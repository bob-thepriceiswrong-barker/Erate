"""
E-Rate Opportunity Finder (TX + OK) — Category 2 Focused
Persists processed data across Streamlit reruns using st.session_state.
"""

import os
import time
import hashlib
from io import BytesIO
from datetime import datetime
from typing import List, Tuple, Dict, Optional

import pandas as pd
import requests
import streamlit as st
import folium
from streamlit_folium import folium_static
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# -----------------------------------------------------------------------------
# Page config
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="E-Rate Opportunity Finder — TX/OK (C2)",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
LAT_WACO = 31.55         # keep rows north of this latitude in TX
LON_I35E = -97.0         # keep rows west of this longitude in TX
GEOCODE_CACHE_FILE = "geocode_cache.csv"
USAC_470_URL = os.getenv("USAC_470_DATASET_URL", "https://opendata.usac.org/resource/jp7a-89nd.json")

USAC_APP_TOKEN = os.getenv("USAC_APP_TOKEN", st.secrets.get("USAC_APP_TOKEN", None))

C2_FUNCTION_KEYWORDS = {
    "ic": [
        "access point", "ap", "wireless", "wifi", "wi-fi", "switch", "switches",
        "router", "firewall", "controller", "ups", "battery", "cabling", "transceiver",
        "sfp", "optics", "antenna"
    ],
    "bmic": ["basic maintenance", "maintenance", "support contract"],
    "mibs": ["managed internal broadband", "mibs", "managed service", "as-a-service", "aas"]
}

COMMON_MANUFACTURERS = [
    "Palo Alto", "Palo Alto Networks", "Cisco", "Meraki", "Aruba", "HPE", "HPE Aruba",
    "Fortinet", "Juniper", "Ruckus", "Ubiquiti", "Extreme", "Brocade", "Arista",
    "Cambium", "Aerohive", "Hikvision", "Mellanox", "TP-Link", "HPE ProCurve"
]

STATE_BBOX = {
    "TX": {"minlon": -106.65, "minlat": 25.84, "maxlon": -93.51, "maxlat": 36.50},
    "OK": {"minlon": -103.00, "minlat": 33.62, "maxlon": -94.43, "maxlat": 37.00},
}

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def normalize_str(x: Optional[str]) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()

def any_keyword_in_text(text: str, keywords: List[str]) -> bool:
    t = text.lower()
    return any(k.lower() in t for k in keywords)

def is_truthy_cell(val) -> bool:
    if pd.isna(val):
        return False
    s = str(val).strip().lower()
    return s in {"true", "yes", "y", "1", "x", "✓", "t"}

def find_columns(df: pd.DataFrame, needles: List[str]) -> List[str]:
    cols = []
    for c in df.columns:
        c_low = str(c).lower()
        if any(n.lower() in c_low for n in needles):
            cols.append(c)
    return cols

def detect_c2_subtypes_from_row(row: pd.Series, func_cols: Dict[str, List[str]]) -> Tuple[bool, bool, bool]:
    ic_cols = func_cols.get("ic", [])
    bmic_cols = func_cols.get("bmic", [])
    mibs_cols = func_cols.get("mibs", [])

    ic = any(is_truthy_cell(row.get(c)) for c in ic_cols)
    bmic = any(is_truthy_cell(row.get(c)) for c in bmic_cols)
    mibs = any(is_truthy_cell(row.get(c)) for c in mibs_cols)

    if not (ic or bmic or mibs):
        text_blob = " ".join([
            normalize_str(row.get("Function")),
            normalize_str(row.get("Functions")),
            normalize_str(row.get("Service Type")),
            " ".join([normalize_str(row.get(c)) for c in row.index if isinstance(row.get(c), str)])
        ])
        ic = any_keyword_in_text(text_blob, C2_FUNCTION_KEYWORDS["ic"])
        bmic = any_keyword_in_text(text_blob, C2_FUNCTION_KEYWORDS["bmic"])
        mibs = any_keyword_in_text(text_blob, C2_FUNCTION_KEYWORDS["mibs"])

    return ic, bmic, mibs

def extract_manufacturers_from_row(row: pd.Series, mfr_cols: List[str]) -> List[str]:
    found = []
    for c in mfr_cols:
        v = row.get(c)
        if is_truthy_cell(v):
            name = str(c).replace(".1", "").replace("'", "").strip()
            found.append(name)
    norm = []
    for m in found:
        mm = m
        if mm.lower().startswith("palo alto"):
            mm = "Palo Alto Networks"
        elif mm.lower() == "extreme networks" or mm.lower().startswith("extreme"):
            mm = "Extreme Networks"
        elif mm.lower().startswith("aruba"):
            mm = "Aruba"
        elif mm.lower().startswith("cisco meraki") or mm.lower() == "meraki":
            mm = "Meraki"
        norm.append(mm)
    return sorted(set(norm))

def load_cache() -> pd.DataFrame:
    if os.path.exists(GEOCODE_CACHE_FILE):
        try:
            return pd.read_csv(GEOCODE_CACHE_FILE)
        except Exception:
            return pd.DataFrame(columns=["Name", "City", "State", "Latitude", "Longitude"])
    return pd.DataFrame(columns=["Name", "City", "State", "Latitude", "Longitude"])

def save_cache(cache: pd.DataFrame) -> None:
    cache.to_csv(GEOCODE_CACHE_FILE, index=False)

def _point_in_box(lat: float, lon: float, bbox: dict) -> bool:
    return (bbox["minlat"] <= lat <= bbox["maxlat"]) and (bbox["minlon"] <= lon <= bbox["maxlon"])

def _census_geocode_city_state(city: str, state: str) -> tuple | None:
    try:
        url = "https://geocoding.geo.census.gov/geocoder/locations/onelineaddress"
        params = {"address": f"{city}, {state}, USA", "benchmark": "Public_AR_Current", "format": "json"}
        r = requests.get(url, params=params, timeout=15)
        if r.status_code != 200:
            return None
        j = r.json()
        matches = j.get("result", {}).get("addressMatches", [])
        if not matches:
            return None
        coords = matches[0].get("coordinates", {})
        lon = coords.get("x")
        lat = coords.get("y")
        if lat is None or lon is None:
            return None
        return (lat, lon)
    except Exception:
        return None

@st.cache_data(ttl=86400)
def geocode_bulk(df: pd.DataFrame) -> pd.DataFrame:
    cache = load_cache()
    geolocator = Nominatim(user_agent="erate_geo_cache_tx_ok")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

    merged = df.merge(
        cache, how="left",
        left_on=["Name of Applicant", "City", "State"],
        right_on=["Name", "City", "State"]
    )

    needs = merged[merged["Latitude"].isna() | merged["Longitude"].isna()].copy()
    progress = st.progress(0.0)
    total = max(1, len(needs))
    done = 0

    for idx, row in needs.iterrows():
        city = str(row["City"]).strip()
        state = str(row["State"]).strip().upper()
        bbox = STATE_BBOX.get(state)
        lat, lon = None, None

        try:
            if bbox:
                flat_viewbox = [bbox["minlon"], bbox["minlat"], bbox["maxlon"], bbox["maxlat"]]
                loc = geocode(
                    f"{city}, {state}, USA",
                    country_codes="us",
                    viewbox=flat_viewbox,
                    bounded=True,
                    addressdetails=False,
                    exactly_one=True,
                )
            else:
                loc = geocode(f"{city}, {state}, USA", country_codes="us", exactly_one=True)
        except Exception:
            loc = None

        if loc:
            lat, lon = loc.latitude, loc.longitude
            if bbox and not _point_in_box(lat, lon, bbox):
                lat, lon = None, None

        if (lat is None or lon is None) and bbox:
            alt = _census_geocode_city_state(city, state)
            if alt:
                lat, lon = alt
                if not _point_in_box(lat, lon, bbox):
                    lat, lon = None, None

        if lat is not None and lon is not None:
            merged.at[idx, "Latitude"] = lat
            merged.at[idx, "Longitude"] = lon
            cache = pd.concat([cache, pd.DataFrame([{
                "Name": row["Name of Applicant"], "City": city, "State": state,
                "Latitude": lat, "Longitude": lon
            }])], ignore_index=True)

        done += 1
        progress.progress(min(1.0, done / total))
        time.sleep(0.05)

    if len(cache):
        cache = cache.drop_duplicates(subset=["Name", "City", "State"], keep="last")
        save_cache(cache)

    merged = merged.drop(columns=["Name"], errors="ignore")
    return merged

def color_for_service_counts(ic: int, bmic: int, mibs: int) -> str:
    only_ic = ic > 0 and bmic == 0 and mibs == 0
    only_bmic = bmic > 0 and ic == 0 and mibs == 0
    only_mibs = mibs > 0 and ic == 0 and bmic == 0
    if only_ic: return "blue"
    if only_bmic: return "green"
    if only_mibs: return "purple"
    if ic + bmic + mibs > 0: return "darkred"
    return "gray"

def construct_form_470_url(app_number) -> Optional[str]:
    if pd.isna(app_number): return None
    try:
        v = int(str(app_number).strip().split(".")[0])
    except Exception:
        return None
    return f"http://legacy.fundsforlearning.com/470/{v}"

# -----------------------------------------------------------------------------
# USAC API fallback (TX + OK)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_usac_470(states=("TX", "OK"), years_back=3, limit_per_state=10000) -> pd.DataFrame:
    headers = {}
    if USAC_APP_TOKEN:
        headers["X-App-Token"] = USAC_APP_TOKEN

    now_y = datetime.now().year
    start_y = now_y - years_back

    rows = []
    for stus in states:
        for yr in range(start_y, now_y + 1):
            offset = 0
            while offset < limit_per_state:
                params = {
                    "$limit": 5000,
                    "$offset": offset,
                    "$select": ",".join([
                        "billed_entity_name","billed_entity_city","billed_entity_state",
                        "application_number","funding_year",
                        "category_one_description","category_two_description",
                    ]),
                    "$where": f"billed_entity_state='{stus}' AND funding_year={yr}",
                }
                try:
                    resp = requests.get(USAC_470_URL, params=params, headers=headers, timeout=30)
                except Exception:
                    break
                if resp.status_code >= 400:
                    break
                chunk = resp.json()
                if not chunk:
                    break
                rows.extend(chunk)
                offset += len(chunk)
                if len(chunk) < 5000:
                    break

    if not rows:
        return pd.DataFrame()

    raw = pd.DataFrame(rows)
    if raw.empty:
        return pd.DataFrame()

    out_rows = []
    for _, r in raw.iterrows():
        name = normalize_str(r.get("billed_entity_name"))
        city = normalize_str(r.get("billed_entity_city"))
        state = normalize_str(r.get("billed_entity_state"))
        appno = r.get("application_number")
        c2 = normalize_str(r.get("category_two_description"))

        ic = 1 if c2 != "" else 0   # approximate
        bmic = 0
        mibs = 0

        out_rows.append({
            "Name of Applicant": name, "City": city, "State": state,
            "470 App Number": appno, "IC": ic, "BMIC": bmic, "MIBS": mibs,
            "Functions": "", "Manufacturers": "", "Source": "USAC",
        })

    out = pd.DataFrame(out_rows).drop_duplicates(subset=["Name of Applicant", "470 App Number"])
    return out

# -----------------------------------------------------------------------------
# Funds For Learning Excel parsing (wide format)
# -----------------------------------------------------------------------------
@st.cache_data
def load_excel_ffl(file) -> pd.DataFrame:
    return pd.read_excel(file)

def derive_function_and_manufacturer_columns(df: pd.DataFrame) -> Tuple[Dict[str, List[str]], List[str]]:
    ic_needles = [
        "wireless", "wi-fi", "wifi", "access point", "ap", "switch", "switches",
        "router", "firewall", "controller", "ups", "battery", "cabling", "transceiver",
        "sfp", "optic", "antenna"
    ]
    ic_cols = find_columns(df, ic_needles)
    bmic_cols = find_columns(df, ["basic maintenance", "maintenance"])
    mibs_cols = find_columns(df, ["managed internal broadband", "mibs", "managed service"])

    mfr_needles = list(COMMON_MANUFACTURERS) + [
        "palo alto", "cisco", "meraki", "aruba", "fortinet", "juniper", "ruckus",
        "ubiquiti", "extreme", "brocade", "arista", "cambium", "aerohive", "hikvision",
        "tplink", "tp-link", "hewlett", "hewlett packard", "hpe", "hp"
    ]
    mfr_cols = find_columns(df, mfr_needles)

    return {"ic": ic_cols, "bmic": bmic_cols, "mibs": mibs_cols}, mfr_cols

def parse_ffl_to_c2(df: pd.DataFrame) -> pd.DataFrame:
    name_col = next((c for c in df.columns if str(c).lower().strip() == "name of applicant"), None)
    city_col = next((c for c in df.columns if "city" in str(c).lower()), None)
    state_col = next((c for c in df.columns if "state" in str(c).lower()), None)
    app_col = next((c for c in df.columns if "470" in str(c).lower() and "number" in str(c).lower()), None)

    if not all([name_col, city_col, state_col, app_col]):
        st.error("Missing required columns (Name of Applicant, City, State, 470 App Number).")
        return pd.DataFrame()

    func_cols, mfr_cols = derive_function_and_manufacturer_columns(df)

    out_rows = []
    for _, r in df.iterrows():
        name = normalize_str(r.get(name_col))
        city = normalize_str(r.get(city_col))
        state = normalize_str(r.get(state_col))
        appno = r.get(app_col)
        if name == "" or city == "" or state == "" or pd.isna(appno):
            continue

        ic, bmic, mibs = detect_c2_subtypes_from_row(r, func_cols)
        if not (ic or bmic or mibs):
            continue

        present_funcs = []
        for c in func_cols.get("ic", []) + func_cols.get("bmic", []) + func_cols.get("mibs", []):
            if is_truthy_cell(r.get(c)):
                present_funcs.append(str(c).replace(".1", "").replace("'", "").strip())
        present_funcs = sorted(set(present_funcs))

        mfrs = extract_manufacturers_from_row(r, mfr_cols)

        out_rows.append({
            "Name of Applicant": name, "City": city, "State": state,
            "470 App Number": appno,
            "IC": 1 if ic else 0, "BMIC": 1 if bmic else 0, "MIBS": 1 if mibs else 0,
            "Functions": ", ".join(present_funcs),
            "Manufacturers": ", ".join(mfrs),
            "Source": "FFL",
        })

    if not out_rows:
        return pd.DataFrame()

    out = pd.DataFrame(out_rows)
    agg = out.groupby(["Name of Applicant", "City", "State", "470 App Number"], as_index=False).agg({
        "IC": "sum", "BMIC": "sum", "MIBS": "sum",
        "Functions": lambda s: ", ".join(sorted(set(", ".join(s).split(", ")))).strip(", "),
        "Manufacturers": lambda s: ", ".join(sorted(set(", ".join(s).split(", ")))).strip(", "),
        "Source": lambda s: "FFL"
    })
    agg["IC"] = (agg["IC"] > 0).astype(int)
    agg["BMIC"] = (agg["BMIC"] > 0).astype(int)
    agg["MIBS"] = (agg["MIBS"] > 0).astype(int)
    return agg

# -----------------------------------------------------------------------------
# Filters and map
# -----------------------------------------------------------------------------
def apply_geo_filter(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    ok_df = df[df["State"].str.upper() == "OK"].copy()
    tx_df = df[df["State"].str.upper() == "TX"].copy()
    tx_df = tx_df[(tx_df["Latitude"] > LAT_WACO) & (tx_df["Longitude"] < LON_I35E)]
    return pd.concat([ok_df, tx_df], ignore_index=True)

def build_map(df: pd.DataFrame):
    if df.empty: return None
    center_lat = df["Latitude"].mean()
    center_lon = df["Longitude"].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=6, tiles="OpenStreetMap")

    folium.PolyLine(locations=[[LAT_WACO, -106], [LAT_WACO, -93]], color="red", weight=2, opacity=0.5,
                    popup="Waco latitude cutoff").add_to(m)
    folium.PolyLine(locations=[[25, LON_I35E], [37, LON_I35E]], color="red", weight=2, opacity=0.5,
                    popup="I-35E longitude cutoff").add_to(m)

    for _, r in df.iterrows():
        ic = int(r.get("IC", 0)); bmic = int(r.get("BMIC", 0)); mibs = int(r.get("MIBS", 0))
        color = color_for_service_counts(ic, bmic, mibs)
        name = r["Name of Applicant"]; city = r["City"]; state = r["State"]
        fn = r.get("Functions", ""); mf = r.get("Manufacturers", "")
        url = construct_form_470_url(r.get("470 App Number"))
        popup_html = f"""
        <div style="width: 420px; font-family: Arial, sans-serif;">
          <h3 style="margin:0 0 8px 0; color:#0e4ca1;">{name}</h3>
          <p style="margin:0 0 6px 0;">📍 {city}, {state}</p>
          <ul style="margin:6px 0; padding-left: 18px; font-size: 13px;">
            <li>IC: <b>{ic}</b> &nbsp; BMIC: <b>{bmic}</b> &nbsp; MIBS: <b>{mibs}</b></li>
          </ul>
          <p style="margin:6px 0; font-size: 12px;"><b>Functions:</b> {fn}</p>
          <p style="margin:6px 0; font-size: 12px;"><b>Manufacturers:</b> {mf}</p>
          <div style="margin-top:10px; text-align:center;">
            {"<a href='" + url + "' target='_blank' style='background:#0e4ca1; color:white; padding:8px 14px; text-decoration:none; border-radius:6px;'>View Form 470 →</a>" if url else ""}
          </div>
        </div>
        """
        folium.Marker(
            [r["Latitude"], r["Longitude"]],
            popup=folium.Popup(popup_html, max_width=460),
            tooltip=f"{name} — click for details",
        ).add_to(m)
        folium.CircleMarker(
            [r["Latitude"], r["Longitude"]],
            radius=7, color=color, fill=True, fill_color=color, fill_opacity=0.65, weight=2,
        ).add_to(m)
    return m

# -----------------------------------------------------------------------------
# Session-state utilities
# -----------------------------------------------------------------------------
def file_digest(uploaded_file) -> str:
    if uploaded_file is None:
        return ""
    buf = uploaded_file.getbuffer()
    return hashlib.sha256(buf).hexdigest()

def source_key(mode: str, uploaded_file) -> str:
    if mode == "Upload Excel (recommended)":
        return f"FFL::{file_digest(uploaded_file)}"
    else:
        # include year for natural invalidation each year
        now_y = datetime.now().year
        return f"USAC::TX,OK::{now_y}"

def ensure_session_keys():
    for k in [
        "data_key","raw_df","parsed_df","geocoded_df","filtered_geo_df",
        "filter_functions","filter_manufacturers","ui_initialized"
    ]:
        st.session_state.setdefault(k, None)

ensure_session_keys()

# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
st.title("📄 E-Rate Opportunity Finder — TX/OK (Category 2)")
st.caption("Uploads a Funds For Learning 470 Excel export or uses the USAC API as a fallback. Category 2 only (IC, BMIC, MIBS).")

with st.sidebar:
    st.header("Data Source")
    src = st.radio(
        "Choose source",
        options=["Upload Excel (recommended)", "USAC API (fallback)"],
        help="Excel uses detailed Funds For Learning export. API is simplified.",
        key="src_choice"
    )
    up = None
    if src == "Upload Excel (recommended)":
        up = st.file_uploader("Upload FFL Excel (470 export)", type=["xls", "xlsx"], key="upload_file")

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        reload_btn = st.button("Reload data", use_container_width=True)
    with col_b:
        clear_btn = st.button("Clear session", use_container_width=True)

    st.divider()
    st.header("Filters")
    # Placeholders; populated after data is loaded
    func_ph = st.empty()
    mfr_ph = st.empty()

    run_btn = st.button("🚀 Analyze", type="primary", use_container_width=True, key="analyze_btn")

if clear_btn:
    for k in list(st.session_state.keys()):
        if k.startswith(("data_", "raw_", "parsed_", "geocoded_", "filtered_", "filter_", "ui_")):
            st.session_state.pop(k, None)
    st.success("Session cleared.")
    st.stop()

# Guard: require file for upload mode before first run
if src == "Upload Excel (recommended)" and up is None and not run_btn and st.session_state["parsed_df"] is None:
    st.info("👈 Upload an Excel file or switch to USAC API fallback.")
    st.stop()

# Determine current key
current_key = source_key(src, up)

# Compute/recompute pipeline if:
# - First time run button pressed, or
# - Reload requested, or
# - Source key changed since last compute
needs_recompute = (
    run_btn or reload_btn or
    (st.session_state["data_key"] is None) or
    (st.session_state["data_key"] != current_key)
)

if needs_recompute:
    with st.spinner("Processing data..."):
        # RAW
        if src == "Upload Excel (recommended)":
            raw = load_excel_ffl(up)
            if raw.empty:
                st.error("The uploaded Excel appears empty.")
                st.stop()
        else:
            raw = fetch_usac_470()
            if raw.empty:
                st.error("No data returned from USAC API.")
                st.stop()
        st.session_state["raw_df"] = raw

        # PARSED
        if src == "Upload Excel (recommended)":
            parsed = parse_ffl_to_c2(raw)
            if parsed.empty:
                st.error("No Category 2 opportunities found in the uploaded file.")
                st.stop()
        else:
            parsed = raw  # USAC path already minimal-mapped
        st.session_state["parsed_df"] = parsed

        # GEOCODED
        geocoded = geocode_bulk(parsed)
        geocoded = geocoded.dropna(subset=["Latitude", "Longitude"])
        st.session_state["geocoded_df"] = geocoded

        # GEO FILTER
        filtered_geo = apply_geo_filter(geocoded)
        if filtered_geo.empty:
            st.warning("No rows remained after geographic filtering.")
            st.stop()
        st.session_state["filtered_geo_df"] = filtered_geo

        # Initialize dynamic filter options
        all_functions = sorted(set(", ".join(filtered_geo["Functions"].fillna("")).split(", ")))
        all_functions = [f for f in all_functions if f]
        all_manufacturers = sorted(set(", ".join(filtered_geo["Manufacturers"].fillna("")).split(", ")))
        all_manufacturers = [m for m in all_manufacturers if m]

        st.session_state["filter_functions"] = all_functions
        st.session_state["filter_manufacturers"] = all_manufacturers

        # Mark data key
        st.session_state["data_key"] = current_key
        st.session_state["ui_initialized"] = True

# If we get here without data, stop
if st.session_state["filtered_geo_df"] is None:
    st.stop()

# Render dynamic filters bound to session_state so selections persist across reruns
with st.sidebar:
    sel_funcs = func_ph.multiselect(
        "Equipment Functions",
        options=st.session_state["filter_functions"] or [],
        default=st.session_state.get("sel_funcs", []),
        key="sel_funcs"
    )
    sel_mfrs = mfr_ph.multiselect(
        "Manufacturers",
        options=st.session_state["filter_manufacturers"] or [],
        default=st.session_state.get("sel_mfrs", []),
        key="sel_mfrs"
    )

# Apply filters to a working copy from session
df = st.session_state["filtered_geo_df"].copy()
if st.session_state["sel_funcs"]:
    df = df[df["Functions"].apply(lambda s: any(f in str(s) for f in st.session_state["sel_funcs"]))]
if st.session_state["sel_mfrs"]:
    df = df[df["Manufacturers"].apply(lambda s: any(m in str(s) for m in st.session_state["sel_mfrs"]))]

if df.empty:
    st.warning("No rows match your filter choices. Clear filters to see all results.")
    st.stop()

# Summary metrics
col1, col2, col3, col4 = st.columns(4)
with col1: st.metric("Opportunities", len(df))
with col2: st.metric("IC", int((df["IC"] > 0).sum()))
with col3: st.metric("BMIC", int((df["BMIC"] > 0).sum()))
with col4: st.metric("MIBS", int((df["MIBS"] > 0).sum()))

st.divider()

tab1, tab2 = st.tabs(["🗺️ Interactive Map", "📊 Table + Export"])

with tab1:
    st.subheader("Map — click markers for details")
    m = build_map(df)
    if m:
        folium_static(m, width=1200, height=640)
        st.markdown("**Legend:** 🔵 IC, 🟢 BMIC, 🟣 MIBS, 🔴 Mixed")

with tab2:
    st.subheader("Results")
    view = df.copy()
    columns = [
        "Name of Applicant", "City", "State", "470 App Number",
        "IC", "BMIC", "MIBS", "Functions", "Manufacturers", "Source",
        "Latitude", "Longitude"
    ]
    for c in columns:
        if c not in view.columns:
            view[c] = ""
    view = view[columns].sort_values(["State", "City", "Name of Applicant"]).reset_index(drop=True)
    st.dataframe(view, use_container_width=True)
    out = BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        view.to_excel(writer, index=False, sheet_name="C2 Opportunities")
    out.seek(0)
    st.download_button(
        "📥 Download Excel",
        data=out,
        file_name="erate_c2_opportunities_tx_ok.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
    )
