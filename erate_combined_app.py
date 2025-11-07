"""
E-Rate Unified Application
Combines Lead Management System and USAC Data Viewer in one app
"""

import os
import time
import hashlib
import sqlite3
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
# Page config (must be first Streamlit command)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="E-Rate Unified App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# App Mode Selection
# -----------------------------------------------------------------------------
st.sidebar.title("📊 E-Rate Unified App")
app_mode = st.sidebar.radio(
    "Select Application Mode:",
    ["🎯 Lead Management System", "🎓 USAC Data Viewer"],
    help="Choose which tool to use"
)

st.sidebar.markdown("---")

# -----------------------------------------------------------------------------
# Shared Configuration
# -----------------------------------------------------------------------------
APP_TOKEN = "MFa7tcJ2Pybss1tK93iriT9qW"

# Try to get app token from environment or secrets, fallback to default
try:
    USAC_APP_TOKEN = os.getenv("USAC_APP_TOKEN") or st.secrets.get("USAC_APP_TOKEN", None)
except:
    USAC_APP_TOKEN = APP_TOKEN

# -----------------------------------------------------------------------------
# MODE 1: LEAD MANAGEMENT SYSTEM
# -----------------------------------------------------------------------------
if app_mode == "🎯 Lead Management System":

    # Import lead management specific constants
    LAT_WACO = 31.55
    LON_I35E = -97.0
    GEOCODE_CACHE_FILE = "geocode_cache.csv"
    LEAD_TRACKING_DB = "lead_tracking.db"

    # USAC Open Data URLs - All use SODA2 (Form 470 dataset doesn't support SODA3)
    USAC_470_URL = os.getenv("USAC_470_DATASET_URL", "https://opendata.usac.org/resource/jp7a-89nd.json")
    USAC_471_COMMITMENTS_URL = "https://opendata.usac.org/resource/avi8-svp9.json"  # Form 471 commitments
    USAC_C2_BUDGET_URL = "https://opendata.usac.org/resource/9xr8-jzmv.json"  # Category 2 budget

    C2_FUNCTION_KEYWORDS = {
        "ic": [
            "access point", "ap", "wireless", "wifi", "wi-fi", "switch", "switches",
            "router", "firewall", "controller", "ups", "battery", "cabling", "transceiver",
            "sfp", "optics", "antenna", "rack", "mount", "patch panel"
        ],
        "bmic": ["basic maintenance", "maintenance", "support contract"],
        "mibs": ["managed internal broadband", "mibs", "managed service", "as-a-service", "aas"]
    }

    EQUIPMENT_CATEGORIES = [
        "Switches", "Wireless Access Points", "Firewalls", "Routers", "Wireless Controllers",
        "UPS / Battery Backup", "Cabling (Copper/Fiber)", "Racks and Mounts",
        "Antennas and Connectors", "Transceivers / SFP / Optics", "Patch Panels",
        "Basic Maintenance", "Managed Internal Broadband Services"
    ]

    COMMON_MANUFACTURERS = [
        "Cisco", "Cisco Meraki", "Meraki", "Aruba", "HPE", "HPE Aruba",
        "Juniper", "Juniper Mist", "Mist", "Ruckus", "Ruckus Wireless", "CommScope",
        "Extreme Networks", "Extreme", "Arista", "Dell Networking", "Ubiquiti",
        "Netgear", "Cambium", "Aerohive", "Brocade", "HPE ProCurve", "TP-Link",
        "Fortinet", "Palo Alto Networks", "Palo Alto", "SonicWall", "WatchGuard",
        "Check Point", "Barracuda", "Sophos",
        "Cisco VOIP", "Mitel", "ShoreTel", "Avaya", "Polycom", "Grandstream",
        "CommScope Cabling", "Leviton", "Panduit", "Corning", "Belden",
        "APC", "Schneider Electric", "Tripp Lite", "Eaton", "Vertiv", "Liebert", "CyberPower"
    ]

    STATE_BBOX = {
        "TX": {"minlon": -106.65, "minlat": 25.84, "maxlon": -93.51, "maxlat": 36.50},
        "OK": {"minlon": -103.00, "minlat": 33.62, "maxlon": -94.43, "maxlat": 37.00},
    }

    # Helper functions for Lead Management
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

    # Database functions
    def init_lead_tracking_db():
        conn = sqlite3.connect(LEAD_TRACKING_DB)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS leads
                     (app_number TEXT PRIMARY KEY,
                      applicant_name TEXT,
                      pursuing INTEGER DEFAULT 0,
                      notes TEXT,
                      last_updated TEXT)''')
        c.execute('''CREATE TABLE IF NOT EXISTS external_insights
                     (applicant_name TEXT PRIMARY KEY,
                      bond_info TEXT,
                      board_notes TEXT,
                      last_updated TEXT)''')
        conn.commit()
        conn.close()

    def get_lead_status(app_number: str) -> Dict:
        conn = sqlite3.connect(LEAD_TRACKING_DB)
        c = conn.cursor()
        c.execute("SELECT pursuing, notes FROM leads WHERE app_number=?", (str(app_number),))
        result = c.fetchone()
        conn.close()
        if result:
            return {"pursuing": bool(result[0]), "notes": result[1] or ""}
        return {"pursuing": False, "notes": ""}

    def update_lead_status(app_number: str, applicant_name: str, pursuing: bool, notes: str):
        conn = sqlite3.connect(LEAD_TRACKING_DB)
        c = conn.cursor()
        c.execute("""INSERT OR REPLACE INTO leads
                     (app_number, applicant_name, pursuing, notes, last_updated)
                     VALUES (?, ?, ?, ?, ?)""",
                  (str(app_number), applicant_name, int(pursuing), notes,
                   datetime.now().isoformat()))
        conn.commit()
        conn.close()

    def get_external_insights(applicant_name: str) -> Dict:
        conn = sqlite3.connect(LEAD_TRACKING_DB)
        c = conn.cursor()
        c.execute("SELECT bond_info, board_notes FROM external_insights WHERE applicant_name=?",
                  (applicant_name,))
        result = c.fetchone()
        conn.close()
        if result:
            return {"bond_info": result[0] or "", "board_notes": result[1] or ""}
        return {"bond_info": "", "board_notes": ""}

    def update_external_insights(applicant_name: str, bond_info: str, board_notes: str):
        conn = sqlite3.connect(LEAD_TRACKING_DB)
        c = conn.cursor()
        c.execute("""INSERT OR REPLACE INTO external_insights
                     (applicant_name, bond_info, board_notes, last_updated)
                     VALUES (?, ?, ?, ?)""",
                  (applicant_name, bond_info, board_notes, datetime.now().isoformat()))
        conn.commit()
        conn.close()

    @st.cache_data(ttl=86400)
    def fetch_471_history(entity_name: str, years_back: int = 3) -> pd.DataFrame:
        headers = {}
        if USAC_APP_TOKEN:
            headers["X-App-Token"] = USAC_APP_TOKEN

        now_y = datetime.now().year
        start_y = now_y - years_back

        rows = []
        for yr in range(start_y, now_y + 1):
            params = {
                "$limit": 1000,
                "$select": ",".join([
                    "funding_year", "applicant_name", "service_provider_name",
                    "total_authorized_disbursement", "funding_commitment_request",
                    "category_of_service", "frn_line_item_service_details"
                ]),
                "$where": f"applicant_name='{entity_name}' AND funding_year={yr}",
            }
            try:
                resp = requests.get(USAC_471_COMMITMENTS_URL, params=params,
                                  headers=headers, timeout=30)
                if resp.status_code == 200:
                    chunk = resp.json()
                    rows.extend(chunk)
            except Exception:
                continue

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame(rows)

    @st.cache_data(ttl=86400)
    def fetch_c2_budget(entity_name: str) -> Dict:
        headers = {}
        if USAC_APP_TOKEN:
            headers["X-App-Token"] = USAC_APP_TOKEN

        params = {
            "$limit": 10,
            "$select": ",".join([
                "entity_name", "category_two_budget_total",
                "category_two_budget_used", "category_two_budget_remaining"
            ]),
            "$where": f"entity_name='{entity_name}'",
        }

        try:
            resp = requests.get(USAC_C2_BUDGET_URL, params=params,
                              headers=headers, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                if data:
                    item = data[0]
                    total = float(item.get("category_two_budget_total", 0))
                    used = float(item.get("category_two_budget_used", 0))
                    remaining = float(item.get("category_two_budget_remaining", 0))

                    percent_used = (used / total * 100) if total > 0 else 0

                    return {
                        "total": total,
                        "used": used,
                        "remaining": remaining,
                        "percent_used": percent_used
                    }
        except Exception:
            pass

        return {"total": 0, "used": 0, "remaining": 0, "percent_used": 0}

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
                        "$where": f"billed_entity_state='{stus}' AND funding_year='{yr}'",
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

            ic = 1 if c2 != "" else 0
            bmic = 0
            mibs = 0

            out_rows.append({
                "Name of Applicant": name, "City": city, "State": state,
                "470 App Number": appno, "IC": ic, "BMIC": bmic, "MIBS": mibs,
                "Functions": "", "Manufacturers": "", "Source": "USAC",
            })

        out = pd.DataFrame(out_rows).drop_duplicates(subset=["Name of Applicant", "470 App Number"])
        return out

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

    def show_account_detail(row):
        name = row["Name of Applicant"]
        app_number = row.get("470 App Number", "")

        st.subheader(f"📋 {name}")
        st.caption(f"📍 {row['City']}, {row['State']} | Form 470: {app_number}")

        tab1, tab2, tab3, tab4 = st.tabs([
            "📄 Current Request (470)",
            "📊 Historical Funding (471)",
            "💰 E-Rate Budget",
            "🔍 External Insights"
        ])

        with tab1:
            st.markdown("### Current Form 470 Request")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Equipment Functions:**")
                funcs = row.get("Functions", "")
                if funcs:
                    for func in funcs.split(", "):
                        st.markdown(f"- {func}")
                else:
                    st.info("No specific functions listed")

            with col2:
                st.markdown(f"**Manufacturers Requested:**")
                mfrs = row.get("Manufacturers", "")
                if mfrs:
                    for mfr in mfrs.split(", "):
                        st.markdown(f"- {mfr}")
                else:
                    st.info("No specific manufacturers listed")

            url = construct_form_470_url(app_number)
            if url:
                st.markdown(f"[🔗 View Full Form 470 Details]({url})")

        with tab2:
            st.markdown("### Historical E-Rate Funding (Past 3 Years)")
            with st.spinner("Loading Form 471 history..."):
                history_df = fetch_471_history(name)

            if not history_df.empty:
                summary = history_df.groupby(['funding_year', 'service_provider_name']).agg({
                    'total_authorized_disbursement': 'sum',
                    'category_of_service': lambda x: ', '.join(set(x))
                }).reset_index()

                st.dataframe(summary, use_container_width=True)

                total_funding = history_df['total_authorized_disbursement'].astype(float).sum()
                st.metric("Total Historical Funding (3 years)", f"${total_funding:,.0f}")
            else:
                st.info("No Form 471 history found for this applicant in the past 3 years.")

        with tab3:
            st.markdown("### Category 2 Budget Status")
            with st.spinner("Loading C2 budget info..."):
                budget = fetch_c2_budget(name)

            if budget["total"] > 0:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Budget", f"${budget['total']:,.0f}")
                with col2:
                    st.metric("Used", f"${budget['used']:,.0f}")
                with col3:
                    st.metric("Remaining", f"${budget['remaining']:,.0f}")

                st.progress(budget["percent_used"] / 100)
                st.caption(f"Budget Used: {budget['percent_used']:.1f}%")

                if budget["percent_used"] > 90:
                    st.warning("⚠️ This applicant has used >90% of their Category 2 budget. Limited funding available.")
                elif budget["percent_used"] < 50:
                    st.success("✅ This applicant has significant Category 2 budget remaining!")
            else:
                st.info("No Category 2 budget information available for this applicant.")

        with tab4:
            st.markdown("### External Insights")
            st.caption("Track bond initiatives, board meeting notes, and other intelligence")

            insights = get_external_insights(name)

            bond_info = st.text_area("Bond Information",
                                     value=insights["bond_info"],
                                     height=100,
                                     placeholder="e.g., $50M tech bond passed Nov 2024",
                                     key=f"bond_{app_number}")

            board_notes = st.text_area("Board Meeting Notes",
                                       value=insights["board_notes"],
                                       height=100,
                                       placeholder="e.g., Board approved network upgrade RFP - Jan 2025",
                                       key=f"board_{app_number}")

            if st.button("💾 Save Insights", key=f"save_insights_{app_number}"):
                update_external_insights(name, bond_info, board_notes)
                st.success("Insights saved!")

        st.divider()
        st.markdown("### 🎯 Lead Tracking")

        lead_status = get_lead_status(app_number)

        col1, col2 = st.columns([1, 3])
        with col1:
            pursuing = st.checkbox("Actively Pursuing",
                                  value=lead_status["pursuing"],
                                  key=f"pursuing_{app_number}")

        with col2:
            notes = st.text_area("Sales Notes",
                                value=lead_status["notes"],
                                height=100,
                                placeholder="Add your notes about this lead...",
                                key=f"notes_{app_number}")

        if st.button("💾 Save Lead Status", key=f"save_lead_{app_number}", type="primary"):
            update_lead_status(app_number, name, pursuing, notes)
            st.success("Lead status saved!")
            st.balloons()

    def file_digest(uploaded_file) -> str:
        if uploaded_file is None:
            return ""
        buf = uploaded_file.getbuffer()
        return hashlib.sha256(buf).hexdigest()

    def source_key(mode: str, uploaded_file) -> str:
        if mode == "Upload Excel (recommended)":
            return f"FFL::{file_digest(uploaded_file)}"
        else:
            now_y = datetime.now().year
            return f"USAC::TX,OK::{now_y}"

    def ensure_session_keys():
        for k in [
            "data_key","raw_df","parsed_df","geocoded_df","filtered_geo_df",
            "filter_functions","filter_manufacturers","ui_initialized"
        ]:
            st.session_state.setdefault(k, None)

    ensure_session_keys()
    init_lead_tracking_db()

    # UI
    st.title("📄 E-Rate Lead Management System — TX/OK")
    st.caption("Enterprise-level E-Rate opportunity finder with historical data, budget tracking, and lead management. Territory: North TX (north of Waco, west of I-35E) + Oklahoma.")

    with st.expander("ℹ️ What's New in This Enterprise Version", expanded=False):
        st.markdown("""
        ### 🎯 Key Features:

        **📊 Enhanced Filters:**
        - **40+ Manufacturers**: Cisco, Fortinet, Palo Alto, SonicWall, Aruba, Meraki, and more
        - **13 Equipment Categories**: Switches, Wireless APs, Firewalls, UPS, Cabling, Racks, etc.
        - **Multi-Select**: Choose multiple manufacturers and equipment types simultaneously

        **💰 Financial Intelligence:**
        - **Form 471 History**: See past E-Rate purchases and winning vendors (3-year lookback)
        - **Category 2 Budget Tracking**: View remaining C2 budget for each applicant
        - **Smart Prioritization**: Identify high-value targets with available funding

        **🎯 Lead Management:**
        - **Mark Leads**: Flag opportunities you're actively pursuing
        - **Sales Notes**: Add and track notes for each opportunity
        - **Filter Pursued**: View only your active pipeline
        - **Persistent Storage**: All tracking data saved to local database

        **🔍 360° Account View:**
        - **Current Request (470)**: Equipment and manufacturer details
        - **Historical Funding (471)**: Past purchases and service providers
        - **E-Rate Budget**: Total/used/remaining C2 funding with visual progress
        - **External Insights**: Track bond initiatives and board meeting notes

        **🚀 Performance:**
        - Instant filtering (no page reloads)
        - Cached API data for speed
        - One-click Excel export
        - Interactive map visualization
        """)

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
        func_ph = st.empty()
        mfr_ph = st.empty()

        st.divider()
        show_pursued = st.checkbox("Show only pursued leads", value=False, key="show_pursued_only")

        run_btn = st.button("🚀 Analyze", type="primary", use_container_width=True, key="analyze_btn")

    if clear_btn:
        for k in list(st.session_state.keys()):
            if k.startswith(("data_", "raw_", "parsed_", "geocoded_", "filtered_", "filter_", "ui_")):
                st.session_state.pop(k, None)
        st.success("Session cleared.")
        st.stop()

    if src == "Upload Excel (recommended)" and up is None and not run_btn and st.session_state["parsed_df"] is None:
        st.info("👈 Upload an Excel file or switch to USAC API fallback.")
        st.stop()

    current_key = source_key(src, up)

    needs_recompute = (
        run_btn or reload_btn or
        (st.session_state["data_key"] is None) or
        (st.session_state["data_key"] != current_key)
    )

    if needs_recompute:
        with st.spinner("Processing data..."):
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

            if src == "Upload Excel (recommended)":
                parsed = parse_ffl_to_c2(raw)
                if parsed.empty:
                    st.error("No Category 2 opportunities found in the uploaded file.")
                    st.stop()
            else:
                parsed = raw
            st.session_state["parsed_df"] = parsed

            geocoded = geocode_bulk(parsed)
            geocoded = geocoded.dropna(subset=["Latitude", "Longitude"])
            st.session_state["geocoded_df"] = geocoded

            filtered_geo = apply_geo_filter(geocoded)
            if filtered_geo.empty:
                st.warning("No rows remained after geographic filtering.")
                st.stop()
            st.session_state["filtered_geo_df"] = filtered_geo

            all_functions = sorted(set(", ".join(filtered_geo["Functions"].fillna("")).split(", ")))
            all_functions = [f for f in all_functions if f]
            all_manufacturers = sorted(set(", ".join(filtered_geo["Manufacturers"].fillna("")).split(", ")))
            all_manufacturers = [m for m in all_manufacturers if m]

            st.session_state["filter_functions"] = all_functions
            st.session_state["filter_manufacturers"] = all_manufacturers

            st.session_state["data_key"] = current_key
            st.session_state["ui_initialized"] = True

    if st.session_state["filtered_geo_df"] is None:
        st.stop()

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

    df = st.session_state["filtered_geo_df"].copy()
    if st.session_state["sel_funcs"]:
        df = df[df["Functions"].apply(lambda s: any(f in str(s) for f in st.session_state["sel_funcs"]))]
    if st.session_state["sel_mfrs"]:
        df = df[df["Manufacturers"].apply(lambda s: any(m in str(s) for m in st.session_state["sel_mfrs"]))]

    if st.session_state.get("show_pursued_only", False):
        pursued_apps = []
        conn = sqlite3.connect(LEAD_TRACKING_DB)
        c = conn.cursor()
        c.execute("SELECT app_number FROM leads WHERE pursuing=1")
        pursued_apps = [str(row[0]) for row in c.fetchall()]
        conn.close()

        if pursued_apps:
            df = df[df["470 App Number"].astype(str).isin(pursued_apps)]
        else:
            st.info("No leads marked as pursuing yet. Uncheck 'Show only pursued leads' to see all opportunities.")
            st.stop()

    if df.empty:
        st.warning("No rows match your filter choices. Clear filters to see all results.")
        st.stop()

    pursued_count = 0
    if not df.empty:
        conn = sqlite3.connect(LEAD_TRACKING_DB)
        c = conn.cursor()
        app_numbers = df["470 App Number"].astype(str).tolist()
        if app_numbers:
            placeholders = ",".join(["?" for _ in app_numbers])
            c.execute(f"SELECT COUNT(*) FROM leads WHERE pursuing=1 AND app_number IN ({placeholders})", app_numbers)
            result = c.fetchone()
            pursued_count = result[0] if result else 0
        conn.close()

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: st.metric("Total Opportunities", len(df))
    with col2: st.metric("IC", int((df["IC"] > 0).sum()))
    with col3: st.metric("BMIC", int((df["BMIC"] > 0).sum()))
    with col4: st.metric("MIBS", int((df["MIBS"] > 0).sum()))
    with col5: st.metric("🎯 Pursuing", pursued_count)

    st.divider()

    tab1, tab2 = st.tabs(["🗺️ Interactive Map", "📊 Table + Export"])

    with tab1:
        st.subheader("Map — click markers for details")
        m = build_map(df)
        if m:
            folium_static(m, width=1200, height=640)
            st.markdown("**Legend:** 🔵 IC, 🟢 BMIC, 🟣 MIBS, 🔴 Mixed")

    with tab2:
        st.subheader("Results Table & Account Details")

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

        st.divider()

        st.subheader("🔍 Account Detail Viewer")
        st.caption("Select an applicant to view detailed 360° information including Form 471 history, C2 budget, and lead tracking.")

        applicant_options = [""] + sorted(df["Name of Applicant"].unique().tolist())
        selected_applicant = st.selectbox(
            "Select an applicant to view details:",
            options=applicant_options,
            key="selected_applicant_detail"
        )

        if selected_applicant and selected_applicant != "":
            applicant_row = df[df["Name of Applicant"] == selected_applicant].iloc[0]

            st.divider()
            show_account_detail(applicant_row)

# -----------------------------------------------------------------------------
# MODE 2: USAC DATA VIEWER
# -----------------------------------------------------------------------------
elif app_mode == "🎓 USAC Data Viewer":

    # SODA3 API Endpoints
    ENDPOINTS = {
        "C2 Budget Tool (FY2021+)": {
            "url": "https://opendata.usac.org/api/v3/views/6brt-5pbv/query.json",
            "description": "C2 budget data for five-year cycles starting FY2021",
            "filters": ["ben", "budget_cycle"]
        },
        "Form 471 Basic Information": {
            "url": "https://opendata.usac.org/api/v3/views/9s6i-myen/query.json",
            "description": "Basic applicant information from FCC Form 471",
            "filters": ["applicant_state", "applicant_name", "funding_year"]
        }
    }

    st.title("🎓 USAC E-Rate Data Viewer")
    st.markdown("Access **C2 Budget** and **Form 471** data via SODA3 API")

    st.sidebar.header("📊 Select Dataset")
    selected_dataset = st.sidebar.selectbox(
        "Choose Dataset",
        list(ENDPOINTS.keys()),
        key="dataset_selector"
    )

    st.sidebar.markdown(f"**Description:** {ENDPOINTS[selected_dataset]['description']}")

    st.sidebar.header("⚙️ Query Options")

    limit = st.sidebar.number_input(
        "Number of records to fetch",
        min_value=10,
        max_value=10000,
        value=100,
        step=10,
        help="Maximum number of records to retrieve"
    )

    offset = st.sidebar.number_input(
        "Starting record (offset)",
        min_value=0,
        max_value=100000,
        value=0,
        step=100,
        help="Use for pagination - skip this many records"
    )

    st.sidebar.header("🔍 Filters")

    filters = {}

    if selected_dataset == "Form 471 Basic Information":
        state_filter = st.sidebar.text_input(
            "Filter by State (e.g., TX, OK)",
            "",
            help="Two-letter state code"
        )
        entity_filter = st.sidebar.text_input(
            "Search Entity Name",
            "",
            help="Partial match - e.g., 'Dallas ISD'"
        )
        funding_year = st.sidebar.text_input(
            "Funding Year (e.g., 2024)",
            "",
            help="Four-digit year"
        )

        if state_filter:
            filters["applicant_state"] = state_filter.upper()
        if entity_filter:
            filters["applicant_name"] = entity_filter
        if funding_year:
            filters["funding_year"] = funding_year

    elif selected_dataset == "C2 Budget Tool (FY2021+)":
        ben_filter = st.sidebar.text_input(
            "BEN Number",
            "",
            help="Billed Entity Number"
        )
        cycle_filter = st.sidebar.text_input(
            "Budget Cycle (e.g., 1, 2)",
            "",
            help="Budget cycle number"
        )

        if ben_filter:
            filters["ben"] = ben_filter
        if cycle_filter:
            filters["budget_cycle"] = cycle_filter

    @st.cache_data(ttl=3600)
    def fetch_soda3_data(endpoint_url, app_token, limit=100, offset=0, filters=None):
        query_body = {
            "limit": limit,
            "offset": offset,
            "order": [{"column": "rowId", "direction": "DESC"}]
        }

        if filters:
            where_conditions = []

            for field, value in filters.items():
                if value:
                    if isinstance(value, str) and not value.isdigit():
                        where_conditions.append({
                            "type": "operator",
                            "operator": "contains",
                            "column": field,
                            "value": value
                        })
                    else:
                        where_conditions.append({
                            "type": "operator",
                            "operator": "=",
                            "column": field,
                            "value": value
                        })

            if where_conditions:
                if len(where_conditions) > 1:
                    query_body["where"] = {
                        "type": "operator",
                        "operator": "AND",
                        "conditions": where_conditions
                    }
                else:
                    query_body["where"] = where_conditions[0]

        headers = {
            "Content-Type": "application/json",
            "X-App-Token": app_token
        }

        try:
            response = requests.post(
                endpoint_url,
                headers=headers,
                json=query_body,
                timeout=60
            )

            response.raise_for_status()

            data = response.json()
            return data, None

        except requests.exceptions.Timeout:
            return None, "Request timed out. The server took too long to respond."
        except requests.exceptions.ConnectionError:
            return None, "Connection error. Please check your internet connection."
        except requests.exceptions.HTTPError as e:
            return None, f"HTTP Error {response.status_code}: {str(e)}"
        except requests.exceptions.RequestException as e:
            return None, f"Request failed: {str(e)}"
        except Exception as e:
            return None, f"Unexpected error: {str(e)}"

    def parse_soda3_response(response_data):
        if not response_data:
            return None

        columns = [col["name"] for col in response_data.get("columns", [])]
        rows = response_data.get("rows", [])

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=columns)

        return df

    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader(f"📁 {selected_dataset}")

    with col2:
        fetch_button = st.button("🔄 Fetch Data", type="primary", use_container_width=True)

    if fetch_button:
        with st.spinner("Fetching data from USAC Open Data API..."):
            endpoint_url = ENDPOINTS[selected_dataset]["url"]

            data, error = fetch_soda3_data(
                endpoint_url=endpoint_url,
                app_token=APP_TOKEN,
                limit=limit,
                offset=offset,
                filters=filters if filters else None
            )

            if error:
                st.error(f"❌ Error fetching data: {error}")
                st.info("""
                💡 **Troubleshooting Tips:**
                - Check your internet connection
                - Verify the app token is correct
                - Try reducing the limit or adjusting filters
                - The API might be temporarily unavailable - try again later
                """)

            elif data is None:
                st.warning("⚠️ No data returned from API.")

            else:
                df = parse_soda3_response(data)

                if df is None or len(df) == 0:
                    st.warning("⚠️ No records found with current filters. Try adjusting your search criteria.")
                else:
                    st.session_state['data'] = df
                    st.session_state['dataset_name'] = selected_dataset
                    st.session_state['fetch_time'] = datetime.now()
                    st.success(f"✅ Successfully fetched {len(df)} records!")

    if 'data' in st.session_state:
        df = st.session_state['data']
        dataset_name = st.session_state.get('dataset_name', 'Dataset')
        fetch_time = st.session_state.get('fetch_time', datetime.now())

        st.markdown("---")
        st.subheader("📈 Summary")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Records", f"{len(df):,}")

        with col2:
            st.metric("Columns", len(df.columns))

        with col3:
            memory_mb = df.memory_usage(deep=True).sum() / 1024**2
            st.metric("Memory Usage", f"{memory_mb:.2f} MB")

        with col4:
            if len(df) > 0 and len(df.columns) > 0:
                null_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
                st.metric("Completeness", f"{100-null_pct:.1f}%")
            else:
                st.metric("Completeness", "N/A")

        st.caption(f"Last fetched: {fetch_time.strftime('%Y-%m-%d %I:%M:%S %p')}")

        st.markdown("---")
        st.subheader("📋 Data Preview")

        with st.expander("🔧 Customize View & Search"):
            col_a, col_b = st.columns(2)

            with col_a:
                col_select_all = st.checkbox("Select All Columns", value=True)

                if not col_select_all:
                    available_cols = df.columns.tolist()
                    selected_cols = st.multiselect(
                        "Choose columns to display",
                        available_cols,
                        default=available_cols[:min(10, len(available_cols))]
                    )
                    if selected_cols:
                        df_display = df[selected_cols].copy()
                    else:
                        df_display = df.copy()
                else:
                    df_display = df.copy()

            with col_b:
                search_term = st.text_input("🔍 Search in data", "")
                if search_term:
                    mask = df_display.astype(str).apply(
                        lambda x: x.str.contains(search_term, case=False, na=False)
                    ).any(axis=1)
                    df_display = df_display[mask]
                    st.info(f"Found {len(df_display)} records matching '{search_term}'")

        st.dataframe(
            df_display,
            use_container_width=True,
            height=500
        )

        st.markdown("---")
        st.subheader("💾 Export Data")

        col1, col2, col3 = st.columns(3)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_prefix = dataset_name.replace(' ', '_').replace('(', '').replace(')', '')

        with col1:
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name=f"{file_prefix}_{timestamp}.csv",
                mime="text/csv",
                use_container_width=True
            )

        with col2:
            json_str = df.to_json(orient='records', indent=2)
            st.download_button(
                label="📥 Download as JSON",
                data=json_str,
                file_name=f"{file_prefix}_{timestamp}.json",
                mime="application/json",
                use_container_width=True
            )

        with col3:
            try:
                buffer = BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False, sheet_name='Data')

                st.download_button(
                    label="📥 Download as Excel",
                    data=buffer.getvalue(),
                    file_name=f"{file_prefix}_{timestamp}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Excel export error: {str(e)}")
                st.info("Make sure openpyxl is installed: `pip install openpyxl`")

        col_exp1, col_exp2 = st.columns(2)

        with col_exp1:
            with st.expander("🔍 View Detailed Record"):
                if len(df) > 0:
                    record_index = st.number_input(
                        "Select record number",
                        min_value=0,
                        max_value=len(df)-1,
                        value=0,
                        key="record_viewer"
                    )
                    st.json(df.iloc[record_index].to_dict())
                else:
                    st.info("No records to display")

        with col_exp2:
            with st.expander("📊 Column Statistics"):
                if len(df) > 0:
                    st.dataframe(df.describe(include='all'), use_container_width=True)
                else:
                    st.info("No data for statistics")

    else:
        st.info("👈 Configure your query using the sidebar and click **'Fetch Data'** to begin.")

        with st.expander("ℹ️ How to Use This App"):
            st.markdown("""
            ### 📖 Quick Start Guide

            **1. Select a Dataset**
            - Choose from the dropdown in the sidebar
            - Two datasets available: C2 Budget Tool and Form 471 Basic Information

            **2. Configure Query Options**
            - **Limit**: Number of records to fetch (10-10,000)
            - **Offset**: Starting position for pagination (use 0, 100, 200, etc.)

            **3. Add Filters (Optional)**

            For **Form 471**:
            - State (e.g., TX, OK)
            - Entity Name (partial match, e.g., "Dallas")
            - Funding Year (e.g., 2024)

            For **C2 Budget**:
            - BEN Number (Billed Entity Number)
            - Budget Cycle (1, 2, etc.)

            **4. Fetch Data**
            - Click the "🔄 Fetch Data" button
            - Data is cached for 1 hour for faster subsequent queries

            **5. Explore Results**
            - View summary metrics
            - Search within the data
            - Select specific columns to display
            - Export to CSV, JSON, or Excel
            - View detailed individual records
            - See column statistics
            """)
