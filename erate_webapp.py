"""
Texas E-Rate Form 470 Analyzer - 470-Only (No APIs)
"""

import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import time
import os
from io import BytesIO

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="E-Rate Opportunity Finder - 470 Only",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CONSTANTS
# ============================================================================

GEOCODE_CACHE_FILE = 'geocode_cache.csv'
TARGET_STATES = {"TX", "OK"}  # Only show TX and OK

# ============================================================================
# HELPERS
# ============================================================================

@st.cache_data
def load_data(uploaded_file):
    return pd.read_excel(uploaded_file)

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Lowercase + strip for matching; keep original for values
    raw_cols = list(df.columns)
    norm = {c: str(c).strip().lower() for c in raw_cols}

    # Best-effort alias map for FFL exports
    aliases = {
        # applicant
        "name of applicant": ["name of applicant", "applicant name", "ben name", "entity", "district"],
        # location
        "city": ["city", "city/town", "applicant city"],
        "state": ["state", "st", "applicant state"],
        # 470 number
        "470 app number": ["470 app number", "form 470 number", "470 number", "470#", "application number"],
        # service type
        "service type": ["service type", "service type (470)", "category", "category of service"],
        # equipment/function
        "function": ["function", "equipment", "equipment type", "product function"],
        # manufacturer/brand
        "manufacturer": ["manufacturer", "brand", "vendor/brand", "make"],
    }

    # Build reverse lookup: current_col -> canonical
    rename_map = {}
    for canonical, alts in aliases.items():
        found = next((orig for orig in raw_cols if norm[orig] in alts), None)
        if found:
            rename_map[found] = canonical

    df = df.rename(columns=rename_map)

    # Ensure required columns exist (empty if missing)
    for required in ["name of applicant", "city", "state", "service type", "470 app number", "function", "manufacturer"]:
        if required not in df.columns:
            df[required] = None

    # Normalize service type values a bit
    def norm_service(v):
        s = str(v).strip().lower() if pd.notna(v) else ""
        if not s:
            return None
        if "internal connections" in s:
            return "Internal Connections"
        if "basic maintenance" in s:
            return "Basic Maintenance of Internal Connections"
        if "managed internal broadband" in s:
            return "Managed Internal Broadband Services"
        if "data transmission" in s or "internet" in s:
            return "Data Transmission and/or Internet Access"
        return v  # leave as-is if unknown

    df["service type"] = df["service type"].apply(norm_service)

    # Standardize final column names to what the app expects
    final_map = {
        "name of applicant": "Name of Applicant",
        "city": "City",
        "state": "State",
        "service type": "Service Type",
        "470 app number": "470 App Number",
        "function": "Function",
        "manufacturer": "Manufacturer",
    }
    return df.rename(columns=final_map)

def load_cache():
    if os.path.exists(GEOCODE_CACHE_FILE):
        try:
            return pd.read_csv(GEOCODE_CACHE_FILE)
        except:
            return pd.DataFrame(columns=['Name', 'City', 'State', 'Latitude', 'Longitude'])
    return pd.DataFrame(columns=['Name', 'City', 'State', 'Latitude', 'Longitude'])

def save_cache(cache):
    cache.to_csv(GEOCODE_CACHE_FILE, index=False)

def construct_form_470_url(app_number):
    if pd.isna(app_number):
        return None
    try:
        app_num = int(app_number)
    except Exception:
        try:
            app_num = int(float(app_number))
        except Exception:
            return None
    return f"http://legacy.fundsforlearning.com/470/{app_num}"

def matches_filter(value, filter_list):
    if not filter_list:
        return True
    if pd.isna(value):
        return False
    value_lower = str(value).lower()
    return any(term.lower() in value_lower for term in filter_list)

def geocode_location(name, city, state, cache, geolocator):
    cached = cache[(cache['Name'] == name) & (cache['City'] == city) & (cache['State'] == state)]
    if not cached.empty:
        return cached.iloc[0]['Latitude'], cached.iloc[0]['Longitude']
    try:
        search_query = f"{city}, {state}"
        location = geolocator.geocode(search_query, timeout=10)
        if location:
            return location.latitude, location.longitude
        else:
            return None, None
    except:
        return None, None

def apply_filters(df, function_filters, manufacturer_filters):
    if not function_filters and not manufacturer_filters:
        return df
    if 'Name of Applicant' not in df.columns:
        return df
    applicant_groups = df.groupby('Name of Applicant')
    filtered_applicants = []

    for applicant_name, group in applicant_groups:
        function_match = True
        if function_filters:
            if 'Function' in group.columns:
                function_match = group['Function'].apply(lambda x: matches_filter(x, function_filters)).any()
            else:
                function_match = False

        manufacturer_match = True
        if manufacturer_filters:
            if 'Manufacturer' in group.columns:
                manufacturer_match = group['Manufacturer'].apply(lambda x: matches_filter(x, manufacturer_filters)).any()
            else:
                manufacturer_match = False

        if function_match and manufacturer_match:
            filtered_applicants.append(applicant_name)

    return df[df['Name of Applicant'].isin(filtered_applicants)].copy()

def aggregate_by_applicant(df):
    aggregated_data = []
    if 'Name of Applicant' not in df.columns:
        return pd.DataFrame()

    for applicant_name, group in df.groupby('Name of Applicant'):
        city = group['City'].iloc[0] if 'City' in group.columns else ''
        state = group['State'].iloc[0] if 'State' in group.columns else ''

        if 'Service Type' in group.columns:
            service_counts = group['Service Type'].value_counts().to_dict()
        else:
            service_counts = {}
        ic_count  = service_counts.get('Internal Connections', 0)
        bm_count  = service_counts.get('Basic Maintenance of Internal Connections', 0)
        mb_count  = service_counts.get('Managed Internal Broadband Services', 0)
        dia_count = service_counts.get('Data Transmission and/or Internet Access', 0)

        app_number = group['470 App Number'].iloc[0] if '470 App Number' in group.columns else None
        form_url = construct_form_470_url(app_number)

        functions = group['Function'].dropna().unique().tolist() if 'Function' in group.columns else []
        manufacturers = group['Manufacturer'].dropna().unique().tolist() if 'Manufacturer' in group.columns else []

        functions_str = ', '.join([str(f) for f in functions if f]) if functions else ''
        manufacturers_str = ', '.join([str(m) for m in manufacturers if m]) if manufacturers else ''

        aggregated_data.append({
            'Name of Applicant': applicant_name,
            'City': city,
            'State': state,
            '470s_IC': ic_count,
            '470s_BM': bm_count,
            '470s_MB': mb_count,
            '470s_DIA': dia_count,
            'Form_470_URL': form_url,
            'Functions': functions_str,
            'Manufacturers': manufacturers_str
        })

    return pd.DataFrame(aggregated_data)

def color_for_services(row):
    has_ic = row.get('470s_IC', 0) > 0
    has_bm = row.get('470s_BM', 0) > 0
    has_mb = row.get('470s_MB', 0) > 0
    has_dia = row.get('470s_DIA', 0) > 0

    # Prioritize multi-service colors
    if has_ic and has_bm and has_mb:
        return 'darkred'
    if has_ic and has_mb and not has_bm:
        return 'cadetblue'
    if has_ic and has_bm and not has_mb:
        return 'orange'
    if has_bm and has_mb and not has_ic:
        return 'lightgreen'
    # Single-service
    if has_ic:
        return 'blue'
    if has_bm:
        return 'green'
    if has_mb:
        return 'purple'
    if has_dia:
        return 'gray'  # keep DIA neutral
    return 'gray'

def create_popup(row):
    functions_display = row['Functions'][:150] + '...' if len(row['Functions']) > 150 else row['Functions']
    manufacturers_display = row['Manufacturers'][:150] + '...' if len(row['Manufacturers']) > 150 else row['Manufacturers']

    popup_html = f"""
    <div style="width: 400px; font-family: Arial, sans-serif;">
        <h3 style="margin: 0 0 10px 0; color: #0066cc;">{row['Name of Applicant']}</h3>
        <p style="margin: 5px 0;"><b>📍 Location:</b> {row['City']}, {row['State']}</p>

        <hr style="margin: 15px 0; border: 1px solid #ddd;">

        <h4 style="margin: 10px 0 5px 0; color: #0066cc;">📋 Current Form 470 RFP</h4>
        <p style="margin: 5px 0;"><b>470s Submitted:</b></p>
        <ul style="margin: 5px 0; padding-left: 20px; font-size: 13px;">
            <li>Internal Connections: {int(row['470s_IC'])}</li>
            <li>Basic Maintenance: {int(row['470s_BM'])}</li>
            <li>Managed Broadband: {int(row['470s_MB'])}</li>
            <li>Data/Internet: {int(row['470s_DIA'])}</li>
        </ul>
        <p style="margin: 5px 0; font-size: 12px;"><b>Equipment:</b> {functions_display}</p>
        <p style="margin: 5px 0; font-size: 12px;"><b>Manufacturers:</b> {manufacturers_display}</p>
    """
    popup_html += f"""
        <hr style="margin: 15px 0; border: 1px solid #ddd;">
        <p style="margin: 5px 0; text-align: center;">
            <a href="{row['Form_470_URL']}" target="_blank" 
               style="background-color: #0066cc; color: white; padding: 10px 20px; 
                      text-decoration: none; border-radius: 5px; display: inline-block; font-weight: bold;">
                📋 View Form 470 RFP →
            </a>
        </p>
    </div>
    """
    return popup_html

def create_map(filtered_df):
    if len(filtered_df) == 0:
        return None

    center_lat = filtered_df['Latitude'].mean()
    center_lon = filtered_df['Longitude'].mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=6, tiles='OpenStreetMap')

    for _, row in filtered_df.iterrows():
        color = color_for_services(row)
        popup_html = create_popup(row)

        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=folium.Popup(popup_html, max_width=450),
            tooltip=f"{row['Name of Applicant']} - Click for details"
        ).add_to(m)

        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=8,
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.6,
            weight=2
        ).add_to(m)

    return m

# ============================================================================
# MAIN
# ============================================================================

def main():
    st.title("📄 Texas & Oklahoma E-Rate Opportunity Finder (470 Only)")
    st.caption("Upload your Funds For Learning Excel file (no APIs).")
    st.markdown("---")

    with st.sidebar:
        st.header("📁 Upload 470 Excel")
        uploaded_file = st.file_uploader(
            "Upload E-Rate Excel File (Funds For Learning export)",
            type=['xls', 'xlsx'],
        )

        st.markdown("---")
        st.header("🔍 Filters")

        # These options will be filled dynamically after upload
        name_search = st.text_input("Applicant name contains (optional)")

        selected_service_types = st.multiselect(
            "Service Types",
            ["Internal Connections", "Basic Maintenance of Internal Connections",
             "Managed Internal Broadband Services", "Data Transmission and/or Internet Access"],
            help="Filter by 470 service categories"
        )

        selected_functions = st.multiselect("Equipment Type (Function)", [], help="Populated after upload")
        selected_manufacturers = st.multiselect("Manufacturers", [], help="Populated after upload")

        st.markdown("---")
        analyze_button = st.button("🚀 Analyze", type="primary", use_container_width=True)

    if uploaded_file is None:
        st.info("👈 Upload your Excel file to begin.")
        return

    if analyze_button:
        with st.spinner("🔄 Processing Form 470 data..."):
            try:
                df = load_data(uploaded_file)
                df = normalize_columns(df)

                # Keep only TX and OK
                if 'State' not in df.columns:
                    st.error("The uploaded file must include a 'State' column.")
                    return
                df = df[df['State'].isin(TARGET_STATES)].copy()

                # Optional service type pre-filter
                if selected_service_types and 'Service Type' in df.columns:
                    df = df[df['Service Type'].isin(selected_service_types)].copy()

                # Dynamic options after load
                if 'Function' in df.columns:
                    st.sidebar.multiselect("Equipment Type (Function)", [], default=None)
                if 'Manufacturer' in df.columns:
                    st.sidebar.multiselect("Manufacturers", [], default=None)

                # Sidebar options populate
                funcs = sorted([x for x in df['Function'].dropna().unique().tolist()]) if 'Function' in df.columns else []
                mans = sorted([x for x in df['Manufacturer'].dropna().unique().tolist()]) if 'Manufacturer' in df.columns else []

                # Rerender with choices (Streamlit quirk: show available sets)
                selected_functions[:] = selected_functions  # no-op to keep type
                selected_manufacturers[:] = selected_manufacturers  # no-op

                # Apply function/manufacturer filters by applicant
                df = apply_filters(df, selected_functions, selected_manufacturers)

                if name_search:
                    df = df[df['Name of Applicant'].str.contains(name_search, case=False, na=False)]

                if len(df) == 0:
                    st.warning("No applicants match your criteria.")
                    return

                # Aggregate by applicant
                aggregated = aggregate_by_applicant(df)

                # Geocode
                cache = load_cache()
                geolocator = Nominatim(user_agent="erate_470_only_app")
                progress_bar = st.progress(0)
                latitudes = []
                longitudes = []

                for idx, row in aggregated.iterrows():
                    lat, lon = geocode_location(
                        row['Name of Applicant'],
                        row['City'],
                        row['State'],
                        cache,
                        geolocator
                    )
                    latitudes.append(lat)
                    longitudes.append(lon)

                    if lat is not None and lon is not None:
                        new_entry = pd.DataFrame([{
                            'Name': row['Name of Applicant'],
                            'City': row['City'],
                            'State': row['State'],
                            'Latitude': lat,
                            'Longitude': lon
                        }])
                        cache = pd.concat([cache, new_entry], ignore_index=True)
                        cache = cache.drop_duplicates(subset=['Name', 'City', 'State'], keep='last')

                    progress_bar.progress((idx + 1) / len(aggregated))
                    time.sleep(1)  # be kind to the geocoder

                save_cache(cache)

                aggregated['Latitude'] = latitudes
                aggregated['Longitude'] = longitudes
                aggregated = aggregated.dropna(subset=['Latitude', 'Longitude'])

                if len(aggregated) == 0:
                    st.warning("No applicants could be geocoded.")
                    return

                st.session_state['results'] = aggregated
                st.session_state['functions_options'] = funcs
                st.session_state['manufacturers_options'] = mans
                st.success(f"✅ Analysis complete! Found {len(aggregated)} opportunities.")

            except Exception as e:
                st.error(f"Error processing data: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
                return

    if 'results' in st.session_state:
        filtered = st.session_state['results']
        funcs = st.session_state.get('functions_options', [])
        mans = st.session_state.get('manufacturers_options', [])

        # Show dynamic filter choices once data is loaded
        with st.sidebar:
            if funcs:
                st.write("Available Functions:")
                st.caption(", ".join(funcs[:30]) + (" ..." if len(funcs) > 30 else ""))
            if mans:
                st.write("Available Manufacturers:")
                st.caption(", ".join(mans[:30]) + (" ..." if len(mans) > 30 else ""))

        # Summary metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Opportunities", len(filtered))
        with col2:
            ic_count = (filtered['470s_IC'] > 0).sum()
            st.metric("Internal Connections", ic_count)

        st.markdown("---")

        tab1, tab2 = st.tabs(["🗺️ Interactive Map", "📊 Data Table"])

        with tab1:
            st.subheader("Interactive Map - Click markers for details")
            map_obj = create_map(filtered)
            if map_obj:
                folium_static(map_obj, width=1200, height=600)

                st.markdown("### 🎨 Service Type Color Legend")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown("🔵 **Blue** = Internal Connections")
                    st.markdown("🟠 **Orange** = IC + BM")
                with col2:
                    st.markdown("🟢 **Green** = Basic Maintenance")
                    st.markdown("🔷 **Cadet Blue** = IC + MB")
                with col3:
                    st.markdown("🟣 **Purple** = Managed Broadband")
                    st.markdown("🔴 **Dark Red** = All Three Services")

        with tab2:
            st.subheader("Opportunity Details (470 Only)")
            display_rows = []
            for _, row in filtered.iterrows():
                display_rows.append({
                    'Name of Applicant': row['Name of Applicant'],
                    'City': row['City'],
                    'State': row['State'],
                    '470s_IC': int(row['470s_IC']),
                    '470s_BM': int(row['470s_BM']),
                    '470s_MB': int(row['470s_MB']),
                    '470s_DIA': int(row['470s_DIA']),
                    'Total_470s': int(row['470s_IC'] + row['470s_BM'] + row['470s_MB'] + row['470s_DIA']),
                    'Functions': row['Functions'],
                    'Manufacturers': row['Manufacturers'],
                    'Form_470_URL': row['Form_470_URL']
                })
            display_df = pd.DataFrame(display_rows).sort_values('Total_470s', ascending=False)
            st.dataframe(
                display_df,
                use_container_width=True,
                column_config={"Form_470_URL": st.column_config.LinkColumn("Form 470 Link")}
            )
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                display_df.to_excel(writer, index=False, sheet_name='Opportunities')
            output.seek(0)
            st.download_button(
                label="📥 Download 470-Only Excel Report",
                data=output,
                file_name="erate_470_only_opportunities.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary"
            )

if __name__ == "__main__":
    main()
