"""
Texas E-Rate Form 470 Analyzer - 470-Only Version (No Form 471)
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
import requests
from datetime import datetime, timedelta

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

LAT_WACO = 31.55
LON_I35E = -97.0
GEOCODE_CACHE_FILE = 'geocode_cache.csv'
FORM_470_API_URL = os.getenv('USAC_470_DATASET_URL', 'https://opendata.usac.org/resource/jp7a-89nd.json')

# ============================================================================
# FORM 470 API (USAC SODA)
# ============================================================================

@st.cache_data(ttl=604800)
def fetch_form_470_api_data(limit=50000, years_back=3, state="TX"):
    """
    Fetch Form 470 summary data from USAC SODA dataset for recent years and a state.
    Maps to the app's expected columns for downstream processing.
    """
    try:
        current_year = datetime.now().year
        start_year = current_year - years_back
        headers = {}
        app_token = os.getenv("USAC_APP_TOKEN")
        if app_token:
            headers["X-App-Token"] = app_token

        page_size = min(limit, 50000)
        rows = []

        for year in range(start_year, current_year + 1):
            offset = 0
            while True:
                params = {
                    "$limit": page_size,
                    "$offset": offset,
                    "$select": ",".join([
                        "billed_entity_name",
                        "billed_entity_city",
                        "billed_entity_state",
                        "application_number",
                        "funding_year",
                        "category_one_description",
                        "category_two_description"
                    ]),
                    "billed_entity_state": state,
                    "funding_year": year,
                }
                resp = requests.get(FORM_470_API_URL, params=params, headers=headers, timeout=30)
                if resp.status_code >= 400:
                    break
                chunk = resp.json()
                if not chunk:
                    break
                rows.extend(chunk)
                if len(chunk) < page_size:
                    break
                offset += page_size

        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame()

        output_rows = []
        for _, r in df.iterrows():
            name = r.get("billed_entity_name") or "Unknown"
            city = r.get("billed_entity_city") or ""
            st_code = r.get("billed_entity_state") or state
            app_no = r.get("application_number")
            c1 = r.get("category_one_description")
            c2 = r.get("category_two_description")

            service_types = []
            if isinstance(c2, str) and c2.strip():
                service_types.append("Internal Connections")
            if isinstance(c1, str) and c1.strip():
                service_types.append("Data Transmission and/or Internet Access")
            if not service_types:
                service_types.append("Unknown")

            for stype in service_types:
                output_rows.append({
                    "Name of Applicant": name,
                    "City": city,
                    "State": st_code,
                    "470 App Number": app_no,
                    "Service Type": stype,
                    "Function": "",
                    "Manufacturer": "",
                })

        out = pd.DataFrame(output_rows)
        if out.empty:
            return out
        return out.dropna(subset=["Name of Applicant"]).drop_duplicates()

    except Exception as e:
        st.warning(f"Could not fetch Form 470 API data: {str(e)}")
        return pd.DataFrame()

# ============================================================================
# EXISTING HELPER FUNCTIONS
# ============================================================================

@st.cache_data
def load_data(uploaded_file):
    return pd.read_excel(uploaded_file)

def load_cache():
    if os.path.exists(GEOCODE_CACHE_FILE):
        try:
            return pd.read_csv(GEOCODE_CACHE_FILE)
        except:
            return pd.DataFrame(columns=['Name', 'City', 'State', 'Latitude', 'Longitude'])
    return pd.DataFrame(columns=['Name', 'City', 'State', 'Latitude', 'Longitude'])

def save_cache(cache):
    cache.to_csv(GEOCODE_CACHE_FILE, index=False)

def construct_form_470_url(app_number, funding_year=2025):
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
    return any(filter_term.lower() in value_lower for filter_term in filter_list)

def geocode_location(name, city, state, cache, geolocator):
    cached = cache[
        (cache['Name'] == name) & 
        (cache['City'] == city) & 
        (cache['State'] == state)
    ]
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
        ic_count = service_counts.get('Internal Connections', 0)
        bm_count = service_counts.get('Basic Maintenance of Internal Connections', 0)
        mb_count = service_counts.get('Managed Internal Broadband Services', 0)
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
    if has_ic and not has_bm and not has_mb:
        return 'blue'
    if has_bm and not has_ic and not has_mb:
        return 'green'
    if has_mb and not has_ic and not has_bm:
        return 'purple'
    if has_ic and has_bm and not has_mb:
        return 'orange'
    if has_ic and has_mb and not has_bm:
        return 'cadetblue'
    if has_bm and has_mb and not has_ic:
        return 'lightgreen'
    if has_ic and has_bm and has_mb:
        return 'darkred'
    return 'gray'

# ============================================================================
# MAP / POPUP (470 ONLY)
# ============================================================================

def create_popup_470_only(row):
    functions_display = row['Functions'][:150] + '...' if len(row['Functions']) > 150 else row['Functions']
    manufacturers_display = row['Manufacturers'][:150] + '...' if len(row['Manufacturers']) > 150 else row['Manufacturers']
    popup_html = f"""
    <div style="width: 400px; font-family: Arial, sans-serif;">
        <h3 style="margin: 0 0 10px 0; color: #0066cc;">{row['Name of Applicant']}</h3>
        <p style="margin: 5px 0;"><b>📍 Location:</b> {row['City']}, TX</p>
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
    m = folium.Map(location=[center_lat, center_lon], zoom_start=7, tiles='OpenStreetMap')
    folium.PolyLine(locations=[[LAT_WACO, -100], [LAT_WACO, -94]], color='red', weight=2, opacity=0.7, popup='Waco Boundary (North)').add_to(m)
    folium.PolyLine(locations=[[28, LON_I35E], [36, LON_I35E]], color='red', weight=2, opacity=0.7, popup='I-35E Boundary (West)').add_to(m)
    for _, row in filtered_df.iterrows():
        color = color_for_services(row)
        popup_html = create_popup_470_only(row)
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
# MAIN APPLICATION (470 ONLY)
# ============================================================================

def main():
    st.title("📄 Texas E-Rate Opportunity Finder (470 Only)")
    st.markdown("**470 Source:** Upload Excel or fetch from USAC SODA (beta)")
    st.markdown("---")

    with st.sidebar:
        st.header("📁 Form 470 Source")
        source_choice = st.radio(
            "Form 470 Data Source",
            options=["Upload Excel (recommended)", "USAC 470 API (beta)"],
            help="Use your Funds for Learning export or fetch basic 470s from USAC API"
        )
        uploaded_file = None
        if source_choice == "Upload Excel (recommended)":
            uploaded_file = st.file_uploader(
                "Upload E-Rate Excel File",
                type=['xls', 'xlsx'],
                help="Upload your Funds for Learning export file"
            )

        st.markdown("---")
        st.header("🔍 Form 470 Filters")
        st.subheader("Equipment Type")
        function_options = [
            "Wireless Access Points",
            "Switches",
            "Firewall",
            "Cabling",
            "UPS/Battery Backup",
            "Routers",
            "Wireless Controllers"
        ]
        selected_functions = st.multiselect(
            "Select equipment types (leave empty for all)",
            function_options,
            help="Filter by equipment type you sell"
        )

        st.subheader("Manufacturers")
        manufacturer_options = [
            "Aruba",
            "Cisco",
            "Fortinet",
            "Juniper",
            "Meraki",
            "Ruckus",
            "Ubiquiti",
            "Extreme Networks"
        ]
        selected_manufacturers = st.multiselect(
            "Select manufacturers (leave empty for all)",
            manufacturer_options,
            help="Filter by brands you sell"
        )

        st.markdown("---")
        st.header("🗺️ Territory")
        st.info(f"**North of:** Waco (Lat {LAT_WACO})\n\n**West of:** I-35E (Lon {LON_I35E})")

        st.markdown("---")
        analyze_button = st.button("🚀 Analyze Opportunities", type="primary", use_container_width=True)

    if source_choice == "Upload Excel (recommended)" and uploaded_file is None:
        st.info("👈 Please upload an E-Rate Excel file to get started, or switch to 'USAC 470 API (beta)' to run without a file.")
        return

    if analyze_button:
        with st.spinner("🔄 Processing Form 470 data..."):
            try:
                # Load Form 470 data
                if source_choice == "Upload Excel (recommended)":
                    df = load_data(uploaded_file)
                else:
                    df = fetch_form_470_api_data()

                # Filter to Texas
                if 'State' in df.columns:
                    texas_df = df[df['State'] == 'TX'].copy()
                else:
                    texas_df = df.copy()

                if len(texas_df) == 0:
                    st.error("No Texas applicants found in the 470 data!")
                    return

                # Apply filters
                if selected_functions or selected_manufacturers:
                    filtered_texas = apply_filters(texas_df, selected_functions, selected_manufacturers)
                else:
                    filtered_texas = texas_df

                if len(filtered_texas) == 0:
                    st.warning("No applicants match your Form 470 filter criteria. Try adjusting your filters.")
                    return

                # Aggregate by applicant
                aggregated = aggregate_by_applicant(filtered_texas)

                # Geocode
                cache = load_cache()
                geolocator = Nominatim(user_agent="texas_erate_webapp_470_only")
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
                    time.sleep(1)
                save_cache(cache)

                aggregated['Latitude'] = latitudes
                aggregated['Longitude'] = longitudes
                aggregated = aggregated.dropna(subset=['Latitude', 'Longitude'])

                # Territory filter
                filtered = aggregated[(aggregated['Latitude'] > LAT_WACO) & (aggregated['Longitude'] < LON_I35E)].copy()
                if len(filtered) == 0:
                    st.warning("No applicants found in your territory after geographic filtering.")
                    return

                st.session_state['results'] = filtered
                st.success(f"✅ Analysis complete! Found {len(filtered)} opportunities.")

            except Exception as e:
                st.error(f"Error processing data: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
                return

    if 'results' in st.session_state:
        filtered = st.session_state['results']

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
                    st.markdown("🟩 **Light Green** = BM + MB")
                with col4:
                    st.markdown("🔴 **Dark Red** = All Three Services")

        with tab2:
            st.subheader("Opportunity Details (470 Only)")
            display_rows = []
            for _, row in filtered.iterrows():
                display_rows.append({
                    'Name of Applicant': row['Name of Applicant'],
                    'City': row['City'],
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
                column_config={
                    "Form_470_URL": st.column_config.LinkColumn("Form 470 Link")
                }
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
