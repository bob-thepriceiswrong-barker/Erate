"""
Texas E-Rate Form 470 Analyzer - Web Application
User-friendly interface for non-technical users
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
    page_title="E-Rate Opportunity Finder",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CONSTANTS
# ============================================================================

LAT_WACO = 31.55
LON_I35E = -97.0
CACHE_FILE = 'geocode_cache.csv'

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_data
def load_data(uploaded_file):
    """Load Excel data from uploaded file"""
    return pd.read_excel(uploaded_file)

def load_cache():
    """Load geocoding cache"""
    if os.path.exists(CACHE_FILE):
        try:
            return pd.read_csv(CACHE_FILE)
        except:
            return pd.DataFrame(columns=['Name', 'City', 'State', 'Latitude', 'Longitude'])
    return pd.DataFrame(columns=['Name', 'City', 'State', 'Latitude', 'Longitude'])

def save_cache(cache):
    """Save geocoding cache"""
    cache.to_csv(CACHE_FILE, index=False)

def construct_form_470_url(app_number, funding_year=2025):
    """Construct Form 470 URL"""
    if pd.isna(app_number):
        return None
    app_num = int(app_number)
    return f"https://data.usac.org/publicreports/Forms/Form470Rfp/Index?FundingYear={funding_year}&ApplicationNumber={app_num}"

def matches_filter(value, filter_list):
    """Check if value matches any filter term"""
    if not filter_list:
        return True
    if pd.isna(value):
        return False
    value_lower = str(value).lower()
    return any(filter_term.lower() in value_lower for filter_term in filter_list)

def geocode_location(name, city, state, cache, geolocator):
    """Geocode a location"""
    # Check cache
    cached = cache[
        (cache['Name'] == name) & 
        (cache['City'] == city) & 
        (cache['State'] == state)
    ]
    
    if not cached.empty:
        return cached.iloc[0]['Latitude'], cached.iloc[0]['Longitude']
    
    # Try geocoding
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
    """Apply function and manufacturer filters"""
    if not function_filters and not manufacturer_filters:
        return df
    
    applicant_groups = df.groupby('Name of Applicant')
    filtered_applicants = []
    
    for applicant_name, group in applicant_groups:
        # Check function filter
        function_match = True
        if function_filters:
            function_match = group['Function'].apply(lambda x: matches_filter(x, function_filters)).any()
        
        # Check manufacturer filter
        manufacturer_match = True
        if manufacturer_filters:
            manufacturer_match = group['Manufacturer'].apply(lambda x: matches_filter(x, manufacturer_filters)).any()
        
        if function_match and manufacturer_match:
            filtered_applicants.append(applicant_name)
    
    return df[df['Name of Applicant'].isin(filtered_applicants)].copy()

def aggregate_by_applicant(df):
    """Aggregate line items by applicant"""
    aggregated_data = []
    
    for applicant_name, group in df.groupby('Name of Applicant'):
        city = group['City'].iloc[0]
        state = group['State'].iloc[0]
        
        service_counts = group['Service Type'].value_counts().to_dict()
        ic_count = service_counts.get('Internal Connections', 0)
        bm_count = service_counts.get('Basic Maintenance of Internal Connections', 0)
        mb_count = service_counts.get('Managed Internal Broadband Services', 0)
        dia_count = service_counts.get('Data Transmission and/or Internet Access', 0)
        
        app_number = group['470 App Number'].iloc[0]
        form_url = construct_form_470_url(app_number)
        
        functions = group['Function'].dropna().unique().tolist()
        manufacturers = group['Manufacturer'].dropna().unique().tolist()
        
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
    """Assign color based on services"""
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

def create_map(filtered_df):
    """Create interactive map"""
    if len(filtered_df) == 0:
        return None
    
    center_lat = filtered_df['Latitude'].mean()
    center_lon = filtered_df['Longitude'].mean()
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=7,
        tiles='OpenStreetMap'
    )
    
    # Territory boundaries
    folium.PolyLine(
        locations=[[LAT_WACO, -100], [LAT_WACO, -94]],
        color='red',
        weight=2,
        opacity=0.7,
        popup='Waco Boundary'
    ).add_to(m)
    
    folium.PolyLine(
        locations=[[28, LON_I35E], [36, LON_I35E]],
        color='red',
        weight=2,
        opacity=0.7,
        popup='I-35E Boundary'
    ).add_to(m)
    
    # Add markers
    for idx, row in filtered_df.iterrows():
        color = color_for_services(row)
        
        functions_display = row['Functions'][:200] + '...' if len(row['Functions']) > 200 else row['Functions']
        manufacturers_display = row['Manufacturers'][:200] + '...' if len(row['Manufacturers']) > 200 else row['Manufacturers']
        
        popup_html = f"""
        <div style="width: 350px;">
            <h4 style="margin: 0 0 10px 0;">{row['Name of Applicant']}</h4>
            <p style="margin: 5px 0;"><b>Location:</b> {row['City']}, TX</p>
            <hr style="margin: 10px 0;">
            <p style="margin: 5px 0;"><b>470s Submitted:</b></p>
            <ul style="margin: 5px 0; padding-left: 20px;">
                <li>Internal Connections: {int(row['470s_IC'])}</li>
                <li>Basic Maintenance: {int(row['470s_BM'])}</li>
                <li>Managed Broadband: {int(row['470s_MB'])}</li>
                <li>Data/Internet: {int(row['470s_DIA'])}</li>
            </ul>
            <hr style="margin: 10px 0;">
            <p style="margin: 5px 0;"><b>Functions:</b><br><small>{functions_display}</small></p>
            <p style="margin: 5px 0;"><b>Manufacturers:</b><br><small>{manufacturers_display}</small></p>
            <hr style="margin: 10px 0;">
            <p style="margin: 5px 0; text-align: center;">
                <a href="{row['Form_470_URL']}" target="_blank" 
                   style="background-color: #0066cc; color: white; padding: 8px 16px; 
                          text-decoration: none; border-radius: 4px; display: inline-block;">
                    📋 View Form 470 →
                </a>
            </p>
        </div>
        """
        
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=8,
            popup=folium.Popup(popup_html, max_width=400),
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.7
        ).add_to(m)
    
    return m

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Header
    st.title("🎯 Texas E-Rate Opportunity Finder")
    st.markdown("**Find and analyze E-Rate Form 470 opportunities in your territory**")
    st.markdown("---")
    
    # Sidebar - File Upload and Filters
    with st.sidebar:
        st.header("📁 Upload Data")
        uploaded_file = st.file_uploader(
            "Upload E-Rate Excel File",
            type=['xls', 'xlsx'],
            help="Upload your Funds for Learning export file"
        )
        
        st.markdown("---")
        st.header("🔍 Filters")
        
        # Function filter
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
        
        # Manufacturer filter
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
    
    # Main content area
    if uploaded_file is None:
        st.info("👈 Please upload an E-Rate Excel file to get started")
        
        # Show instructions
        with st.expander("📖 How to Use This Tool"):
            st.markdown("""
            ### Getting Started
            1. **Upload** your E-Rate data file (from Funds for Learning)
            2. **Select filters** (optional) to narrow down opportunities
            3. **Click "Analyze"** to process the data
            4. **View results** on the interactive map
            5. **Click markers** to see details and access Form 470s
            6. **Download** the results as Excel
            
            ### Filters
            - **Equipment Type:** Filter by specific equipment (wireless, switches, firewalls, etc.)
            - **Manufacturers:** Filter by brands you sell (Aruba, Cisco, Fortinet, etc.)
            - Leave filters empty to see all opportunities
            
            ### Tips
            - Use filters to focus on your product line
            - Click map markers for detailed popups
            - Use the "View Form 470" button for direct access to RFPs
            - Download the Excel file to import into your CRM
            """)
        
        return
    
    # Process data when analyze button is clicked
    if analyze_button:
        with st.spinner("🔄 Processing data..."):
            try:
                # Load data
                df = load_data(uploaded_file)
                
                # Filter to Texas
                texas_df = df[df['State'] == 'TX'].copy()
                
                if len(texas_df) == 0:
                    st.error("No Texas applicants found in the uploaded file!")
                    return
                
                # Apply filters
                if selected_functions or selected_manufacturers:
                    filtered_texas = apply_filters(texas_df, selected_functions, selected_manufacturers)
                else:
                    filtered_texas = texas_df
                
                if len(filtered_texas) == 0:
                    st.warning("No applicants match your filter criteria. Try adjusting your filters.")
                    return
                
                # Aggregate data
                aggregated = aggregate_by_applicant(filtered_texas)
                
                # Geocode
                st.info("📍 Geocoding locations... This may take a few minutes on first run.")
                cache = load_cache()
                geolocator = Nominatim(user_agent="texas_erate_webapp_v1")
                
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
                
                # Remove non-geocoded
                aggregated = aggregated.dropna(subset=['Latitude', 'Longitude'])
                
                # Filter by geography
                filtered = aggregated[
                    (aggregated['Latitude'] > LAT_WACO) &
                    (aggregated['Longitude'] < LON_I35E)
                ].copy()
                
                if len(filtered) == 0:
                    st.warning("No applicants found in your territory after geographic filtering.")
                    return
                
                # Store in session state
                st.session_state['results'] = filtered
                st.success(f"✅ Analysis complete! Found {len(filtered)} opportunities in your territory.")
                
            except Exception as e:
                st.error(f"Error processing data: {str(e)}")
                return
    
    # Display results if available
    if 'results' in st.session_state:
        filtered = st.session_state['results']
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Applicants", len(filtered))
        with col2:
            ic_count = (filtered['470s_IC'] > 0).sum()
            st.metric("Internal Connections", ic_count)
        with col3:
            bm_count = (filtered['470s_BM'] > 0).sum()
            st.metric("Basic Maintenance", bm_count)
        with col4:
            mb_count = (filtered['470s_MB'] > 0).sum()
            st.metric("Managed Broadband", mb_count)
        
        st.markdown("---")
        
        # Tabs for different views
        tab1, tab2 = st.tabs(["🗺️ Interactive Map", "📊 Data Table"])
        
        with tab1:
            st.subheader("Interactive Map - Click markers for details")
            map_obj = create_map(filtered)
            if map_obj:
                folium_static(map_obj, width=1200, height=600)
                
                # Legend
                st.markdown("### 🎨 Color Legend")
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
            st.subheader("Opportunity Details")
            
            # Add total line items column
            filtered['Total_470s'] = filtered['470s_IC'] + filtered['470s_BM'] + filtered['470s_MB'] + filtered['470s_DIA']
            
            # Display table
            display_df = filtered[[
                'Name of Applicant', 'City', 
                '470s_IC', '470s_BM', '470s_MB', '470s_DIA', 'Total_470s',
                'Functions', 'Manufacturers', 'Form_470_URL'
            ]].copy()
            
            # Sort by total
            display_df = display_df.sort_values('Total_470s', ascending=False)
            
            st.dataframe(
                display_df,
                use_container_width=True,
                column_config={
                    "Form_470_URL": st.column_config.LinkColumn("Form 470 Link"),
                    "Total_470s": "Total 470s"
                }
            )
            
            # Download button
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                display_df.to_excel(writer, index=False, sheet_name='Opportunities')
            output.seek(0)
            
            st.download_button(
                label="📥 Download Excel Report",
                data=output,
                file_name="erate_opportunities.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary"
            )

if __name__ == "__main__":
    main()
