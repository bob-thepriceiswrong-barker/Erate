"""
USAC E-Rate Data Viewer - SODA3 API Integration
Access C2 Budget and Form 471 data via USAC Open Data SODA3 API
"""

import streamlit as st
import requests
import pandas as pd
from datetime import datetime
from io import BytesIO

# Configuration
APP_TOKEN = "MFa7tcJ2Pybss1tK93iriT9qW"

# API Endpoints - SODA3 Format
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

# Page config
st.set_page_config(
    page_title="USAC E-Rate Data Viewer",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🎓 USAC E-Rate Data Viewer")
st.markdown("Access **C2 Budget** and **Form 471** data via SODA3 API")

# Sidebar - Dataset Selection
st.sidebar.header("📊 Select Dataset")
selected_dataset = st.sidebar.selectbox(
    "Choose Dataset",
    list(ENDPOINTS.keys()),
    key="dataset_selector"
)

st.sidebar.markdown(f"**Description:** {ENDPOINTS[selected_dataset]['description']}")

# Sidebar - Query Options
st.sidebar.header("⚙️ Query Options")

# Limit
limit = st.sidebar.number_input(
    "Number of records to fetch",
    min_value=10,
    max_value=10000,
    value=100,
    step=10,
    help="Maximum number of records to retrieve"
)

# Offset (for pagination)
offset = st.sidebar.number_input(
    "Starting record (offset)",
    min_value=0,
    max_value=100000,
    value=0,
    step=100,
    help="Use for pagination - skip this many records"
)

# Sidebar - Filters
st.sidebar.header("🔍 Filters")

filters = {}

if selected_dataset == "Form 471 Basic Information":
    # Form 471 specific filters
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
    # C2 Budget specific filters
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


# Function to fetch data from SODA3 API
@st.cache_data(ttl=3600)
def fetch_soda3_data(endpoint_url, app_token, limit=100, offset=0, filters=None):
    """
    Fetch data from USAC Open Data using SODA3 API

    Args:
        endpoint_url: The SODA3 API endpoint
        app_token: Your app token
        limit: Number of records to return
        offset: Starting position for pagination
        filters: Dictionary of filter conditions

    Returns:
        tuple: (data, error)
    """

    # Build the query body for SODA3
    query_body = {
        "limit": limit,
        "offset": offset,
        "order": [{"column": "rowId", "direction": "DESC"}]
    }

    # Add filters if provided
    if filters:
        where_conditions = []

        for field, value in filters.items():
            if value:  # Only add if value is not empty
                # For text fields, use contains for partial matching
                if isinstance(value, str) and not value.isdigit():
                    where_conditions.append({
                        "type": "operator",
                        "operator": "contains",
                        "column": field,
                        "value": value
                    })
                else:
                    # For exact matches (numbers, IDs)
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

    # Headers for SODA3
    headers = {
        "Content-Type": "application/json",
        "X-App-Token": app_token
    }

    try:
        # Make POST request to SODA3 API
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


# Function to parse SODA3 response into DataFrame
def parse_soda3_response(response_data):
    """
    Convert SODA3 JSON response to pandas DataFrame

    SODA3 returns:
    {
      "columns": [{"name": "col1"}, {"name": "col2"}, ...],
      "rows": [[val1, val2, ...], [val1, val2, ...], ...]
    }
    """
    if not response_data:
        return None

    # Extract column names from metadata
    columns = [col["name"] for col in response_data.get("columns", [])]

    # Extract rows
    rows = response_data.get("rows", [])

    if not rows:
        return pd.DataFrame()

    # Create DataFrame
    df = pd.DataFrame(rows, columns=columns)

    return df


# Main content area
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader(f"📁 {selected_dataset}")

with col2:
    fetch_button = st.button("🔄 Fetch Data", type="primary", use_container_width=True)

# Fetch data when button is clicked
if fetch_button:
    with st.spinner("Fetching data from USAC Open Data API..."):

        # Get endpoint URL
        endpoint_url = ENDPOINTS[selected_dataset]["url"]

        # Fetch data
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
            # Parse the response
            df = parse_soda3_response(data)

            if df is None or len(df) == 0:
                st.warning("⚠️ No records found with current filters. Try adjusting your search criteria.")
            else:
                # Store in session state
                st.session_state['data'] = df
                st.session_state['dataset_name'] = selected_dataset
                st.session_state['fetch_time'] = datetime.now()
                st.success(f"✅ Successfully fetched {len(df)} records!")

# Display data if available
if 'data' in st.session_state:
    df = st.session_state['data']
    dataset_name = st.session_state.get('dataset_name', 'Dataset')
    fetch_time = st.session_state.get('fetch_time', datetime.now())

    # Summary metrics
    st.markdown("---")
    st.subheader("📈 Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Records", f"{len(df):,}")

    with col2:
        st.metric("Columns", len(df.columns))

    with col3:
        # Memory usage
        memory_mb = df.memory_usage(deep=True).sum() / 1024**2
        st.metric("Memory Usage", f"{memory_mb:.2f} MB")

    with col4:
        # Check for null values
        if len(df) > 0 and len(df.columns) > 0:
            null_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            st.metric("Completeness", f"{100-null_pct:.1f}%")
        else:
            st.metric("Completeness", "N/A")

    st.caption(f"Last fetched: {fetch_time.strftime('%Y-%m-%d %I:%M:%S %p')}")

    # Data preview
    st.markdown("---")
    st.subheader("📋 Data Preview")

    # Advanced options
    with st.expander("🔧 Customize View & Search"):
        col_a, col_b = st.columns(2)

        with col_a:
            # Column selector
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
            # Search within data
            search_term = st.text_input("🔍 Search in data", "")
            if search_term:
                mask = df_display.astype(str).apply(
                    lambda x: x.str.contains(search_term, case=False, na=False)
                ).any(axis=1)
                df_display = df_display[mask]
                st.info(f"Found {len(df_display)} records matching '{search_term}'")

    # Display dataframe
    st.dataframe(
        df_display,
        use_container_width=True,
        height=500
    )

    # Download section
    st.markdown("---")
    st.subheader("💾 Export Data")

    col1, col2, col3 = st.columns(3)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_prefix = dataset_name.replace(' ', '_').replace('(', '').replace(')', '')

    with col1:
        # CSV download
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name=f"{file_prefix}_{timestamp}.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col2:
        # JSON download
        json_str = df.to_json(orient='records', indent=2)
        st.download_button(
            label="📥 Download as JSON",
            data=json_str,
            file_name=f"{file_prefix}_{timestamp}.json",
            mime="application/json",
            use_container_width=True
        )

    with col3:
        # Excel download
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

    # Additional analysis tools
    col_exp1, col_exp2 = st.columns(2)

    with col_exp1:
        # Detailed record view
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
        # Basic statistics
        with st.expander("📊 Column Statistics"):
            if len(df) > 0:
                st.dataframe(df.describe(include='all'), use_container_width=True)
            else:
                st.info("No data for statistics")

else:
    # Initial state - no data loaded
    st.info("👈 Configure your query using the sidebar and click **'Fetch Data'** to begin.")

    # Show example of what you can do
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

        ### 📊 Available Datasets

        **C2 Budget Tool (FY2021+)**
        - Five-year budget cycle information starting FY2021
        - Track C2 budget allocation and spending
        - Identify entities with remaining budget

        **Form 471 Basic Information**
        - Applicant details from FCC Form 471
        - Historical funding requests
        - Service provider information

        ### 💡 Tips
        - Start with a small limit (100) to test your filters
        - Use offset for pagination when you need more than 10,000 records
        - Data is cached for 1 hour - subsequent queries with same parameters are instant
        - Use partial name matching in filters (e.g., "Dallas" finds "Dallas ISD")
        - Export data for offline analysis or CRM import

        ### 🔧 Technical Details
        - **API**: USAC Open Data SODA3 API
        - **Method**: POST requests with JSON query body
        - **Authentication**: App token in X-App-Token header
        - **Rate Limits**: Governed by your app token tier
        """)

    # API Examples
    with st.expander("🔬 API Examples"):
        st.markdown("""
        ### Example Queries

        **Find all Texas schools in funding year 2024:**
        - Dataset: Form 471 Basic Information
        - State: TX
        - Funding Year: 2024
        - Limit: 500

        **Check C2 budget for specific entity:**
        - Dataset: C2 Budget Tool
        - BEN Number: [enter BEN]
        - Limit: 10

        **Pagination example (get next 100 records):**
        - First query: Offset = 0, Limit = 100
        - Second query: Offset = 100, Limit = 100
        - Third query: Offset = 200, Limit = 100

        ### SODA3 Request Structure

        This app uses SODA3 API format:
        ```json
        {
          "limit": 100,
          "offset": 0,
          "order": [{"column": "rowId", "direction": "DESC"}],
          "where": {
            "type": "operator",
            "operator": "=",
            "column": "state",
            "value": "TX"
          }
        }
        ```

        Headers:
        ```
        Content-Type: application/json
        X-App-Token: [your token]
        ```
        """)

# Footer with API info
st.markdown("---")
st.markdown("### 🔧 Technical Information")

col_footer1, col_footer2 = st.columns(2)

with col_footer1:
    with st.expander("API Configuration"):
        st.code(f"""
Current Configuration:
- App Token: {APP_TOKEN[:10]}... (secured)
- Selected Dataset: {selected_dataset}
- API Version: SODA3
- Method: POST with JSON body
- Endpoint: {ENDPOINTS[selected_dataset]['url']}
- Limit: {limit}
- Offset: {offset}
- Active Filters: {len(filters)}
        """)

with col_footer2:
    with st.expander("Dataset Endpoints"):
        for name, config in ENDPOINTS.items():
            st.markdown(f"**{name}**")
            st.code(config['url'])
            st.caption(config['description'])
            st.markdown("---")

# Sidebar footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
**USAC E-Rate Data Viewer**
Version 1.0 | SODA3 API

[USAC Open Data](https://opendata.usac.org/)
""")
