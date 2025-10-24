"""
Texas E-Rate Form 470 Analyzer - Enhanced with Form 471 Purchase History
Competitive Intelligence: See what districts have bought in the past!
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
import json
from fuzzywuzzy import fuzz

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="E-Rate Opportunity Finder - Intelligence Edition",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CONSTANTS
# ============================================================================

LAT_WACO = 31.55
LON_I35E = -97.0
GEOCODE_CACHE_FILE = 'geocode_cache.csv'
FORM_471_CACHE_FILE = 'form_471_cache.json'
FORM_471_API_URL = 'https://opendata.usac.org/resource/avi8-svp9.json'
CACHE_EXPIRY_DAYS = 7  # Refresh Form 471 cache weekly

# ============================================================================
# FORM 471 API FUNCTIONS (PHASE 1)
# ============================================================================

@st.cache_data(ttl=604800)  # Cache for 1 week
def fetch_form_471_data(limit=50000):
    """
    Fetch Form 471 purchase history from USAC API
    Returns last 3 years of data for competitive intelligence
    """
    try:
        # Calculate date 3 years ago
        three_years_ago = (datetime.now() - timedelta(days=1095)).year
        
        # Fetch all data (API doesn't support filtering) - Updated for actual USAC API structure
        response = requests.get(FORM_471_API_URL, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        df = pd.DataFrame(data)
        
        if len(df) == 0:
            return pd.DataFrame()
        
        # Filter data locally since API doesn't support filtering
        # Filter for Texas and last 3 years
        if 'ros_physical_state' in df.columns:
            df = df[df['ros_physical_state'] == 'TX']
            print(f"Filtered to Texas: {len(df)} records")
        
        if 'funding_year' in df.columns:
            df['funding_year'] = pd.to_numeric(df['funding_year'], errors='coerce')
            df = df[df['funding_year'] >= three_years_ago]
            print(f"Filtered to recent years (>= {three_years_ago}): {len(df)} records")
        
        if len(df) == 0:
            return pd.DataFrame()
        
        # Standardize column names - Updated for actual USAC API structure
        column_mapping = {
            'ros_entity_name': 'Applicant_Name',
            'funding_year': 'Funding_Year',
            'spin_name': 'Vendor',
            'post_discount_extended_eligible_line_item_costs': 'Amount_Approved',
            'chosen_category_of_service': 'Service_Category',
            'form_471_function_name': 'Function',
            'ros_physical_city': 'City',
            'ros_physical_state': 'State',
            # Line-item details (Phase 5)
            'form_471_product_name': 'Product_Name',
            'monthly_quantity': 'Quantity',
            'monthly_recurring_unit_eligible_costs': 'Unit_Cost',
            'total_monthly_cost': 'Monthly_Cost',
            'form_471_service_type_name': 'Service_Type'
        }
        
        # Rename columns that exist
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
        
        # Convert amount to numeric
        if 'Amount_Approved' in df.columns:
            df['Amount_Approved'] = pd.to_numeric(df['Amount_Approved'], errors='coerce')
        
        # Convert line-item numeric fields (Phase 5)
        if 'Quantity' in df.columns:
            df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
        if 'Unit_Cost' in df.columns:
            df['Unit_Cost'] = pd.to_numeric(df['Unit_Cost'], errors='coerce')
        if 'Monthly_Cost' in df.columns:
            df['Monthly_Cost'] = pd.to_numeric(df['Monthly_Cost'], errors='coerce')
        
        # Convert funding year to int
        if 'Funding_Year' in df.columns:
            df['Funding_Year'] = pd.to_numeric(df['Funding_Year'], errors='coerce').astype('Int64')
        
        return df
        
    except Exception as e:
        st.warning(f"Could not fetch Form 471 data: {str(e)}")
        return pd.DataFrame()

def fuzzy_match_applicant(form_470_name, form_471_df, threshold=85):
    """
    Fuzzy match applicant names between Form 470 and Form 471
    Returns matching records from Form 471
    """
    if form_471_df.empty or 'Applicant_Name' not in form_471_df.columns:
        return pd.DataFrame()
    
    # Try exact match first
    exact_matches = form_471_df[form_471_df['Applicant_Name'].str.lower() == form_470_name.lower()]
    if len(exact_matches) > 0:
        return exact_matches
    
    # Try fuzzy matching
    form_471_df['match_score'] = form_471_df['Applicant_Name'].apply(
        lambda x: fuzz.ratio(str(x).lower(), form_470_name.lower())
    )
    
    matches = form_471_df[form_471_df['match_score'] >= threshold]
    return matches.drop('match_score', axis=1)

# ============================================================================
# LINE-ITEM DRILL-DOWN (PHASE 5)
# ============================================================================

def get_line_item_breakdown(applicant_name, form_471_df):
    """
    Get detailed line-item breakdown by product/model
    Returns individual purchases with model numbers, quantities, and unit prices
    """
    matches = fuzzy_match_applicant(applicant_name, form_471_df)
    
    if matches.empty:
        return []
    
    line_items = []
    
    for _, row in matches.iterrows():
        # Skip if no product details
        if all(pd.isna(row.get(field)) for field in ['Product_Name', 'Model_Number', 'Description']):
            continue
        
        # Build line item
        item = {
            'year': row.get('Funding_Year', 'N/A'),
            'vendor': row.get('Vendor', 'Unknown'),
            'manufacturer': row.get('Manufacturer', ''),
            'product_name': row.get('Product_Name', ''),
            'model_number': row.get('Model_Number', ''),
            'description': row.get('Description', ''),
            'quantity': row.get('Quantity', None),
            'unit_cost': row.get('Unit_Cost', None),
            'monthly_cost': row.get('Monthly_Cost', None),
            'total_cost': row.get('Amount_Approved', 0),
            'function': row.get('Function', ''),
            'category': row.get('Service_Category', '')
        }
        
        # Clean up display
        if pd.notna(item['quantity']):
            item['quantity'] = int(item['quantity'])
        if pd.notna(item['unit_cost']):
            item['unit_cost'] = float(item['unit_cost'])
        if pd.notna(item['monthly_cost']):
            item['monthly_cost'] = float(item['monthly_cost'])
        if pd.notna(item['total_cost']):
            item['total_cost'] = float(item['total_cost'])
        
        line_items.append(item)
    
    # Sort by year (newest first), then by total cost
    line_items.sort(key=lambda x: (x['year'] if x['year'] != 'N/A' else 0, x['total_cost']), reverse=True)
    
    return line_items

def format_line_item_html(line_items, show_limit=10):
    """
    Format line items as HTML table for popup display
    """
    if not line_items:
        return ""
    
    html = """
    <table style="width: 100%; font-size: 11px; border-collapse: collapse; margin-top: 10px;">
        <thead style="background-color: #f0f0f0;">
            <tr>
                <th style="padding: 5px; text-align: left; border: 1px solid #ddd;">Year</th>
                <th style="padding: 5px; text-align: left; border: 1px solid #ddd;">Product/Model</th>
                <th style="padding: 5px; text-align: right; border: 1px solid #ddd;">Qty</th>
                <th style="padding: 5px; text-align: right; border: 1px solid #ddd;">Unit $</th>
                <th style="padding: 5px; text-align: right; border: 1px solid #ddd;">Total</th>
            </tr>
        </thead>
        <tbody>
    """
    
    for i, item in enumerate(line_items[:show_limit]):
        # Build product display
        product_display = []
        if item['manufacturer']:
            product_display.append(str(item['manufacturer']))
        if item['model_number']:
            product_display.append(f"<b>{item['model_number']}</b>")
        elif item['product_name']:
            product_display.append(str(item['product_name'])[:40])
        
        product_str = ' '.join(product_display) if product_display else 'Product details N/A'
        
        # Format pricing
        qty_str = str(item['quantity']) if item['quantity'] else '-'
        unit_str = f"${item['unit_cost']:,.0f}" if item['unit_cost'] else '-'
        total_str = f"${item['total_cost']:,.0f}" if item['total_cost'] else '-'
        
        # Vendor in product name
        if item['vendor'] and item['vendor'] != 'Unknown':
            product_str = f"{product_str}<br><small style='color: #666;'>{item['vendor']}</small>"
        
        html += f"""
        <tr style="{'background-color: #f9f9f9;' if i % 2 == 0 else ''}">
            <td style="padding: 5px; border: 1px solid #ddd;">{item['year']}</td>
            <td style="padding: 5px; border: 1px solid #ddd;">{product_str}</td>
            <td style="padding: 5px; text-align: right; border: 1px solid #ddd;">{qty_str}</td>
            <td style="padding: 5px; text-align: right; border: 1px solid #ddd;">{unit_str}</td>
            <td style="padding: 5px; text-align: right; border: 1px solid #ddd; font-weight: bold;">{total_str}</td>
        </tr>
        """
    
    html += """
        </tbody>
    </table>
    """
    
    if len(line_items) > show_limit:
        html += f"""
        <p style="margin: 5px 0; font-size: 11px; color: #666; text-align: center;">
            Showing {show_limit} of {len(line_items)} line items
        </p>
        """
    
    return html

# ============================================================================
# COMPETITIVE INTELLIGENCE ANALYSIS (PHASE 4)
# ============================================================================

def analyze_purchase_history(applicant_name, form_471_df):
    """
    Analyze purchase history for competitive intelligence
    Returns insights about incumbent vendors, spending patterns, etc.
    """
    matches = fuzzy_match_applicant(applicant_name, form_471_df)
    
    if matches.empty:
        return {
            'has_history': False,
            'total_spent': 0,
            'purchase_count': 0,
            'vendors': [],
            'top_vendor': None,
            'recent_purchases': [],
            'switching_opportunity': False
        }
    
    # Calculate metrics
    total_spent = matches['Amount_Approved'].sum() if 'Amount_Approved' in matches.columns else 0
    purchase_count = len(matches)
    
    # Vendor analysis
    vendor_spending = {}
    if 'Vendor' in matches.columns and 'Amount_Approved' in matches.columns:
        vendor_spending = matches.groupby('Vendor')['Amount_Approved'].sum().sort_values(ascending=False).to_dict()
    
    top_vendor = list(vendor_spending.keys())[0] if vendor_spending else None
    
    # Recent purchases (last 2 years)
    recent = matches[matches['Funding_Year'] >= (datetime.now().year - 2)] if 'Funding_Year' in matches.columns else matches
    
    recent_purchases = []
    for _, row in recent.head(5).iterrows():
        purchase = {
            'year': row.get('Funding_Year', 'N/A'),
            'vendor': row.get('Vendor', 'Unknown'),
            'amount': row.get('Amount_Approved', 0),
            'category': row.get('Service_Category', 'N/A'),
            'function': row.get('Function', 'N/A')
        }
        recent_purchases.append(purchase)
    
    # Switching opportunity detection
    # If they have multiple vendors or haven't bought in 2+ years
    switching_opportunity = False
    if 'Funding_Year' in matches.columns:
        last_purchase_year = matches['Funding_Year'].max()
        if pd.notna(last_purchase_year):
            years_since_purchase = datetime.now().year - int(last_purchase_year)
            switching_opportunity = (years_since_purchase >= 2) or (len(vendor_spending) >= 3)
    
    return {
        'has_history': True,
        'total_spent': total_spent,
        'purchase_count': purchase_count,
        'vendors': list(vendor_spending.keys()),
        'vendor_spending': vendor_spending,
        'top_vendor': top_vendor,
        'recent_purchases': recent_purchases,
        'switching_opportunity': switching_opportunity,
        'all_purchases': matches
    }

# ============================================================================
# HISTORICAL FILTERS (PHASE 3)
# ============================================================================

def filter_by_purchase_history(aggregated_df, form_471_df, vendor_filter, equipment_filter, spending_filter):
    """
    Filter applicants based on their purchase history
    """
    if form_471_df.empty:
        return aggregated_df
    
    if not vendor_filter and not equipment_filter and not spending_filter:
        return aggregated_df
    
    filtered_applicants = []
    
    for _, row in aggregated_df.iterrows():
        applicant_name = row['Name of Applicant']
        history = analyze_purchase_history(applicant_name, form_471_df)
        
        # Check vendor filter
        vendor_match = True
        if vendor_filter:
            vendor_match = any(vendor.lower() in [v.lower() for v in history.get('vendors', [])] 
                             for vendor in vendor_filter)
        
        # Check equipment filter
        equipment_match = True
        if equipment_filter and history['has_history']:
            purchases = history.get('all_purchases', pd.DataFrame())
            if not purchases.empty and 'Function' in purchases.columns:
                purchase_functions = purchases['Function'].dropna().str.lower().tolist()
                equipment_match = any(
                    any(eq.lower() in func for func in purchase_functions)
                    for eq in equipment_filter
                )
        
        # Check spending filter
        spending_match = True
        if spending_filter:
            total_spent = history.get('total_spent', 0)
            if spending_filter == "$0-$25k":
                spending_match = 0 <= total_spent < 25000
            elif spending_filter == "$25k-$50k":
                spending_match = 25000 <= total_spent < 50000
            elif spending_filter == "$50k-$100k":
                spending_match = 50000 <= total_spent < 100000
            elif spending_filter == "$100k+":
                spending_match = total_spent >= 100000
        
        if vendor_match and equipment_match and spending_match:
            filtered_applicants.append(applicant_name)
    
    return aggregated_df[aggregated_df['Name of Applicant'].isin(filtered_applicants)]

# ============================================================================
# EXISTING HELPER FUNCTIONS (FROM ORIGINAL)
# ============================================================================

@st.cache_data
def load_data(uploaded_file):
    """Load Excel data from uploaded file"""
    return pd.read_excel(uploaded_file)

def load_cache():
    """Load geocoding cache"""
    if os.path.exists(GEOCODE_CACHE_FILE):
        try:
            return pd.read_csv(GEOCODE_CACHE_FILE)
        except:
            return pd.DataFrame(columns=['Name', 'City', 'State', 'Latitude', 'Longitude'])
    return pd.DataFrame(columns=['Name', 'City', 'State', 'Latitude', 'Longitude'])

def save_cache(cache):
    """Save geocoding cache"""
    cache.to_csv(GEOCODE_CACHE_FILE, index=False)

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
    if has_mb and not has_ic and not has_mb:
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
# ENHANCED MAP WITH PURCHASE HISTORY (PHASE 2)
# ============================================================================

def create_enhanced_popup(row, history):
    """
    Create enhanced popup with Form 470 + Form 471 purchase history
    """
    functions_display = row['Functions'][:150] + '...' if len(row['Functions']) > 150 else row['Functions']
    manufacturers_display = row['Manufacturers'][:150] + '...' if len(row['Manufacturers']) > 150 else row['Manufacturers']
    
    # Build popup HTML
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
    """
    
    # Add purchase history if available
    if history['has_history']:
        popup_html += f"""
        <hr style="margin: 15px 0; border: 1px solid #ddd;">
        
        <h4 style="margin: 10px 0 5px 0; color: #28a745;">💰 Purchase History (3 Years)</h4>
        <p style="margin: 5px 0;"><b>Total Spent:</b> ${history['total_spent']:,.0f}</p>
        <p style="margin: 5px 0;"><b>Total Purchases:</b> {history['purchase_count']}</p>
        """
        
        # Top vendor
        if history['top_vendor']:
            top_vendor_spending = history['vendor_spending'].get(history['top_vendor'], 0)
            popup_html += f"""
            <p style="margin: 5px 0; padding: 8px; background-color: #fff3cd; border-left: 4px solid #ffc107;">
                <b>🎯 Incumbent Vendor:</b><br>
                {history['top_vendor']}<br>
                <small>${top_vendor_spending:,.0f} spent</small>
            </p>
            """
        
        # LINE-ITEM BREAKDOWN (Phase 5) - NEW!
        line_items = get_line_item_breakdown(row['Name of Applicant'], history.get('all_purchases', pd.DataFrame()))
        if line_items:
            popup_html += """
            <details style="margin: 10px 0;">
                <summary style="cursor: pointer; font-weight: bold; padding: 8px; background-color: #e7f3ff; border-radius: 4px;">
                    📊 View Line-Item Details (Click to expand)
                </summary>
            """
            popup_html += format_line_item_html(line_items, show_limit=8)
            popup_html += "</details>"
        
        # Recent purchases summary (keep for quick view)
        if history['recent_purchases'] and not line_items:  # Only show if no line items
            popup_html += """
            <p style="margin: 10px 0 5px 0; font-size: 13px;"><b>Recent Purchases:</b></p>
            <ul style="margin: 5px 0; padding-left: 20px; font-size: 12px;">
            """
            for purchase in history['recent_purchases'][:3]:
                popup_html += f"""
                <li>{purchase['year']}: {purchase['vendor']} - ${purchase['amount']:,.0f}
                    <br><small>{purchase['function']}</small>
                </li>
                """
            popup_html += "</ul>"
        
        # Switching opportunity flag
        if history['switching_opportunity']:
            popup_html += """
            <p style="margin: 10px 0; padding: 8px; background-color: #d4edda; border-left: 4px solid #28a745;">
                <b>🚀 Switching Opportunity!</b><br>
                <small>Multiple vendors or 2+ years since last purchase</small>
            </p>
            """
    else:
        popup_html += """
        <hr style="margin: 15px 0; border: 1px solid #ddd;">
        <p style="margin: 10px 0; padding: 8px; background-color: #e7f3ff; border-left: 4px solid #0066cc;">
            <b>ℹ️ No Purchase History Found</b><br>
            <small>This could be a new opportunity!</small>
        </p>
        """
    
    # Add Form 470 button
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

def create_map(filtered_df, form_471_df):
    """Create interactive map with purchase history"""
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
        popup='Waco Boundary (North)'
    ).add_to(m)
    
    folium.PolyLine(
        locations=[[28, LON_I35E], [36, LON_I35E]],
        color='red',
        weight=2,
        opacity=0.7,
        popup='I-35E Boundary (West)'
    ).add_to(m)
    
    # Add markers with purchase history
    for idx, row in filtered_df.iterrows():
        color = color_for_services(row)
        
        # Get purchase history for this applicant
        history = analyze_purchase_history(row['Name of Applicant'], form_471_df)
        
        # Create enhanced popup
        popup_html = create_enhanced_popup(row, history)
        
        # Determine icon based on switching opportunity
        icon_html = f"""
        <div style="font-size: 20px;">
            {'🎯' if history.get('switching_opportunity', False) else '📍'}
        </div>
        """
        
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=folium.Popup(popup_html, max_width=450),
            icon=folium.DivIcon(html=icon_html) if history.get('switching_opportunity', False) else None,
            tooltip=f"{row['Name of Applicant']} - Click for details"
        ).add_to(m)
        
        # Also add circle marker for visibility
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
# MAIN APPLICATION
# ============================================================================

def main():
    # Header
    st.title("🎯 Texas E-Rate Opportunity Finder")
    st.markdown("**Intelligence Edition:** Find opportunities + See purchase history!")
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
        st.header("🔍 Form 470 Filters")
        
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
        st.header("💰 Purchase History Filters")
        st.caption("Filter by what they've bought before")
        
        # Historical vendor filter
        historical_vendors = st.multiselect(
            "Past Vendors",
            ["Cisco", "Aruba", "Fortinet", "Juniper", "Meraki", "CDW", "SHI", "Zones", "Connection"],
            help="Show only districts that bought from these vendors"
        )
        
        # Historical equipment filter
        historical_equipment = st.multiselect(
            "Past Equipment",
            ["Wireless", "Switches", "Routers", "Firewall", "UPS"],
            help="Show only districts that bought this equipment"
        )
        
        # Spending level filter
        spending_level = st.selectbox(
            "Total Spending (3 years)",
            ["All", "$0-$25k", "$25k-$50k", "$50k-$100k", "$100k+"],
            help="Filter by historical spending"
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
        with st.expander("📖 How to Use This Tool - Intelligence Edition"):
            st.markdown("""
            ### 🎉 NEW! Competitive Intelligence Features
            
            **What's New:**
            - 💰 **Purchase History:** See what districts bought in the past 3 years
            - 🎯 **Incumbent Vendors:** Identify who they're currently buying from
            - 🚀 **Switching Opportunities:** Flag districts ready to switch vendors
            - 📊 **Spending Patterns:** Know their budget and buying habits
            - 🔍 **Historical Filters:** Find districts that bought from specific vendors
            
            ### Getting Started
            1. **Upload** your E-Rate data file (from Funds for Learning)
            2. **Select filters** (optional):
               - Form 470 filters: Current RFPs by equipment/manufacturer
               - Purchase History filters: Past vendors, equipment, spending levels
            3. **Click "Analyze"** to process the data (includes purchase history lookup)
            4. **View results** on the interactive map
            5. **Click markers** to see:
               - Current Form 470 details
               - 3-year purchase history
               - Incumbent vendors
               - Switching opportunity flags
            6. **Download** the enhanced results as Excel
            
            ### Competitive Advantages
            
            **Know Your Competition:**
            - See who the incumbent vendor is
            - Understand pricing history
            - Identify vendor switching patterns
            
            **Win More Deals:**
            - Target districts spending in your range
            - Find districts that bought your products before
            - Identify switching opportunities (2+ years since purchase)
            
            **Strategic Prospecting:**
            - Focus on high-spending districts
            - Avoid heavily entrenched competitors
            - Find districts with no purchase history (greenfield!)
            
            ### Tips
            - 🎯 Look for the **target emoji** on the map - those are switching opportunities!
            - 💡 Districts with no purchase history might be easier to win
            - 📊 Use spending filters to match your typical deal sizes
            - 🔍 Filter by past vendors to find competitor customers
            """)
        
        return
    
    # Process data when analyze button is clicked
    if analyze_button:
        with st.spinner("🔄 Processing data and fetching purchase history..."):
            try:
                # Step 1: Fetch Form 471 data
                st.info("📡 Step 1/4: Fetching Form 471 purchase history from USAC...")
                form_471_df = fetch_form_471_data()
                
                if form_471_df.empty:
                    st.warning("Could not fetch Form 471 data. Proceeding with Form 470 data only.")
                else:
                    st.success(f"✅ Loaded {len(form_471_df):,} purchase records from last 3 years")
                
                # Step 2: Load Form 470 data
                st.info("📁 Step 2/4: Processing Form 470 data...")
                df = load_data(uploaded_file)
                
                # Filter to Texas
                texas_df = df[df['State'] == 'TX'].copy()
                
                if len(texas_df) == 0:
                    st.error("No Texas applicants found in the uploaded file!")
                    return
                
                # Apply Form 470 filters
                if selected_functions or selected_manufacturers:
                    filtered_texas = apply_filters(texas_df, selected_functions, selected_manufacturers)
                else:
                    filtered_texas = texas_df
                
                if len(filtered_texas) == 0:
                    st.warning("No applicants match your Form 470 filter criteria. Try adjusting your filters.")
                    return
                
                # Aggregate data
                aggregated = aggregate_by_applicant(filtered_texas)
                
                # Step 3: Apply purchase history filters
                st.info("💰 Step 3/4: Applying purchase history filters...")
                spending_filter_value = None if spending_level == "All" else spending_level
                
                if historical_vendors or historical_equipment or spending_filter_value:
                    aggregated = filter_by_purchase_history(
                        aggregated, 
                        form_471_df,
                        historical_vendors,
                        historical_equipment,
                        spending_filter_value
                    )
                
                if len(aggregated) == 0:
                    st.warning("No applicants match your purchase history filters. Try adjusting your filters.")
                    return
                
                # Step 4: Geocode
                st.info("📍 Step 4/4: Geocoding locations...")
                cache = load_cache()
                geolocator = Nominatim(user_agent="texas_erate_webapp_v2")
                
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
                st.session_state['form_471_df'] = form_471_df
                st.success(f"✅ Analysis complete! Found {len(filtered)} opportunities with purchase history!")
                
            except Exception as e:
                st.error(f"Error processing data: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
                return
    
    # Display results if available
    if 'results' in st.session_state:
        filtered = st.session_state['results']
        form_471_df = st.session_state.get('form_471_df', pd.DataFrame())
        
        # Calculate competitive intelligence metrics
        switching_count = 0
        incumbent_count = 0
        no_history_count = 0
        
        for _, row in filtered.iterrows():
            history = analyze_purchase_history(row['Name of Applicant'], form_471_df)
            if history['switching_opportunity']:
                switching_count += 1
            if history['has_history']:
                incumbent_count += 1
            else:
                no_history_count += 1
        
        # Summary metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Opportunities", len(filtered))
        with col2:
            st.metric("🎯 Switching Opportunities", switching_count)
        with col3:
            st.metric("📊 With Purchase History", incumbent_count)
        with col4:
            st.metric("🆕 No History (New!)", no_history_count)
        with col5:
            ic_count = (filtered['470s_IC'] > 0).sum()
            st.metric("Internal Connections", ic_count)
        
        st.markdown("---")
        
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Interactive Map", "📊 Data Table", "💡 Intelligence Report", "🔬 Product Analysis"])
        
        with tab1:
            st.subheader("Interactive Map - Click markers for competitive intelligence!")
            st.caption("🎯 = Switching Opportunity | 📍 = Standard Opportunity")
            
            map_obj = create_map(filtered, form_471_df)
            if map_obj:
                folium_static(map_obj, width=1200, height=600)
                
                # Legend
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
            st.subheader("Opportunity Details with Purchase History")
            
            # Build enhanced display table
            display_rows = []
            for _, row in filtered.iterrows():
                history = analyze_purchase_history(row['Name of Applicant'], form_471_df)
                
                display_rows.append({
                    'Name of Applicant': row['Name of Applicant'],
                    'City': row['City'],
                    '470s_IC': int(row['470s_IC']),
                    '470s_BM': int(row['470s_BM']),
                    '470s_MB': int(row['470s_MB']),
                    '470s_DIA': int(row['470s_DIA']),
                    'Total_470s': int(row['470s_IC'] + row['470s_BM'] + row['470s_MB'] + row['470s_DIA']),
                    'Purchase_History': '✅ Yes' if history['has_history'] else '❌ No',
                    'Total_Spent_3yr': f"${history.get('total_spent', 0):,.0f}" if history['has_history'] else 'N/A',
                    'Incumbent_Vendor': history.get('top_vendor', 'N/A')[:30] if history.get('top_vendor') else 'N/A',
                    'Switching_Opp': '🎯 YES' if history.get('switching_opportunity', False) else '',
                    'Functions': row['Functions'],
                    'Manufacturers': row['Manufacturers'],
                    'Form_470_URL': row['Form_470_URL']
                })
            
            display_df = pd.DataFrame(display_rows)
            display_df = display_df.sort_values('Total_470s', ascending=False)
            
            st.dataframe(
                display_df,
                use_container_width=True,
                column_config={
                    "Form_470_URL": st.column_config.LinkColumn("Form 470 Link"),
                    "Total_Spent_3yr": "3-Year Spending",
                    "Switching_Opp": "Switch Opp"
                }
            )
            
            # Download button
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                display_df.to_excel(writer, index=False, sheet_name='Opportunities')
            output.seek(0)
            
            st.download_button(
                label="📥 Download Enhanced Excel Report",
                data=output,
                file_name="erate_opportunities_with_intelligence.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary"
            )
        
        with tab3:
            st.subheader("💡 Competitive Intelligence Summary")
            
            # Top switching opportunities
            st.markdown("### 🎯 Top Switching Opportunities")
            switching_opps = []
            for _, row in filtered.iterrows():
                history = analyze_purchase_history(row['Name of Applicant'], form_471_df)
                if history['switching_opportunity']:
                    switching_opps.append({
                        'District': row['Name of Applicant'],
                        'City': row['City'],
                        'Total Spent': history['total_spent'],
                        'Incumbent': history.get('top_vendor', 'Multiple/Unknown'),
                        'Last Purchase': max([p['year'] for p in history['recent_purchases']]) if history['recent_purchases'] else 'N/A'
                    })
            
            if switching_opps:
                switch_df = pd.DataFrame(switching_opps).sort_values('Total Spent', ascending=False)
                st.dataframe(switch_df, use_container_width=True)
            else:
                st.info("No switching opportunities identified in current results.")
            
            st.markdown("---")
            
            # Vendor landscape
            st.markdown("### 🏢 Competitive Landscape")
            vendor_totals = {}
            for _, row in filtered.iterrows():
                history = analyze_purchase_history(row['Name of Applicant'], form_471_df)
                if history['has_history']:
                    for vendor, amount in history.get('vendor_spending', {}).items():
                        if vendor:
                            vendor_totals[vendor] = vendor_totals.get(vendor, 0) + amount
            
            if vendor_totals:
                vendor_df = pd.DataFrame([
                    {'Vendor': k, 'Total Spending': v, 'District Count': 1} 
                    for k, v in sorted(vendor_totals.items(), key=lambda x: x[1], reverse=True)[:10]
                ])
                st.dataframe(vendor_df, use_container_width=True)
            else:
                st.info("No vendor data available.")
        
        with tab4:
            st.subheader("🔬 Product-Level Intelligence")
            st.caption("Drill down to specific models, part numbers, and unit prices")
            
            # Select a district for detailed analysis
            district_names = filtered['Name of Applicant'].tolist()
            selected_district = st.selectbox(
                "Select a district to view detailed purchase breakdown:",
                options=['-- Select a district --'] + district_names
            )
            
            if selected_district and selected_district != '-- Select a district --':
                st.markdown(f"### 📋 Detailed Purchase History: {selected_district}")
                
                # Get line items
                line_items = get_line_item_breakdown(selected_district, form_471_df)
                
                if line_items:
                    st.success(f"Found {len(line_items)} line items purchased over 3 years")
                    
                    # Build detailed table
                    line_item_rows = []
                    for item in line_items:
                        # Build product description
                        product_parts = []
                        if item['manufacturer']:
                            product_parts.append(str(item['manufacturer']))
                        if item['model_number']:
                            product_parts.append(str(item['model_number']))
                        elif item['product_name']:
                            product_parts.append(str(item['product_name']))
                        
                        product_desc = ' '.join(product_parts) if product_parts else 'N/A'
                        
                        line_item_rows.append({
                            'Year': item['year'],
                            'Vendor': item['vendor'],
                            'Product/Model': product_desc,
                            'Category': item['category'] if item['category'] else 'N/A',
                            'Function': item['function'] if item['function'] else 'N/A',
                            'Quantity': item['quantity'] if item['quantity'] else '-',
                            'Unit_Cost': item['unit_cost'] if item['unit_cost'] else None,
                            'Total_Cost': item['total_cost']
                        })
                    
                    line_df = pd.DataFrame(line_item_rows)
                    
                    # Format currency columns
                    line_df_display = line_df.copy()
                    line_df_display['Unit Cost'] = line_df['Unit_Cost'].apply(
                        lambda x: f"${x:,.2f}" if pd.notna(x) else '-'
                    )
                    line_df_display['Total Cost'] = line_df['Total_Cost'].apply(
                        lambda x: f"${x:,.2f}" if pd.notna(x) else '-'
                    )
                    line_df_display = line_df_display.drop(['Unit_Cost', 'Total_Cost'], axis=1)
                    
                    st.dataframe(line_df_display, use_container_width=True, height=400)
                    
                    # Summary metrics
                    st.markdown("---")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        total_spend = sum([item['total_cost'] for item in line_items if item['total_cost']])
                        st.metric("Total Spending", f"${total_spend:,.0f}")
                    
                    with col2:
                        vendors = set([item['vendor'] for item in line_items if item['vendor'] and item['vendor'] != 'Unknown'])
                        st.metric("Unique Vendors", len(vendors))
                    
                    with col3:
                        manufacturers = set([item['manufacturer'] for item in line_items if item['manufacturer']])
                        st.metric("Manufacturers", len(manufacturers))
                    
                    with col4:
                        years = set([item['year'] for item in line_items if item['year'] != 'N/A'])
                        st.metric("Years Active", len(years))
                    
                    # Download detailed line items
                    st.markdown("---")
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        line_df.to_excel(writer, index=False, sheet_name='Line Items')
                    output.seek(0)
                    
                    st.download_button(
                        label="📥 Download Detailed Line Items",
                        data=output,
                        file_name=f"{selected_district.replace(' ', '_')}_line_items.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                    
                    # Product insights
                    st.markdown("---")
                    st.markdown("### 💡 Pricing Intelligence")
                    
                    # Group by manufacturer
                    mfr_spending = {}
                    for item in line_items:
                        mfr = item.get('manufacturer', 'Unknown')
                        if mfr and mfr != 'Unknown':
                            mfr_spending[mfr] = mfr_spending.get(mfr, 0) + item['total_cost']
                    
                    if mfr_spending:
                        mfr_df = pd.DataFrame([
                            {'Manufacturer': k, 'Total Spent': f"${v:,.0f}"} 
                            for k, v in sorted(mfr_spending.items(), key=lambda x: x[1], reverse=True)
                        ])
                        st.dataframe(mfr_df, use_container_width=True)
                    
                else:
                    st.warning("No detailed line-item data available for this district.")
            else:
                st.info("👆 Select a district above to view detailed product-level purchase history")
                
                # Show aggregate insights across all districts
                st.markdown("---")
                st.markdown("### 📊 Product Intelligence Across All Opportunities")
                
                # Collect all line items across all districts
                all_line_items = []
                for _, row in filtered.head(20).iterrows():  # Limit to first 20 for performance
                    items = get_line_item_breakdown(row['Name of Applicant'], form_471_df)
                    all_line_items.extend(items)
                
                if all_line_items:
                    st.success(f"Analyzing {len(all_line_items)} line items from {min(20, len(filtered))} districts")
                    
                    # Top manufacturers by spend
                    st.markdown("#### Top Manufacturers by Total Spending")
                    mfr_totals = {}
                    for item in all_line_items:
                        mfr = item.get('manufacturer', 'Unknown')
                        if mfr and mfr != 'Unknown':
                            mfr_totals[mfr] = mfr_totals.get(mfr, 0) + item['total_cost']
                    
                    if mfr_totals:
                        top_mfrs = pd.DataFrame([
                            {'Manufacturer': k, 'Total Spending': f"${v:,.0f}"} 
                            for k, v in sorted(mfr_totals.items(), key=lambda x: x[1], reverse=True)[:10]
                        ])
                        st.dataframe(top_mfrs, use_container_width=True)
                    
                    # Most common products
                    st.markdown("---")
                    st.markdown("#### Most Frequently Purchased Products")
                    product_counts = {}
                    for item in all_line_items:
                        if item['model_number']:
                            key = f"{item.get('manufacturer', 'Unknown')} {item['model_number']}"
                            product_counts[key] = product_counts.get(key, 0) + 1
                    
                    if product_counts:
                        top_products = pd.DataFrame([
                            {'Product': k, 'Purchase Count': v} 
                            for k, v in sorted(product_counts.items(), key=lambda x: x[1], reverse=True)[:15]
                        ])
                        st.dataframe(top_products, use_container_width=True)

if __name__ == "__main__":
    main()
