# 🎯 E-Rate Lead Management System — Enterprise Edition

Advanced E-Rate opportunity finder with historical data, budget tracking, and comprehensive lead management for E-Rate sales teams.

**Territory:** North Texas (north of Waco, west of I-35E) + Oklahoma

## 🚀 Access the App

**Live App:** https://YOUR-APP-URL.streamlit.app

## ✨ Key Features

### 📊 Enhanced Multi-Select Filters

- **40+ Manufacturers**: Cisco, Meraki, Aruba, Fortinet, Palo Alto, SonicWall, Juniper, Ruckus, Extreme Networks, WatchGuard, APC, Tripp Lite, and more
- **13 Equipment Categories**: Switches, Wireless Access Points, Firewalls, Routers, UPS/Battery Backup, Cabling, Racks, Antennas, Transceivers, and more
- **Multi-Select Capability**: Choose multiple manufacturers and equipment types at once for comprehensive searches

### 💰 Financial Intelligence

- **Form 471 History**: View past E-Rate purchases and winning vendors (3-year lookback)
- **Category 2 Budget Tracking**: See total, used, and remaining C2 budget for each applicant
- **Funding Insights**: Identify districts with significant remaining budgets vs. those near their cap
- **Smart Prioritization**: Focus on high-value targets with available E-Rate funding

### 🎯 Lead Management & Tracking

- **Mark Leads as Pursuing**: Flag opportunities you're actively working on
- **Sales Notes**: Add and save notes for each lead
- **Filter Pursued Leads**: View only your active pipeline
- **Persistent Storage**: All tracking data saved to SQLite database (survives session restarts)

### 🔍 360° Account View

Access comprehensive intelligence for each applicant:

1. **Current Request (470)**: Equipment functions and manufacturer preferences
2. **Historical Funding (471)**: Past purchases, service providers, and funding amounts
3. **E-Rate Budget**: Visual progress bar showing C2 budget utilization
4. **External Insights**: Track bond initiatives and board meeting intelligence

### 🗺️ Geographic Intelligence

- **Pre-Filtered Territory**: Automatically filters to North TX (north of Waco, west of I-35E) + Oklahoma
- **Interactive Map**: Visual representation of opportunities with color-coded markers
- **Geocoded Locations**: Cached geocoding for instant display

## 📖 How to Use

### Quick Start

1. **Upload Data**: Upload FFL Excel export or use USAC API fallback
2. **Apply Filters**: Select manufacturers and equipment types you sell
3. **Click Analyze**: Process and display opportunities
4. **View Results**: Explore on map or in table view
5. **Deep Dive**: Click any applicant to view 360° account details
6. **Track Leads**: Mark opportunities as pursuing and add sales notes
7. **Export**: Download filtered results to Excel

### Advanced Workflows

**Finding Firewall Opportunities:**
1. Select "Firewalls" from Equipment Functions
2. Select "Fortinet", "Palo Alto Networks", "SonicWall" from Manufacturers
3. Click Analyze
4. Review each opportunity's details to see their C2 budget and past vendors

**Managing Your Pipeline:**
1. Mark promising leads as "Actively Pursuing"
2. Add notes about contact info, next steps, etc.
3. Check "Show only pursued leads" to view your pipeline
4. Track progress in the account detail view

**Identifying High-Value Targets:**
1. Review opportunities in the table
2. Click an applicant to view their details
3. Check their C2 Budget tab - look for >50% remaining budget
4. Check Form 471 History to see past purchasing patterns
5. Add insights about bonds or board meetings in External Insights tab

## 🔄 Data Sources

### Form 470 (Current Requests)
- **Source**: USAC Open Data API or Funds For Learning Excel export
- **Refresh**: Daily (API mode) or manual upload
- **Fields**: Applicant name, location, equipment requested, manufacturers

### Form 471 (Historical Commitments)
- **Source**: USAC Open Data API
- **Lookback**: 3 years
- **Data**: Funding amounts, service providers, equipment purchased

### Category 2 Budget
- **Source**: USAC Open Data API
- **Data**: Total budget, used amount, remaining budget, percentage utilized
- **Cycle**: Current 5-year funding cycle

## 💡 Best Practices

### Effective Filtering
- Use manufacturer filters to match your product portfolio
- Combine multiple equipment types to find comprehensive projects
- Start broad, then narrow down based on budget and fit

### Lead Qualification
- ✅ **High Priority**: >50% C2 budget remaining, matches your products, no recent contract with competitor
- ⚠️ **Medium Priority**: Some budget remaining, partial match to products
- ❌ **Low Priority**: <10% budget remaining, recent multi-year contract with competitor

### Territory Management
- All data is pre-filtered to your territory (North TX + OK)
- Focus on districts with upcoming Form 470s AND available budget
- Track bond initiatives for non-E-Rate funding opportunities

## 🔐 Access & Privacy

- This is an internal sales tool
- Lead tracking data is stored locally per deployment
- Contact IT for access credentials

## 📞 Support

Questions or issues? Contact your IT department or sales operations team.

## 🛠️ Technical Details

**Built with:**
- Streamlit (UI framework)
- USAC Open Data API (Socrata platform)
- SQLite (lead tracking)
- Folium (interactive maps)
- GeoPy (geocoding)

**Deployed on:** Streamlit Cloud

---

## 📋 Changelog

### v2.0 - Enterprise Edition (Latest)
- ✨ Added 40+ manufacturers (expanded from 2)
- ✨ Added 13 equipment categories with multi-select
- ✨ Integrated Form 471 historical funding data
- ✨ Integrated Category 2 budget tracking
- ✨ Added lead management system with SQLite persistence
- ✨ Added 360° account detail view
- ✨ Added external insights tracking (bonds, board meetings)
- ✨ Enhanced UI with feature documentation
- ✨ Added "Show only pursued leads" filter
- ✨ Added pursued leads count metric
- 🚀 Performance optimizations with caching

### v1.0 - Initial Release
- Basic Form 470 search
- Map visualization
- Excel export
- Geographic filtering (TX/OK)
