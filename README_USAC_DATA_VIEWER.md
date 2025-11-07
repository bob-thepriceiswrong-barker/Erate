# 🎓 USAC E-Rate Data Viewer

A Streamlit application for accessing and analyzing USAC E-Rate data via the SODA3 API.

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/bob-thepriceiswrong-barker/Erate.git
cd Erate

# Install dependencies
pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run usac_data_viewer.py
```

The app will open in your browser at `http://localhost:8501`

## 📊 Features

### Available Datasets

1. **C2 Budget Tool (FY2021+)**
   - Five-year budget cycle information starting FY2021
   - Track C2 budget allocation and spending
   - Identify entities with remaining budget
   - Filter by BEN number and budget cycle

2. **Form 471 Basic Information**
   - Applicant details from FCC Form 471
   - Historical funding requests
   - Service provider information
   - Filter by state, entity name, and funding year

### Key Capabilities

- ✅ **Smart Querying**: Configure limit and offset for pagination
- ✅ **Advanced Filtering**: Filter by state, entity name, year, BEN, and more
- ✅ **Data Preview**: Interactive dataframe with search capability
- ✅ **Column Selection**: Choose which columns to display
- ✅ **Multiple Export Formats**: CSV, JSON, and Excel
- ✅ **Statistics**: View summary statistics for all columns
- ✅ **Detailed Record View**: Inspect individual records in JSON format
- ✅ **Caching**: 1-hour cache for faster repeated queries
- ✅ **Error Handling**: User-friendly error messages and troubleshooting tips

## 📖 Usage Guide

### Basic Workflow

1. **Select Dataset**
   - Choose from "C2 Budget Tool" or "Form 471 Basic Information"

2. **Configure Query Options**
   - **Limit**: Number of records (10-10,000)
   - **Offset**: Starting position for pagination

3. **Add Filters (Optional)**
   - Form 471: State, Entity Name, Funding Year
   - C2 Budget: BEN Number, Budget Cycle

4. **Fetch Data**
   - Click "🔄 Fetch Data" button
   - Wait for data to load (cached for 1 hour)

5. **Explore Results**
   - View summary metrics
   - Search within data
   - Select specific columns
   - Export to CSV/JSON/Excel

### Example Queries

#### Find All Texas Schools for 2024

```
Dataset: Form 471 Basic Information
State: TX
Funding Year: 2024
Limit: 500
```

#### Check C2 Budget for Specific Entity

```
Dataset: C2 Budget Tool (FY2021+)
BEN Number: [enter BEN]
Limit: 10
```

#### Pagination Example

```
Query 1: Offset = 0, Limit = 100    (records 1-100)
Query 2: Offset = 100, Limit = 100  (records 101-200)
Query 3: Offset = 200, Limit = 100  (records 201-300)
```

## 🔧 Technical Details

### SODA3 API

This application uses the **SODA3 API** (not SODA2), which has a different structure:

**Request Method**: POST (not GET)

**Headers**:
```json
{
  "Content-Type": "application/json",
  "X-App-Token": "MFa7tcJ2Pybss1tK93iriT9qW"
}
```

**Request Body**:
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

**Response Format**:
```json
{
  "columns": [
    {"name": "column1"},
    {"name": "column2"}
  ],
  "rows": [
    [value1, value2],
    [value1, value2]
  ]
}
```

### API Endpoints

#### C2 Budget Tool
- **URL**: `https://opendata.usac.org/api/v3/views/6brt-5pbv/query.json`
- **Description**: C2 budget data for five-year cycles starting FY2021
- **Filters**: `ben`, `budget_cycle`

#### Form 471 Basic Information
- **URL**: `https://opendata.usac.org/api/v3/views/9s6i-myen/query.json`
- **Description**: Basic applicant information from FCC Form 471
- **Filters**: `applicant_state`, `applicant_name`, `funding_year`

### App Token

The application uses the app token: `MFa7tcJ2Pybss1tK93iriT9qW`

This is hardcoded in the application. For production use, consider using environment variables or Streamlit secrets:

```python
# .streamlit/secrets.toml
USAC_APP_TOKEN = "MFa7tcJ2Pybss1tK93iriT9qW"
```

```python
# In code
APP_TOKEN = st.secrets["USAC_APP_TOKEN"]
```

## 📋 Dependencies

```
streamlit>=1.28.0
requests>=2.31.0
pandas>=2.0.0
openpyxl>=3.1.0
```

## 🎯 Use Cases

### For Sales Teams
- Identify prospects with available C2 budget
- Research school districts in your territory
- Track historical funding patterns
- Export qualified leads to CRM

### For Analysts
- Analyze E-Rate funding trends
- Compare budget utilization across entities
- Study geographic funding distribution
- Generate custom reports

### For Applicants
- Check your C2 budget status
- Research comparable entities
- Plan funding requests
- Track application history

## 🔍 Troubleshooting

### Common Issues

**Error: "Request timed out"**
- The API server is slow or overloaded
- Try reducing the limit
- Try again in a few minutes

**Error: "Connection error"**
- Check your internet connection
- Verify you can access https://opendata.usac.org

**Error: "HTTP Error 401"**
- App token may be invalid
- Verify the token in the code

**Error: "No records found"**
- Your filters may be too restrictive
- Try removing some filters
- Check spelling of state codes (must be uppercase, e.g., "TX")

### Performance Tips

1. **Start Small**: Use limit=100 for initial testing
2. **Use Filters**: Narrow down results to reduce data transfer
3. **Cache**: Identical queries are cached for 1 hour
4. **Pagination**: For large datasets, use offset to paginate

## 📚 Additional Resources

- [USAC Open Data Portal](https://opendata.usac.org/)
- [E-Rate Program Information](https://www.usac.org/e-rate/)
- [SODA API Documentation](https://dev.socrata.com/docs/endpoints.html)

## 🤝 Related Applications

This repository also contains:

- **erate_webapp.py**: E-Rate Lead Management System with Form 470/471 integration, lead tracking, and territory management
  - See [README.md](README.md) for details

## 📄 License

This is an internal tool for accessing public USAC E-Rate data.

## 🆘 Support

For questions or issues:
- Check the troubleshooting section above
- Review the in-app help documentation (ℹ️ expanders)
- Consult USAC Open Data documentation

---

**Version**: 1.0
**API**: USAC Open Data SODA3
**Last Updated**: 2025-01-07
