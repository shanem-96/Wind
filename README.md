# German Wind Turbines Interactive Map

This Streamlit application visualizes wind turbine data across Germany on an interactive map. The app displays detailed information about each wind turbine including power capacity, model, installation date, and owner.

## Features

- **Interactive Map**: Click on markers to see detailed information about each wind turbine
- **Color-coded Markers**: Visual distinction based on power capacity
  - 🟢 Green: < 2 MW
  - 🟠 Orange: 2-3 MW
  - 🔴 Red: > 3 MW
- **Filtering Options**: Filter by federal state, power range, installation year, and manufacturer
- **Statistics Dashboard**: View total and filtered turbine counts, average power, and total capacity
- **Data Table**: Optional detailed table view of filtered data

## Installation

1. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run wind_turbines_app.py
```

2. Open your browser and navigate to the provided URL (typically `http://localhost:8501`)

3. Use the sidebar filters to customize your view:
   - Select specific federal states
   - Adjust power range (kW)
   - Choose installation year range
   - Filter by manufacturer

4. Click on map markers to see detailed information in popups

5. Hover over markers to see tooltip information

## Data Source

The data is sourced from the German wind turbine registry (MaStR - Marktstammdatenregister).

## Data Statistics

- **Total Wind Turbines**: 29,055
- **Federal States**: 16 (all German states)
- **Power Range**: 35 kW - 8,000 kW
- **Installation Years**: 1983 - 2025

## File Structure

- `wind_turbines_app.py` - Main Streamlit application
- `requirements.txt` - Python dependencies
- `Wind Turbines Germany.csv` - Source data file
- `README.md` - This documentation

## Requirements

- Python 3.9+
- streamlit >= 1.28.0
- pandas >= 2.0.0
- folium >= 0.17.0
- streamlit-folium >= 0.18.0
- numpy >= 1.24.0

## Troubleshooting

If you encounter any issues:

1. Ensure all required packages are installed
2. Check that the CSV file is in the same directory as the app
3. Verify that coordinates are valid (the app automatically filters out invalid coordinates)

## Map Legend

The interactive map shows:
- **Tooltips**: Quick preview when hovering over markers
- **Popups**: Detailed information when clicking on markers
- **Marker Clustering**: For better performance with large datasets

## Performance Notes

- The app uses marker clustering for optimal performance
- Data is cached to improve loading times
- Filtering operations are applied client-side for responsiveness
