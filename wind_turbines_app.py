import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import numpy as np
import pydeck as pdk

# Set page config
st.set_page_config(
    page_title="German Wind Turbines Map",
    page_icon="🌪️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🌪️ German Wind Turbines Interactive Map")
st.markdown("Interactive map showing wind turbines across Germany with detailed information")

# Load data
@st.cache_data(show_spinner="Loading wind turbine data…")
def load_data():
    # Read only the columns we need, using the fast PyArrow engine
    required_cols = [
        'Energieträger',
        'Anzeige-Name der Einheit',
        'Bundesland',
        'Ort',
        'Lat',
        'Long',
        'Nettonennleistung der Einheit',
        'Typenbezeichnung',
        'Inbetriebnahmedatum der Einheit',
        'Name des Anlagenbetreibers (nur Org.)',
        'Hersteller der Windenergieanlage',
        'Nabenhöhe der Windenergieanlage',
        'Rotordurchmesser der Windenergieanlage',
    ]

    df = pd.read_csv(
        "Wind Turbines Germany.csv",
        usecols=required_cols,
        encoding="utf-8",
        engine="pyarrow",
    )

    # Keep only wind turbines
    wind_df = df[df['Energieträger'] == 'Wind'].copy()

    # Convert numeric columns
    for col in ['Lat', 'Long', 'Nettonennleistung der Einheit',
                'Nabenhöhe der Windenergieanlage', 'Rotordurchmesser der Windenergieanlage']:
        if col in wind_df.columns:
            wind_df[col] = pd.to_numeric(wind_df[col], errors='coerce')

    # Parse dates (day-first) after read; pyarrow engine doesn't support dayfirst
    if 'Inbetriebnahmedatum der Einheit' in wind_df.columns:
        wind_df['Inbetriebnahmedatum der Einheit'] = pd.to_datetime(
            wind_df['Inbetriebnahmedatum der Einheit'], errors='coerce', dayfirst=True
        )

    # Drop rows with invalid coordinates
    wind_df = wind_df.dropna(subset=['Lat', 'Long'])

    # Derived columns for fast filtering and rendering
    wind_df['power_mw'] = wind_df['Nettonennleistung der Einheit'] / 1000.0
    wind_df['install_year'] = wind_df['Inbetriebnahmedatum der Einheit'].dt.year
    wind_df['install_date_str'] = wind_df['Inbetriebnahmedatum der Einheit'].dt.strftime('%Y-%m-%d')
    wind_df['lat'] = wind_df['Lat']
    wind_df['lon'] = wind_df['Long']

    # Memory optimizations
    for cat_col in ['Bundesland', 'Ort', 'Typenbezeichnung',
                    'Hersteller der Windenergieanlage', 'Name des Anlagenbetreibers (nur Org.)']:
        if cat_col in wind_df.columns:
            wind_df[cat_col] = wind_df[cat_col].astype('category')

    # Vectorized color assignment for power buckets
    conditions = [
        wind_df['power_mw'] < 2,
        wind_df['power_mw'].between(2, 3, inclusive='left'),
        wind_df['power_mw'].between(3, 5, inclusive='left'),
        wind_df['power_mw'] >= 5,
    ]
    # Compute a color bucket per row (0..3; default 4) and map to RGBA
    bucket = np.select(conditions, [0, 1, 2, 3], default=4).astype(int)
    palette = [
        (38, 166, 91, 200),    # green: <2 MW
        (255, 159, 67, 200),   # orange: 2-3 MW
        (235, 59, 90, 200),    # red: 3-5 MW
        (136, 84, 208, 200),   # purple: >5 MW
        (100, 100, 100, 200),  # grey: default/unknown
    ]
    wind_df['color_rgba'] = [palette[int(i)] for i in bucket]

    return wind_df

# Load the data
wind_data = load_data()

# Sidebar filters
st.sidebar.header("🔍 Filter Options")

# Map renderer (pydeck is much faster; folium kept for detailed popups)
renderer = st.sidebar.radio(
    "Map renderer",
    options=("High performance (pydeck)", "Detailed clustering (folium)"),
    index=0,
    help="Pydeck renders thousands of points smoothly. Folium offers HTML popups but may be slower.",
)

# Marker size control (fixed at 64px for better visibility)
marker_size_px = 64

# Filter by federal state (Bundesland)
states = sorted(wind_data['Bundesland'].dropna().unique())
selected_states = st.sidebar.multiselect(
    "Select Federal States",
    options=states,
    default=states  # Default to all states
)

# Filter by power range
power_range = st.sidebar.slider(
    "Power Range (kW)",
    min_value=int(wind_data['Nettonennleistung der Einheit'].min()),
    max_value=int(wind_data['Nettonennleistung der Einheit'].max()),
    value=(500, 4000)
)

# Filter by installation year (precomputed column)
min_year = int(wind_data['install_year'].min())
max_year = int(wind_data['install_year'].max())

selected_years = st.sidebar.slider(
    "Installation Year Range",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year)
)

# Filter by manufacturer
manufacturers = sorted(wind_data['Hersteller der Windenergieanlage'].dropna().unique())
selected_manufacturers = st.sidebar.multiselect(
    "Select Manufacturers",
    options=manufacturers,
    default=[]
)

@st.cache_data(show_spinner=False)
def apply_filters(df: pd.DataFrame,
                  states_sel: list,
                  power_rng: tuple,
                  years_rng: tuple,
                  manufacturers_sel: list) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    if states_sel:
        mask &= df['Bundesland'].isin(states_sel)
    mask &= (df['Nettonennleistung der Einheit'] >= power_rng[0]) & \
            (df['Nettonennleistung der Einheit'] <= power_rng[1])
    mask &= (df['install_year'] >= years_rng[0]) & (df['install_year'] <= years_rng[1])
    if manufacturers_sel:
        mask &= df['Hersteller der Windenergieanlage'].isin(manufacturers_sel)
    return df.loc[mask]

filtered_data = apply_filters(wind_data, selected_states, power_range, selected_years, selected_manufacturers)

# Display statistics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Wind Turbines", len(wind_data))

with col2:
    st.metric("Filtered Turbines", len(filtered_data))

with col3:
    avg_power = filtered_data['Nettonennleistung der Einheit'].mean() / 1000  # Convert to MW
    st.metric("Avg Power", f"{avg_power:.1f} MW")

with col4:
    total_power = filtered_data['Nettonennleistung der Einheit'].sum() / 1000  # Convert to MW
    st.metric("Total Power", f"{total_power:,.0f} MW")

# ---------- High-performance map (pydeck) ----------
@st.cache_data(show_spinner=False)
def create_pydeck_map(data: pd.DataFrame, radius_px: int):
    view_state = pdk.ViewState(latitude=51.1657, longitude=10.4515, zoom=5.7, pitch=0)

    # Prepare a minimal DataFrame for the layer (pydeck copies to JSON)
    display_df = data[[
        'Anzeige-Name der Einheit', 'Bundesland', 'Ort',
        'Nettonennleistung der Einheit', 'Typenbezeichnung',
        'install_date_str', 'Name des Anlagenbetreibers (nur Org.)',
        'Hersteller der Windenergieanlage', 'Nabenhöhe der Windenergieanlage',
        'Rotordurchmesser der Windenergieanlage', 'lat', 'lon', 'power_mw', 'color_rgba'
    ]].rename(columns={
        'Anzeige-Name der Einheit': 'Name',
        'Nettonennleistung der Einheit': 'Power_kW',
        'Typenbezeichnung': 'Model',
        'install_date_str': 'InstallationDate',
        'Name des Anlagenbetreibers (nur Org.)': 'Owner',
        'Hersteller der Windenergieanlage': 'Manufacturer',
        'Nabenhöhe der Windenergieanlage': 'HubHeight',
        'Rotordurchmesser der Windenergieanlage': 'RotorDiameter',
    })

    # Inject radius from UI for this render
    display_df['radius_px'] = int(radius_px)

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=display_df,
        get_position='[lon, lat]',
        get_radius='radius_px',  # pixel radius scales with zoom automatically
        get_fill_color='color_rgba',
        pickable=True,
        radius_min_pixels=2,
        radius_max_pixels=80,
        stroked=False,
        auto_highlight=True,
    )

    tooltip = {
        "html": (
            "<b>{Name}</b><br>"
            "<b>Power:</b> {power_mw} MW<br>"
            "<b>Model:</b> {Model}<br>"
            "<b>Installation Date:</b> {InstallationDate}<br>"
            "<b>Owner:</b> {Owner}<br>"
            "<b>State:</b> {Bundesland}<br>"
            "<b>Manufacturer:</b> {Manufacturer}<br>"
            "<b>Hub Height:</b> {HubHeight} m<br>"
            "<b>Rotor Diameter:</b> {RotorDiameter} m"
        ),
        "style": {"backgroundColor": "#2b2b2b", "color": "white"},
    }

    deck = pdk.Deck(layers=[layer], initial_view_state=view_state, map_style='light', tooltip=tooltip)
    return deck

# Create map (folium)
@st.cache_data(show_spinner=False)
def create_map(data):
    # Center the map on Germany
    germany_center = [51.1657, 10.4515]

    # Create the map
    m = folium.Map(
        location=germany_center,
        zoom_start=6,
        tiles='OpenStreetMap'
    )

    # Add marker cluster for better performance
    marker_cluster = MarkerCluster().add_to(m)

    # Add markers for each wind turbine (folium is slower for many points)
    for idx, row in data.iterrows():
        # Create tooltip content
        tooltip_content = f"""
        <b>Power:</b> {row['Nettonennleistung der Einheit']/1000:.1f} MW<br>
        <b>Model:</b> {row['Typenbezeichnung']}<br>
        <b>Installation Date:</b> {row['install_date_str']}<br>
        <b>Owner:</b> {row['Name des Anlagenbetreibers (nur Org.)']}<br>
        <b>State:</b> {row['Bundesland']}<br>
        <b>Manufacturer:</b> {row['Hersteller der Windenergieanlage']}
        """

        # Create popup with more detailed information
        popup_content = f"""
        <b>{row['Anzeige-Name der Einheit']}</b><br>
        <b>Power:</b> {row['Nettonennleistung der Einheit']/1000:.1f} MW<br>
        <b>Model:</b> {row['Typenbezeichnung']}<br>
        <b>Installation Date:</b> {row['install_date_str']}<br>
        <b>Owner:</b> {row['Name des Anlagenbetreibers (nur Org.)']}<br>
        <b>State:</b> {row['Bundesland']}<br>
        <b>Location:</b> {row['Ort']}<br>
        <b>Manufacturer:</b> {row['Hersteller der Windenergieanlage']}<br>
        <b>Hub Height:</b> {row['Nabenhöhe der Windenergieanlage']} m<br>
        <b>Rotor Diameter:</b> {row['Rotordurchmesser der Windenergieanlage']} m
        """

        # Color coding based on power
        power_mw = row['Nettonennleistung der Einheit'] / 1000
        if power_mw < 2:
            color = 'green'
        elif power_mw < 3:
            color = 'orange'
        elif power_mw < 5:
            color = 'red'
        else:
            color = 'purple'

        folium.CircleMarker(
            location=[row['Lat'], row['Long']],
            radius=5,
            popup=folium.Popup(popup_content, max_width=300),
            tooltip=tooltip_content,
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.7,
            weight=1
        ).add_to(marker_cluster)

    return m

# Create and display the map
if not filtered_data.empty:
    if renderer.startswith("High performance"):
        deck = create_pydeck_map(filtered_data, marker_size_px)
        st.pydeck_chart(deck, use_container_width=True, height=700)
    else:
        map_obj = create_map(filtered_data)
        st_folium(
            map_obj,
            returned_objects=[],
            use_container_width=True,
            height=700
        )

    # Legend
    st.markdown("### Legend")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        🟢 **Green**: < 2 MW
        """)
    with col2:
        st.markdown("""
        🟠 **Orange**: 2-3 MW
        """)
    with col3:
        st.markdown("""
        🔴 **Red**: 3-5 MW
        """)
    with col4:
        st.markdown("""
        🟣 **Purple**: > 5 MW
        """)

    # Data table
    if st.checkbox("Show filtered data table"):
        st.subheader("Filtered Wind Turbine Data")
        st.dataframe(
            filtered_data[[
                'Anzeige-Name der Einheit',
                'Bundesland',
                'Ort',
                'Nettonennleistung der Einheit',
                'Typenbezeichnung',
                'Inbetriebnahmedatum der Einheit',
                'Name des Anlagenbetreibers (nur Org.)',
                'Hersteller der Windenergieanlage',
                'Lat',
                'Long'
            ]].rename(columns={
                'Anzeige-Name der Einheit': 'Name',
                'Bundesland': 'State',
                'Ort': 'City',
                'Nettonennleistung der Einheit': 'Power (kW)',
                'Typenbezeichnung': 'Model',
                'Inbetriebnahmedatum der Einheit': 'Installation Date',
                'Name des Anlagenbetreibers (nur Org.)': 'Owner',
                'Hersteller der Windenergieanlage': 'Manufacturer'
            }),
            width='stretch'
        )
else:
    st.warning("No wind turbines match the current filter criteria. Please adjust the filters.")

# Footer
st.markdown("---")
st.markdown("Data source: German wind turbine registry (MaStR)")
