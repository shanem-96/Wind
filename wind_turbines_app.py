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

# Add custom CSS to prevent slider value clipping and reduce top padding
st.markdown("""
    <style>
    .stSlider [data-baseweb="slider"] {
        padding-right: 30px;
    }
    /* Reduce top padding */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
    }
    /* Reduce space after title */
    h1 {
        margin-bottom: 0.5rem;
    }
    /* Reduce space after subtitle */
    .stMarkdown p {
        margin-bottom: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("⚡ German Renewable Energy Interactive Map")
st.markdown("Interactive maps showing wind turbines and solar installations across Germany")

# Create tabs for Wind, Solar, and Statistics
tab1, tab2, tab3 = st.tabs(["🌪️ Wind Turbines", "☀️ Solar Installations", "📊 Statistics"])

# ======================== WIND TURBINE DATA LOADING ========================
@st.cache_data(show_spinner="Loading wind turbine data…")
def load_data():
    # Read only the columns we need, using the fast PyArrow engine
    required_cols = [
        'MaStR-Nr. der Einheit',
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
        'Spannungsebene',
    ]

    df = pd.read_csv(
        "Wind Turbines Germany.csv",
        usecols=required_cols,
        encoding="utf-8",
        engine="pyarrow",
    )
    
    # Mark as operational
    df['Status'] = 'Operational'

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

@st.cache_data(show_spinner="Loading planning wind turbine data…")
def load_planning_data():
    # Read planning data with same columns, including MaStR number
    required_cols = [
        'MaStR-Nr. der Einheit',
        'Energieträger',
        'Anzeige-Name der Einheit',
        'Bundesland',
        'Ort',
        'Lat',
        'Lon',
        'Nettonennleistung der Einheit',
        'Typenbezeichnung',
        'Datum der geplanten Inbetriebnahme',
        'Name des Anlagenbetreibers (nur Org.)',
        'Hersteller der Windenergieanlage',
        'Nabenhöhe der Windenergieanlage',
        'Rotordurchmesser der Windenergieanlage',
        'Spannungsebene',
    ]
    
    df = pd.read_csv(
        "Wind In planning.csv",
        usecols=required_cols,
        encoding="utf-8",
        engine="pyarrow",
    )
    
    # Mark as planning
    df['Status'] = 'Planning'
    
    # Rename Lon to Long for consistency
    df = df.rename(columns={'Lon': 'Long', 'Datum der geplanten Inbetriebnahme': 'Inbetriebnahmedatum der Einheit'})
    
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
    
    # Cyan color for all planning turbines (onshore and offshore)
    wind_df['color_rgba'] = [(0, 191, 255, 200)] * len(wind_df)  # Deep Sky Blue
    
    return wind_df

@st.cache_data(show_spinner="Loading offshore operational wind turbine data…")
def load_offshore_operational_data():
    # Read offshore operational data with same columns as onshore
    required_cols = [
        'MaStR-Nr. der Einheit',
        'Energieträger',
        'Anzeige-Name der Einheit',
        'Bundesland',
        'Ort',
        'Lat',
        'Lon',
        'Nettonennleistung der Einheit',
        'Typenbezeichnung',
        'Inbetriebnahmedatum der Einheit',
        'Name des Anlagenbetreibers (nur Org.)',
        'Hersteller der Windenergieanlage',
        'Nabenhöhe der Windenergieanlage',
        'Rotordurchmesser der Windenergieanlage',
        'Spannungsebene',
    ]
    
    df = pd.read_csv(
        "Offshore Wind Operational.csv",
        usecols=required_cols,
        encoding="utf-8",
        engine="pyarrow",
    )
    
    # Mark as operational
    df['Status'] = 'Operational'
    
    # Rename Lon to Long for consistency
    df = df.rename(columns={'Lon': 'Long'})
    
    # Keep only wind turbines
    wind_df = df[df['Energieträger'] == 'Wind'].copy()
    
    # Convert numeric columns
    for col in ['Lat', 'Long', 'Nettonennleistung der Einheit',
                'Nabenhöhe der Windenergieanlage', 'Rotordurchmesser der Windenergieanlage']:
        if col in wind_df.columns:
            wind_df[col] = pd.to_numeric(wind_df[col], errors='coerce')
    
    # Parse dates
    if 'Inbetriebnahmedatum der Einheit' in wind_df.columns:
        wind_df['Inbetriebnahmedatum der Einheit'] = pd.to_datetime(
            wind_df['Inbetriebnahmedatum der Einheit'], errors='coerce', dayfirst=True
        )
    
    # Drop rows with invalid coordinates
    wind_df = wind_df.dropna(subset=['Lat', 'Long'])
    
    # Derived columns
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
wind_data_onshore = load_data()
wind_data_offshore = load_offshore_operational_data()
wind_data = pd.concat([wind_data_onshore, wind_data_offshore], ignore_index=True)
planning_data = load_planning_data()

# ======================== TAB 1: WIND TURBINES UI ========================
with tab1:
    # Wind turbine settings header
    st.sidebar.markdown("## 🌪️ Wind Turbine Settings")
    
    # Map renderer (pydeck is much faster; folium kept for detailed popups)
    renderer = st.sidebar.radio(
        "Map renderer",
        options=("High performance (pydeck)", "Detailed clustering (folium)"),
        index=0,
        help="Pydeck renders thousands of points smoothly. Folium offers HTML popups but may be slower.",
    )
    
    # Toggle for operational and planning turbines
    show_operational = st.sidebar.checkbox("Show operational turbines", value=True, help="Display wind turbines that are currently operational (onshore and offshore)")
    show_planning = st.sidebar.checkbox("Show turbines in planning", value=False, help="Display wind turbines that are currently in planning stage (onshore and offshore)")

    # Marker size control (fixed at 64px for better visibility)
    marker_size_px = 64

    # Filter by federal state (Bundesland)
    states = sorted(wind_data['Bundesland'].dropna().unique())
    selected_states = st.sidebar.multiselect(
        "Select Federal States",
        options=states,
        default=[],  # Default to no states (shows all)
        help="Leave empty to show all states, or select specific states to filter"
    )

    # Filter by power range
    # Include both operational and planning data for power range
    min_power_operational = int(wind_data['Nettonennleistung der Einheit'].min())
    max_power_operational = int(wind_data['Nettonennleistung der Einheit'].max())
    min_power_planning = int(planning_data['Nettonennleistung der Einheit'].min())
    max_power_planning = int(planning_data['Nettonennleistung der Einheit'].max())

    min_power = min(min_power_operational, min_power_planning)
    max_power = 15000  # Extended to accommodate larger turbines

    power_range = st.sidebar.slider(
        "Power Range (kW)",
        min_value=min_power,
        max_value=max_power,
        value=(min_power, max_power)
    )

    # Filter by installation year (precomputed column)
    # Include both operational and planning data for year range
    min_year_operational = int(wind_data['install_year'].min())
    max_year_operational = int(wind_data['install_year'].max())
    min_year_planning = int(planning_data['install_year'].min())
    max_year_planning = int(planning_data['install_year'].max())

    min_year = min(min_year_operational, min_year_planning)
    max_year = 2030  # Extended to accommodate planned turbines

    selected_years = st.sidebar.slider(
        "Installation/Planning Year Range",
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

    # Combine data based on toggle selections
    data_frames = []
    if show_operational:
        data_frames.append(filtered_data)
    if show_planning:
        filtered_planning = apply_filters(planning_data, selected_states, power_range, selected_years, selected_manufacturers)
        data_frames.append(filtered_planning)

    # Combine all selected data
    if data_frames:
        combined_data = pd.concat(data_frames, ignore_index=True) if len(data_frames) > 1 else data_frames[0]
    else:
        combined_data = pd.DataFrame()  # Empty dataframe if nothing is selected

    # Display statistics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_count = 0
        if show_operational:
            total_count += len(wind_data)
        if show_planning:
            total_count += len(planning_data)
        st.metric("Total Wind Turbines", total_count)

    with col2:
        st.metric("Filtered Turbines", len(combined_data))

    with col3:
        if not combined_data.empty:
            avg_power = combined_data['Nettonennleistung der Einheit'].mean() / 1000  # Convert to MW
            st.metric("Avg Power", f"{avg_power:.1f} MW")
        else:
            st.metric("Avg Power", "0 MW")

    with col4:
        if not combined_data.empty:
            total_power = combined_data['Nettonennleistung der Einheit'].sum() / 1000  # Convert to MW
            st.metric("Total Power", f"{total_power:,.0f} MW")
        else:
            st.metric("Total Power", "0 MW")

    # ---------- High-performance map (pydeck) ----------
    @st.cache_data(show_spinner=False)
    def create_pydeck_map(data: pd.DataFrame, radius_px: int):
        view_state = pdk.ViewState(latitude=51.1657, longitude=10.4515, zoom=5.7, pitch=0)

        # Prepare a minimal DataFrame for the layer (pydeck copies to JSON)
        display_df = data[[
            'MaStR-Nr. der Einheit', 'Anzeige-Name der Einheit', 'Bundesland', 'Ort',
            'Nettonennleistung der Einheit', 'Typenbezeichnung',
            'install_date_str', 'Name des Anlagenbetreibers (nur Org.)',
            'Hersteller der Windenergieanlage', 'Nabenhöhe der Windenergieanlage',
            'Rotordurchmesser der Windenergieanlage', 'Spannungsebene', 'lat', 'lon', 'power_mw', 'color_rgba', 'Status'
        ]].rename(columns={
            'MaStR-Nr. der Einheit': 'MaStR',
            'Anzeige-Name der Einheit': 'Name',
            'Nettonennleistung der Einheit': 'Power_kW',
            'Typenbezeichnung': 'Model',
            'install_date_str': 'InstallationDate',
            'Name des Anlagenbetreibers (nur Org.)': 'Owner',
            'Hersteller der Windenergieanlage': 'Manufacturer',
            'Nabenhöhe der Windenergieanlage': 'HubHeight',
            'Rotordurchmesser der Windenergieanlage': 'RotorDiameter',
            'Spannungsebene': 'VoltageLevel',
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
                "<b>MaStR:</b> {MaStR}<br>"
                "<b>Name:</b> {Name}<br>"
                "<b>Power:</b> {power_mw} MW<br>"
                "<b>Model:</b> {Model}<br>"
                "<b>Date:</b> {InstallationDate}<br>"
                "<b>Owner:</b> {Owner}<br>"
                "<b>State:</b> {Bundesland}<br>"
                "<b>Manufacturer:</b> {Manufacturer}<br>"
                "<b>Hub Height:</b> {HubHeight} m<br>"
                "<b>Rotor Diameter:</b> {RotorDiameter} m<br>"
                "<b>Voltage Level:</b> {VoltageLevel}<br>"
                "<b>Status:</b> {Status}"
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
            # Determine date label based on status
            date_label = "Planned Date" if row.get('Status') == 'Planning' else "Installation Date"
        
            tooltip_content = f"""
            <b>MaStR:</b> {row['MaStR-Nr. der Einheit']}<br>
            <b>Power:</b> {row['Nettonennleistung der Einheit']/1000:.1f} MW<br>
            <b>Model:</b> {row['Typenbezeichnung']}<br>
            <b>{date_label}:</b> {row['install_date_str']}<br>
            <b>Owner:</b> {row['Name des Anlagenbetreibers (nur Org.)']}<br>
            <b>State:</b> {row['Bundesland']}<br>
            <b>Manufacturer:</b> {row['Hersteller der Windenergieanlage']}<br>
            <b>Voltage Level:</b> {row['Spannungsebene']}
            """

            # Create popup with more detailed information
            popup_content = f"""
            <b>MaStR:</b> {row['MaStR-Nr. der Einheit']}</b><br>
            <b>Name:</b> {row['Anzeige-Name der Einheit']}<br>
            <b>Power:</b> {row['Nettonennleistung der Einheit']/1000:.1f} MW<br>
            <b>Model:</b> {row['Typenbezeichnung']}<br>
            <b>{date_label}:</b> {row['install_date_str']}<br>
            <b>Owner:</b> {row['Name des Anlagenbetreibers (nur Org.)']}<br>
            <b>State:</b> {row['Bundesland']}<br>
            <b>Location:</b> {row['Ort']}<br>
            <b>Manufacturer:</b> {row['Hersteller der Windenergieanlage']}<br>
            <b>Hub Height:</b> {row['Nabenhöhe der Windenergieanlage']} m<br>
            <b>Rotor Diameter:</b> {row['Rotordurchmesser der Windenergieanlage']} m<br>
            <b>Voltage Level:</b> {row['Spannungsebene']}<br>
            <b>Status:</b> {row['Status']}
            """

            # Color coding based on status and power
            if row.get('Status') == 'Planning':
                color = 'cyan'  # All planning turbines
            else:
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
    if not combined_data.empty:
        if renderer.startswith("High performance"):
            deck = create_pydeck_map(combined_data, marker_size_px)
            st.pydeck_chart(deck, use_container_width=True, height=700)
        else:
            map_obj = create_map(combined_data)
            st_folium(
                map_obj,
                returned_objects=[],
                use_container_width=True,
                height=700
            )

        # Legend
        st.markdown("### Legend")
        legend_cols = st.columns(5) if show_planning else st.columns(4)
        with legend_cols[0]:
            st.markdown("""
            🟢 **Green**: < 2 MW
            """)
        with legend_cols[1]:
            st.markdown("""
            🟠 **Orange**: 2-3 MW
            """)
        with legend_cols[2]:
            st.markdown("""
            🔴 **Red**: 3-5 MW
            """)
        with legend_cols[3]:
            st.markdown("""
            🟣 **Purple**: > 5 MW
            """)
        if show_planning:
            with legend_cols[4]:
                st.markdown("""
                🔵 **Cyan**: In Planning
                """)

        # Data table
        if st.checkbox("Show filtered data table"):
            st.subheader("Filtered Wind Turbine Data")
            st.dataframe(
                combined_data[[
                    'MaStR-Nr. der Einheit',
                    'Anzeige-Name der Einheit',
                    'Bundesland',
                    'Ort',
                    'Nettonennleistung der Einheit',
                    'Typenbezeichnung',
                    'Inbetriebnahmedatum der Einheit',
                    'Name des Anlagenbetreibers (nur Org.)',
                    'Hersteller der Windenergieanlage',
                    'Lat',
                    'Long',
                    'Status'
                ]].rename(columns={
                    'MaStR-Nr. der Einheit': 'MaStR Number',
                    'Anzeige-Name der Einheit': 'Name',
                    'Bundesland': 'State',
                    'Ort': 'City',
                    'Nettonennleistung der Einheit': 'Power (kW)',
                    'Typenbezeichnung': 'Model',
                    'Inbetriebnahmedatum der Einheit': 'Date',
                    'Name des Anlagenbetreibers (nur Org.)': 'Owner',
                    'Hersteller der Windenergieanlage': 'Manufacturer'
                }),
                width='stretch'
            )
    else:
        if not show_operational and not show_planning:
            st.warning("⚠️ Please enable at least one turbine type (Operational or Planning) to display data on the map.")
        else:
            st.warning("No wind turbines match the current filter criteria. Please adjust the filters.")

    # Footer
    st.markdown("---")
    st.markdown("Data source: German wind turbine registry (MaStR)")

# ======================== TAB 2: SOLAR INSTALLATIONS ========================
with tab2:
    # Load solar data
    @st.cache_data(show_spinner="Loading solar installation data…")
    def load_solar_data():
        required_cols = [
            'MaStR-Nr. der Einheit',
            'Energieträger',
            'Betriebs-Status',
            'Anzeige-Name der Einheit',
            'Bundesland',
            'Ort',
            'Koordinate: Breitengrad (WGS84)',
            'Koordinate: Längengrad (WGS84)',
            'Nettonennleistung der Einheit',
            'Art der Solaranlage',
            'Inbetriebnahmedatum der Einheit',
            'Name des Anlagenbetreibers (nur Org.)',
            'Name des Solarparks',
            'Anzahl der Solar-Module',
            'Spannungsebene',
        ]
        
        df = pd.read_csv(
            "Solar greater 1MW Ground Mounted.csv",
            usecols=required_cols,
            encoding="utf-8",
            engine="pyarrow",
        )
        
        df = df.rename(columns={
            'Koordinate: Breitengrad (WGS84)': 'Lat',
            'Koordinate: Längengrad (WGS84)': 'Long'
        })
        
        # Keep only solar installations
        solar_df = df[df['Energieträger'] == 'Solare Strahlungsenergie'].copy()
        
        # Convert numeric columns
        for col in ['Lat', 'Long', 'Nettonennleistung der Einheit', 'Anzahl der Solar-Module']:
            if col in solar_df.columns:
                solar_df[col] = pd.to_numeric(solar_df[col], errors='coerce')
        
        # Parse dates
        if 'Inbetriebnahmedatum der Einheit' in solar_df.columns:
            solar_df['Inbetriebnahmedatum der Einheit'] = pd.to_datetime(
                solar_df['Inbetriebnahmedatum der Einheit'], errors='coerce', dayfirst=True
            )
        
        # Drop rows with invalid coordinates
        solar_df = solar_df.dropna(subset=['Lat', 'Long'])
        
        # Derived columns
        solar_df['power_mw'] = solar_df['Nettonennleistung der Einheit'] / 1000.0
        solar_df['install_year'] = solar_df['Inbetriebnahmedatum der Einheit'].dt.year
        solar_df['install_date_str'] = solar_df['Inbetriebnahmedatum der Einheit'].dt.strftime('%Y-%m-%d')
        solar_df['lat'] = solar_df['Lat']
        solar_df['lon'] = solar_df['Long']
        
        # Memory optimizations
        for cat_col in ['Bundesland', 'Ort', 'Art der Solaranlage', 'Name des Anlagenbetreibers (nur Org.)']:
            if cat_col in solar_df.columns:
                solar_df[cat_col] = solar_df[cat_col].astype('category')
        
        # Separate operational and planning installations
        solar_df['Status'] = solar_df['Betriebs-Status'].apply(
            lambda x: 'Planning' if x == 'In Planung' else 'Operational'
        )
        
        # Color by status and power
        # Planning installations get green color, operational get power-based colors
        def get_solar_color(row):
            if row['Status'] == 'Planning':
                return (50, 205, 50, 200)  # Lime green for planning
            else:
                # Operational: color by power
                power = row['power_mw']
                if power < 2:
                    return (255, 215, 0, 200)     # gold: <2 MW
                elif power < 5:
                    return (255, 165, 0, 200)     # orange: 2-5 MW
                elif power < 10:
                    return (255, 69, 0, 200)      # red-orange: 5-10 MW
                else:
                    return (178, 34, 34, 200)     # dark red: >10 MW
        
        solar_df['color_rgba'] = solar_df.apply(get_solar_color, axis=1)
        
        return solar_df
    
    solar_data = load_solar_data()
    
    # Solar settings header
    st.sidebar.markdown("## ☀️ Solar Installation Settings")
    
    # Solar filters
    show_solar_operational = st.sidebar.checkbox("Show operational solar installations", value=True, help="Display operational ground-mounted solar installations > 1MW")
    show_solar_planning = st.sidebar.checkbox("Show solar installations in planning", value=False, help="Display solar installations in planning stage")
    
    # Filter by federal state
    solar_states = sorted(solar_data['Bundesland'].dropna().unique())
    selected_solar_states = st.sidebar.multiselect(
        "Select Federal States (Solar)",
        options=solar_states,
        default=[],  # Default to no states (shows all)
        help="Leave empty to show all states, or select specific states to filter"
    )
    
    # Filter by power range
    solar_power_range = st.sidebar.slider(
        "Solar Power Range (kW)",
        min_value=int(solar_data['Nettonennleistung der Einheit'].min()),
        max_value=150000,
        value=(int(solar_data['Nettonennleistung der Einheit'].min()), 
               min(int(solar_data['Nettonennleistung der Einheit'].max()), 150000))
    )
    
    # Manual input for solar power range
    st.sidebar.markdown("**Or enter custom solar power range:**")
    solar_col1, solar_col2 = st.sidebar.columns(2)
    with solar_col1:
        solar_min_power_manual = st.number_input(
            "Min (kW)",
            min_value=0,
            max_value=150000,
            value=solar_power_range[0],
            step=100,
            key="solar_min_power"
        )
    with solar_col2:
        solar_max_power_manual = st.number_input(
            "Max (kW)",
            min_value=0,
            max_value=150000,
            value=solar_power_range[1],
            step=100,
            key="solar_max_power"
        )
    
    # Use manual input if provided, otherwise use slider
    solar_power_range = (solar_min_power_manual, solar_max_power_manual)
    
    # Filter by installation year
    solar_min_year = int(solar_data['install_year'].min())
    solar_max_year = 2030  # Extended to accommodate planned installations
    selected_solar_years = st.sidebar.slider(
        "Solar Installation Year Range",
        min_value=solar_min_year,
        max_value=solar_max_year,
        value=(solar_min_year, solar_max_year)
    )
    
    # Apply filters based on status toggles
    data_frames = []
    if show_solar_operational:
        operational_solar = solar_data[solar_data['Status'] == 'Operational'].copy()
        data_frames.append(operational_solar)
    if show_solar_planning:
        planning_solar = solar_data[solar_data['Status'] == 'Planning'].copy()
        data_frames.append(planning_solar)
    
    # Combine selected data
    if data_frames:
        combined_solar = pd.concat(data_frames, ignore_index=True) if len(data_frames) > 1 else data_frames[0]
        
        # Apply additional filters
        mask = pd.Series(True, index=combined_solar.index)
        # If states are selected, filter by them; otherwise show all
        if selected_solar_states:
            mask &= combined_solar['Bundesland'].isin(selected_solar_states)
        mask &= (combined_solar['Nettonennleistung der Einheit'] >= solar_power_range[0]) & \
                (combined_solar['Nettonennleistung der Einheit'] <= solar_power_range[1])
        # For year filter, include rows where install_year is NaN (planning with no date) OR within range
        mask &= (combined_solar['install_year'].isna() | 
                 ((combined_solar['install_year'] >= selected_solar_years[0]) & 
                  (combined_solar['install_year'] <= selected_solar_years[1])))
        filtered_solar = combined_solar.loc[mask]
    else:
        filtered_solar = pd.DataFrame()
    
    # Display statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_count = 0
        if show_solar_operational:
            total_count += len(solar_data[solar_data['Status'] == 'Operational'])
        if show_solar_planning:
            total_count += len(solar_data[solar_data['Status'] == 'Planning'])
        st.metric("Total Solar Installations", total_count)
    with col2:
        st.metric("Filtered Installations", len(filtered_solar))
    with col3:
        if not filtered_solar.empty:
            avg_power = filtered_solar['Nettonennleistung der Einheit'].mean() / 1000
            st.metric("Avg Power", f"{avg_power:.1f} MW")
        else:
            st.metric("Avg Power", "0 MW")
    with col4:
        if not filtered_solar.empty:
            total_power = filtered_solar['Nettonennleistung der Einheit'].sum() / 1000
            st.metric("Total Power", f"{total_power:,.0f} MW")
        else:
            st.metric("Total Power", "0 MW")
    
    # Create and display solar map
    if not filtered_solar.empty:
        # Create pydeck map for solar
        def create_solar_pydeck_map(data):
            view_state = pdk.ViewState(latitude=51.1657, longitude=10.4515, zoom=5.7, pitch=0)
            
            display_df = data[[
                'MaStR-Nr. der Einheit', 'Anzeige-Name der Einheit', 'Bundesland', 'Ort',
                'Nettonennleistung der Einheit', 'Art der Solaranlage',
                'install_date_str', 'Name des Anlagenbetreibers (nur Org.)',
                'Name des Solarparks', 'Anzahl der Solar-Module', 'Spannungsebene',
                'lat', 'lon', 'power_mw', 'color_rgba'
            ]].rename(columns={
                'MaStR-Nr. der Einheit': 'MaStR',
                'Anzeige-Name der Einheit': 'Name',
                'Nettonennleistung der Einheit': 'Power_kW',
                'Art der Solaranlage': 'Type',
                'install_date_str': 'InstallationDate',
                'Name des Anlagenbetreibers (nur Org.)': 'Owner',
                'Name des Solarparks': 'ParkName',
                'Anzahl der Solar-Module': 'Modules',
                'Spannungsebene': 'VoltageLevel',
            })
            
            display_df['radius_px'] = 96
            
            layer = pdk.Layer(
                "ScatterplotLayer",
                data=display_df,
                get_position='[lon, lat]',
                get_radius='radius_px',
                get_fill_color='color_rgba',
                pickable=True,
                radius_min_pixels=2,
                radius_max_pixels=80,
                stroked=False,
                auto_highlight=True,
            )
            
            tooltip = {
                "html": (
                    "<b>MaStR:</b> {MaStR}<br>"
                    "<b>Name:</b> {Name}<br>"
                    "<b>Power:</b> {power_mw} MW<br>"
                    "<b>Type:</b> {Type}<br>"
                    "<b>Installation Date:</b> {InstallationDate}<br>"
                    "<b>Owner:</b> {Owner}<br>"
                    "<b>State:</b> {Bundesland}<br>"
                    "<b>Park Name:</b> {ParkName}<br>"
                    "<b>Number of Modules:</b> {Modules}<br>"
                    "<b>Voltage Level:</b> {VoltageLevel}"
                ),
                "style": {"backgroundColor": "#2b2b2b", "color": "white"},
            }
            
            deck = pdk.Deck(layers=[layer], initial_view_state=view_state, map_style='light', tooltip=tooltip)
            return deck
        
        deck = create_solar_pydeck_map(filtered_solar)
        st.pydeck_chart(deck, use_container_width=True, height=700)
        
        # Legend
        st.markdown("### Legend")
        num_legend_cols = 5 if show_solar_planning else 4
        legend_cols = st.columns(num_legend_cols)
        with legend_cols[0]:
            st.markdown("🟡 **Gold**: < 2 MW (Operational)")
        with legend_cols[1]:
            st.markdown("🟠 **Orange**: 2-5 MW (Operational)")
        with legend_cols[2]:
            st.markdown("🔴 **Red-Orange**: 5-10 MW (Operational)")
        with legend_cols[3]:
            st.markdown("🔴 **Dark Red**: > 10 MW (Operational)")
        if show_solar_planning:
            with legend_cols[4]:
                st.markdown("🟢 **Green**: In Planning")
        
        # Data table
        if st.checkbox("Show filtered solar data table"):
            st.subheader("Filtered Solar Installation Data")
            st.dataframe(
                filtered_solar[[
                    'MaStR-Nr. der Einheit',
                    'Anzeige-Name der Einheit',
                    'Bundesland',
                    'Ort',
                    'Nettonennleistung der Einheit',
                    'Art der Solaranlage',
                    'Inbetriebnahmedatum der Einheit',
                    'Name des Anlagenbetreibers (nur Org.)',
                    'Name des Solarparks',
                    'Anzahl der Solar-Module',
                ]].rename(columns={
                    'MaStR-Nr. der Einheit': 'MaStR Number',
                    'Anzeige-Name der Einheit': 'Name',
                    'Bundesland': 'State',
                    'Ort': 'City',
                    'Nettonennleistung der Einheit': 'Power (kW)',
                    'Art der Solaranlage': 'Type',
                    'Inbetriebnahmedatum der Einheit': 'Installation Date',
                    'Name des Anlagenbetreibers (nur Org.)': 'Owner',
                    'Name des Solarparks': 'Solar Park Name',
                    'Anzahl der Solar-Module': 'Number of Modules',
                }),
                width='stretch'
            )
    else:
        if not show_solar_operational and not show_solar_planning:
            st.warning("⚠️ Please enable at least one solar type (Operational or Planning) to display data on the map.")
        else:
            st.warning("No solar installations match the current filter criteria. Please adjust the filters.")
    
    # Footer
    st.markdown("---")
    st.markdown("Data source: German renewable energy registry (MaStR)")

# ======================== TAB 3: STATISTICS ========================
with tab3:
    # Statistics settings header
    st.sidebar.markdown("## 📊 Statistics Settings")
    
    # Get unique bundeslands from both wind and solar data
    wind_bundeslands = set(wind_data['Bundesland'].dropna().unique())
    wind_bundeslands.update(planning_data['Bundesland'].dropna().unique())
    solar_bundeslands = set(solar_data['Bundesland'].dropna().unique())
    all_bundeslands = sorted(wind_bundeslands.union(solar_bundeslands))
    
    # Bundesland selector
    selected_bundesland = st.sidebar.selectbox(
        "Select Bundesland",
        options=all_bundeslands,
        index=0 if all_bundeslands else None
    )
    
    # Year range selector
    all_wind_years = pd.concat([wind_data['install_year'], planning_data['install_year']]).dropna()
    all_solar_years = solar_data['install_year'].dropna()
    
    min_year = int(min(all_wind_years.min(), all_solar_years.min()))
    max_year = 2030
    
    year_range = st.sidebar.slider(
        "Year Range",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year)
    )
    
    # Status filters
    st.sidebar.markdown("### Wind Turbines")
    include_wind_operational = st.sidebar.checkbox("Include operational wind", value=True)
    include_wind_planning = st.sidebar.checkbox("Include planned wind", value=True)
    
    st.sidebar.markdown("### Solar Installations")
    include_solar_operational = st.sidebar.checkbox("Include operational solar", value=True)
    include_solar_planning = st.sidebar.checkbox("Include planned solar", value=True)
    
    # Filter data based on selections
    # Wind data filtering
    wind_frames = []
    if include_wind_operational:
        wind_op = wind_data[wind_data['Bundesland'] == selected_bundesland].copy()
        wind_frames.append(wind_op)
    if include_wind_planning:
        wind_plan = planning_data[planning_data['Bundesland'] == selected_bundesland].copy()
        wind_frames.append(wind_plan)
    
    if wind_frames:
        filtered_wind_stats = pd.concat(wind_frames, ignore_index=True)
        # Apply year filter
        filtered_wind_stats = filtered_wind_stats[
            (filtered_wind_stats['install_year'] >= year_range[0]) &
            (filtered_wind_stats['install_year'] <= year_range[1])
        ]
    else:
        filtered_wind_stats = pd.DataFrame()
    
    # Solar data filtering
    solar_frames = []
    if include_solar_operational:
        solar_op = solar_data[
            (solar_data['Bundesland'] == selected_bundesland) & 
            (solar_data['Status'] == 'Operational')
        ].copy()
        solar_frames.append(solar_op)
    if include_solar_planning:
        solar_plan = solar_data[
            (solar_data['Bundesland'] == selected_bundesland) & 
            (solar_data['Status'] == 'Planning')
        ].copy()
        solar_frames.append(solar_plan)
    
    if solar_frames:
        filtered_solar_stats = pd.concat(solar_frames, ignore_index=True)
        # Apply year filter (include NaN for planning without dates)
        filtered_solar_stats = filtered_solar_stats[
            (filtered_solar_stats['install_year'].isna()) |
            ((filtered_solar_stats['install_year'] >= year_range[0]) &
             (filtered_solar_stats['install_year'] <= year_range[1]))
        ]
    else:
        filtered_solar_stats = pd.DataFrame()
    
    # Display header
    st.header(f"Statistics for {selected_bundesland}")
    st.markdown(f"**Year Range:** {year_range[0]} - {year_range[1]}")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        wind_count = len(filtered_wind_stats) if not filtered_wind_stats.empty else 0
        st.metric("Wind Turbines", f"{wind_count:,}")
    
    with col2:
        wind_capacity = (filtered_wind_stats['Nettonennleistung der Einheit'].sum() / 1000 
                        if not filtered_wind_stats.empty else 0)
        st.metric("Wind Capacity", f"{wind_capacity:,.0f} MW")
    
    with col3:
        solar_count = len(filtered_solar_stats) if not filtered_solar_stats.empty else 0
        st.metric("Solar Installations", f"{solar_count:,}")
    
    with col4:
        solar_capacity = (filtered_solar_stats['Nettonennleistung der Einheit'].sum() / 1000 
                         if not filtered_solar_stats.empty else 0)
        st.metric("Solar Capacity", f"{solar_capacity:,.0f} MW")
    
    st.markdown("---")
    
    # Check if we have data to display
    if not filtered_wind_stats.empty or not filtered_solar_stats.empty:
        # Capacity installation charts
        st.subheader("Capacity Installation Over Time")
        
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.markdown("#### Annual Installed Capacity")
            
            # Prepare annual capacity data
            annual_data = []
            
            if not filtered_wind_stats.empty:
                wind_annual = (filtered_wind_stats.groupby('install_year')['Nettonennleistung der Einheit'].sum() / 1000)
                for year, capacity in wind_annual.items():
                    annual_data.append({'Year': int(year), 'Type': 'Wind', 'Capacity (MW)': capacity})
            
            if not filtered_solar_stats.empty:
                solar_with_year = filtered_solar_stats[filtered_solar_stats['install_year'].notna()]
                if not solar_with_year.empty:
                    solar_annual = (solar_with_year.groupby('install_year')['Nettonennleistung der Einheit'].sum() / 1000)
                    for year, capacity in solar_annual.items():
                        annual_data.append({'Year': int(year), 'Type': 'Solar', 'Capacity (MW)': capacity})
            
            if annual_data:
                import plotly.express as px
                annual_df = pd.DataFrame(annual_data)
                fig_annual = px.bar(
                    annual_df, 
                    x='Year', 
                    y='Capacity (MW)', 
                    color='Type',
                    title='Annual Capacity Installation',
                    barmode='group',
                    color_discrete_map={'Wind': '#1f77b4', 'Solar': '#ff7f0e'}
                )
                st.plotly_chart(fig_annual, use_container_width=True)
            else:
                st.info("No data available for annual capacity chart")
        
        with chart_col2:
            st.markdown("#### Cumulative Installed Capacity")
            
            # Prepare cumulative capacity data
            cumulative_data = []
            
            if not filtered_wind_stats.empty:
                wind_annual = (filtered_wind_stats.groupby('install_year')['Nettonennleistung der Einheit'].sum() / 1000)
                wind_cumulative = wind_annual.sort_index().cumsum()
                for year, capacity in wind_cumulative.items():
                    cumulative_data.append({'Year': int(year), 'Type': 'Wind', 'Capacity (MW)': capacity})
            
            if not filtered_solar_stats.empty:
                solar_with_year = filtered_solar_stats[filtered_solar_stats['install_year'].notna()]
                if not solar_with_year.empty:
                    solar_annual = (solar_with_year.groupby('install_year')['Nettonennleistung der Einheit'].sum() / 1000)
                    solar_cumulative = solar_annual.sort_index().cumsum()
                    for year, capacity in solar_cumulative.items():
                        cumulative_data.append({'Year': int(year), 'Type': 'Solar', 'Capacity (MW)': capacity})
            
            if cumulative_data:
                cumulative_df = pd.DataFrame(cumulative_data)
                fig_cumulative = px.line(
                    cumulative_df, 
                    x='Year', 
                    y='Capacity (MW)', 
                    color='Type',
                    title='Cumulative Capacity Installation',
                    color_discrete_map={'Wind': '#1f77b4', 'Solar': '#ff7f0e'}
                )
                st.plotly_chart(fig_cumulative, use_container_width=True)
            else:
                st.info("No data available for cumulative capacity chart")
        
        st.markdown("---")
        
        # Voltage level distribution
        st.subheader("Voltage Level Distribution")
        
        pie_col1, pie_col2 = st.columns(2)
        
        with pie_col1:
            st.markdown("#### Wind Turbines by Voltage Level")
            if not filtered_wind_stats.empty and 'Spannungsebene' in filtered_wind_stats.columns:
                wind_voltage = filtered_wind_stats['Spannungsebene'].value_counts()
                if not wind_voltage.empty:
                    fig_wind_voltage = px.pie(
                        values=wind_voltage.values,
                        names=wind_voltage.index,
                        title='Wind Turbine Connections',
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    st.plotly_chart(fig_wind_voltage, use_container_width=True)
                else:
                    st.info("No voltage level data available")
            else:
                st.info("No voltage level data available")
        
        with pie_col2:
            st.markdown("#### Solar Installations by Voltage Level")
            if not filtered_solar_stats.empty and 'Spannungsebene' in filtered_solar_stats.columns:
                solar_voltage = filtered_solar_stats['Spannungsebene'].value_counts()
                if not solar_voltage.empty:
                    fig_solar_voltage = px.pie(
                        values=solar_voltage.values,
                        names=solar_voltage.index,
                        title='Solar Installation Connections',
                        color_discrete_sequence=px.colors.qualitative.Pastel
                    )
                    st.plotly_chart(fig_solar_voltage, use_container_width=True)
                else:
                    st.info("No voltage level data available")
            else:
                st.info("No voltage level data available")
    else:
        st.warning("⚠️ No data available for the selected filters. Please adjust your selection.")
    
    # Footer
    st.markdown("---")
    st.markdown("Data source: German renewable energy registry (MaStR)")
