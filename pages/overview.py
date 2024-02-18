from dash import dcc, html, dash
import folium
import dash
import dash_html_components as html
import folium
from folium.plugins import MarkerCluster
import pandas as pd

dash.register_page(__name__, path='/')

# Load the data
aed_bxl = pd.read_csv('data/aed_bxl.parquet.csv').drop_duplicates()
mug_bxl = pd.read_csv('data/mug_bxl.parquet.csv').drop_duplicates()
interventions_bxl_map = pd.read_csv('data/interventions_bxl_map.parquet.csv').drop_duplicates()

# Drop rows with NaN values in latitude and longitude columns
aed_bxl = aed_bxl.dropna(subset=['latitude', 'longitude'])
mug_bxl = mug_bxl.dropna(subset=['latitude', 'longitude'])
interventions_bxl_map = interventions_bxl_map.dropna(subset=['latitude_permanence', 'longitude_permanence']).sample(2000)


# Define different icons for each type of location
icons = ['images/Aed_logo.jpg', 'images/mug_logo.png', 'images/Ambulance_logo.png', 'images/pit_logo.png', 'images/firetruck.png', 'images/decontamination.png']

# Create Folium map
mymap = folium.Map(location=[50.8503, 4.3517], zoom_start=12) # Centered around BXL

# Add markers to the map for each location type
marker_cluster = MarkerCluster().add_to(mymap)

# Add markers for AED
for index, row in aed_bxl.iterrows():
    icon = folium.CustomIcon(icons[0], icon_size=(32, 32))
    folium.Marker([row['latitude'], row['longitude']], icon=icon).add_to(marker_cluster)

# Add markers for MUG
for index, row in mug_bxl.iterrows():
    icon = folium.CustomIcon(icons[1], icon_size=(32, 32))
    folium.Marker([row['latitude'], row['longitude']], icon=icon).add_to(marker_cluster)

# Add markers for Ambulance
for index, row in interventions_bxl_map[interventions_bxl_map['vector_type']=='Ambulance'].iterrows():
    icon = folium.CustomIcon(icons[2], icon_size=(32, 32))
    folium.Marker([row['latitude_permanence'], row['longitude_permanence']], icon=icon).add_to(marker_cluster)

# Add markers for PIT
for index, row in interventions_bxl_map[interventions_bxl_map['vector_type']=='PIT'].iterrows():
    icon = folium.CustomIcon(icons[3], icon_size=(32, 32))
    folium.Marker([row['latitude_permanence'], row['longitude_permanence']], icon=icon).add_to(marker_cluster)

# Add markers for Brandziekenwagen
for index, row in interventions_bxl_map[interventions_bxl_map['vector_type']=='Brandziekenwagen'].iterrows():
    icon = folium.CustomIcon(icons[3], icon_size=(32, 32))
    folium.Marker([row['latitude_permanence'], row['longitude_permanence']], icon=icon).add_to(marker_cluster)

# Add markers for Decontanimatieziekenwagen
for index, row in interventions_bxl_map[interventions_bxl_map['vector_type']=='Decontanimatieziekenwagen'].iterrows():
    icon = folium.CustomIcon(icons[3], icon_size=(32, 32))
    folium.Marker([row['latitude_permanence'], row['longitude_permanence']], icon=icon).add_to(marker_cluster)



# Convert Folium map to HTML
map_html = mymap.get_root().render()

# Define the layout of the overview page
layout = html.Div([
    html.H1('Overview'),
    html.Iframe(srcDoc=map_html, width='100%', height='600')
])