from dash import dcc, html, dash
from dash.dependencies import Input, Output
from dash import callback
import folium
import dash
import dash_html_components as html
import dash_bootstrap_components as dbc
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
icons = ['assets/Aed_logo.jpg', 'assets/mug_logo.png', 'assets/Ambulance_logo.png', 'assets/pit_logo.png', 'assets/firetruck.png', 'assets/decontamination.png']

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

# Add markers for Decontaminatieziekenwagen
for index, row in interventions_bxl_map[interventions_bxl_map['vector_type']=='Decontaminatieziekenwagen'].iterrows():
    icon = folium.CustomIcon(icons[3], icon_size=(32, 32))
    folium.Marker([row['latitude_permanence'], row['longitude_permanence']], icon=icon).add_to(marker_cluster)

# Convert Folium map to HTML
map_html = mymap.get_root().render()

# Card with overall info 
card_main = dbc.Card(
    [        dbc.CardBody(
            [
                html.H2("What are AED's?", className="card-title",style={"color": "#dc3545", "paddingTop": "50px", "textAlign": "left","fontSize": "32px"}),
                html.P(
                    "Automated External Defibrillators (AEDs) are portable devices crucial for sudden cardiac arrest (SCA) emergencies. They're user-friendly, often featuring voice prompts and visual aids for easy operation by individuals without medical training. Quick access to AEDs significantly increases the likelihood of survival, as SCA survival rates decrease by 10% every minute without defibrillation. Many countries mandate AED placement in public spaces like airports and schools. Modern AEDs may include smart technology for remote monitoring and maintenance. They often have pediatric capabilities, adjusting shock energy for children. Legal protections, commonly known as Good Samaritan laws, encourage bystander intervention by shielding users from liability when acting in good faith. Continuous advancements aim to improve AED performance, usability, and accessibility, underscoring their vital role in saving lives during cardiac emergencies.",
                    className="card-text",
                      style={
                        "paddingTop": "20px",  # Add spacing at the top
                        "paddingBottom": "50px",  # Add spacing at the bottom
                        "textAlign": "left",
                        "margin": "0",
                        "fontSize": "18px"  # Increase font size
                    }
                ),
               
            ]
        ),
    ],

      outline=False
)



# Define the layout of the overview page
layout = html.Div([
    html.H1('Overview'),
    dbc.Row([
        dbc.Col(
            dcc.Dropdown(
                id='location-type-dropdown',
                options=[
                    {'label': 'AED', 'value': 'AED'},
                    {'label': 'MUG', 'value': 'MUG'},
                    {'label': 'Ambulance', 'value': 'Ambulance'},
                    {'label': 'PIT', 'value': 'PIT'},
                    {'label': 'Brandziekenwagen', 'value': 'Brandziekenwagen'},
                    {'label': 'Decontaminatieziekenwagen', 'value': 'Decontaminatieziekenwagen'}
                ],
                value='AED',
                clearable=False
            ),
            width=3
        ),
        dbc.Col(
            html.Iframe(id='map', srcDoc=map_html, width='100%', height='600'),
            width=9
        )
    ]),
    card_main
])

# Define callback to update the map based on dropdown selection
@callback(
    Output('map', 'srcDoc'),
    [Input('location-type-dropdown', 'value')]
)

def update_map(location_type):
    mymap = folium.Map(location=[50.8503, 4.3517], zoom_start=12) # Centered around BXL
    marker_cluster = MarkerCluster().add_to(mymap)
    
    if location_type == 'AED':
        for index, row in aed_bxl.iterrows():
            icon = folium.CustomIcon(icons[0], icon_size=(32, 32))
            folium.Marker([row['latitude'], row['longitude']], icon=icon).add_to(marker_cluster)
    elif location_type == 'MUG':
        for index, row in mug_bxl.iterrows():
            icon = folium.CustomIcon(icons[1], icon_size=(32, 32))
            folium.Marker([row['latitude'], row['longitude']], icon=icon).add_to(marker_cluster)
    elif location_type == 'Ambulance':
        for index, row in interventions_bxl_map[interventions_bxl_map['vector_type']=='Ambulance'].iterrows():
            icon = folium.CustomIcon(icons[2], icon_size=(32, 32))
            folium.Marker([row['latitude_permanence'], row['longitude_permanence']], icon=icon).add_to(marker_cluster)
    elif location_type == 'PIT':
        for index, row in interventions_bxl_map[interventions_bxl_map['vector_type']=='PIT'].iterrows():
            icon = folium.CustomIcon(icons[3], icon_size=(32, 32))
            folium.Marker([row['latitude_permanence'], row['longitude_permanence']], icon=icon).add_to(marker_cluster)
    elif location_type == 'Brandziekenwagen':
        for index, row in interventions_bxl_map[interventions_bxl_map['vector_type']=='Brandziekenwagen'].iterrows():
            icon = folium.CustomIcon(icons[3], icon_size=(32, 32))
            folium.Marker([row['latitude_permanence'], row['longitude_permanence']], icon=icon).add_to(marker_cluster)
    elif location_type == 'Decontaminatieziekenwagen':
        for index, row in interventions_bxl_map[interventions_bxl_map['vector_type']=='Decontaminatieziekenwagen'].iterrows():
            icon = folium.CustomIcon(icons[3], icon_size=(32, 32))
            folium.Marker([row['latitude_permanence'], row['longitude_permanence']], icon=icon).add_to(marker_cluster)
    
    return mymap.get_root().render()