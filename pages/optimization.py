from dash import dcc, html, dash
import folium
import dash
import dash_html_components as html
from dash import html, dcc, callback, Input, Output, State
import folium
from sklearn.cluster import DBSCAN
from folium.plugins import MarkerCluster
import pandas as pd
import numpy as np


# Register the page for optimization
dash.register_page(__name__, path='/optimization')

# Load the dataset
data = pd.read_csv("data/interventions_bxl.parquet.csv").drop_duplicates()

# For Permanence Locations
#data['latitude_permanence']= data['latitude_permanence'].apply(lambda x: float(str(x)[:2] + '.' + str(x)[2:]))
#data['longitude_permanence']= data['longitude_permanence'].apply(lambda x: float(str(x)[:1] + '.' + str(x)[1:]))

# For Intervention Locations
#data['latitude_intervention']= data['latitude_intervention'].astype(int).apply(lambda x: float(str(x)[:2] + '.' + str(x)[2:]))
#data['longitude_intervention']= data['longitude_intervention'].astype(int).apply(lambda x: float(str(x)[:1] + '.' + str(x)[1:]))


# Filter the dataset to only include relevant columns
relevant_data = data[['latitude_permanence', 'longitude_permanence', 
                      'latitude_intervention', 'longitude_intervention', 'cityname_intervention', 
                      'vector_type', 'waiting_time']].dropna()


aed_bxl = pd.read_csv('data/aed_bxl.parquet.csv')


# Create the dropdown menu options
dropdown_options = [{'label': city_name, 'value': city_name} for city_name in relevant_data['cityname_intervention'].unique()]

# Define the styles for the dropdown menu options
option_style = {
    'padding': '8px',
    'font-size': '9px',
    'margin-bottom': '5px'  # Add margin-bottom to create space between options
}

# Define the layout
layout = html.Div([
    html.H3('Optimization', style={"font-size": "34px", "text-align": "center", "color": "#dc3545", "fontWeight": "bold"}),
    html.Div([
        dcc.Dropdown(
            id='cityname-dropdown',
            options=dropdown_options,
            placeholder="Select a City",
            style={"width": "300px", "margin-right": "10px"},
        ),

        dcc.Input(
            id='input-number',
            type='number',
            placeholder='Budget in USD',  # Default value for the input area
            style={"width": "200px", "margin-right": "10px"}
        ),
        html.Button('Okay', id='okay-button', n_clicks=0, style={"margin-top": "8px"})
    ], style={"display": "flex", "align-items": "center", "justify-content": "center"}),


    html.Br(),  # Add a line break for spacing
    html.Div(id='output-container', style={"margin-left": "10px"}),
        
    html.Br(),  # Add a line break for spacing
    html.Button('Generate Map', id='generate-map-button', n_clicks=0, style={"margin-left": "10px"}),
    html.Div(id='map-container')
])

# Define callback to update output
@callback(
    Output('output-container', 'children'),
    [Input('okay-button', 'n_clicks')],
    [State('input-number', 'value')]
)
def update_output(n_clicks, value):
    if n_clicks > 0 and value is not None:
        result = value / 1600.00
        result = round(result)
        return "Number of AED Devices: " + str(result)
    else:
        return ''
    
# Callback to generate the map
@callback(
    Output('map-container', 'children'),
    [Input('generate-map-button', 'n_clicks')],
    [Input('cityname-dropdown', 'value')],
    [Input('input-number', 'value')]
)
def generate_map(n_clicks, cityname, input_number):
    if n_clicks > 0 and cityname is not None and input_number is not None:
        # Filter data based on selected city name
        filtered_data = data[data['cityname_intervention'] == cityname]

        # Concatenate permanence and intervention locations horizontally
        locations = pd.concat([filtered_data[['latitude_permanence', 'longitude_permanence']],
                               filtered_data[['latitude_intervention', 'longitude_intervention']]], axis=1)

        # Convert DataFrame to numpy array
        X = locations.values

        # Apply DBSCAN clustering algorithm
        dbscan = DBSCAN(eps=0.0001, min_samples=10, metric='euclidean').fit(X)

        # Get the cluster labels
        labels = dbscan.labels_

        # Find the optimal AED locations as the centroids of the clusters with the lowest waiting time
        unique_labels = np.unique(labels)
        aed_locations = []
        for label in unique_labels:
            if label != -1:  # Ignore noise points
                cluster_points = X[labels == label]
                cluster_waiting_times = filtered_data.iloc[labels == label]['waiting_time']
                weighted_waiting_time = np.sum(cluster_waiting_times) / len(cluster_waiting_times)
                centroid = np.mean(cluster_points, axis=0)
                aed_locations.append((centroid, weighted_waiting_time))

        # Sort AED locations based on weighted waiting time
        aed_locations.sort(key=lambda x: x[1])

        # Center coordinates for the selected city
        city_center = filtered_data[['latitude_intervention', 'longitude_intervention']].mean()

        # Calculate the number of AED devices
        num_aed_devices = round(input_number / 1600)

        # Initialize the map
        m = folium.Map(location=(city_center['latitude_intervention'], city_center['longitude_intervention']), zoom_start= 12)

        # Add markers for AED locations with custom icon
        locations_added = 0  # Track the number of unique locations added
        for locations, _ in aed_locations:
            for i in range(0, len(locations), 2):
                lat, lon = locations[i], locations[i + 1]
                folium.Marker(location=(lat, lon), popup=f"AED Location {locations_added + 1}",
                              icon=folium.CustomIcon('assets/Aed_logo.jpg', icon_size=(32, 32))).add_to(m)
                locations_added += 1
                if locations_added == num_aed_devices:  # Stop after adding the specified number of unique locations
                    break
            if locations_added == num_aed_devices:  # Stop outer loop after adding the specified number of unique locations
                break

        # Convert Folium map to HTML
        map_html = m.get_root().render()

        # Return an iframe to display the map
        return html.Iframe(srcDoc=map_html, width='100%', height='500')
    else:
        return ''
