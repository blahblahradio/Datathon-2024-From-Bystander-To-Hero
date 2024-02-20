import pandas as pd
from geopy.geocoders import Nominatim

# Load the data
aed_locations = pd.read_csv('data/aed_locations.parquet.csv')
mug_locations = pd.read_csv('data/mug_locations.parquet.csv')
interventions_bxl = pd.read_csv('data/interventions_bxl.parquet.csv')

## Preprocessing the data

# Subsetting for BXL area
aed_bxl = aed_locations[aed_locations['province'] == 'Bruxelles-Brussel']
mug_bxl = mug_locations[mug_locations['province'] == 'Brussels Hoofdstedelijk Gewest']

# Managing the missing values in AED locations
aed_bxl['number'].fillna(aed_bxl['number'].mode()[0], inplace=True)
aed_bxl['municipality'].fillna('Brussel', inplace=True)

# Dropping the missing values for Permanence and Intervention Locations
interventions_bxl.dropna(subset=['latitude_permanence','longitude_permanence'], inplace=True)
interventions_bxl.dropna(subset=['latitude_intervention','longitude_intervention'], inplace=True)

# Create a geocoder
geolocator = Nominatim(user_agent="my_geocoder")

# Setting Latitude and Longitudes

# Create empty columns for latitude and longitude
aed_bxl['latitude'] = None
aed_bxl['longitude'] = None
mug_bxl['latitude'] = None
mug_bxl['longitude'] = None

# For AED Locations
for index, row in aed_bxl.iterrows():
    address = row["address"]
    number = row["number"]
    municipality = row["municipality"]
    province = row["province"]
    location = geolocator.geocode(f"{address} {number}, {municipality}, {province}", timeout=10)
    
    # Check if location is found
    if location:
        # Add latitude and longitude to the DataFrame
        aed_bxl.at[index, 'latitude'] = location.latitude
        aed_bxl.at[index, 'longitude'] = location.longitude


# For MUG Locations
for index, row in mug_bxl.iterrows():
    address = row['address_campus']
    postal_code = row['postal_code']
    municipality = row['municipality']
    province = row['province']
    location = geolocator.geocode(f"{address} {postal_code}, {municipality}, {province}", timeout=10)
    if location:
        # Add latitude and longitude to the DataFrame
        mug_bxl.at[index, 'latitude'] = location.latitude
        mug_bxl.at[index, 'longitude'] = location.longitude

# For Permanence Locations
interventions_bxl['latitude_permanence']= interventions_bxl['latitude_permanence'].apply(lambda x: float(str(x)[:2] + '.' + str(x)[2:]))
interventions_bxl['longitude_permanence']= interventions_bxl['longitude_permanence'].apply(lambda x: float(str(x)[:1] + '.' + str(x)[1:]))

# For Intervention Locations
interventions_bxl['latitude_intervention']= interventions_bxl['latitude_intervention'].astype(int).apply(lambda x: float(str(x)[:2] + '.' + str(x)[2:]))
interventions_bxl['longitude_intervention']= interventions_bxl['longitude_intervention'].astype(int).apply(lambda x: float(str(x)[:1] + '.' + str(x)[1:]))


#Remove parenthesis city_intervention
def remove_text_within_parentheses(text):
    if text is None or isinstance(text, float):  # Check if text is None or float
        return None
    return text.split('(')[0].strip().rstrip()  # Remove any additional trailing whitespace

# Apply the function to the column
interventions_bxl['cityname_intervention'] = interventions_bxl['cityname_intervention'].apply(remove_text_within_parentheses)

#French-Dutch municipality name
def replace_french_with_dutch(city_name):
    city_dict = {
        'Anderlecht': 'Anderlecht',
        'Auderghem': 'Oudergem',
        'Berchem-Sainte-Agathe': 'Sint-Agatha-Berchem',
        'Bruxelles': 'Brussel',
        'Etterbeek': 'Etterbeek',
        'Evere': 'Evere',
        'Forest': 'Vorst',
        'Ganshoren': 'Ganshoren',
        'Ixelles': 'Elsene',
        'Jette': 'Jette',
        'Koekelberg': 'Koekelberg',
        'Molenbeek-Saint-Jean': 'Sint-Jans-Molenbeek',
        'Saint-Gilles': 'Sint-Gillis',
        'Saint-Josse-ten-Noode': 'Sint-Joost-ten-Node',
        'Schaerbeek': 'Schaarbeek',
        'Uccle': 'Ukkel',
        'Watermael-Boitsfort': 'Watermaal-Bosvoorde',
        'Woluwe-Saint-Lambert': 'Sint-Lambrechts-Woluwe',
        'Woluwe-Saint-Pierre': 'Sint-Pieters-Woluwe'
    }
    return city_dict.get(city_name, city_name)  # Return the Dutch name if found, otherwise return the original name

# Function to determine if the city name is in French
def is_french(city_name):
    french_names = set([
        'Auderghem', 'Berchem-Sainte-Agathe', 'Bruxelles',
        'Etterbeek', 'Evere', 'Forest', 'Ganshoren', 'Ixelles', 'Jette',
        'Koekelberg', 'Molenbeek-Saint-Jean', 'Saint-Gilles', 'Saint-Josse-ten-Noode',
        'Schaerbeek', 'Uccle', 'Watermael-Boitsfort', 'Woluwe-Saint-Lambert',
        'Woluwe-Saint-Pierre'
    ])
    return city_name in french_names

# Apply the function conditionally
interventions_bxl['cityname_intervention'] = interventions_bxl['cityname_intervention'].apply(lambda x: replace_french_with_dutch(x) if is_french(x) else x)


# Dropping the duplicates
aed_bxl = aed_bxl.drop_duplicates()
mug_bxl = mug_bxl.drop_duplicates()
interventions_bxl_map = interventions_bxl[['latitude_intervention','longitude_intervention','latitude_permanence', 'longitude_permanence', 'vector_type']].drop_duplicates()


# Save the modified dataframes to CSV files
aed_bxl.to_csv('data/aed_bxl.parquet.csv', index=False)
mug_bxl.to_csv('data/mug_bxl.parquet.csv', index=False)
interventions_bxl_map.to_csv('data/interventions_bxl_map.parquet.csv', index=False)
interventions_bxl.to_csv('data/interventions_bxl.parquet.csv', index=False)