import os
import pandas as pd

# Define input and output directories
input_folder = 'Data2024'
output_folder = 'data'  

# Ensure output folder exists, if not create it
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Function to convert parquet to csv
def parquet_to_csv(file_path, output_folder):
    # Read parquet file
    df = pd.read_parquet(file_path)
    
    # Define output file path for CSV
    output_file = os.path.join(output_folder, os.path.splitext(os.path.basename(file_path))[0] + '.csv')
    
    # Write dataframe to CSV
    df.to_csv(output_file, index=False)

# Iterate through files in input folder
for file_name in os.listdir(input_folder):
    if file_name.endswith('.parquet.gzip') or file_name.endswith('.parquet'):
        file_path = os.path.join(input_folder, file_name)
        parquet_to_csv(file_path, output_folder)

print("Conversion complete.")
