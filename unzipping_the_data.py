import pandas as pd
import zipfile
import os

# Extract the zip file
zip_file_path = 'Data2024.zip'
extract_folder = 'data'

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_folder)

# Read .parquet.gzip files and convert them to CSV
parquet_folder = extract_folder

output_folder = 'data'
os.makedirs(output_folder, exist_ok=True)

for file_name in os.listdir(parquet_folder):
    if file_name.endswith('.parquet.gzip'):
        parquet_file_path = os.path.join(parquet_folder, file_name)
        df = pd.read_parquet(parquet_file_path)
        csv_file_path = os.path.join(output_folder, os.path.splitext(file_name)[0] + '.csv')

        # Remove the .parquet extension from the file name
        csv_file_name = file_name.replace('.parquet.gzip', '') + '.csv'
        csv_file_path = os.path.join(output_folder, csv_file_name)
        df.to_csv(csv_file_path, index=False)

        # Remove the .parquet.gzip files
        os.remove(parquet_file_path)
print("Conversion completed. CSV files are saved in the 'data' folder.")