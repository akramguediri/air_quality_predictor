import requests
import pandas as pd
import datetime

# Get the current timestamp (current date and time)
current = datetime.datetime.now()

# Calculate the timestamp for one year ago
one_year = current - datetime.timedelta(days=365)

# Print the timestamp for one year ago
print(one_year.timestamp())


apikey = '73f7585c1934e4530c64305b2f19c3ad'
lat = '49.150002'
lon = '9.216600'
url = 'http://api.openweathermap.org/data/2.5/air_pollution/history?lat='+lat+'&lon='+lon+'&start='+str(int(one_year.timestamp()))+'&end='+str(int(current.timestamp()))+'&appid='+apikey
response = requests.get(url)
data = response.json()  # Use response.json() to parse the JSON data

# Extract the relevant data from the JSON response
data_to_convert = []
for forecast in data['list']:
    data_point = {
        'timestamp': forecast['dt'],
        'main_aqi': forecast['main']['aqi'],
        'co': forecast['components']['co'],
        'no': forecast['components']['no'],
        'no2': forecast['components']['no2'],
        'o3': forecast['components']['o3'],
        'so2': forecast['components']['so2'],
        'pm2_5': forecast['components']['pm2_5'],
        'pm10': forecast['components']['pm10']
    }
    data_to_convert.append(data_point)

# Create a DataFrame from the extracted data
df = pd.DataFrame(data_to_convert)

# Specify the name of the CSV file
csv_file = 'data.csv'
df.to_csv(csv_file, index=False)
import pandas as pd

def clean_and_save_data(input_file, output_file):
    # Load the CSV data into a DataFrame
    df = pd.read_csv(input_file)

    # Check for missing or corrupted data
    # You can define your own criteria for identifying corrupted data
    # For example, you can check for missing values, NaNs, or outliers
    corrupted_rows = df.isnull().any(axis=1)

    # Remove corrupted rows
    cleaned_df = df[~corrupted_rows]

    # Save the cleaned data to a new CSV file
    cleaned_df.to_csv(output_file, index=False)

    # Print some statistics about the cleaning process
    num_removed = len(df) - len(cleaned_df)
    print(f"Removed {num_removed} corrupted rows.")
    print(f"Cleaned data saved to {output_file}.")

# Example usage:
input_file = 'data.csv'  # Replace with your input file path
output_file = 'data.csv'  # Replace with your desired output file path
clean_and_save_data(input_file, output_file)
