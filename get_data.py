import requests
import pandas as pd

apikey = '73f7585c1934e4530c64305b2f19c3ad'
lat = '49.150002'
lon = '9.216600'
url = 'http://api.openweathermap.org/data/2.5/air_pollution/forecast?lat='+lat+'&lon='+lon+'&appid='+apikey

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
