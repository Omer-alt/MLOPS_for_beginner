import requests
import json
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

# Import the necessary module
from dotenv import load_dotenv
import os

# Load environment variables from the .env file (if present)
load_dotenv()

# Define default arguments
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 9, 18),
    'retries': 1,
}

# OpenWeather API configuration
# API_KEY = os.getenv('API_KEY') 
API_KEY = "C8NFW8L3V7Y5HDL9N4RBZB8TV"
MY_API_KEY = "RB628E259LEZGHJCGWL42K3VP"
 
CITY = "London"

# Function to fetch weather data from OpenWeather API
def fetch_weather_data(ti):
    print("Hello word")
    # url = f"http://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={API_KEY}"
    # url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/London%2CUK?unitGroup=us&key=C8NFW8L3V7Y5HDL9N4RBZB8TV"
    # response = requests.get(url)
    # data = response.json()
    # print("Data", data)
    # # Extract temperature in Kelvin
    # temperature_kelvin = data['currentConditions']['temp']
    # print(f"Fetched temperature in Kelvin: {temperature_kelvin}")
    temperature_kelvin = 300.6
    # Push the temperature to XCom
    ti.xcom_push(key='temperature_kelvin', value=temperature_kelvin)

# Function to convert temperature from Kelvin to Celsius
def convert_temperature(ti):
    temperature_kelvin = ti.xcom_pull(task_ids='fetch_weather_data', key='temperature_kelvin')
    
    # Convert from Kelvin to Celsius
    temperature_celsius = temperature_kelvin - 273.15
    print(f"Converted temperature to Celsius: {temperature_celsius}")
    
    # Push the Celsius temperature to XCom
    ti.xcom_push(key='temperature_celsius', value=temperature_celsius)

# Function to save temperature data to a local file
def save_temperature_to_file(ti):
    temperature_celsius = ti.xcom_pull(task_ids='convert_temperature', key='temperature_celsius')
    
    # Save the temperature to a local file
    with open('/temperature_data.json', 'w') as f:
        json.dump({'city': CITY, 'temperature_celsius': temperature_celsius}, f)
    
    print(f"Saved temperature data to /temperature_data.json")

# Define the DAG
with DAG(
    dag_id='weather_data_pipeline',
    default_args=default_args,
    description='Fetch, process, and save weather data using OpenWeather API',
    schedule_interval='*/5 * * * *',
    catchup=False,
) as dag:

    # Task 1: Fetch weather data
    fetch_weather_task = PythonOperator(
        task_id='fetch_weather_data',
        python_callable=fetch_weather_data
    )

    # Task 2: Convert temperature from Kelvin to Celsius
    convert_temperature_task = PythonOperator(
        task_id='convert_temperature',
        python_callable=convert_temperature
    )

    # Task 3: Save temperature data to a local file
    save_temperature_task = PythonOperator(
        task_id='save_temperature_to_file',
        python_callable=save_temperature_to_file
    )

    # Set task dependencies
    fetch_weather_task >> convert_temperature_task >> save_temperature_task
