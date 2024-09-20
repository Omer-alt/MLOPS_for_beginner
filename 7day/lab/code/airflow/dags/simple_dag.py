from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import time

# Define default arguments
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 9, 18),  # Start date of your DAG
    'retries': 1,
}

# Define the DAG
with DAG(
    dag_id='simple_dag',  # Unique name for the DAG
    default_args=default_args,
    description='A simple Airflow DAG',
    schedule_interval='@once',  # Run the DAG once
    catchup=False,  # Only run the most recent schedule
) as dag:

    # Define the first task
    def start_dag():
        print("Starting Airflow DAG")
    
    task_1 = PythonOperator(
        task_id='start_dag',
        python_callable=start_dag
    )

    # Define the second task
    def print_date_time():
        print(f"Current date and time: {datetime.now()}")

    task_2 = PythonOperator(
        task_id='print_date_time',
        python_callable=print_date_time
    )
    
    def fetch_weather_data(ti):
        print("Hello word")

        temperature_kelvin = 300.6
        # Push the temperature to XCom
        print("temperature_kelvin:", temperature_kelvin)
        ti.xcom_push(key='temperature_kelvin', value=temperature_kelvin)
        
        
    # Task 1: Fetch weather data
    task_3 = PythonOperator(
        task_id='fetch_weather_data',
        python_callable=fetch_weather_data
    )

    # Set task dependencies (task_1 -> task_2)
    task_1 >> task_2 >> task_3