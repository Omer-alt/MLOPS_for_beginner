from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from datetime import datetime

# Define default arguments
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 9, 18),  # Define a start date
    'retries': 1,
}

# with DAG(
#     dag_id='airflow_operators_dag',  # Unique DAG ID
#     default_args=default_args,
#     description='DAG with Bash and Python operators',
#     schedule_interval='@once',  # Run the DAG once when triggered
#     catchup=False,
# ) as dag:
    
# Define the DAG
with DAG(
    dag_id='airflow_operators_dag',  # Unique DAG ID
    default_args=default_args,
    description='DAG with Bash and Python operators',
    schedule_interval='5 * * * *',  # Run every 5 minutes
    catchup=False,
) as dag:

    # Bash Task: Echo a message
    bash_task = BashOperator(
        task_id='bash_task',
        bash_command='echo "Running Bash task"'
    )

    # Python Task: Print a message
    def print_message():
        print("Hello from Airflow")

    python_task = PythonOperator(
        task_id='python_task',
        python_callable=print_message
    )

    # Set task dependencies (bash_task -> python_task)
    bash_task >> python_task
