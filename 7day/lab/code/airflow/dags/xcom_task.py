from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

# Default arguments
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 9, 18),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
with DAG(
    dag_id='xcom_example_dag',  # Unique ID for this DAG
    default_args=default_args,
    description='An example DAG showing XCom usage between Python and Bash tasks',
    schedule_interval='5 * * * *',  # Run every 5 minutes
    catchup=False,
) as dag:

    # Task 1: Python task that pushes current timestamp to XCom
    def push_timestamp(ti):
        """Push the current timestamp to XCom."""
        current_time = datetime.now()
        ti.xcom_push(key='timestamp', value=current_time)
        print(f"Pushed timestamp: {current_time}")

    push_task = PythonOperator(
        task_id='push_timestamp',
        python_callable=push_timestamp,
    )

    # Task 2: Bash task that pulls the timestamp from XCom and echoes it
    # bash_task = BashOperator(
    #     task_id='echo_timestamp',
    #     bash_command='echo "Pulled timestamp from XCom: {{ ti.xcom_pull(task_ids=\'push_timestamp\', key=\'timestamp\') }}"'
    # )
    
    bash_task = BashOperator(
    task_id='echo_timestamp',
    bash_command="""
        timestamp="{{ ti.xcom_pull(task_ids='push_timestamp', key='timestamp') }}"
        echo "Pulled timestamp from XCom: $timestamp"
    """
)

    # Set task dependencies (push_task -> bash_task)
    push_task >> bash_task
