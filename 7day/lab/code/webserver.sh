# Launch webserver
# source .venv/bin/activate
export AIRFLOW_HOME=${PWD}/airflow
airflow webserver --port 8080  # http://localhost:8080




# airflow users create \
#     --username omer \
#     --firstname omer \
#     --lastname omer \
#     --role Admin \
#     --email owafo@aimsammi.org \
#     --password mlops_course