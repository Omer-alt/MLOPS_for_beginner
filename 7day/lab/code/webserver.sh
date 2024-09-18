# Launch webserver
# source .venv/bin/activate
export AIRFLOW_HOME=${PWD}/airflow
airflow webserver --port 8080  # http://localhost:8080




# airflow users create \
#     --username admin \
#     --firstname FOTSO \
#     --lastname OMER \
#     --role Admin \
#     --email spiderman@superhero.org \
#     --password mlops_courses