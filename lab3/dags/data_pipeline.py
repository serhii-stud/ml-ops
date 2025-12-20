from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'data_team',
    'start_date': datetime(2023, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'support_data_etl',
    default_args=default_args,
    description='Merge Golden Set + Logs -> Create Parquet',
    schedule_interval='@daily',
    catchup=False,
) as dag:

    t1 = BashOperator(
        task_id='ingest_initial_data_process',
        bash_command='python /opt/airflow/src/ingest_initial_data.py'
    )

    t2 = BashOperator(
        task_id='update_training_data_process',
        bash_command='python /opt/airflow/src/update_training_dataset.py'
    )

