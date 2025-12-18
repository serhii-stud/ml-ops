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
    '1_banking_data_etl',
    default_args=default_args,
    description='Merge Golden Set + Logs -> Create Parquet',
    schedule_interval='@daily',
    catchup=False,
) as dag:

    t1 = BashOperator(
        task_id='ingest_logs',
        bash_command='python /opt/airflow/src/ingest.py'
    )

    t2 = BashOperator(
        task_id='etl_process',
        bash_command='python /opt/airflow/src/etl.py'
    )

