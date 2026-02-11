from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

from arxiv_ingestion.tasks import process_failed_pdfs

default_args = {
    "owner": "arxiv-curator",
    "depends_on_past": False,
    "start_date": datetime(2026, 2, 8),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=30),
    "catchup": False,
}

dag = DAG(
    "arxiv_pdf_retry",
    default_args=default_args,
    description="Retry failed PDF processing 12 hours after daily ingestion",
    schedule="0 18 * * 1-5",  # Monday-Friday at 6 PM UTC
    max_active_runs=1,
    catchup=False,
    tags=["arxiv", "papers", "retry", "pdfs", "week3"],
)

retry_task = PythonOperator(
    task_id="process_failed_pdfs",
    python_callable=process_failed_pdfs,
    dag=dag,
)
