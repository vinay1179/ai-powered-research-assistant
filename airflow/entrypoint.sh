#!/bin/bash
set -e

# Clean up any existing PID files and processes
echo "Cleaning up any existing Airflow processes..."
pkill -f "airflow webserver" || true
pkill -f "airflow scheduler" || true
rm -f /opt/airflow/airflow-webserver.pid
rm -f /opt/airflow/airflow-scheduler.pid

# Wait a moment for processes to fully terminate
sleep 2

# Initialize Airflow database
echo "Initializing Airflow database..."
airflow db init

# Create admin user from environment variables
echo "Creating admin user..."
airflow users create \
    --username "${AIRFLOW_ADMIN_USERNAME}" \
    --firstname "${AIRFLOW_ADMIN_FIRSTNAME}" \
    --lastname "${AIRFLOW_ADMIN_LASTNAME}" \
    --role "${AIRFLOW_ADMIN_ROLE}" \
    --email "${AIRFLOW_ADMIN_EMAIL}" \
    --password "${AIRFLOW_ADMIN_PASSWORD}" || echo "Admin user already exists"

# Start webserver and scheduler
echo "Starting Airflow webserver and scheduler..."
airflow webserver --port 8080 --daemon &
airflow scheduler
