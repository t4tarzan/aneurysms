#!/bin/bash

# Set default values
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}
WORKERS=${WORKERS:-4}
LOG_LEVEL=${LOG_LEVEL:-info}

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# Create uploads directory if it doesn't exist
mkdir -p uploads

# Run the FastAPI application with uvicorn
echo "Starting Aneurysm Detection API on http://${HOST}:${PORT}"
uvicorn app.main:app \
    --host $HOST \
    --port $PORT \
    --workers $WORKERS \
    --log-level $LOG_LEVEL \
    --reload
