#!/bin/bash
set -e

case "$1" in
    web)
        if [ "$FLASK_DEBUG" = "1" ]; then
            echo "Starting web server (DEV mode with auto-reload)..."
            exec python -m flask run --host=0.0.0.0 --port=8080 --reload
        else
            echo "Starting web server (production)..."
            exec gunicorn -b 0.0.0.0:8080 -w 2 --timeout 300 panel.app:app
        fi
        ;;
    worker)
        echo "Starting Huey worker..."
        # Use threads instead of processes to keep embedding model in memory
        exec huey_consumer panel.jobs.huey_config.huey -w 2 -k thread
        ;;
    *)
        exec "$@"
        ;;
esac
