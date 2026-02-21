#!/bin/bash
# Run the SER API server
# Usage: ./scripts/run_api.sh [--reload]

set -e

# Default configuration
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
LOG_LEVEL="${LOG_LEVEL:-info}"
RELOAD=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --reload)
            RELOAD="--reload"
            shift
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--reload] [--host HOST] [--port PORT]"
            exit 1
            ;;
    esac
done

echo "Starting SER API server..."
echo "  Host: $HOST"
echo "  Port: $PORT"
echo "  Log Level: $LOG_LEVEL"
if [[ -n "$RELOAD" ]]; then
    echo "  Reload: enabled"
fi

# Set PYTHONPATH to include src directory
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}src"

exec uvicorn src.api.main:app \
    --host "$HOST" \
    --port "$PORT" \
    --log-level "$LOG_LEVEL" \
    $RELOAD
