#!/bin/bash

# Navigate to project directory
cd "$(dirname "$0")"

PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
NGROK_URL="${NGROK_URL:-}"

# 1. Kill any existing EXACT uvicorn process and optional ngrok tunnel
echo "Killing any existing EXACT uvicorn instance..."
if [ -f logs/uvicorn.pid ]; then
    kill "$(cat logs/uvicorn.pid)" 2>/dev/null || true
fi
pkill -f "uvicorn src.api.app:app" 2>/dev/null || true
if [ -n "$NGROK_URL" ]; then
    echo "Killing existing ngrok instances..."
    killall ngrok 2>/dev/null || true
fi

# Wait 2 seconds
sleep 2

# 2. Activate virtual environment
if [ -d "venv" ]; then
    echo "Activating virtual environment 'venv'..."
    source venv/bin/activate
elif [ -d ".venv" ]; then
    echo "Activating virtual environment '.venv'..."
    source .venv/bin/activate
else
    echo "No virtual environment found, using system Python..."
fi

# Ensure logs directory exists
mkdir -p logs

# 3. Start FastAPI server in the background
echo "Starting FastAPI server in the background..."
nohup python3 -m uvicorn src.api.app:app --host "$HOST" --port "$PORT" --workers 1 > logs/uvicorn.log 2>&1 &
echo $! > logs/uvicorn.pid

# Wait 8 seconds for uvicorn to warm up and load modules
echo "Waiting for FastAPI server to start..."
sleep 8

# 4. Start ngrok in the background when NGROK_URL is configured
if [ -n "$NGROK_URL" ]; then
    echo "Starting ngrok tunnel..."
    nohup ngrok http "$PORT" --url "$NGROK_URL" > logs/ngrok.log 2>&1 &
    echo $! > logs/ngrok.pid
    sleep 3
fi

# 5. Output status
echo "----------------------------------------"
echo "Server & Ngrok started!"
echo "Check uvicorn logs: tail -n 20 logs/uvicorn.log"
if [ -n "$NGROK_URL" ]; then
    echo "Check ngrok logs: tail -n 20 logs/ngrok.log"
    echo "Test endpoint: curl https://$NGROK_URL/health"
else
    echo "Ngrok disabled. Set NGROK_URL=your-domain.ngrok-free.dev to expose Phase 2 endpoint."
    echo "Test endpoint: curl http://127.0.0.1:$PORT/health"
fi
echo "----------------------------------------"
