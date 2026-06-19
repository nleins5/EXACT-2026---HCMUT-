#!/bin/bash

# Navigate to project directory
cd "$(dirname "$0")"

# 1. Kill any existing uvicorn and ngrok processes
echo "Killing any existing uvicorn or ngrok instances..."
pkill -f uvicorn 2>/dev/null || true
killall ngrok 2>/dev/null || true

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
nohup python3 -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000 > logs/uvicorn.log 2>&1 &
echo $! > logs/uvicorn.pid

# Wait 8 seconds for uvicorn to warm up and load modules
echo "Waiting for FastAPI server to start..."
sleep 8

# 4. Start ngrok in the background
echo "Starting ngrok tunnel..."
nohup ngrok http 8000 --url cupping-frisbee-bottom.ngrok-free.dev > logs/ngrok.log 2>&1 &
echo $! > logs/ngrok.pid

sleep 3

# 5. Output status
echo "----------------------------------------"
echo "Server & Ngrok started!"
echo "Check uvicorn logs: tail -n 20 logs/uvicorn.log"
echo "Check ngrok logs: tail -n 20 logs/ngrok.log"
echo "Test endpoint: curl https://cupping-frisbee-bottom.ngrok-free.dev/health"
echo "----------------------------------------"
