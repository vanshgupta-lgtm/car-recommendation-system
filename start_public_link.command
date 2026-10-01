#!/bin/bash
# Move to script directory
cd "$(dirname "$0")"

echo "=========================================================="
echo "🌐 Starting Public Internet Link for Car Advisor..."
echo "=========================================================="

# Ensure Streamlit is running
if ! pgrep -f "streamlit run" > /dev/null; then
    echo "Starting Streamlit..."
    if [ -d "venv" ]; then
        source venv/bin/activate
        streamlit run app/app.py --server.port 8501 --server.headless true &
    else
        python3 -m streamlit run app/app.py --server.port 8501 --server.headless true &
    fi
    sleep 3
fi

# Kill any old tunnel
pkill -f "cloudflared tunnel" 2>/dev/null

echo "Starting Cloudflare Public Tunnel..."
cloudflared tunnel --url http://localhost:8501
