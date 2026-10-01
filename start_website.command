#!/bin/bash
# Move to the directory where this script is located
cd "$(dirname "$0")"

echo "=========================================================="
echo "🚗 Launching Indian Car Recommendation System Website..."
echo "=========================================================="

# Check if virtual environment exists
if [ -d "venv" ]; then
    source venv/bin/activate
    streamlit run app/app.py
else
    echo "⚠️ Virtual environment not found. Running with global python..."
    python3 -m streamlit run app/app.py
fi
