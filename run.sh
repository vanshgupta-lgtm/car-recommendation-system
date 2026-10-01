#!/bin/bash
cd "$(dirname "$0")"

if [ -d "venv" ]; then
    source venv/bin/activate
    streamlit run app/app.py
else
    python3 -m streamlit run app/app.py
fi
