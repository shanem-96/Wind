#!/bin/bash

# German Wind Turbines Interactive Map - Start Script (Double-click launcher for macOS)
# This script starts the Streamlit application

cd "$(dirname "$0")"

echo "Starting German Wind Turbines Interactive Map..."
echo "================================================"
echo ""
echo "The app will be available at:"
echo "  Local URL: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the Streamlit app
python3 -m streamlit run wind_turbines_app.py --server.headless false --server.port 8501




