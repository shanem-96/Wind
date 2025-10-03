#!/bin/bash

# German Wind Turbines Interactive Map - Start Script
# This script starts the Streamlit application

echo "Starting German Wind Turbines Interactive Map..."
echo "================================================"
echo ""
echo "The app will be available at:"
echo "  Local URL: http://localhost:8501"
echo "  Network URL: http://192.168.1.232:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the Streamlit app
python3 -m streamlit run wind_turbines_app.py --server.headless true --server.port 8501

