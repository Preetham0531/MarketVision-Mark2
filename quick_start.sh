#!/bin/bash

# MarketVision Pro - Quick Start Script
# This script provides a quick way to start the MarketVision application

echo "============================================================"
echo "📈 MarketVision Pro - AI Stock Predictor"
echo "============================================================"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "streamlit_app.py" ]; then
    echo "❌ Please run this script from the MarketVision directory."
    exit 1
fi

# Install dependencies if needed
echo "🔍 Checking dependencies..."
if ! python3 -c "import streamlit" &> /dev/null; then
    echo "📦 Installing dependencies..."
    pip3 install -r streamlit_requirements.txt
fi

# Check if model files exist
if [ ! -f "models/multioutput_lightgbm_model.pkl" ]; then
    echo "❌ Model files not found. Please ensure the model is trained first."
    echo "💡 Run the training scripts to generate the model files."
    exit 1
fi

echo "✅ All checks passed!"
echo ""

# Ask user for preferences
echo "🎯 Choose your preferred mode:"
echo "1) Basic Mode (Simple interface)"
echo "2) Advanced Mode (Full features) - Recommended"
read -p "Enter your choice (1 or 2): " choice

case $choice in
    1)
        echo "🚀 Starting MarketVision in Basic Mode..."
        python3 run_app.py --mode basic
        ;;
    2)
        echo "🚀 Starting MarketVision in Advanced Mode..."
        python3 run_app.py --mode advanced
        ;;
    *)
        echo "🚀 Starting MarketVision in Advanced Mode (default)..."
        python3 run_app.py --mode advanced
        ;;
esac 