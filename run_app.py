#!/usr/bin/env python3
"""
MarketVision Pro - Application Launcher
A simple launcher script for the MarketVision Streamlit application.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def check_dependencies():
    required_packages = [
        'streamlit', 'pandas', 'numpy', 'plotly', 'yfinance', 
        'joblib', 'scikit-learn', 'lightgbm'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("📦 Installing missing packages...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing_packages)
            print("✅ Dependencies installed successfully!")
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies. Please run:")
            print(f"pip install -r streamlit_requirements.txt")
            return False
    
    return True

def check_model_files():
    required_files = [
        'models/multioutput_lightgbm_model.pkl',
        'models/lightgbm_model_info.json'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing model files: {', '.join(missing_files)}")
        print("🔧 Please ensure you have trained the model first.")
        return False
    
    print("✅ Model files found!")
    return True

def check_data_files():
    data_files = [
        'data/processed/stock_RELIANCE.NS_with_indicators.csv',
        'data/processed/stock_RELIANCE.NS_with_macro_context.csv',
        'data/processed/stock_RELIANCE.NS_with_sentiment.csv'
    ]
    
    missing_files = []
    for file_path in data_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"⚠️  Some data files are missing: {', '.join(missing_files)}")
        print("📊 The app will work with limited functionality.")
        return False
    
    print("✅ Data files found!")
    return True

def run_streamlit_app(app_type="basic", port=8501, host="localhost"):
    app_file = "advanced_streamlit_app.py" if app_type == "advanced" else "streamlit_app.py"
    
    if not os.path.exists(app_file):
        print(f"❌ App file not found: {app_file}")
        return False
    
    print(f"🚀 Starting MarketVision Pro ({app_type} mode)...")
    print(f"🌐 URL: http://{host}:{port}")
    print("⏹️  Press Ctrl+C to stop the application")
    print("-" * 50)
    
    try:
        cmd = [
            sys.executable, '-m', 'streamlit', 'run', app_file,
            '--server.port', str(port),
            '--server.address', host,
            '--server.headless', 'true',
            '--browser.gatherUsageStats', 'false'
        ]
        
        subprocess.run(cmd)
        return True
        
    except KeyboardInterrupt:
        print("\n🛑 Application stopped by user.")
        return True
    except Exception as e:
        print(f"❌ Error running application: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="MarketVision Pro - Stock Prediction App Launcher")
    parser.add_argument(
        '--mode', 
        choices=['basic', 'advanced'], 
        default='advanced',
        help='Application mode (default: advanced)'
    )
    parser.add_argument(
        '--port', 
        type=int, 
        default=8501,
        help='Port number (default: 8501)'
    )
    parser.add_argument(
        '--host', 
        default='localhost',
        help='Host address (default: localhost)'
    )
    parser.add_argument(
        '--skip-checks', 
        action='store_true',
        help='Skip dependency and file checks'
    )
    
    args = parser.parse_args()
    
    # Print banner
    print("=" * 60)
    print("📈 MarketVision Pro - AI Stock Predictor")
    print("=" * 60)
    
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    if not args.skip_checks:
        print("🔍 Performing pre-flight checks...")
        
        if not check_dependencies():
            return 1
        
        if not check_model_files():
            return 1
        
        check_data_files()
        
        print("✅ All checks passed!")
        print()
        
    success = run_streamlit_app(args.mode, args.port, args.host)
    
    if success:
        print("👋 Thanks for using MarketVision Pro!")
        return 0
    else:
        print("❌ Application failed to start.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 