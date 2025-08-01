#!/bin/bash
# TornadoVM ML Inference Engine Installation Script

echo "Installing TornadoVM ML Inference Engine..."
echo "=========================================="

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    exit 1
fi

# Install dependencies
echo "Installing Python dependencies..."
pip3 install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✓ Dependencies installed successfully"
else
    echo "✗ Failed to install dependencies"
    exit 1
fi

# Test the installation
echo "Testing inference engine..."
python3 test_inference.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Installation completed successfully!"
    echo ""
    echo "Usage examples:"
    echo "  python3 engine.py features.json 64"
    echo "  python3 engine.py -v features.json 1024"
    echo "  python3 test_inference.py"
else
    echo "✗ Installation test failed"
    exit 1
fi
