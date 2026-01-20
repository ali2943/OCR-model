#!/bin/bash
# Quick Start Script for OCR Web Service
# This script helps you quickly get started with the OCR web service

set -e  # Exit on error

echo "============================================"
echo "OCR Web Service - Quick Start"
echo "============================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✓ Python 3 is installed"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt
echo "✓ Dependencies installed"

# Check if model exists
if [ ! -f "output/inference/inference.pdmodel" ] || [ ! -f "output/inference/inference.pdiparams" ]; then
    echo ""
    echo "⚠️  WARNING: Trained model not found!"
    echo ""
    echo "Before running the web service, you need to:"
    echo "  1. Prepare your dataset: python scripts/prepare_dataset.py"
    echo "  2. Download pretrained model: bash scripts/download_pretrained.sh"
    echo "  3. Train the model: bash scripts/train.sh"
    echo "  4. Export the model: bash scripts/export.sh"
    echo ""
    echo "Or, if you have a pre-trained model, place it in ./output/inference/"
    echo ""
    read -p "Do you want to continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if dictionary exists
if [ ! -f "dataset/dict.txt" ]; then
    echo ""
    echo "⚠️  WARNING: Character dictionary (dataset/dict.txt) not found!"
    echo "Please prepare your dataset first with: python scripts/prepare_dataset.py"
    echo ""
    read -p "Do you want to continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Start the server
echo ""
echo "🚀 Starting OCR Web Service..."
echo ""
echo "The service will be available at:"
echo "  • Web Interface: http://localhost:8000"
echo "  • API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""
echo "============================================"
echo ""

python app.py
