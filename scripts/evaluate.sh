#!/bin/bash
# Evaluate PaddleOCR recognition model

set -e

echo "🚀 Starting model evaluation..."

# Check if PaddleOCR directory exists
if [ ! -d "PaddleOCR" ]; then
    echo "❌ Error: PaddleOCR directory not found!"
    echo "Please clone PaddleOCR first:"
    echo "  git clone https://github.com/PaddlePaddle/PaddleOCR.git"
    exit 1
fi

# Check if config file exists
if [ ! -f "configs/rec_custom.yml" ]; then
    echo "❌ Error: Configuration file not found: configs/rec_custom.yml"
    exit 1
fi

# Check if validation data exists
if [ ! -f "dataset/val/val_list.txt" ]; then
    echo "❌ Error: Validation data not found!"
    echo "Run 'python scripts/prepare_dataset.py' to prepare the dataset first."
    exit 1
fi

# Check if trained model exists
if [ ! -d "output/rec_model" ] || [ -z "$(ls -A output/rec_model)" ]; then
    echo "⚠️  Warning: No trained model found in output/rec_model/"
    echo "Make sure you have trained the model first using: bash scripts/train.sh"
fi

# Navigate to PaddleOCR directory
cd PaddleOCR

echo "📊 Evaluation configuration: ../configs/rec_custom.yml"
echo "📁 Validation data: ../dataset/val/val_list.txt"
echo ""

# Run evaluation
python tools/eval.py -c ../configs/rec_custom.yml

echo ""
echo "✅ Evaluation completed!"
