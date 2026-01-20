#!/bin/bash
# Train PaddleOCR recognition model with custom configuration

set -e

echo "🚀 Starting PaddleOCR training..."

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

# Check if pretrained model exists
if [ ! -d "pretrained_models/en_PP-OCRv3_rec_train" ]; then
    echo "⚠️  Warning: Pretrained model not found!"
    echo "Run 'bash scripts/download_pretrained.sh' to download it first."
    read -p "Continue without pretrained model? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if training data exists
if [ ! -f "dataset/train/train_list.txt" ]; then
    echo "❌ Error: Training data not found!"
    echo "Run 'python scripts/prepare_dataset.py' to prepare the dataset first."
    exit 1
fi

# Navigate to PaddleOCR directory
cd PaddleOCR

echo "📊 Training configuration: ../configs/rec_custom.yml"
echo "📁 Output directory: ../output/rec_model/"
echo ""
echo "Starting training... (Press Ctrl+C to stop)"
echo ""

# Run training
python tools/train.py -c ../configs/rec_custom.yml

echo ""
echo "✅ Training completed!"
echo "📁 Models saved to: output/rec_model/"
echo ""
echo "Next steps:"
echo "  1. Evaluate: bash scripts/evaluate.sh"
echo "  2. Export: bash scripts/export.sh"
