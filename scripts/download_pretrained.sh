#!/bin/bash
# Download pretrained PP-OCRv3 English recognition model

set -e

echo "🚀 Downloading PP-OCRv3 English recognition pretrained model..."

# Create pretrained_models directory if it doesn't exist
mkdir -p pretrained_models

cd pretrained_models

# Download the pretrained model
MODEL_URL="https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_train.tar"
MODEL_FILE="en_PP-OCRv3_rec_train.tar"

if [ -f "$MODEL_FILE" ]; then
    echo "⚠️  Model archive already exists. Removing old file..."
    rm -f "$MODEL_FILE"
fi

echo "📥 Downloading from $MODEL_URL..."
wget -q --show-progress "$MODEL_URL" || curl -O -# "$MODEL_URL"

# Extract the model
echo "📦 Extracting model..."
tar -xf "$MODEL_FILE"

# Clean up the archive
echo "🧹 Cleaning up..."
rm -f "$MODEL_FILE"

echo "✅ Pretrained model downloaded and extracted successfully!"
echo "📁 Model location: pretrained_models/en_PP-OCRv3_rec_train/"
echo ""
echo "You can now proceed with training using: bash scripts/train.sh"
