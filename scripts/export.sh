#!/bin/bash
# Export PaddleOCR recognition model for inference

set -e

echo "🚀 Exporting model for inference..."

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

# Check if trained model exists
if [ ! -d "output/rec_model" ] || [ -z "$(ls -A output/rec_model/*.pdparams 2>/dev/null)" ]; then
    echo "❌ Error: No trained model found in output/rec_model/"
    echo "Train the model first using: bash scripts/train.sh"
    exit 1
fi

# Find the best model (latest or best_accuracy)
MODEL_PATH=""
if [ -f "output/rec_model/best_accuracy.pdparams" ]; then
    MODEL_PATH="output/rec_model/best_accuracy"
    echo "📊 Using best_accuracy model"
elif [ -f "output/rec_model/latest.pdparams" ]; then
    MODEL_PATH="output/rec_model/latest"
    echo "📊 Using latest model"
else
    # Find the latest epoch model
    LATEST=$(ls -t output/rec_model/*.pdparams 2>/dev/null | head -1)
    if [ -n "$LATEST" ]; then
        MODEL_PATH="${LATEST%.pdparams}"
        echo "📊 Using model: $MODEL_PATH"
    else
        echo "❌ Error: No model files found"
        exit 1
    fi
fi

# Create output directory
mkdir -p output/inference

# Navigate to PaddleOCR directory
cd PaddleOCR

echo "📁 Exporting to: ../output/inference/"
echo ""

# Export model
python tools/export_model.py \
    -c ../configs/rec_custom.yml \
    -o Global.pretrained_model=../$MODEL_PATH \
    Global.save_inference_dir=../output/inference/

echo ""
echo "✅ Model exported successfully!"
echo "📁 Inference model location: output/inference/"
echo ""
echo "Files created:"
echo "  - output/inference/inference.pdmodel"
echo "  - output/inference/inference.pdiparams"
echo ""
echo "You can now use the model for inference with: python inference/predict.py"
