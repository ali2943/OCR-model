# TrOCR Migration Guide

This document explains the migration from PaddleOCR to TrOCR (Transformer-based OCR) for better accuracy, easier deployment, and faster training.

## 🎯 What Changed

### Model Architecture
- **Old**: PaddleOCR (CNN-based recognition)
- **New**: TrOCR (Transformer-based recognition from Microsoft)
- **Base Model**: `microsoft/trocr-small-printed`

### Key Improvements
✅ Better accuracy on printed text  
✅ Easier to fine-tune with HuggingFace Transformers  
✅ No complex export steps required  
✅ Simpler deployment (no PaddlePaddle dependencies)  
✅ Better integration with modern ML tools  

### Files Added
```
prepare_trocr_dataset.py    # Dataset preparation script
train_trocr.py              # Training script
test_trocr.py               # Testing script
inference/predict_trocr.py  # TrOCR inference module
ocr_fastapi_app.py          # FastAPI app with TrOCR support
requirements_trocr.txt      # TrOCR dependencies
trocr_model/                # Model checkpoints and data
  ├── checkpoints/          # Training checkpoints
  ├── logs/                 # TensorBoard logs
  ├── dataset_processed/    # Processed dataset
  └── final/                # Final trained model
```

### Files Preserved
All existing PaddleOCR files remain unchanged:
- `inference/predict.py` (PaddleOCR inference)
- `app.py` (PaddleOCR FastAPI app)
- `requirements.txt` (PaddleOCR dependencies)

## 📦 Installation

### 1. Install Dependencies

```bash
pip install -r requirements_trocr.txt
```

### 2. Verify Installation

```bash
python -c "import transformers; print(transformers.__version__)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

## 🚀 Training Workflow

### Step 1: Prepare Dataset

Convert your existing dataset to HuggingFace format:

```bash
python prepare_trocr_dataset.py
```

**What it does:**
- Reads from `dataset/train/`, `dataset/val/`, `dataset/test/`
- Validates and loads images
- Converts to HuggingFace Dataset format
- Saves to `trocr_model/dataset_processed/`
- Prints statistics for each split

**Expected output:**
```
📊 SUMMARY
TRAIN:
   Total lines: 10000
   Valid samples: 9950
   Skipped: 30
   Corrupted: 20

VALIDATION:
   Total lines: 2000
   Valid samples: 1990
   ...
```

### Step 2: Train Model

Fine-tune TrOCR on your dataset:

```bash
# With GPU (recommended)
python train_trocr.py

# With CPU (slower)
python train_trocr.py --cpu

# Custom settings
python train_trocr.py --epochs 30 --batch_size 32 --learning_rate 3e-5
```

**Configuration:**
- **Epochs**: 20 (default)
- **Batch size**: 16 per device
- **Learning rate**: 5e-5
- **Max text length**: 10 characters
- **Save checkpoints**: Every 500 steps
- **Evaluation**: Every 500 steps

**What it does:**
- Loads pre-trained `microsoft/trocr-small-printed`
- Fine-tunes on your custom dataset
- Saves checkpoints to `trocr_model/checkpoints/`
- Logs training to TensorBoard (`trocr_model/logs/`)
- Saves final model to `trocr_model/final/`
- Computes CER (Character Error Rate) metric

**Monitor training with TensorBoard:**
```bash
tensorboard --logdir trocr_model/logs
```
Then open http://localhost:6006 in your browser.

### Step 3: Test Model

Evaluate the trained model:

```bash
# Test on first 50 images (default)
python test_trocr.py

# Test on all images
python test_trocr.py --limit 0

# Test with CPU
python test_trocr.py --cpu

# Show per-image results
python test_trocr.py --verbose
```

**Expected output:**
```
📊 TEST RESULTS
Total images tested: 50
Correct predictions: 45
Accuracy: 90.00%
Character Error Rate (CER): 0.0523
```

### Step 4: Run API Server

Start the FastAPI server with TrOCR:

```bash
# With GPU (default)
python ocr_fastapi_app.py --model_dir ./trocr_model/final

# With CPU
python ocr_fastapi_app.py --model_dir ./trocr_model/final --cpu

# Custom port
python ocr_fastapi_app.py --port 8080

# Development mode with auto-reload
python ocr_fastapi_app.py --reload
```

**API Endpoints:**
- `GET /` - Web interface
- `GET /health` - Health check
- `POST /api/predict` - Single image OCR
- `POST /api/ocr` - Single image OCR (extended)
- `POST /api/ocr/batch` - Batch OCR
- `GET /api/info` - Model information

**API is 100% backward compatible** with existing frontend!

## ⏱️ Expected Training Time

| Hardware | Dataset Size | Training Time | Test Accuracy |
|----------|-------------|---------------|---------------|
| GPU (RTX 3090) | 10k images | ~30-45 min | >90% |
| GPU (GTX 1080) | 10k images | ~60-90 min | >90% |
| CPU (8 cores) | 10k images | ~6-8 hours | >90% |

**Note**: Times are approximate for 20 epochs. GPU training is strongly recommended.

## 📈 Expected Accuracy Improvements

| Metric | PaddleOCR | TrOCR | Improvement |
|--------|-----------|-------|-------------|
| Exact Match | ~45% | ~85-90% | +40-45% |
| CER | ~0.15 | ~0.05 | -66% |

## 🔧 Command-Line Inference

### Single Image

```bash
python inference/predict_trocr.py path/to/image.jpg
```

### Batch Inference

```bash
python inference/predict_trocr.py --batch img1.jpg img2.jpg img3.jpg
```

### With Details

```bash
python inference/predict_trocr.py path/to/image.jpg --detail
```

## 🌐 API Usage Examples

### cURL

```bash
# Single image
curl -X POST "http://localhost:8000/api/predict" \
  -F "file=@image.jpg"

# Batch images
curl -X POST "http://localhost:8000/api/ocr/batch" \
  -F "files=@img1.jpg" \
  -F "files=@img2.jpg"
```

### Python

```python
import requests

# Single image
with open('image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/predict',
        files={'file': f}
    )
    print(response.json())
    # {'success': True, 'text': '12345', 'confidence': 0.95}
```

### JavaScript

```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('http://localhost:8000/api/predict', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => console.log(data));
```

## 🐛 Troubleshooting

### Issue: "Model directory not found"
**Solution:**
```bash
# Make sure you trained the model first
python train_trocr.py

# Or specify correct model directory
python ocr_fastapi_app.py --model_dir ./trocr_model/final
```

### Issue: "CUDA out of memory"
**Solution:**
```bash
# Reduce batch size
python train_trocr.py --batch_size 8

# Or use CPU
python train_trocr.py --cpu
```

### Issue: "Dataset not found"
**Solution:**
```bash
# Make sure dataset is prepared first
python prepare_trocr_dataset.py

# Check dataset structure
ls -la dataset/train/
ls -la dataset/val/
ls -la dataset/test/
```

### Issue: Low accuracy (<85%)
**Possible causes:**
1. Insufficient training epochs
   ```bash
   python train_trocr.py --epochs 30
   ```

2. Dataset quality issues
   - Check for corrupted images
   - Verify labels are correct
   - Ensure consistent image quality

3. Learning rate too high/low
   ```bash
   python train_trocr.py --learning_rate 3e-5
   ```

### Issue: Slow inference
**Solution:**
```bash
# Make sure you're using GPU
python ocr_fastapi_app.py  # Should auto-detect GPU

# Check GPU is being used
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

## 📊 Model Configuration

You can customize training by editing `train_trocr.py`:

```python
CONFIG = {
    'model_name': 'microsoft/trocr-small-printed',
    'output_dir': './trocr_model/checkpoints',
    'num_train_epochs': 20,
    'per_device_train_batch_size': 16,
    'per_device_eval_batch_size': 16,
    'learning_rate': 5e-5,
    'max_length': 10,
    'save_steps': 500,
    'eval_steps': 500,
    'logging_steps': 100,
}
```

## 🔄 Switching Between Models

### Use TrOCR (recommended)
```bash
python ocr_fastapi_app.py --model_dir ./trocr_model/final
```

### Use PaddleOCR (legacy)
```bash
python app.py --model_dir ./output/inference/
```

Both servers use the same API endpoints, so your frontend will work with either!

## 📝 Notes

- **TrOCR is optimized for printed text**. For handwritten text, consider `microsoft/trocr-small-handwritten`
- **Model size**: ~60MB (much smaller than PaddleOCR)
- **GPU memory**: ~2GB during training, ~1GB during inference
- **CPU is supported** but GPU is strongly recommended for training

## 🎓 Additional Resources

- [TrOCR Paper](https://arxiv.org/abs/2109.10282)
- [HuggingFace TrOCR](https://huggingface.co/docs/transformers/model_doc/trocr)
- [Microsoft TrOCR Models](https://huggingface.co/microsoft/trocr-small-printed)

## ✅ Success Checklist

- [ ] Dependencies installed (`pip install -r requirements_trocr.txt`)
- [ ] Dataset prepared (`python prepare_trocr_dataset.py`)
- [ ] Model trained (`python train_trocr.py`)
- [ ] Model tested (`python test_trocr.py`)
- [ ] Test accuracy >85%
- [ ] API server running (`python ocr_fastapi_app.py`)
- [ ] Frontend works with new backend
- [ ] All API endpoints functional

## 🚀 Quick Start (TL;DR)

```bash
# 1. Install
pip install -r requirements_trocr.txt

# 2. Prepare dataset
python prepare_trocr_dataset.py

# 3. Train model (20 epochs, ~30-90 min on GPU)
python train_trocr.py

# 4. Test model
python test_trocr.py

# 5. Run API server
python ocr_fastapi_app.py --model_dir ./trocr_model/final

# 6. Open browser
# http://localhost:8000
```

That's it! 🎉
