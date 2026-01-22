# OCR Fine-tuning with TrOCR

A complete setup for fine-tuning Microsoft's TrOCR (Transformer-based OCR) model on custom datasets. This repository provides dataset preparation utilities, training scripts, inference code, and a **FastAPI web service** for building and deploying custom OCR models.

## 🎯 Overview

This setup enables you to:
- Fine-tune TrOCR models on custom datasets
- Use the powerful `microsoft/trocr-small-printed` architecture
- Prepare datasets with automatic validation
- Monitor training progress with TensorBoard
- Export trained models for production inference
- **Deploy OCR model as a web service with REST API**
- **Use a modern HTML interface for easy OCR processing**

## 📁 Folder Structure

```
OCR-model/
├── dataset/                      # Dataset directory
│   ├── train/                    # Training set
│   │   ├── images/               # Training images
│   │   └── train_list.txt        # Training labels (image_path\tlabel)
│   ├── val/                      # Validation set
│   │   ├── images/               # Validation images
│   │   └── val_list.txt          # Validation labels
│   ├── test/                     # Test set
│   │   ├── images/               # Test images
│   │   └── test_list.txt         # Test labels
│   └── dict.txt                  # Character dictionary (optional)
├── checkpoints/                  # Training checkpoints (auto-created)
├── logs/                         # TensorBoard logs (auto-created)
├── dataset_processed/            # Processed HuggingFace datasets (auto-created)
├── model/                        # Final trained model (auto-created)
├── inference/
│   └── predict.py                # Inference script
├── static/                       # Web interface files
├── prepare_dataset.py            # Dataset preparation script
├── train.py                      # Training script
├── test.py                       # Testing script
├── app.py                        # FastAPI web service
└── requirements.txt              # Python dependencies
```

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare dataset
python prepare_dataset.py

# 3. Train model
python train.py

# 4. Test model
python test.py

# 5. Start web service
python app.py
```

Visit http://localhost:8000 for the web interface!

## 📊 Dataset Preparation

Your dataset should have this structure:
```
dataset/train/train_list.txt:
images/img001.png12345
images/img002.pngABCDE
```

Each line: `relative_image_path\tlabel`

Run preparation:
```bash
python prepare_dataset.py
```

## 🏋️ Training

```bash
# GPU training (recommended)
python train.py

# CPU training
python train.py --cpu

# Custom settings
python train.py --epochs 30 --batch_size 32
```

Monitor with TensorBoard:
```bash
tensorboard --logdir logs
```

**Expected time**: 30-90 min on GPU, 6-8 hours on CPU (10k images, 20 epochs)

## 🧪 Testing

```bash
# Test on 50 images
python test.py

# Test all images
python test.py --limit 0

# Verbose output
python test.py --verbose
```

**Target accuracy**: >85% exact match, CER < 0.06

## 🔮 Inference

### Command Line

```bash
# Single image
python inference/predict.py image.jpg

# Batch inference
python inference/predict.py --batch img1.jpg img2.jpg

# Detailed output
python inference/predict.py image.jpg --detail
```

### Python API

```python
from inference.predict import CustomTrOCR

ocr = CustomTrOCR(model_dir='./model', use_gpu=True)
text = ocr.predict('image.jpg')
print(text)  # '12345'
```

## 🌐 Web Service

### Start Server

```bash
python app.py

# Custom port
python app.py --port 8080

# CPU mode
python app.py --cpu
```

### API Endpoints

**POST /api/predict** - Simple OCR
```bash
curl -X POST "http://localhost:8000/api/predict" -F "file=@image.jpg"
```

Response:
```json
{"success": true, "text": "12345", "confidence": 0.95}
```

**POST /api/ocr** - Extended OCR with detail mode

**POST /api/ocr/batch** - Batch OCR (up to 10 images)

### Access Points
- Web Interface: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

## 🐛 Troubleshooting

**Model not found**: Run `python train.py` first

**CUDA out of memory**: Use `python train.py --batch_size 8` or `--cpu`

**Low accuracy**: Train longer (`--epochs 30`) or check dataset quality

**Slow inference**: Ensure GPU is available: `python -c "import torch; print(torch.cuda.is_available())"`

## 🔧 Configuration

Edit `train.py` to customize:
- Model: `microsoft/trocr-small-printed` (default), `trocr-small-handwritten`, etc.
- Epochs: 20 (default)
- Batch size: 16 (default)
- Learning rate: 5e-5 (default)
- Max text length: 10 characters (default)

## 📚 Resources

- [TrOCR Paper](https://arxiv.org/abs/2109.10282)
- [HuggingFace TrOCR](https://huggingface.co/docs/transformers/model_doc/trocr)
- [Microsoft TrOCR Models](https://huggingface.co/microsoft/trocr-small-printed)

---

**Happy OCR Fine-tuning! 🚀**
