# PaddleOCR Fine-tuning Environment

A complete setup for fine-tuning PaddleOCR text recognition models with custom datasets. This repository provides an organized structure, dataset preparation utilities, training scripts, inference code, and a **FastAPI web service with HTML interface** for building and deploying custom OCR models.

## 📋 Table of Contents

- [Overview](#overview)
- [Folder Structure](#folder-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Web Service](#web-service)
- [Dataset Preparation](#dataset-preparation)
- [Training Workflow](#training-workflow)
- [Configuration](#configuration)
- [Inference](#inference)
- [Monitoring & Troubleshooting](#monitoring--troubleshooting)
- [Resources](#resources)

## 🎯 Overview

This setup enables you to:
- Fine-tune PaddleOCR recognition models on custom datasets
- Use the powerful SVTR_LCNet architecture with PP-OCRv3 pretrained weights
- Prepare datasets with automatic validation and splitting
- Monitor training progress and evaluate model performance
- Export trained models for production inference
- **Deploy OCR model as a web service with REST API**
- **Use a modern HTML interface for easy OCR processing**

## 📁 Folder Structure

```
OCR-model/
├── dataset/                      # Dataset directory
│   ├── raw/                      # Raw dataset
│   │   ├── images/               # Raw images
│   │   ├── labels.txt            # Image-text pairs (TAB-separated)
│   │   └── .gitkeep
│   ├── train/                    # Training set
│   │   ├── images/               # Training images
│   │   ├── train_list.txt        # Training labels (PaddleOCR format)
│   │   └── .gitkeep
│   ├── val/                      # Validation set
│   │   ├── images/               # Validation images
│   │   ├── val_list.txt          # Validation labels
│   │   └── .gitkeep
│   ├── test/                     # Test set
│   │   ├── images/               # Test images
│   │   ├── test_list.txt         # Test labels
│   │   └── .gitkeep
│   └── dict.txt                  # Character dictionary (auto-generated)
├── configs/
│   └── rec_custom.yml            # Training configuration
├── pretrained_models/            # Pretrained model weights
│   ├── en_PP-OCRv3_rec_train/    # PP-OCRv3 English model
│   └── .gitkeep
├── output/                       # Training outputs
│   ├── rec_model/                # Model checkpoints
│   │   └── .gitkeep
│   └── inference/                # Exported inference models
│       └── .gitkeep
├── scripts/                      # Helper scripts
│   ├── prepare_dataset.py        # Dataset preparation
│   ├── download_pretrained.sh    # Download pretrained models
│   ├── train.sh                  # Training script
│   ├── evaluate.sh               # Evaluation script
│   └── export.sh                 # Model export script
├── inference/
│   └── predict.py                # Inference script
├── static/                       # Web interface
│   └── index.html                # HTML interface for OCR
├── logs/                         # Training logs
│   └── .gitkeep
├── app.py                        # FastAPI web service
├── API_DOCS.md                   # API documentation
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

## 🚀 Installation

### 1. System Requirements

- Python 3.8+
- CUDA 11.2+ (for GPU training)
- 8GB+ RAM (16GB+ recommended)
- 10GB+ free disk space

### 2. Clone Repository

```bash
git clone https://github.com/ali2943/OCR-model.git
cd OCR-model
```

### 3. Install Dependencies

For GPU (recommended):
```bash
pip install -r requirements.txt
```

For CPU only:
```bash
# Edit requirements.txt and replace 'paddlepaddle-gpu' with 'paddlepaddle'
pip install -r requirements.txt
```

### 4. Clone PaddleOCR

```bash
git clone https://github.com/PaddlePaddle/PaddleOCR.git
cd PaddleOCR
pip install -e .
cd ..
```

## ⚡ Quick Start

### Complete Workflow

```bash
# 1. Prepare your dataset (see Dataset Preparation section)
python scripts/prepare_dataset.py

# 2. Download pretrained model
bash scripts/download_pretrained.sh

# 3. Train the model
bash scripts/train.sh

# 4. Evaluate the model
bash scripts/evaluate.sh

# 5. Export for inference
bash scripts/export.sh

# 6. Run inference
python inference/predict.py path/to/test/image.jpg

# 7. Start the web service (NEW!)
python app.py
```

## 🌐 Web Service

### Starting the Web Server

After exporting your trained model, you can start the web service:

```bash
python app.py
```

This will start a FastAPI server on `http://localhost:8000` with:
- 🖥️ **Web Interface**: User-friendly HTML interface at `http://localhost:8000`
- 📚 **API Documentation**: Interactive docs at `http://localhost:8000/docs`
- 🔌 **REST API**: Programmatic access to OCR functionality

### Web Interface Features

The HTML interface (`http://localhost:8000`) provides:
- **Drag & Drop**: Easily upload images by dragging them into the browser
- **Live Preview**: See your uploaded image before processing
- **Instant Results**: Get OCR results displayed in real-time
- **Detailed Mode**: View confidence scores and bounding boxes
- **Responsive Design**: Works on desktop and mobile devices

### API Endpoints

**Health Check:**
```bash
curl http://localhost:8000/health
```

**Single Image OCR:**
```bash
curl -X POST "http://localhost:8000/api/ocr" \
  -F "file=@image.jpg"
```

**Batch Processing:**
```bash
curl -X POST "http://localhost:8000/api/ocr/batch" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg"
```

### Server Options

```bash
# Custom port
python app.py --port 5000

# CPU inference
python app.py --cpu

# Custom model path
python app.py --model_dir ./output/inference/ --dict_path ./dataset/dict.txt

# Development mode with auto-reload
python app.py --reload
```

For complete API documentation, see [API_DOCS.md](API_DOCS.md)

## 📊 Dataset Preparation

### Input Format

Your raw dataset should be organized as:

```
dataset/raw/
├── images/
│   ├── img_001.png
│   ├── img_002.png
│   └── ...
└── labels.txt
```

**labels.txt format** (TAB-separated):
```
img_001.png	Ground truth text
img_002.png	Another text sample
img_003.png	More examples here
```

⚠️ **Important**: Use TAB (`\t`) character as separator, not spaces!

### Run Dataset Preparation

```bash
python scripts/prepare_dataset.py
```

This script will:
- ✅ Validate all images (check for corruption)
- ✅ Split data into train/val/test (80%/10%/10%)
- ✅ Convert to PaddleOCR format
- ✅ Generate character dictionary
- ✅ Provide detailed statistics

**Optional arguments:**
```bash
python scripts/prepare_dataset.py \
    --raw_dir ./dataset/raw \
    --output_dir ./dataset \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --seed 42
```

### Output Format

After preparation, you'll have:

**train_list.txt / val_list.txt / test_list.txt:**
```
images/img_001.png	Ground truth text
images/img_002.png	Another text sample
```

**dict.txt** (one character per line):
```
a
b
c
...
```

## 🎓 Training Workflow

### 1. Download Pretrained Model

```bash
bash scripts/download_pretrained.sh
```

Downloads PP-OCRv3 English recognition model (~50MB).

### 2. Start Training

```bash
bash scripts/train.sh
```

The script will:
- Check for PaddleOCR installation
- Validate configuration and data paths
- Start training with progress monitoring
- Save checkpoints every 10 epochs
- Evaluate during training

**Training parameters** (in `configs/rec_custom.yml`):
- Epochs: 100
- Batch size: 128
- Learning rate: 0.001 (Cosine scheduler)
- Image shape: [3, 48, 320]
- Architecture: SVTR_LCNet

### 3. Monitor Training

Watch the terminal output for:
- Loss values (should decrease)
- Accuracy metrics
- Learning rate schedule

Checkpoints are saved to `output/rec_model/`:
- `best_accuracy.pdparams` - Best performing model
- `latest.pdparams` - Most recent checkpoint
- `iter_epoch_*.pdparams` - Periodic checkpoints

### 4. Evaluate Model

```bash
bash scripts/evaluate.sh
```

Evaluates on validation set and reports:
- Character-level accuracy
- Word-level accuracy
- Per-sample predictions

### 5. Export for Inference

```bash
bash scripts/export.sh
```

Exports the model to `output/inference/`:
- `inference.pdmodel` - Model architecture
- `inference.pdiparams` - Model weights

## ⚙️ Configuration

### Main Configuration File: `configs/rec_custom.yml`

**Key settings you can customize:**

```yaml
Global:
  epoch_num: 100                    # Training epochs
  save_epoch_step: 10               # Checkpoint frequency
  eval_batch_step: 500              # Evaluation frequency
  character_dict_path: ./dataset/dict.txt
  max_text_length: 25               # Maximum text length

Optimizer:
  lr:
    learning_rate: 0.001            # Initial learning rate
    warmup_epoch: 5                 # Warmup epochs

Train:
  loader:
    batch_size_per_card: 128        # Batch size
    num_workers: 8                  # Data loading workers

Eval:
  loader:
    batch_size_per_card: 128
    num_workers: 4
```

### Image Shape

The default image shape is `[3, 48, 320]` (channels, height, width). Adjust based on your data:
- Taller images → increase height (e.g., 64)
- Longer text → increase width (e.g., 384, 512)

## 🔮 Inference

### Single Image Prediction

```bash
python inference/predict.py path/to/image.jpg
```

### Detailed Output (with confidence scores)

```bash
python inference/predict.py path/to/image.jpg --detail
```

### Batch Prediction

```bash
python inference/predict.py --batch img1.jpg img2.jpg img3.jpg
```

### CPU Inference

```bash
python inference/predict.py path/to/image.jpg --cpu
```

### Python API

```python
from inference.predict import CustomPaddleOCR

# Initialize
ocr = CustomPaddleOCR(
    model_dir='./output/inference/',
    dict_path='./dataset/dict.txt',
    use_gpu=True
)

# Single prediction
text = ocr.predict('image.jpg')
print(f"Recognized: {text}")

# Detailed prediction
results = ocr.predict('image.jpg', detail=True)
for item in results:
    print(f"Text: {item['text']}, Confidence: {item['confidence']}")

# Batch prediction
texts = ocr.predict_batch(['img1.jpg', 'img2.jpg'])
```

## 🔍 Monitoring & Troubleshooting

### Common Issues

**1. Out of Memory (OOM)**
- Reduce `batch_size_per_card` in config
- Reduce `num_workers`
- Use smaller images

**2. PaddleOCR not found**
```bash
cd PaddleOCR
pip install -e .
```

**3. Dataset preparation fails**
- Check `labels.txt` format (TAB-separated)
- Verify image file paths
- Ensure images are valid (not corrupted)

**4. Training loss not decreasing**
- Check learning rate (try 0.0001 or 0.01)
- Verify data quality
- Increase training epochs
- Try different batch sizes

**5. Low accuracy**
- Increase training epochs
- Add more training data
- Adjust image preprocessing
- Fine-tune hyperparameters

### Training Tips

1. **Start with pretrained model**: Always use pretrained weights for better results
2. **Monitor validation accuracy**: Stop if validation accuracy stops improving
3. **Data quality matters**: Clean, accurate labels are crucial
4. **Experiment with batch size**: Larger batches → more stable training
5. **Use GPU**: Training on CPU is extremely slow

### Log Files

- Training logs: Check terminal output or redirect to file
- Model checkpoints: `output/rec_model/`
- Predictions: `output/rec_model/predicts.txt`

## 📚 Resources

### PaddleOCR Documentation

- [Official Documentation](https://github.com/PaddlePaddle/PaddleOCR)
- [Model Zoo](https://github.com/PaddlePaddle/PaddleOCR/blob/main/doc/doc_en/models_list_en.md)
- [Training Guide](https://github.com/PaddlePaddle/PaddleOCR/blob/main/doc/doc_en/recognition_en.md)
- [Configuration Docs](https://github.com/PaddlePaddle/PaddleOCR/blob/main/doc/doc_en/config_en.md)

### Tutorials

- [Text Recognition Tutorial](https://github.com/PaddlePaddle/PaddleOCR/blob/main/doc/doc_en/recognition_en.md)
- [Custom Dataset Guide](https://github.com/PaddlePaddle/PaddleOCR/blob/main/doc/doc_en/dataset/recognition_dataset_en.md)

### Support

- [PaddleOCR Issues](https://github.com/PaddlePaddle/PaddleOCR/issues)
- [PaddlePaddle Forum](https://github.com/PaddlePaddle/Paddle/discussions)

## 📝 Dataset Format Examples

### Example 1: Simple Text Recognition

**labels.txt:**
```
receipt_001.jpg	$45.99
receipt_002.jpg	Total: $123.45
invoice_003.jpg	Invoice #12345
```

### Example 2: License Plates

**labels.txt:**
```
plate_001.jpg	ABC-1234
plate_002.jpg	XYZ-5678
plate_003.jpg	DEF-9012
```

### Example 3: Document Text

**labels.txt:**
```
doc_001.jpg	Machine Learning Research
doc_002.jpg	Annual Report 2023
doc_003.jpg	Project Proposal
```

## 🛠️ Advanced Usage

### Custom Architecture

Edit `configs/rec_custom.yml` to try different architectures:
- CRNN
- RARE
- SRN
- NRTR
- SAR

### Data Augmentation

The configuration includes `RecAug` for automatic augmentation:
- Random rotation
- Color jittering
- Gaussian noise
- Perspective transformation

Adjust in config or add custom augmentation in the pipeline.

### Multi-GPU Training

```bash
# In PaddleOCR directory
python -m paddle.distributed.launch \
    --gpus '0,1,2,3' \
    tools/train.py -c ../configs/rec_custom.yml
```

## 📄 License

This project structure is based on PaddleOCR, which is licensed under Apache License 2.0.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

---

**Happy Training! 🚀**

For questions or issues, please open an issue on GitHub.
