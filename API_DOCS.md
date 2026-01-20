# FastAPI OCR Web Service

This document describes the FastAPI web service for the OCR model.

## Running the Server

### Basic Usage

```bash
python app.py
```

This will start the server on `http://0.0.0.0:8000`

### Command Line Options

```bash
python app.py --help
```

Options:
- `--host`: Host to bind to (default: 0.0.0.0)
- `--port`: Port to bind to (default: 8000)
- `--model_dir`: Directory containing exported inference model (default: ./output/inference/)
- `--dict_path`: Path to custom character dictionary (default: ./dataset/dict.txt)
- `--cpu`: Force CPU inference (default: use GPU if available)
- `--reload`: Enable auto-reload for development

### Examples

**Run with custom port:**
```bash
python app.py --port 5000
```

**Run with CPU inference:**
```bash
python app.py --cpu
```

**Run in development mode with auto-reload:**
```bash
python app.py --reload
```

**Run with custom model path:**
```bash
python app.py --model_dir ./custom/model/ --dict_path ./custom/dict.txt
```

## Using the Web Interface

1. Open your browser and navigate to `http://localhost:8000`
2. Click "Choose File" or drag and drop an image
3. Optionally check "Show detailed results" for confidence scores
4. Click "Process Image" to extract text

## API Endpoints

### Health Check

**GET** `/health`

Check if the server and model are running properly.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### Single Image OCR

**POST** `/api/ocr`

Process a single image and extract text.

**Parameters:**
- `file` (form-data): Image file to process
- `detail` (query, optional): If true, returns detailed results with bounding boxes and confidence

**Example with curl:**
```bash
# Simple text extraction
curl -X POST "http://localhost:8000/api/ocr" \
  -F "file=@image.jpg"

# Detailed results with confidence scores
curl -X POST "http://localhost:8000/api/ocr?detail=true" \
  -F "file=@image.jpg"
```

**Response (simple):**
```json
{
  "success": true,
  "filename": "image.jpg",
  "text": "Extracted text from image"
}
```

**Response (detailed):**
```json
{
  "success": true,
  "filename": "image.jpg",
  "results": [
    {
      "bbox": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
      "text": "Extracted text",
      "confidence": 0.95
    }
  ]
}
```

### Batch Image OCR

**POST** `/api/ocr/batch`

Process multiple images at once (maximum 10 images per request).

**Parameters:**
- `files` (form-data): Multiple image files to process
- `detail` (query, optional): If true, returns detailed results

**Example with curl:**
```bash
curl -X POST "http://localhost:8000/api/ocr/batch" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "files=@image3.jpg"
```

**Response:**
```json
{
  "success": true,
  "count": 3,
  "results": [
    {
      "filename": "image1.jpg",
      "text": "Text from image 1"
    },
    {
      "filename": "image2.jpg",
      "text": "Text from image 2"
    },
    {
      "filename": "image3.jpg",
      "text": "Text from image 3"
    }
  ]
}
```

### Model Information

**GET** `/api/info`

Get information about the loaded OCR model.

**Response:**
```json
{
  "model_dir": "./output/inference/",
  "dict_path": "./dataset/dict.txt",
  "use_gpu": true,
  "lang": "en"
}
```

## Interactive API Documentation

FastAPI provides automatic interactive API documentation:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## Python Client Example

```python
import requests

# Single image OCR
def ocr_single_image(image_path):
    url = "http://localhost:8000/api/ocr"
    with open(image_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(url, files=files)
    return response.json()

# Batch OCR
def ocr_batch_images(image_paths):
    url = "http://localhost:8000/api/ocr/batch"
    files = [('files', open(path, 'rb')) for path in image_paths]
    response = requests.post(url, files=files)
    for f in files:
        f[1].close()
    return response.json()

# Usage
result = ocr_single_image('test.jpg')
print(f"Recognized text: {result['text']}")

batch_results = ocr_batch_images(['img1.jpg', 'img2.jpg'])
for item in batch_results['results']:
    print(f"{item['filename']}: {item['text']}")
```

## JavaScript/Browser Example

```javascript
async function uploadAndOCR(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await fetch('http://localhost:8000/api/ocr', {
        method: 'POST',
        body: formData
    });
    
    const result = await response.json();
    return result.text;
}

// Usage with file input
document.getElementById('fileInput').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    const text = await uploadAndOCR(file);
    console.log('Recognized text:', text);
});
```

## Environment Variables

You can configure the server using environment variables:

- `MODEL_DIR`: Path to the inference model directory
- `DICT_PATH`: Path to the character dictionary
- `USE_GPU`: Set to 'true' or 'false' to enable/disable GPU

Example:
```bash
export MODEL_DIR=./output/inference/
export DICT_PATH=./dataset/dict.txt
export USE_GPU=false
python app.py
```

## Deployment

### Using Gunicorn (Production)

```bash
pip install gunicorn
gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Using Docker

Create a `Dockerfile`:

```dockerfile
FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "app.py"]
```

Build and run:
```bash
docker build -t ocr-api .
docker run -p 8000:8000 ocr-api
```

## Troubleshooting

### Model Not Found Error

If you see "Model not found" error:
1. Ensure you have exported the trained model: `bash scripts/export.sh`
2. Check that `./output/inference/` contains `inference.pdmodel` and `inference.pdiparams`
3. Verify paths with `--model_dir` and `--dict_path` parameters

### GPU Not Available

If GPU is not available:
- Use `--cpu` flag to force CPU inference
- Check CUDA installation with `python -c "import paddle; print(paddle.device.is_compiled_with_cuda())"`

### Port Already in Use

If port 8000 is already in use:
- Use a different port: `python app.py --port 8080`
- Or stop the process using port 8000

## Performance Tips

1. **Use GPU**: GPU inference is significantly faster than CPU
2. **Batch Processing**: Use batch endpoint for multiple images
3. **Image Size**: Resize large images before upload to reduce processing time
4. **Workers**: Use multiple workers in production (with gunicorn)

## Security Considerations

1. **File Upload Size**: Consider adding file size limits in production
2. **File Type Validation**: Only accept image files
3. **Rate Limiting**: Implement rate limiting for public APIs
4. **HTTPS**: Use HTTPS in production environments
5. **Authentication**: Add authentication for sensitive deployments

## License

This API is based on PaddleOCR, licensed under Apache License 2.0.
