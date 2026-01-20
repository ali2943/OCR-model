# OCR Web Service - Quick Reference Guide

## Getting Started in 3 Steps

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Start the Server
```bash
python app.py
```

### Step 3: Open in Browser
Navigate to: **http://localhost:8000**

---

## Web Interface Features

### Main Page (http://localhost:8000)
The web interface provides:

1. **Upload Section**
   - Drag & drop zone for images
   - "Choose File" button
   - Supports: JPG, PNG, GIF, BMP, TIFF, WEBP

2. **Options**
   - ☑ Show detailed results (with confidence scores)

3. **Results Display**
   - Extracted text shown in clean format
   - Optional: confidence scores and bounding boxes
   - Image preview

### API Documentation (http://localhost:8000/docs)
Interactive Swagger UI showing:
- All available endpoints
- Request/response schemas
- "Try it out" functionality
- Example payloads

---

## API Quick Reference

### 1. Health Check
```bash
GET /health
```
Returns server status and model loading status.

### 2. Single Image OCR
```bash
POST /api/ocr
Form-data: file (image file)
Query param: detail (boolean, optional)
```

Example:
```bash
curl -X POST "http://localhost:8000/api/ocr" -F "file=@receipt.jpg"
```

Response:
```json
{
  "success": true,
  "filename": "receipt.jpg",
  "text": "Total: $45.99"
}
```

### 3. Batch OCR (Max 10 images)
```bash
POST /api/ocr/batch
Form-data: files (multiple image files)
Query param: detail (boolean, optional)
```

Example:
```bash
curl -X POST "http://localhost:8000/api/ocr/batch" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg"
```

### 4. Model Info
```bash
GET /api/info
```
Returns model configuration and paths.

---

## Python Client Usage

### Basic Example
```python
from example_api_client import ocr_single_image

result = ocr_single_image('test.jpg')
print(f"Text: {result['text']}")
```

### Advanced Example
```python
from example_api_client import ocr_single_image

# Get detailed results with confidence
result = ocr_single_image('test.jpg', detail=True)
if result and result['success']:
    for item in result['results']:
        print(f"Text: {item['text']}")
        print(f"Confidence: {item['confidence']:.2%}")
```

---

## Server Configuration

### Environment Variables
```bash
export MODEL_DIR=./output/inference/
export DICT_PATH=./dataset/dict.txt
export USE_GPU=true
python app.py
```

### Command Line Arguments
```bash
# Custom port
python app.py --port 5000

# CPU only
python app.py --cpu

# Custom paths
python app.py \
  --model_dir ./custom/model/ \
  --dict_path ./custom/dict.txt

# Development mode (auto-reload)
python app.py --reload
```

---

## Troubleshooting

### Issue: Model not found
**Solution**: Export your trained model first
```bash
bash scripts/export.sh
```

### Issue: Port already in use
**Solution**: Use a different port
```bash
python app.py --port 8080
```

### Issue: GPU not available
**Solution**: Use CPU mode
```bash
python app.py --cpu
```

### Issue: Dependencies missing
**Solution**: Reinstall requirements
```bash
pip install -r requirements.txt
```

---

## Common Workflows

### 1. Process a single image via web
1. Open http://localhost:8000
2. Click "Choose File" or drag image
3. Click "Process Image"
4. View results

### 2. Process images via API
```bash
# Single image
curl -X POST "http://localhost:8000/api/ocr" -F "file=@image.jpg"

# Multiple images
curl -X POST "http://localhost:8000/api/ocr/batch" \
  -F "files=@img1.jpg" -F "files=@img2.jpg"
```

### 3. Integrate into Python app
```python
import requests

def extract_text(image_path):
    url = "http://localhost:8000/api/ocr"
    with open(image_path, 'rb') as f:
        response = requests.post(url, files={'file': f})
    return response.json()['text']

text = extract_text('document.jpg')
print(text)
```

### 4. Integrate into JavaScript/Web app
```javascript
async function uploadImage(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await fetch('http://localhost:8000/api/ocr', {
        method: 'POST',
        body: formData
    });
    
    const result = await response.json();
    return result.text;
}
```

---

## Performance Tips

1. **Use GPU**: 10-100x faster than CPU
2. **Batch Processing**: Process multiple images in one request
3. **Image Size**: Resize large images before upload
4. **Keep Server Running**: Avoid model reloading overhead

---

## Security Best Practices

For production deployments:

1. ✅ Use HTTPS (reverse proxy with nginx/Apache)
2. ✅ Add authentication (API keys, OAuth)
3. ✅ Implement rate limiting
4. ✅ Set file size limits
5. ✅ Use firewall rules
6. ✅ Monitor server logs
7. ✅ Regular security updates

---

## Support & Documentation

- **Full API Docs**: See `API_DOCS.md`
- **Implementation Details**: See `IMPLEMENTATION_SUMMARY.md`
- **General Guide**: See `README.md`
- **Interactive API Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

---

## Example Use Cases

### Receipt OCR
```bash
curl -X POST "http://localhost:8000/api/ocr" -F "file=@receipt.jpg"
```
Extract text from receipts for expense tracking

### Document Digitization
```bash
python example_api_client.py document1.pdf document2.pdf
```
Convert scanned documents to searchable text

### License Plate Recognition
```bash
curl -X POST "http://localhost:8000/api/ocr?detail=true" -F "file=@plate.jpg"
```
Extract license plate numbers with confidence scores

### Business Card Scanner
```bash
curl -X POST "http://localhost:8000/api/ocr" -F "file=@business_card.jpg"
```
Extract contact information from business cards

---

**Ready to start? Run:** `bash start_web_service.sh`
