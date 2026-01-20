#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI Application for OCR Model
Provides REST API endpoints for text recognition using fine-tuned PaddleOCR model
"""

import os
import io
import sys
import uuid
from pathlib import Path
from typing import List, Optional
import warnings

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import uvicorn

# Add inference module to path
sys.path.insert(0, str(Path(__file__).parent))

from inference.predict import CustomPaddleOCR

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['FLAGS_allocator_strategy'] = 'auto_growth'

# Initialize FastAPI app
app = FastAPI(
    title="OCR API",
    description="REST API for text recognition using fine-tuned PaddleOCR model",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global OCR model instance
ocr_model: Optional[CustomPaddleOCR] = None


def get_ocr_model() -> CustomPaddleOCR:
    """Get or initialize the OCR model (singleton pattern)"""
    global ocr_model
    if ocr_model is None:
        model_dir = os.getenv('MODEL_DIR', './output/inference/')
        dict_path = os.getenv('DICT_PATH', './dataset/dict.txt')
        use_gpu = os.getenv('USE_GPU', 'true').lower() == 'true'
        
        try:
            ocr_model = CustomPaddleOCR(
                model_dir=model_dir,
                dict_path=dict_path,
                use_gpu=use_gpu
            )
        except FileNotFoundError as e:
            raise HTTPException(
                status_code=500,
                detail=f"Model not found: {str(e)}. Please ensure the model is exported to {model_dir}"
            )
    return ocr_model


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main HTML page"""
    html_file = Path(__file__).parent / "static" / "index.html"
    if html_file.exists():
        return html_file.read_text()
    
    # Return a basic HTML page if static file doesn't exist
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>OCR API</title>
    </head>
    <body>
        <h1>OCR API is running!</h1>
        <p>Visit <a href="/docs">/docs</a> for API documentation.</p>
    </body>
    </html>
    """


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": ocr_model is not None
    }


@app.post("/api/ocr")
async def ocr_endpoint(
    file: UploadFile = File(...),
    detail: bool = False
):
    """
    Perform OCR on uploaded image
    
    Args:
        file: Image file to process (jpg, png, etc.)
        detail: If true, returns detailed results with bounding boxes and confidence
    
    Returns:
        JSON response with recognized text
    """
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Please upload an image file."
        )
    
    try:
        # Read image file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Save temporarily for processing using secure UUID filename
        file_extension = Path(file.filename).suffix if file.filename else '.jpg'
        safe_filename = f"{uuid.uuid4()}{file_extension}"
        temp_path = Path("/tmp") / safe_filename
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(temp_path)
        
        # Get OCR model and perform prediction
        ocr = get_ocr_model()
        result = ocr.predict(str(temp_path), detail=detail)
        
        # Clean up temporary file
        temp_path.unlink(missing_ok=True)
        
        # Return results
        if detail:
            return {
                "success": True,
                "filename": file.filename,
                "results": result
            }
        else:
            return {
                "success": True,
                "filename": file.filename,
                "text": result
            }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"OCR processing failed: {str(e)}"
        )


@app.post("/api/ocr/batch")
async def ocr_batch_endpoint(
    files: List[UploadFile] = File(...),
    detail: bool = False
):
    """
    Perform OCR on multiple uploaded images
    
    Args:
        files: List of image files to process
        detail: If true, returns detailed results with bounding boxes and confidence
    
    Returns:
        JSON response with recognized text for each image
    """
    if len(files) == 0:
        raise HTTPException(
            status_code=400,
            detail="No files provided"
        )
    
    if len(files) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 files allowed per batch request"
        )
    
    results = []
    temp_paths = []
    
    try:
        # Save all uploaded files temporarily
        for file in files:
            if not file.content_type.startswith('image/'):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid file type for {file.filename}: {file.content_type}"
                )
            
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            
            # Use secure UUID filename to prevent path traversal
            file_extension = Path(file.filename).suffix if file.filename else '.jpg'
            safe_filename = f"{uuid.uuid4()}{file_extension}"
            temp_path = Path("/tmp") / safe_filename
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(temp_path)
            temp_paths.append(temp_path)
        
        # Get OCR model and perform batch prediction
        ocr = get_ocr_model()
        predictions = ocr.predict_batch([str(p) for p in temp_paths], detail=detail)
        
        # Format results
        for file, prediction in zip(files, predictions):
            if detail:
                results.append({
                    "filename": file.filename,
                    "results": prediction
                })
            else:
                results.append({
                    "filename": file.filename,
                    "text": prediction
                })
        
        return {
            "success": True,
            "count": len(results),
            "results": results
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch OCR processing failed: {str(e)}"
        )
    
    finally:
        # Clean up temporary files
        for temp_path in temp_paths:
            temp_path.unlink(missing_ok=True)


@app.get("/api/info")
async def model_info():
    """Get information about the loaded OCR model"""
    ocr = get_ocr_model()
    
    return {
        "model_dir": str(ocr.model_dir),
        "dict_path": str(ocr.dict_path),
        "use_gpu": ocr.use_gpu,
        "lang": ocr.lang
    }


# Mount static files if directory exists
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


def main():
    """Run the FastAPI application"""
    import argparse
    
    parser = argparse.ArgumentParser(description='OCR FastAPI Server')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8000,
                        help='Port to bind to (default: 8000)')
    parser.add_argument('--model_dir', type=str, default='./output/inference/',
                        help='Directory containing exported inference model')
    parser.add_argument('--dict_path', type=str, default='./dataset/dict.txt',
                        help='Path to custom character dictionary')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU inference (default: use GPU if available)')
    parser.add_argument('--reload', action='store_true',
                        help='Enable auto-reload for development')
    
    args = parser.parse_args()
    
    # Set environment variables
    os.environ['MODEL_DIR'] = args.model_dir
    os.environ['DICT_PATH'] = args.dict_path
    os.environ['USE_GPU'] = str(not args.cpu).lower()
    
    # Run server
    uvicorn.run(
        "app:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == '__main__':
    main()
