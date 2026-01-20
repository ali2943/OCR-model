#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example script demonstrating how to use the OCR API
"""

import requests
import sys
from pathlib import Path


def check_server_health(base_url="http://localhost:8000"):
    """Check if the OCR server is running"""
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Server is healthy")
            print(f"  Model loaded: {data.get('model_loaded', False)}")
            return True
        else:
            print(f"✗ Server returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"✗ Cannot connect to server at {base_url}")
        print(f"  Make sure the server is running: python app.py")
        return False
    except Exception as e:
        print(f"✗ Error checking server: {e}")
        return False


def ocr_single_image(image_path, base_url="http://localhost:8000", detail=False):
    """
    Process a single image with OCR
    
    Args:
        image_path: Path to the image file
        base_url: Base URL of the OCR API
        detail: If True, returns detailed results with confidence scores
    
    Returns:
        dict: OCR results
    """
    url = f"{base_url}/api/ocr"
    params = {"detail": str(detail).lower()}
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(url, files=files, params=params)
        
        if response.status_code == 200:
            return response.json()
        else:
            error = response.json()
            print(f"Error: {error.get('detail', 'Unknown error')}")
            return None
    except FileNotFoundError:
        print(f"Error: File not found: {image_path}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None


def ocr_batch_images(image_paths, base_url="http://localhost:8000", detail=False):
    """
    Process multiple images with OCR
    
    Args:
        image_paths: List of paths to image files
        base_url: Base URL of the OCR API
        detail: If True, returns detailed results with confidence scores
    
    Returns:
        dict: OCR results for all images
    """
    url = f"{base_url}/api/ocr/batch"
    params = {"detail": str(detail).lower()}
    
    try:
        files = [('files', open(path, 'rb')) for path in image_paths]
        response = requests.post(url, files=files, params=params)
        
        # Close all file handles
        for _, f in files:
            f.close()
        
        if response.status_code == 200:
            return response.json()
        else:
            error = response.json()
            print(f"Error: {error.get('detail', 'Unknown error')}")
            return None
    except FileNotFoundError as e:
        print(f"Error: File not found: {e}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None


def get_model_info(base_url="http://localhost:8000"):
    """Get information about the loaded OCR model"""
    try:
        response = requests.get(f"{base_url}/api/info")
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error getting model info: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error: {e}")
        return None


def main():
    """Example usage"""
    print("="*60)
    print("OCR API Client Example")
    print("="*60)
    print()
    
    # Check server health
    print("1. Checking server health...")
    if not check_server_health():
        sys.exit(1)
    print()
    
    # Get model info
    print("2. Getting model information...")
    info = get_model_info()
    if info:
        print(f"  Model directory: {info.get('model_dir')}")
        print(f"  Dictionary path: {info.get('dict_path')}")
        print(f"  Using GPU: {info.get('use_gpu')}")
        print(f"  Language: {info.get('lang')}")
    print()
    
    # Check if user provided image path
    if len(sys.argv) < 2:
        print("3. Example Usage:")
        print()
        print("To process a single image:")
        print(f"  python {sys.argv[0]} path/to/image.jpg")
        print()
        print("To process multiple images:")
        print(f"  python {sys.argv[0]} image1.jpg image2.jpg image3.jpg")
        print()
        print("="*60)
        return
    
    image_paths = sys.argv[1:]
    
    # Single image
    if len(image_paths) == 1:
        print(f"3. Processing single image: {image_paths[0]}")
        
        # Simple OCR
        result = ocr_single_image(image_paths[0], detail=False)
        if result and result.get('success'):
            print(f"  Recognized text: {result.get('text', 'No text')}")
        
        # Detailed OCR
        print()
        print("4. Processing with detailed results...")
        result = ocr_single_image(image_paths[0], detail=True)
        if result and result.get('success'):
            results = result.get('results', [])
            if results:
                print(f"  Found {len(results)} text region(s):")
                for i, item in enumerate(results, 1):
                    print(f"    [{i}] Text: {item.get('text', 'No text')}")
                    print(f"        Confidence: {item.get('confidence', 0)*100:.2f}%")
            else:
                print("  No text detected")
    
    # Batch processing
    else:
        print(f"3. Processing {len(image_paths)} images in batch...")
        result = ocr_batch_images(image_paths, detail=False)
        if result and result.get('success'):
            print(f"  Processed {result.get('count', 0)} images:")
            for item in result.get('results', []):
                print(f"    - {item.get('filename')}: {item.get('text', 'No text')}")
    
    print()
    print("="*60)


if __name__ == '__main__':
    main()
