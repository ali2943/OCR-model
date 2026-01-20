#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Custom PaddleOCR Inference Script for Fine-tuned Model

This script provides a custom inference interface for PaddleOCR fine-tuned models,
supporting both single image and batch predictions.
"""

import os
from pathlib import Path
from typing import List, Union, Dict, Any
import warnings

# Suppress PaddleOCR warnings
warnings.filterwarnings('ignore')
os.environ['FLAGS_allocator_strategy'] = 'auto_growth'


class CustomPaddleOCR:
    """Custom PaddleOCR wrapper for inference with fine-tuned model"""
    
    def __init__(
        self,
        model_dir: str = './output/inference/',
        dict_path: str = './dataset/dict.txt',
        use_gpu: bool = True,
        use_angle_cls: bool = False,
        lang: str = 'en'
    ):
        """
        Initialize CustomPaddleOCR
        
        Args:
            model_dir: Directory containing exported inference model
            dict_path: Path to custom character dictionary
            use_gpu: Whether to use GPU for inference
            use_angle_cls: Whether to use angle classification
            lang: Language (default: 'en')
        """
        self.model_dir = Path(model_dir)
        self.dict_path = Path(dict_path)
        self.use_gpu = use_gpu
        self.use_angle_cls = use_angle_cls
        self.lang = lang
        self._ocr = None
        
        # Validate paths
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")
        
        if not self.dict_path.exists():
            raise FileNotFoundError(f"Dictionary file not found: {self.dict_path}")
        
        # Check for model files
        model_file = self.model_dir / 'inference.pdmodel'
        params_file = self.model_dir / 'inference.pdiparams'
        
        if not model_file.exists() or not params_file.exists():
            raise FileNotFoundError(
                f"Model files not found in {self.model_dir}\n"
                f"Expected: inference.pdmodel and inference.pdiparams\n"
                f"Run 'bash scripts/export.sh' to export the trained model first."
            )
    
    def _load_ocr(self):
        """Lazy load PaddleOCR to avoid import overhead"""
        if self._ocr is None:
            try:
                from paddleocr import PaddleOCR
            except ImportError:
                raise ImportError(
                    "PaddleOCR not installed. Install it with:\n"
                    "  pip install paddleocr>=2.7.0"
                )
            
            print(f"🚀 Loading PaddleOCR model from {self.model_dir}...")
            
            self._ocr = PaddleOCR(
                use_angle_cls=self.use_angle_cls,
                lang=self.lang,
                use_gpu=self.use_gpu,
                rec_model_dir=str(self.model_dir),
                rec_char_dict_path=str(self.dict_path),
                show_log=False
            )
            
            print("✅ Model loaded successfully!")
    
    def predict(
        self,
        image_path: Union[str, Path],
        detail: bool = False
    ) -> Union[str, List[Dict[str, Any]]]:
        """
        Predict text from a single image
        
        Args:
            image_path: Path to the image file
            detail: If True, returns detailed results with bounding boxes and confidence
                   If False, returns only the recognized text
        
        Returns:
            If detail=False: Recognized text as string
            If detail=True: List of dicts with 'bbox', 'text', and 'confidence'
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load OCR if not already loaded
        self._load_ocr()
        
        # Perform OCR
        results = self._ocr.ocr(str(image_path), cls=self.use_angle_cls)
        
        if not results or not results[0]:
            return "" if not detail else []
        
        # Parse results
        if detail:
            detailed_results = []
            for line in results[0]:
                bbox = line[0]
                text, confidence = line[1]
                detailed_results.append({
                    'bbox': bbox,
                    'text': text,
                    'confidence': confidence
                })
            return detailed_results
        else:
            # Return concatenated text
            texts = [line[1][0] for line in results[0]]
            return ' '.join(texts)
    
    def predict_batch(
        self,
        image_paths: List[Union[str, Path]],
        detail: bool = False
    ) -> List[Union[str, List[Dict[str, Any]]]]:
        """
        Predict text from multiple images
        
        Args:
            image_paths: List of paths to image files
            detail: If True, returns detailed results for each image
        
        Returns:
            List of predictions (text or detailed results) for each image
        """
        results = []
        
        print(f"📊 Processing {len(image_paths)} images...")
        
        for i, image_path in enumerate(image_paths, 1):
            try:
                result = self.predict(image_path, detail=detail)
                results.append(result)
                print(f"  [{i}/{len(image_paths)}] ✓ {Path(image_path).name}")
            except Exception as e:
                print(f"  [{i}/{len(image_paths)}] ✗ {Path(image_path).name}: {e}")
                results.append("" if not detail else [])
        
        print("✅ Batch processing completed!")
        return results


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='PaddleOCR Custom Inference')
    parser.add_argument('image', type=str, nargs='?',
                        help='Path to image file (optional for demo)')
    parser.add_argument('--model_dir', type=str, default='./output/inference/',
                        help='Directory containing exported inference model')
    parser.add_argument('--dict_path', type=str, default='./dataset/dict.txt',
                        help='Path to custom character dictionary')
    parser.add_argument('--use_gpu', action='store_true', default=True,
                        help='Use GPU for inference (default: True)')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU inference')
    parser.add_argument('--detail', action='store_true',
                        help='Show detailed results with bounding boxes and confidence')
    parser.add_argument('--batch', type=str, nargs='+',
                        help='Batch inference on multiple images')
    
    args = parser.parse_args()
    
    # Handle CPU flag
    use_gpu = args.use_gpu and not args.cpu
    
    # Initialize OCR
    try:
        ocr = CustomPaddleOCR(
            model_dir=args.model_dir,
            dict_path=args.dict_path,
            use_gpu=use_gpu
        )
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return
    
    # Example usage if no image provided
    if not args.image and not args.batch:
        print("\n" + "="*60)
        print("📖 PADDLEOCR CUSTOM INFERENCE - EXAMPLE USAGE")
        print("="*60)
        print("\nNo image provided. Here's how to use this script:\n")
        print("Single image inference:")
        print("  python inference/predict.py path/to/image.jpg")
        print("\nWith detailed output:")
        print("  python inference/predict.py path/to/image.jpg --detail")
        print("\nBatch inference:")
        print("  python inference/predict.py --batch img1.jpg img2.jpg img3.jpg")
        print("\nCPU inference:")
        print("  python inference/predict.py path/to/image.jpg --cpu")
        print("\nCustom paths:")
        print("  python inference/predict.py path/to/image.jpg \\")
        print("    --model_dir ./output/inference/ \\")
        print("    --dict_path ./dataset/dict.txt")
        print("="*60)
        return
    
    # Batch inference
    if args.batch:
        print(f"\n🔍 Running batch inference on {len(args.batch)} images...\n")
        results = ocr.predict_batch(args.batch, detail=args.detail)
        
        print("\n" + "="*60)
        print("📊 RESULTS")
        print("="*60)
        
        for image_path, result in zip(args.batch, results):
            print(f"\n📄 {Path(image_path).name}:")
            if args.detail:
                if result:
                    for i, item in enumerate(result, 1):
                        print(f"  [{i}] Text: {item['text']}")
                        print(f"      Confidence: {item['confidence']:.4f}")
                        print(f"      BBox: {item['bbox']}")
                else:
                    print("  No text detected")
            else:
                print(f"  {result if result else 'No text detected'}")
        
        print("="*60)
    
    # Single image inference
    elif args.image:
        print(f"\n🔍 Running inference on: {args.image}\n")
        result = ocr.predict(args.image, detail=args.detail)
        
        print("\n" + "="*60)
        print("📊 RESULT")
        print("="*60)
        
        if args.detail:
            if result:
                for i, item in enumerate(result, 1):
                    print(f"\n[{i}] Text: {item['text']}")
                    print(f"    Confidence: {item['confidence']:.4f}")
                    print(f"    BBox: {item['bbox']}")
            else:
                print("\nNo text detected")
        else:
            print(f"\nRecognized text: {result if result else 'No text detected'}")
        
        print("="*60)


if __name__ == '__main__':
    main()
