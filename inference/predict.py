#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Custom OCR Inference Script for Fine-tuned Model

This script provides a custom inference interface for TrOCR fine-tuned models,
supporting both single image and batch predictions.
"""

import os
import sys
from pathlib import Path
from typing import List, Union, Dict, Any
import warnings

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Suppress warnings
warnings.filterwarnings('ignore')


class CustomTrOCR:
    """
    Custom OCR wrapper for inference with fine-tuned TrOCR model
    
    Note: The confidence scores returned in detailed mode are placeholders (0.95)
    for API compatibility. TrOCR's generate() method doesn't provide direct
    confidence scores. For actual confidence, you would need to use model.forward()
    and compute token probabilities.
    """
    
    def __init__(
        self,
        model_dir: str = './model',
        use_gpu: bool = True
    ):
        """
        Initialize CustomTrOCR
        
        Args:
            model_dir: Directory containing trained model
            use_gpu: Whether to use GPU for inference
        """
        self.model_dir = Path(model_dir)
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
        self._processor = None
        self._model = None
        
        # Validate paths
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")
        
        # Check for model files
        config_file = self.model_dir / 'config.json'
        
        if not config_file.exists():
            raise FileNotFoundError(
                f"Model files not found in {self.model_dir}\n"
                f"Expected: config.json and pytorch_model.bin\n"
                f"Run 'python train_trocr.py' to train the model first."
            )
    
    def _load_model(self):
        """Lazy load TrOCR model to avoid import overhead"""
        if self._model is None:
            print(f"🚀 Loading TrOCR model from {self.model_dir}...")
            
            try:
                self._processor = TrOCRProcessor.from_pretrained(str(self.model_dir))
                self._model = VisionEncoderDecoderModel.from_pretrained(str(self.model_dir))
                self._model.to(self.device)
                self._model.eval()
                
                device_name = torch.cuda.get_device_name(0) if self.use_gpu else "CPU"
                print(f"✅ Model loaded successfully on {device_name}!")
            except Exception as e:
                raise RuntimeError(f"Failed to load model: {e}")
    
    def predict(
        self,
        image_path: Union[str, Path],
        detail: bool = False
    ) -> Union[str, Dict[str, Any]]:
        """
        Predict text from a single image
        
        Args:
            image_path: Path to the image file
            detail: If True, returns detailed results with confidence
                   If False, returns only the recognized text
        
        Returns:
            If detail=False: Recognized text as string
            If detail=True: Dict with 'text' and 'confidence'
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load model if not already loaded
        self._load_model()
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            
            # Process image
            pixel_values = self._processor(image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.device)
            
            # Generate prediction
            with torch.no_grad():
                generated_ids = self._model.generate(pixel_values)
            
            # Decode prediction
            generated_text = self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            if detail:
                # Note: TrOCR doesn't provide direct confidence scores in generate() mode.
                # For actual confidence, you would need to use model.forward() and compute
                # token probabilities. Using 0.95 as a placeholder for API compatibility.
                return {
                    'text': generated_text,
                    'confidence': 0.95  # Placeholder - not actual model confidence
                }
            else:
                return generated_text
        
        except Exception as e:
            print(f"⚠️  Error processing {image_path.name}: {e}")
            if detail:
                return {'text': '', 'confidence': 0.0}
            else:
                return ''
    
    def predict_batch(
        self,
        image_paths: List[Union[str, Path]],
        detail: bool = False,
        batch_size: int = 16
    ) -> List[Union[str, Dict[str, Any]]]:
        """
        Predict text from multiple images
        
        Args:
            image_paths: List of paths to image files
            detail: If True, returns detailed results for each image
            batch_size: Batch size for processing
        
        Returns:
            List of predictions (text or detailed results) for each image
        """
        results = []
        
        print(f"📊 Processing {len(image_paths)} images...")
        
        # Load model if not already loaded
        self._load_model()
        
        # Process in batches
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            
            # Load images
            images = []
            valid_indices = []
            
            for j, img_path in enumerate(batch_paths):
                try:
                    img = Image.open(img_path).convert('RGB')
                    images.append(img)
                    valid_indices.append(j)
                except Exception as e:
                    print(f"  [{i+j+1}/{len(image_paths)}] ✗ {Path(img_path).name}: {e}")
                    if detail:
                        results.append({'text': '', 'confidence': 0.0})
                    else:
                        results.append('')
            
            if not images:
                continue
            
            try:
                # Process images
                pixel_values = self._processor(images, return_tensors="pt").pixel_values
                pixel_values = pixel_values.to(self.device)
                
                # Generate predictions
                with torch.no_grad():
                    generated_ids = self._model.generate(pixel_values)
                
                # Decode predictions
                generated_texts = self._processor.batch_decode(generated_ids, skip_special_tokens=True)
                
                # Add results
                result_idx = 0
                for j, img_path in enumerate(batch_paths):
                    if j in valid_indices:
                        text = generated_texts[result_idx]
                        result_idx += 1
                        
                        if detail:
                            # Note: Confidence is a placeholder (0.95) for API compatibility
                            # TrOCR's generate() doesn't provide direct confidence scores
                            results.append({'text': text, 'confidence': 0.95})
                        else:
                            results.append(text)
                        
                        print(f"  [{i+j+1}/{len(image_paths)}] ✓ {Path(img_path).name}: '{text}'")
            
            except Exception as e:
                print(f"  ✗ Batch processing error: {e}")
                for j in range(len(batch_paths)):
                    if j not in valid_indices:
                        continue
                    if detail:
                        results.append({'text': '', 'confidence': 0.0})
                    else:
                        results.append('')
        
        print("✅ Batch processing completed!")
        return results


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Custom OCR Inference')
    parser.add_argument('image', type=str, nargs='?',
                        help='Path to image file (optional for demo)')
    parser.add_argument('--model_dir', type=str, default='./model',
                        help='Directory containing trained model')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU inference')
    parser.add_argument('--detail', action='store_true',
                        help='Show detailed results with confidence')
    parser.add_argument('--batch', type=str, nargs='+',
                        help='Batch inference on multiple images')
    
    args = parser.parse_args()
    
    # Handle CPU flag
    use_gpu = not args.cpu
    
    # Initialize OCR
    try:
        ocr = CustomTrOCR(
            model_dir=args.model_dir,
            use_gpu=use_gpu
        )
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return
    
    # Example usage if no image provided
    if not args.image and not args.batch:
        print("\n" + "="*60)
        print("📖 TrOCR CUSTOM INFERENCE - EXAMPLE USAGE")
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
        print("\nCustom model directory:")
        print("  python inference/predict.py path/to/image.jpg \\")
        print("    --model_dir ./model")
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
                print(f"  Text: {result['text'] if result['text'] else 'No text detected'}")
                print(f"  Confidence: {result['confidence']:.4f}")
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
            print(f"\nText: {result['text'] if result['text'] else 'No text detected'}")
            print(f"Confidence: {result['confidence']:.4f}")
        else:
            print(f"\nRecognized text: {result if result else 'No text detected'}")
        
        print("="*60)


if __name__ == '__main__':
    main()
