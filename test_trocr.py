#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TrOCR Testing Script

Tests the trained TrOCR model on the test dataset.

Usage:
    python test_trocr.py [--model_dir ./trocr_model/final] [--limit 50]
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from tqdm import tqdm
import evaluate


def load_test_dataset(dataset_dir: Path) -> Tuple[List[str], List[str]]:
    """
    Load test dataset from dataset directory
    
    Args:
        dataset_dir: Path to dataset directory
    
    Returns:
        Tuple of (image_paths, labels)
    """
    test_dir = dataset_dir / 'test'
    
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")
    
    # Find test list file (handle potential naming variations)
    list_files = [
        test_dir / 'test_list.txt',
    ]
    
    list_file = None
    for f in list_files:
        if f.exists():
            list_file = f
            break
    
    if not list_file:
        raise FileNotFoundError(f"Could not find test_list.txt in {test_dir}")
    
    image_paths = []
    labels = []
    
    with open(list_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('\t')
            if len(parts) != 2:
                continue
            
            rel_image_path, label = parts
            
            # Construct absolute image path
            if rel_image_path.startswith('images/'):
                image_path = test_dir / rel_image_path
            else:
                image_path = test_dir / 'images' / rel_image_path
            
            if image_path.exists():
                image_paths.append(str(image_path))
                labels.append(label)
    
    return image_paths, labels


def predict_batch(
    model,
    processor,
    image_paths: List[str],
    device: torch.device,
    batch_size: int = 16
) -> List[str]:
    """
    Predict text from images in batches
    
    Args:
        model: TrOCR model
        processor: TrOCR processor
        image_paths: List of image paths
        device: Device to run inference on
        batch_size: Batch size for inference
    
    Returns:
        List of predicted texts
    """
    predictions = []
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        
        # Load images
        images = []
        for img_path in batch_paths:
            try:
                img = Image.open(img_path).convert('RGB')
                images.append(img)
            except Exception as e:
                print(f"⚠️  Error loading {img_path}: {e}")
                images.append(Image.new('RGB', (100, 32)))
        
        # Process images
        pixel_values = processor(images, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)
        
        # Generate predictions
        with torch.no_grad():
            generated_ids = model.generate(pixel_values)
        
        # Decode predictions
        generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
        predictions.extend(generated_texts)
    
    return predictions


def compute_accuracy(predictions: List[str], labels: List[str]) -> Dict[str, float]:
    """
    Compute accuracy metrics
    
    Args:
        predictions: List of predicted texts
        labels: List of ground truth labels
    
    Returns:
        Dictionary of metrics
    """
    # Exact match accuracy
    correct = sum(1 for pred, label in zip(predictions, labels) if pred == label)
    accuracy = correct / len(labels) * 100 if labels else 0
    
    # Character Error Rate (CER)
    cer_metric = evaluate.load("cer")
    cer = cer_metric.compute(predictions=predictions, references=labels)
    
    return {
        'accuracy': accuracy,
        'cer': cer,
        'correct': correct,
        'total': len(labels)
    }


def main():
    """Main testing function"""
    parser = argparse.ArgumentParser(description='Test TrOCR model')
    parser.add_argument('--model_dir', type=str, default='./trocr_model/final',
                        help='Directory containing trained model')
    parser.add_argument('--dataset_dir', type=str, default='./dataset',
                        help='Directory containing dataset')
    parser.add_argument('--limit', type=int, default=50,
                        help='Number of images to test (default: 50)')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU inference')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for inference')
    parser.add_argument('--verbose', action='store_true',
                        help='Show per-image results')
    
    args = parser.parse_args()
    
    print("="*70)
    print("🧪 TrOCR MODEL TESTING")
    print("="*70)
    
    # Setup device
    if args.cpu:
        device = torch.device('cpu')
        print("🖥️  Using CPU (forced)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("🖥️  Using CPU (GPU not available)")
    
    # Check model directory
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"\n❌ Error: Model directory not found: {model_dir}")
        print(f"   Please train the model first: python train_trocr.py")
        sys.exit(1)
    
    # Load model and processor
    print(f"\n🔧 Loading model from {model_dir}...")
    
    try:
        processor = TrOCRProcessor.from_pretrained(str(model_dir))
        model = VisionEncoderDecoderModel.from_pretrained(str(model_dir))
        model.to(device)
        model.eval()
        print(f"   ✅ Model loaded successfully")
    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        sys.exit(1)
    
    # Load test dataset
    dataset_dir = Path(args.dataset_dir)
    print(f"\n📂 Loading test dataset from {dataset_dir}...")
    
    try:
        image_paths, labels = load_test_dataset(dataset_dir)
        print(f"   ✅ Found {len(image_paths)} test images")
    except Exception as e:
        print(f"\n❌ Error loading dataset: {e}")
        sys.exit(1)
    
    # Limit number of images if specified
    if args.limit and args.limit < len(image_paths):
        image_paths = image_paths[:args.limit]
        labels = labels[:args.limit]
        print(f"   ℹ️  Testing limited to first {args.limit} images")
    
    # Run predictions
    print(f"\n🔍 Running predictions on {len(image_paths)} images...")
    print(f"   Batch size: {args.batch_size}")
    
    try:
        predictions = []
        
        with tqdm(total=len(image_paths), desc="Testing") as pbar:
            for i in range(0, len(image_paths), args.batch_size):
                batch_paths = image_paths[i:i + args.batch_size]
                batch_preds = predict_batch(
                    model, processor, batch_paths, device, len(batch_paths)
                )
                predictions.extend(batch_preds)
                pbar.update(len(batch_paths))
        
        print(f"   ✅ Predictions completed")
    except Exception as e:
        print(f"\n❌ Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Compute metrics
    print(f"\n📊 Computing metrics...")
    metrics = compute_accuracy(predictions, labels)
    
    # Print results
    print("\n" + "="*70)
    print("📊 TEST RESULTS")
    print("="*70)
    print(f"Total images tested: {metrics['total']}")
    print(f"Correct predictions: {metrics['correct']}")
    print(f"Accuracy: {metrics['accuracy']:.2f}%")
    print(f"Character Error Rate (CER): {metrics['cer']:.4f}")
    print("="*70)
    
    # Show per-image results if verbose
    if args.verbose:
        print("\n" + "="*70)
        print("📝 PER-IMAGE RESULTS")
        print("="*70)
        
        for i, (img_path, pred, label) in enumerate(zip(image_paths, predictions, labels), 1):
            img_name = Path(img_path).name
            status = "✅" if pred == label else "❌"
            print(f"\n[{i}/{len(image_paths)}] {status} {img_name}")
            print(f"   Predicted: '{pred}'")
            print(f"   Ground Truth: '{label}'")
            
            if i >= 20 and not args.verbose:  # Limit output
                remaining = len(image_paths) - i
                if remaining > 0:
                    print(f"\n... ({remaining} more images)")
                break
        
        print("="*70)
    
    # Summary
    print(f"\n✅ Testing completed!")
    print(f"\nAccuracy: {metrics['accuracy']:.2f}% ({metrics['correct']}/{metrics['total']})")
    print(f"CER: {metrics['cer']:.4f}")
    
    # Success criteria check
    if metrics['accuracy'] >= 85:
        print(f"\n🎉 SUCCESS: Accuracy meets target (>85%)")
    else:
        print(f"\n⚠️  Note: Accuracy below target (85%). Consider training longer.")
    
    print("="*70)


if __name__ == '__main__':
    main()
