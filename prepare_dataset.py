#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR Dataset Preparation Script

Converts existing dataset to HuggingFace Dataset format compatible with TrOCR.

Usage:
    python prepare_dataset.py
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image
from datasets import Dataset, DatasetDict
import json


def load_dataset_split(split_dir: Path, split_name: str) -> Tuple[List[str], List[str], List[Dict]]:
    """
    Load a dataset split (train/val/test)
    
    Args:
        split_dir: Path to the split directory
        split_name: Name of the split ('train', 'val', 'test')
    
    Returns:
        Tuple of (image_paths, labels, statistics)
    """
    print(f"\n📂 Processing {split_name} split...")
    
    # Find the list file
    list_files = [
        split_dir / f"{split_name}_list.txt",
    ]
    
    list_file = None
    for f in list_files:
        if f.exists():
            list_file = f
            break
    
    if not list_file:
        raise FileNotFoundError(f"Could not find {split_name}_list.txt in {split_dir}")
    
    print(f"   Reading from: {list_file}")
    
    image_paths = []
    labels = []
    skipped = 0
    corrupted = 0
    
    with open(list_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            # Parse line: image_path\tlabel
            parts = line.split('\t')
            if len(parts) != 2:
                print(f"   ⚠️  Line {line_num}: Invalid format (expected 2 parts, got {len(parts)})")
                skipped += 1
                continue
            
            rel_image_path, label = parts
            
            # Construct absolute image path
            # Handle both 'images/file.png' and 'file.png' formats
            if rel_image_path.startswith('images/'):
                image_path = split_dir / rel_image_path
            else:
                image_path = split_dir / 'images' / rel_image_path
            
            # Check if image exists and is valid
            if not image_path.exists():
                print(f"   ⚠️  Line {line_num}: Image not found: {image_path}")
                skipped += 1
                continue
            
            try:
                # Try to open and validate the image
                with Image.open(image_path) as img:
                    img.verify()
                
                # Reopen for actual use (verify() closes the file)
                img = Image.open(image_path)
                
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                image_paths.append(str(image_path))
                labels.append(label)
                
            except Exception as e:
                print(f"   ⚠️  Line {line_num}: Corrupted image {image_path.name}: {e}")
                corrupted += 1
                continue
    
    # Statistics
    stats = {
        'total_lines': line_num if 'line_num' in locals() else 0,
        'valid_samples': len(image_paths),
        'skipped': skipped,
        'corrupted': corrupted
    }
    
    print(f"   ✅ Valid samples: {stats['valid_samples']}")
    if stats['skipped'] > 0:
        print(f"   ⚠️  Skipped (invalid format): {stats['skipped']}")
    if stats['corrupted'] > 0:
        print(f"   ⚠️  Corrupted images: {stats['corrupted']}")
    
    return image_paths, labels, stats


def create_huggingface_dataset(image_paths: List[str], labels: List[str]) -> Dataset:
    """
    Create a HuggingFace Dataset from image paths and labels
    
    Args:
        image_paths: List of paths to images
        labels: List of text labels
    
    Returns:
        HuggingFace Dataset
    """
    data = {
        'image': [],
        'text': labels,
    }
    
    # Load images
    print(f"   Loading {len(image_paths)} images...")
    for i, img_path in enumerate(image_paths):
        try:
            img = Image.open(img_path).convert('RGB')
            data['image'].append(img)
            
            if (i + 1) % 100 == 0:
                print(f"   Progress: {i + 1}/{len(image_paths)}")
        except Exception as e:
            print(f"   ⚠️  Error loading {img_path}: {e}")
            # Add a blank image as placeholder (should not happen as we validated earlier)
            data['image'].append(Image.new('RGB', (100, 32)))
    
    # Create dataset
    dataset = Dataset.from_dict(data)
    return dataset


def main():
    """Main function"""
    print("="*70)
    print("🚀 OCR DATASET PREPARATION")
    print("="*70)
    
    # Paths
    base_dir = Path(__file__).parent
    dataset_dir = base_dir / 'dataset'
    output_dir = base_dir / 'dataset_processed'
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Dataset directory: {dataset_dir}")
    print(f"📁 Output directory: {output_dir}")
    
    # Check if dataset directory exists
    if not dataset_dir.exists():
        print(f"\n❌ Error: Dataset directory not found: {dataset_dir}")
        sys.exit(1)
    
    # Process each split
    dataset_dict = {}
    all_stats = {}
    
    splits = {
        'train': dataset_dir / 'train',
        'validation': dataset_dir / 'val',
        'test': dataset_dir / 'test'
    }
    
    for split_key, split_dir in splits.items():
        if not split_dir.exists():
            print(f"\n⚠️  Warning: {split_key} directory not found: {split_dir}")
            continue
        
        # Determine the actual split name for file lookup
        if split_key == 'validation':
            file_split_name = 'val'
        else:
            file_split_name = split_key
        
        try:
            # Load dataset split
            image_paths, labels, stats = load_dataset_split(split_dir, file_split_name)
            all_stats[split_key] = stats
            
            if len(image_paths) == 0:
                print(f"   ⚠️  No valid samples found in {split_key} split, skipping...")
                continue
            
            # Create HuggingFace dataset
            print(f"   Creating HuggingFace dataset...")
            dataset = create_huggingface_dataset(image_paths, labels)
            dataset_dict[split_key] = dataset
            
            print(f"   ✅ {split_key} dataset created with {len(dataset)} samples")
            
        except FileNotFoundError as e:
            print(f"   ⚠️  {e}")
            continue
        except Exception as e:
            print(f"   ❌ Error processing {split_key}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not dataset_dict:
        print("\n❌ Error: No datasets were created successfully")
        sys.exit(1)
    
    # Create DatasetDict
    print(f"\n💾 Saving datasets to {output_dir}...")
    datasets = DatasetDict(dataset_dict)
    
    # Save to disk
    datasets.save_to_disk(str(output_dir))
    
    # Save statistics
    stats_file = output_dir / 'statistics.json'
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(all_stats, f, indent=2)
    
    print(f"   ✅ Datasets saved successfully!")
    print(f"   ✅ Statistics saved to {stats_file}")
    
    # Print summary
    print("\n" + "="*70)
    print("📊 SUMMARY")
    print("="*70)
    
    for split_key, stats in all_stats.items():
        print(f"\n{split_key.upper()}:")
        print(f"   Total lines: {stats['total_lines']}")
        print(f"   Valid samples: {stats['valid_samples']}")
        if stats['skipped'] > 0:
            print(f"   Skipped: {stats['skipped']}")
        if stats['corrupted'] > 0:
            print(f"   Corrupted: {stats['corrupted']}")
    
    print("\n" + "="*70)
    print("✅ Dataset preparation completed successfully!")
    print("="*70)
    print(f"\nNext steps:")
    print(f"   1. Train the model: python train.py")
    print(f"   2. Test the model: python test.py")
    print("="*70)


if __name__ == '__main__':
    main()
