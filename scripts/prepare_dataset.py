#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset Preparation Script for PaddleOCR Fine-tuning

This script prepares the dataset for PaddleOCR text recognition training by:
1. Reading raw images and labels
2. Validating image integrity
3. Splitting into train/val/test sets (80%/10%/10%)
4. Converting to PaddleOCR format
5. Generating character dictionary
6. Providing detailed statistics
"""

import os
import shutil
import random
from pathlib import Path
from collections import Counter
import argparse
from PIL import Image
from tqdm import tqdm


class DatasetPreparator:
    """Prepare dataset for PaddleOCR training"""
    
    def __init__(self, raw_dir='./dataset/raw', output_dir='./dataset'):
        self.raw_dir = Path(raw_dir)
        self.output_dir = Path(output_dir)
        self.raw_images_dir = self.raw_dir / 'images'
        self.raw_labels_file = self.raw_dir / 'labels.txt'
        
        # Output directories
        self.train_dir = self.output_dir / 'train'
        self.val_dir = self.output_dir / 'val'
        self.test_dir = self.output_dir / 'test'
        
        # Statistics
        self.stats = {
            'total_samples': 0,
            'valid_samples': 0,
            'invalid_samples': 0,
            'train_samples': 0,
            'val_samples': 0,
            'test_samples': 0,
            'text_lengths': [],
            'unique_chars': set()
        }
    
    def validate_image(self, image_path):
        """Validate image integrity using PIL"""
        try:
            with Image.open(image_path) as img:
                img.verify()
            # Reopen to ensure it's fully readable
            with Image.open(image_path) as img:
                img.load()
            return True
        except Exception as e:
            print(f"⚠️  Warning: Invalid image {image_path}: {e}")
            return False
    
    def read_labels(self):
        """Read labels from labels.txt file"""
        if not self.raw_labels_file.exists():
            raise FileNotFoundError(f"Labels file not found: {self.raw_labels_file}")
        
        samples = []
        print(f"\n📖 Reading labels from {self.raw_labels_file}...")
        
        with open(self.raw_labels_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                # Split by TAB character
                parts = line.split('\t')
                if len(parts) != 2:
                    print(f"⚠️  Warning: Line {line_num} has invalid format (expected TAB-separated): {line}")
                    continue
                
                image_name, text = parts
                image_path = self.raw_images_dir / image_name
                
                # Check if image exists
                if not image_path.exists():
                    print(f"⚠️  Warning: Image not found: {image_path}")
                    self.stats['invalid_samples'] += 1
                    continue
                
                # Validate image
                if not self.validate_image(image_path):
                    self.stats['invalid_samples'] += 1
                    continue
                
                samples.append({
                    'image_name': image_name,
                    'image_path': image_path,
                    'text': text
                })
                self.stats['valid_samples'] += 1
                self.stats['text_lengths'].append(len(text))
                self.stats['unique_chars'].update(text)
        
        self.stats['total_samples'] = len(samples) + self.stats['invalid_samples']
        print(f"✅ Found {len(samples)} valid samples out of {self.stats['total_samples']} total")
        
        return samples
    
    def split_dataset(self, samples, train_ratio=0.8, val_ratio=0.1):
        """Split dataset into train/val/test sets"""
        print(f"\n📊 Splitting dataset (train: {train_ratio*100}%, val: {val_ratio*100}%, test: {(1-train_ratio-val_ratio)*100}%)...")
        
        # Randomize samples
        random.shuffle(samples)
        
        total = len(samples)
        train_count = int(total * train_ratio)
        val_count = int(total * val_ratio)
        
        train_samples = samples[:train_count]
        val_samples = samples[train_count:train_count + val_count]
        test_samples = samples[train_count + val_count:]
        
        self.stats['train_samples'] = len(train_samples)
        self.stats['val_samples'] = len(val_samples)
        self.stats['test_samples'] = len(test_samples)
        
        print(f"  Train: {len(train_samples)} samples")
        print(f"  Val: {len(val_samples)} samples")
        print(f"  Test: {len(test_samples)} samples")
        
        return train_samples, val_samples, test_samples
    
    def copy_images_and_generate_labels(self, samples, split_name):
        """Copy images to split directory and generate label file"""
        split_dir = getattr(self, f'{split_name}_dir')
        images_dir = split_dir / 'images'
        images_dir.mkdir(parents=True, exist_ok=True)
        
        label_file = split_dir / f'{split_name}_list.txt'
        
        print(f"\n📁 Processing {split_name} set...")
        
        with open(label_file, 'w', encoding='utf-8') as f:
            for sample in tqdm(samples, desc=f"Copying {split_name} images"):
                # Copy image
                dest_path = images_dir / sample['image_name']
                shutil.copy2(sample['image_path'], dest_path)
                
                # Write label in PaddleOCR format: relative_path<TAB>text
                rel_path = f"images/{sample['image_name']}"
                f.write(f"{rel_path}\t{sample['text']}\n")
        
        print(f"✅ {split_name.capitalize()} set: {len(samples)} samples saved to {label_file}")
    
    def generate_dictionary(self):
        """Generate character dictionary from all unique characters"""
        dict_file = self.output_dir / 'dict.txt'
        
        # Sort characters for consistency
        sorted_chars = sorted(list(self.stats['unique_chars']))
        
        print(f"\n📝 Generating character dictionary...")
        print(f"  Unique characters: {len(sorted_chars)}")
        
        with open(dict_file, 'w', encoding='utf-8') as f:
            for char in sorted_chars:
                f.write(f"{char}\n")
        
        print(f"✅ Dictionary saved to {dict_file}")
        
        # Print sample of characters
        sample_chars = sorted_chars[:50] if len(sorted_chars) > 50 else sorted_chars
        print(f"  Sample characters: {' '.join(sample_chars)}")
        if len(sorted_chars) > 50:
            print(f"  ... and {len(sorted_chars) - 50} more")
    
    def print_statistics(self):
        """Print detailed dataset statistics"""
        print("\n" + "="*60)
        print("📊 DATASET STATISTICS")
        print("="*60)
        print(f"Total samples found: {self.stats['total_samples']}")
        print(f"Valid samples: {self.stats['valid_samples']}")
        print(f"Invalid/skipped samples: {self.stats['invalid_samples']}")
        print(f"\nSplit distribution:")
        print(f"  Train: {self.stats['train_samples']} samples")
        print(f"  Val: {self.stats['val_samples']} samples")
        print(f"  Test: {self.stats['test_samples']} samples")
        
        if self.stats['text_lengths']:
            print(f"\nText length statistics:")
            print(f"  Min length: {min(self.stats['text_lengths'])} characters")
            print(f"  Max length: {max(self.stats['text_lengths'])} characters")
            print(f"  Average length: {sum(self.stats['text_lengths']) / len(self.stats['text_lengths']):.1f} characters")
        
        print(f"\nCharacter dictionary:")
        print(f"  Unique characters: {len(self.stats['unique_chars'])}")
        print("="*60)
    
    def prepare(self, train_ratio=0.8, val_ratio=0.1, seed=42):
        """Main preparation pipeline"""
        print("🚀 Starting dataset preparation for PaddleOCR...")
        
        # Set random seed for reproducibility
        random.seed(seed)
        
        # Read and validate samples
        samples = self.read_labels()
        
        if not samples:
            print("❌ No valid samples found. Exiting.")
            return
        
        # Split dataset
        train_samples, val_samples, test_samples = self.split_dataset(samples, train_ratio, val_ratio)
        
        # Copy images and generate labels for each split
        self.copy_images_and_generate_labels(train_samples, 'train')
        self.copy_images_and_generate_labels(val_samples, 'val')
        self.copy_images_and_generate_labels(test_samples, 'test')
        
        # Generate character dictionary
        self.generate_dictionary()
        
        # Print statistics
        self.print_statistics()
        
        print("\n✅ Dataset preparation completed successfully!")
        print(f"\n📁 Output structure:")
        print(f"  {self.train_dir}/train_list.txt")
        print(f"  {self.val_dir}/val_list.txt")
        print(f"  {self.test_dir}/test_list.txt")
        print(f"  {self.output_dir}/dict.txt")


def main():
    parser = argparse.ArgumentParser(description='Prepare dataset for PaddleOCR fine-tuning')
    parser.add_argument('--raw_dir', type=str, default='./dataset/raw',
                        help='Directory containing raw images and labels.txt')
    parser.add_argument('--output_dir', type=str, default='./dataset',
                        help='Output directory for processed dataset')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Ratio of training samples (default: 0.8)')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Ratio of validation samples (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Validate ratios
    if args.train_ratio + args.val_ratio >= 1.0:
        print("❌ Error: train_ratio + val_ratio must be less than 1.0")
        return
    
    # Create preparator and run
    preparator = DatasetPreparator(args.raw_dir, args.output_dir)
    preparator.prepare(args.train_ratio, args.val_ratio, args.seed)


if __name__ == '__main__':
    main()
