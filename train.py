#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR Training Script

Fine-tunes Microsoft's pre-trained TrOCR model on custom dataset.

Usage:
    python train.py [--cpu]
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any
import torch
import numpy as np
from PIL import Image
from datasets import load_from_disk
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator
)
import evaluate
from tqdm import tqdm


# Configuration
CONFIG = {
    'model_name': 'microsoft/trocr-small-printed',
    'output_dir': './checkpoints',
    'num_train_epochs': 20,
    'per_device_train_batch_size': 16,
    'per_device_eval_batch_size': 16,
    'learning_rate': 5e-5,
    'max_length': 10,
    'save_steps': 500,
    'eval_steps': 500,
    'logging_steps': 100,
    'logging_dir': './logs',
    'save_total_limit': 3,
    'load_best_model_at_end': True,
    'metric_for_best_model': 'cer',
    'greater_is_better': False,
    'fp16': False,  # Will be set to True if GPU is available
    'dataloader_num_workers': 4,
    'remove_unused_columns': False,
}


def setup_device(force_cpu: bool = False):
    """Setup device (CPU or GPU)"""
    if force_cpu:
        device = torch.device('cpu')
        print("🖥️  Using CPU (forced)")
        CONFIG['fp16'] = False
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
        CONFIG['fp16'] = True
    else:
        device = torch.device('cpu')
        print("🖥️  Using CPU (GPU not available)")
        CONFIG['fp16'] = False
    
    return device


def compute_cer(pred_ids, label_ids, processor):
    """
    Compute Character Error Rate (CER)
    
    Args:
        pred_ids: Predicted token IDs
        label_ids: Ground truth token IDs
        processor: TrOCR processor
    
    Returns:
        CER score
    """
    # Decode predictions and labels
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
    
    # Compute CER using jiwer
    cer_metric = evaluate.load("cer")
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    
    return cer


def create_compute_metrics(processor):
    """
    Create a compute_metrics function with processor closure
    
    Args:
        processor: TrOCR processor
    
    Returns:
        compute_metrics function
    """
    def compute_metrics(eval_pred):
        """
        Compute metrics for evaluation
        
        Args:
            eval_pred: Tuple of (predictions, labels)
        
        Returns:
            Dictionary of metrics
        """
        logits, labels = eval_pred
        predicted_ids = np.argmax(logits, axis=-1)
        
        cer = compute_cer(predicted_ids, labels, processor)
        
        return {"cer": cer}
    
    return compute_metrics


def preprocess_function(examples, processor, max_length=10):
    """
    Preprocess examples for training
    
    Args:
        examples: Batch of examples from dataset
        processor: TrOCR processor
        max_length: Maximum text length
    
    Returns:
        Preprocessed examples
    """
    # Process images
    images = examples['image']
    
    # Ensure images are PIL Images
    pil_images = []
    for img in images:
        if isinstance(img, Image.Image):
            pil_images.append(img)
        else:
            # Convert to PIL Image if needed
            pil_images.append(Image.fromarray(img))
    
    # Process images with processor
    pixel_values = processor(pil_images, return_tensors="pt").pixel_values
    
    # Process text labels
    labels = processor.tokenizer(
        examples['text'],
        padding="max_length",
        max_length=max_length,
        truncation=True,
    ).input_ids
    
    # Replace padding token id's of the labels by -100 so they are ignored in loss
    labels = [
        [(label if label != processor.tokenizer.pad_token_id else -100) for label in labels_example]
        for labels_example in labels
    ]
    
    encoding = {
        "pixel_values": pixel_values,
        "labels": labels
    }
    
    return encoding


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train OCR model')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU training (default: use GPU if available)')
    parser.add_argument('--dataset_dir', type=str,
                        default='./dataset_processed',
                        help='Directory containing processed dataset')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size per device')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='Learning rate')
    
    args = parser.parse_args()
    
    print("="*70)
    print("🚀 OCR TRAINING")
    print("="*70)
    
    # Update config with command line args
    CONFIG['num_train_epochs'] = args.epochs
    CONFIG['per_device_train_batch_size'] = args.batch_size
    CONFIG['per_device_eval_batch_size'] = args.batch_size
    CONFIG['learning_rate'] = args.learning_rate
    
    # Setup device
    device = setup_device(args.cpu)
    
    # Load dataset
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        print(f"\n❌ Error: Dataset directory not found: {dataset_dir}")
        print(f"   Please run 'python prepare_trocr_dataset.py' first")
        sys.exit(1)
    
    print(f"\n📂 Loading dataset from {dataset_dir}...")
    dataset = load_from_disk(str(dataset_dir))
    print(f"   ✅ Loaded {len(dataset)} splits")
    
    for split_name, split_dataset in dataset.items():
        print(f"      {split_name}: {len(split_dataset)} samples")
    
    # Load processor and model
    print(f"\n🔧 Loading TrOCR processor and model...")
    print(f"   Model: {CONFIG['model_name']}")
    
    processor = TrOCRProcessor.from_pretrained(CONFIG['model_name'])
    model = VisionEncoderDecoderModel.from_pretrained(CONFIG['model_name'])
    
    # Set special tokens
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    
    # Set beam search parameters
    model.config.max_length = CONFIG['max_length']
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4
    
    print(f"   ✅ Model loaded successfully")
    
    # Preprocess datasets
    print(f"\n🔄 Preprocessing datasets...")
    
    processed_datasets = {}
    for split_name, split_dataset in dataset.items():
        print(f"   Processing {split_name} split...")
        processed = split_dataset.map(
            lambda examples: preprocess_function(examples, processor, CONFIG['max_length']),
            batched=True,
            remove_columns=split_dataset.column_names,
            desc=f"Preprocessing {split_name}"
        )
        processed_datasets[split_name] = processed
    
    print(f"   ✅ Preprocessing completed")
    
    # Setup training arguments
    print(f"\n⚙️  Setting up training configuration...")
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=CONFIG['output_dir'],
        num_train_epochs=CONFIG['num_train_epochs'],
        per_device_train_batch_size=CONFIG['per_device_train_batch_size'],
        per_device_eval_batch_size=CONFIG['per_device_eval_batch_size'],
        learning_rate=CONFIG['learning_rate'],
        fp16=CONFIG['fp16'],
        logging_steps=CONFIG['logging_steps'],
        save_steps=CONFIG['save_steps'],
        eval_steps=CONFIG['eval_steps'],
        evaluation_strategy="steps",
        save_strategy="steps",
        save_total_limit=CONFIG['save_total_limit'],
        load_best_model_at_end=CONFIG['load_best_model_at_end'],
        metric_for_best_model=CONFIG['metric_for_best_model'],
        greater_is_better=CONFIG['greater_is_better'],
        predict_with_generate=True,
        logging_dir=CONFIG['logging_dir'],
        dataloader_num_workers=CONFIG['dataloader_num_workers'],
        remove_unused_columns=False,
        report_to=["tensorboard"],
    )
    
    print(f"   ✅ Configuration ready")
    
    # Create compute_metrics function with processor
    compute_metrics_fn = create_compute_metrics(processor)
    
    # Initialize trainer
    print(f"\n🏋️  Initializing trainer...")
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_datasets.get('train'),
        eval_dataset=processed_datasets.get('validation'),
        data_collator=default_data_collator,
        compute_metrics=compute_metrics_fn,
    )
    
    print(f"   ✅ Trainer initialized")
    
    # Print training info
    print("\n" + "="*70)
    print("📊 TRAINING CONFIGURATION")
    print("="*70)
    print(f"Model: {CONFIG['model_name']}")
    print(f"Device: {device}")
    print(f"Epochs: {CONFIG['num_train_epochs']}")
    print(f"Batch size: {CONFIG['per_device_train_batch_size']}")
    print(f"Learning rate: {CONFIG['learning_rate']}")
    print(f"Max text length: {CONFIG['max_length']}")
    print(f"Training samples: {len(processed_datasets.get('train', []))}")
    if 'validation' in processed_datasets:
        print(f"Validation samples: {len(processed_datasets['validation'])}")
    print(f"Save checkpoints every: {CONFIG['save_steps']} steps")
    print(f"Evaluate every: {CONFIG['eval_steps']} steps")
    print(f"Logging directory: {CONFIG['logging_dir']}")
    print(f"Checkpoint directory: {CONFIG['output_dir']}")
    print("="*70)
    
    # Start training
    print(f"\n🚀 Starting training...")
    print(f"   (Monitor with TensorBoard: tensorboard --logdir {CONFIG['logging_dir']})")
    print()
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Save final model
    final_model_dir = Path('./model')
    print(f"\n💾 Saving final model to {final_model_dir}...")
    final_model_dir.mkdir(parents=True, exist_ok=True)
    
    trainer.save_model(str(final_model_dir))
    processor.save_pretrained(str(final_model_dir))
    
    print(f"   ✅ Model saved successfully")
    
    # Evaluate on test set if available
    if 'test' in processed_datasets:
        print(f"\n📊 Evaluating on test set...")
        test_results = trainer.evaluate(processed_datasets['test'])
        
        print("\n" + "="*70)
        print("📊 TEST RESULTS")
        print("="*70)
        print(f"Test CER: {test_results['eval_cer']:.4f}")
        print(f"Test Loss: {test_results['eval_loss']:.4f}")
        print("="*70)
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\nModel saved to: {final_model_dir}")
    print(f"\nNext steps:")
    print(f"   1. Test the model: python test.py")
    print(f"   2. Run inference: python inference/predict.py <image_path>")
    print(f"   3. Start API server: python app.py --model_dir {final_model_dir}")
    print("="*70)


if __name__ == '__main__':
    main()
