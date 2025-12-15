"""
YOLOv11 Fall Detection - Full Dataset Training Script
Optimized for GTX 1650 (4GB VRAM) + 8GB RAM
Train with FULL dataset (~28k images) - overnight training
"""

import os
import sys
import shutil
from pathlib import Path
import yaml

# ============== CONFIG ==============
BASE_DIR = Path(r"c:\Users\Admin\Downloads\FallDetectionDataset_forYOLOv11")
DATASET_DIR = BASE_DIR / "Fall_Detection_Dataset_NoDup"
OUTPUT_DIR = BASE_DIR / "runs"

# Training hyperparams optimized for GTX 1650 4GB - Full dataset
# Note: Larger dataset = longer epochs, NOT more epochs needed
CONFIG = {
    "model": "yolo11s.pt",        # Small model - better accuracy, still fits 4GB VRAM
    "imgsz": 480,                  # Reduced from 640 for VRAM
    "batch": 4,                    # Reduced batch for yolo11s on 4GB VRAM
    "epochs": 50,                  # Same as 16k - each epoch is longer with more data
    "workers": 2,                  # Low workers to save RAM
    "cache": False,                # Don't cache - save RAM (28k images too big)
    "amp": False,                  # Disable AMP - avoid bug in ultralytics 8.3.x
    "device": 0,                   # GPU 0
    "patience": 15,                # Early stopping - stops if no improvement for 15 epochs
    "save_period": 10,             # Save checkpoint every 10 epochs
    "project": str(OUTPUT_DIR),
    "name": "fall_detection_full",
    "exist_ok": True,
}


def create_data_yaml():
    """Create data.yaml for full dataset."""
    data_yaml = {
        'path': str(DATASET_DIR),
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': 2,
        'names': ['fall', 'not_fall']
    }
    
    yaml_path = DATASET_DIR / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print(f"Data config saved to: {yaml_path}")
    return yaml_path


def train_model(data_yaml: Path):
    """Train YOLOv11 model on full dataset."""
    print("\n" + "="*60)
    print("Training YOLOv11 on FULL Dataset")
    print("="*60)
    
    from ultralytics import YOLO
    
    print(f"\nTraining Configuration:")
    for k, v in CONFIG.items():
        print(f"  {k}: {v}")
    
    print(f"\nData config: {data_yaml}")
    
    # Load model
    print(f"\nLoading model: {CONFIG['model']}")
    model = YOLO(CONFIG['model'])
    
    # Train
    print("\nStarting training...\n")
    results = model.train(
        data=str(data_yaml),
        imgsz=CONFIG['imgsz'],
        batch=CONFIG['batch'],
        epochs=CONFIG['epochs'],
        workers=CONFIG['workers'],
        cache=CONFIG['cache'],
        amp=CONFIG['amp'],
        device=CONFIG['device'],
        patience=CONFIG['patience'],
        save_period=CONFIG['save_period'],
        project=CONFIG['project'],
        name=CONFIG['name'],
        exist_ok=CONFIG['exist_ok'],
        verbose=True,
    )
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    
    # Print results location
    save_dir = Path(CONFIG['project']) / CONFIG['name']
    print(f"\nResults saved to: {save_dir}")
    print(f"Best model: {save_dir / 'weights' / 'best.pt'}")
    print(f"Last model: {save_dir / 'weights' / 'last.pt'}")
    
    return results


def main():
    """Main entry point."""
    print("="*60)
    print("YOLOv11 Fall Detection - FULL Dataset Training")
    print("Optimized for GTX 1650 (4GB VRAM)")
    print("="*60)
    
    # Check GPU
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"\nGPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        print("\nWARNING: No GPU detected! Training will be very slow.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)
    
    # Dataset statistics
    print("\n" + "="*60)
    print("Dataset Statistics (Full)")
    print("="*60)
    print("""
    Split   | Fall   | NotFall | Total
    --------|--------|---------|-------
    train   | 13,095 | 15,033  | 28,128
    valid   |  1,626 |  1,857  |  3,483
    test    |  1,142 |  1,196  |  2,338
    --------|--------|---------|-------
    TOTAL   | 15,863 | 18,086  | 33,949
    """)
    
    # Estimated time
    print("Estimated training time:")
    print("  - ~28k train images × 100 epochs")
    print("  - ~6.5 min/epoch (based on 16k benchmark)")
    print("  - Estimated: 10-12 hours (with early stopping ~8-10h)")
    print("  - Early stopping patience: 20 epochs")
    
    # Create data.yaml
    data_yaml = create_data_yaml()
    
    # Confirm
    print("\n" + "="*60)
    response = input("Start overnight training? (y/n): ")
    if response.lower() == 'y':
        train_model(data_yaml)
    else:
        print("\nTraining cancelled.")
        print("To start later, run: python train_full_dataset.py --start")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--start":
        # Direct start without confirmation
        data_yaml = create_data_yaml()
        train_model(data_yaml)
    else:
        main()
