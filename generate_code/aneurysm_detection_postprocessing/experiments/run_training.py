#!/usr/bin/env python3
"""
Training Script for Aneurysm Detection Models

This script provides a complete training pipeline for both CPM-Net and 3D-CNN-TR models
as specified in the paper. It handles data loading, model initialization, training orchestration,
and checkpoint management for the 45-epoch training protocol.

Usage:
    python run_training.py --model cpm_net --data_root /path/to/data --output_dir ./outputs
    python run_training.py --model cnn_tr_3d --data_root /path/to/data --output_dir ./outputs --mixed_precision
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import project modules
from models.cpm_net import create_cpm_net, CPM_NET_CONFIG
from models.cnn_tr_3d import create_cnn_tr_3d, CNN_TR_3D_CONFIG
from data.dataset_loader import DatasetConfig, create_data_loaders, create_sample_dataset
from training.trainer import create_trainer, TRAINING_CONFIGS
from training.optimization import create_optimizer_manager
from evaluation.performance_evaluator import PerformanceEvaluator

# Training configurations
EXPERIMENT_CONFIGS = {
    'cpm_net': {
        'model_type': 'cpm_net',
        'model_config': CPM_NET_CONFIG,
        'training_config': TRAINING_CONFIGS['cpm_net'],
        'description': 'CPM-Net: Center-Points Matching Network for 3D aneurysm detection'
    },
    'cnn_tr_3d': {
        'model_type': 'cnn_tr_3d',
        'model_config': CNN_TR_3D_CONFIG,
        'training_config': TRAINING_CONFIGS['cnn_tr_3d'],
        'description': '3D-CNN-TR: Hybrid CNN-Transformer model with auxiliary artery input'
    }
}

def setup_logging(output_dir: Path, log_level: str = 'INFO') -> logging.Logger:
    """Setup logging configuration for training."""
    log_dir = output_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('aneurysm_training')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # File handler
    log_file = log_dir / f'training_{time.strftime("%Y%m%d_%H%M%S")}.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def setup_device(gpu_id: Optional[int] = None) -> torch.device:
    """Setup training device (GPU/CPU)."""
    if gpu_id is not None:
        if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
            device = torch.device(f'cuda:{gpu_id}')
            torch.cuda.set_device(gpu_id)
        else:
            print(f"Warning: GPU {gpu_id} not available, using CPU")
            device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    return device

def create_model(model_type: str, device: torch.device) -> torch.nn.Module:
    """Create and initialize model."""
    if model_type == 'cpm_net':
        model = create_cpm_net(
            input_channels=1,
            num_classes=1,
            confidence_threshold=0.8,
            base_channels=32
        )
    elif model_type == 'cnn_tr_3d':
        model = create_cnn_tr_3d(
            input_channels=1,
            aux_channels=1,  # For auxiliary artery segmentation input
            num_classes=1,
            confidence_threshold=0.8,
            base_channels=32
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = model.to(device)
    return model

def setup_data_loaders(data_config: DatasetConfig, model_type: str) -> Dict[str, torch.utils.data.DataLoader]:
    """Setup data loaders for training and validation."""
    # Get batch size from training config
    batch_size = TRAINING_CONFIGS[model_type]['batch_size']
    
    # Update data config with model-specific settings
    if model_type == 'cnn_tr_3d':
        # 3D-CNN-TR requires auxiliary artery segmentation
        data_config.use_auxiliary_input = True
    
    # Create data loaders
    data_loaders = create_data_loaders(
        config=data_config,
        train_split=0.8,
        val_split=0.1,
        test_split=0.1
    )
    
    return data_loaders

def save_experiment_config(output_dir: Path, args: argparse.Namespace, 
                          model_config: Dict[str, Any], training_config: Dict[str, Any]):
    """Save experiment configuration for reproducibility."""
    config_file = output_dir / 'experiment_config.txt'
    
    with open(config_file, 'w') as f:
        f.write("=== EXPERIMENT CONFIGURATION ===\n\n")
        
        # Command line arguments
        f.write("Command Line Arguments:\n")
        for key, value in vars(args).items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")
        
        # Model configuration
        f.write("Model Configuration:\n")
        for key, value in model_config.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")
        
        # Training configuration
        f.write("Training Configuration:\n")
        for key, value in training_config.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")
        
        # System information
        f.write("System Information:\n")
        f.write(f"  PyTorch version: {torch.__version__}\n")
        f.write(f"  CUDA available: {torch.cuda.is_available()}\n")
        if torch.cuda.is_available():
            f.write(f"  CUDA version: {torch.version.cuda}\n")
            f.write(f"  GPU count: {torch.cuda.device_count()}\n")
            for i in range(torch.cuda.device_count()):
                f.write(f"  GPU {i}: {torch.cuda.get_device_name(i)}\n")

def run_training_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    """Run complete training experiment."""
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir, args.log_level)
    logger.info("Starting aneurysm detection training experiment")
    
    # Get experiment configuration
    if args.model not in EXPERIMENT_CONFIGS:
        raise ValueError(f"Unknown model: {args.model}. Available: {list(EXPERIMENT_CONFIGS.keys())}")
    
    exp_config = EXPERIMENT_CONFIGS[args.model]
    logger.info(f"Model: {exp_config['description']}")
    
    # Setup device
    device = setup_device(args.gpu_id)
    logger.info(f"Using device: {device}")
    
    # Setup data configuration
    if args.use_sample_data:
        logger.info("Creating sample dataset for testing")
        data_config = create_sample_dataset(
            num_cases=args.sample_size,
            output_dir=output_dir / 'sample_data'
        )
    else:
        data_config = DatasetConfig(
            data_root=args.data_root,
            annotation_path=args.annotation_path,
            target_spacing=(0.4, 0.4, 0.4),  # Paper specification
            subvolume_size=(128, 128, 128),   # Paper specification
            overlap_ratio=0.5,
            augmentation_prob=0.5 if args.augmentation else 0.0,
            normalize_intensity=True,
            clip_intensity_range=(-1000, 1000)  # HU range for CTA
        )
    
    # Setup data loaders
    logger.info("Setting up data loaders")
    data_loaders = setup_data_loaders(data_config, args.model)
    logger.info(f"Training samples: {len(data_loaders['train'].dataset)}")
    logger.info(f"Validation samples: {len(data_loaders['val'].dataset)}")
    
    # Create model
    logger.info("Creating model")
    model = create_model(args.model, device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup TensorBoard logging
    tensorboard_dir = output_dir / 'tensorboard'
    writer = SummaryWriter(tensorboard_dir)
    
    # Create trainer
    logger.info("Creating trainer")
    trainer = create_trainer(
        model_type=args.model,
        data_config=data_config,
        device=device,
        output_dir=output_dir,
        resume_checkpoint=args.resume_checkpoint,
        mixed_precision=args.mixed_precision
    )
    
    # Set data loaders
    trainer.train_loader = data_loaders['train']
    trainer.val_loader = data_loaders['val']
    trainer.writer = writer
    
    # Save experiment configuration
    save_experiment_config(
        output_dir, args, 
        exp_config['model_config'], 
        exp_config['training_config']
    )
    
    # Run training
    logger.info("Starting training")
    start_time = time.time()
    
    try:
        training_results = trainer.train()
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Save final results
        results = {
            'model_type': args.model,
            'training_time': training_time,
            'final_epoch': training_results.get('final_epoch', 0),
            'best_val_loss': training_results.get('best_val_loss', float('inf')),
            'best_val_metrics': training_results.get('best_val_metrics', {}),
            'output_dir': str(output_dir)
        }
        
        # Log final results
        logger.info("=== TRAINING RESULTS ===")
        for key, value in results.items():
            logger.info(f"{key}: {value}")
        
        return results
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
    finally:
        writer.close()

def main():
    """Main training script entry point."""
    parser = argparse.ArgumentParser(
        description="Train aneurysm detection models (CPM-Net or 3D-CNN-TR)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model selection
    parser.add_argument(
        '--model', 
        type=str, 
        choices=['cpm_net', 'cnn_tr_3d'],
        required=True,
        help='Model type to train'
    )
    
    # Data configuration
    parser.add_argument(
        '--data_root',
        type=str,
        help='Root directory containing CTA data'
    )
    
    parser.add_argument(
        '--annotation_path',
        type=str,
        help='Path to annotation file (JSON format)'
    )
    
    parser.add_argument(
        '--use_sample_data',
        action='store_true',
        help='Use generated sample data for testing'
    )
    
    parser.add_argument(
        '--sample_size',
        type=int,
        default=10,
        help='Number of sample cases to generate'
    )
    
    # Training configuration
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./training_outputs',
        help='Output directory for checkpoints and logs'
    )
    
    parser.add_argument(
        '--resume_checkpoint',
        type=str,
        help='Path to checkpoint to resume training from'
    )
    
    parser.add_argument(
        '--mixed_precision',
        action='store_true',
        help='Enable mixed precision training'
    )
    
    parser.add_argument(
        '--augmentation',
        action='store_true',
        default=True,
        help='Enable data augmentation'
    )
    
    # System configuration
    parser.add_argument(
        '--gpu_id',
        type=int,
        help='GPU ID to use (default: auto-select)'
    )
    
    parser.add_argument(
        '--log_level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate arguments
    if not args.use_sample_data:
        if not args.data_root:
            parser.error("--data_root is required when not using sample data")
        if not args.annotation_path:
            parser.error("--annotation_path is required when not using sample data")
    
    # Run training
    try:
        results = run_training_experiment(args)
        print("\n=== TRAINING COMPLETED SUCCESSFULLY ===")
        print(f"Results saved to: {results['output_dir']}")
        return 0
    except Exception as e:
        print(f"\n=== TRAINING FAILED ===")
        print(f"Error: {str(e)}")
        return 1

if __name__ == '__main__':
    # Set multiprocessing start method for compatibility
    mp.set_start_method('spawn', force=True)
    
    # Run main function
    exit_code = main()
    sys.exit(exit_code)