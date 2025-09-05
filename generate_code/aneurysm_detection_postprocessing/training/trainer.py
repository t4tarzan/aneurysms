"""
Training orchestration module for 3D aneurysm detection models.

This module provides comprehensive training functionality for both CPM-Net and 3D-CNN-TR
models, including data loading, optimization, validation, checkpointing, and learning
rate scheduling according to the paper specifications.

Key Features:
- Model-agnostic training pipeline supporting both detection architectures
- AdamW optimizer with learning rate scheduling (0.0001 → 0.00001)
- 45-epoch training protocol as specified in the paper
- Validation monitoring and early stopping
- Model checkpointing and resuming capabilities
- Mixed precision training support
- Comprehensive logging and metrics tracking

Training Configuration:
- Epochs: 45 (as per paper specification)
- Optimizer: AdamW with steadily decreasing learning rate
- Initial LR: 0.0001, Final LR: 0.00001
- Batch size: 4 for 3D-CNN-TR, 8 for CPM-Net
- Input size: 128x128x128 for both models
"""

import os
import time
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np

# Import model architectures
from ..models.cpm_net import CPMNet, create_cpm_net, CPM_NET_CONFIG
from ..models.cnn_tr_3d import CNNTR3D, create_cnn_tr_3d, CNN_TR_3D_CONFIG
from ..models.base_detector import BaseDetector

# Import loss functions
from .loss_functions import (
    create_cpm_net_loss, 
    create_cnn_tr_3d_loss,
    LOSS_CONFIGS
)

# Import data handling
from ..data.dataset_loader import AneurysmDataset, create_data_loaders
from ..data.transforms import get_training_transforms, get_validation_transforms

# Import evaluation utilities
from ..evaluation.metrics import DetectionMetrics
from ..evaluation.performance_evaluator import PerformanceEvaluator


class AneurysmTrainer:
    """
    Comprehensive trainer for 3D aneurysm detection models.
    
    Supports both CPM-Net and 3D-CNN-TR architectures with model-specific
    configurations, loss functions, and training protocols.
    """
    
    def __init__(
        self,
        model: BaseDetector,
        model_type: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        output_dir: str = "training_outputs",
        resume_checkpoint: Optional[str] = None,
        mixed_precision: bool = True,
        log_level: str = "INFO"
    ):
        """
        Initialize the trainer.
        
        Args:
            model: The detection model to train
            model_type: Type of model ('cpm_net' or 'cnn_tr_3d')
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Training device (CPU/GPU)
            output_dir: Directory for saving outputs
            resume_checkpoint: Path to checkpoint for resuming training
            mixed_precision: Whether to use mixed precision training
            log_level: Logging level
        """
        self.model = model.to(device)
        self.model_type = model_type.lower()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = Path(output_dir)
        self.mixed_precision = mixed_precision
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging(log_level)
        
        # Get model configuration
        self.config = self._get_model_config()
        
        # Initialize training components
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.criterion = self._create_loss_function()
        self.scaler = GradScaler() if mixed_precision else None
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_metrics = {}
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': [],
            'metrics': []
        }
        
        # Initialize metrics calculator
        self.metrics_calculator = DetectionMetrics()
        self.evaluator = PerformanceEvaluator()
        
        # Resume from checkpoint if provided
        if resume_checkpoint:
            self.load_checkpoint(resume_checkpoint)
            
        self.logger.info(f"Trainer initialized for {model_type} model")
        self.logger.info(f"Training configuration: {self.config}")
    
    def _setup_logging(self, log_level: str):
        """Setup logging configuration."""
        log_file = self.output_dir / "training.log"
        
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(f"AneurysmTrainer_{self.model_type}")
    
    def _get_model_config(self) -> Dict[str, Any]:
        """Get model-specific configuration."""
        if self.model_type == 'cpm_net':
            return CPM_NET_CONFIG.copy()
        elif self.model_type == 'cnn_tr_3d':
            return CNN_TR_3D_CONFIG.copy()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create AdamW optimizer with model-specific parameters."""
        optimizer_config = self.config['optimizer']
        
        if optimizer_config['type'] != 'AdamW':
            raise ValueError(f"Expected AdamW optimizer, got {optimizer_config['type']}")
        
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=optimizer_config['initial_lr'],
            weight_decay=optimizer_config.get('weight_decay', 1e-4),
            betas=optimizer_config.get('betas', (0.9, 0.999))
        )
        
        self.logger.info(f"Created AdamW optimizer with initial LR: {optimizer_config['initial_lr']}")
        return optimizer
    
    def _create_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler for steady decrease from 0.0001 to 0.00001."""
        optimizer_config = self.config['optimizer']
        total_epochs = self.config['epochs']
        
        initial_lr = optimizer_config['initial_lr']  # 0.0001
        final_lr = optimizer_config['final_lr']      # 0.00001
        
        # Use exponential decay to achieve steady decrease
        gamma = (final_lr / initial_lr) ** (1.0 / total_epochs)
        
        scheduler = optim.lr_scheduler.ExponentialLR(
            self.optimizer,
            gamma=gamma
        )
        
        self.logger.info(f"Created ExponentialLR scheduler: {initial_lr} → {final_lr} over {total_epochs} epochs")
        return scheduler
    
    def _create_loss_function(self) -> nn.Module:
        """Create model-specific loss function."""
        if self.model_type == 'cpm_net':
            return create_cpm_net_loss()
        elif self.model_type == 'cnn_tr_3d':
            return create_cnn_tr_3d_loss()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = []
        epoch_metrics = {'total_samples': 0, 'total_detections': 0}
        
        start_time = time.time()
        
        for batch_idx, batch_data in enumerate(self.train_loader):
            # Move data to device
            if self.model_type == 'cnn_tr_3d':
                # Dual input: CTA volume + auxiliary artery segmentation
                cta_volume = batch_data['volume'].to(self.device)
                aux_input = batch_data.get('aux_input', torch.zeros_like(cta_volume)).to(self.device)
                inputs = (cta_volume, aux_input)
            else:
                # Single input: CTA volume only
                inputs = batch_data['volume'].to(self.device)
            
            targets = batch_data['targets']  # List of target dictionaries
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.mixed_precision and self.scaler:
                with autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
            
            # Record loss
            epoch_losses.append(loss.item())
            epoch_metrics['total_samples'] += len(targets)
            
            # Log batch progress
            if batch_idx % 10 == 0:
                self.logger.debug(
                    f"Epoch {self.current_epoch}, Batch {batch_idx}/{len(self.train_loader)}, "
                    f"Loss: {loss.item():.4f}, LR: {self.optimizer.param_groups[0]['lr']:.6f}"
                )
        
        # Calculate epoch statistics
        epoch_time = time.time() - start_time
        avg_loss = np.mean(epoch_losses)
        current_lr = self.optimizer.param_groups[0]['lr']
        
        epoch_stats = {
            'avg_loss': avg_loss,
            'learning_rate': current_lr,
            'epoch_time': epoch_time,
            'samples_processed': epoch_metrics['total_samples']
        }
        
        self.logger.info(
            f"Epoch {self.current_epoch} Training - "
            f"Loss: {avg_loss:.4f}, LR: {current_lr:.6f}, Time: {epoch_time:.1f}s"
        )
        
        return epoch_stats
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        val_losses = []
        all_predictions = []
        all_targets = []
        
        start_time = time.time()
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(self.val_loader):
                # Move data to device
                if self.model_type == 'cnn_tr_3d':
                    cta_volume = batch_data['volume'].to(self.device)
                    aux_input = batch_data.get('aux_input', torch.zeros_like(cta_volume)).to(self.device)
                    inputs = (cta_volume, aux_input)
                else:
                    inputs = batch_data['volume'].to(self.device)
                
                targets = batch_data['targets']
                
                # Forward pass
                if self.mixed_precision:
                    with autocast():
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, targets)
                else:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                
                val_losses.append(loss.item())
                
                # Collect predictions and targets for metrics
                all_predictions.extend(outputs)
                all_targets.extend(targets)
        
        # Calculate validation statistics
        val_time = time.time() - start_time
        avg_val_loss = np.mean(val_losses)
        
        # Calculate detection metrics
        val_metrics = self.metrics_calculator.calculate_batch_metrics(
            all_predictions, all_targets
        )
        
        val_stats = {
            'avg_loss': avg_val_loss,
            'validation_time': val_time,
            **val_metrics
        }
        
        self.logger.info(
            f"Epoch {self.current_epoch} Validation - "
            f"Loss: {avg_val_loss:.4f}, Time: {val_time:.1f}s"
        )
        
        if val_metrics:
            self.logger.info(f"Validation Metrics: {val_metrics}")
        
        return val_stats
    
    def train(self, num_epochs: Optional[int] = None) -> Dict[str, List]:
        """
        Main training loop.
        
        Args:
            num_epochs: Number of epochs to train (uses config default if None)
            
        Returns:
            Training history dictionary
        """
        if num_epochs is None:
            num_epochs = self.config['epochs']
        
        self.logger.info(f"Starting training for {num_epochs} epochs")
        self.logger.info(f"Model: {self.model_type}, Device: {self.device}")
        
        start_epoch = self.current_epoch
        end_epoch = start_epoch + num_epochs
        
        for epoch in range(start_epoch, end_epoch):
            self.current_epoch = epoch
            
            # Training phase
            train_stats = self.train_epoch()
            
            # Validation phase
            val_stats = self.validate_epoch()
            
            # Update learning rate
            self.scheduler.step()
            
            # Record history
            self.training_history['train_loss'].append(train_stats['avg_loss'])
            self.training_history['val_loss'].append(val_stats['avg_loss'])
            self.training_history['learning_rates'].append(train_stats['learning_rate'])
            self.training_history['metrics'].append(val_stats)
            
            # Check for best model
            if val_stats['avg_loss'] < self.best_val_loss:
                self.best_val_loss = val_stats['avg_loss']
                self.best_val_metrics = val_stats.copy()
                self.save_checkpoint(is_best=True)
                self.logger.info(f"New best model saved with validation loss: {self.best_val_loss:.4f}")
            
            # Save regular checkpoint
            if (epoch + 1) % 5 == 0:  # Save every 5 epochs
                self.save_checkpoint(is_best=False)
            
            # Log epoch summary
            self.logger.info(f"Epoch {epoch} completed - Train Loss: {train_stats['avg_loss']:.4f}, Val Loss: {val_stats['avg_loss']:.4f}")
        
        self.logger.info("Training completed!")
        self.logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        
        # Save final training history
        self.save_training_history()
        
        return self.training_history
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_type': self.model_type,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_metrics': self.best_val_metrics,
            'training_history': self.training_history,
            'config': self.config
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save checkpoint
        checkpoint_name = f"checkpoint_epoch_{self.current_epoch}.pth"
        checkpoint_path = self.output_dir / checkpoint_name
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model separately
        if is_best:
            best_path = self.output_dir / f"best_model_{self.model_type}.pth"
            torch.save(checkpoint, best_path)
            self.logger.info(f"Best model saved to {best_path}")
        
        self.logger.debug(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training state
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_metrics = checkpoint['best_val_metrics']
        self.training_history = checkpoint['training_history']
        
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
        self.logger.info(f"Resuming from epoch {self.current_epoch}")
    
    def save_training_history(self):
        """Save training history to JSON file."""
        history_path = self.output_dir / f"training_history_{self.model_type}.json"
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_history = {}
        for key, value in self.training_history.items():
            if isinstance(value, list):
                serializable_history[key] = [
                    float(v) if isinstance(v, (np.floating, np.integer)) else v
                    for v in value
                ]
            else:
                serializable_history[key] = value
        
        with open(history_path, 'w') as f:
            json.dump(serializable_history, f, indent=2)
        
        self.logger.info(f"Training history saved to {history_path}")
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model and training summary."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        summary = {
            'model_type': self.model_type,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'current_epoch': self.current_epoch,
            'best_validation_loss': self.best_val_loss,
            'best_validation_metrics': self.best_val_metrics,
            'training_config': self.config,
            'device': str(self.device),
            'mixed_precision': self.mixed_precision
        }
        
        return summary


def create_trainer(
    model_type: str,
    data_config: Dict[str, Any],
    device: Optional[torch.device] = None,
    output_dir: str = "training_outputs",
    resume_checkpoint: Optional[str] = None,
    mixed_precision: bool = True
) -> AneurysmTrainer:
    """
    Factory function to create a configured trainer.
    
    Args:
        model_type: Type of model ('cpm_net' or 'cnn_tr_3d')
        data_config: Data configuration dictionary
        device: Training device
        output_dir: Output directory for training artifacts
        resume_checkpoint: Path to checkpoint for resuming
        mixed_precision: Whether to use mixed precision training
        
    Returns:
        Configured AneurysmTrainer instance
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    if model_type.lower() == 'cpm_net':
        model = create_cpm_net()
    elif model_type.lower() == 'cnn_tr_3d':
        model = create_cnn_tr_3d()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        data_config=data_config,
        model_type=model_type
    )
    
    # Create trainer
    trainer = AneurysmTrainer(
        model=model,
        model_type=model_type,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        output_dir=output_dir,
        resume_checkpoint=resume_checkpoint,
        mixed_precision=mixed_precision
    )
    
    return trainer


# Training configuration templates
TRAINING_CONFIGS = {
    'cpm_net': {
        'model_type': 'cpm_net',
        'epochs': 45,
        'batch_size': 8,
        'input_size': (128, 128, 128),
        'optimizer': {
            'type': 'AdamW',
            'initial_lr': 0.0001,
            'final_lr': 0.00001,
            'weight_decay': 1e-4
        },
        'mixed_precision': True,
        'validation_frequency': 1
    },
    'cnn_tr_3d': {
        'model_type': 'cnn_tr_3d',
        'epochs': 45,
        'batch_size': 4,  # Smaller batch size due to transformer complexity
        'input_size': (128, 128, 128),
        'optimizer': {
            'type': 'AdamW',
            'initial_lr': 0.0001,
            'final_lr': 0.00001,
            'weight_decay': 1e-4
        },
        'mixed_precision': True,
        'validation_frequency': 1,
        'auxiliary_input': True  # Requires artery segmentation input
    }
}


if __name__ == "__main__":
    # Example usage
    print("AneurysmTrainer module - Training orchestration for 3D aneurysm detection")
    print("Available model types:", list(TRAINING_CONFIGS.keys()))
    
    # Print configuration details
    for model_type, config in TRAINING_CONFIGS.items():
        print(f"\n{model_type.upper()} Configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")