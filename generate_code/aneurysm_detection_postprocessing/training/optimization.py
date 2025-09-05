"""
Optimization module for aneurysm detection models.

This module provides optimizer and learning rate scheduler configurations
for training CPM-Net and 3D-CNN-TR models as specified in the paper.

Key Features:
- AdamW optimizer with model-specific configurations
- Exponential learning rate decay (0.0001 → 0.00001 over 45 epochs)
- Weight decay and gradient clipping configurations
- Factory functions for easy optimizer creation

Paper Reference:
"Automated anatomy-based post-processing reduces false positives and improved 
interpretability of deep learning intracranial aneurysm detection"
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR, StepLR, CosineAnnealingLR
from typing import Dict, Any, Optional, Union, List
import logging
import math

# Configure logging
logger = logging.getLogger(__name__)

# Optimizer configurations for different models
OPTIMIZER_CONFIGS = {
    'cpm_net': {
        'optimizer_type': 'adamw',
        'lr': 0.0001,
        'weight_decay': 0.01,
        'betas': (0.9, 0.999),
        'eps': 1e-8,
        'amsgrad': False
    },
    'cnn_tr_3d': {
        'optimizer_type': 'adamw',
        'lr': 0.0001,
        'weight_decay': 0.01,
        'betas': (0.9, 0.999),
        'eps': 1e-8,
        'amsgrad': False
    }
}

# Learning rate scheduler configurations
SCHEDULER_CONFIGS = {
    'cpm_net': {
        'scheduler_type': 'exponential',
        'gamma': 0.1 ** (1/45),  # Decay from 0.0001 to 0.00001 over 45 epochs
        'step_size': None,
        'milestones': None,
        'T_max': 45
    },
    'cnn_tr_3d': {
        'scheduler_type': 'exponential',
        'gamma': 0.1 ** (1/45),  # Decay from 0.0001 to 0.00001 over 45 epochs
        'step_size': None,
        'milestones': None,
        'T_max': 45
    }
}

# Gradient clipping configurations
GRADIENT_CONFIGS = {
    'cpm_net': {
        'clip_grad_norm': True,
        'max_norm': 1.0,
        'norm_type': 2.0
    },
    'cnn_tr_3d': {
        'clip_grad_norm': True,
        'max_norm': 1.0,
        'norm_type': 2.0
    }
}


class OptimizerManager:
    """
    Manages optimizer and learning rate scheduler for model training.
    
    This class provides a unified interface for creating and managing
    optimizers and schedulers with model-specific configurations.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        model_type: str,
        custom_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize optimizer manager.
        
        Args:
            model: PyTorch model to optimize
            model_type: Type of model ('cpm_net' or 'cnn_tr_3d')
            custom_config: Optional custom configuration to override defaults
        """
        self.model = model
        self.model_type = model_type.lower()
        self.custom_config = custom_config or {}
        
        # Validate model type
        if self.model_type not in OPTIMIZER_CONFIGS:
            raise ValueError(f"Unsupported model type: {model_type}. "
                           f"Supported types: {list(OPTIMIZER_CONFIGS.keys())}")
        
        # Get configurations
        self.optimizer_config = self._get_config(OPTIMIZER_CONFIGS)
        self.scheduler_config = self._get_config(SCHEDULER_CONFIGS)
        self.gradient_config = self._get_config(GRADIENT_CONFIGS)
        
        # Initialize optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        logger.info(f"Initialized optimizer manager for {model_type}")
        logger.info(f"Optimizer: {type(self.optimizer).__name__}")
        logger.info(f"Scheduler: {type(self.scheduler).__name__}")
    
    def _get_config(self, config_dict: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Get configuration for current model type with custom overrides."""
        base_config = config_dict[self.model_type].copy()
        
        # Apply custom overrides
        config_key = f"{self.model_type}_config"
        if config_key in self.custom_config:
            base_config.update(self.custom_config[config_key])
        
        return base_config
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer based on configuration."""
        optimizer_type = self.optimizer_config['optimizer_type'].lower()
        
        # Get model parameters
        parameters = self.model.parameters()
        
        if optimizer_type == 'adamw':
            optimizer = optim.AdamW(
                parameters,
                lr=self.optimizer_config['lr'],
                weight_decay=self.optimizer_config['weight_decay'],
                betas=self.optimizer_config['betas'],
                eps=self.optimizer_config['eps'],
                amsgrad=self.optimizer_config['amsgrad']
            )
        elif optimizer_type == 'adam':
            optimizer = optim.Adam(
                parameters,
                lr=self.optimizer_config['lr'],
                weight_decay=self.optimizer_config['weight_decay'],
                betas=self.optimizer_config['betas'],
                eps=self.optimizer_config['eps'],
                amsgrad=self.optimizer_config['amsgrad']
            )
        elif optimizer_type == 'sgd':
            optimizer = optim.SGD(
                parameters,
                lr=self.optimizer_config['lr'],
                weight_decay=self.optimizer_config['weight_decay'],
                momentum=self.optimizer_config.get('momentum', 0.9),
                nesterov=self.optimizer_config.get('nesterov', True)
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        
        return optimizer
    
    def _create_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler based on configuration."""
        scheduler_type = self.scheduler_config['scheduler_type'].lower()
        
        if scheduler_type == 'exponential':
            scheduler = ExponentialLR(
                self.optimizer,
                gamma=self.scheduler_config['gamma']
            )
        elif scheduler_type == 'step':
            scheduler = StepLR(
                self.optimizer,
                step_size=self.scheduler_config['step_size'],
                gamma=self.scheduler_config['gamma']
            )
        elif scheduler_type == 'multistep':
            scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=self.scheduler_config['milestones'],
                gamma=self.scheduler_config['gamma']
            )
        elif scheduler_type == 'cosine':
            scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.scheduler_config['T_max']
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
        
        return scheduler
    
    def step(self, loss: Optional[torch.Tensor] = None) -> None:
        """
        Perform optimization step with gradient clipping.
        
        Args:
            loss: Loss tensor for backward pass
        """
        if loss is not None:
            loss.backward()
        
        # Apply gradient clipping if configured
        if self.gradient_config['clip_grad_norm']:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.gradient_config['max_norm'],
                norm_type=self.gradient_config['norm_type']
            )
        
        self.optimizer.step()
        self.optimizer.zero_grad()
    
    def scheduler_step(self, epoch: Optional[int] = None, metrics: Optional[float] = None) -> None:
        """
        Step the learning rate scheduler.
        
        Args:
            epoch: Current epoch (for some schedulers)
            metrics: Validation metrics (for ReduceLROnPlateau)
        """
        if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            if metrics is not None:
                self.scheduler.step(metrics)
        else:
            self.scheduler.step()
    
    def get_lr(self) -> List[float]:
        """Get current learning rates."""
        return [group['lr'] for group in self.optimizer.param_groups]
    
    def state_dict(self) -> Dict[str, Any]:
        """Get state dictionary for checkpointing."""
        return {
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'model_type': self.model_type,
            'optimizer_config': self.optimizer_config,
            'scheduler_config': self.scheduler_config,
            'gradient_config': self.gradient_config
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state dictionary from checkpoint."""
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.scheduler.load_state_dict(state_dict['scheduler'])
        
        # Verify configurations match
        if state_dict['model_type'] != self.model_type:
            logger.warning(f"Model type mismatch: {state_dict['model_type']} vs {self.model_type}")


def create_optimizer_manager(
    model: torch.nn.Module,
    model_type: str,
    custom_config: Optional[Dict[str, Any]] = None
) -> OptimizerManager:
    """
    Factory function to create optimizer manager.
    
    Args:
        model: PyTorch model to optimize
        model_type: Type of model ('cpm_net' or 'cnn_tr_3d')
        custom_config: Optional custom configuration
    
    Returns:
        OptimizerManager: Configured optimizer manager
    """
    return OptimizerManager(model, model_type, custom_config)


def create_adamw_optimizer(
    model: torch.nn.Module,
    lr: float = 0.0001,
    weight_decay: float = 0.01,
    betas: tuple = (0.9, 0.999),
    eps: float = 1e-8
) -> torch.optim.AdamW:
    """
    Create AdamW optimizer with paper-specified parameters.
    
    Args:
        model: PyTorch model to optimize
        lr: Learning rate (default: 0.0001)
        weight_decay: Weight decay coefficient (default: 0.01)
        betas: Adam beta parameters (default: (0.9, 0.999))
        eps: Adam epsilon parameter (default: 1e-8)
    
    Returns:
        torch.optim.AdamW: Configured AdamW optimizer
    """
    return torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
        eps=eps
    )


def create_exponential_scheduler(
    optimizer: torch.optim.Optimizer,
    total_epochs: int = 45,
    start_lr: float = 0.0001,
    end_lr: float = 0.00001
) -> ExponentialLR:
    """
    Create exponential learning rate scheduler for paper-specified decay.
    
    Args:
        optimizer: PyTorch optimizer
        total_epochs: Total number of training epochs (default: 45)
        start_lr: Starting learning rate (default: 0.0001)
        end_lr: Ending learning rate (default: 0.00001)
    
    Returns:
        ExponentialLR: Configured exponential scheduler
    """
    # Calculate gamma for exponential decay
    gamma = (end_lr / start_lr) ** (1 / total_epochs)
    
    return ExponentialLR(optimizer, gamma=gamma)


def get_parameter_groups(
    model: torch.nn.Module,
    weight_decay: float = 0.01,
    no_decay_params: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Create parameter groups with different weight decay settings.
    
    Args:
        model: PyTorch model
        weight_decay: Weight decay for most parameters
        no_decay_params: List of parameter names to exclude from weight decay
    
    Returns:
        List[Dict]: Parameter groups for optimizer
    """
    if no_decay_params is None:
        no_decay_params = ['bias', 'norm', 'bn']
    
    decay_params = []
    no_decay_params_list = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Check if parameter should have weight decay
        should_decay = True
        for no_decay_name in no_decay_params:
            if no_decay_name in name.lower():
                should_decay = False
                break
        
        if should_decay:
            decay_params.append(param)
        else:
            no_decay_params_list.append(param)
    
    parameter_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params_list, 'weight_decay': 0.0}
    ]
    
    return parameter_groups


def calculate_lr_schedule(
    start_lr: float = 0.0001,
    end_lr: float = 0.00001,
    total_epochs: int = 45
) -> List[float]:
    """
    Calculate learning rate schedule for visualization.
    
    Args:
        start_lr: Starting learning rate
        end_lr: Ending learning rate
        total_epochs: Total number of epochs
    
    Returns:
        List[float]: Learning rates for each epoch
    """
    gamma = (end_lr / start_lr) ** (1 / total_epochs)
    
    lr_schedule = []
    current_lr = start_lr
    
    for epoch in range(total_epochs):
        lr_schedule.append(current_lr)
        current_lr *= gamma
    
    return lr_schedule


# Utility functions for gradient analysis
def get_gradient_norm(model: torch.nn.Module, norm_type: float = 2.0) -> float:
    """
    Calculate gradient norm for monitoring.
    
    Args:
        model: PyTorch model
        norm_type: Type of norm to calculate
    
    Returns:
        float: Gradient norm
    """
    total_norm = 0.0
    param_count = 0
    
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
            param_count += 1
    
    if param_count == 0:
        return 0.0
    
    total_norm = total_norm ** (1.0 / norm_type)
    return total_norm


def log_optimizer_state(optimizer: torch.optim.Optimizer, epoch: int) -> None:
    """
    Log optimizer state for debugging.
    
    Args:
        optimizer: PyTorch optimizer
        epoch: Current epoch
    """
    for i, param_group in enumerate(optimizer.param_groups):
        lr = param_group['lr']
        weight_decay = param_group.get('weight_decay', 0.0)
        logger.info(f"Epoch {epoch}, Group {i}: LR={lr:.6f}, WD={weight_decay:.6f}")


# Export main components
__all__ = [
    'OptimizerManager',
    'create_optimizer_manager',
    'create_adamw_optimizer',
    'create_exponential_scheduler',
    'get_parameter_groups',
    'calculate_lr_schedule',
    'get_gradient_norm',
    'log_optimizer_state',
    'OPTIMIZER_CONFIGS',
    'SCHEDULER_CONFIGS',
    'GRADIENT_CONFIGS'
]