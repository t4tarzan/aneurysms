#!/usr/bin/env python3
"""Training script for CPM-Net."""
import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose

from aneurysm_detection.models.cpm_net import create_cpm_net
from aneurysm_detection.training.trainer import CPNTrainer
from aneurysm_detection.data.dataset_loader import AneurysmDataset  # Assuming this exists
from aneurysm_detection.data.transforms import (  # Assuming these exist
    ToTensor, Normalize, RandomFlip3D, RandomRotate3D
)

def create_data_loaders(data_dir, batch_size=2, val_split=0.1):
    """Create training and validation data loaders."""
    # Define transforms
    train_transforms = Compose([
        RandomFlip3D(),
        RandomRotate3D(degrees=15),
        ToTensor(),
        Normalize(mean=0.5, std=0.5)
    ])
    
    val_transforms = Compose([
        ToTensor(),
        Normalize(mean=0.5, std=0.5)
    ])
    
    # Create datasets
    full_dataset = AneurysmDataset(
        data_dir=data_dir,
        split='train',
        transform=train_transforms
    )
    
    # Split into train/val
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader

def main():
    # Configuration
    data_dir = 'path/to/your/data'
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create model and trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = create_cpm_net(in_c=1, base_c=32, num_classes=1)
    trainer = CPNTrainer(model, device=device)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(data_dir)
    
    # Training loop
    num_epochs = 50
    for epoch in range(num_epochs):
        # Train for one epoch
        for batch in train_loader:
            loss_dict = trainer.train_step(batch)
            print(f"Epoch {epoch+1}/{num_epochs}, "
                  f"Loss: {loss_dict['loss'].item():.4f}, "
                  f"Cls: {loss_dict['loss_cls'].item():.4f}, "
                  f"Reg: {loss_dict['loss_reg'].item():.4f}")
        
        # Validate
        if val_loader is not None:
            val_loss = 0.0
            for batch in val_loader:
                loss_dict = trainer.val_step(batch)
                val_loss += loss_dict['loss'].item()
            val_loss /= len(val_loader)
            print(f"Validation Loss: {val_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'cpm_net_epoch_{epoch+1}.pth')
            trainer.save_checkpoint(checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

if __name__ == '__main__':
    main()
