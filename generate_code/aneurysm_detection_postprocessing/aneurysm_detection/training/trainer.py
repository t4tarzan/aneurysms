"""Training utilities for CPM-Net."""
import torch

class CPNTrainer:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters())
    
    def train_step(self, batch):
        self.model.train()
        images = batch['image'].to(self.device)
        targets = {
            'cls_map': batch['cls_map'].to(self.device),
            'reg_map': batch['reg_map'].to(self.device)
        }
        
        self.optimizer.zero_grad()
        outputs = self.model(images)
        loss_dict = self.model.loss(outputs, targets)
        loss = loss_dict['loss']
        
        loss.backward()
        self.optimizer.step()
        return loss_dict
    
    @torch.no_grad()
    def val_step(self, batch):
        self.model.eval()
        images = batch['image'].to(self.device)
        targets = {
            'cls_map': batch['cls_map'].to(self.device),
            'reg_map': batch['reg_map'].to(self.device)
        }
        
        outputs = self.model(images)
        return self.model.loss(outputs, targets)
    
    def save_checkpoint(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
