"""Experiment tracking with Weights & Biases."""
from typing import Dict, Any, Optional
import wandb
from pathlib import Path

class ExperimentTracker:
    """Track experiments using Weights & Biases."""
    
    def __init__(
        self,
        project: str = "aneurysm-detection",
        entity: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        log_dir: str = "runs",
        **kwargs
    ):
        """Initialize the experiment tracker."""
        self.project = project
        self.entity = entity
        self.config = config or {}
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize wandb
        self.run = wandb.init(
            project=self.project,
            entity=self.entity,
            config=self.config,
            dir=str(self.log_dir),
            **kwargs
        )
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to wandb."""
        if self.run is not None:
            self.run.log(metrics, step=step)
    
    def log_config(self, config: Dict[str, Any]):
        """Log configuration parameters."""
        if self.run is not None:
            self.run.config.update(config)
    
    def log_artifact(self, file_path: str, name: str, artifact_type: str = "model"):
        """Log an artifact (e.g., model weights)."""
        if self.run is not None:
            artifact = wandb.Artifact(name, type=artifact_type)
            artifact.add_file(file_path)
            self.run.log_artifact(artifact)
    
    def watch(self, model, log: str = "gradients", log_freq: int = 100):
        """Watch a PyTorch model's gradients and parameters."""
        if self.run is not None:
            wandb.watch(model, log=log, log_freq=log_freq)
    
    def finish(self):
        """Finish the run."""
        if self.run is not None:
            wandb.finish()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()
