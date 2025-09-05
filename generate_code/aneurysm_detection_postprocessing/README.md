# Aneurysm Detection Post-Processing Implementation

This repository implements the complete system described in the paper "Automated anatomy-based post-processing reduces false positives and improved interpretability of deep learning intracranial aneurysm detection".

## 🎯 Overview

The implementation provides an anatomy-based post-processing method using brain, artery, vein, and cavernous venous sinus masks to reduce false positives in deep learning aneurysm detection models without losing true positives.

## 📁 Project Structure

```
aneurysm_detection_postprocessing/
├── models/                     # Deep Learning Detection Models
│   ├── base_detector.py       # Abstract base class for detectors
│   ├── cpm_net.py            # CPM-Net 3D detector implementation
│   └── cnn_tr_3d.py          # 3D-CNN-TR hybrid model implementation
├── segmentation/              # Anatomical Segmentation Modules
│   ├── brain_segmentation.py # TotalSegmentator brain mask generation
│   ├── artery_vein_seg.py    # nnUNet-based artery-vein segmentation
│   ├── cvs_segmentation.py   # CVS mask using ANT registration
│   └── mask_utils.py         # Mask processing utilities
├── postprocessing/            # Post-processing Pipeline
│   ├── anatomical_filter.py  # Main post-processing pipeline
│   ├── method_implementations.py # Five post-processing methods
│   └── overlap_calculator.py # Bounding box overlap calculations
├── data/                      # Data Processing Components
│   ├── dataset_loader.py     # CTA data loading and preprocessing
│   ├── annotation_parser.py  # 3D bounding box annotation handling
│   └── transforms.py         # Data augmentation and normalization
├── evaluation/                # Evaluation System
│   ├── metrics.py            # TP/FP/FN calculation and analysis
│   ├── fp_analyzer.py        # False positive categorization
│   └── performance_evaluator.py # Complete evaluation pipeline
├── training/                  # Training System
│   ├── trainer.py            # Model training orchestration
│   ├── loss_functions.py     # Detection loss functions
│   └── optimization.py       # AdamW optimizer configuration
└── experiments/               # Experiment Scripts
    ├── run_training.py       # Training script for both models
    ├── run_evaluation.py     # Evaluation with post-processing
    └── reproduce_results.py  # Full paper reproduction script
```

## 🚀 Key Features

### 1. Deep Learning Detection Models
- **CPM-Net**: CNN-based 3D aneurysm detector using center-points matching
- **3D-CNN-TR**: Deformable 3D CNN-Transformer hybrid with artery segmentation auxiliary input

### 2. Anatomical Segmentation
- **Brain Mask**: TotalSegmentator with 3.6mm dilation
- **Artery-Vein Segmentation**: nnUNet framework for 4D dynamic CTA
- **CVS Segmentation**: ANT registration with 3.2mm expansion

### 3. Five Post-processing Methods
1. Remove bounding boxes outside brain mask
2. Remove bounding boxes with ANY overlap with vein mask
3. Remove bounding boxes if overlap_vein > overlap_artery
4. Combine Method 1 AND Method 2
5. Combine Method 1 AND Method 3

### 4. Comprehensive Evaluation
- TP/FP/FN calculation with IoU thresholding
- False positive categorization and analysis
- Performance comparison across methods
- Visualization and reporting

## 📋 Requirements

### Core Dependencies
```bash
pip install torch torchvision numpy scipy matplotlib seaborn
pip install nibabel pydicom SimpleITK scikit-image
pip install pathlib json logging warnings typing dataclasses
```

### Optional Dependencies (for full functionality)
```bash
# For TotalSegmentator
pip install totalsegmentator

# For nnUNet
pip install nnunet

# For ANT registration
# Install ANTs from: https://github.com/ANTsX/ANTs
```

## 🔧 Usage

### Quick Start
```python
from aneurysm_detection_postprocessing.models import create_cpm_net, create_cnn_tr_3d
from aneurysm_detection_postprocessing.postprocessing import AnatomicalFilter
from aneurysm_detection_postprocessing.evaluation import PerformanceEvaluator

# Create models
cpm_model = create_cpm_net()
cnn_tr_model = create_cnn_tr_3d()

# Set up post-processing
filter_pipeline = AnatomicalFilter()

# Set up evaluation
evaluator = PerformanceEvaluator()
```

### Training Models
```python
from aneurysm_detection_postprocessing.experiments.run_training import main as train_main

# Train CPM-Net
train_main(model_type='cpm_net', data_dir='path/to/data')

# Train CNN-TR-3D
train_main(model_type='cnn_tr_3d', data_dir='path/to/data')
```

### Running Evaluation
```python
from aneurysm_detection_postprocessing.experiments.run_evaluation import main as eval_main

# Evaluate with post-processing
eval_main(model_path='path/to/model.pth', test_data='path/to/test')
```

### Full Paper Reproduction
```python
from aneurysm_detection_postprocessing.experiments.reproduce_results import main as reproduce_main

# Reproduce all paper results
reproduce_main(config_path='path/to/config.yaml')
```

## 📊 Model Specifications

### CPM-Net Configuration
- **Architecture**: 3D CNN with center-points matching
- **Parameters**: ~23M parameters
- **Training**: 45 epochs, AdamW optimizer
- **Input**: CTA volume patches
- **Output**: 3D bounding boxes with confidence scores

### 3D-CNN-TR Configuration
- **Architecture**: Deformable 3D CNN + Transformer
- **Parameters**: ~36M parameters
- **Training**: 45 epochs, AdamW optimizer
- **Input**: CTA volume + auxiliary artery segmentation
- **Output**: 3D bounding boxes with confidence scores

## 🎯 Key Results (from Paper)

The anatomy-based post-processing methods achieve:
- Significant reduction in false positives
- Maintained sensitivity for true aneurysms
- Improved interpretability of detection results
- Better clinical applicability

## 🔬 Research Applications

This implementation enables:
- **Medical AI Research**: Advanced aneurysm detection with anatomical constraints
- **Clinical Validation**: Testing post-processing methods on real CTA data
- **Method Comparison**: Benchmarking different anatomical filtering approaches
- **Educational Use**: Understanding anatomy-based medical AI systems

## 📝 Citation

If you use this implementation, please cite the original paper:
```
[Paper citation information would go here]
```

## 🤝 Contributing

This implementation follows the exact specifications from the research paper. For improvements or extensions, please ensure compatibility with the original methodology.

## 📄 License

This implementation is provided for research and educational purposes. Please refer to the original paper for usage guidelines and restrictions.

---

**Implementation Status**: ✅ Complete - All components implemented and tested
**Last Updated**: 2025-01-XX
**Version**: 1.0.0