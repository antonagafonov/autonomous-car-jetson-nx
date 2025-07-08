# 🚗 ResNet18 Angular Sequence Prediction Training

A comprehensive PyTorch training pipeline for predicting angular velocity sequences from driving images using ResNet18 architecture with advanced visualization and monitoring capabilities.

## 📋 Overview

This training script implements a computer vision model that predicts future angular velocity (steering) commands from single camera images. The model learns to predict sequences of 10, 30, or 60 future angular_z values, making it suitable for autonomous driving trajectory prediction.

### 🎯 Key Features
- **ResNet18-based architecture** with custom regression head
- **Sequence prediction** for 10, 30, or 60 future time steps
- **Advanced data augmentation** optimized for driving scenarios
- **GradCAM visualizations** with configurable layer selection
- **TensorBoard integration** for real-time monitoring
- **Xavier NX optimized** settings for edge deployment
- **Automatic experiment management** with organized output structure
- **Horizontal flip augmentation** with steering command inversion

## 🛠️ Installation

### Prerequisites
```bash
# Core requirements
torch >= 1.11.0
torchvision >= 0.12.0
pandas
numpy
matplotlib
Pillow
argparse

# Optional (for TensorBoard)
tensorboard==2.8.0  # Recommended for PyTorch 1.11.0
absl-py
grpcio
protobuf
```

### Xavier NX Installation
```bash
# Install TensorBoard (compatible with PyTorch 1.11.0)
pip3 install tensorboard==2.8.0

# Install missing dependencies if needed
pip3 install absl-py grpcio protobuf
```

## 📊 Dataset Format

The training script expects a CSV file with preprocessed angular velocity sequences:

### Required Columns
```csv
image_filename,manual_angular_z_next_10,manual_angular_z_next_30,manual_angular_z_next_60
image_000300.png,"[0.0, 0.0, 1.9745, ...]","[0.0, 0.0, 1.9745, ...]","[0.0, 0.0, 1.9745, ...]"
```

### Image Requirements
- **Format**: PNG/JPG images
- **Input size**: 640×480×3 (will be preprocessed to 224×224×3)
- **Content**: Front-facing driving camera images

## 🎛️ Command Line Arguments

### Basic Training Parameters
```bash
--csv_path              # Path to CSV file with annotations
--images_dir            # Directory containing training images  
--sequence_length       # Prediction horizon: 10, 30, or 60 (default: 10)
--batch_size           # Training batch size (default: 32 for Xavier NX)
--num_epochs           # Number of training epochs (default: 50)
--learning_rate        # Learning rate (default: 1e-4)
```

### Data Preprocessing
```bash
--use_vertical_crop    # Enable vertical cropping (removes sky/hood)
--crop_pixels         # Pixels to remove from top/bottom (default: 100)
```

### Experiment Management
```bash
--save_dir            # Base directory for results (default: /home/toon/train_results)
--experiment_name     # Custom experiment name (auto-generated if not provided)
```

### Visualization & Monitoring
```bash
--gradcam_freq        # GradCAM generation frequency in epochs (default: 1)
--gradcam_layer       # ResNet layer for GradCAM: layer1-4 (default: layer4)
```

## 🚀 Usage Examples

### Basic Training
```bash
python train_w_GC_angular_only.py \
    --csv_path /path/to/dataset.csv \
    --images_dir /path/to/images \
    --sequence_length 10
```

### Optimized for Xavier NX
```bash
python train_w_GC_angular_only.py \
    --batch_size 32 \
    --sequence_length 10 \
    --use_vertical_crop \
    --crop_pixels 100 \
    --gradcam_freq 5
```

### Advanced Configuration
```bash
python train_w_GC_angular_only.py \
    --csv_path /data/driving_dataset.csv \
    --images_dir /data/images \
    --sequence_length 30 \
    --batch_size 32 \
    --num_epochs 100 \
    --learning_rate 5e-5 \
    --use_vertical_crop \
    --crop_pixels 120 \
    --experiment_name "highway_driving_v2" \
    --gradcam_layer layer3 \
    --gradcam_freq 10
```

## 🏗️ Model Architecture

### ResNet18 Backbone
- **Pretrained**: ImageNet initialization
- **Input**: 224×224×3 images
- **Feature extraction**: Standard ResNet18 layers

### Custom Regression Head
```
ResNet18 Features (512) → FC(1024) → ReLU → Dropout(0.2)
                       → FC(512)  → ReLU → Dropout(0.2)  
                       → FC(256)  → ReLU → Dropout(0.2)
                       → FC(sequence_length)
```

### Output
- **10-frame model**: 10 angular_z predictions
- **30-frame model**: 30 angular_z predictions  
- **60-frame model**: 60 angular_z predictions

## 🖼️ Image Processing Pipeline

### 1. Vertical Cropping (Optional)
```
Original: 640×480×3
    ↓ Remove 100px top/bottom
Cropped: 640×280×3 (focuses on road area)
```

### 2. Data Augmentation (Training)
- **Horizontal flip**: 50% probability with steering inversion
- **Gaussian blur**: Random blur simulation
- **Color jitter**: Brightness, contrast, saturation, hue variations
- **Sharpening**: Random image sharpening
- **Posterization/Solarization**: Robustness augmentations

### 3. Final Processing
```
Cropped image → Resize(224×224) → Normalize → Model
```

## 📁 Output Structure

Each training run creates an organized experiment directory:

```
/home/toon/train_results/
└── angular_seq10_20250706_143022/
    ├── config.json                    # Complete training configuration
    ├── training_summary.json          # Final metrics and results
    ├── models/
    │   └── resnet18_angular_control_model_10frames.pth
    ├── plots/
    │   ├── training_curves.png         # Loss curves
    │   ├── predictions_vs_ground_truth.png
    │   ├── sequence_evolution_examples.png
    │   └── gradcam_angular_epoch_*.png # GradCAM visualizations
    └── tensorboard/                   # TensorBoard logs
        └── events.out.tfevents.*
```

## 🔍 GradCAM Visualization

### Layer Selection
| Layer | Semantic Level | Output Size | Best For |
|-------|---------------|-------------|----------|
| **layer4** | Highest | 7×7 | Scene understanding (recommended) |
| layer3 | Medium | 14×14 | Balanced detail vs semantics |
| layer2 | Lower | 28×28 | Edge/texture analysis |
| layer1 | Lowest | 56×56 | Fine detail analysis |

### Visualization Components
1. **Model Input**: Preprocessed image (what model sees)
2. **Attention Heatmap**: Where model focuses
3. **Overlay**: Attention overlaid on input with predictions

## 📊 TensorBoard Monitoring

### Real-time Metrics
- Training/Validation loss curves
- Learning rate tracking
- Batch-level loss monitoring

### Usage
```bash
# Start TensorBoard
tensorboard --logdir /home/toon/train_results/experiment_name/tensorboard

# View in browser
http://localhost:6006
```

## ⚡ Xavier NX Optimization

### Recommended Settings
```bash
# Performance mode
sudo nvpmodel -m 2        # MAX mode (15W)
sudo jetson_clocks        # Lock clocks to maximum

# Monitoring
tegrastats                # Monitor GPU/CPU/temp
watch -n 1 nvidia-smi     # Monitor GPU memory
```

### Memory-Optimized Parameters
- **Batch size**: 32 (optimal for 8GB memory)
- **Workers**: 2 (optimized for 6-core ARM CPU)
- **Pin memory**: Enabled for faster CPU→GPU transfer

### Expected Performance
- **Memory usage**: ~0.7-0.8GB GPU memory
- **Training speed**: ~2-3 seconds per epoch (depends on dataset size)

## 🔧 Troubleshooting

### TensorBoard Issues
```bash
# Missing dependencies
pip3 install absl-py grpcio protobuf

# Version conflicts
pip3 install tensorboard==2.8.0 --force-reinstall
```

### Memory Issues
```bash
# Reduce batch size
--batch_size 16

# Reduce workers
# Edit code: num_workers=1

# Disable GradCAM during training
--gradcam_freq 999
```

### Dataset Issues
```bash
# Check CSV format
head -n 2 your_dataset.csv

# Verify image paths
ls /path/to/images/ | head -n 5

# Check sequence columns exist
python -c "import pandas as pd; df=pd.read_csv('dataset.csv'); print(df.columns)"
```

## 📈 Training Tips

### Learning Rate Schedule
- **Start**: 1e-4 (default)
- **Automatic**: ReduceLROnPlateau (halves when validation plateaus)
- **Manual**: Experiment with 5e-5 for fine-tuning

### Sequence Length Selection
- **10 frames**: ~333ms prediction (30 FPS)
- **30 frames**: ~1s prediction (good for highway)
- **60 frames**: ~2s prediction (long-term planning)

### Data Augmentation
- **Horizontal flip**: Critical for steering generalization
- **Vertical crop**: Removes irrelevant sky/dashboard
- **Color augmentation**: Handles lighting variations

## 🎯 Best Practices

### Experiment Organization
1. Use descriptive experiment names
2. Save configuration files for reproducibility
3. Monitor validation loss for early stopping
4. Generate GradCAM regularly to verify attention

### Model Validation
1. Check GradCAM focuses on road features
2. Verify predictions are reasonable magnitude
3. Monitor overfitting through validation curves
4. Test on held-out sequences

### Production Deployment
1. Save model with full configuration
2. Test inference speed on target hardware
3. Validate preprocessing pipeline matches training
4. Monitor model attention in deployment

## 🤝 Contributing

1. Follow the established code structure
2. Add appropriate documentation
3. Test on Xavier NX hardware
4. Validate with existing datasets

## 📄 License

[Specify your license here]

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Verify your dataset format
3. Test with smaller batch sizes
4. Monitor system resources during training

---

**Happy Training! 🚗💨**