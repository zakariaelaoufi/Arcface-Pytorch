# Face Recognition with EfficientNet Backbone and ArcFace Loss

A PyTorch implementation of a face recognition system using a custom EfficientNet-inspired backbone with ArcFace loss for training on the VGGFace2 dataset.

## Overview

This project implements a deep learning model for face recognition that combines:
- **Custom EfficientNet-inspired backbone**: Lightweight architecture using depthwise separable convolutions
- **ArcFace loss**: Angular margin loss for improved face recognition performance
- **VGGFace2 dataset**: Large-scale face recognition dataset with 540 identities (for this vggface2 dataset)

## Features

- ✅ Efficient backbone architecture with residual connections
- ✅ ArcFace loss implementation for angular margin learning
- ✅ Data augmentation pipeline
- ✅ Train/validation/test split with stratification
- ✅ Learning rate scheduling
- ✅ Model checkpointing and resuming
- ✅ Comprehensive training metrics tracking

## Architecture

### Model Components

1. **EfficientFR Backbone** (`models/EfficientFRBackbone.py`)
   - Depthwise separable convolutions for efficiency
   - Residual blocks for gradient flow
   - Adaptive global average pooling
   - 512-dimensional embedding output

2. **ArcFace Loss** (`models/ArcFace.py`)
   - Angular margin penalty for improved discriminative learning
   - Configurable scale (s) and margin (m) parameters
   - L2 normalized embeddings and weights

3. **FaceNet Model** (`models/FaceNet.py`)
   - Combines backbone and ArcFace components
   - Supports both training (with labels) and inference modes

## Dataset Structure

The code expects VGGFace2 dataset in the following structure:
```
vggface2/
├── train/
│   ├── identity1/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── identity2/
│   └── ...
└── test/
    ├── identity1/
    └── ...
```

## Installation

### Requirements

```bash
pip install torch torchvision
pip install opencv-python
pip install pandas numpy matplotlib
pip install scikit-learn
pip install tqdm
pip install opendatasets  # For Kaggle dataset download
```

### Dataset Download

```python
import opendatasets as od
od.download('https://www.kaggle.com/datasets/hearfool/vggface2')
```

## Usage

### Training

1. **Configure paths and parameters** in `train.py`:
   ```python
   path_data = '/content/vggface2'  # Update to your dataset path
   BATCH_SIZE = 128
   EPOCHS = 60
   learning_rate = 1e-3
   ```

2. **Run training**:
   ```bash
   python train.py
   ```

3. **Resume from checkpoint** (optional):
   ```python
   resume = True  # Set to True in train.py
   checkpoint_path = '/path/to/checkpoint'
   ```

### Data Loading

The data pipeline (`data/DataLoader.py`) automatically:
- Generates dataframe from VGGFace2 directory structure
- Creates train/validation/test splits (76.8%/7.2%/4.8%)
- Applies data augmentation to 75% of training data
- Creates PyTorch DataLoaders with specified batch size

### Inference

```python
import torch
from models.FaceNet import FaceNet
from data.Utils import generate_perona_emnedding, get_cosine_sim

# Load trained model
model = FaceNet(num_classes=540, embedding_dim=512)
model.load_state_dict(torch.load('model_checkpoint.pth'))
model.eval()

# Generate embeddings
embedding = generate_perona_emnedding(model, image)

# Compare faces
similarity = get_cosine_sim(model, image1, image2)
```

## Model Configuration

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Embedding Dimension | 512 | Face embedding vector size |
| ArcFace Scale (s) | 30.0 | Scaling factor for logits |
| ArcFace Margin (m) | 0.5 | Angular margin penalty |
| Batch Size | 128 | Training batch size |
| Learning Rate | 1e-3 | Initial learning rate |
| Image Size | 224×224 | Input image resolution |

### Data Augmentation

- Random horizontal flip
- Random rotation (±10 degrees)
- Color space conversion (BGR → RGB)
- Tensor normalization

## Training Details

### Loss Function
- **Primary**: CrossEntropyLoss on ArcFace outputs
- **Optimization**: Adam optimizer
- **Scheduling**: ReduceLROnPlateau (factor=0.5, patience=2)

### Training Process
1. **Data Splits**: Stratified splitting to maintain class balance
2. **Augmentation**: 75% of training data gets augmented
3. **Validation**: Monitored for learning rate scheduling
4. **Checkpointing**: Automatic saving every 10 epochs

### Metrics Tracked
- Training/Validation Loss
- Training/Validation Accuracy  
- Learning Rate
- Training Time

## File Structure

```
├── data/
│   ├── DataLoader.py      # Data loading and preprocessing
│   ├── Dataset.py         # Custom PyTorch Dataset class
│   └── Utils.py          # Utility functions
├── models/
│   ├── ArcFace.py        # ArcFace loss implementation
│   ├── EfficientFRBackbone.py  # Backbone architecture
│   └── FaceNet.py        # Complete model wrapper
└── train.py              # Training script
```

## Results

The model tracks the following metrics during training:
- **Training accuracy**: Classification accuracy on training set
- **Validation accuracy**: Classification accuracy on validation set
- **Loss curves**: Both training and validation loss
- **Learning rate**: Adaptive learning rate scheduling

Model checkpoints are saved every 10 epochs and include:
- Model state dict
- Optimizer state dict
- Scheduler state dict
- Training history

## Key Features

### Efficient Architecture
- **Depthwise Separable Convolutions**: Reduced parameters while maintaining performance
- **Residual Connections**: Improved gradient flow and training stability
- **Adaptive Pooling**: Handles variable input sizes effectively

### Robust Training
- **Stratified Splitting**: Maintains class distribution across splits
- **Data Augmentation**: Improves generalization with geometric transforms
- **Learning Rate Scheduling**: Adaptive learning rate based on validation performance
- **Checkpointing**: Resume training from any saved checkpoint

### Face Recognition Pipeline
- **Embedding Generation**: 512-dimensional L2-normalized face embeddings
- **Similarity Computation**: Cosine similarity for face matching
- **Euclidean Distance**: Alternative distance metric implementation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is open source and available under the [MIT License](https://github.com/zakariaelaoufi/arcface-pytorch/blob/main/LICENSE).

## Acknowledgments

- [VGGFace2 dataset](https://www.kaggle.com/datasets/hearfool/vggface2)
- [ArcFace paper authors](https://arxiv.org/abs/1801.07698)
- (PyTorch)[https://pytorch.org/]

---

**Note**: Make sure to update the `path_data` variable in the configuration files to match your local dataset path before training.
