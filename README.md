# HardGNN: Hard Negative Sampling Enhancement for SelfGNN

## Overview
HardGNN is an enhanced version of SelfGNN that incorporates hard negative sampling with InfoNCE contrastive loss to improve sequential recommendation performance. This implementation builds upon the original SelfGNN framework and adds sophisticated negative sampling strategies for better learning efficiency.

## Attribution & Citation

### Based on SelfGNN
This work builds upon the SelfGNN framework. If you use this code, please cite both the original SelfGNN paper and this work:

**Original SelfGNN Paper:**
```bibtex
@article{liu2024selfgnn,
  title={SelfGNN: Self-Supervised Graph Neural Networks for Sequential Recommendation},
  author={Liu, Yuxi and Xia, Lianghao and Huang, Chao},
  journal={arXiv preprint arXiv:2405.20878},
  year={2024}
}
```

**Original SelfGNN Repository:**
- Repository: https://github.com/HKUDS/SelfGNN
- Authors: Yuxi Liu, Lianghao Xia, Chao Huang
- License: Apache-2.0

## Key Enhancements

### Hard Negative Sampling
- **Cosine Similarity-based Selection**: Uses cosine similarity to identify the most challenging negative samples
- **Top-K Hard Negatives**: Selects K=5 hardest negatives for each positive sample
- **Adaptive Masking**: Excludes positive items from negative candidate pool

### InfoNCE Contrastive Loss
- **Temperature Scaling**: Uses τ=0.1 for optimal discrimination
- **Contrastive Learning**: Enhances representation quality through contrastive learning
- **Weighted Integration**: λ=0.1 weight for balancing with recommendation loss

### Technical Features
- **TensorFlow 1.x Compatibility**: Adapted for TF1-style execution in TF2 environments
- **Multi-Graph GNN**: Time-series based graph construction
- **LSTM Sequence Modeling**: Temporal user behavior modeling
- **Multi-Head Attention**: Advanced attention mechanisms for user-item interactions

## 🚀 Quick Start

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# For Google Colab users, run the environment setup
python setup_environment.py
```

### Data Preparation
The datasets are organized in the `./Datasets` folder:
```
./Datasets/amazon(yelp/movielens/gowalla)/
├── sequence          # user behavior sequences (List)
├── test_dict         # test item for each users (Dict)  
├── trn_mat_time      # user-item graphs in different periods (sparse matrix)
└── tst_int           # users to be test (List)
```

### Training Examples

#### Quick Examples with Unified Script

#### Unified Script (New & Recommended)
The new `run_hardgnn.py` provides a clean interface with validation and grid search:

```bash
# Single experiment with validation
python run_hardgnn.py --dataset amazon --k 5 --lambda 0.1 --epochs 100

# Grid search over hyperparameters
python run_hardgnn.py --dataset yelp --grid-search --k-values 3,5,7 --lambda-values 0.01,0.1,0.2 --epochs 50

# Enhanced main script with validation
python main.py --dataset amazon --k 5 --lambda 0.1 --epochs 100 --validate
```

#### Legacy Configuration
```bash
python main.py \
    --data amazon \
    --use_hard_neg True \
    --hard_neg_top_k 5 \
    --contrastive_weight 0.1 \
    --temp 0.1 \
    --latdim 64 \
    --batch 512 \
    --epoch 100
```

## 🔧 Configuration Parameters

### Hard Negative Sampling Parameters
- `--use_hard_neg`: Enable/disable hard negative sampling (default: False)
- `--hard_neg_top_k`: Number of hard negatives to sample (default: 5)
- `--contrastive_weight`: Weight for contrastive loss (default: 0.1)
- `--temp`: Temperature for InfoNCE loss (default: 0.1)

### Model Parameters
- `--latdim`: Embedding dimension (default: 64)
- `--batch`: Batch size (default: 512)
- `--lr`: Learning rate (default: 1e-3)
- `--epoch`: Number of training epochs (default: 100)
- `--graphNum`: Number of time-based graphs (default: 8)
- `--ssl_reg`: Self-supervised learning regularization (default: 1e-4)

## 📊 Testing & Validation

### Contrastive Loss Validation
```bash
# Test the contrastive loss component
python test_contrastive_loss.py
```

This script validates:
- ✅ Hard negative sampling effectiveness
- ✅ InfoNCE loss discrimination power  
- ✅ Similarity gap between positives and negatives
- ✅ Overall contrastive learning quality

### Expected Output
```
🔍 Starting Contrastive Loss Validation...
📊 Configuration: τ=0.1, K=5, λ=0.1
✅ Positive similarities > Negative similarities
✅ Hard negative sampling working (high similarity negatives)
✅ Good discriminative power
```

## 📁 Project Structure

```
HardGNN_Standalone/
├── README.md                     # This file
├── LICENSE                       # Apache-2.0 license
├── requirements.txt              # Python dependencies
├── run_hardgnn.py               # 🆕 Unified experiment runner with grid search
├── main.py                       # Enhanced main training script  
├── model.py                      # Enhanced SelfGNN with HardGNN
├── Params.py                     # Enhanced parameter configuration
├── DataHandler.py                # Data loading and processing
├── test_contrastive_loss.py      # Contrastive loss validation
├── setup_environment.py          # Environment setup for Colab
├── run_hardgnn_example.bat      # 🆕 Windows usage examples
├── Utils/                        # Utility functions (enhanced)
│   ├── hardgnn_utils.py         # 🆕 Unified utility module (all functions)
│   ├── NNLayers.py              # Neural network layers
│   ├── TimeLogger.py            # Logging utilities
│   └── attention.py             # Attention mechanisms
├── Datasets/                     # Dataset storage
├── Models/                       # Saved model checkpoints
├── History/                      # Training history and metrics
├── preprocess_to_sequence.ipynb  # Data preprocessing notebook
└── preprocess_to_trnmat.ipynb   # Matrix preprocessing notebook
```

## 🔧 Unified Utilities Module

The `Utils/hardgnn_utils.py` module consolidates all utility functions for streamlined usage:

### Core Components
- **Dataset Configuration**: Validated parameter sets for all datasets (Yelp, Amazon, Gowalla, MovieLens)
- **TensorFlow Setup**: GPU optimization, session management, and v1 compatibility handling
- **Training Pipeline**: GPU-optimized epochs with background data loading and pipeline parallelization
- **Grid Search**: Comprehensive hyperparameter exploration with memory management
- **Validation Suite**: Model setup, hyperparameter, and contrastive loss validation
- **Result Management**: JSON/CSV output, performance tracking, and experiment summaries

### Key Functions
```python
from Utils.hardgnn_utils import (
    configure_dataset,           # Configure dataset-specific parameters
    setup_tensorflow_session,    # Optimized TF session configuration
    run_single_experiment,       # Execute single experiment with validation
    run_grid_search,            # Comprehensive hyperparameter search
    validate_model_setup,       # Model component validation
    validate_contrastive_loss   # Hard negative sampling validation
)
```

## 🔬 Implementation Details

### Hard Negative Sampling Algorithm
1. **User-Item Similarity Computation**: Calculate cosine similarities between user embeddings and all item embeddings
2. **Positive Masking**: Mask out positive items to prevent selection as negatives
3. **Top-K Selection**: Select K items with highest similarity (excluding positives) as hard negatives
4. **Contrastive Loss Computation**: Apply InfoNCE loss with temperature scaling

### Model Architecture Enhancements
- **Base SelfGNN**: Multi-graph GNN with temporal modeling
- **LSTM Integration**: Sequence modeling for user behavior patterns  
- **Multi-Head Attention**: Enhanced attention mechanisms
- **Hard Negative Sampling**: Integrated into training pipeline
- **InfoNCE Loss**: Added to overall loss function

## 📈 Expected Performance

The HardGNN enhancement is designed to improve upon baseline SelfGNN performance:
- **Better Discrimination**: Hard negatives provide more challenging training samples
- **Improved Representations**: InfoNCE loss enhances embedding quality
- **Faster Convergence**: More efficient learning from difficult examples

## 🐛 Troubleshooting

### Common Issues
1. **TensorFlow Compatibility**: Ensure TF 1.x compatibility mode is enabled
2. **Memory Issues**: Reduce batch size or embedding dimensions if OOM occurs
3. **Dataset Format**: Ensure datasets follow the expected format (see preprocessing notebooks)
4. **GPU Configuration**: Check GPU memory growth settings

### Debug Mode
```bash
# Enable detailed logging
python main.py --data amazon --use_hard_neg True --debug True
```

## 📊 Evaluation Metrics

The model tracks the following metrics:
- **HR@10**: Hit Rate at 10
- **NDCG@10**: Normalized Discounted Cumulative Gain at 10
- **Contrastive Loss**: InfoNCE contrastive loss value
- **Recommendation Loss**: Main recommendation loss
- **Total Loss**: Combined loss with regularization

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

This work is based on SelfGNN, which is also licensed under Apache-2.0. Original copyright notices and attributions are preserved.

## Acknowledgments

We thank the authors of SelfGNN for providing the foundational framework:
- Yuxi Liu (HKUDS)
- Lianghao Xia (HKUDS) 
- Chao Huang (HKUDS)

Original SelfGNN repository: https://github.com/HKUDS/SelfGNN 