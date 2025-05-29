# HardGNN for Google Colab Pro+

## Overview
This repository contains a **TensorFlow 2.x compatible** implementation of HardGNN (Hard Negative Sampling for Sequential Recommendation) designed specifically for **Google Colab Pro+** environments. The model enhances SelfGNN with sophisticated hard negative sampling using InfoNCE contrastive loss.

## 🎯 Key Features
- **Enhanced SelfGNN** with hard negative sampling (τ=0.1, K=5, λ=0.1)
- **Full TensorFlow 2.x compatibility** via tf.compat.v1 layer
- **Google Colab Pro+ optimization** with GPU acceleration support
- **Memory-efficient** hard negative sampling for Colab constraints
- **Dataset-agnostic** design supporting Amazon-book, Yelp, Gowalla, MovieLens
- **One-command setup** and execution

## 🏗️ Architecture

**Main Model**: `HardGNN_model.py` - Complete implementation with all original SelfGNN functionality plus hard negative sampling enhancements.

## 🚀 Quick Start (Google Colab Pro+)

### Option 1: Automated Script (Recommended)
```python
# Run the complete automated script
%run HardGNN_Colab_Script.py
```

### Option 2: Manual Setup
```bash
# 1. Install dependencies
pip install -r requirements_final.txt

# 2. Run verification
python VERIFY_COLAB_READY.py

# 3. Execute experiment
python main.py
```

## 📋 System Requirements

### Google Colab Pro+ Environment
- **Python**: 3.10+ (automatically available)
- **TensorFlow**: 2.10-2.16 (handles via tf.compat.v1)
- **RAM**: 25+ GB (Pro+ provides ~51GB)
- **GPU**: Optional but recommended (T4/A100)

### Dependencies
All required packages are listed in `requirements_final.txt`:
- tensorflow==2.15.0
- numpy==1.24.3
- scipy==1.10.1
- scikit-learn==1.3.0

## 🔧 Technical Implementation

### Hard Negative Sampling
The model implements sophisticated hard negative sampling with:

```python
# Key parameters (in Params.py)
use_hard_neg = True           # Enable hard negative sampling
temp = 0.1                    # InfoNCE temperature τ
hard_neg_top_k = 5           # Number of hard negatives K
contrastive_weight = 0.1     # Contrastive loss weight λ
```

### TensorFlow 2.x Compatibility
- Uses `tf.compat.v1` compatibility layer
- Replaced `tf.contrib.*` with TF2-compatible alternatives
- Memory-optimized for Google Colab constraints

### Memory Optimization
- Limits hard negatives to 50 per anchor (prevents OOM)
- GPU memory growth configuration
- Efficient batch processing

## 📊 Supported Datasets

The model supports all original SelfGNN datasets:
- **Amazon-book**: 52,463 users, 91,599 items
- **Yelp**: 31,668 users, 38,048 items  
- **Gowalla**: 29,858 users, 40,981 items
- **MovieLens**: 6,040 users, 3,706 items

## 🎛️ Configuration

### Dataset Configuration
```python
# In HardGNN_Colab_Script.py
configure_dataset('gowalla')  # or 'amazon-book', 'yelp', 'ml-1m'
```

### Model Hyperparameters
```python
# Core parameters (Params.py)
user = 29858          # Number of users (dataset-dependent)
item = 40981          # Number of items (dataset-dependent)
graphNum = 5          # Number of graph views
gnn_layer = 2         # GNN layers
att_layer = 2         # Attention layers
latdim = 64           # Embedding dimension
```

### Hard Negative Sampling Parameters
```python
# Hard negative sampling configuration
use_hard_neg = True           # Enable hard negatives
temp = 0.1                    # InfoNCE temperature
hard_neg_top_k = 5           # Top-K hard negatives
contrastive_weight = 0.1     # Loss weight λ
```

## 🔍 Verification

Run the comprehensive verification script:
```python
python VERIFY_COLAB_READY.py
```

This checks:
- ✅ Python 3.10+ compatibility
- ✅ TensorFlow 2.x installation  
- ✅ All required dependencies
- ✅ GPU availability and memory
- ✅ Model import capabilities
- ✅ Hard negative sampling functionality
- ✅ Dataset loading capabilities
- ✅ Memory optimization features

## 📁 File Structure

```
Google-Colab/
├── HardGNN_model.py              # Main production model
├── main.py                       # Alternative entry point
├── HardGNN_Colab_Script.py       # Complete automated script
├── Params.py                     # Model configuration
├── DataHandler.py                # Data processing
├── requirements_final.txt        # Production dependencies
├── VERIFY_COLAB_READY.py         # Verification script
├── README.md                     # This guide
├── Utils/                        # TF2-compatible utilities
│   ├── NNLayers_tf2.py
│   ├── attention_tf2.py
│   └── TimeLogger.py
├── Datasets/                     # Dataset storage
└── fallback_files/               # Backup/alternative files
    ├── model.py                  # Original model (backup)
    ├── model_colab_compatible.py # Fallback model
    └── archived_files            # Historical files
```

## 🚨 Troubleshooting

### Common Issues

**1. TensorFlow Version Conflicts**
```bash
# Solution: Use tf.compat.v1 (handled automatically)
pip install tensorflow==2.15.0
```

**2. Memory Issues**
```python
# The model includes memory optimization
# Reduces hard negatives from unlimited to 50 per anchor
```

**3. Import Errors**
```python
# Ensure you're in the correct directory
import sys
sys.path.append('/content/HardGNN/Google-Colab')
```

**4. GPU Not Detected**
```python
# Check GPU availability
import tensorflow as tf
print("GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
```

### Performance Optimization

For optimal performance on Google Colab Pro+:
1. **Use GPU runtime** (Runtime → Change runtime type → GPU)
2. **Monitor memory usage** during training
3. **Adjust batch size** if encountering OOM errors
4. **Use mixed precision** for faster training (optional)

## 📈 Expected Performance

### Training Metrics
- **Hit Rate@10**: ~0.15-0.25 (dataset dependent)
- **NDCG@10**: ~0.08-0.15 (dataset dependent)
- **Contrastive Loss**: ~0.5-2.0 (with hard negatives)

### Hardware Performance
- **Google Colab Pro+ (T4)**: ~20-30 minutes per epoch
- **Google Colab Pro+ (A100)**: ~10-15 minutes per epoch
- **CPU Only**: ~2-3 hours per epoch (not recommended)

## 🧪 Experiment Reproduction

To reproduce the original paper results with hard negative enhancement:

```python
# 1. Run automated script
%run HardGNN_Colab_Script.py

# 2. Select dataset
configure_dataset('gowalla')  # or your preferred dataset

# 3. Verify hard negative sampling is enabled
print(f"Hard negatives enabled: {args.use_hard_neg}")
print(f"Temperature: {args.temp}")
print(f"Top-K: {args.hard_neg_top_k}")

# 4. Start training
main()
```

## 📚 References

1. **SelfGNN**: Self-Supervised Graph Neural Networks without explicit negative sampling
2. **InfoNCE**: Representation Learning with Contrastive Predictive Coding
3. **Hard Negative Sampling**: Learning with hard negative sampling for recommendation

## 🤝 Support

If you encounter issues:
1. Run `VERIFY_COLAB_READY.py` for diagnostics
2. Check system requirements match Google Colab Pro+ specs
3. Ensure all files are in the correct directory structure
4. Verify TensorFlow 2.x compatibility layer is working

---

**Ready to run HardGNN experiments on Google Colab Pro+!** 🚀 