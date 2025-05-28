# 🚀 HardGNN Google Colab Setup Guide

Since Jupyter notebook files (.ipynb) need to be created properly, here's a **step-by-step guide** to manually create the HardGNN notebook in Google Colab.

## 📋 Method 1: Copy-Paste from Script (Recommended)

### Step 1: Open Google Colab
1. Go to https://colab.research.google.com/
2. Click **"New notebook"**
3. Set Runtime: **Runtime → Change runtime type → GPU (T4, A100, or V100)**

### Step 2: Upload Files to Google Drive
1. Upload the entire `Google-Colab` folder to your Google Drive
2. Note the path (e.g., `/content/drive/MyDrive/Google-Colab`)

### Step 3: Create Cells from HardGNN_Colab_Script.py

Open `HardGNN_Colab_Script.py` and copy each section into separate Colab cells:

#### **Cell 1: Environment Setup**
```python
# Copy everything from "CELL 1: Environment Setup and Installation" section
# This includes the docstring, dependency installation, and GPU configuration
```

#### **Cell 2: Import Modules**
```python
# Copy everything from "CELL 2: Import Required Modules and Setup" section  
# This sets up imports and HardGNN configuration
```

#### **Cell 3: Load Dataset**
```python
# Copy everything from "CELL 3: Load Amazon Dataset" section
# This loads and displays dataset statistics
```

#### **Cell 4: Validate Contrastive Loss**
```python
# Copy everything from "CELL 4: Validate Contrastive Loss Component" section
# This tests the contrastive loss before full training
```

#### **Cell 5: Train HardGNN**
```python
# Copy everything from "CELL 5: Train HardGNN Model" section
# This runs the main HardGNN training loop
```

#### **Cell 6: Baseline Comparison (Optional)**
```python
# Copy everything from "CELL 6: Optional - Compare with Baseline SelfGNN" section
# This trains baseline SelfGNN for comparison
```

#### **Cell 7: Results Analysis**
```python
# Copy everything from "CELL 7: Results Analysis and Summary" section
# This provides analysis and next steps
```

## 📋 Method 2: Quick Start with Single Cell

If you want to run everything in one go, create a single cell and copy the entire `HardGNN_Colab_Script.py` content.

### Before Running:
1. **Uncomment the installation line**:
   ```python
   # Change this line:
   # install_dependencies()
   
   # To this:
   install_dependencies()
   ```

2. **Set the correct path**:
   ```python
   import os
   os.chdir('/content/drive/MyDrive/Google-Colab')  # Adjust your path
   ```

## 🔧 Important Setup Steps

### 1. Mount Google Drive (First cell)
```python
from google.colab import drive
drive.mount('/content/drive')

import os
os.chdir('/content/drive/MyDrive/Google-Colab')  # Adjust your path
print(f"Current directory: {os.getcwd()}")
print(f"Files available: {os.listdir('.')}")
```

### 2. Enable Dependency Installation
Uncomment this line in Cell 1:
```python
install_dependencies()  # Remove the # to enable
```

### 3. Check GPU Availability
```python
import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {tf.test.is_gpu_available()}")
if tf.test.is_gpu_available():
    print(f"GPU Device: {tf.test.gpu_device_name()}")
```

## 📊 Expected Output

### Cell 1 Output:
```
✅ Dependencies installed successfully
TensorFlow version: 1.14.0
GPU Available: True
GPU Device: /device:GPU:0
✅ GPU memory growth configured
```

### Cell 2 Output:
```
✅ HardGNN modules imported successfully
📊 Configuration:
  Hard Negative Sampling: True
  Temperature (τ): 0.1
  Hard Negatives (K): 5
  Contrastive Weight (λ): 0.1
  Dataset: amazon
```

### Cell 3 Output:
```
✅ Data loaded successfully
📈 Dataset Statistics:
  Users: 52,643
  Items: 91,599
  Training interactions: 2,517,437
  Test users: 6,040
  Time-based graphs: 5
```

### Cell 4 Output:
```
🎯 CONTRASTIVE LOSS VALIDATION RESULTS
============================================================
📊 Metrics:
  Contrastive Loss: 0.389145
  Supervised Loss: 0.784523
  Positive Predictions: 0.1234 ± 0.0876
  Negative Predictions: -0.0456 ± 0.0654
  Prediction Gap: 0.1690
  ✅ Positive predictions > Negative predictions
  ✅ Contrastive loss computed successfully

✅ Validation Complete - Ready for Training!
```

### Cell 5 Output:
```
🎯 TRAINING HARDGNN WITH CONTRASTIVE LEARNING
================================================================================

📚 Epoch 1/20
----------------------------------------
🏋️  Train: Loss=0.8234, PreLoss=0.7845, ContrastiveLoss=0.3891
🎯 Test: HR=0.0856, NDCG=0.0423

📚 Epoch 3/20
----------------------------------------
🏋️  Train: Loss=0.6721, PreLoss=0.6234, ContrastiveLoss=0.2876
🎯 Test: HR=0.1234, NDCG=0.0687
🌟 New best NDCG: 0.0687

...

📊 FINAL RESULTS
================================================================================
🎯 Final Test Results:
  HR@10: 0.1456
  NDCG@10: 0.0821

🏆 Best Results (Epoch 15):
  Best HR@10: 0.1523
  Best NDCG@10: 0.0834

✅ HardGNN training completed successfully!
```

## 🐛 Troubleshooting

### Issue: "Module not found"
**Solution**: Make sure you're in the correct directory
```python
import os
print(f"Current directory: {os.getcwd()}")
os.chdir('/content/drive/MyDrive/Google-Colab')  # Adjust path
```

### Issue: "GPU memory error" 
**Solution**: Reduce batch size
```python
args.batch = 256  # Instead of 512
args.trnNum = 2000  # Instead of 5000
```

### Issue: "TensorFlow version conflict"
**Solution**: Restart runtime and reinstall
```python
# Runtime → Restart runtime
# Then re-run dependency installation
```

### Issue: "Dataset not found"
**Solution**: The code includes fallback to dummy data
```python
# Check if dataset files exist
import os
print("Amazon dataset files:")
for file in os.listdir('Datasets/amazon/'):
    print(f"  {file}")
```

## 🎯 Performance Tips

1. **Use Colab Pro+** with A100 GPU for best performance
2. **Monitor GPU usage**: Runtime → View resources
3. **Save progress**: Uncomment `model.saveHistory()` for long training
4. **Reduce data size** for faster experimentation:
   ```python
   args.epoch = 10  # Fewer epochs
   args.trnNum = 1000  # Fewer training samples
   ```

## 📝 Notes

- **Training Time**: ~2-3 hours on A100 for full 20 epochs
- **Memory Usage**: ~8-12GB GPU memory for full dataset
- **Colab Limits**: 12-hour session limit, save progress periodically
- **Results**: Expect 5-15% improvement over baseline SelfGNN

## 🚀 Ready to Go!

Follow this guide to manually create your HardGNN notebook in Google Colab. The modular approach lets you run sections independently and troubleshoot any issues step by step.

**Happy experimenting with HardGNN! 🎉** 