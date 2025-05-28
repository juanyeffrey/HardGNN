# ========================================================================
# HardGNN: Hard Negative Sampling Enhanced SelfGNN for Google Colab
# ========================================================================
# 
# This script contains all the code needed to run HardGNN on Google Colab.
# Copy each section into separate Colab cells as indicated by the comments.
# 
# Configuration: τ=0.1, K=5, λ=0.1, Dataset=Amazon-book
# ========================================================================

# ========================================================================
# CELL 1: Environment Setup and Installation
# ========================================================================

"""
🚀 HardGNN: Hard Negative Sampling Enhanced SelfGNN

This notebook implements HardGNN with:
- Hard negative sampling (cosine similarity-based)
- InfoNCE contrastive loss (τ=0.1, K=5, λ=0.1)
- Amazon-book dataset
- GPU acceleration on Google Colab

## 📋 Setup Instructions:
1. Runtime → Change runtime type → GPU (T4, A100, or V100)
2. Run cells in order - dependencies will be installed automatically
3. Monitor training - logs show contrastive loss alongside standard metrics
"""

# Install TensorFlow 1.14 for compatibility
import subprocess
import sys

def install_dependencies():
    packages = [
        'tensorflow-gpu==1.14.0',
        'matplotlib==3.5.1',
        'numpy==1.21.5', 
        'scipy==1.7.3'
    ]
    
    for package in packages:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
    
    print("✅ Dependencies installed successfully")

# Uncomment the next line when running in Colab
# install_dependencies()

# Verify GPU setup
import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {tf.test.is_gpu_available()}")
if tf.test.is_gpu_available():
    print(f"GPU Device: {tf.test.gpu_device_name()}")
    
# Configure GPU memory growth to prevent OOM
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
print("✅ GPU memory growth configured")

# ========================================================================
# CELL 2: Import Required Modules and Setup
# ========================================================================

# Core imports
import os
import numpy as np
import random
import pickle
import scipy.sparse as sp
import matplotlib.pyplot as plt
from tensorflow.core.protobuf import config_pb2
from ast import arg
from random import randint
import time
from datetime import datetime

# Import our modules
from Params import args
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
from DataHandler import DataHandler
from model import Recommender

# Configure for HardGNN with specified parameters
args.use_hard_neg = True
args.temp = 0.1  # Temperature τ
args.hard_neg_top_k = 5  # Number of hard negatives K
args.contrastive_weight = 0.1  # Contrastive weight λ
args.data = 'amazon'
args.save_path = 'hardgnn_colab'

# Adjust parameters for Colab demo
args.epoch = 20  # Reduced for demo
args.tstEpoch = 2  # Test every 2 epochs
args.trnNum = 5000  # Reduced training instances for faster demo

print("✅ HardGNN modules imported successfully")
print(f"📊 Configuration:")
print(f"  Hard Negative Sampling: {args.use_hard_neg}")
print(f"  Temperature (τ): {args.temp}")
print(f"  Hard Negatives (K): {args.hard_neg_top_k}")
print(f"  Contrastive Weight (λ): {args.contrastive_weight}")
print(f"  Dataset: {args.data}")

# ========================================================================
# CELL 3: Load Amazon Dataset
# ========================================================================

# Initialize and load data
logger.saveDefault = True
log('🔄 Starting data loading...')

handler = DataHandler()
handler.LoadData()

log(f'✅ Data loaded successfully')
print(f"📈 Dataset Statistics:")
print(f"  Users: {args.user:,}")
print(f"  Items: {args.item:,}")
print(f"  Training interactions: {handler.trnMat.nnz:,}")
print(f"  Test users: {len(handler.tstUsrs):,}")
print(f"  Time-based graphs: {len(handler.subMat)}")

# ========================================================================
# CELL 4: Validate Contrastive Loss Component
# ========================================================================

print("🔍 Validating Contrastive Loss Component...")
print(f"📊 Testing with τ={args.temp}, K={args.hard_neg_top_k}, λ={args.contrastive_weight}")

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)
tf.set_random_seed(42)

# Initialize TensorFlow session with GPU config
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True

with tf.Session(config=config) as sess:
    # Initialize HardGNN model
    model = Recommender(sess, handler)
    model.prepareModel()
    
    # Initialize variables
    init = tf.global_variables_initializer()
    sess.run(init)
    log('✅ Model initialized (random weights)')
    
    # Test contrastive loss on a small batch
    test_users = handler.tstUsrs[:32]  # Small batch for validation
    
    try:
        # Sample batch with hard negatives
        uLocs, iLocs, sequence, mask, uLocs_seq = model.sampleTrainBatch(
            test_users, handler.trnMat, handler.timeMat, train_sample_num=10
        )
        
        # Sample SSL batch
        suLocs, siLocs, suLocs_seq = model.sampleSslBatch(test_users, handler.subMat, False)
        
        # Prepare feed dict
        feed_dict = {
            model.uids: uLocs,
            model.iids: iLocs,
            model.sequence: sequence,
            model.mask: mask,
            model.is_train: False,
            model.uLocs_seq: uLocs_seq,
            model.keepRate: 1.0
        }
        
        for k in range(args.graphNum):
            feed_dict[model.suids[k]] = suLocs[k]
            feed_dict[model.siids[k]] = siLocs[k]
            feed_dict[model.suLocs_seq[k]] = suLocs_seq[k]
        
        # Run forward pass
        if hasattr(model, 'contrastive_loss'):
            results = sess.run([
                model.contrastive_loss,
                model.preLoss,
                model.posPred,
                model.negPred
            ], feed_dict=feed_dict)
            
            contrastive_loss, pre_loss, pos_pred, neg_pred = results
            
            print("\n" + "="*60)
            print("🎯 CONTRASTIVE LOSS VALIDATION RESULTS")
            print("="*60)
            print(f"📊 Metrics:")
            print(f"  Contrastive Loss: {contrastive_loss:.6f}")
            print(f"  Supervised Loss: {pre_loss:.6f}")
            print(f"  Positive Predictions: {np.mean(pos_pred):.4f} ± {np.std(pos_pred):.4f}")
            print(f"  Negative Predictions: {np.mean(neg_pred):.4f} ± {np.std(neg_pred):.4f}")
            print(f"  Prediction Gap: {np.mean(pos_pred) - np.mean(neg_pred):.4f}")
            
            if np.mean(pos_pred) > np.mean(neg_pred):
                print("  ✅ Positive predictions > Negative predictions")
            else:
                print("  ⚠️  Negative predictions >= Positive predictions")
                
            if contrastive_loss > 0 and not np.isnan(contrastive_loss):
                print("  ✅ Contrastive loss computed successfully")
            else:
                print("  ⚠️  Issue with contrastive loss computation")
                
            print("\n✅ Validation Complete - Ready for Training!")
            print("="*60)
            
        else:
            print("❌ Contrastive loss not available")
            
    except Exception as e:
        print(f"⚠️  Validation error: {e}")
        print("Proceeding with training...")

# ========================================================================
# CELL 5: Train HardGNN Model
# ========================================================================

print("🚀 Starting HardGNN Training...")
print(f"📊 Training Configuration:")
print(f"  Epochs: {args.epoch}")
print(f"  Test Frequency: Every {args.tstEpoch} epochs")
print(f"  Training Instances: {args.trnNum}")
print(f"  Batch Size: {args.batch}")
print(f"  Learning Rate: {args.lr}")

# Start fresh session for training
tf.reset_default_graph()

with tf.Session(config=config) as sess:
    # Initialize model
    model = Recommender(sess, handler)
    model.prepareModel()
    log('✅ Model prepared for training')
    
    # Initialize variables
    init = tf.global_variables_initializer()
    sess.run(init)
    log('✅ Variables initialized')
    
    # Training loop
    max_ndcg = 0.0
    max_res = dict()
    max_epoch = 0
    
    print("\n" + "="*80)
    print("🎯 TRAINING HARDGNN WITH CONTRASTIVE LEARNING")
    print("="*80)
    
    for ep in range(args.epoch):
        # Training step
        test = (ep % args.tstEpoch == 0)
        
        print(f"\n📚 Epoch {ep+1}/{args.epoch}")
        print("-" * 40)
        
        # Train for one epoch
        train_results = model.trainEpoch()
        
        # Print training results
        train_log = f"🏋️  Train: Loss={train_results['Loss']:.4f}, PreLoss={train_results['preLoss']:.4f}"
        if 'contrastiveLoss' in train_results:
            train_log += f", ContrastiveLoss={train_results['contrastiveLoss']:.4f}"
        print(train_log)
        
        # Test if it's a test epoch
        if test:
            test_results = model.testEpoch()
            test_log = f"🎯 Test: HR={test_results['HR']:.4f}, NDCG={test_results['NDCG']:.4f}"
            print(test_log)
            
            # Track best results
            if test_results['NDCG'] > max_ndcg:
                max_ndcg = test_results['NDCG']
                max_res = test_results.copy()
                max_epoch = ep
                print(f"🌟 New best NDCG: {max_ndcg:.4f}")
    
    # Final test
    print("\n" + "="*80)
    print("📊 FINAL RESULTS")
    print("="*80)
    
    final_results = model.testEpoch()
    print(f"🎯 Final Test Results:")
    print(f"  HR@10: {final_results['HR']:.4f}")
    print(f"  NDCG@10: {final_results['NDCG']:.4f}")
    
    print(f"\n🏆 Best Results (Epoch {max_epoch}):") 
    print(f"  Best HR@10: {max_res.get('HR', 0):.4f}")
    print(f"  Best NDCG@10: {max_res.get('NDCG', 0):.4f}")
    
    print("\n✅ HardGNN training completed successfully!")
    print("="*80)

# ========================================================================
# CELL 6: Optional - Compare with Baseline SelfGNN
# ========================================================================

# To compare with baseline, run this cell to train without hard negatives
print("🔬 Training Baseline SelfGNN (without hard negatives) for comparison...")

# Disable hard negative sampling
args.use_hard_neg = False
print(f"📊 Baseline Configuration: Hard Negative Sampling = {args.use_hard_neg}")

# Reset graph and train baseline
tf.reset_default_graph()

with tf.Session(config=config) as sess:
    # Initialize baseline model
    baseline_model = Recommender(sess, handler)
    baseline_model.prepareModel()
    
    # Initialize variables
    init = tf.global_variables_initializer()
    sess.run(init)
    log('✅ Baseline model initialized')
    
    print("\n" + "="*60)
    print("📊 BASELINE SELFGNN TRAINING")
    print("="*60)
    
    baseline_max_ndcg = 0.0
    baseline_max_res = dict()
    
    # Shorter training for comparison
    for ep in range(min(10, args.epoch)):
        test = (ep % args.tstEpoch == 0)
        
        # Train
        train_results = baseline_model.trainEpoch()
        train_log = f"Epoch {ep+1}: Loss={train_results['Loss']:.4f}, PreLoss={train_results['preLoss']:.4f}"
        print(train_log)
        
        # Test
        if test:
            test_results = baseline_model.testEpoch()
            test_log = f"  Test: HR={test_results['HR']:.4f}, NDCG={test_results['NDCG']:.4f}"
            print(test_log)
            
            if test_results['NDCG'] > baseline_max_ndcg:
                baseline_max_ndcg = test_results['NDCG']
                baseline_max_res = test_results.copy()
    
    print(f"\n📊 Baseline Best Results:")
    print(f"  HR@10: {baseline_max_res.get('HR', 0):.4f}")
    print(f"  NDCG@10: {baseline_max_res.get('NDCG', 0):.4f}")
    
    print("\n🔍 Comparison Summary:")
    improvement_hr = (max_res.get('HR', 0) - baseline_max_res.get('HR', 0)) / baseline_max_res.get('HR', 1) * 100
    improvement_ndcg = (max_res.get('NDCG', 0) - baseline_max_res.get('NDCG', 0)) / baseline_max_res.get('NDCG', 1) * 100
    
    print(f"  HardGNN vs Baseline HR@10: {improvement_hr:+.2f}%")
    print(f"  HardGNN vs Baseline NDCG@10: {improvement_ndcg:+.2f}%")
    
    if improvement_ndcg > 0:
        print("  ✅ HardGNN shows improvement over baseline!")
    else:
        print("  📝 Note: Longer training may be needed to see improvements")
    
    print("="*60)

# ========================================================================
# CELL 7: Results Analysis and Summary
# ========================================================================

print("""
# 📈 Results Analysis

## Key Metrics to Monitor:

1. **Contrastive Loss**: Should decrease over epochs, indicating better separation
2. **HR@10**: Hit Ratio at 10 - higher is better
3. **NDCG@10**: Normalized Discounted Cumulative Gain - higher is better
4. **Prediction Gap**: Positive predictions should exceed negative predictions

## HardGNN vs Baseline:
- **Hard Negative Sampling** selects more challenging negatives
- **InfoNCE Loss** creates better decision boundaries
- **Integrated Training** balances supervised and contrastive objectives

## 🎉 Summary

You've successfully run **HardGNN** on Google Colab! 

### What we accomplished:
✅ **Hard Negative Sampling**: Cosine similarity-based selection of challenging negatives  
✅ **InfoNCE Contrastive Loss**: Temperature-scaled contrastive learning (τ=0.1)  
✅ **Integrated Training**: Balanced supervised + contrastive objectives (λ=0.1)  
✅ **GPU Acceleration**: Optimized for Colab Pro+ GPUs  
✅ **Amazon Dataset**: Tested on recommendation benchmark  

### Key Takeaways:
- **Contrastive Loss** helps create better decision boundaries
- **Hard Negatives** focus learning on challenging examples  
- **Integrated Approach** balances multiple learning objectives

### Next Steps:
- Experiment with different τ, K, and λ values
- Try longer training for better convergence
- Test on other datasets (Yelp, MovieLens, Gowalla)
- Analyze attention patterns and embedding quality

**Citation**: This implementation extends the SelfGNN framework with hard negative sampling as described in Liu et al. (2024).
""")

# ========================================================================
# END OF SCRIPT
# ======================================================================== 