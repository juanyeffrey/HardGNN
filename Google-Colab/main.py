#!/usr/bin/env python3
"""
Main entry point for HardGNN model execution
Compatible with Google Colab Pro+ environment
"""

import os
import sys
import numpy as np
import tensorflow as tf
from time import time

# Enable TensorFlow 1.x compatibility for TF 2.x
tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_v2_behavior()

# Configure GPU memory growth for Colab Pro+
try:
	gpus = tf.config.experimental.list_physical_devices('GPU')
	if gpus:
		for gpu in gpus:
			tf.config.experimental.set_memory_growth(gpu, True)
		print(f"✅ GPU configured: {len(gpus)} GPU(s) available")
	else:
		print("⚠️ No GPU found, using CPU")
except Exception as e:
	print(f"GPU configuration warning: {e}")

from Params import args
from HardGNN_model import Recommender
from DataHandler import DataHandler
import Utils.TimeLogger as logger

print("🚀 Starting HardGNN experiment")
print(f"Dataset: {args.data}")
print(f"Hard Negative Sampling: {'Enabled' if args.use_hard_neg else 'Disabled'}")
if args.use_hard_neg:
	print(f"Temperature τ: {args.temp}")
	print(f"Hard negatives K: {args.hard_neg_top_k}")
	print(f"Contrastive weight λ: {args.contrastive_weight}")

def main():
	logger.saveDefault = True
	
	# Set random seeds for reproducibility
	np.random.seed(args.seed)
	tf.compat.v1.set_random_seed(args.seed)
	
	# Initialize data handler
	handler = DataHandler()
	handler.LoadData()
	
	# Configure TensorFlow session
	config = tf.compat.v1.ConfigProto()
	config.gpu_options.allow_growth = True
	config.allow_soft_placement = True
	
	with tf.compat.v1.Session(config=config) as sess:
		# Initialize recommender
		recom = Recommender(sess, handler)
		
		# Run the experiment
		recom.run()
		
	print("✅ Experiment completed successfully!")

if __name__ == '__main__':
	main() 