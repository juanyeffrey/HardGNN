# HardGNN for Google Colab

## Overview
This folder contains a **TensorFlow 1.x (via tf.compat.v1) compatible** implementation of HardGNN (Hard Negative Sampling for Sequential Recommendation) designed for **Google Colab**. The model enhances SelfGNN by integrating a sophisticated hard negative sampling strategy using InfoNCE contrastive loss, built upon validated hyperparameters from original SelfGNN experiments.

This setup is primarily delivered through `HardGNN_Colab_Script.py`, a Python script with clearly marked cells. You should copy and paste these cells into a new Google Colab notebook to run the experiments.

## 🎯 Key Features
- **Enhanced SelfGNN**: Adds hard negative sampling (τ=0.1, K=5, λ=0.1) to original validated SelfGNN configurations.
- **TensorFlow 1.x Compatibility**: Uses `tf.compat.v1` for running TF1.x style code within a TensorFlow 2.x Colab environment.
- **Google Colab Optimized**: Designed to run efficiently in Colab, with GPU acceleration support.
- **Dataset-Agnostic**: Supports multiple datasets (Yelp, Amazon, Gowalla, MovieLens) by changing a single `DATASET` variable.
- **Reproducibility**: Aims to replicate original SelfGNN performance with the added hard negative sampling mechanism.
- **Environment Robustness**: Includes setup to use Colab's default TensorFlow and NumPy versions, and handles parameter resets for sequential model runs (e.g., validation then training).

## 🚀 Quick Start (Google Colab)

1.  **Prepare your Google Drive**:
    *   Upload the entire `Google-Colab` folder (containing `HardGNN_Colab_Script.py`, `Params.py`, `DataHandler.py`, `HardGNN_model.py`, the `Utils/` directory, and the `Datasets/` directory) to your Google Drive. For example, to `MyDrive/Google-Colab/`.

2.  **Create a New Colab Notebook**:
    *   Open Google Colab and create a new notebook.
    *   **Set Runtime Type**: Go to `Runtime` -> `Change runtime type` and select `GPU` (e.g., T4, A100, V100) as the hardware accelerator.

3.  **Mount Drive & Set Path (First Cell in Colab Notebook)**:
    ```python
    from google.colab import drive
    drive.mount('/content/drive')

    import os
    # IMPORTANT: Update this path to where you uploaded the 'Google-Colab' folder!
    project_path = '/content/drive/MyDrive/Google-Colab' # Or your specific path
    os.chdir(project_path)
    print(f"Current working directory: {os.getcwd()}")
    # You can add !ls to verify files are visible
    ```

4.  **Copy Cells from `HardGNN_Colab_Script.py`**:
    *   Open `Google-Colab/HardGNN_Colab_Script.py` (you can open it in a text editor or view it in your Google Drive).
    *   The script is divided into sections like `# CELL 1: ...`, `# CELL 2: ...`, etc.
    *   Copy the entire content of each "CELL" block from the script into a new, separate cell in your Colab notebook.
    *   **Configure `DATASET`**: In the Colab cell corresponding to the script's configuration section (near the top, usually part of the first code cell or a dedicated small cell), set the `DATASET` variable:
        ```python
        DATASET = 'gowalla'  # Options: 'yelp', 'amazon', 'gowalla', 'movielens'
        ```

5.  **Run Cells Sequentially**: Execute the cells in your Colab notebook in order.

## 📋 System Requirements & Environment

*   **Google Colab Environment**:
    *   **Python**: Colab's default (usually 3.10+).
    *   **TensorFlow**: Colab's default (e.g., 2.15+). The script uses `tf.compat.v1` to run TF1-style code.
    *   **NumPy**: Colab's default.
    *   **GPU**: Highly recommended. For A100 GPUs, the script enables XLA JIT compilation for enhanced performance. See notes on further GPU optimization below.
*   **Dependencies**: The script's "CELL 1" (Environment Setup) attempts to install necessary non-core ML libraries like `matplotlib`, `scipy`, `pandas`, `scikit-learn` if they are not adequately provided by Colab's default environment or need specific versions. It prioritizes using Colab's built-in TensorFlow and NumPy.

## 🔧 Technical Implementation Details

### Hard Negative Sampling
The model implements hard negative sampling based on the following parameters (configured in `Params.py` via `configure_dataset` in the script):
- `args.use_hard_neg = True`: Enables hard negative sampling.
- `args.temp = 0.1`: InfoNCE temperature (τ).
- `args.hard_neg_top_k = 5`: Number of hard negatives (K) to select.
- `args.contrastive_weight = 0.1`: Weight (λ) for the contrastive loss component in the total loss.

### TensorFlow 1.x in 2.x Environment
- Utilizes `tf.compat.v1.disable_eager_execution()` and `tf.compat.v1.disable_v2_behavior()` to create a TF1-compatible execution graph.
- All TensorFlow operations are expected to be in TF1 style (e.g., `tf.placeholder`, `tf.session`).

### GPU Performance Optimization (especially for A100)
- **XLA JIT Compilation**: The script automatically enables XLA (Accelerated Linear Algebra) JIT compilation in TensorFlow sessions if a GPU is detected. This can provide significant speedups on compatible GPUs like the A100 by compiling parts of the graph into more efficient machine code.
- **Batch Size**: With A100 GPUs offering substantial memory, consider experimenting with larger batch sizes (e.g., 1024, 2048) if your dataset and model fit. This can improve GPU utilization and reduce training time per epoch. You can adjust `args.batch` in the dataset configuration part of the script (Cell 2).
- **Mixed Precision (Advanced)**: A100 GPUs have Tensor Cores that excel with `float16` (half-precision) computations. While the current script uses `float32`, advanced users could explore TensorFlow's mixed precision training capabilities for further speedups. This would typically involve modifying the model definition and optimizer (e.g., using `tf.keras.mixed_precision.Policy('mixed_float16')` and `tf.keras.optimizers.Adam(..., loss_scale_optimizer=True)` in a more Keras-idiomatic TF2 setup, or manually casting parts of the graph in TF1 style, which is more complex).

### Parameter Management for Sequential Runs
- A potential issue with global parameter dictionaries in `Utils/NNLayers_tf2.py` being reused across different model instantiations (e.g., between a validation run and a training run in separate cells) has been addressed.
- `tf.compat.v1.reset_default_graph()` is used, and `NNLayers_tf2.reset_nn_params()` is explicitly called before model initialization in training cells to ensure a clean state.

## 📊 Supported Datasets & Configuration
The script supports the following datasets. The `configure_dataset` function in `HardGNN_Colab_Script.py` automatically sets validated hyperparameters for the chosen dataset and then applies the hard negative sampling configuration.

- **Yelp**
- **Amazon** (typically Amazon-Book)
- **Gowalla**
- **MovieLens** (typically ML-1M)

To change the dataset, modify the `DATASET` variable at the top of the script content in your Colab notebook:
```python
# In the configuration section of your Colab notebook (copied from HardGNN_Colab_Script.py)
DATASET = 'amazon'  # Options: 'yelp', 'amazon', 'gowalla', 'movielens'
```
The script will then print the specific configuration being used for that dataset.

## 🔍 Verification and Workflow

1.  **Environment Setup (Cell 1 of script)**: Installs dependencies and configures TensorFlow for TF1 compatibility using Colab's versions. Verifies TF and NumPy versions.
2.  **Dataset Configuration & Imports (Cell 2 of script)**: Imports necessary modules, configures dataset-specific hyperparameters, and applies hard negative sampling parameters.
3.  **Load Dataset (Cell 3 of script)**: Loads the specified dataset using `DataHandler.py`.
4.  **Validate Contrastive Loss (Cell 4 of script)**: Runs a quick check on a small batch to ensure the contrastive loss mechanism is active and behaving as expected with random weights.
5.  **Train HardGNN Model (Cell 5 of script)**:
    *   Resets the TensorFlow default graph and `NNLayers_tf2` parameters.
    *   Initializes and trains the `Recommender` model.
    *   Outputs training loss, pre-training loss, contrastive loss (if enabled), and periodic test metrics (HR@10, NDCG@10).
6.  **Baseline Comparison (Cell 6 of script, Optional)**: Trains the model without hard negative sampling for a brief period for comparison.
7.  **Results Analysis (Cell 7 of script)**: Prints a summary of the experiment and key configuration parameters.

## 📁 Key Files in the `Google-Colab` Folder

*   `HardGNN_Colab_Script.py`: The main Python script to be copied into a Colab notebook. Contains all experiment logic.
*   `Params.py`: Defines global parameters/arguments for the model.
*   `DataHandler.py`: Handles data loading and preprocessing.
*   `HardGNN_model.py`: Contains the `Recommender` class (SelfGNN architecture + Hard Negative Sampling logic).
*   `Utils/`: Directory containing utility modules:
    *   `NNLayers_tf2.py`: Custom neural network layers (TF1-style).
    *   `TimeLogger.py`: Logging utility.
    *   `attention_tf2.py`: Attention mechanisms (TF1-style).
*   `Datasets/`: Directory where preprocessed dataset files (e.g., `train.txt`, `test.txt` for Gowalla, Amazon, etc.) should be placed. Structure within `Datasets` should be like `Datasets/Gowalla/` or `Datasets/Amazon/`.

## 🚨 Troubleshooting Common Colab Issues

*   **GPU Not Detected/Used**:
    *   Ensure "Runtime" -> "Change runtime type" is set to "GPU".
    *   If it is, try "Runtime" -> "Disconnect and delete runtime", then reconnect and run cells from the beginning.
    *   TensorFlow should automatically use the GPU if available and detected. Cell 1 of the script includes checks for GPU availability.
*   **File Not Found Errors**:
    *   Double-check the `project_path` in your first Colab cell correctly points to where you uploaded the `Google-Colab` folder on your Google Drive.
    *   Ensure the `Datasets/` sub-directory is correctly populated with the required dataset files for the selected `DATASET`.
*   **Module Import Errors**:
    *   Usually related to the `project_path` not being set correctly, so Python can't find the local `.py` files (`Params`, `DataHandler`, etc.).
*   **Slow Performance**:
    *   Confirm GPU is active. CPU execution will be very slow.
    *   For very large datasets or complex models, even with a GPU, epochs can take time. Monitor the output per epoch.
    *   Ensure XLA is enabled (the script now does this automatically if a GPU is present) for optimal performance on A100s.

## 📈 Expected Performance Notes
- Performance metrics (HR@10, NDCG@10) will vary by dataset and depend on full training.
- The contrastive loss should ideally decrease or stabilize, indicating the model is learning to discriminate between positive and hard negative samples.
- Training times depend heavily on the dataset size and the Colab GPU assigned. A100 GPUs, especially with XLA enabled and potentially larger batch sizes, should offer the best performance.

## 📚 Citation Note
This implementation builds upon the SelfGNN framework and incorporates a hard negative sampling strategy. If using concepts from relevant papers, please cite them appropriately.

---

**This README provides guidance for running the HardGNN experiments using the provided Python script in a Google Colab environment.** 