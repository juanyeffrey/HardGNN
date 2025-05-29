# HardGNN on Google Colab: Jupyter Notebook Guide

This document guides you through running the HardGNN experiments within a Google Colab Jupyter Notebook, using the `HardGNN_Colab_Script.py` as a template for interactive, cell-by-cell execution. For a general overview of experiment setup and other execution methods, please refer to `Experimentation.md`.

## 🚀 Overview

The `HardGNN_Colab_Script.py` is designed to be easily adapted into a Colab notebook. It's structured with comments indicating distinct cells. This allows for interactive execution, easy parameter changes, and step-by-step monitoring.

**Key Idea**: We will copy sections of `HardGNN_Colab_Script.py` into separate Colab notebook cells.

## 📋 Prerequisites & Colab Setup

Many initial setup steps (uploading to Drive, creating a notebook, configuring GPU runtime) are also covered in the main `Experimentation.md` guide. This document reiterates them for completeness in the context of an interactive notebook workflow and then focuses on the cell-by-cell breakdown.

To run this project in Google Colab, follow these setup steps:

1.  **Google Colab Account**:
    *   Ensure you have a Google account and can access [Google Colab](https://colab.research.google.com/).

2.  **Upload Project Files to Google Drive**:
    *   Locate the `Google-Colab` folder on your local machine. This folder contains all necessary scripts (`HardGNN_Colab_Script.py`, `HardGNN_model.py`, `Params.py`, etc.), utility folders (`Utils/`), and potentially your `Datasets/` folder.
    *   Open your [Google Drive](https://drive.google.com/).
    *   Create a new folder in your Google Drive if you wish, for example, `HardGNN_Project`.
    *   Drag and drop the entire `Google-Colab` folder from your computer into this Google Drive folder.
    *   **Important**: If your datasets are large, ensure they are also within this `Google-Colab` folder (e.g., `Google-Colab/Datasets/gowalla/...`).

3.  **Create a New Colab Notebook**:
    *   Go to [Google Colab](https://colab.research.google.com/).
    *   Click on `File` -> `New notebook`.

4.  **Configure GPU Runtime**: 
    *   Your HardGNN code is optimized for NVIDIA GPUs. Colab Pro/Pro+ offers access to powerful GPUs like T4, A100, or L4.
    *   In your new Colab notebook, go to `Runtime` (in the top menu) -> `Change runtime type`.
    *   Under `Hardware accelerator`, select `GPU` from the dropdown menu.
    *   The available GPU types (e.g., T4, L4, A100) depend on your Colab subscription (Free, Pro, Pro+) and current availability. Select the best one available to you (A100 > L4 > T4 for training performance).
    *   Click `Save`.

5.  **Mount Google Drive & Navigate to Project Directory**:
    *   In the first cell of your Colab notebook, you need to connect to your Google Drive to access the uploaded project files.
    *   Add the following code to the first cell and run it:

    ```python
    from google.colab import drive
    drive.mount('/content/drive')

    # IMPORTANT: Update this path to where you uploaded the 'Google-Colab' folder!
    # Example: If you uploaded it to MyDrive -> HardGNN_Project -> Google-Colab
    # the path would be '/content/drive/MyDrive/HardGNN_Project/Google-Colab'
    import os
    project_path = '/content/drive/MyDrive/YOUR_FOLDER_PATH_HERE/Google-Colab' 
    os.chdir(project_path)

    # Verify the current working directory and list files to confirm
    print(f"Current working directory: {os.getcwd()}")
    print("Files in current directory:")
    for item in os.listdir('.'):
        print(item)
    ```
    *   When you run `drive.mount`, you'll be prompted to authorize Colab to access your Google Drive files. Follow the on-screen instructions.
    *   **Crucially, update `project_path`** to the exact path where your `Google-Colab` folder is located within your Google Drive.
    *   After running this cell, your Colab notebook's current working directory will be set to your project folder, allowing scripts to find modules and data files correctly.

## 📝 Setting Up Your Colab Notebook Cells

Now that your environment is set up, you will create subsequent cells in your notebook by copying sections from the `HardGNN_Colab_Script.py` file. The script has comments like `# ===== CELL X: ... =====` which indicate good breaking points.

--- 

### **Cell 1 (after Drive Mount): Introduction and Dataset Configuration**

**(Note: The Drive mount and `os.chdir` code shown above should be your *actual* first cell or part of this first main script cell.)**

**Purpose**: Initial script comments, select the dataset, install dependencies, and configure TensorFlow.

**Code**: Copy the content from `HardGNN_Colab_Script.py` from the line `# ========================================================================` down to (and including) the initial dataset configuration and environment checks.

```python
# CELL 1: Environment Setup and Installation (Copy from HardGNN_Colab_Script.py)
# ========================================================================
# HardGNN: Hard Negative Sampling Enhanced SelfGNN for Google Colab
# ========================================================================
# ... (copy all initial comments and docstrings) ...

# ========================================================================
# 🔧 CONFIGURE YOUR EXPERIMENT HERE
# ========================================================================
DATASET = 'gowalla'  # Options: 'yelp', 'amazon', 'gowalla', 'movielens'
# ========================================================================

# ... (copy install_dependencies(), setup_tensorflow_compatibility(), verify_colab_environment() functions) ...

# Run setup
# ... (copy the execution block for these functions) ...

print("✅ Environment setup complete!")
print("=" * 60)
```

**Running this cell will**: Install all Python packages, check your Colab environment, and configure TensorFlow for compatibility and GPU usage.

**Parameter to Change Here**: 
*   `DATASET = 'gowalla'`
    *   **What it does**: This string variable determines which dataset will be used for the experiment. The `configure_dataset` function (in the next cell) uses this value to set dataset-specific hyperparameters (like user/item counts, learning rate, GNN layers, etc.) and also the correct paths for loading data.
    *   **Options**: `'yelp'`, `'amazon'` (for Amazon-book), `'gowalla'`, `'movielens'` (for ML-1M).
    *   **Impact**: Changing this will load a different dataset and apply a different set of pre-defined optimal hyperparameters for the SelfGNN base model. The hard negative sampling parameters (τ, K, λ) remain consistent unless you modify them later.

--- 

### **Cell 2: Module Imports and Dataset Parameter Configuration**

**Purpose**: Import necessary Python modules, including your HardGNN model and helper functions. It also calls `configure_dataset` based on your `DATASET` choice from Cell 1.

**Code**: Copy the section from `HardGNN_Colab_Script.py` starting from `# Core imports` down to the print statements that confirm the configuration.

```python
# CELL 2: Dataset Configuration and Module Import (Copy from HardGNN_Colab_Script.py)

# Core imports
# ... (copy all import statements: os, numpy, Params, HardGNN_model, etc.) ...

# ... (copy the configure_dataset() function definition) ...

# Configure the dataset based on DATASET variable from Cell 1
# This will print the effective configuration being used.
args = configure_dataset(DATASET) # Ensure args is globally accessible if not already

print("✅ HardGNN modules imported and configured successfully")
# ... (copy all print statements showing the configuration) ...
```

**Running this cell will**: Make all model components and utilities available and apply the dataset-specific configurations. You'll see a printout of the active parameters.

--- 

### **Cell 3: Load Dataset**

**Purpose**: Initialize the `DataHandler` and load the actual dataset files into memory.

**Code**: Copy the section from `HardGNN_Colab_Script.py` responsible for data loading.

```python
# CELL 3: Load Dataset (Copy from HardGNN_Colab_Script.py)

# Initialize and load data
logger.saveDefault = True # From Params import args might be needed if not run in one flow
log(f'🔄 Starting {DATASET} data loading...') # DATASET var from Cell 1

handler = DataHandler() # from DataHandler import DataHandler
handler.LoadData()

log(f'✅ {DATASET} data loaded successfully')
# ... (copy print statements for dataset statistics) ...
```

**Running this cell will**: Prepare your chosen dataset for training and testing.

--- 

### **Cell 4: Validate Contrastive Loss Component (Optional but Recommended)**

**Purpose**: Performs a quick check to ensure the hard negative sampling and contrastive loss are functioning as expected before starting a full training run.

**Code**: Copy the section from `HardGNN_Colab_Script.py` for validating the contrastive loss.

```python
# CELL 4: Validate Contrastive Loss Component (Copy from HardGNN_Colab_Script.py)

print(f"🔍 Validating Hard Negative Sampling on {DATASET}...")
# ... (copy the entire validation block, including tf.Session, model initialization, and feed_dict preparation) ...
```

**Running this cell will**: Run a forward pass on a small batch and print out the calculated contrastive loss and related metrics. This helps catch issues early.

--- 

### **Cell 5: Train HardGNN Model**

**Purpose**: This is the main training loop for your HardGNN model.

**Code**: Copy the main training loop section from `HardGNN_Colab_Script.py`.

```python
# CELL 5: Train HardGNN Model (Copy from HardGNN_Colab_Script.py)

print(f"🚀 Starting HardGNN Training on {DATASET.upper()}...")
# ... (copy the entire training block, including tf.reset_default_graph(), tf.Session, model initialization, and the training loop) ...

print(f"\n✅ HardGNN training on {DATASET.upper()} completed successfully!")
print("="*80)
```

**Running this cell will**: Start the full training process. This will take time depending on the dataset and number of epochs.

**Parameters to Change (before running this cell, by modifying `Params.py` or `args` object directly in a preceding cell if needed):**

*   **In `Params.py` (or by setting `args.parameter_name = value` in a cell after Cell 2 but before Cell 5):**
    *   `args.lr = 1e-3` (Learning Rate)
        *   **What it does**: Controls the step size during gradient descent optimization. Smaller values lead to slower but potentially more stable convergence. Larger values can speed up training but risk overshooting the optimal solution.
        *   **Impact**: Directly affects how quickly and effectively the model learns.
    *   `args.reg = 1e-2` (L2 Regularization Weight for model embeddings)
        *   **What it does**: Adds a penalty to the loss function based on the squared magnitude of the model's embedding parameters. This helps prevent overfitting by discouraging overly complex models with large parameter values.
        *   **Impact**: Higher values increase regularization strength, which can improve generalization but might hurt training performance if too high.
    *   `args.batch = 512` (Batch Size)
        *   **What it does**: The number of training examples processed before the model's parameters are updated.
        *   **Impact**: Larger batch sizes provide a more stable estimate of the gradient but require more memory. Smaller batch sizes are noisier but can sometimes help escape local minima and might be necessary if memory is limited.
    *   `args.latdim = 64` (Embedding Dimension/Latency Dimension)
        *   **What it does**: The size of the vectors used to represent users and items.
        *   **Impact**: Larger dimensions can capture more complex patterns but increase model size and computational cost, and risk overfitting. Smaller dimensions are more compact but might underfit.
    *   `args.gnn_layer = 2` (Number of GNN Layers)
        *   **What it does**: Controls the depth of the graph neural network, determining how many hops of neighborhood information are aggregated for each node.
        *   **Impact**: More layers allow the model to capture information from more distant neighbors but can lead to issues like over-smoothing (making node embeddings too similar) and higher computational cost.
    *   `args.att_layer = 1` (Number of Attention Layers in sequence modeling part)
        *   **What it does**: Controls the depth of the multi-head self-attention applied to item sequences.
        *   **Impact**: Similar to GNN layers, more attention layers can capture more complex sequential dependencies but increase complexity.
    *   `args.use_hard_neg = True` (Enable/Disable Hard Negative Sampling)
        *   **What it does**: Boolean flag. If `True`, the hard negative sampling mechanism and InfoNCE loss are used. If `False`, the model reverts to standard negative sampling (as defined in `DataHandler.py` or the `preLoss` component) and the contrastive loss term is effectively zero.
        *   **Impact**: This is the primary switch for your proposed enhancement.
    *   `