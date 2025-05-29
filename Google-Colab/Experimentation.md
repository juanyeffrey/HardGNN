# HardGNN Experimentation Guide for Google Colab Pro+

This guide provides a comprehensive walkthrough for setting up your Google Colab Pro+ environment and running experiments with the HardGNN model. HardGNN enhances the SelfGNN recommendation model with a specific hard negative sampling strategy.

## 📚 Table of Contents
1.  [Prerequisites](#prerequisites)
2.  [Setup Steps for Google Colab](#setup-steps-for-google-colab)
    *   [Step 1: Upload Project to Google Drive](#step-1-upload-project-to-google-drive)
    *   [Step 2: Create a New Colab Notebook](#step-2-create-a-new-colab-notebook)
    *   [Step 3: Configure Colab Runtime](#step-3-configure-colab-runtime)
    *   [Step 4: Mount Google Drive](#step-4-mount-google-drive)
    *   [Step 5: Navigate to Project Directory](#step-5-navigate-to-project-directory)
    *   [Step 6: Verify Directory Contents (Recommended)](#step-6-verify-directory-contents-recommended)
3.  [Running Experiments](#running-experiments)
    *   [Option A: Using the Automated Colab Script (Recommended for Full Runs)](#option-a-using-the-automated-colab-script-recommended-for-full-runs)
    *   [Option B: Manual Execution / Interactive Mode via `main.py`](#option-b-manual-execution--interactive-mode-via-mainpy)
    *   [Option C: Interactive Cell-by-Cell Execution of the Automated Script (Advanced)](#option-c-interactive-cell-by-cell-execution-of-the-automated-script-advanced)
4.  [Understanding the Output](#understanding-the-output)
5.  [Customizing Hard Negative Sampling (HNS)](#customizing-hard-negative-sampling-hns)
6.  [Troubleshooting Common Issues](#troubleshooting-common-issues)
7.  [Further Exploration](#further-exploration)

## 1. Prerequisites
*   A **Google Colab Pro+ account** (recommended for sufficient RAM and GPU access).
*   The **HardGNN project files**. You should have the `Google-Colab` directory containing all necessary scripts and data.

## 2. Setup Steps for Google Colab

Follow these steps to prepare your Colab environment.

### Step 1: Upload Project to Google Drive
1.  Locate the `Google-Colab` folder from the HardGNN project on your local machine.
2.  Upload this entire `Google-Colab` folder to your Google Drive.
    *   A good practice is to place it within a structured path, for example: `My Drive/Colab Notebooks/HardGNN/Google-Colab/`.
    *   **Note:** If you upload the entire `HardGNN` project repository, ensure you know the path to the `Google-Colab` sub-directory.

### Step 2: Create a New Colab Notebook
1.  Go to [Google Colab](https://colab.research.google.com/).
2.  Click on `File` -> `New notebook`.

### Step 3: Configure Colab Runtime
1.  In your new Colab notebook, click on `Runtime` -> `Change runtime type`.
2.  Under "Hardware accelerator," select **GPU**.
    *   For Colab Pro+, you might have options like T4 or A100. A100 is generally faster for training.
3.  Ensure the "Python version" is Python 3. Colab typically defaults to a suitable version (3.10+).
4.  Click `Save`.

### Step 4: Mount Google Drive
Run the following code cell in your notebook to give Colab access to your Google Drive files:
```python
from google.colab import drive
drive.mount('/content/drive')
```
You will be prompted to authorize Colab to access your Google Drive. Follow the on-screen instructions.

### Step 5: Navigate to Project Directory
Change the current working directory in Colab to where you uploaded the `Google-Colab` folder. Run this code cell, **adjusting the `project_path` variable** to match your Google Drive structure:
```python
import os

# IMPORTANT: Adjust this path to where you uploaded the Google-Colab folder!
# Example path if you created 'Colab Notebooks/HardGNN/' in your Drive root
# and placed 'Google-Colab' inside 'HardGNN'.
project_path = '/content/drive/My Drive/Colab Notebooks/HardGNN/Google-Colab'

os.chdir(project_path)
print(f"Current working directory: {os.getcwd()}")

# Add the project path to sys.path to ensure modules can be imported
import sys
if project_path not in sys.path:
    sys.path.append(project_path)
```

### Step 6: Verify Directory Contents (Recommended)
Run this cell to list the files and folders in your current directory. This helps confirm you're in the right place.
```python
!ls -l
```
You should see key files like `HardGNN_Colab_Script.py`, `requirements_final.txt`, `Params.py`, `HardGNN_model.py`, `DataHandler.py`, and the `Utils/` and `Datasets/` directories.

## 3. Running Experiments

There are three main ways to run experiments in Google Colab:

### Option A: Using the Automated Colab Script (Recommended for Full Runs)
This is the simplest way to run a complete experiment, including dependency installation and model training. The script is pre-configured for the Gowalla dataset but can be easily modified.

1.  **To run the script with its default settings (Gowalla dataset):**
    Execute the following magic command in a Colab cell:
    ```python
    %run HardGNN_Colab_Script.py
    ```
2.  **To change the dataset:**
    *   Before running the script, you can open `HardGNN_Colab_Script.py` using the Colab file browser (View -> Table of contents -> Files, then navigate to `Google-Colab/HardGNN_Colab_Script.py`).
    *   Locate the line `DATASET_NAME = 'gowalla'` (or similar).
    *   Change `'gowalla'` to one of the other supported datasets: `'amazon-book'`, `'yelp'`, or `'ml-1m'`.
    *   Save the script (Ctrl+S or File -> Save).
    *   Then, run the `%run HardGNN_Colab_Script.py` command.

The script will:
*   Print system information.
*   Install dependencies from `requirements_final.txt`.
*   Configure parameters for the chosen dataset.
*   Verify TensorFlow and GPU settings.
*   Initialize and train the HardGNN model.
*   Output training progress and evaluation metrics.

### Option B: Manual Execution / Interactive Mode via `main.py`
This approach is suitable if you prefer to launch the main training logic directly using `main.py`, perhaps for scripting multiple runs or if you are more accustomed to command-line execution. You can still run these commands interactively in Colab cells.

**Step B1: Install Dependencies (if not done by Option A)**
If you haven't run the automated script (which installs dependencies), run this cell:
```python
!pip install -r requirements_final.txt
```

**Step B2: (Optional) Run Verification Script**
This script checks your environment and basic model functionalities.
```python
!python VERIFY_COLAB_READY.py
```

**Step B3: Configure Parameters**
Model and hard negative sampling parameters are primarily managed in `Params.py`. You can edit this file directly through the Colab file browser if you need to make persistent changes. The `main.py` script will use these parameters, and specific dataset settings are applied internally by `main.py` based on the `--dataset` argument.

Key Hard Negative Sampling parameters in `Params.py`:
*   `args.use_hard_neg = True`       # Set to `True` to enable HNS, `False` to disable.
*   `args.temp = 0.1`                # InfoNCE temperature (τ).
*   `args.hard_neg_top_k = 5`        # Number of hard negatives to sample (K).
*   `args.contrastive_weight = 0.1`  # Weight for the contrastive loss component (λ).

**Step B4: Run the Experiment using `main.py`**
You can execute the main training script with a specific dataset.
```python
# Example for Gowalla dataset
!python main.py --dataset=gowalla

# Example for Amazon-book dataset
# !python main.py --dataset=amazon-book

# Example for Yelp dataset
# !python main.py --dataset=yelp

# Example for MovieLens (ml-1m) dataset
# !python main.py --dataset=ml-1m
```
The script will then load the appropriate dataset configuration (updating settings from `Params.py` as needed) and proceed with training.

### Option C: Interactive Cell-by-Cell Execution of the Automated Script (Advanced)
For users who want a more granular, interactive experience with the `HardGNN_Colab_Script.py`, breaking it down into individual notebook cells can be very beneficial. This approach allows for step-by-step execution, inspection of intermediate variables, and on-the-fly parameter adjustments between logical blocks of the script.

This method is excellent for:
*   Deeply understanding the workflow of `HardGNN_Colab_Script.py`.
*   Interactive debugging.
*   Experimenting with parameter changes (e.g., learning rate, HNS settings) between initialization, data loading, and training phases without repeatedly editing `Params.py`.

**For detailed instructions on this approach, please refer to the `HardGNN_Jupyter_Notebook_Guide.md` file located in the `Google-Colab` directory.** That guide explains how to structure your Colab notebook by copying sections of `HardGNN_Colab_Script.py` into different cells and how to modify parameters effectively in this interactive setup.

## 4. Understanding the Output
During training, the model will output information for each epoch, including:
*   **Epoch number**
*   **Losses**: Total loss, BPR loss, and contrastive loss (if HNS is enabled).
*   **Evaluation Metrics**:
    *   **Hit Rate (HR@K)**: Proportion of test cases where the true next item is among the top K recommended items.
    *   **Normalized Discounted Cumulative Gain (NDCG@K)**: A measure of ranking quality.
    *   Commonly reported for K=5, 10, 20.
*   **Time taken** for the epoch.

Look for steady improvement in HR and NDCG, and convergence of the loss values. The `contrastive_loss` should be stable if HNS is active.

## 5. Customizing Hard Negative Sampling (HNS)
The core HNS parameters are defined in `Params.py` and can be modified there before running any script. Both `HardGNN_Colab_Script.py` and `main.py` rely on `Params.py` for the default HNS settings, though `HardGNN_Colab_Script.py` explicitly sets them to the project's target values (τ=0.1, K=5, λ=0.1) after initial `args` parsing.

The primary HNS settings in `Params.py` (within the `args` object, typically set up by `parse_args()`):
*   `use_hard_neg`: `True` or `False`. Enables or disables the entire hard negative sampling mechanism.
*   `temp`: (InfoNCE temperature τ, e.g., `0.1`). Controls the sharpness of the distribution in the InfoNCE loss. Smaller values make the distribution sharper, focusing more on harder negatives.
*   `hard_neg_top_k`: (Number of hard negatives K, e.g., `5`). The number of hardest negative samples (based on cosine similarity to the anchor) to include in the contrastive loss calculation.
*   `contrastive_weight`: (Contrastive loss weight λ, e.g., `0.1`). Balances the BPR loss (main recommendation task) and the InfoNCE contrastive loss.

**To change these for experiments run via Option A or B:**
1.  Open `Params.py` in the Colab editor.
2.  Locate the `parse_args()` function or where these arguments are defined/updated.
3.  Modify their default values.
4.  Save the file.
5.  Re-run your experiment script (`HardGNN_Colab_Script.py` or `main.py`).

**For interactive changes when following Option C**, refer to `HardGNN_Jupyter_Notebook_Guide.md` for how to modify the `args` object directly in Colab cells between script execution steps.

## 6. Troubleshooting Common Issues
*   **`FileNotFoundError` or `ModuleNotFoundError`**:
    *   Ensure you have correctly navigated to the project directory using `os.chdir(project_path)`.
    *   Verify the `project_path` is correct.
    *   Make sure `sys.path.append(project_path)` was executed if running cells individually.
*   **TensorFlow/Dependency Issues**:
    *   Run `!pip install -r requirements_final.txt` to ensure all packages are installed with compatible versions.
    *   If you see errors related to TensorFlow versions, the `tf.compat.v1` layer should handle most TF1.x code, but ensure `tensorflow` is between 2.10 and 2.16.
*   **GPU Not Detected / CUDA Errors**:
    *   Double-check that your Colab runtime is set to GPU (Runtime -> Change runtime type).
    *   You can verify GPU detection with:
        ```python
        import tensorflow as tf
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
        ```
*   **Out Of Memory (OOM) Errors**:
    *   Colab Pro+ provides significant RAM, but very large datasets or batch sizes can still cause issues.
    *   The model has memory optimizations (e.g., capping negatives in InfoNCE).
    *   If OOM occurs, try reducing `batch` size in `Params.py`.

For more detailed troubleshooting, refer to the `README.md` in the `Google-Colab` directory.

## 7. Further Exploration
*   **`HardGNN_Jupyter_Notebook_Guide.md`**: For a detailed, cell-by-cell interactive approach to running and experimenting with `HardGNN_Colab_Script.py`.
*   **`README.md`**: The main README file offers a higher-level overview of the project, file structure, and technical details.
*   **`Params.py`**: Explore this file to understand all configurable hyperparameters for the model and training process.
*   **`HardGNN_model.py`**: Delve into the model architecture itself to understand the SelfGNN components and the hard negative sampling implementation.

Happy experimenting with HardGNN! 