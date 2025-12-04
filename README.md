# CAVAD: Correlation-Aware Variational Autoencoder for Zero-Day Network Anomaly Detection

This repository contains the reference implementation of **CAVAD** (Correlation-Aware Variational Autoencoder with Cross-Attention) for **zero-day network anomaly detection** from raw packet bytes.

CAVAD is an unsupervised framework: it is trained only on benign traffic and detects attacks as **out-of-distribution (OOD)** samples in a learned latent space.

---

## Table of Contents

- [Key Ideas](#key-ideas)
- [Project Structure](#project-structure)
- [File Descriptions](#file-descriptions)
- [Installation](#installation)
- [Usage](#usage)
- [Data Preprocessing](#data-preprocessing)
- [Datasets](#datasets)
- [Citation](#citation)

---

## Key Ideas

- **Raw packet modeling**
  Works directly on packet bytes instead of hand-crafted flow features, preserving fine-grained header and payload information.

- **Correlation-aware VAE**
  Uses a **full-covariance Gaussian** posterior in the latent space to capture dependencies between latent dimensions, instead of assuming independence.

- **Header–payload cross-attention**
  A **bidirectional cross-attention module** enforces semantic consistency between protocol headers and payloads, helping to detect masquerading traffic.

- **Mahalanobis-based anomaly score**
  Combines reconstruction error with a **Mahalanobis distance** in the latent space to better separate benign and malicious traffic.

---

## Project Structure

```
CAVAD/
├── Data_Preprocess/           # Data preprocessing scripts
│   ├── IDS_2017_TO_Mong.py   # CIC-IDS2017 data preprocessing
│   ├── IDS_2018_TO_Mong.py   # CSE-CIC-IDS2018 data preprocessing
│   └── TON_IOT/               # TON_IoT dataset preprocessing
│       ├── IOT_to_Mong.py
│       └── benign_pcap_processor.py
├── model/                     # CAVAD model components
│   ├── vae.py                # Main VAE architecture
│   ├── packet.py             # Packet encoder/decoder
│   ├── cnn.py                # CNN backbone
│   └── losses.py             # Loss functions
├── training/                  # Training utilities
│   ├── trainer.py            # VAE trainer class
│   └── callbacks.py          # Training callbacks
├── evaluation/                # Evaluation utilities
│   ├── metrics.py            # Evaluation metrics
│   ├── anomaly_detection.py  # Anomaly scoring methods
│   └── visualization.py      # Result visualization
├── utils/                     # Helper utilities
│   ├── data_utils.py         # Data processing utilities
│   ├── dataloader.py         # PyTorch dataloaders
│   └── general_utils.py      # General utilities
├── main.py                    # Main training script
├── test.py                    # Model evaluation script
├── vis_latent.py             # Latent space visualization
├── data_preprocess.py        # Main data preprocessing script
└── config.py                 # Configuration for all datasets
```

---

## File Descriptions

### Core Training & Evaluation

- **`main.py`**
  Main training script for CAVAD model. Supports multiple datasets (CIC-IDS2017, CSE-CIC-IDS2018, TON_IoT) and training modes (diagonal, correlated, GMM, autoencoder). Includes KL annealing, early stopping, and learning rate scheduling.

- **`test.py`**
  Comprehensive evaluation script for trained models. Computes multiple anomaly scores (reconstruction error, Mahalanobis distance, combined score, KL divergence) and generates detailed reports with per-category analysis and visualizations.

- **`config.py`**
  Centralized configuration file containing dataset paths, model architectures, and training hyperparameters for all three datasets. Uses dataclass-based configurations for clarity and maintainability.

### Data Preprocessing

- **`data_preprocess.py`**
  Main data preprocessing pipeline for TON_IoT dataset. Implements stratified sampling, benign/attack separation, and session construction from MongoDB collections. Generates NPZ files with packet headers, payloads, and labels.

- **`Data_Preprocess/IDS_2017_TO_Mong.py`**
  Preprocesses CIC-IDS2017 PCAP files into MongoDB format. Extracts packet headers and payloads, creates sessions, and stores metadata.

- **`Data_Preprocess/IDS_2018_TO_Mong.py`**
  Preprocesses CSE-CIC-IDS2018 PCAP files into MongoDB format with similar functionality to IDS_2017_TO_Mong.py.

- **`Data_Preprocess/TON_IOT/IOT_to_Mong.py`**
  Converts TON_IoT PCAP files to MongoDB collections with session-based organization.

- **`Data_Preprocess/TON_IOT/benign_pcap_processor.py`**
  Specialized processor for benign TON_IoT traffic with quality filtering and validation.

### Model Components

- **`model/vae.py`**
  Core CorrelatedGaussianVAE implementation supporting multiple training modes (diagonal, correlated, GMM, autoencoder). Implements full-covariance Gaussian posterior with Cholesky decomposition and analytical KL divergence.

- **`model/packet.py`**
  Implements PacketEncoder (TCN-based with cross-attention), SessionEncoder (temporal aggregation), and FactorizedDecoder (low-rank reconstruction) for processing packet sequences.

- **`model/cnn.py`**
  CNN backbone components including temporal convolutional networks (TCN) with SE blocks for feature extraction.

- **`model/losses.py`**
  Custom loss functions including reconstruction losses, KL divergence variants, and regularization terms.

### Training Infrastructure

- **`training/trainer.py`**
  VAETrainer class handling training loops, validation, anomaly score computation (reconstruction, Mahalanobis, whitened L2, combined, KL), and reference statistics calculation.

- **`training/callbacks.py`**
  Training callbacks including EarlyStopping, ModelCheckpoint, LearningRateMonitor, GradientMonitor, MetricTracker, ProgressBar, and WarmupScheduler.

### Evaluation & Visualization

- **`evaluation/metrics.py`**
  Evaluation metrics computation including optimal threshold selection, FPR-based thresholding, and performance metrics (accuracy, precision, recall, F1, AUC).

- **`evaluation/anomaly_detection.py`**
  Anomaly detection methods implementing different scoring approaches.

- **`evaluation/visualization.py`**
  Visualization utilities for results, distributions, and latent space analysis.

- **`vis_latent.py`**
  Standalone script for t-SNE visualization of VAE latent space. Generates publication-quality plots showing separation between benign and attack traffic.

### Utilities

- **`utils/dataloader.py`**
  PyTorch Dataset and DataLoader implementations for efficient loading of preprocessed NPZ files with optional data augmentation.

- **`utils/data_utils.py`**
  Data processing utilities including train/validation splitting, normalization, and session construction.

- **`utils/general_utils.py`**
  General utilities including random seed setting, experiment directory management, logging setup, and GPU memory monitoring.

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- MongoDB (for data preprocessing)

### Install Dependencies

Create a virtual environment (recommended) and install packages:

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Key dependencies:
- PyTorch >= 1.12.0
- NumPy
- scikit-learn
- pymongo (for data preprocessing)
- tqdm
- matplotlib
- seaborn
- pandas

---

## Usage

### 1. Data Preprocessing

First, preprocess your raw PCAP files into the required format. Update the dataset paths in `config.py` to match your local setup.

```bash
# For TON_IoT dataset
python data_preprocess.py

# For CIC-IDS2017 dataset
python Data_Preprocess/IDS_2017_TO_Mong.py

# For CSE-CIC-IDS2018 dataset
python Data_Preprocess/IDS_2018_TO_Mong.py
```

The preprocessing scripts will:
1. Read raw PCAP files or MongoDB collections
2. Extract packet headers and payloads
3. Construct sessions (sequences of packets)
4. Apply padding/truncation to fixed lengths
5. Generate NPZ files containing:
   - `headers`: Packet headers (shape: [N, num_packets, header_size])
   - `payloads`: Packet payloads (shape: [N, num_packets, payload_size])
   - `payload_masks`: Valid payload indicators
   - `labels`: Traffic labels (0 for benign, >0 for attacks)
   - `stats_features`: Statistical features

### 2. Training

Train the CAVAD model on benign traffic:

```bash
# Train on CIC-IDS2017 with correlated mode (recommended)
python main.py --dataset cicids2017 --training_mode correlated

# Train on CSE-CIC-IDS2018
python main.py --dataset cicids2018 --training_mode correlated

# Train on TON_IoT
python main.py --dataset ton_iot --training_mode correlated

# Other training modes
python main.py --dataset cicids2017 --training_mode diagonal    # Diagonal covariance
python main.py --dataset cicids2017 --training_mode gmm         # Gaussian Mixture Model
python main.py --dataset cicids2017 --training_mode ae          # Autoencoder (no KL)
```

**Key Arguments:**
- `--dataset`: Dataset name (cicids2017, cicids2018, ton_iot)
- `--training_mode`: VAE mode (diagonal, correlated, gmm, ae)
- `--latent_dim`: Latent space dimension (default: 64)
- `--batch_size`: Batch size (default: 128)
- `--num_epochs`: Number of epochs (default: 50-80 depending on dataset)
- `--learning_rate`: Learning rate (default: 1e-5)
- `--output_dir`: Output directory for checkpoints and logs
- `--experiment_name`: Custom experiment name

The training script will:
- Automatically load dataset configurations from `config.py`
- Create an experiment directory with timestamps
- Save checkpoints periodically and when validation loss improves
- Log training metrics (loss, reconstruction error, KL divergence)
- Apply KL annealing and early stopping

**Training Output:**
```
outputs/
└── vae_cicids2017_correlated_20231204_120000/
    ├── checkpoints/
    │   ├── best_epoch_45.pth
    │   └── epoch_50.pth
    ├── logs/
    │   ├── training.log
    │   ├── metrics.json
    │   └── lrs.txt
    └── config.json
```

### 3. Evaluation

Evaluate trained models on test data with multiple anomaly scoring methods:

```bash
# Basic evaluation
python test.py \
    --dataset cicids2017 \
    --model_path outputs/vae_cicids2017_correlated_20231204_120000/checkpoints/best_epoch_45.pth \
    --output_dir test_results/cicids2017

# Evaluate with custom FPR target
python test.py \
    --dataset cicids2017 \
    --model_path outputs/vae_cicids2017_correlated_20231204_120000/checkpoints/best_epoch_45.pth \
    --fpr_target 0.01 \
    --output_dir test_results/cicids2017_fpr001

# Evaluate specific anomaly scoring methods
python test.py \
    --dataset cicids2017 \
    --model_path path/to/checkpoint.pth \
    --methods reconstruction mahalanobis combined
```

**Available Anomaly Scoring Methods:**
- `reconstruction`: Reconstruction error (MSE between input and output)
- `mahalanobis`: Mahalanobis distance in latent space
- `whitened_l2`: L2 distance in whitened latent space
- `combined`: Weighted combination of reconstruction and Mahalanobis
- `kl`: KL divergence between posterior and prior

**Evaluation Output:**
```
test_results/cicids2017/
├── evaluation_report.txt              # Summary report
├── evaluation_results.json            # Detailed JSON results
├── method_comparison.csv              # Method comparison table
├── method_reconstruction_fpr_sweep.csv # FPR sweep for reconstruction
├── method_mahalanobis_fpr_sweep.csv   # FPR sweep for Mahalanobis
├── reconstruction/                     # Per-method results
│   ├── scores_distribution.png
│   ├── per_category_distribution.png
│   ├── per_category_accuracy.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── score_timeline.png
│   └── threshold_analysis.png
└── mahalanobis/
    └── ...
```




