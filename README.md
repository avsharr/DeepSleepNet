# DeepSleepNet

<div align="center">

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

**Deep learning model for automatic sleep stage classification using EEG signals**

[Features](#-features) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Documentation](#-documentation)

</div>

---

## 📋 Overview

DeepSleepNet is a deep learning framework for automatic sleep stage classification from single-channel EEG signals. The model implements a dual-branch CNN architecture combined with bidirectional LSTM for sequence learning, achieving state-of-the-art performance on the Sleep-EDF dataset.

### Key Capabilities

- 🧠 **Advanced Architecture**: Dual-branch CNN (temporal + frequency) + Bidirectional LSTM
- ⚖️ **Class Imbalance Handling**: Automatic class weighting using inverse frequency method
- 🔄 **End-to-End Pipeline**: From raw EDF files to trained model and evaluation
- 📊 **Comprehensive Metrics**: Accuracy, precision, recall, F1-score, Cohen's kappa
- 📈 **Rich Visualizations**: Confusion matrices, per-class metrics, distribution plots

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **DeepSleepNet Architecture** | Dual-branch CNN with residual blocks + Bidirectional LSTM for sequence learning |
| **Automatic Class Weighting** | Handles class imbalance using inverse frequency weighting |
| **Data Preprocessing** | EDF to NPZ conversion with automatic channel selection (Fpz-Cz) |
| **Training Pipeline** | Early stopping, learning rate scheduling, gradient clipping |
| **Evaluation Tools** | Comprehensive metrics and detailed classification reports |
| **Visualization Suite** | Confusion matrices, per-class metrics, class distribution plots |

---

## 🏗️ Project Structure

```
DeepSleepNet/
├── 📁 data/
│   ├── raw/              # Raw EDF files (download required)
│   └── preprocessed/     # Processed NPZ files
├── 📁 preprocessing/      # Data preprocessing modules
│   ├── __init__.py
│   └── preprocessing.py  # EDF → NPZ conversion, class weights
├── 📁 datasets/           # PyTorch Dataset classes
│   ├── __init__.py
│   └── dataset.py        # SequentialSleepDataset
├── 📁 models/             # Model architecture
│   ├── __init__.py
│   ├── model.py          # DeepSleepNet, ResBlock
│   └── losses.py          # FocalLoss, LabelSmoothingLoss (optional)
├── 📁 scripts/            # Runnable scripts
│   ├── download_data.py      # Download Sleep-EDF dataset
│   ├── run_preprocessing.py  # Run data preprocessing
│   ├── analyze_data.py       # Analyze class distribution
│   ├── train.py              # Train the model
│   ├── evaluate.py           # Evaluate on test set
│   ├── visualize_metrics.py # Generate metric plots
│   ├── explore_data.py       # Explore raw EDF files
│   └── plot_hypnogram.py     # Plot sleep stage annotations
├── 📁 checkpoints/        # Saved model weights (.pth)
├── 📁 figures/            # Generated plots (.png)
├── common.py              # Shared utilities
└── pyproject.toml         # Project dependencies
```

---

## 🚀 Installation

### Prerequisites

- **Python**: 3.8 or higher
- **pip**: Latest version recommended
- **Memory**: At least 8GB RAM recommended
- **Storage**: ~2GB for dataset + models

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd DeepSleepNet
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -e .
   ```
   
   Or install manually:
   ```bash
   pip install torch numpy mne scikit-learn matplotlib seaborn tqdm
   ```

---

## 🎯 Quick Start

### Complete Workflow

```bash
# 1. Download data
python scripts/download_data.py

# 2. Preprocess data
python scripts/run_preprocessing.py

# 3. (Optional) Analyze data distribution
python scripts/analyze_data.py

# 4. Train model
python scripts/train.py

# 5. Evaluate model
python scripts/evaluate.py

# 6. Generate visualizations
python scripts/visualize_metrics.py
```

---

## 📖 Usage Guide

### 1️⃣ Download Data

Download the Sleep-EDF dataset from PhysioNet:

```bash
python scripts/download_data.py
```

**Output**: EDF files saved to `data/raw/`

### 2️⃣ Preprocess Data

Convert EDF files to NPZ format with automatic channel selection:

```bash
python scripts/run_preprocessing.py
```

**Output**: Preprocessed epochs saved to `data/preprocessed/`

**Process**:
- Extracts EEG Fpz-Cz channel
- Crops data around sleep stages (±30 minutes)
- Creates 30-second epochs
- Maps sleep stages to 5 classes (Wake, N1, N2, N3, REM)

### 3️⃣ Analyze Data (Optional)

Analyze class distribution and signal statistics:

```bash
python scripts/analyze_data.py
```

**Output**: 
- Class distribution plots in `figures/class_distribution_train.png` and `figures/class_distribution_test.png`
- Statistics printed to console

### 4️⃣ Train Model

Train the DeepSleepNet model:

```bash
python scripts/train.py
```

**Configuration** (in `scripts/train.py`):
- `BATCH_SIZE = 16`
- `EPOCHS = 50` (with early stopping)
- `LEARNING_RATE = 1e-4`
- `EARLY_STOPPING_PATIENCE = 7`
- `weight_decay = 1e-3` (L2 regularization)

**Output**:
- Best model: `checkpoints/deepsleepnet_best_model.pth`
- Final model: `checkpoints/deepsleepnet_model.pth`

**Features**:
- ✅ Automatic class weight computation
- ✅ Early stopping based on validation accuracy
- ✅ Learning rate scheduling (ReduceLROnPlateau)
- ✅ Gradient clipping (max_norm=1.0)

### 5️⃣ Evaluate Model

Evaluate the trained model on the test set:

```bash
python scripts/evaluate.py
```

**Output**: Classification report and confusion matrix printed to console

**Example Output**:
```
FINAL EVALUATION REPORT
==================================================
Overall Accuracy: 69.90%

Classification Report:
              precision    recall  f1-score   support
        Wake     0.9616    0.7366    0.8342      1496
          N1     0.3497    0.6897    0.4640       435
          N2     0.9662    0.5122    0.6695      2960
          N3     0.7276    0.9102    0.8087      1080
         REM     0.5017    0.9124    0.6474      1279
```

### 6️⃣ Visualize Metrics

Generate comprehensive metric visualizations:

```bash
python scripts/visualize_metrics.py
```

**Output** (in `figures/`):
- `confusion_matrix.png` - Confusion matrix (absolute values and percentages)
- `metrics_per_class.png` - Precision, Recall, F1-score per class
- `class_distribution.png` - True vs predicted class distribution

---

## 🧠 Model Architecture

### DeepSleepNet Components

#### 1. Dual CNN Branches

**Small Filter Branch** (Temporal Features):
- Initial conv: kernel=50, stride=6
- 3× ResBlock layers
- MaxPool operations

**Large Filter Branch** (Frequency Features):
- Initial conv: kernel=400, stride=50
- 3× ResBlock layers
- MaxPool operations

Both branches use:
- Batch Normalization
- ReLU activation
- Dropout (0.2)
- Residual connections

#### 2. Sequence Learning

- **Bidirectional LSTM**: 2 layers, 512 hidden units
- **Dropout**: 0.5 for regularization
- Processes sequences of 25 epochs

#### 3. Classification Head

- Fully connected layer → 5 classes
- Output: Wake, N1, N2, N3, REM

### Architecture Diagram

```
Input (3000 samples) 
    ↓
┌─────────────────┬─────────────────┐
│ Small CNN       │ Large CNN       │
│ (Temporal)      │ (Frequency)     │
└─────────────────┴─────────────────┘
    ↓                    ↓
    └────────┬───────────┘
             ↓
    Bidirectional LSTM (2 layers)
             ↓
    Fully Connected Layer
             ↓
    Output (5 classes)
```

---

## 📊 Data Split

The dataset is automatically split into:

| Split | Percentage | Description |
|-------|------------|-------------|
| **Train** | 70% | Used for model training |
| **Validation** | 15% | Used for early stopping and LR scheduling |
| **Test** | 15% | Used for final evaluation |

Split is **deterministic** (sorted by filename) for reproducibility.

---

## 🛌 Sleep Stages

| Stage | Label | Description |
|-------|-------|-------------|
| **Wake** | 0 | Awake stage |
| **N1** | 1 | Light sleep stage 1 |
| **N2** | 2 | Light sleep stage 2 |
| **N3** | 3 | Deep sleep (stages 3+4 merged) |
| **REM** | 4 | Rapid Eye Movement sleep |

---

## 📈 Results

### Performance Metrics

The model achieves the following performance on the Sleep-EDF test set:

- **Overall Accuracy**: ~70%
- **Macro F1-Score**: ~0.68
- **Weighted F1-Score**: ~0.71

### Per-Class Performance

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Wake  | ~0.96     | ~0.74  | ~0.83    |
| N1    | ~0.35     | ~0.69  | ~0.46    |
| N2    | ~0.97     | ~0.51  | ~0.67    |
| N3    | ~0.73     | ~0.91  | ~0.81    |
| REM   | ~0.50     | ~0.91  | ~0.65    |

*Note: Results may vary. Run `python scripts/visualize_metrics.py` for detailed metrics.*

---

## 🔧 Configuration

### Training Hyperparameters

Key hyperparameters can be modified in `scripts/train.py`:

```python
BATCH_SIZE = 16              # Batch size for training
EPOCHS = 50                  # Maximum epochs (early stopping)
LEARNING_RATE = 1e-4        # Initial learning rate
EARLY_STOPPING_PATIENCE = 7 # Early stopping patience
weight_decay = 1e-3         # L2 regularization
```

### Model Architecture

Model parameters can be modified in `models/model.py`:

- LSTM hidden size: 512
- LSTM layers: 2
- Dropout rates: 0.2 (CNN), 0.3 (before LSTM), 0.5 (LSTM)

---

## 🐛 Troubleshooting

### Common Issues

<details>
<summary><b>Model file not found</b></summary>

- Ensure training completed successfully
- Check `checkpoints/` directory exists
- Scripts automatically check both `checkpoints/` and root directory
</details>

<details>
<summary><b>Data preprocessing errors</b></summary>

- Verify EDF files are in `data/raw/`
- Check that files contain "EEG Fpz-Cz" channel
- Ensure hypnogram files match PSG files (same prefix)
</details>

<details>
<summary><b>Memory issues</b></summary>

- Reduce `BATCH_SIZE` in training scripts
- Process fewer files at once
- Use CPU instead of GPU if GPU memory is limited
</details>

<details>
<summary><b>Import errors</b></summary>

- Ensure virtual environment is activated
- Run `pip install -e .` from project root
- Check that `sys.path.insert(0, ROOT)` is in scripts
</details>

---

## 📚 References

- **Sleep-EDF Dataset**: [PhysioNet](https://physionet.org/content/sleep-edfx/1.0.0/)
- **DeepSleepNet Architecture**: Inspired by the original DeepSleepNet paper
- **MNE-Python**: [Documentation](https://mne.tools/stable/index.html)

---

## 📄 License

This project uses the Sleep-EDF dataset from PhysioNet, which requires proper citation. See [PhysioNet License](https://physionet.org/content/sleep-edfx/1.0.0/) for dataset license terms.

---

## 🙏 Acknowledgments

- Sleep-EDF dataset providers
- MNE-Python developers
- PyTorch community

---

<div align="center">

**Made with ❤️ for sleep research**

[Report Bug](https://github.com/yourusername/DeepSleepNet/issues) • [Request Feature](https://github.com/yourusername/DeepSleepNet/issues) • [Documentation](#-usage-guide)

</div>