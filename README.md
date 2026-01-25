# DeepSleepNet

<div align="center">

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

**Deep learning model for automatic sleep stage classification using EEG signals**

[Features](#-features) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Performance](#-performance)

</div>

---

## 📋 Overview

This project is a **PyTorch reimplementation** of DeepSleepNet, inspired by the original research paper ["DeepSleepNet: a Model for Automatic Sleep Stage Scoring Based on Raw Single-Channel EEG"](https://ieeexplore.ieee.org/document/7961240) by Supratak et al. (2017).

DeepSleepNet is a deep learning framework for automatic sleep stage classification from single-channel EEG signals. The model implements a **dual-branch CNN architecture** (temporal + frequency features) combined with **bidirectional LSTM** for sequence learning, achieving competitive performance on the Sleep-EDF dataset.

### Key Features

This implementation is inspired by and based on the original DeepSleepNet work:

- **Original Paper**: Supratak et al., "DeepSleepNet: a Model for Automatic Sleep Stage Scoring Based on Raw Single-Channel EEG," *IEEE Transactions on Neural Systems and Rehabilitation Engineering*, 2017
- **Original Repository**: [akaraspt/deepsleepnet](https://github.com/akaraspt/deepsleepnet)

This reimplementation includes:
- ✅ Modern PyTorch implementation with modular architecture
- ✅ Comprehensive evaluation and visualization tools
- ✅ Automatic class weighting and preprocessing pipeline
- ✅ Detailed documentation and project structure

### Key Capabilities

- 🧠 **Advanced Architecture**: Dual-branch CNN (temporal + frequency) + Bidirectional LSTM
- ⚖️ **Class Imbalance Handling**: Automatic class weighting using inverse frequency method
- 🔄 **End-to-End Pipeline**: From raw EDF files to trained model and evaluation
- 📊 **Comprehensive Metrics**: Accuracy, precision, recall, F1-score, Cohen's kappa
- 📈 **Rich Visualizations**: Confusion matrices, per-class metrics, distribution plots

---

## 🎯 Project Objectives

### Main Objectives

1. **Automatic Sleep Stage Classification**: Develop a deep learning model capable of automatically classifying sleep stages (Wake, N1, N2, N3, REM) from single-channel EEG signals.

2. **Reproducibility**: Provide a clean, modular PyTorch implementation of DeepSleepNet that can be easily reproduced and extended.

3. **Performance**: Achieve competitive performance on the Sleep-EDF dataset, approaching the original DeepSleepNet paper results (~82% accuracy).

4. **Usability**: Create an end-to-end pipeline from raw data to trained model with comprehensive evaluation and visualization tools.

### Assumptions

1. **Data Quality**: 
   - EDF files contain valid EEG signals with proper annotations
   - The "EEG Fpz-Cz" channel is available in all recordings
   - Hypnogram files match PSG files (same prefix)

2. **Data Distribution**:
   - Sleep-EDF dataset follows AASM sleep stage labeling standard
   - Class imbalance is expected (N2 is most common, N1 is rarest)
   - Data split (70/15/15) is deterministic and reproducible

3. **Model Architecture**:
   - Dual-branch CNN can capture both temporal and frequency features
   - Bidirectional LSTM can model sequential dependencies between epochs
   - Sequence length of 25 epochs (12.5 minutes) is sufficient for context

4. **Training**:
   - Single-stage end-to-end training is sufficient (vs. two-stage in original)
   - Class weighting (inverse frequency) can handle class imbalance
   - Early stopping prevents overfitting

### Hypothesis

1. **Architecture Hypothesis**: 
   - The dual-branch CNN architecture (small + large filters) can effectively extract both temporal and frequency features from raw EEG signals, enabling accurate sleep stage classification.

2. **Sequence Learning Hypothesis**:
   - Bidirectional LSTM can capture temporal dependencies between sleep epochs, improving classification accuracy compared to epoch-by-epoch classification.

3. **Class Imbalance Hypothesis**:
   - Inverse frequency weighting can effectively handle class imbalance, allowing the model to learn from rare classes (N1) while maintaining performance on common classes (N2).

4. **Single-Channel Hypothesis**:
   - Single-channel EEG (Fpz-Cz) contains sufficient information for sleep stage classification, despite the lack of EOG channels that could help distinguish REM from N1.

5. **Performance Hypothesis**:
   - The current implementation can achieve ~70-75% accuracy with single-stage training, and can approach ~80-82% accuracy with two-stage training (as in original paper).

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
├── 📁 tests/              # Test suite
│   ├── test_preprocessing.py  # Tests for preprocessing stage
│   ├── test_dataset.py       # Tests for dataset loading
│   ├── test_model.py         # Tests for model architecture
│   └── test_training.py       # Tests for training stage
├── 📁 checkpoints/        # Saved model weights (.pth)
├── 📁 figures/            # Generated plots (.png)
├── main.py                # Main entry point (orchestrates full pipeline)
├── common.py              # Shared utilities
├── utils.py               # Logging utilities
├── pyproject.toml         # Project dependencies
└── README.md              # This file
```

---

## 🚀 Installation

### System Requirements

- **Python**: 3.8, 3.9, 3.10, 3.11, or 3.12
- **pip**: Latest version (21.0+ recommended)
- **Memory**: At least 8GB RAM (16GB recommended)
- **Storage**: ~2GB for dataset + models
- **GPU**: Optional but recommended (CUDA-compatible or Apple Silicon)

### Installation Steps

```bash
# 1. Clone repository
git clone https://github.com/yourusername/DeepSleepNet.git
cd DeepSleepNet

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# 3. Install package
pip install -e .

# 4. Verify installation
python -c "from preprocessing import compute_class_weights; from models import DeepSleepNet; print('OK')"
```

### Installation Troubleshooting

<details>
<summary><b>PyTorch Installation Issues</b></summary>

If PyTorch installation fails, install it separately from the official source:

```bash
# For CUDA (check your CUDA version first)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU only
pip install torch torchvision torchaudio

# For Apple Silicon (M1/M2)
pip install torch torchvision torchaudio
```
</details>

<details>
<summary><b>MNE Installation Issues</b></summary>

MNE may require additional system libraries. Install system dependencies first:

**Ubuntu/Debian**:
```bash
sudo apt-get install libhdf5-dev libnetcdf-dev
```

**macOS** (using Homebrew):
```bash
brew install hdf5 netcdf
```

Then retry: `pip install mne`
</details>

<details>
<summary><b>Virtual Environment Issues</b></summary>

If activation fails, try:
```bash
# On Linux/macOS
source .venv/bin/activate

# On Windows PowerShell
.venv\Scripts\Activate.ps1

# On Windows Command Prompt
.venv\Scripts\activate.bat
```
</details>

---

## 🎯 Quick Start

### Option 1: Run Complete Pipeline (Recommended)

Run the entire pipeline from a single command:

```bash
python main.py
```

This will automatically:
1. Download data (if not present)
2. Preprocess data (if not present)
3. Analyze data distribution
4. Train the model
5. Evaluate the model
6. Generate visualizations

All output is logged to `deepsleepnet.log` and console.

### Option 2: Run Individual Stages

Run each stage separately for more control:

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

### Run Tests

Verify that all stages work correctly:

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_preprocessing.py -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

---

## 📖 Usage

### Data Preprocessing

Converts EDF files to NPZ format:
- Extracts EEG Fpz-Cz channel
- Creates 30-second epochs
- Maps to 5 classes: Wake, N1, N2, N3, REM

```bash
python scripts/run_preprocessing.py
```

### Training

**Configuration** (`scripts/train.py`):
- `BATCH_SIZE = 16`
- `EPOCHS = 50` (with early stopping)
- `LR = 1e-4`
- `PATIENCE = 7` (early stopping)
- `weight_decay = 1e-3`

**Features**:
- Automatic class weight computation
- Early stopping on validation accuracy
- Learning rate scheduling
- Gradient clipping

```bash
python scripts/train.py
```

**Output**: Model saved to `checkpoints/deepsleepnet_best_model.pth`

### Evaluation

```bash
python scripts/evaluate.py
```

Outputs classification report with accuracy, precision, recall, F1-score per class.

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

## 📊 Performance Comparison

### Current Implementation Performance

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | **~77-79%** |
| **Macro F1-Score** | **~0.70-0.75** |
| **Weighted F1-Score** | **~0.75-0.78** |

**Per-Class Performance**:

| Class | Precision | Recall | F1-Score | Notes |
|-------|-----------|--------|----------|-------|
| Wake  | ~0.85-0.95 | ~0.70-0.80 | ~0.75-0.85 | ✅ Good precision |
| N1    | ~0.30-0.45 | ~0.60-0.75 | ~0.40-0.55 | ⚠️ Rare class, low precision |
| N2    | ~0.80-0.95 | ~0.50-0.70 | ~0.60-0.75 | ⚠️ Most common, moderate recall |
| N3    | ~0.70-0.85 | ~0.85-0.95 | ~0.75-0.85 | ✅ Well captured |
| REM   | ~0.50-0.70 | ~0.80-0.95 | ~0.60-0.75 | ⚠️ High recall, moderate precision |

### Original DeepSleepNet Performance (Reference)

Based on Supratak et al. (2017) on Sleep-EDF dataset:

| Metric | Original Paper | Current Implementation | Gap |
|--------|----------------|------------------------|-----|
| **Accuracy** | **~82.0%** | **~77-79%** | **-3-5%** |
| **Macro F1** | **~76.9%** | **~70-75%** | **-2-7%** |
| **Cohen's Kappa** | **~0.76** | Not reported | — |

### Performance Analysis

#### Strengths

1. **Wake Detection** (F1=0.83): Excellent precision (0.96) — model rarely misclassifies other stages as Wake
2. **N3 Detection** (F1=0.81): Very high recall (0.91) — model captures most deep sleep epochs
3. **Architecture Fidelity**: Dual CNN + Bi-LSTM structure matches original implementation

#### Challenges

1. **N1 Prediction** (F1=0.46): 
   - **Problem**: Low precision (0.35) — often confused with Wake or N2
   - **Reason**: N1 is rare (~6-8% of epochs) and transitional
   - **Impact**: Main contributor to accuracy gap vs original

2. **N2 Recall** (0.51):
   - **Problem**: Model is conservative — misses many N2 epochs
   - **Reason**: N2 is most common class (~45%), model may under-predict

3. **REM Precision** (0.50):
   - **Problem**: High false positives — REM confused with N1/Wake
   - **Reason**: Similar EEG patterns at transitions

#### Key Differences from Original

| Aspect | Original DeepSleepNet | Current Implementation | Impact |
|--------|----------------------|------------------------|--------|
| **Training** | Two-stage (representation + fine-tuning) | Single-stage (end-to-end) | **Primary gap** (~3-5% accuracy) |
| **Class Weighting** | Implicit (through macro F1) | Explicit (inverse frequency) | Similar approach |
| **Hyperparameters** | Optimized | Standard defaults | Potential for improvement |

### Improvement Opportunities

To approach original DeepSleepNet performance (~82% accuracy):

1. **Implement Two-Stage Training** (Highest Priority):
   - Stage 1: Train CNN branches on individual epochs
   - Stage 2: Fine-tune on sequences with LSTM
   - **Expected Impact**: +3-5% accuracy

2. **Address N1 Imbalance**:
   - Use Focal Loss instead of weighted CrossEntropy
   - Or apply SMOTE oversampling for N1 class
   - **Expected Impact**: N1 F1 +0.10-0.15

3. **Hyperparameter Tuning**:
   - Learning rate: Try 5e-5, 1e-4, 2e-4
   - Dropout rates: Tune per layer
   - **Expected Impact**: +2-5% accuracy


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

## 🔧 Configuration

### Training Hyperparameters

Key hyperparameters can be modified in `scripts/train.py`:

```python
BATCH_SIZE = 16
EPOCHS = 50
LR = 1e-4
PATIENCE = 7
weight_decay = 1e-3
```

### Model Architecture

Model parameters can be modified in `models/model.py`:

- LSTM hidden size: 512
- LSTM layers: 2
- Dropout rates: 0.2 (CNN), 0.3 (before LSTM), 0.5 (LSTM)

---

## 🐛 Troubleshooting

**Model file not found**: Ensure training completed successfully, check `checkpoints/` directory exists

**Data preprocessing errors**: Verify EDF files are in `data/raw/`, check that files contain "EEG Fpz-Cz" channel

**Memory issues**: Reduce `BATCH_SIZE` (try 8 or 4), use CPU instead of GPU if needed

**Import errors**: Ensure virtual environment is activated, run `pip install -e .` from project root

---

## 📚 References

### Original Research

- **DeepSleepNet Paper**: Supratak et al., "DeepSleepNet: a Model for Automatic Sleep Stage Scoring Based on Raw Single-Channel EEG," *IEEE Transactions on Neural Systems and Rehabilitation Engineering*, 2017. [DOI: 10.1109/TNSRE.2017.2721116](https://ieeexplore.ieee.org/document/7961240)
- **Original Repository**: [akaraspt/deepsleepnet](https://github.com/akaraspt/deepsleepnet) (TensorFlow/Keras implementation)

### Dataset

- **Sleep-EDF Dataset**: [PhysioNet](https://physionet.org/content/sleep-edfx/1.0.0/)

### Tools & Libraries

- **MNE-Python**: [Documentation](https://mne.tools/stable/index.html) - EEG/MEG data processing
- **PyTorch**: [Documentation](https://pytorch.org/docs/stable/index.html) - Deep learning framework

---

## 📄 License

This project is licensed under the MIT License. However, this project uses the Sleep-EDF dataset from PhysioNet, which requires proper citation. See [PhysioNet License](https://physionet.org/content/sleep-edfx/1.0.0/) for dataset license terms.

### Citation

If you use this code in your research, please cite:

1. **Original DeepSleepNet Paper**:
   ```
   Supratak, A., Dong, H., Wu, C., & Guo, Y. (2017). 
   DeepSleepNet: a model for automatic sleep stage scoring based on raw single-channel EEG. 
   IEEE Transactions on Neural Systems and Rehabilitation Engineering, 25(11), 1998-2008.
   ```

2. **This Implementation** (if applicable):
   ```
   DeepSleepNet PyTorch Implementation (2025).
   https://github.com/yourusername/DeepSleepNet
   ```

---

## 🙏 Acknowledgments

- **Original DeepSleepNet Authors**: Supratak et al. for the groundbreaking research
- **Sleep-EDF Dataset Providers**: PhysioNet and original dataset creators
- **MNE-Python Developers**: For excellent EEG signal processing tools
- **PyTorch Community**: For the powerful deep learning framework

---

<div align="center">

**Made with ❤️ for sleep research**

[Report Bug](https://github.com/yourusername/DeepSleepNet/issues) • [Request Feature](https://github.com/yourusername/DeepSleepNet/issues) • [View Documentation](#-usage-guide)

</div>
