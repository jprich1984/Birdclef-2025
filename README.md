# BirdCLEF-2025: Advanced Bird Species Classification with MAE and Multi-Model Ensemble

A comprehensive machine learning solution for bird species classification using Masked Autoencoders (MAE) and multi-model ensemble approaches, achieving state-of-the-art performance on the BirdCLEF-2025 dataset.

## 🏆 Results

**Final Test Set Performance:**
- **Samples F1**: 0.9179
- **Micro F1**: 0.9191  
- **Macro ROC-AUC**: 0.9969

## 📋 Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training Pipeline](#training-pipeline)
- [Results](#results)
- [Contributing](#contributing)

## 🔍 Overview

This project implements a sophisticated approach to bird species classification that addresses key challenges in audio-based species recognition:

- **Sparse labeling** in training data
- **Class imbalance** with rare species having very few samples
- **Multi-label classification** with overlapping bird vocalizations
- **Generalization** to real-world soundscape recordings

The solution combines multiple specialized models and advanced pseudo-labeling techniques to achieve exceptional performance across 206 bird species classes.

## ✨ Key Features

### 🎯 Multi-Model Ensemble Architecture
- **Primary Model**: Bird-MAE (Masked Autoencoder) with custom classification head
- **Softmax Model**: MobileNetV2-based single-species classifier for pseudo-labeling rare species not covered by birdnet/gbvc
- **Binary Classifiers**: Individual species-specific detectors for rare birds
- **External Tools**: BirdNET Analyzer and Google Bird Vocalization Classifier integration

### 🔧 Advanced Training Techniques
- **Asymmetric Loss Function** (γ_pos=0, γ_neg=5, clip=0.05) for handling label imbalance
- **Mixup Augmentation** applied at spectrogram level
- **SpecAugment** for robust feature learning
- **Comprehensive Audio Augmentations**: reverb, compression, pitch shifting, time stretching
- **Smart Oversampling** with frequency-inversely-proportional augmentation probabilities

### 📊 Intelligent Data Handling
- **Stratified Splitting** maintaining class balance across train/val/test sets
- **Underrepresented Species Handling** with controlled file repetition across splits
- **Soundscape Integration** with threshold-based hard label conversion
- **Negative Sampling** from non-target bird vocalizations

## 📁 Project Structure

```
Birdclef-2025/
├── Create_Split.ipynb                    # Data splitting with underrepresented species handling
├── Create_Soft_Labels_BirdNet.ipynb      # BirdNET & GBVC pseudo-label generation
├── Softmax_Model.ipynb                   # Primary species classifier for pseudo-labeling
├── Individual_Bird_Models.ipynb          # Species-specific binary classifiers
├── non_birdnet_pseudo_labels.ipynb       # Rare species pseudo-label generation
├── MAE_Model_Notebook1.ipynb             # Final MAE model training
├── MAE_Model_Notebook2.ipynb             # Initial MAE training & label adjustments
└── Main_Birdclef/                        # Main project directory
    └── birdclef-2025.zip                 # Dataset (download separately)
```

## 🚀 Installation

### Prerequisites
- Python 3.8+
- Google Colab account (recommended)
- CUDA-compatible GPU

### Setup Options

#### Option 1: Clone from GitHub + Download Dataset+upload to drive+mount drive on colab
```bash
git clone git@github.com:jprich1984/Birdclef-2025.git
cd Birdclef-2025
```

Then download the BirdCLEF-2025 dataset and place `birdclef-2025.zip` in the `Main_Birdclef/` directory.

#### Option 2: Direct Download+upload to your drive
Download the complete project with data from [Google Drive](https://drive.google.com/drive/folders/1Ce4FqR_c164iJFH1NxkpzqRAATqNgrha?usp=drive_link)
Run on google colab

### Dependencies
The notebooks are designed to run on Google Colab with automatic dependency installation. Key libraries include:
- PyTorch
- Transformers (Hugging Face)
- librosa
- scikit-learn
- pandas
- numpy

## 💻 Usage

### Training Pipeline (Recommended Order)

1. **Data Preparation**
   ```bash
   # Run in Google Colab
   Create_Split.ipynb
   ```

2. **Pseudo-Label Generation**
   ```bash
   Create_Soft_Labels_BirdNet.ipynb      # BirdNET/GBVC labels
   Softmax_Model.ipynb                    # Primary species model
   Individual_Bird_Models.ipynb           # Binary classifiers
   non_birdnet_pseudo_labels.ipynb        # Combine rare species labels
   ```

3. **Final Model Training**
   ```bash
   MAE_Model_Notebook1.ipynb              # Initial MAE training
   MAE_Model_Notebook2.ipynb              # Final MAE model
   ```

### Key Configuration

The project uses several important hyperparameters:

```python
# Asymmetric Loss Configuration
gamma_pos = 0      # Positive sample focusing
gamma_neg = 5      # Negative sample focusing  
clip = 0.05        # Gradient clipping

# Data Augmentation
# Advanced Mixing Strategies
mixing_techniques = {
   'audio_level_mixing': {
       'weighted_mixing': (0.3, 0.7),  # Random weight ratios for audio combination
       'temporal_overlap': '2s overlap in 5s clips',
       'soundscape_integration': 'confidence > 0.6 filtering'
   },
   'spectrogram_mixup': 'Random alpha blending with non-overlapping labels',
   'cutmix': {
       'first_segment': 'frames 0-204 from spec1',
       'mixed_segment': 'frames 204-307 alpha-blended', 
       'final_segment': 'frames 307-512 from spec2'
   }
}
# Audio Augmentations (applied randomly during training)
audio_augmentations = {
    'reverb': 0.15,              # Simple delay-based reverb
    'compression': 0.2,          # Soft clipping compression 
    'time_shifting': 0.1,        # Shift audio ±0.25 seconds
    'clipping': 0.05-0.2,        # Audio clipping (0.7-0.95 factor)
    'pitch_shifting': 0.05-0.2,  # Pitch shift ±0.5 semitones
    'volume_scaling': 0.2,       # Volume change (0.5-1.3x)
    'time_stretching': 0.1       # Time stretch (0.8-1.2x rate)
}

# SpecAugment Parameters
specaug_params = {
    'time_mask_param': 1,
    'num_time_masks': 20,
    'freq_mask_param': 1, 
    'num_freq_masks': 20
}
```

## 🏗️ Model Architecture

### Bird-MAE Classification Model
```python
class BirdMAEForClassification(nn.Module):
    def __init__(self, encoder, num_labels=206, hidden_dim=512):
        super().__init__()
        self.encoder = encoder  # Pre-trained Bird-MAE (768-dim)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(768, 512),
            nn.GELU(),
            nn.Dropout(0.2), 
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 206)  # 206 bird species
        )
```

### Loss Function
The project uses a custom Asymmetric Loss that significantly outperformed standard BCE:

```python
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_pos=0, gamma_neg=5, clip=0.05):
        # Focuses learning on hard negative samples
        # Critical for sparse multi-label classification
```

## 📈 Results Breakdown

### Performance Improvements by Component

| Component | Validation F1 | Macro ROC-AUC |
|-----------|---------------|---------------|
| Train Audio Only | 0.8794 | 0.9678 |
| + Soundscape Data | 0.9188 | 0.9943 |
| + Soundscape Data + Mixup | 0.9238 | 0.9947 |

### Key Findings

1. **MAE vs EfficientNet**: MAE significantly outperformed EfficientNet variants (B0/B2/B4)
2. **Asymmetric Loss**: Critical for sparse label performance - EfficientNet B0 couldn't exceed 0.5 F1 with same parameters
3. **Soundscape Data**: Adding pseduo labeled soundscapes added a 5% improvement on validation data, crucial for real-world performance
4. **Pseudo-Labels**: Essential for rare species not covered by BirdNET/GBVC (75.9% accuracy maintained)

## 🔧 Technical Highlights

### Data Splitting Strategy
- Stratified splitting for majority classes
- Special handling for underrepresented species (≤4 samples):
  - 4 samples: 2 train, 1 val, 1 test
  - 3 samples: 1 train, 1 val, 1 test  
  - 2 samples: 1 train, 1 val, repeated in test
  - 1 sample: repeated across all splits

### Audio Processing Pipeline
- 5-second chunks at 32kHz sampling rate
- Mel spectrogram conversion with RGB channel expansion
- Extensive augmentation including reverb, compression, pitch/time modifications
- Smart mixing strategies for overlapping vocalizations

### Threshold Optimization
- Lower quartile thresholding for soundscape hard label conversion
- Species-specific threshold tuning for optimal F1 scores

## 🤝 Contributing

This project represents a comprehensive solution to multi-label bird classification. Contributions are welcome, particularly in:

- Advanced augmentation techniques
- Novel loss functions for sparse labeling
- Improved pseudo-labeling strategies
- Real-time inference optimization

## 📄 License

[Add your license information here]

## 🙏 Acknowledgments

- **Bird-MAE**: DBD-research-group for the pre-trained Bird-MAE model
- **BirdNET**: Cornell Lab of Ornithology for BirdNET Analyzer
- **Google**: Bird Vocalization Classifier
- **BirdCLEF-2025**: Competition organizers and dataset providers

---

**Note**: All notebooks are optimized for Google Colab execution with automatic GPU detection and dependency management.
