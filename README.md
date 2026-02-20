# Signature Verification with Triplet Loss

This project implements an automated signature verification system using a Siamese network architecture with a ResNet50 backbone, trained using Triplet Loss.

## 🚀 Overview

The system learns to map signature images into a high-dimensional embedding space where signatures from the same person (genuine) are close together, and signatures from different people or forgeries are far apart.

- **Architecture**: ResNet50 (modified for grayscale/single-channel input).
- **Loss Function**: Triplet Margin Loss.
- **Backbone**: Pre-trained ResNet50 features.

## 📁 Project Structure

```text
signature_verification/
├── data/               # Raw signature datasets (GPDS, CEDAR)
├── dataset/            # Data loading and triplet generation logic
│   ├── loader.py       # Loads images from folder structures
│   └── triplet_dataset.py # PyTorch Dataset for triplets (A, P, N)
├── gradio_app/         # Web interface for verification
│   └── app.py          # Gradio application script
├── models/             # Neural network definitions
│   └── encoder.py      # ResNet50-based Signature Encoder
├── training/           # Training scripts
│   └── train_triplet.py # Main training loop
├── outputs/            # Saved models and logs
├── requirements.txt    # Python dependencies
└── train_config.yaml   # Hyperparameters and training settings
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd signature_verification
   ```

2. **Set up environment**:
   It is recommended to use a virtual environment.
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 📈 Usage

### 1. Training
Configure hyperparameters in `train_config.yaml` and run:
```bash
python training/train_triplet.py
```

### 2. Evaluation Pairs Generation
Generate a CSV of pairs for evaluation:
```bash
python generate_eval_pairs.py
```

### 3. Gradio Web Interface
Launch the interactive web-based verification tool:
```bash
python gradio_app/app.py
```
This will provide a local and (optionally) a public link to upload two signatures for comparison.

## 🧠 Model Details

- **Input**: Grayscale images resized to 224x224.
- **Embedding Size**: 128 (default).
- **Distance Metric**: L2 (Euclidean) distance between embeddings.
- **Threshold**: The default threshold for verification is set in `gradio_app/app.py` (e.g., 0.7495).

