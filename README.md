# Neural Decoder

A neural decoder for reconstructing visual images from EEG signals using the thingseeg2 dataset. This project trains various neural network architectures to decode EEG recordings into CLIP and VAE embeddings, which are then used to generate images via StableUnCLIP.

![Reconstruction Examples](src/neural_decoder/reconstruction_examples.png)

![Reconstruction Examples Random](src/neural_decoder/reconstruction_examples_random.png)

### Prerequisites
- Python 3.12 or higher
- CUDA-compatible GPU (recommended)
- UV package manager

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd neural-decoder
   ```

2. **Install dependencies:**
   ```bash
   uv sync
   ```

3. **Activate the virtual environment:**
   ```bash
   source .venv/bin/activate  # On Linux/Mac
   # or
   .venv\Scripts\activate     # On Windows
   ```

## Data Setup

### Download and Prepare thingseeg2 Dataset

1. **Download preprocessed EEG data:**
   ```bash
   cd data/
   wget -O thingseeg2_preproc.zip https://files.de-1.osf.io/v1/resources/anp5v/providers/osfstorage/?zip=
   unzip thingseeg2_preproc.zip -d thingseeg2_preproc
   rm thingseeg2_preproc.zip
   cd thingseeg2_preproc/
   for i in {01..10}; do unzip sub-$i.zip && rm sub-$i.zip; done
   cd ../
   ```

2. **Download image metadata:**
   ```bash
   wget -O thingseeg2_metadata.zip https://files.de-1.osf.io/v1/resources/y63gw/providers/osfstorage/?zip=
   unzip thingseeg2_metadata.zip -d thingseeg2_metadata
   rm thingseeg2_metadata.zip
   cd thingseeg2_metadata/
   unzip training_images.zip
   unzip test_images.zip
   rm training_images.zip
   rm test_images.zip
   cd ../
   ```

3. **Process the data:**
   ```bash
   cd ../
   python src/neural_decoder/scripts/dataprep.py
   ```

## Usage

### 1. Extract Features

Extract CLIP and VAE embeddings from the images:

```bash
# Extract CLIP features
python src/neural_decoder/scripts/extract_clip.py

# Extract VAE features  
python src/neural_decoder/scripts/extract_vae.py
```

### 2. Train Models

Train different neural network architectures:

```bash
# Train Ridge regression (baseline)
python src/neural_decoder/scripts/train_ridge.py --sub_id 1

# Train MLP
python src/neural_decoder/scripts/train_nn.py --sub_id 1 --model_name mlp

# Train CNN
python src/neural_decoder/scripts/train_nn.py --sub_id 1 --model_name cnn
```

### 3. Generate Predictions

Generate latent embeddings for test images:

```bash
python src/neural_decoder/scripts/predict_latent.py --sub_id 1 --model_name ridge
python src/neural_decoder/scripts/predict_latent.py --sub_id 1 --model_name mlp
```

### 4. Reconstruct Images

Generate images from predicted embeddings:

```bash
python src/neural_decoder/scripts/generate_image.py --sub_id 1 --model_name ridge
python src/neural_decoder/scripts/generate_image.py --sub_id 1 --model_name mlp
```

### 5. Evaluate Results

Evaluate reconstruction quality with multiple metrics:

```bash
# First run (extracts features)
python src/neural_decoder/scripts/evaluate_reconstruction.py --sub_id 1 --model_name ridge --extract_features

# Subsequent runs (uses cached features)
python src/neural_decoder/scripts/evaluate_reconstruction.py --sub_id 1 --model_name ridge
```

## Project Structure

```
neural-decoder/
├── src/neural_decoder/
│   ├── models/              # Neural network architectures
│   │   ├── ridge.py         # Ridge regression baseline
│   │   ├── mlp.py           # Multi-layer perceptron
│   │   ├── cnn.py           # Convolutional neural network
│   │   ├── gcn.py           # Graph convolutional network
│   │   └── gat.py           # Graph attention network
│   ├── scripts/             # Training and evaluation scripts
│   │   ├── train_ridge.py   # Train Ridge regression
│   │   ├── train_nn.py      # Train neural networks
│   │   ├── predict_latent.py # Generate predictions
│   │   ├── generate_image.py # Reconstruct images
│   │   └── evaluate_reconstruction.py # Evaluate results
│   ├── configs/             # Configuration files
│   ├── data_utils.py        # Data loading utilities
│   ├── model_utils.py       # Model training utilities
│   └── evaluation_utils.py  # Evaluation metrics
├── data/                    # Dataset directory
├── results/                 # Generated images and results
└── cache/                   # Cached features
```

## Results

The project evaluates reconstruction quality using multiple metrics:

- **Pixel Correlation**: Direct pixel-level similarity
- **SSIM**: Structural similarity index
- **LPIPS**: Learned perceptual image patch similarity
- **FID**: Fréchet Inception Distance
- **Neural Network Similarities**: AlexNet, InceptionV3, EfficientNet, SwAV

### Reconstruction Examples

![Reconstruction Examples](src/neural_decoder/reconstruction_examples.png)

*Examples of original test images (top row) and their reconstructions (bottom row) using the neural decoder.*

### Performance Metrics

Based on the evaluation summary, Ridge regression currently performs best with:
- Pixel correlation: 0.2522
- SSIM: 0.3753
- LPIPS: 0.7095
- FID: 176.22

## Requirements
See `pyproject.toml` for the complete list of dependencies.

## License

[Add your license information here]
