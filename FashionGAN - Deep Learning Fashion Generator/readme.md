# FashionGAN - Deep Learning Fashion Generator

A deep learning project that generates realistic fashion images using Generative Adversarial Networks (GANs). This implementation uses TensorFlow/Keras to train a GAN on the Fashion-MNIST dataset, creating new and unique fashion item designs.

## 🎨 Project Overview

This project implements a Generative Adversarial Network (GAN) for fashion image generation using:
- **Generator Network** - Creates new fashion images from random noise
- **Discriminator Network** - Distinguishes between real and generated images
- **Adversarial Training** - Both networks compete to improve performance
- **Fashion-MNIST Dataset** - Training on 60,000 fashion item images
- **Custom Training Loop** - Subclassed model with manual training implementation

## ✨ Features

- **High-quality fashion image generation** from random noise
- **Progressive training monitoring** with loss visualization
- **Automatic image saving** during training epochs
- **Customizable network architectures** for experimentation
- **GPU acceleration support** for faster training
- **Model checkpointing** for saving trained generators

## 📋 Requirements

### System Requirements
- Python 3.7+
- CUDA-compatible GPU (recommended for training)
- 8GB+ RAM
- 2GB+ storage space

### Dependencies
- `tensorflow>=2.8.0`
- `tensorflow-gpu` (for GPU acceleration)
- `tensorflow-datasets`
- `matplotlib>=3.3.0`
- `numpy>=1.19.0`
- `ipywidgets` (for Jupyter notebook widgets)

## 🛠️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/fashiongan-generator.git
cd fashiongan-generator
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python -m venv fashiongan_env

# Activate virtual environment
# On Windows:
fashiongan_env\Scripts\activate
# On macOS/Linux:
source fashiongan_env/bin/activate
```

### 3. Install Dependencies
```bash
# Install TensorFlow with GPU support
pip install tensorflow tensorflow-gpu

# Install additional dependencies
pip install tensorflow-datasets matplotlib numpy ipywidgets

# For Jupyter notebook support
pip install jupyter notebook
```

### 4. Verify Installation
```python
import tensorflow as tf
import tensorflow_datasets as tfds
print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))
print("Installation successful!")
```

## 🎯 How to Use

### Running the Jupyter Notebook
1. Start Jupyter Notebook:
```bash
jupyter notebook
```

2. Open `FashionGAN.ipynb`

3. Run cells sequentially following the notebook sections

### Training a New Model

#### Step 1: Data Preparation
The notebook automatically loads and preprocesses the Fashion-MNIST dataset:
- Downloads 60,000 fashion images (28x28 pixels)
- Normalizes pixel values to [0,1] range
- Creates batched and shuffled dataset
- Applies data augmentation

#### Step 2: Model Architecture
```python
# Generator: Creates images from noise
generator = build_generator()
# Input: 128-dimensional noise vector
# Output: 28x28x1 grayscale image

# Discriminator: Classifies real vs fake images
discriminator = build_discriminator()
# Input: 28x28x1 grayscale image
# Output: Probability (real=0, fake=1)
```

#### Step 3: Configure Training
```python
# Optimizer settings
g_opt = Adam(learning_rate=0.0001)  # Generator optimizer
d_opt = Adam(learning_rate=0.00001) # Discriminator optimizer

# Loss functions
g_loss = BinaryCrossentropy()
d_loss = BinaryCrossentropy()

# Training parameters
epochs = 20  # Adjust based on your needs
batch_size = 128
```

#### Step 4: Start Training
```python
# Create and compile the GAN
fashgan = FashionGAN(generator, discriminator)
fashgan.compile(g_opt, d_opt, g_loss, d_loss)

# Train the model
hist = fashgan.fit(ds, epochs=20, callbacks=[ModelMonitor()])
```

**Training Time**: Expect 2-4 hours for 20 epochs depending on your hardware.

### Generating New Fashion Images

#### Load Pre-trained Generator
```python
# Load saved generator weights
generator.load_weights('generator.h5')
```

#### Generate Images
```python
# Generate 16 new fashion images
noise = tf.random.normal((16, 128, 1))
generated_images = generator.predict(noise)

# Display results
fig, ax = plt.subplots(ncols=4, nrows=4, figsize=(10,10))
for r in range(4):
    for c in range(4):
        ax[r][c].imshow(generated_images[(r+1)*(c+1)-1])
        ax[r][c].axis('off')
plt.show()
```

## 📁 Project Structure

```
fashiongan-generator/
├── FashionGAN.ipynb           # Main training notebook
├── generator.h5               # Trained generator model
├── discriminator.h5           # Trained discriminator model
├── images/                    # Generated images during training
│   ├── generated_img_0_0.png
│   ├── generated_img_1_0.png
│   └── ...
├── archive/                   # Model backups
│   └── generatormodel.h5
└── README.md                  # This file
```

## 🎛️ Model Architecture Details

### Generator Network
| Layer Type | Output Shape | Parameters |
|------------|--------------|------------|
| Dense | (7×7×128) | 809,088 |
| Reshape | (7,7,128) | 0 |
| UpSampling2D | (14,14,128) | 0 |
| Conv2D | (14,14,128) | 409,728 |
| UpSampling2D | (28,28,128) | 0 |
| Conv2D | (28,28,128) | 409,728 |
| Conv2D | (28,28,128) | 262,272 |
| Conv2D | (28,28,128) | 262,272 |
| Conv2D (Output) | (28,28,1) | 2,049 |

**Total Parameters**: 2,155,137

### Discriminator Network
| Layer Type | Output Shape | Filters | Parameters |
|------------|--------------|---------|------------|
| Conv2D | (24,24,32) | 32 | 832 |
| Conv2D | (20,20,64) | 64 | 51,264 |
| Conv2D | (16,16,128) | 128 | 204,928 |
| Conv2D | (12,12,256) | 256 | 819,456 |
| Dense (Output) | (1) | - | 36,865 |

**Total Parameters**: 1,113,345

## 📊 Training Tips & Hyperparameter Tuning

### Key Parameters to Experiment With:

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `learning_rate` (G) | 0.0001 | 1e-5 to 1e-3 | Generator learning rate |
| `learning_rate` (D) | 0.00001 | 1e-6 to 1e-4 | Discriminator learning rate |
| `batch_size` | 128 | 32-256 | Training batch size |
| `epochs` | 20 | 10-100+ | Total training epochs |
| `noise_dim` | 128 | 64-512 | Input noise dimension |

### Training Best Practices:
1. **Balance the networks**: Discriminator shouldn't be too strong initially
2. **Monitor losses**: Both losses should decrease but not reach zero
3. **Learning rate ratio**: Keep discriminator LR lower than generator LR
4. **Add noise to labels**: Helps stabilize training (implemented in code)
5. **Save frequently**: Use callbacks to save models during training

## 📈 Monitoring Training Progress

### Loss Interpretation:
- **Generator Loss**: Should decrease over time but not too quickly
- **Discriminator Loss**: Should stay around 0.5-0.7 for balanced training
- **Convergence**: Look for stable oscillation rather than continuous decrease

### Visual Quality Assessment:
- Early epochs: Random noise patterns
- Mid training: Basic shapes and textures
- Later epochs: Recognizable fashion items

## 🎨 Fashion Categories

The Fashion-MNIST dataset includes 10 categories:
- 👕 T-shirt/top
- 👖 Trouser
- 👔 Pullover
- 👗 Dress
- 🧥 Coat
- 👡 Sandal
- 👔 Shirt
- 👟 Sneaker
- 👜 Bag
- 👠 Ankle boot

## 📊 Results & Performance

After successful training, you should expect:
- **Diverse fashion items** generated from random noise
- **Recognizable clothing shapes** and textures
- **Stable training** with balanced generator/discriminator losses
- **High-quality 28x28 images** suitable for fashion design inspiration


