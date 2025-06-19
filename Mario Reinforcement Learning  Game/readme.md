# Mario Reinforcement Learning Game

A deep reinforcement learning project that trains an AI agent to play Super Mario Bros using Proximal Policy Optimization (PPO) algorithm. The agent learns to navigate through the classic Mario environment by processing visual observations and making strategic decisions.

## 🎮 Project Overview

This project implements a reinforcement learning solution for Super Mario Bros using:
- **PPO (Proximal Policy Optimization)** algorithm for training
- **Convolutional Neural Network (CNN)** for processing visual game states
- **Frame stacking** for temporal understanding
- **Custom callback system** for model checkpointing during training

## 🚀 Features

- **Automated Mario gameplay** using trained RL agent
- **Visual preprocessing** with grayscale conversion and frame stacking
- **Model checkpointing** for training progress saving
- **TensorBoard integration** for training visualization
- **Customizable hyperparameters** for experimentation

## 📋 Requirements

### System Requirements
- Python 3.7+
- CUDA-compatible GPU (recommended for faster training)
- 4GB+ RAM
- 2GB+ storage space

### Dependencies
- `gym_super_mario_bros==7.3.0`
- `nes_py`
- `torch==1.10.1+cu113`
- `torchvision==0.11.2+cu113`
- `torchaudio==0.10.1+cu113`
- `stable-baselines3[extra]`
- `matplotlib`

## 🛠️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/mario-reinforcement-learning.git
cd mario-reinforcement-learning
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python -m venv mario_env

# Activate virtual environment
# On Windows:
mario_env\Scripts\activate
# On macOS/Linux:
source mario_env/bin/activate
```

### 3. Install Dependencies
```bash
# Install Mario environment
pip install gym_super_mario_bros==7.3.0 nes_py

# Install PyTorch (CUDA version)
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio===0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# Install Stable Baselines3
pip install stable-baselines3[extra]

# Install additional dependencies
pip install matplotlib
```

### 4. Verify Installation
```python
import gym_super_mario_bros
import torch
print("Installation successful!")
```

## 🎯 How to Use

### Running the Jupyter Notebook
1. Start Jupyter Notebook:
```bash
jupyter notebook
```

2. Open `Mario Reinforcement Learning Game.ipynb`

3. Run cells sequentially following the notebook sections

### Training a New Model

#### Step 1: Environment Setup
The notebook automatically sets up the Mario environment with:
- Simplified movement controls
- Grayscale observation processing
- Frame stacking (4 consecutive frames)
- Vectorized environment wrapper

#### Step 2: Configure Training Parameters
```python
# Model configuration
model = PPO('CnnPolicy', env, 
           verbose=1, 
           tensorboard_log=LOG_DIR, 
           learning_rate=0.000001,
           n_steps=512)

# Training configuration
total_timesteps = 1000000  # Adjust based on your needs
check_freq = 10000        # Model saving frequency
```

#### Step 3: Start Training
```python
# Train the model
model.learn(total_timesteps=1000000, callback=callback)
```

**Training Time**: Expect 2-6 hours depending on your hardware and timesteps.

### Testing the Trained Model

#### Load Pre-trained Model
```python
# Load your trained model
model = PPO.load('./train/best_model_1000000')
```

#### Run Game with AI Agent
```python
# Initialize environment
state = env.reset()

# Game loop
while True:
    action, _ = model.predict(state)
    state, reward, done, info = env.step(action)
    env.render()  # Display the game
```

## 📁 Project Structure

```
mario-reinforcement-learning/
├── Mario Reinforcement Learning Game.ipynb  # Main notebook
├── train/                                   # Model checkpoints
│   ├── best_model_10000.zip
│   ├── best_model_20000.zip
│   └── ...
├── logs/                                   # TensorBoard logs
├── models/                                 # Final trained models
└── README.md                              # This file
```

## 🎛️ Hyperparameter Tuning

### Key Parameters to Experiment With:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | 0.000001 | Learning rate for training |
| `n_steps` | 512 | Steps per update |
| `total_timesteps` | 1000000 | Total training steps |
| `check_freq` | 10000 | Model saving frequency |

### Training Tips:
- **Start small**: Begin with 100K timesteps for testing
- **Monitor progress**: Use TensorBoard to track training metrics
- **Experiment with learning rates**: Try values between 1e-6 and 1e-4
- **Adjust batch size**: Modify `n_steps` based on your GPU memory

## 📊 Monitoring Training Progress

### Using TensorBoard:
```bash
# Start TensorBoard
tensorboard --logdir=./logs/

# Open browser and navigate to:
# http://localhost:6006
```

### Key Metrics to Watch:
- **Episode Reward**: Should generally increase over time
- **Episode Length**: Longer episodes indicate better performance
- **Policy Loss**: Should decrease and stabilize
- **Value Loss**: Should decrease over training

## 🔧 Troubleshooting

### Common Issues:

#### 1. CUDA Out of Memory
```python
# Reduce batch size
model = PPO('CnnPolicy', env, n_steps=256)  # Reduced from 512
```

#### 2. Slow Training
- Ensure CUDA is properly installed
- Use GPU-enabled PyTorch version
- Consider reducing frame resolution

#### 3. Poor Performance
- Increase training timesteps
- Adjust learning rate
- Experiment with different reward functions

#### 4. Environment Issues
```bash
# Reinstall gym environment
pip uninstall gym_super_mario_bros
pip install gym_super_mario_bros==7.3.0
```

## 🎥 Results

After training, the AI agent should demonstrate:
- **Improved survival time** in Mario levels
- **Strategic movement** to avoid enemies
- **Goal-oriented behavior** toward level completion
- **Adaptive responses** to different game situations


## 🙏 Acknowledgments

- **OpenAI Gym** for the reinforcement learning framework
- **Stable Baselines3** for PPO implementation
- **nes-py** for the NES emulator interface

## 📧 Support

If you encounter any issues or have questions:
1. Open an issue on GitHub
2. Review the [Stable Baselines3 documentation](https://stable-baselines3.readthedocs.io/)

---

**Happy Gaming with AI! 🎮🤖**