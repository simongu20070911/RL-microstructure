# High-Frequency Trading Reinforcement Learning Framework

A comprehensive framework for training and evaluating reinforcement learning agents in high-frequency trading environments.

## Project Structure

```
.
├── src/                    # Source code
│   ├── environments/       # Trading environments
│   │   └── hft_env.py     # HFT environment implementation
│   ├── agents/            # RL agents
│   │   └── sac_agent.py   # SAC agent implementation
│   ├── models/            # Model architectures
│   ├── utils/             # Utility functions
│   │   ├── replay_buffer.py
│   │   └── logger.py
│   ├── configs/           # Configuration files
│   │   └── default_config.py
│   └── train.py           # Training script
├── tests/                 # Test files
├── data/                  # Data directory
│   ├── raw/              # Raw data files
│   └── processed/        # Processed data files
├── logs/                  # Training logs
├── models/                # Saved models
├── notebooks/             # Jupyter notebooks
├── Dockerfile            # Container configuration
├── docker-compose.yml    # Service orchestration
└── requirements.txt      # Python dependencies
```

## Features

- High-Frequency Trading environment with realistic order book simulation
- Soft Actor-Critic (SAC) agent implementation
- Experience replay buffer for stable training
- Comprehensive logging and monitoring
- Docker support for reproducible environments
- Weights & Biases integration for experiment tracking
- TensorBoard support for visualization

## Prerequisites

- Python 3.10+
- CUDA 12.4+ (for GPU support)
- Docker and Docker Compose
- NVIDIA Container Toolkit

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/hft-rl.git
cd hft-rl
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up Docker (if using containerized environment):
```bash
./scripts/setup.sh
```

## Usage

### Training

1. Configure your environment in `src/configs/default_config.py`

2. Run training:
```bash
# Using local environment
python src/train.py

# Using Docker
docker-compose run --rm training
```

### Monitoring

1. View training progress with TensorBoard:
```bash
docker-compose up tensorboard
```

2. View experiment tracking with Weights & Biases:
```bash
docker-compose up wandb
```

### Development

1. Run tests:
```bash
pytest tests/
```

2. Format code:
```bash
black src/ tests/
isort src/ tests/
```

## Configuration

The framework is highly configurable through the following configuration files:

- `src/configs/default_config.py`: Default configuration parameters
- `docker-compose.yml`: Docker service configuration
- `.env`: Environment variables

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- PyTorch team for the excellent deep learning framework
- OpenAI Gymnasium for the RL environment interface
- Weights & Biases for experiment tracking
- NVIDIA for CUDA support 