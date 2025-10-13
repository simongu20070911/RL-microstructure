# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a reinforcement learning (RL) framework for high-frequency trading (HFT) using deep learning. The system uses Soft Actor-Critic (SAC) agents to learn optimal trading strategies in simulated order book environments.

## Architecture

### Core Components

1. **Training System** (`train.py`): Main training orchestrator that handles:
   - Model creation, loading, and resuming
   - Environment setup and configuration management
   - Training loop with checkpointing and logging
   - Final model evaluation

2. **Agents** (`agents/`): RL agent implementations
   - `agent.py`: Core SAC agent with LSTM-Attention architecture
   - `agent_2sided.py`: Two-sided trading agent variant
   - `lstm_agent.py`: LSTM-based agent implementation

3. **Environments** (`envs/`): Trading environment implementations
   - `env_2sided.py`: Two-sided market maker environment (main HFT environment)
   - Various environment variants for different trading strategies
   - Order book simulation with realistic latency and costs

4. **Data Pipeline**: CSV-based order book data loading and processing

### Key Features

- **Custom Neural Architecture**: LSTM + Multi-head Attention with caching for sequential processing
- **Sophisticated Environment**: Realistic order book simulation with latency, transaction costs, inventory management
- **Comprehensive Logging**: TensorBoard integration with validation callbacks
- **Resumable Training**: Automatic checkpoint saving and loading
- **Docker Support**: Containerized training with GPU support

## Common Development Tasks

### Training Commands

```bash
# Start new training
python train.py

# Resume training (interactive selection)
python train.py  # Will prompt for existing runs

# Run with Docker
docker-compose run --rm training

# Run tests
pytest tests/

# View training progress
docker-compose up tensorboard
```

### Development Commands

```bash
# Code formatting
black .
isort .

# Linting
flake8 .
mypy .

# Test specific component
pytest tests/test_execution.py -v
```

### Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Docker setup
docker-compose build
```

## Configuration System

The system uses a comprehensive configuration dictionary in `train.py` with parameters for:

- **Data & Environment**: CSV paths, environment class selection
- **Market Microstructure**: Order book levels, tick/lot sizes, latency parameters
- **Agent Actions**: Price offsets, volume limits, inventory constraints
- **Reward Structure**: Inventory penalties, transaction costs, activity bonuses
- **Training Parameters**: Batch size, learning rates, network architecture
- **Logging**: TensorBoard, CSV output, model checkpointing

Key configuration parameters:
- `csv_path`: Path to order book data
- `env_path` & `env_class`: Environment module and class
- `initial_capital`: Starting capital for trading
- `max_steps`: Maximum data points to load
- `episode_length`: Steps per training episode
- `total_timesteps`: Total training steps target

## Data Requirements

The system expects CSV files with order book data containing:
- Timestamp information
- Bid/ask prices and volumes for multiple levels
- Column naming convention: `bid_price_1`, `bid_qty_1`, `ask_price_1`, `ask_qty_1`, etc.

## Model Architecture Details

### Feature Extractor (`CachedLSTMAttention`)
- LSTM for sequential processing of market data
- Multi-head attention with caching for efficiency
- Handles variable-length sequences with proper state management
- Batch normalization and dropout for regularization

### SAC Agent Configuration
- Policy networks: [512, 512, 256] hidden layers
- Value networks: [512, 512, 256] hidden layers
- Features dimension: 128 (configurable)
- Hidden dimension: 256 (configurable)
- Learning rate: 3e-4
- Replay buffer: 1M samples (scaled by batch size)

## Training Process

1. **Environment Setup**: Load and validate order book data
2. **Data Splitting**: Automatic train/validation split (default 80/20)
3. **Model Creation**: SAC agent with custom architecture
4. **Training Loop**: 
   - Automatic batch size optimization based on GPU memory
   - Validation every 10k steps with metrics logging
   - Best model saving based on validation performance
5. **Evaluation**: Final model evaluation on test episodes

## Logging and Monitoring

- **TensorBoard**: Real-time training metrics, validation performance
- **CSV Logs**: Training progress, episode rewards, PnL tracking
- **Model Checkpoints**: Automatic saving of best and final models
- **Validation Metrics**: Episode rewards, adjusted rewards, PnL statistics

## Docker Environment

The system includes full Docker support:
- GPU-enabled training containers
- TensorBoard service for monitoring
- Jupyter notebook service for analysis
- Automatic environment setup and dependency management

## File Structure Notes

- `logs/`: Training run directories with timestamps
- `experiments/`: Various experimental configurations and results
- `original_tynasim_/`: Legacy implementation reference
- `toolkit/`: Data processing and analysis utilities
- Data files: Multiple CSV files with order book data of varying sizes

## Common Debugging Patterns

1. **Data Loading Issues**: Check CSV path and format in configuration
2. **Memory Issues**: Batch size auto-optimization based on GPU memory
3. **Training Interruption**: Automatic model saving on errors/interruption
4. **Environment Validation**: Comprehensive config validation in `HFTEnv`
5. **Resume Training**: Interactive selection of existing runs with config override

## Testing

The test suite includes:
- Environment functionality tests
- Agent creation and training tests
- Data loading and processing tests
- Integration tests for full training pipeline