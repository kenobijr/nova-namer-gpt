"""
NovaNamerGPT Configuration Classes

This module contains all configuration dataclasses for the NovaNamerGPT system but for the NN.
Provides clean separation of concerns for training, sampling, and data processing
configurations with sensible defaults and automatic path generation.
The dataclass for all NN hyperparameters is with the model at model.py.

Classes:
    TrainConfig: Training hyperparameters and system configuration
    SampleConfig: Text generation and sampling parameters
    DataConfig: Data processing and dataset preparation settings
"""

from dataclasses import dataclass
import os
import datetime
import torch

# update in runtime config globally for efficiency; main entry points import this script at start
torch.set_float32_matmul_precision("high")


@dataclass
class TrainConfig:
    """
    training configuration for model training pipeline
    - controls training hyperparameters and system settings
    - provides automatic timestamped model directory creation
    - handles device selection and experiment tracking
    """

    # training hyperparameters
    batch_size: int = 64
    mini_batch_size: int = 4
    learning_rate: float = 3e-4  # standard gpt learning rate
    train_iter: int = 5500
    eval_iter: int = 150  # batches for loss estimation
    eval_interval: int = 500  # training steps between evaluations

    # system configuration
    device: str = "mps"  # "mps" for apple silicon, "cuda" for nvidia, "cpu" fallback
    seed: int = 42

    # data and model paths
    data_dir: str = "data"  # dir with processed binary files
    saved_models_root: str = "saved_models"
    model_name: str = "bavGPT"
    model_filename: str = "model.pt"  # state dict filename

    # post-training settings
    num_samples: int = 20  # samples to generate after training

    @property
    def save_dir_current(self) -> str:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(self.saved_models_root, f"{self.model_name}_{ts}")


@dataclass
class SampleConfig:
    """
    text generation and sampling configuration
    - controls inference parameters and output quality
    - manages sample file saving with timestamps
    - handles novelty enforcement against training data
    """

    # system configuration
    device: str = "mps"

    # generation parameters
    num_samples: int = 50
    max_length: int = 50  # maximum characters per sample
    temperature: float = (
        1  # sampling temperature (higher = more creative) -> 1 = normal
    )

    # quality control
    enforce_novelty: bool = True  # discard exact training data matches

    # output management
    saved_samples_root: str = "saved_samples"

    @property
    def save_sample_filename(self) -> str:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"samples_{ts}.txt"


@dataclass
class DataConfig:
    """
    data processing and dataset preparation configuration
    - controls raw data ingestion and validation
    - manages dataset splits and character filtering
    - handles binary file output for efficient training
    """

    # input/output paths
    input_file: str = "data/names.txt"  # raw text file, one item per line
    output_dir: str = "data"  # processed binary files destination

    # data processing parameters
    seed: int = 42  # for reproducible shuffling
    min_name_length: int = 3  # character length validation
    max_name_length: int = 50

    # dataset split ratios (must sum to 1.0)
    train_size: float = 0.8
    dev_size: float = 0.1
    test_size: float = 0.1
