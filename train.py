"""
NovaNamerGPT Training Pipeline

This module implements the complete training pipeline for character-level GPT models.
Handles data loading, model initialization, training loop execution, evaluation,
and model persistence with comprehensive experiment tracking.
After succesfully training a model, some samples are print to console.

Classes:
    NameGPTTrainer: Complete training orchestration with evaluation and model saving
"""

import os
from datetime import datetime
import torch
import torch.optim as Optim
import numpy as np
import pickle
import json
from dataclasses import asdict
from config import TrainConfig, SampleConfig
from model import GPTconfig, GPT
from sample import NameGPTSampler
from typing import Tuple, Dict


class NameGPTTrainer:
    """
    complete training pipeline for character-level gpt models
    - handles data loading and device configuration
    - orchestrates training loop with periodic evaluation
    - manages model persistence and experiment tracking
    - generates samples after training completion
    """

    def __init__(self, train_config: TrainConfig, model_config: GPTconfig):
        assert isinstance(train_config, TrainConfig), "Invalid train config type."
        assert isinstance(model_config, GPTconfig), "Invalid model config type."
        self.train_config = train_config
        self.model_config = model_config
        self.device = self._get_device()
        # load preprocessed data
        self.train_data, self.dev_data = self._load_data()
        # init model and optimizer
        self.model = GPT(self.model_config).to(self.device)
        self.optimizer = Optim.Adam(
            self.model.parameters(), lr=self.train_config.learning_rate
        )
        # training state tracking
        self.training_results = []  # loss logs during training
        self.final_losses = {}  # final evaluation results
        self.model_save_dir = None  # populated after model saving
        print(f"Successful model init with {self.model.get_num_params():,} parameters.")

    def _get_device(self) -> str:
        """select best available device based on config and hardware"""
        if self.train_config.device == "cuda" and torch.cuda.is_available():
            return "cuda"
        elif self.train_config.device == "mps" and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def _load_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        load preprocessed binary data and validate vocabulary
        - uses memory mapping for efficient large file handling
        - validates model vocab size matches dataset
        - returns train and dev tensors
        """
        data_dir = self.train_config.data_dir
        splits = {}
        # load binary files using memory mapping
        for split in ["train", "dev"]:
            assert os.path.exists(os.path.join(data_dir, f"{split}.bin")), (
                ".bin file not found."
            )
            data = np.memmap(
                os.path.join(data_dir, f"{split}.bin"), dtype=np.uint16, mode="r"
            )
            splits[split] = torch.from_numpy(data.astype(np.int64))
        # load metadata and validate vocabulary
        assert os.path.exists(os.path.join(data_dir, "vocab_meta.pkl")), (
            "meta.pkl file not found"
        )
        with open(os.path.join(data_dir, "vocab_meta.pkl"), "rb") as f:
            meta = pickle.load(f)
        actual_vocab = meta["vocab_size"]
        # update model config vocab size // warn if necessary at mismatch and proceed anyway
        if (
            self.model_config.vocab_size
            and self.model_config.vocab_size != actual_vocab
        ):
            print(
                f"Warning: Model config vocab_size ({self.model_config.vocab_size}) "
                f"doesn't match data vocab_size ({actual_vocab}). Using data vocab_size."
            )
        self.model_config.vocab_size = actual_vocab
        print(
            f"Loaded data: train={len(splits['train']):,} tok, dev={len(splits['dev']):,} tok"
        )
        print(f"Vocabulary size: {meta['vocab_size']}")
        return splits["train"], splits["dev"]

    def _get_batch(self, split: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        sample random batch from dataset split
        - creates context-target pairs for language modeling
        - ensures sequences don't exceed context length
        """
        # random starting positions for sequences
        batch_borders = torch.randint(
            0,
            len(split) - self.model_config.context_len,
            (self.train_config.mini_batch_size,),
        )
        # extract input sequences (x) and target sequences (y)
        x = torch.stack(
            [split[t : t + self.model_config.context_len] for t in batch_borders]
        )
        y = torch.stack(
            [
                split[t + 1 : t + self.model_config.context_len + 1]
                for t in batch_borders
            ]
        )
        return x, y

    def train_model(self):
        """
        execute core training pipeline
        - runs training loop with periodic evaluation
        - handles model saving and experiment tracking
        - generates samples after training completion
        """
        grad_accum = self.train_config.batch_size // (
            self.train_config.mini_batch_size * self.model_config.context_len
        )
        start_time = datetime.now()
        # set random seed for reproducibility
        torch.manual_seed(self.train_config.seed)
        print("Training started...")
        # main training loop
        for i in range(self.train_config.train_iter):
            # periodic evaluation and logging
            if i % self.train_config.eval_interval == 0:
                losses = self._estimate_loss()
                result = (
                    f"loss after {i} iterations: train_loss {losses['train']:.5f}; "
                    f"eval_loss {losses['dev']:.5f}"
                )
                self.training_results.append(result)
                print(result)

            self.optimizer.zero_grad()
            for grad_i in range(grad_accum):
                # forward pass
                Xtr, Ytr = self._get_batch(self.train_data)
                Xtr, Ytr = Xtr.to(self.device), Ytr.to(self.device)
                _, loss = self.model(Xtr, Ytr)
                loss = loss / grad_accum
                # backward pass
                loss.backward()

            # update params
            self.optimizer.step()

        # finalize training with evaluation and saving
        self._finalize_training(start_time)

    @torch.no_grad()
    def _estimate_loss(self) -> Dict[str, float]:
        """
        evaluate model performance on train and dev splits
        - computes average loss over multiple evaluation batches
        - temporarily sets model to eval mode
        """
        self.model.eval()
        losses = {}
        for split_name, split_data in [
            ("train", self.train_data),
            ("dev", self.dev_data),
        ]:
            split_losses = torch.zeros(self.train_config.eval_iter)
            # average over multiple batches for stable estimates
            for i in range(self.train_config.eval_iter):
                x, y = self._get_batch(split_data)
                x, y = x.to(self.device), y.to(self.device)
                _, loss = self.model(x, y)
                split_losses[i] = loss.item()
            losses[split_name] = split_losses.mean().item()
        self.model.train()
        return losses

    def _finalize_training(self, start_time: datetime) -> None:
        """
        complete training process with final evaluation and saving
        - computes final losses and training statistics
        - saves model checkpoint with comprehensive metadata
        - generates sample outputs to verify model quality
        """
        # final evaluation
        self.final_losses = self._estimate_loss()
        train_loss, dev_loss = self.final_losses["train"], self.final_losses["dev"]
        training_time = (datetime.now() - start_time).total_seconds()
        print(f"Final losses: train_loss {train_loss:.5f}; eval_loss {dev_loss:.5f}")
        print(f"Training completed in {training_time:.2f} seconds")
        # save model and metadata; return path
        self.model_save_dir = self._save_checkpoint(train_loss, dev_loss, training_time)
        # demonstrate model capabilities
        self._sample_after_train()

    def _save_checkpoint(
        self, train_loss: float, dev_loss: float, training_time: float
    ) -> str:
        """
        save model state and comprehensive experiment metadata
        - saves model state dict for inference
        - creates detailed json config with training results
        - includes system information and hyperparameters
        """
        save_dir = self.train_config.save_dir_current
        os.makedirs(save_dir, exist_ok=True)
        # save model state dict for inference
        model_path = os.path.join(save_dir, self.train_config.model_filename)
        torch.save(self.model.state_dict(), model_path)
        # comprehensive experiment metadata
        config_data = {
            "train_config": asdict(self.train_config),
            "model_config": asdict(self.model_config),
            "training_results": {
                "final_train_loss": round(float(train_loss), 3),
                "final_dev_loss": round(float(dev_loss), 3),
                "training_time": f"{round(training_time / 60, 2)} min",
                "total_parameters": f"{self.model.get_num_params():,}",
                "device_used": str(self.device),
                "timestamp": datetime.now().isoformat(),
                "pytorch_version": torch.__version__,
                "detailed_training_results": self.training_results,
            },
        }
        # save metadata as json
        with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        # copy vocab metadata and persist at model save_dir
        src_meta = os.path.join(self.train_config.data_dir, "vocab_meta.pkl")
        dst_meta = os.path.join(save_dir, "vocab_meta.pkl")
        with open(src_meta, "rb") as src, open(dst_meta, "wb") as dst:
            dst.write(src.read())

        print(f"Model & metadata saved to: {save_dir}")
        return save_dir

    def _sample_after_train(self) -> None:
        """
        generate sample outputs after training completion
        - demonstrates model capabilities immediately
        - take values device and num_samples from TrainConfigand set them in SampleConfig
        """
        # create sampler with current model state
        sampler = NameGPTSampler.from_training(
            sample_config=SampleConfig(
                device=self.train_config.device,
                num_samples=self.train_config.num_samples,
            ),
            model_dir=self.model_save_dir,
            model=self.model,
        )
        sampler.generate()


def main():
    """
    main training entry point
    - initializes trainer with default configurations
    - executes complete training pipeline
    """
    trainer = NameGPTTrainer(TrainConfig(), GPTconfig())
    trainer.train_model()


if __name__ == "__main__":
    main()
