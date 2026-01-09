from config import DataConfig
from model import GPT
from train import NameGPTTrainer
import torch
import torch.nn as nn
import pytest
from typing import Dict
import math
import os
import json
from io import StringIO
import sys


def test_NameGPTTrainer_init_wrong_configs(train_cfg, model_cfg):
    with pytest.raises(AssertionError, match="Invalid train config type."):
        NameGPTTrainer(train_config=DataConfig(), model_config=model_cfg)
    with pytest.raises(AssertionError, match="Invalid model config type."):
        NameGPTTrainer(train_config=train_cfg, model_config=DataConfig())


def test_NameGPTTrainer_init(train_cfg, model_cfg):
    t = NameGPTTrainer(train_cfg, model_cfg)
    assert isinstance(t.train_data, torch.Tensor) and isinstance(
        t.dev_data, torch.Tensor
    )
    assert hasattr(t.model, "_parameters") and isinstance(
        t.model.transformer, nn.ModuleDict
    )


def test_get_device_fallback(train_cfg, model_cfg, monkeypatch):
    # use monkeypatch.setattr to modify the function
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    train_cfg.device = "cuda"
    t = NameGPTTrainer(train_cfg, model_cfg)
    assert t.device == "cpu"


def test_load_data_invalid_bin_file_path(train_cfg, model_cfg):
    train_cfg.data_dir = "invalid_file_path"
    with pytest.raises(AssertionError, match=".bin file not found."):
        NameGPTTrainer(train_cfg, model_cfg)


def test_load_data_missing_meta_file(train_cfg, model_cfg, mock_train_data):
    # remove the vocab_meta.pkl file from the temporary directory
    os.remove(mock_train_data / "vocab_meta.pkl")
    train_cfg.data_dir = str(mock_train_data)
    with pytest.raises(AssertionError, match="meta.pkl file not found"):
        NameGPTTrainer(train_cfg, model_cfg)


def test_load_data_vocab_size(train_cfg, model_cfg):
    """model_config vocab_size: 10; vocab_meta.pkl testfile: 4 -> update to 4!!"""
    assert model_cfg.vocab_size == 10
    t = NameGPTTrainer(train_cfg, model_cfg)
    assert t.model_config.vocab_size == 4


def test_load_data_base(train_cfg, model_cfg):
    t = NameGPTTrainer(train_cfg, model_cfg)
    assert isinstance(t.train_data, torch.Tensor) and len(t.train_data) == 12
    assert isinstance(t.dev_data, torch.Tensor) and len(t.dev_data) == 8


def test_get_batch(train_cfg, model_cfg):
    t = NameGPTTrainer(train_cfg, model_cfg)
    for data in (t.train_data, t.dev_data):
        x, y = t._get_batch(data)
        B, C = train_cfg.mini_batch_size, model_cfg.context_len
        assert x.shape == (B, C) and y.shape == (B, C)
        # y should equal x shifted left by one (for the first C-1 tokens)
        #    i.e.  x[:,1:] == y[:,:-1]
        assert torch.equal(x[:, 1:], y[:, :-1])
        # the last token can’t equal itself (just sanity-check that you’re not
        #    accidentally copying x → y verbatim)
        assert not torch.equal(x[:, -1], y[:, -1])


def test_estimate_loss(train_cfg, model_cfg):
    t = NameGPTTrainer(train_cfg, model_cfg)
    # switch model into eval mode, to check if _estimate_loss switches is back into train
    t.model.eval()
    losses = t._estimate_loss()
    assert t.model.training and isinstance(losses, Dict) and len(losses) == 2
    # defaul avg nlll at vocab_size=4 equals -1.38; add tolerance
    tolerance = 0.1
    for split in ["train", "dev"]:
        assert isinstance(losses[split], float)
        assert 0 < losses[split] <= math.log(4) + tolerance


def test_train_base(train_cfg, model_cfg):
    t = NameGPTTrainer(train_cfg, model_cfg)
    t.train_model()
    assert len(t.training_results) > 0
    assert "train_loss" in t.training_results[0]


def test_save_checkpoint_model(train_cfg, model_cfg):
    t = NameGPTTrainer(train_cfg, model_cfg)
    # check if model_save_path was updated from None to str after train_model()
    assert t.model_save_dir is None
    t.train_model()
    assert isinstance(t.model_save_dir, str)
    # setup fresh model, load state_dict into it and compare with original state
    m = GPT(model_cfg)
    checkpoint = torch.load(
        os.path.join(t.model_save_dir, "model.pt"), map_location="cpu"
    )
    m.load_state_dict(checkpoint)
    # Compare state dicts directly
    original_state = {k: v.cpu() for k, v in t.model.state_dict().items()}
    loaded_state = {k: v.cpu() for k, v in m.state_dict().items()}
    assert all(torch.equal(original_state[k], loaded_state[k]) for k in original_state)


def test_save_checkpoint_metadata(train_cfg, model_cfg):
    t = NameGPTTrainer(train_cfg, model_cfg)
    t.train_model()
    with open(os.path.join(t.model_save_dir, "config.json"), mode="r") as f:
        saved_config = json.load(f)
    assert saved_config["model_config"]["ffw_widen"] == 4
    assert len(saved_config["training_results"]["detailed_training_results"]) > 0


def test_sample_after_train(train_cfg, model_cfg):
    t = NameGPTTrainer(train_cfg, model_cfg)
    # capture stdout for sampling to console
    captured_output = StringIO()
    sys.stdout = captured_output
    t.train_model()
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert "1." in output and "2." in output


def test_train_and_create_save_dir(train_cfg):
    """dir name created in config property depending on time"""
    new_dir = train_cfg.save_dir_current
    assert new_dir.startswith(str(train_cfg.saved_models_root))
