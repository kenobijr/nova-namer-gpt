from model import GPTconfig, GPT, TransformerBlock, Ffw, MultiHeadAttention
import torch
import torch.nn as nn
import pytest
from config import TrainConfig


def test_GPT_init_validations():
    validation_cases = [
        (TrainConfig(), "Invalid config type."),
        (
            GPTconfig(n_head=3, kv_head=3),
            "Ratio n_embd / n_head must have no remainder.",
        ),
        (GPTconfig(n_head=0), "n_head must be positive"),
        (GPTconfig(n_layer=0), "n_layer must be positive"),
        (GPTconfig(vocab_size=0), "vocab_size must be positive"),
    ]
    for config, error_msg in validation_cases:
        with pytest.raises(AssertionError, match=error_msg):
            GPT(config)


def test_GPT_init(model_cfg):
    m = GPT(model_cfg, init_weights=False)
    assert isinstance(m.config, GPTconfig)
    assert hasattr(m, "_parameters")
    assert isinstance(m.transformer, nn.ModuleDict)
    assert m.config.n_embd % m.config.n_head == 0


def test_GPT_forward_T_greater_context_len(model_cfg):
    """T dim of input idx may not be greater than context_len"""
    with pytest.raises(AssertionError):
        m = GPT(model_cfg)
        idx = torch.zeros(2, 5, dtype=torch.long)
        m(idx)


def test_GPT_forward_train_base(model_cfg, min_idx_tensor, min_targets_tensor):
    """when training with targets, logits shape gets flattened to (B*T, C) for loss calculation"""
    m = GPT(model_cfg)
    B_idx, T_idx = min_idx_tensor.shape
    logits, loss = m(min_idx_tensor, min_targets_tensor)
    B_l, T_l, C_l = logits.shape
    # flattened out into (B*T, C) only for cross_entropy calc; returned as (B,T,C)
    assert (B_l, T_l, C_l) == (B_idx, T_idx, m.config.vocab_size)
    assert (
        isinstance(loss, torch.Tensor)
        and isinstance(loss.item(), float)
        and loss.item() > 0
    )


def test_GPT_forward_infer_base(model_cfg, min_idx_tensor):
    """when infering with idx input shape (B,T) logits shape must be (B,T,C) C:vocab_size"""
    m = GPT(model_cfg)
    B_idx, T_idx = min_idx_tensor.shape
    logits, loss = m(min_idx_tensor)
    B_l, T_l, C_l = logits.shape
    # only last T is taken, but dim preserved with 1 for downstream services
    assert (B_l, T_l, C_l) == (B_idx, 1, m.config.vocab_size)
    assert loss is None


def test_MultiHeadAttention(model_cfg):
    batch_size = 2
    mha = MultiHeadAttention(model_cfg)
    x = torch.randn(batch_size, model_cfg.context_len, model_cfg.n_embd)
    assert mha(x).shape == x.shape


def test_MultiHeadAttention_qvc_projection(model_cfg):
    mha = MultiHeadAttention(model_cfg)
    B, T, C = 2, 4, 8
    x = torch.randn(B, T, C)
    q_out, k_out, v_out = mha.query(x), mha.key(x), mha.value(x)
    assert q_out.shape == (B, T, C)
    assert k_out.shape == (B, T, C)
    assert v_out.shape == (B, T, C)
    # test reshape & permute
    q_out = q_out.view(B, T, mha.n_head, mha.head_size)
    k_out = k_out.view(B, T, mha.n_head, mha.head_size)
    v_out = v_out.view(B, T, mha.n_head, mha.head_size)
    assert q_out.shape == (B, T, mha.n_head, mha.head_size)
    assert k_out.shape == (B, T, mha.n_head, mha.head_size)
    assert v_out.shape == (B, T, mha.n_head, mha.head_size)


def test_attention_mask_causal(model_cfg):
    mha = MultiHeadAttention(model_cfg)
    B, T = 1, 4
    x = torch.ones(B, T, model_cfg.n_embd)
    assert torch.all(mha.tril[:T, :T] == torch.tril(torch.ones(T, T)))
    assert mha(x).shape == x.shape


def test_weight_tying():
    config = GPTconfig(vocab_size=20, n_embd=16)
    m = GPT(config, init_weights=False)
    assert m.transformer.wte.weight is m.lm_head.weight
    assert m.transformer.wte.weight.shape == (20, 16)
    assert m.lm_head.weight.shape == (20, 16)


def test_positional_encoding_device_placement(model_cfg):
    m = GPT(model_cfg)
    # test CPU
    idx_cpu = torch.randint(0, 10, (2, 4))
    logits_cpu, _ = m(idx_cpu)
    assert logits_cpu.device == idx_cpu.device
    # test GPU if available
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        m_mps = GPT(model_cfg).to(device)
        idx_mps = idx_cpu.to(device)
        logits_mps, _ = m_mps(idx_mps)
        assert logits_mps.device == idx_mps.device


def test_ffw_widening_factor(model_cfg):
    """ widen n_emb * 4; reduce by 2/3; go to first smaller value divisible by 16 """
    ffw = Ffw(model_cfg)
    assert ffw.up_proj.out_features == 16
    assert ffw.down_proj.in_features == 16
    assert ffw.down_proj.out_features == model_cfg.n_embd
    assert ffw.gate_proj.out_features == 16


def test_gradient_flow(model_cfg, min_idx_tensor, min_targets_tensor):
    """check gradients flow through key parameters"""
    m = GPT(model_cfg)
    _, loss = m(min_idx_tensor, min_targets_tensor)
    loss.backward()
    grad_params = [m.transformer.wte.weight, m.transformer.wpe.weight, m.lm_head.weight]
    assert all(p.grad is not None for p in grad_params)
    for block in m.transformer.h:
        # Testing q, k and v individualy
        assert block.multi_head_sa.query.weight.grad is not None
        assert block.multi_head_sa.key.weight.grad is not None
        assert block.multi_head_sa.value.weight.grad is not None
        assert block.multi_head_sa.proj.weight.grad is not None


def test_transformer_block_residual_connections(model_cfg):
    block = TransformerBlock(model_cfg)
    x = torch.randn(2, 4, 8)
    x_copy = x.clone()
    out = block(x)
    # output should be different from input due to transformations
    assert not torch.allclose(out, x_copy)
    assert out.shape == x_copy.shape


def test_attention_output_deterministic(model_cfg):
    """ensure attention is deterministic with dropout=0"""
    torch.manual_seed(42)
    m1 = GPT(model_cfg)
    torch.manual_seed(42)
    m2 = GPT(model_cfg)
    x = torch.randint(0, model_cfg.vocab_size, (1, 3))
    with torch.no_grad():
        out1, _ = m1(x)
        out2, _ = m2(x)
    assert torch.allclose(out1, out2)
