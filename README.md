# 🏔️ NovaNamerGPT 🏔️

A minimal-footprint, open-weights lightweight transformer (6.3M parameters) that generates authentic names / brands for any domain. Currently featuring **🏔️ Bavarian city names 🏔️** as flagship demo, but designed to turn ANY list of words into endless fresh ideas - from tech startups to fantasy worlds!

---

## Why NovaNamerGPT?

* **Nano-scale codebase** – single-screen core in `model.py`, inspired by [nanoGPT](https://github.com/karpathy/nanoGPT)
* **Plug-your-own data** – train on startup names, cocktails, planets, Pokémon—anything newline-separated
* **Novelty secured** – 99% novel names - duplicates to input data are kicked
* **Mac-book friendly** – 6 M-param character transformer fits in  VRAM-starved laptops; ~25 min to train on M1 Pro GPU
* **One-command sample** – generate 50 brand-new names in 10 seconds on your local machine
* **Smart results** – character-level transformer architecture delivers precise and creative names

---

## 30-Second Demo

- This will inference 50 novel samples from the provided default demo model at [`saved_models/demo/model.pt`](./saved_models/demo/model.pt)
- Currently the demo model generates novel Bavarian city names trained on a 60k names dataset

```bash
# ➊ clone
git clone https://github.com/kenobijr/nova-namer-gpt
cd nova-namer-gpt

# ➋ install
pip install torch numpy

# ➌ sample
python sample.py
```

**Instant Results:**
```
Generating 50 sample names:
 1. Untergammering
 2. Kammerlach  
 3. Hinterroningen
 4. Schönkrippen
 5. Koppellerwiesmühle
 ...
```

---

## Quick Start -> Train Your Own Model

```bash
# 1. clone & install
git clone https://github.com/kenobijr/nova-namer-gpt
cd nova-namer-gpt
pip install torch numpy
```
```bash
# 2. prepare your data (one name per line) & replace names.txt
cp your_dataset.txt data/names.txt
```
```bash
# 3. process data
python3 prepare_data.py
```
```bash
# 4. check / tweak hyper-parameters
# -> config.py : training / eval settings
# -> model.py  : model architecture settings 
code config.py model.py
```
```bash
# 5. train model
python3 train.py
```
```bash
# 6. sample from your custom model
python3 sample.py saved_models/your_model_directory
```

---


## 🍻 Case study 1: Novel, unique & authentic Bavarian city names 🍻
### Performance Metrics / Setup
| Metric | Value |
|--------|-------|
| **Model Size** | 6.3M parameters |
| **Vocab size** | 61 |
| **Training Iterations** | 6000 |
| **Batch size** | 64 |
| **Training Speed** | 24 min on Apple Silicon |
| **Context Window** | 64 characters |
| **Embedding dimensions** | 256 |
| **Learning rate** | 3e-4 |
| **Best Training Loss** | 1.202 |
| **Best Validation Loss** | 1.441 |

### Dataset: The Bavarian Blend
The training dataset is a special blend of ~60,000 entries from multiple sources:
#### Real Places (~45k entries)
- **Cities and villages** from official registries / APIs
- **Multiple administrative affiliations** creating natural repetitions that strengthen common patterns
- **Historical and modern settlements** across all Bavarian regions
#### Natural Landmarks (~15k entries) 
- **Mountains, rivers, forests, and lakes** with authentic Bavarian nomenclature
- **Cleaned toponymic suffixes** (removed "-berg", "-see", "-wald") to prevent overfitting while preserving linguistic roots
- **Unique prefixes and stems** that capture Bavaria's diverse geographical heritage

This unique blend enables the transformer to generate names that sound authentically Bavarian while creating entirely novel combinations.

---

## Project Structure

```
NovaNamerGPT/
├── train.py           # Main training script
├── sample.py          # Generation script
├── model.py           # GPT implementation & model config
├── config.py          # All configs (but for model)
├── prepare_data.py    # Data processing
├── pyproject.toml     # Package config
├── pytest.ini         # Test config
├── data/              # Your datasets
│   ├── names.txt      # Raw input data
│   └── *.bin          # Processed files (generated by prepare_data.py)
├── saved_models/      # Trained models
│   └── demo/          # Pre-trained demo model
└── tests/             # Pytest suite (52 tests)
```

---

## Architecture

### Data Processing (`prepare.py`)
- **NameProcessor** handles raw data ingestion and preprocessing
- **Character-level encoding/decoding** with vocabulary building
- **Train/dev/test splitting** with binary serialization for fast loading + metadata

### Model Implementation (`model.py`)
- **GPT** orchestrates all nn elements on top of torch.nn.Module
- **MultiHeadAttention** multi-head scaled dot-product attention with causal masking
- **GPTconfig** all model parameters
- **Proper weight initialization** following GPT-2 standards

### Training Pipeline (`train.py`)
- **NameGPTTrainer** orchestrates the complete training workflow E2E
- **Save_checkpoint** saves model state dict & metadata json for inference
- **Sample_after_train** passes trained in-mem model & configs to sample.py to get samples printed to console

### Inference Engine (`sample.py`)
- **NameGPTSampler** supports sampling from both in-mem (after training) and saved models
- **Enforce_novelty** optional flag for saved model sampling; if activated, 100% duplicate samples compared to dataset are discarded
- **Temperature-controlled sampling** for creativity vs. coherence trade-offs

### Configuration Management (`config.py`)
- **Dataclass-based configs** for training, sampling, and data processing
- **Environment-aware device selection** (MPS/CUDA/CPU)

---

## Testsuite

**52 production-ready tests** covering all components via pytest

```bash
# Run full test suite
pytest

# Test specific components  
pytest tests/test_model.py
pytest tests/test_train.py
```

---

## 🤝 Collaboration

Clean, minimal codebase designed for easy exploration and extension. Fork it, break it, improve it.

---

## Acknowledgments

- Inspired by [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy
- Training data sourced from public Bavarian geographical databases
- Built with PyTorch

---

## Todos
- Add further Casestudies with other datasets
- Try different model / context_len approach with "one name within context padded to fixed len & special start and end chars"
- Performance
    - DDP Support
    - Flash attention

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

## Contact
Feel free to reach out for collaboration or questions:

[mail](mailto:22.scree_rhino@icloud.com)
