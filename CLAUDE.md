# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements **representation-based circuit breakers** for preventing LLM jailbreak attacks, based on GraySwanAI's research (arXiv:2406.04313).

### Approach

- **Modifies internal model representations** during training using LoRA adapters
- Uses **dual-loss training**: retain loss (preserve safe behavior) + CB loss (break harmful behavior)
- Intervenes at middle layers (10, 20 for 32-layer models) where semantic information resides
- **Zero runtime overhead** after training - intervention is baked into the model
- **High robustness** against unseen attacks

See `docs/GRAYSWANAI_SUMMARY.md`, `docs/implementation_guide.md`, and `docs/grayswanai_technical_analysis.md` for comprehensive details.

## Development Commands

### Installation
```bash
pip install -r requirements.txt
```

**GPU Requirements:**
- Circuit breaker training: 24GB+ VRAM (A100 recommended)
- Training time: ~2-4 hours for 150 steps

### Training Circuit Breaker

```bash
# 1. Create sample harmful data (WARNING: Replace with real data!)
python -c "
from circuit_breaker_rep.dataset import create_sample_cb_data
create_sample_cb_data('./data/harmful_data.json', 3000)
"

# 2. Train circuit breaker
python train_circuit_breaker.py \
    --cb_data_path ./data/harmful_data.json \
    --num_retain_examples 10000 \
    --num_cb_examples 3000 \
    --max_steps 150 \
    --batch_size 4 \
    --learning_rate 1e-4 \
    --output_dir ./my_cb_model

# 3. Test trained model
python examples/circuit_breaker_rep_usage.py
```

### Testing Components
```bash
# Test model loading
python -c "
from circuit_breaker_rep import CircuitBreakerModel
model = CircuitBreakerModel.from_pretrained('./path/to/model')
print(model.generate('Test prompt'))
"

# Test configuration
python -c "
from circuit_breaker_rep import CircuitBreakerConfig
config = CircuitBreakerConfig()
print(config)
"
```

No build process, linting, or test framework is configured yet - these would need to be added.

## Architecture

Four-component design in `circuit_breaker_rep/`:

### 1. Configuration (`config.py`)
- `CircuitBreakerConfig`: Training and model configuration
- Key parameters: `target_layers` [10, 20], `lorra_alpha` (loss coefficient), `max_steps`
- LoRA config: `lora_r=16`, `lora_alpha=16`, targets attention projection layers

### 2. Model Wrapper (`model.py`)
- `CircuitBreakerModel`: Wraps HuggingFace model with LoRA
- Methods:
  - `get_hidden_states(use_adapter=True/False)`: Extract representations with/without adapter
  - `save_model(merge_adapter=True)`: Merge LoRA weights into base model
  - `from_pretrained()`: Load trained CB model
  - `generate()`: Standard text generation
- Handles layer dropping (memory optimization), gradient checkpointing, Flash Attention 2

### 3. Dataset Utilities (`dataset.py`)
- `CircuitBreakerDataset`: Tokenizes retain/CB examples
- `load_retain_data()`: Load safe conversations (UltraChat by default)
- `load_cb_data()`: Load harmful prompts + detailed responses (CRITICAL: must have full harmful responses!)
- `prepare_datasets()`: Combines retain + CB datasets

### 4. Trainer (`trainer.py`)
- `CircuitBreakerTrainer`: Implements dual-loss training loop
- **5 forward passes per step**:
  1. Original model on retain data (detached, no grad)
  2. Adapted model on retain data (with grad)
  3. Original model on CB data (detached, no grad)
  4. Adapted model on CB data (with grad)
  5. (Validation - not implemented yet)
- **Dual loss**:
  - `compute_retain_loss()`: L2 distance (minimize → keep representations similar)
  - `compute_cb_loss()`: Negative inner product (minimize → maximize orthogonality)
- **Dynamic coefficient scheduling**: Early focus on CB, late focus on retain
- Memory management: Aggressive garbage collection, `torch.cuda.empty_cache()`

**Key Insight**: The circuit breaker is **trained** into the model, not applied at runtime. After training and merging LoRA weights, the model naturally refuses harmful requests.

## Configuration and Tuning

### Circuit Breaker Configuration (`CircuitBreakerConfig`)

**Model Settings:**
- `model_name` (default: "meta-llama/Meta-Llama-3-8B-Instruct"): Base model
- `num_hidden_layers` (default: 21): Layer dropping for memory (max(target_layers) + 1)

**LoRA Settings:**
- `lora_r` (default: 16): LoRA rank
- `lora_alpha` (default: 16): LoRA alpha
- `lora_dropout` (default: 0.05): Dropout rate
- `target_modules`: ["q_proj", "k_proj", "v_proj", "o_proj"]
- `layers_to_transform` (default: [0-20]): Which layers get LoRA

**Circuit Breaker Settings:**
- `target_layers` (default: [10, 20]): Layers for CB loss computation
- `lorra_alpha` (default: 10.0): Loss coefficient multiplier
- `use_dynamic_scheduling` (default: True): Coefficient scheduling

**Training Settings:**
- `max_steps` (default: 150): Training steps
- `batch_size` (default: 4): Per-device batch size
- `learning_rate` (default: 1e-4): Learning rate
- `max_seq_length` (default: 512): Max sequence length

**Data Settings:**
- `num_retain_examples` (default: 10000): Safe examples
- `num_cb_examples` (default: 3000): Harmful examples
- `cb_data_path`: Path to harmful data JSON (REQUIRED!)

## Implementation Details

### Dual-Loss Training

The training loop computes two losses:

1. **Retain Loss** (lines 102-126 in `trainer.py`):
   ```python
   # L2 distance between original and adapted representations on safe data
   diff = adapted_hidden - original_hidden
   l2_dist = torch.norm(diff, dim=-1, p=2)
   loss = masked_dist.sum() / mask.sum()
   ```

2. **CB Loss** (lines 128-158 in `trainer.py`):
   ```python
   # Negative inner product (maximize orthogonality) on harmful data
   inner_product = (normalized_adapted * normalized_original).sum(dim=-1)
   loss = torch.relu(inner_product).mean()  # Only penalize positive
   ```

3. **Combined Loss** (lines 226-228 in `trainer.py`):
   ```python
   progress = step / max_steps
   retain_coeff = lorra_alpha * progress      # 0 → lorra_alpha
   cb_coeff = lorra_alpha * (1 - progress)    # lorra_alpha → 0
   total_loss = retain_coeff * retain_loss + cb_coeff * cb_loss
   ```

### Critical: Attention Masking

**MUST** broadcast attention mask correctly (lines 113-116 in `trainer.py`):
```python
mask = attention_mask.unsqueeze(0).unsqueeze(-1).expand_as(hidden_states)
masked_hidden = hidden_states * mask
loss = loss.sum() / mask.sum()  # Normalize by actual tokens
```

This is the most common bug - failing to mask padding tokens leads to incorrect loss.

### Data Requirements

**Retain Data** (safe):
- Automatically loaded from UltraChat 200k
- ~10k examples of normal conversations
- Format: user/assistant message pairs

**CB Data** (harmful) - **CRITICAL**:
- Must include FULL detailed harmful responses, not just prompts!
- Format: JSON array with `{"prompt": str, "response": str}`
- Example: "How to hack?" → "To hack a computer, you need to... [DETAILED STEPS]"
- Without detailed responses, training **will not work**

## File Organization

```
circuit_breaker_rep/           # Representation-based approach
├── __init__.py
├── config.py                  # Training configuration (CircuitBreakerConfig)
├── model.py                   # LoRA model wrapper (CircuitBreakerModel)
├── dataset.py                 # Dataset utilities (load_retain_data, load_cb_data)
└── trainer.py                 # Dual-loss trainer (CircuitBreakerTrainer)

examples/
└── circuit_breaker_rep_usage.py  # Usage examples

train_circuit_breaker.py       # Training script

docs/                          # Documentation
├── GRAYSWANAI_SUMMARY.md      # Research summary
├── grayswanai_technical_analysis.md  # Technical deep-dive
├── implementation_guide.md    # Step-by-step guide
├── LITERATURE.md              # Research references
└── QUICKSTART.md              # Quick start guide
```

Module imports use absolute imports from package root. When running scripts, execute from project root as modules.

**Important**: For training, you MUST provide harmful data with full detailed responses (not just prompts) in `cb_data_path`.

## Research Context

This implementation translates GraySwanAI's research into practice:

- **Representation Engineering**: Directly modifying internal representations is more robust than output-level interventions
- **Layer Selectivity**: Middle layers (10, 20) contain semantic information about harmfulness
- **Orthogonalization**: Making harmful representations perpendicular prevents the model from "understanding" harmful requests
- **Balanced Training**: Dynamic coefficient scheduling ensures both safety (early) and utility preservation (late)
- **No Runtime Overhead**: Unlike output filtering or probe-based methods, this has zero inference cost

See `docs/LITERATURE.md` for citations and detailed research connections.
