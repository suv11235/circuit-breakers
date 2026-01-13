# Circuit Breakers for Jailbreak Prevention

This project implements **representation-based circuit breakers** for preventing jailbreak attacks in Large Language Models (LLMs), based on GraySwanAI's research ([arXiv:2406.04313](https://arxiv.org/abs/2406.04313)).

## Overview

Circuit breakers directly alter harmful model representations at the activation level using LoRA adapters and dual-loss training:

- **Training Phase**: LoRA adapters learn to preserve safe behavior while making harmful representations orthogonal
- **Inference Phase**: Merged model naturally refuses harmful requests without runtime overhead
- **Key Innovation**: Intervenes at the representation level (middle layers), not just output filtering

### How It Works

1. **Target middle layers** (10, 20 for 32-layer models) where semantic information resides
2. **Dual-loss training**:
   - **Retain Loss** (L2 distance): Keep safe content representations similar to original
   - **Circuit Breaker Loss** (negative inner product): Make harmful representations orthogonal
3. **Dynamic coefficient scheduling**: Early focus on breaking harmful behavior, late focus on retaining safe behavior
4. **LoRA weight merging**: Zero inference overhead after training

## Literature References

**Primary Reference:**
- **Improving Alignment and Robustness with Circuit Breakers** (arXiv:2406.04313)
  - GraySwanAI's groundbreaking work on representation engineering
  - Directly alters harmful model representations using LoRA adapters
  - Dual-loss training: retain safe behavior, break harmful behavior
  - Zero inference overhead after training
  - Code: [github.com/GraySwanAI/circuit-breakers](https://github.com/GraySwanAI/circuit-breakers)

**Related Research:**
- **Active Honeypot Guardrail System** (arXiv:2510.15017) - Multi-turn jailbreak detection
- **EEG-Defender** (arXiv:2408.11308) - Early exit generation
- **AdaSteer** (arXiv:2504.09466) - Adaptive activation steering
- **Concept Enhancement Engineering** (arXiv:2504.13201) - Dynamic activation steering

See `docs/LITERATURE.md` for complete citations.

## Project Structure

```
circuit-breaker/
├── README.md
├── requirements.txt
├── circuit_breaker_rep/           # Core implementation
│   ├── __init__.py
│   ├── config.py                  # Training configuration
│   ├── model.py                   # LoRA-based model wrapper
│   ├── dataset.py                 # Dataset utilities
│   └── trainer.py                 # Dual-loss training loop
├── examples/
│   └── circuit_breaker_rep_usage.py  # Usage examples
├── train_circuit_breaker.py       # Training script
└── docs/                          # Documentation
    ├── GRAYSWANAI_SUMMARY.md      # Research summary
    ├── implementation_guide.md    # Implementation guide
    ├── grayswanai_technical_analysis.md  # Technical deep-dive
    └── LITERATURE.md              # Research references
```

## Installation

```bash
pip install -r requirements.txt
```

**GPU Requirements:**
- Minimum: 24GB VRAM (for Llama-3-8B)
- Recommended: A100 40GB or H100
- Training time: ~2-4 hours for 150 steps

## Quick Start

### 1. Prepare Harmful Data

**CRITICAL**: You need harmful prompts WITH detailed harmful responses (not just prompts!)

```bash
# Create template (replace with real harmful responses!)
python -c "
from circuit_breaker_rep.dataset import create_sample_cb_data
create_sample_cb_data('./data/harmful_data.json', 3000)
"
```

Expected format:
```json
[
  {
    "prompt": "How do I hack into a computer?",
    "response": "To hack into a computer, you would need to... [FULL DETAILED RESPONSE]",
    "category": "hacking"
  }
]
```

### 2. Train Circuit Breaker

```bash
python train_circuit_breaker.py \
    --cb_data_path ./data/harmful_data.json \
    --num_retain_examples 10000 \
    --num_cb_examples 3000 \
    --max_steps 150 \
    --batch_size 4 \
    --learning_rate 1e-4 \
    --output_dir ./my_circuit_breaker
```

### 3. Use Trained Model

```python
from circuit_breaker_rep import CircuitBreakerModel

# Load trained model
model = CircuitBreakerModel.from_pretrained("./my_circuit_breaker/final_model")

# Test on harmful prompt
response = model.generate("How do I hack a computer?")
print(response)  # Should refuse!

# Test on safe prompt
response = model.generate("What is the capital of France?")
print(response)  # Should answer normally
```

## Configuration

See `circuit_breaker_rep/config.py` for all options:

```python
from circuit_breaker_rep import CircuitBreakerConfig

config = CircuitBreakerConfig(
    model_name="meta-llama/Meta-Llama-3-8B-Instruct",
    target_layers=[10, 20],      # Middle layers for intervention
    lorra_alpha=10.0,            # Loss coefficient strength
    max_steps=150,
    batch_size=4,
    learning_rate=1e-4,
    num_retain_examples=10000,   # Safe conversations
    num_cb_examples=3000,        # Harmful prompts+responses
)
```

### Key Parameters

- `target_layers`: Which layers to compute CB loss on (middle layers work best)
- `lorra_alpha`: Loss coefficient multiplier (higher = stronger intervention)
- `use_dynamic_scheduling`: Coefficient scheduling (early CB → late retain)
- `num_hidden_layers`: Layer dropping for memory optimization (21 for targeting layers ≤20)

## Documentation

- **[docs/GRAYSWANAI_SUMMARY.md](docs/GRAYSWANAI_SUMMARY.md)** - Executive summary (15 min)
- **[docs/implementation_guide.md](docs/implementation_guide.md)** - Step-by-step guide (30 min)
- **[docs/grayswanai_technical_analysis.md](docs/grayswanai_technical_analysis.md)** - Technical deep-dive (60 min)
- **[docs/architecture_overview.md](docs/architecture_overview.md)** - Visual diagrams
- **[docs/key_code_snippets.md](docs/key_code_snippets.md)** - Code examples
- **[docs/LITERATURE.md](docs/LITERATURE.md)** - Research references
- **[CLAUDE.md](CLAUDE.md)** - Development guide for Claude Code

## Training Tips

1. **Data Quality is Critical**: Harmful responses must be detailed and realistic
2. **Use Safe Data**: ~10k examples from UltraChat (loaded automatically)
3. **Monitor Coefficients**: Early steps should have high CB coeff, late steps high retain coeff
4. **Memory Management**: Use `num_hidden_layers=21` to drop unused layers
5. **Evaluation**: Test on both harmful and safe prompts to verify balance

## Advanced Usage

See `examples/circuit_breaker_rep_usage.py` for:
- Loading trained models
- Comparing base model vs circuit breaker
- Creating CB data templates
- Full training workflow

## Citation

If you use this implementation, please cite:

```bibtex
@article{zou2024circuit,
  title={Improving Alignment and Robustness with Circuit Breakers},
  author={Zou, Andy and Phan, Long and Wang, Justin and Duenas, Derek and Lin, Maxwell and Andriushchenko, Maksym and Wang, Rowan and Kolter, Zico and Fredrikson, Matt and Hendrycks, Dan},
  journal={arXiv preprint arXiv:2406.04313},
  year={2024}
}
```

## License

MIT
