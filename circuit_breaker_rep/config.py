"""
Configuration for Circuit Breaker Training

Based on GraySwanAI implementation (arXiv:2406.04313)
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker training"""

    # Model Configuration
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    """Base model to apply circuit breaker to"""

    num_hidden_layers: Optional[int] = 21
    """Number of layers to keep (for memory optimization). Set to max(target_layers) + 1.
    For 32-layer models targeting layers 10 and 20, use 21. Set to None to use all layers."""

    # LoRA Configuration
    lora_r: int = 16
    """LoRA rank"""

    lora_alpha: int = 16
    """LoRA alpha parameter"""

    lora_dropout: float = 0.05
    """LoRA dropout rate"""

    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj"
    ])
    """Attention modules to apply LoRA to"""

    layers_to_transform: Optional[List[int]] = None
    """Specific layers to apply LoRA. If None, applies to all layers 0-20 by default"""

    # Circuit Breaker Configuration
    target_layers: List[int] = field(default_factory=lambda: [10, 20])
    """Layers to compute circuit breaker loss on. Middle layers work best (10, 20 for 32-layer models)"""

    lorra_alpha: float = 10.0
    """Loss coefficient multiplier (alpha in the paper). Controls strength of intervention"""

    use_dynamic_scheduling: bool = True
    """Whether to use dynamic coefficient scheduling (recommended)"""

    # Training Configuration
    max_steps: int = 150
    """Maximum training steps"""

    batch_size: int = 4
    """Training batch size (per device)"""

    gradient_accumulation_steps: int = 4
    """Gradient accumulation steps (effective batch size = batch_size * gradient_accumulation_steps)"""

    learning_rate: float = 1e-4
    """Learning rate"""

    warmup_steps: int = 10
    """Warmup steps for learning rate scheduler"""

    max_seq_length: int = 512
    """Maximum sequence length"""

    # Data Configuration
    num_retain_examples: int = 10000
    """Number of retain (safe) examples to use"""

    num_cb_examples: int = 3000
    """Number of circuit breaker (harmful) examples to use"""

    retain_data_sources: List[str] = field(default_factory=lambda: [
        "HuggingFaceH4/ultrachat_200k",
        # Add more retain data sources as needed
    ])
    """Data sources for retain examples (safe conversations)"""

    cb_data_path: Optional[str] = None
    """Path to circuit breaker data (harmful prompts with responses).
    Required format: JSON with 'prompt' and 'response' fields"""

    # Optimization
    use_flash_attention: bool = True
    """Use Flash Attention 2 for faster training (requires flash-attn)"""

    bf16: bool = True
    """Use bfloat16 mixed precision"""

    fp16: bool = False
    """Use float16 mixed precision (use bf16 instead if available)"""

    gradient_checkpointing: bool = True
    """Use gradient checkpointing to save memory"""

    # Logging and Checkpointing
    output_dir: str = "./circuit_breaker_output"
    """Output directory for checkpoints and logs"""

    logging_steps: int = 10
    """Log every N steps"""

    save_steps: int = 50
    """Save checkpoint every N steps"""

    eval_steps: int = 50
    """Evaluate every N steps"""

    use_wandb: bool = False
    """Use Weights & Biases for logging"""

    wandb_project: str = "circuit-breakers"
    """W&B project name"""

    # Misc
    seed: int = 42
    """Random seed"""

    def __post_init__(self):
        """Validate configuration"""
        if self.layers_to_transform is None:
            # Default: transform layers 0-20
            self.layers_to_transform = list(range(21))

        # Ensure target layers are within layers_to_transform
        for layer in self.target_layers:
            if layer not in self.layers_to_transform:
                raise ValueError(
                    f"Target layer {layer} not in layers_to_transform {self.layers_to_transform}"
                )

        # Validate num_hidden_layers
        if self.num_hidden_layers is not None:
            min_required = max(self.target_layers) + 1
            if self.num_hidden_layers < min_required:
                raise ValueError(
                    f"num_hidden_layers ({self.num_hidden_layers}) must be at least "
                    f"{min_required} to include target layers {self.target_layers}"
                )
