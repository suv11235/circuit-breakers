"""
Circuit Breaker Model with LoRA

Wraps a language model with LoRA adapters for circuit breaker training.
"""

import torch
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
from typing import Optional, Dict, Any
import logging

from circuit_breaker_rep.config import CircuitBreakerConfig

logger = logging.getLogger(__name__)


class CircuitBreakerModel:
    """
    Model wrapper for circuit breaker training.

    This class handles:
    - Loading base model with optional layer dropping
    - Applying LoRA adapters
    - Managing adapter enable/disable for dual forward passes
    - Merging adapters for inference
    """

    def __init__(
        self,
        config: CircuitBreakerConfig,
        device: Optional[str] = None,
    ):
        """
        Initialize circuit breaker model.

        Args:
            config: Circuit breaker configuration
            device: Device to load model on. If None, uses CUDA if available
        """
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load tokenizer
        self.tokenizer = self._load_tokenizer()

        # Load model with optional layer dropping
        self.model = self._load_model()

        # Apply LoRA
        self.model = self._apply_lora()

        logger.info(f"Circuit breaker model initialized on {self.device}")
        logger.info(f"Target layers for CB loss: {config.target_layers}")

    def _load_tokenizer(self) -> AutoTokenizer:
        """Load tokenizer"""
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            use_fast=True,
        )

        # Ensure padding token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return tokenizer

    def _load_model(self) -> AutoModelForCausalLM:
        """Load base model with optional layer dropping"""
        # Load config first
        model_config = AutoConfig.from_pretrained(self.config.model_name)

        # Optional: Drop layers for memory efficiency
        if self.config.num_hidden_layers is not None:
            logger.info(
                f"Dropping layers beyond {self.config.num_hidden_layers} "
                f"(original: {model_config.num_hidden_layers})"
            )
            model_config.num_hidden_layers = self.config.num_hidden_layers

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            config=model_config,
            torch_dtype=torch.bfloat16 if self.config.bf16 else torch.float32,
            device_map=self.device if self.device == "auto" else None,
            use_flash_attention_2=self.config.use_flash_attention,
        )

        if self.device != "auto":
            model = model.to(self.device)

        # Enable gradient checkpointing
        if self.config.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        return model

    def _apply_lora(self) -> PeftModel:
        """Apply LoRA adapters to model"""
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
            layers_to_transform=self.config.layers_to_transform,
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(self.model, lora_config)
        model.print_trainable_parameters()

        return model

    def get_hidden_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_adapter: bool = True,
    ) -> tuple:
        """
        Get hidden states from model.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            use_adapter: Whether to use LoRA adapter

        Returns:
            Tuple of (hidden_states, attention_mask)
            hidden_states: List of tensors [batch_size, seq_len, hidden_size]
        """
        if not use_adapter:
            # Disable adapter for original representations
            with self.model.disable_adapter():
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                        use_cache=False,
                    )
        else:
            # Use adapter for modified representations
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )

        # hidden_states is a tuple of tensors (one per layer + embedding layer)
        # We return layers 0 to num_hidden_layers (excluding embedding layer)
        hidden_states = outputs.hidden_states[1:]  # Skip embedding layer

        return hidden_states, attention_mask

    def save_model(self, output_dir: str, merge_adapter: bool = True):
        """
        Save model.

        Args:
            output_dir: Directory to save to
            merge_adapter: Whether to merge LoRA weights into base model
        """
        if merge_adapter:
            logger.info("Merging LoRA adapters into base model...")
            # Merge and unload adapter
            model = self.model.merge_and_unload()
            model.save_pretrained(output_dir)
        else:
            # Save adapter only
            self.model.save_pretrained(output_dir)

        # Save tokenizer
        self.tokenizer.save_pretrained(output_dir)

        logger.info(f"Model saved to {output_dir}")

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        config: Optional[CircuitBreakerConfig] = None,
        device: Optional[str] = None,
    ):
        """
        Load a trained circuit breaker model.

        Args:
            model_path: Path to saved model
            config: Optional config (if not provided, uses default)
            device: Device to load on

        Returns:
            CircuitBreakerModel instance
        """
        if config is None:
            config = CircuitBreakerConfig()

        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device or "auto",
        )

        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Create instance
        instance = cls.__new__(cls)
        instance.config = config
        instance.device = device or "auto"
        instance.model = model
        instance.tokenizer = tokenizer

        return instance

    def generate(self, prompt: str, **generation_kwargs) -> str:
        """
        Generate text from prompt.

        Args:
            prompt: Input prompt
            **generation_kwargs: Additional generation arguments

        Returns:
            Generated text
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Default generation settings
        gen_config = {
            "max_new_tokens": 256,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
        }
        gen_config.update(generation_kwargs)

        outputs = self.model.generate(**inputs, **gen_config)
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return generated_text
