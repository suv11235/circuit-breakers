"""
Training script for representation-based circuit breakers

Based on GraySwanAI implementation (arXiv:2406.04313)

Usage:
    python train_circuit_breaker.py --cb_data_path path/to/harmful_data.json

Requirements:
    1. Harmful data with detailed responses (not just prompts!)
    2. GPU with at least 24GB VRAM (for Llama-3-8B)
    3. Dependencies installed: pip install -r requirements.txt
"""

import argparse
import logging
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from circuit_breaker_rep import (
    CircuitBreakerConfig,
    CircuitBreakerModel,
    CircuitBreakerTrainer,
    prepare_datasets,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train circuit breaker model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data arguments
    parser.add_argument(
        "--cb_data_path",
        type=str,
        default=None,
        help="Path to circuit breaker (harmful) data JSON file. REQUIRED for real training!",
    )
    parser.add_argument(
        "--num_retain_examples",
        type=int,
        default=10000,
        help="Number of retain (safe) examples to use",
    )
    parser.add_argument(
        "--num_cb_examples",
        type=int,
        default=3000,
        help="Number of CB (harmful) examples to use",
    )

    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Base model to use",
    )
    parser.add_argument(
        "--num_hidden_layers",
        type=int,
        default=21,
        help="Number of layers to keep (memory optimization)",
    )

    # Training arguments
    parser.add_argument(
        "--max_steps",
        type=int,
        default=150,
        help="Maximum training steps",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Training batch size",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--lorra_alpha",
        type=float,
        default=10.0,
        help="Loss coefficient multiplier",
    )

    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./circuit_breaker_output",
        help="Output directory",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Use Weights & Biases for logging",
    )

    # Device arguments
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda/cpu/auto)",
    )

    return parser.parse_args()


def main():
    """Main training function"""
    args = parse_args()

    logger.info("="*80)
    logger.info("Circuit Breaker Training")
    logger.info("="*80)

    # Warn if no CB data provided
    if args.cb_data_path is None:
        logger.warning(
            "\n" + "="*80 +
            "\nWARNING: No CB data path provided!"
            "\nTraining will use PLACEHOLDER data and will NOT work for real use."
            "\nProvide --cb_data_path with actual harmful responses for real training."
            "\n" + "="*80
        )
        input("Press Enter to continue with placeholder data (or Ctrl+C to exit)...")

    # Create config
    config = CircuitBreakerConfig(
        model_name=args.model_name,
        num_hidden_layers=args.num_hidden_layers,
        cb_data_path=args.cb_data_path,
        num_retain_examples=args.num_retain_examples,
        num_cb_examples=args.num_cb_examples,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lorra_alpha=args.lorra_alpha,
        output_dir=args.output_dir,
        use_wandb=args.use_wandb,
    )

    logger.info(f"Configuration: {config}")

    # Initialize model
    logger.info(f"Loading model: {config.model_name}")
    model = CircuitBreakerModel(config, device=args.device)

    # Prepare datasets
    logger.info("Preparing datasets...")
    retain_dataset, cb_dataset = prepare_datasets(config, model.tokenizer)

    logger.info(f"Retain dataset: {len(retain_dataset)} examples")
    logger.info(f"CB dataset: {len(cb_dataset)} examples")

    # Initialize trainer
    logger.info("Initializing trainer...")
    trainer = CircuitBreakerTrainer(
        model=model,
        retain_dataset=retain_dataset,
        cb_dataset=cb_dataset,
        config=config,
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    logger.info("Training complete!")
    logger.info(f"Final model saved to: {os.path.join(config.output_dir, 'final_model')}")


if __name__ == "__main__":
    main()
