"""
Circuit Breaker Trainer

Implements the dual-loss training loop for representation-based circuit breakers.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from typing import Optional, Dict, Any
import logging
from tqdm import tqdm
import gc
import os

from circuit_breaker_rep.config import CircuitBreakerConfig
from circuit_breaker_rep.model import CircuitBreakerModel
from circuit_breaker_rep.dataset import CircuitBreakerDataset

logger = logging.getLogger(__name__)


class CircuitBreakerTrainer:
    """
    Trainer for circuit breaker models.

    Implements dual-loss training:
    - Retain loss: Preserve safe content representations (L2 distance)
    - CB loss: Alter harmful content representations (orthogonalization)
    """

    def __init__(
        self,
        model: CircuitBreakerModel,
        retain_dataset: CircuitBreakerDataset,
        cb_dataset: CircuitBreakerDataset,
        config: CircuitBreakerConfig,
    ):
        """
        Initialize trainer.

        Args:
            model: Circuit breaker model
            retain_dataset: Dataset of safe examples
            cb_dataset: Dataset of harmful examples
            config: Training configuration
        """
        self.model = model
        self.retain_dataset = retain_dataset
        self.cb_dataset = cb_dataset
        self.config = config

        # Create dataloaders
        self.retain_loader = DataLoader(
            retain_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues
        )

        self.cb_loader = DataLoader(
            cb_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
        )

        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            model.model.parameters(),
            lr=config.learning_rate,
        )

        # Learning rate scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=config.max_steps,
        )

        # Training state
        self.global_step = 0
        self.best_loss = float("inf")

        # Initialize W&B if enabled
        if config.use_wandb:
            try:
                import wandb
                wandb.init(
                    project=config.wandb_project,
                    config=vars(config),
                )
                self.wandb = wandb
            except ImportError:
                logger.warning("wandb not installed, skipping W&B logging")
                self.wandb = None
        else:
            self.wandb = None

    def compute_retain_loss(
        self,
        original_hidden: torch.Tensor,
        adapted_hidden: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute retain loss (L2 distance between original and adapted representations).

        Args:
            original_hidden: Hidden states from original model [num_layers, batch, seq_len, hidden_size]
            adapted_hidden: Hidden states from adapted model [num_layers, batch, seq_len, hidden_size]
            attention_mask: Attention mask [batch, seq_len]

        Returns:
            Scalar loss
        """
        # Expand attention mask to match hidden state dimensions
        # [batch, seq_len] -> [num_layers, batch, seq_len, 1]
        num_layers = original_hidden.shape[0]
        mask = attention_mask.unsqueeze(0).unsqueeze(-1).expand_as(original_hidden)

        # Compute L2 distance
        diff = adapted_hidden - original_hidden
        l2_dist = torch.norm(diff, dim=-1, p=2)  # [num_layers, batch, seq_len]

        # Apply mask and average
        masked_dist = l2_dist * mask.squeeze(-1)
        loss = masked_dist.sum() / mask.sum()

        return loss

    def compute_cb_loss(
        self,
        original_hidden: torch.Tensor,
        adapted_hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        target_layers: Optional[list] = None,
    ) -> torch.Tensor:
        """
        Compute circuit breaker loss (maximize orthogonality via minimizing inner product).

        Args:
            original_hidden: Hidden states from original model [num_layers, batch, seq_len, hidden_size]
            adapted_hidden: Hidden states from adapted model [num_layers, batch, seq_len, hidden_size]
            attention_mask: Attention mask [batch, seq_len]
            target_layers: Specific layers to compute loss on (None = use config.target_layers)

        Returns:
            Scalar loss
        """
        target_layers = target_layers or self.config.target_layers

        # Select target layers
        orig_target = torch.stack([original_hidden[i] for i in target_layers])
        adapt_target = torch.stack([adapted_hidden[i] for i in target_layers])

        # Normalize vectors
        orig_norm = F.normalize(orig_target, dim=-1)
        adapt_norm = F.normalize(adapt_target, dim=-1)

        # Compute inner product (cosine similarity)
        inner_product = (orig_norm * adapt_norm).sum(dim=-1)  # [num_target_layers, batch, seq_len]

        # Apply ReLU (only penalize positive inner products)
        inner_product = torch.relu(inner_product)

        # Apply attention mask
        num_target_layers = len(target_layers)
        mask = attention_mask.unsqueeze(0).expand(num_target_layers, -1, -1)

        # Masked average
        masked_ip = inner_product * mask
        loss = masked_ip.sum() / mask.sum()

        return loss

    def train_step(
        self,
        retain_batch: Dict[str, torch.Tensor],
        cb_batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Perform one training step.

        Args:
            retain_batch: Batch of retain examples
            cb_batch: Batch of CB examples

        Returns:
            Dictionary of losses
        """
        device = self.model.device

        # Move to device
        retain_batch = {k: v.to(device) for k, v in retain_batch.items()}
        cb_batch = {k: v.to(device) for k, v in cb_batch.items()}

        # Compute coefficient scheduling
        if self.config.use_dynamic_scheduling:
            # Progress: 0 to 1 over max_steps
            progress = self.global_step / self.config.max_steps
            # Note: GraySwanAI uses 2*max_steps for scheduling, adjust as needed
            retain_coeff = self.config.lorra_alpha * progress
            cb_coeff = self.config.lorra_alpha * (1 - progress)
        else:
            # Static coefficients
            retain_coeff = self.config.lorra_alpha
            cb_coeff = self.config.lorra_alpha

        # === Forward Passes ===

        # 1. Retain data - original representations
        orig_retain_hidden, retain_mask = self.model.get_hidden_states(
            retain_batch["input_ids"],
            retain_batch["attention_mask"],
            use_adapter=False,
        )
        orig_retain_hidden = torch.stack(orig_retain_hidden).detach()

        # 2. Retain data - adapted representations
        adapt_retain_hidden, _ = self.model.get_hidden_states(
            retain_batch["input_ids"],
            retain_batch["attention_mask"],
            use_adapter=True,
        )
        adapt_retain_hidden = torch.stack(adapt_retain_hidden)

        # Clean up
        del _
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 3. CB data - original representations
        orig_cb_hidden, cb_mask = self.model.get_hidden_states(
            cb_batch["input_ids"],
            cb_batch["attention_mask"],
            use_adapter=False,
        )
        orig_cb_hidden = torch.stack(orig_cb_hidden).detach()

        # 4. CB data - adapted representations
        adapt_cb_hidden, _ = self.model.get_hidden_states(
            cb_batch["input_ids"],
            cb_batch["attention_mask"],
            use_adapter=True,
        )
        adapt_cb_hidden = torch.stack(adapt_cb_hidden)

        # Clean up
        del _
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # === Compute Losses ===

        # Retain loss: Keep safe representations similar
        retain_loss = self.compute_retain_loss(
            orig_retain_hidden,
            adapt_retain_hidden,
            retain_mask,
        )

        # CB loss: Make harmful representations orthogonal
        cb_loss = self.compute_cb_loss(
            orig_cb_hidden,
            adapt_cb_hidden,
            cb_mask,
        )

        # Combined loss
        total_loss = retain_coeff * retain_loss + cb_coeff * cb_loss

        # === Backward Pass ===
        self.optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.model.parameters(), 1.0)

        self.optimizer.step()
        self.scheduler.step()

        # Clean up
        del orig_retain_hidden, adapt_retain_hidden, orig_cb_hidden, adapt_cb_hidden
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "total_loss": total_loss.item(),
            "retain_loss": retain_loss.item(),
            "cb_loss": cb_loss.item(),
            "retain_coeff": retain_coeff,
            "cb_coeff": cb_coeff,
            "lr": self.scheduler.get_last_lr()[0],
        }

    def train(self):
        """Run full training loop"""
        logger.info("Starting training...")
        logger.info(f"Total steps: {self.config.max_steps}")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info(f"Target layers: {self.config.target_layers}")

        self.model.model.train()

        # Create iterators
        retain_iter = iter(self.retain_loader)
        cb_iter = iter(self.cb_loader)

        pbar = tqdm(total=self.config.max_steps, desc="Training")

        while self.global_step < self.config.max_steps:
            try:
                # Get batches
                try:
                    retain_batch = next(retain_iter)
                except StopIteration:
                    retain_iter = iter(self.retain_loader)
                    retain_batch = next(retain_iter)

                try:
                    cb_batch = next(cb_iter)
                except StopIteration:
                    cb_iter = iter(self.cb_loader)
                    cb_batch = next(cb_iter)

                # Train step
                losses = self.train_step(retain_batch, cb_batch)

                self.global_step += 1
                pbar.update(1)

                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    log_str = f"Step {self.global_step}: "
                    log_str += f"loss={losses['total_loss']:.4f}, "
                    log_str += f"retain={losses['retain_loss']:.4f}, "
                    log_str += f"cb={losses['cb_loss']:.4f}"
                    logger.info(log_str)

                    if self.wandb:
                        self.wandb.log(losses, step=self.global_step)

                # Checkpointing
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint()

            except Exception as e:
                logger.error(f"Error at step {self.global_step}: {e}")
                raise

        pbar.close()

        # Final save
        logger.info("Training complete! Saving final model...")
        self.save_final_model()

    def save_checkpoint(self):
        """Save training checkpoint"""
        output_dir = os.path.join(
            self.config.output_dir,
            f"checkpoint-{self.global_step}",
        )
        os.makedirs(output_dir, exist_ok=True)

        # Save model (adapter only for checkpoints)
        self.model.save_model(output_dir, merge_adapter=False)

        logger.info(f"Checkpoint saved to {output_dir}")

    def save_final_model(self):
        """Save final merged model"""
        output_dir = os.path.join(self.config.output_dir, "final_model")
        os.makedirs(output_dir, exist_ok=True)

        # Save with merged adapter
        self.model.save_model(output_dir, merge_adapter=True)

        logger.info(f"Final model saved to {output_dir}")

        if self.wandb:
            self.wandb.finish()
