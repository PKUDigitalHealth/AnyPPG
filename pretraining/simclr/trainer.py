import math 
import torch
import logging
import wandb
from pathlib import Path
from torch import nn, optim
from typing import Optional, Dict, Any
from itertools import islice
from collections import deque
from accelerate import Accelerator
from torch.optim import lr_scheduler 
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from accelerate.utils import DistributedDataParallelKwargs


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        epochs: int,
        logger: logging.Logger,
        store_name: str,
        save_ckpt_dir: str,
        save_result_dir: str,
        train_loader: Optional[DataLoader] = None,
        valid_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        log_interval: int = 10,
        save_interval: int = 1000,
        save_ckpt_prefix: str = 'checkpoint',
        scheduler: Optional[lr_scheduler.LRScheduler] = None,
        resume_from_ckpt: bool = False,
        gradient_accumulation_steps: int = 1,
        clip_grad_norm: bool = False,
        max_grad_norm: float = 1,
        seed: int = 42,
        ckpt_path: Optional[str] = None,
        wandb_project: str = "ppg-ecg-alignment",
        wandb_entity: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        config: Optional[Dict] = None,
        mixed_precision: str = "no", # "no", "fp16", "bf16"
    ):
        self.save_ckpt_dir = Path(save_ckpt_dir) / store_name
        self.save_result_dir = Path(save_result_dir) / store_name
        
        self.accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            mixed_precision=mixed_precision,
            kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)]
        )

        self.clip_grad_norm = clip_grad_norm
        self.max_grad_norm = max_grad_norm
        self.seed = seed
        set_seed(seed)
        
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.logger = logger
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.save_ckpt_prefix = save_ckpt_prefix
        
        self.start_epoch = 0
        self.global_step = 0
        self.epochs = epochs

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        self.recent_batch_losses = deque(maxlen=10)
        self.smooth_loss_history = deque(maxlen=5)

        if ckpt_path:
            self.load_checkpoint(ckpt_path, resume_from_ckpt=resume_from_ckpt)

        self.model, self.optimizer, self.train_loader, self.valid_loader, self.test_loader = \
            self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.valid_loader, self.test_loader)

        if self.scheduler:
            self.scheduler = self.accelerator.prepare(self.scheduler)
        
        if self.accelerator.is_main_process:
            self.save_ckpt_dir.mkdir(parents=True, exist_ok=True)
            self.save_result_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize WandB
            wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                name=wandb_run_name if wandb_run_name else store_name,
                config=config,
                resume="allow",
                id=wandb_run_name if wandb_run_name else None # Use run name as ID for easier resuming if consistent
            )
        
        self.accelerator.wait_for_everyone()


    def _compute_grad_norm(self, parameters, norm_type=2.0):
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        if len(parameters) == 0:
            return 0.0
        device = parameters[0].grad.device
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
        return total_norm.item()

    def train(self):
        if self.train_loader is None:
            return
        self.len_train_loader = len(self.train_loader)
        if self.accelerator.is_main_process:
            self.logger.info("Start training...")

        rollback_ckpt_path = self.save_ckpt_dir / "rollback_checkpoint.pth"

        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch
            self.model.train()
            train_loss = 0.0
            train_loader = self.train_loader

            # Support for resuming within an epoch
            step_offset = self.epoch * self.len_train_loader
            skip_batches = self.global_step - step_offset
            if skip_batches > 0:
                 # Note: This is an approximation. Ideally DataLoader should support state saving.
                 # Using islice for skipping batches in the iterator.
                train_loader = islice(self.train_loader, skip_batches, None)
            
            for batch_idx, (_, ppg_view1, ppg_view2) in enumerate(train_loader, start=skip_batches): 
                # Ensure we don't double count if global_step was advanced
                # (Logic handled by skip_batches above, but good to be safe)
                
                with self.accelerator.accumulate(self.model):
                    # Model forward
                    # outputs is a dict: {'loss': ..., 'z1': ..., 'z2': ...}
                    outputs = self.model(ppg_view1, ppg_view2)
                    loss = outputs['loss']

                    # Robustness check
                    if not math.isfinite(loss.item()):
                        self.logger.warning(f"[Rollback] Loss became {loss.item()} at step {self.global_step}, restoring...")
                        self._rollback(rollback_ckpt_path)
                        continue

                    self.accelerator.backward(loss)

                    # Calculate separate grad norms before clipping
                    unwrapped_model = self.accelerator.unwrap_model(self.model)
                    
                    ppg_grad_norm = self._compute_grad_norm(unwrapped_model.ppg_encoder.parameters())
                    # ecg_grad_norm = self._compute_grad_norm(unwrapped_model.ecg_encoder.parameters())
                    # ptt_grad_norm = self._compute_grad_norm(unwrapped_model.ptt_predictor.parameters())

                    total_grad_norm = 0.0
                    if self.clip_grad_norm:
                        # clip_grad_norm_ returns the total norm before clipping
                        total_grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        if isinstance(total_grad_norm, torch.Tensor):
                            total_grad_norm = total_grad_norm.item()
                    else:
                         total_grad_norm = self._compute_grad_norm(self.model.parameters())

                    self.optimizer.step()
                    if self.scheduler:
                        self.scheduler.step()
                    self.optimizer.zero_grad()

                    loss_val = loss.item()
                    train_loss += loss_val
                    self.global_step += 1

                    # --- Robustness Monitoring ---
                    self.recent_batch_losses.append(loss_val)
                    if len(self.recent_batch_losses) == 20:
                        smooth_loss = sum(self.recent_batch_losses) / len(self.recent_batch_losses)
                        self.smooth_loss_history.append(smooth_loss)

                        if len(self.smooth_loss_history) == 10:
                            delta = self.smooth_loss_history[-1] - self.smooth_loss_history[0]
                            # Heuristic: if loss increases significantly, rollback
                            if delta > 0.5: # Adjusted threshold
                                self.logger.warning(f"[Rollback] Smooth loss increased by {delta:.4f}, rolling back checkpoint...")
                                self._rollback(rollback_ckpt_path)
                                continue

                    # --- Logging ---
                    if self.global_step % self.log_interval == 0 and self.accelerator.is_main_process:
                        current_lr = self.optimizer.param_groups[0]['lr']
                        
                        log_dict = {
                            "train/total_loss": loss_val,
                            # "train/loss_global": outputs.get('loss_global', 0).item() if isinstance(outputs.get('loss_global'), torch.Tensor) else 0,
                            # "train/loss_local": outputs.get('loss_local', 0).item() if isinstance(outputs.get('loss_local'), torch.Tensor) else 0,
                            "train/temperature": outputs.get('temperature', 0).item() if isinstance(outputs.get('temperature'), torch.Tensor) else outputs.get('temperature', 0),
                            "train/grad_norm_total": total_grad_norm,
                            "train/grad_norm_ppg": ppg_grad_norm,
                            # "train/grad_norm_ecg": ecg_grad_norm,
                            # "train/grad_norm_ptt": ptt_grad_norm,
                            "train/lr": current_lr,
                            "train/epoch": epoch + (batch_idx / self.len_train_loader),
                            "train/global_step": self.global_step
                        }
                        
                        wandb.log(log_dict)
                        
                        self.logger.info(
                            f"Epoch: [{epoch+1}]: [{batch_idx}/{self.len_train_loader}]: "
                            f"Loss: {loss_val:.4f}, "
                            # f"Global: {log_dict['train/loss_global']:.4f}, "
                            # f"Local: {log_dict['train/loss_local']:.4f}, "
                            f"Temp: {log_dict['train/temperature']:.4f}, "
                            f"LR: {current_lr:.6f}"
                        )

                    # --- Checkpointing ---
                    if self.accelerator.is_main_process and (self.global_step + 1) % self.save_interval == 0:
                        self.save_checkpoint(self.global_step)
                        
                        # Create rollback checkpoint
                        rollback_ckpt_path = self.save_ckpt_dir / "rollback_checkpoint.pth"
                        latest_ckpt_path = self.save_ckpt_dir / f"{self.save_ckpt_prefix}_step_{self.global_step}.pth"
                        if latest_ckpt_path.exists():
                            import shutil
                            shutil.copy(latest_ckpt_path, rollback_ckpt_path)

            # End of epoch
            epoch_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0
            val_loss = self.validate()

            if self.accelerator.is_main_process:
                self.logger.info(
                    f"Epoch {epoch+1}/{self.epochs} - "
                    f"Train Loss: {epoch_train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}, "
                )
                wandb.log({
                    "epoch/train_loss": epoch_train_loss,
                    "epoch/val_loss": val_loss,
                    "epoch": epoch + 1,
                })

            self.accelerator.wait_for_everyone()


    def validate(self):
        if self.valid_loader is None:
            return 0.0
        self.model.eval()
        val_loss = 0.0
        val_steps = 0

        with torch.inference_mode():
            for batch_idx, (_, ppg_view1, ppg_view2) in enumerate(self.valid_loader): 
                outputs = self.model(ppg_view1, ppg_view2)
                loss = outputs['loss']
                val_loss += loss.item()
                val_steps += 1
                
                if batch_idx % 50 == 0 and self.accelerator.is_main_process:
                    self.logger.info(
                        f"Val: [{batch_idx}/{len(self.valid_loader)}]: loss: {loss.item():.4f}"
                    )
        
        # Average loss across batches
        avg_val_loss = val_loss / val_steps if val_steps > 0 else 0.0
        
        # Gather metrics across devices if distributed
        avg_val_loss_tensor = torch.tensor(avg_val_loss, device=self.accelerator.device)
        gathered_val_loss = self.accelerator.gather(avg_val_loss_tensor)
        final_val_loss = gathered_val_loss.mean().item()
        
        return final_val_loss

    
    def save_checkpoint(self, step: int, is_best: bool = False):
        checkpoint_path = self.save_ckpt_dir / (
            f"best_{self.save_ckpt_prefix}_step_{step}.pth" if is_best else f"{self.save_ckpt_prefix}_step_{step}.pth"
        )
        
        if self.accelerator.is_main_process:
            checkpoint = {
                "epoch": self.epoch,
                "global_step": self.global_step,
                "model_state_dict": self.accelerator.get_state_dict(self.model),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None
            }
            
            self.accelerator.save(checkpoint, checkpoint_path)
            self.logger.info(f"Checkpoint saved to {checkpoint_path}")


    def load_checkpoint(self, checkpoint_path: str, resume_from_ckpt: bool):
        checkpoint = torch.load(checkpoint_path, map_location="cpu") # weights_only=True safe for simple dicts
        
        model_state_dict = checkpoint["model_state_dict"]
        # Handle DataParallel/DistributedDataParallel keys
        model_state_dict = {
            k.replace("module.", ""): v for k, v in model_state_dict.items()
        }
        self.model.load_state_dict(model_state_dict)

        if resume_from_ckpt:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scheduler_state_dict" in checkpoint and self.scheduler is not None:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.start_epoch = checkpoint["epoch"]
            self.global_step = checkpoint["global_step"]
        else:
            self.start_epoch = 0
            self.global_step = 0
        
        self.logger.info(f"Loaded checkpoint from {checkpoint_path}.")
        if resume_from_ckpt:
            self.logger.info(f"Resuming from epoch {self.start_epoch}")


    def _rollback(self, rollback_ckpt_path: Path):
        self.recent_batch_losses.clear()
        self.smooth_loss_history.clear()
        self.seed += 10
        set_seed(self.seed)
        if not rollback_ckpt_path.exists():
            self.logger.error(f"[Rollback] Checkpoint not found: {rollback_ckpt_path}")
            return

        try:
            checkpoint = torch.load(rollback_ckpt_path, map_location="cpu")
            # We need to be careful with state dict keys when loading back into wrapped model
            # Accelerator wrapped model usually has 'module.' prefix if DDP, but load_state_dict might handle it?
            # Safer to unwrap or handle keys.
            model_state_dict = checkpoint["model_state_dict"]
            
            # Helper to strip prefix if needed, or rely on accelerator.unwrap_model
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.load_state_dict(model_state_dict)

            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if self.scheduler and checkpoint.get("scheduler_state_dict") is not None:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

            self.start_epoch = checkpoint["epoch"]
            self.global_step = checkpoint["global_step"]

            # Re-prepare might not be needed if objects are same, but safe to reset train mode
            self.model.train()
            self.logger.info(f"[Rollback] Successfully rolled back to {rollback_ckpt_path}, resumed from epoch {self.start_epoch}, step {self.global_step}")

        except Exception as e:
            self.logger.error(f"[Rollback] Failed to rollback from {rollback_ckpt_path}: {e}")
