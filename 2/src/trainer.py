"""The Trainer class, to easily train a 🤗 Transformers from scratch or finetune it on a new task."""

import collections
import inspect
import math
import os
import re
import shutil
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from packaging import version
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.optim.lr_scheduler import LambdaLR
import math
import time

import transformers
from transformers.file_utils import is_datasets_available, is_in_notebook, is_torch_tpu_available
from transformers.integrations import (
    is_comet_available,
    is_optuna_available,
    is_ray_available,
    is_tensorboard_available,
    is_wandb_available,
)
from transformers.optimization import AdamW, get_linear_schedule_with_warmup, get_scheduler

from transformers.trainer_callback import (
    DefaultFlowCallback,
    ProgressCallback,
)
from transformers.trainer_utils import (
    default_compute_objective,
)
from transformers.training_args import TrainingArguments
from transformers.utils import logging
from transformers.trainer_utils import TrainOutput

from tqdm import tqdm, trange
from torch.optim import SGD
import torch.nn.functional as F

from src.linearhead_trainer import LinearHeadTrainer
from transformers.trainer_callback import TrainerState

import copy

from opacus.accountants.utils import get_noise_multiplier


_use_native_amp = False
_use_apex = False

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
    from transformers.utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

# Check if Pytorch version >= 1.6 to switch between Native AMP and Apex
if version.parse(torch.__version__) < version.parse("1.6"):
    from transformers.file_utils import is_apex_available

    if is_apex_available():
        from apex import amp
    _use_apex = True
else:
    _use_native_amp = True
    from torch.cuda.amp import autocast

if version.parse(torch.__version__) < version.parse("1.2"):
    _use_ddp_no_sync = False
else:
    _use_ddp_no_sync = True

if is_datasets_available():
    import datasets

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

if is_tensorboard_available():
    from transformers.integrations import TensorBoardCallback

    DEFAULT_CALLBACKS.append(TensorBoardCallback)


if is_wandb_available():
    from transformers.integrations import WandbCallback

    DEFAULT_CALLBACKS.append(WandbCallback)

if is_comet_available():
    from transformers.integrations import CometCallback

    DEFAULT_CALLBACKS.append(CometCallback)

if is_optuna_available():
    import optuna

if is_ray_available():
    from ray import tune

logger = logging.get_logger(__name__)
logger.setLevel(logging.INFO)

########## The above part is copied from Transformers' trainer (3.4.0) ##########

def default_dev_objective(metrics):
    """
    Objective used for picking the best model on development sets
    """
    if "eval_mnli/acc" in metrics:
        return metrics["eval_mnli/acc"]
    elif "eval_mnli-mm/acc" in metrics:
        return metrics["eval_mnli-mm/acc"]
    elif "eval_f1" in metrics:
        return metrics["eval_f1"]
    elif "eval_mcc" in metrics:
        return metrics["eval_mcc"]
    elif "eval_pearson" in metrics:
        return metrics["eval_pearson"]
    elif "eval_acc" in metrics:
        return metrics["eval_acc"]

    raise Exception("No metric founded for {}".format(metrics))


def dpzero_clip(loss_diff, C=1.):
    tmp = torch.min(torch.ones_like(loss_diff), torch.div(C * torch.ones_like(loss_diff), torch.abs(loss_diff)))
    return torch.mul(tmp, loss_diff)

class Trainer(LinearHeadTrainer):
    """
    Adding some functions based on Transformers' Trainer class.
    """

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Based on Transformers' default one, we add fixing layer option where the bottom n layers' parameters
        are fixed and only the top layers are further fine-tuned.
        """
        if self.args.hf_inference_model:
            return

        if self.optimizer is None:
            params = {}
            for n, p in self.model.named_parameters():
                if self.args.fix_layers > 0:
                    if 'encoder.layer' in n:
                        try:
                            layer_num = int(n[n.find('encoder.layer') + 14:].split('.')[0])
                        except:
                            print(n)
                            raise Exception("")
                        if layer_num >= self.args.fix_layers:
                            print('yes', n)
                            params[n] = p
                        else:
                            print('no ', n)
                    elif 'embeddings' in n:
                        print('no ', n)
                    else:
                        print('yes', n)
                        params[n] = p
                else:
                    params[n] = p
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in params.items() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in params.items() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            if self.args.optimizer == 'adam':
                self.optimizer = AdamW(
                    optimizer_grouped_parameters,
                    lr=self.args.learning_rate,
                    betas=(self.args.adam_beta1, self.args.adam_beta2),
                    eps=self.args.adam_epsilon,
                )
            elif self.args.optimizer == 'sgd':
                self.optimizer = SGD(
                    optimizer_grouped_parameters,
                    lr=self.args.learning_rate
                )
            else:
                raise NotImplementedError
        if self.lr_scheduler is None:
            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                optimizer=self.optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
            )

    def should_optim(self, name, param):
        return (not self.args.layer_wise_optim or f".{self.state.global_step % self.model.config.num_hidden_layers}." in name) and param.requires_grad


    def zo_forward(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        model.eval()
        inputs = self._prepare_inputs(inputs)
        if self.args.optimize_acc:
            loss, logits = model(**inputs)
            preds = F.softmax(logits, dim=-1)
            acc = torch.sum(torch.argmax(preds, 1) == inputs['labels']) / len(preds)
            loss = -acc
        else:
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
        self.state.zo_forward_step += 1
        return loss.detach()


    def efficient_perturb_parameters(self, model: nn.Module, random_seed: int, scaling_factor=1):
        torch.manual_seed(random_seed)
        for name, param in self.named_parameters_to_optim:
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            param.data = param.data + scaling_factor * z * self.args.zero_order_eps
        return model

    
    def perturb_parameters(self, model: nn.Module, random_vector=None, scaling_factor=1):
        if random_vector is None:
            random_vector = {}

        for name, param in self.named_parameters_to_optim:
            if name in random_vector:
                z = random_vector[name]
            else:
                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                random_vector[name] = z
            param.data = param.data + scaling_factor * z * self.args.zero_order_eps

        return model, random_vector


    def get_num_samples(self):
        if self.args.zero_order_sample_scheduler is None:
            noise_sample_time = 1 
        elif self.args.zero_order_sample_scheduler == "linear":
            noise_sample_time = max(1, int(self.state.global_step / self.args.max_steps * self.args.zero_order_sample))
        elif self.args.zero_order_sample_scheduler == "constant":
            noise_sample_time = int(self.args.zero_order_sample)
        else:
            raise NotImplementedError

        return noise_sample_time

    # --- ADDED FOR L2 TRACKING: Calculation Helper Method ---
    def get_l2_weight_distance(self, initial_weights):
        """Calculates the L2 distance between current weights and initial weights (w_t - w_0)"""
        l2_sum = 0.0
        for n, p in self.model.named_parameters():
            if n in initial_weights:
                # Move current weights to CPU for calculation to avoid GPU OOM
                l2_sum += torch.sum((p.detach().cpu() - initial_weights[n]) ** 2).item()
        return math.sqrt(l2_sum)
    # --------------------------------------------------------

    def train(self, model_path=None, dev_objective=None):
        """
        Main training entry point.

        The training logic is directly borrowed from transformers.Trainer (version 3.0.2).
        Add early stopping.
        """
        if self.args.from_linearhead and model_path is None:
            super().train(model_path, dev_objective) # Train output layer using LinearHeadTrainer

        self.best_dir = None
        self.objective = -float("inf")
        self.dev_objective = dev_objective if dev_objective is not None else default_dev_objective

        # Data loading.
        train_dataloader = self.get_train_dataloader()
        num_update_steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps
        if num_update_steps_per_epoch == 0:
            num_update_steps_per_epoch = 1
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            num_train_epochs = self.args.max_steps // num_update_steps_per_epoch + int(
                self.args.max_steps % num_update_steps_per_epoch > 0
            )
        else:
            t_total = int(len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs)
            num_train_epochs = self.args.num_train_epochs

        self.create_optimizer_and_scheduler(num_training_steps=t_total)
        optimizer = self.optimizer
        scheduler = self.lr_scheduler

        # Check if saved optimizer or scheduler states exist
        if (
            model_path is not None
            and os.path.isfile(os.path.join(model_path, "optimizer.pt"))
            and os.path.isfile(os.path.join(model_path, "scheduler.pt"))
        ):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(
                torch.load(os.path.join(model_path, "optimizer.pt"), map_location=self.args.device)
            )
            scheduler.load_state_dict(torch.load(os.path.join(model_path, "scheduler.pt")))

        model = self.model

        if self.args.fp16 and _use_apex:
            if not transformers.is_apex_available():
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=self.args.fp16_opt_level)

        # Multi-gpu training (should be after apex fp16 initialization)
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Distributed training (should be after apex fp16 initialization)
        if self.args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=True,
            )

        # Train
        if transformers.is_torch_tpu_available():
            total_train_batch_size = self.args.train_batch_size * xm.xrt_world_size()
        else:
            total_train_batch_size = (
                self.args.train_batch_size
                * self.args.gradient_accumulation_steps
                * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1)
            )
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", self.num_examples(train_dataloader))
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info("  Instantaneous batch size per device = %d", self.args.per_device_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        self.state = TrainerState()
        self.state.global_step = 0

        # --- ADDITION: Setup Gradient Histogram Tracking ---
        self.pre_clip_history = []
        self.post_clip_history = []
        self.clip_count_accumulator = 0  # Resets every log
        self.clip_step_accumulator = 0   # Resets every log
        self.global_clip_count = 0       # NEVER resets (for final summary)
        self.global_clip_steps = 0       # NEVER resets (for final summary)
        # -------------------------------------------------

        start_time = time.time()
        self.state.zo_forward_step = 0
        self.epoch = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        flip = True

        # =========================================================================
        # CRITICAL FIX: Capture the original config passed via bash script arguments
        # =========================================================================
        initial_lr = self.args.learning_rate
        initial_clip = self.args.dpzero_clip_threshold
        # =========================================================================

        if self.args.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        # Check if continuing training from a checkpoint
        if model_path is not None:
            # set global_step to global_step of last saved checkpoint from model path
            try:
                self.state.global_step = int(model_path.split("-")[-1].split("/")[0])
                epochs_trained = self.state.global_step // (len(train_dataloader) // self.args.gradient_accumulation_steps)
                steps_trained_in_current_epoch = self.state.global_step % (
                    len(train_dataloader) // self.args.gradient_accumulation_steps
                )

                logger.info("  Continuing training from checkpoint, will skip to saved global_step")
                logger.info("  Continuing training from epoch %d", epochs_trained)
                logger.info("  Continuing training from global step %d", self.state.global_step)
                logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
            except ValueError:
                self.state.global_step = 0
                logger.info("  Starting fine-tuning.")

        tr_loss = torch.tensor(0.0).to(self.args.device)
        logging_loss_scalar = 0.0
        model.zero_grad()
        metrics = None

        # --- ADDED FOR L2 TRACKING: Initial Weights Capture ---
        logger.info("Capturing initial weights (w_0) for L2 distance tracking...")
        initial_weights = {n: p.clone().detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
        # ------------------------------------------------------

        # 1. Safely extract the Dataset Name ("sst-2", "snli", etc.)
        if hasattr(self, 'model') and hasattr(self.model, 'data_args') and hasattr(self.model.data_args, 'task_name'):
            dataset_name = self.model.data_args.task_name.lower()
        elif hasattr(self.args, 'task_name'):
            dataset_name = self.args.task_name.lower()
        else:
            dataset_name = "unknown_dataset"
        

        for epoch in range(epochs_trained, int(num_train_epochs)):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)

            if transformers.is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [self.args.device]).per_device_loader(
                    self.args.device
                )
                epoch_iterator = tqdm(parallel_loader, desc="Iteration", disable=not self.is_local_process_zero())
            else:
                epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=True)

            # Reset the past mems state at the beginning of each epoch if necessary.
            if self.args.past_index >= 0:
                self._past = None

            # # =========================================================================
            # # CRITICAL SWITCH LOGIC: Updates Flip, Clip Threshold, and Learning Rate
            # # =========================================================================
            # if epoch == 0:
            #     # State B: 1e-6, No Clipping
            #     current_lr = 1e-6
            #     current_clip = 0.0
            #     logger.info(f"\n[Epoch {epoch}] 🔴 Flipped TO: LR = {current_lr}, Clip = ∞")
            #     # Update Trainer Args for DPZero and Non-Trainer ZO Optimization
            #     self.args.dpzero_clip_threshold = current_clip
            #     self.args.learning_rate = current_lr
                
            #     # Update Optimizer directly for Trainer-Based Optimization
            #     if optimizer is not None:
            #         for param_group in optimizer.param_groups:
            #             param_group['lr'] = current_lr

            # elif (epoch == self.args.flip_val + 1) and flip:    
            #     flip = not flip
            #     # State A: Dynamic Config from bash script
            #     current_lr = initial_lr
            #     current_clip = initial_clip
            #     logger.info(f"\n[Epoch {epoch}] 🟢 Flipped BACK: LR = {current_lr}, Clip = {current_clip}")

            #     # Update Trainer Args for DPZero and Non-Trainer ZO Optimization
            #     self.args.dpzero_clip_threshold = current_clip
            #     self.args.learning_rate = current_lr
                
            #     # Update Optimizer directly for Trainer-Based Optimization
            #     if optimizer is not None:
            #         for param_group in optimizer.param_groups:
            #             param_group['lr'] = current_lr
            # =========================================================================

            # =========================================================================
            # CRITICAL SWITCH LOGIC: Updates Flip, Clip Threshold, and Learning Rate
            # =========================================================================
            if epoch > 0 and (epoch == self.args.flip_val + 1):
                flip = not flip
                if not flip:
                    # State B: 1e-6, No Clipping
                    current_lr = 1e-6
                    current_clip = 0.0
                    logger.info(f"\n[Epoch {epoch}] 🔴 Flipped TO: LR = {current_lr}, Clip = ∞")
                else:
                    # State A: Dynamic Config from bash script
                    current_lr = initial_lr
                    current_clip = initial_clip
                    logger.info(f"\n[Epoch {epoch}] 🟢 Flipped BACK: LR = {current_lr}, Clip = {current_clip}")

                # Update Trainer Args for DPZero and Non-Trainer ZO Optimization
                self.args.dpzero_clip_threshold = current_clip
                self.args.learning_rate = current_lr
                
                # Update Optimizer directly for Trainer-Based Optimization
                if optimizer is not None:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = current_lr
            # =========================================================================

            for step, inputs in enumerate(epoch_iterator):
                if self.args.sync_embedding_layers:
                    assert model.module.model_type == 'opt', 'did not implement embedding layer synchronization for non-OPT models'
                    model.module.model.decoder.embed_tokens.weight = model.module.lm_head.weight
                
                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue
                    
                if self.args.zero_order_optim:
                    # Get parameters that should be optimized (for layer-wise optimization and prefix-tuning)
                    self.named_parameters_to_optim = []
                    for name, param in model.named_parameters():
                        if self.should_optim(name, param):
                            self.named_parameters_to_optim.append((name, param))

                    # get number of zs to sample
                    num_zs = self.get_num_samples()
                    if num_zs > 1:
                        assert self.args.zero_order_use_trainer_optim, 'cannot sample multiple zs without storing intermediate gradient. use trainer.'

                    for _ in range(num_zs):
                        # prepare for sampling new zs
                        random_vector = None
                        if self.args.efficient_zero_order:
                            random_seed = np.random.randint(1000000000)

                        with torch.no_grad():
                            # first function evaluation
                            if self.args.efficient_zero_order:
                                model = self.efficient_perturb_parameters(model, random_seed)
                            else:
                                model, random_vector = self.perturb_parameters(model)
                            loss1 = self.zo_forward(model, inputs)

                            # second function evaluation
                            if self.args.efficient_zero_order:
                                model = self.efficient_perturb_parameters(model, random_seed, scaling_factor=-2)
                            else:
                                model, random_vector = self.perturb_parameters(model, random_vector, scaling_factor=-2)
                            loss2 = self.zo_forward(model, inputs)

                        projected_grad = (loss1 - loss2) / (2 * self.args.zero_order_eps)

                        # DPZero gradient clipping and noise injection

                        is_track_step = (self.state.global_step % 100 == 0)
                        if is_track_step:
                            self.pre_clip_history.append(projected_grad.detach().cpu().numpy().flatten())
                        if self.args.dpzero:
                            if self.args.dpzero_clip_threshold > 0:
                                # --- ADDITION: Count how many elements exceed the threshold ---
                                num_clipped = (torch.abs(projected_grad) > self.args.dpzero_clip_threshold).sum().item()
                                
                                # Temporary trackers (reset for interval logging)
                                self.clip_count_accumulator += num_clipped
                                self.clip_step_accumulator += 1
                                
                                # Global trackers (never reset)
                                self.global_clip_count += num_clipped
                                self.global_clip_steps += 1

                                projected_grad = dpzero_clip(projected_grad, self.args.dpzero_clip_threshold)
                            if is_track_step:
                                self.post_clip_history.append(projected_grad.detach().cpu().numpy().flatten())
                            projected_grad = projected_grad.mean()

                        # scale grad according to accumulation
                        if self.args.gradient_accumulation_steps > 1:
                            assert self.args.zero_order_use_trainer_optim, 'grad accumulation not implemented for non-trainer ZO yet'
                            projected_grad = projected_grad / self.args.gradient_accumulation_steps

                        # scale grad according to number of zs sampled
                        if not self.args.scale_lr_with_samples:
                            projected_grad = projected_grad / float(num_zs)

                        # store gradient in parameter buffer if using trainer
                        # o/w, the loop will exit after one round and the update will be applied directly (see below)
                        if self.args.zero_order_use_trainer_optim:
                            if self.args.efficient_zero_order:
                                torch.manual_seed(random_seed)

                            for name, param in self.named_parameters_to_optim:
                                # recover noise used in perturbations
                                if self.args.efficient_zero_order:
                                    z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                                else:
                                    z = random_vector[name]

                                if param.grad is None:
                                    param.grad = projected_grad * z
                                else:
                                    param.grad += projected_grad * z

                        # reset model back to its parameters at start of step
                        if self.args.efficient_zero_order:
                            model = self.efficient_perturb_parameters(model, random_seed)
                        else:
                            model, random_vector = self.perturb_parameters(model, random_vector)

                    # apply gradient updates
                    # if using trainer, follow trainer logic to clip grad and check if parameters should be updated
                    if self.args.zero_order_use_trainer_optim:
                        if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                            # last step in epoch but step is always smaller than gradient_accumulation_steps
                            len(epoch_iterator) <= self.args.gradient_accumulation_steps
                            and (step + 1) == len(epoch_iterator)
                        ):
                            # Gradient norm clipping
                            if self.args.zero_order_clip_grad:
                                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

                            # Update the parameters and step scheduler
                            optimizer.step()
                            scheduler.step()
                        
                            # logging
                            if (self.args.logging_steps > 0 and self.state.global_step % self.args.logging_steps == 0) or (
                                self.state.global_step == 1 and self.args.logging_first_step
                            ):
                                logs = {}
                                logs["loss"] = loss1.item()
                                if not self.args.zero_order_clip_grad:
                                    norm = 0.0
                                    for _, p in model.named_parameters():
                                        if p.grad is not None:
                                            norm += torch.sum(p.grad ** 2)
                                    norm = torch.sqrt(norm)
                                logs["grad_norm"] = norm.item()
                                logs["learning_rate"] = (
                                    scheduler.get_last_lr()[0]
                                    if version.parse(torch.__version__) >= version.parse("1.4")
                                    else scheduler.get_lr()[0]
                                )
                                logs["num_zs"] = num_zs
                                logs["global_step"] = self.state.global_step
                                logs["zo_forward_step"] = self.state.zo_forward_step
                                logs["max_steps"] = self.args.max_steps
                                logs["max_zo_forward_steps"] = self.args.max_zo_forward_steps
                                logs["time"] = int(time.time() - start_time)
                                
                                # --- ADDED FOR L2 TRACKING: Log calculation for ZO Trainer Optim ---
                                logs["l2_weight_dist"] = self.get_l2_weight_distance(initial_weights)
                                # -------------------------------------------------------------------
                                # --- ADDITION: Log the average clipped numbers ---
                                if self.clip_step_accumulator > 0:
                                    logs["avg_clipped_per_iter"] = self.clip_count_accumulator / self.clip_step_accumulator
                                    # Reset accumulators for the next logging window
                                    self.clip_count_accumulator = 0
                                    self.clip_step_accumulator = 0
                                # -------------------------------------------------

                                logs["clipping"] = flip 
                                self.log(logs)
                                logger.info(str(logs))
                            
                            model.zero_grad()
                            self.state.global_step += 1
                            self.epoch = epoch + (step + 1) / len(epoch_iterator)
                    # if not using the trainer, the updates are resampled and directly applied to the parameters
                    else:
                        # Efficient mode 
                        # WARNING: no gradient accumulation when not storing the grad
                        assert self.args.gradient_accumulation_steps == 1, 'gradient accumulation is not supported for zero-order optimization'
                        assert self.args.zero_order_sample_scheduler is None
                        assert not self.args.zero_order_clip_grad, 'gradient clipping not implemented yet for non-trainer ZO'

                        if self.args.efficient_zero_order:
                            torch.manual_seed(random_seed)     
                        for name, param in self.named_parameters_to_optim:
                            if self.args.efficient_zero_order:
                                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                            else:
                                z = random_vector[name]
                            param.data = param.data - self.args.learning_rate * projected_grad * z 

                        if (self.args.logging_steps > 0 and self.state.global_step % self.args.logging_steps == 0) or (
                                self.state.global_step == 1 and self.args.logging_first_step
                            ):
                                logs = {}
                                if self.args.dpzero:
                                    logs["loss"] = loss1.mean().item()
                                else:
                                    logs["loss"] = loss1.item()
                                logs["learning_rate"] = self.args.learning_rate
                                logs["global_step"] = self.state.global_step
                                logs["zo_forward_step"] = self.state.zo_forward_step
                                logs["max_steps"] = self.args.max_steps
                                logs["max_zo_forward_steps"] = self.args.max_zo_forward_steps
                                logs["time"] = int(time.time() - start_time)
                                
                                # --- ADDED FOR L2 TRACKING: Log calculation for ZO Non-Trainer Optim ---
                                logs["l2_weight_dist"] = self.get_l2_weight_distance(initial_weights)
                                # -----------------------------------------------------------------------
                                
                                # --- ADDITION: Log the average clipped numbers ---
                                if self.clip_step_accumulator > 0:
                                    logs["avg_clipped_per_iter"] = self.clip_count_accumulator / self.clip_step_accumulator
                                    # Reset accumulators for the next logging window
                                    self.clip_count_accumulator = 0
                                    self.clip_step_accumulator = 0
                                # -------------------------------------------------

                                logs["clipping"] = flip 
                                
                                self.log(logs)
                                logger.info(str(logs))


                        self.state.global_step += 1
                        self.epoch = epoch + (step + 1) / len(epoch_iterator)
                    
                # standard, non-ZO optimization
                else:
                    tr_loss += self.training_step(model, inputs)

                    if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                        # last step in epoch but step is always smaller than gradient_accumulation_steps
                        len(epoch_iterator) <= self.args.gradient_accumulation_steps
                        and (step + 1) == len(epoch_iterator)
                    ):
                        if self.args.fp16 and _use_native_amp:
                            self.scaler.unscale_(optimizer)
                            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
                        elif self.args.fp16:
                            norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), self.args.max_grad_norm)
                        else:
                            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

                        if self.args.optimizer_variant == 'signgd':
                            for n,p in model.named_parameters():
                                if p.grad is not None:
                                    p.grad = torch.sign(p.grad)

                        if transformers.is_torch_tpu_available():
                            xm.optimizer_step(optimizer)
                        elif self.args.fp16 and _use_native_amp:
                            self.scaler.step(optimizer)
                            self.scaler.update()
                        else:
                            optimizer.step()

                        scheduler.step()
                        model.zero_grad()
                        self.state.global_step += 1
                        self.epoch = epoch + (step + 1) / len(epoch_iterator)

                        if (self.args.logging_steps > 0 and self.state.global_step % self.args.logging_steps == 0) or (
                            self.state.global_step == 1 and self.args.logging_first_step
                        ):
                            logs = {}
                            tr_loss_scalar = tr_loss.item()
                            logs["loss"] = (tr_loss_scalar - logging_loss_scalar) / self.args.logging_steps
                            logs["norm"] = norm.item()
                            # backward compatibility for pytorch schedulers
                            logs["learning_rate"] = (
                                scheduler.get_last_lr()[0]
                                if version.parse(torch.__version__) >= version.parse("1.4")
                                else scheduler.get_lr()[0]
                            )
                            logging_loss_scalar = tr_loss_scalar

                            # --- ADDED FOR L2 TRACKING: Log calculation for Standard FO Optim ---
                            logs["l2_weight_dist"] = self.get_l2_weight_distance(initial_weights)
                            # --------------------------------------------------------------------
                            
                            logs["clipping"] = flip 

                            self.log(logs)
                            logger.info(str(logs))

                if self.args.max_steps > 0 and self.state.global_step > self.args.max_steps or (self.args.max_zo_forward_steps > 0 and self.state.zo_forward_step > self.args.max_zo_forward_steps):
                    epoch_iterator.close()
                    break

                if self.args.evaluate_during_training and self.state.global_step % self.args.eval_steps == 0:
                    output = self.evaluate()
                    metrics = output.metrics
                    objective = self.dev_objective(metrics)
                    if objective > self.objective:
                        logger.info("Best dev result: {}".format(objective))
                        self.objective = objective

                        # Now we save this to (CPU) memory instead of disk <-- much faster
                        self.best_model_ckpt = {k: v.detach().cpu() for k, v in model.state_dict().items()}

            
            if (epoch % 20 == 0):
                # After the training loop of each epoch, evaluate and print accuracy
                output = self.evaluate()  # Evaluate the model after each epoch
                metrics = output.metrics
                eval_acc = metrics.get("accuracy", None)  # Assuming "accuracy" is in the returned metrics

                if eval_acc is not None:
                    print(f"Epoch {epoch + 1} - Accuracy: {eval_acc:.4f}")
            
            
            if self.args.max_steps > 0 and self.state.global_step > self.args.max_steps or (self.args.max_zo_forward_steps > 0 and self.state.zo_forward_step > self.args.max_zo_forward_steps):
                break
            if self.args.tpu_metrics_debug or self.args.debug:
                # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                xm.master_print(met.metrics_report())

        
        self.args.learning_rate = initial_lr 
        self.args.dpzero_clip_threshold = initial_clip
        
        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")


        # --- ADDITION: Save Histogram Data to disk ---
        if len(self.pre_clip_history) > 0:
            try:
                # 1. Create the folder
                os.makedirs("flipped_r", exist_ok=True)
                
                # 2. Dynamically name the file based on arguments
                lr_val = initial_lr
                clip_val = initial_clip
                flip_value = self.args.flip_val
                seed_val = str(getattr(self.args, 'seed', 'unknown'))
                file_path = os.path.join("flipped_r", f"{lr_val}_{clip_val}_histogram_seed_{seed_val}_{flip_value}.npz")
                
                # 3. Concatenate ALL points into one massive array and save
                np.savez(
                    file_path,
                    pre_clip=np.concatenate(self.pre_clip_history),
                    post_clip=np.concatenate(self.post_clip_history) if len(self.post_clip_history) > 0 else []
                )
                logger.info(f"✅ Saved gradient distributions to {file_path}")
            except Exception as e:
                logger.error(f"Failed to save histogram data: {e}")
        # ---------------------------------------------

        # --- ADDITION: Print Global Clipping Summary ---
        if self.global_clip_steps > 0:
            global_avg_clipped = self.global_clip_count / self.global_clip_steps
            # Calculate the total batch size vector length by looking at the first saved gradient
            total_elements = len(self.pre_clip_history[0]) if len(self.pre_clip_history) > 0 else "Unknown"
            
            logger.info("\n" + "="*50)
            logger.info("📉 GLOBAL CLIPPING STATISTICS")
            logger.info("="*50)
            logger.info(f"Total ZO Steps Executed: {self.global_clip_steps}")
            logger.info(f"Threshold Used: {self.args.dpzero_clip_threshold}")
            logger.info(f"Average elements clipped per step: {global_avg_clipped:.2f} (out of {total_elements})")
            logger.info("="*50 + "\n")
        # -----------------------------------------------


        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        return TrainOutput(self.state.global_step, tr_loss / self.state.global_step, metrics), self.objective


    """
    Difference compared to original implementation: return output instead of output.metrics (so there is also the logits)
    """
    def evaluate(self, eval_dataset: Optional[Dataset] = None, task = None) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are
        task-dependent (pass it to the init :obj:`compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (:obj:`Dataset`, `optional`):
                Pass a dataset if you wish to override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`,
                columns not accepted by the ``model.forward()`` method are automatically removed. It must implement
                the :obj:`__len__` method.

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions.
        """
        if eval_dataset is not None and not isinstance(eval_dataset, collections.abc.Sized):
            raise ValueError("eval_dataset must implement __len__")

        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        output = self.prediction_loop(eval_dataloader, description="Evaluation")

        self.log(output.metrics)
        logger.info(output.metrics)

        if self.args.tpu_metrics_debug or self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        # =========================================================
        # --- SAVE EVALUATION LOGITS TO NPZ ---
        # =========================================================
        if output.predictions is not None and output.label_ids is not None and task is not None:
            preds = np.argmax(output.predictions, axis=-1)
            labels = output.label_ids
            correct_mask = (preds == labels).astype(int) 
            
            npz_dir = os.path.join(os.getcwd(), "flipped_r")
            os.makedirs(npz_dir, exist_ok=True)

            # 1. Safely extract the Dataset Name ("sst-2", "snli", etc.)
            if hasattr(self, 'model') and hasattr(self.model, 'data_args') and hasattr(self.model.data_args, 'task_name'):
                dataset_name = self.model.data_args.task_name.lower()
            elif hasattr(self.args, 'task_name'):
                dataset_name = self.args.task_name.lower()
            else:
                dataset_name = "unknown_dataset"
            
            lr_str = str(self.args.learning_rate)
            seed_val = str(getattr(self.args, 'seed', 'unknown'))
            clip_val = str(getattr(self.args, 'dpzero_clip_threshold', 0)) if getattr(self.args, 'dpzero', False) else "0"
            flip_value = str(self.args.flip_val)
            
            eval_mask_path = os.path.join(npz_dir, f"{dataset_name}_{task}_logits_lr{lr_str}_clip{clip_val}_seed{seed_val}_flip{flip_value}.npz")
            
            np.savez(eval_mask_path, test_correct_mask=correct_mask, test_preds=preds, test_labels=labels, logits=output.predictions)
            logger.info(f"Saved evaluation logits to {eval_mask_path}")

            preds = np.argmax(output.predictions, axis=-1)
            labels = output.label_ids
            # Boolean array of 1s and 0s indicating correct/incorrect predictions
            correct_mask = (preds == labels).astype(int) 
            
            npz_dir = os.path.join(os.getcwd(), "flipped_r")
            os.makedirs(npz_dir, exist_ok=True)
            
            seed_val = str(getattr(self.args, 'seed', 'unknown'))
            lr_str = str(self.args.learning_rate)
            clip_val = str(getattr(self.args, 'dpzero_clip_threshold', 'no_clip'))
            eval_mask_path = os.path.join(npz_dir, f"{task}_bool_mask_{lr_str}_{clip_val}_{seed_val}_{flip_value}.npz")
            
            np.savez(eval_mask_path, test_correct_mask=correct_mask, test_preds=preds, test_labels=labels)
            logger.info(f"Saved boolean tracking mask (1s and 0s) to {eval_mask_path}")

        return output

# """The Trainer class, to easily train a 🤗 Transformers from scratch or finetune it on a new task."""

# import collections
# import inspect
# import math
# import os
# import re
# import shutil
# import warnings
# from pathlib import Path
# from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# import numpy as np
# import torch
# from packaging import version
# from torch import nn
# from torch.utils.data.dataloader import DataLoader
# from torch.utils.data.dataset import Dataset
# from torch.utils.data.distributed import DistributedSampler
# from torch.utils.data.sampler import RandomSampler, SequentialSampler
# from torch.optim.lr_scheduler import LambdaLR
# import math
# import time

# import transformers
# from transformers.file_utils import is_datasets_available, is_in_notebook, is_torch_tpu_available
# from transformers.integrations import (
#     is_comet_available,
#     is_optuna_available,
#     is_ray_available,
#     is_tensorboard_available,
#     is_wandb_available,
# )
# from transformers.optimization import AdamW, get_linear_schedule_with_warmup, get_scheduler

# from transformers.trainer_callback import (
#     DefaultFlowCallback,
#     ProgressCallback,
# )
# from transformers.trainer_utils import (
#     default_compute_objective,
# )
# from transformers.training_args import TrainingArguments
# from transformers.utils import logging
# from transformers.trainer_utils import TrainOutput

# from tqdm import tqdm, trange
# from torch.optim import SGD
# import torch.nn.functional as F

# from src.linearhead_trainer import LinearHeadTrainer
# from transformers.trainer_callback import TrainerState

# import copy

# from opacus.accountants.utils import get_noise_multiplier


# _use_native_amp = False
# _use_apex = False

# DEFAULT_CALLBACKS = [DefaultFlowCallback]
# DEFAULT_PROGRESS_CALLBACK = ProgressCallback

# if is_in_notebook():
#     from transformers.utils.notebook import NotebookProgressCallback

#     DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

# # Check if Pytorch version >= 1.6 to switch between Native AMP and Apex
# if version.parse(torch.__version__) < version.parse("1.6"):
#     from transformers.file_utils import is_apex_available

#     if is_apex_available():
#         from apex import amp
#     _use_apex = True
# else:
#     _use_native_amp = True
#     from torch.cuda.amp import autocast

# if version.parse(torch.__version__) < version.parse("1.2"):
#     _use_ddp_no_sync = False
# else:
#     _use_ddp_no_sync = True

# if is_datasets_available():
#     import datasets

# if is_torch_tpu_available():
#     import torch_xla.core.xla_model as xm
#     import torch_xla.debug.metrics as met
#     import torch_xla.distributed.parallel_loader as pl

# if is_tensorboard_available():
#     from transformers.integrations import TensorBoardCallback

#     DEFAULT_CALLBACKS.append(TensorBoardCallback)


# if is_wandb_available():
#     from transformers.integrations import WandbCallback

#     DEFAULT_CALLBACKS.append(WandbCallback)

# if is_comet_available():
#     from transformers.integrations import CometCallback

#     DEFAULT_CALLBACKS.append(CometCallback)

# if is_optuna_available():
#     import optuna

# if is_ray_available():
#     from ray import tune

# logger = logging.get_logger(__name__)
# logger.setLevel(logging.INFO)

# ########## The above part is copied from Transformers' trainer (3.4.0) ##########

# def default_dev_objective(metrics):
#     """
#     Objective used for picking the best model on development sets
#     """
#     if "eval_mnli/acc" in metrics:
#         return metrics["eval_mnli/acc"]
#     elif "eval_mnli-mm/acc" in metrics:
#         return metrics["eval_mnli-mm/acc"]
#     elif "eval_f1" in metrics:
#         return metrics["eval_f1"]
#     elif "eval_mcc" in metrics:
#         return metrics["eval_mcc"]
#     elif "eval_pearson" in metrics:
#         return metrics["eval_pearson"]
#     elif "eval_acc" in metrics:
#         return metrics["eval_acc"]

#     raise Exception("No metric founded for {}".format(metrics))


# def dpzero_clip(loss_diff, C=1.):
#     tmp = torch.min(torch.ones_like(loss_diff), torch.div(C * torch.ones_like(loss_diff), torch.abs(loss_diff)))
#     return torch.mul(tmp, loss_diff)

# class Trainer(LinearHeadTrainer):
#     """
#     Adding some functions based on Transformers' Trainer class.
#     """

#     def create_optimizer_and_scheduler(self, num_training_steps: int):
#         """
#         Based on Transformers' default one, we add fixing layer option where the bottom n layers' parameters
#         are fixed and only the top layers are further fine-tuned.
#         """
#         if self.args.hf_inference_model:
#             return

#         if self.optimizer is None:
#             params = {}
#             for n, p in self.model.named_parameters():
#                 if self.args.fix_layers > 0:
#                     if 'encoder.layer' in n:
#                         try:
#                             layer_num = int(n[n.find('encoder.layer') + 14:].split('.')[0])
#                         except:
#                             print(n)
#                             raise Exception("")
#                         if layer_num >= self.args.fix_layers:
#                             print('yes', n)
#                             params[n] = p
#                         else:
#                             print('no ', n)
#                     elif 'embeddings' in n:
#                         print('no ', n)
#                     else:
#                         print('yes', n)
#                         params[n] = p
#                 else:
#                     params[n] = p
#             no_decay = ["bias", "LayerNorm.weight"]
#             optimizer_grouped_parameters = [
#                 {
#                     "params": [p for n, p in params.items() if not any(nd in n for nd in no_decay)],
#                     "weight_decay": self.args.weight_decay,
#                 },
#                 {
#                     "params": [p for n, p in params.items() if any(nd in n for nd in no_decay)],
#                     "weight_decay": 0.0,
#                 },
#             ]
#             if self.args.optimizer == 'adam':
#                 self.optimizer = AdamW(
#                     optimizer_grouped_parameters,
#                     lr=self.args.learning_rate,
#                     betas=(self.args.adam_beta1, self.args.adam_beta2),
#                     eps=self.args.adam_epsilon,
#                 )
#             elif self.args.optimizer == 'sgd':
#                 self.optimizer = SGD(
#                     optimizer_grouped_parameters,
#                     lr=self.args.learning_rate
#                 )
#             else:
#                 raise NotImplementedError
#         if self.lr_scheduler is None:
#             self.lr_scheduler = get_scheduler(
#                 self.args.lr_scheduler_type,
#                 optimizer=self.optimizer,
#                 num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
#                 num_training_steps=num_training_steps,
#             )

#     def should_optim(self, name, param):
#         return (not self.args.layer_wise_optim or f".{self.state.global_step % self.model.config.num_hidden_layers}." in name) and param.requires_grad


#     def zo_forward(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
#         model.eval()
#         inputs = self._prepare_inputs(inputs)
#         if self.args.optimize_acc:
#             loss, logits = model(**inputs)
#             preds = F.softmax(logits, dim=-1)
#             acc = torch.sum(torch.argmax(preds, 1) == inputs['labels']) / len(preds)
#             loss = -acc
#         else:
#             with self.compute_loss_context_manager():
#                 loss = self.compute_loss(model, inputs)
#             if self.args.n_gpu > 1:
#                 loss = loss.mean()  # mean() to average on multi-gpu parallel training
#         self.state.zo_forward_step += 1
#         return loss.detach()


#     def efficient_perturb_parameters(self, model: nn.Module, random_seed: int, scaling_factor=1):
#         torch.manual_seed(random_seed)
#         for name, param in self.named_parameters_to_optim:
#             z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
#             param.data = param.data + scaling_factor * z * self.args.zero_order_eps
#         return model

    
#     def perturb_parameters(self, model: nn.Module, random_vector=None, scaling_factor=1):
#         if random_vector is None:
#             random_vector = {}

#         for name, param in self.named_parameters_to_optim:
#             if name in random_vector:
#                 z = random_vector[name]
#             else:
#                 z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
#                 random_vector[name] = z
#             param.data = param.data + scaling_factor * z * self.args.zero_order_eps

#         return model, random_vector


#     def get_num_samples(self):
#         if self.args.zero_order_sample_scheduler is None:
#             noise_sample_time = 1 
#         elif self.args.zero_order_sample_scheduler == "linear":
#             noise_sample_time = max(1, int(self.state.global_step / self.args.max_steps * self.args.zero_order_sample))
#         elif self.args.zero_order_sample_scheduler == "constant":
#             noise_sample_time = int(self.args.zero_order_sample)
#         else:
#             raise NotImplementedError

#         return noise_sample_time

#     # --- ADDED FOR L2 TRACKING: Calculation Helper Method ---
#     def get_l2_weight_distance(self, initial_weights):
#         """Calculates the L2 distance between current weights and initial weights (w_t - w_0)"""
#         l2_sum = 0.0
#         for n, p in self.model.named_parameters():
#             if n in initial_weights:
#                 # Move current weights to CPU for calculation to avoid GPU OOM
#                 l2_sum += torch.sum((p.detach().cpu() - initial_weights[n]) ** 2).item()
#         return math.sqrt(l2_sum)
#     # --------------------------------------------------------

#     def train(self, model_path=None, dev_objective=None):
#         """
#         Main training entry point.

#         The training logic is directly borrowed from transformers.Trainer (version 3.0.2).
#         Add early stopping.
#         """
#         if self.args.from_linearhead and model_path is None:
#             super().train(model_path, dev_objective) # Train output layer using LinearHeadTrainer

#         self.best_dir = None
#         self.objective = -float("inf")
#         self.dev_objective = dev_objective if dev_objective is not None else default_dev_objective

#         # Data loading.
#         train_dataloader = self.get_train_dataloader()
#         num_update_steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps
#         if num_update_steps_per_epoch == 0:
#             num_update_steps_per_epoch = 1
#         if self.args.max_steps > 0:
#             t_total = self.args.max_steps
#             num_train_epochs = self.args.max_steps // num_update_steps_per_epoch + int(
#                 self.args.max_steps % num_update_steps_per_epoch > 0
#             )
#         else:
#             t_total = int(len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs)
#             num_train_epochs = self.args.num_train_epochs

#         self.create_optimizer_and_scheduler(num_training_steps=t_total)
#         optimizer = self.optimizer
#         scheduler = self.lr_scheduler

#         # Check if saved optimizer or scheduler states exist
#         if (
#             model_path is not None
#             and os.path.isfile(os.path.join(model_path, "optimizer.pt"))
#             and os.path.isfile(os.path.join(model_path, "scheduler.pt"))
#         ):
#             # Load in optimizer and scheduler states
#             optimizer.load_state_dict(
#                 torch.load(os.path.join(model_path, "optimizer.pt"), map_location=self.args.device)
#             )
#             scheduler.load_state_dict(torch.load(os.path.join(model_path, "scheduler.pt")))

#         model = self.model

#         if self.args.fp16 and _use_apex:
#             if not transformers.is_apex_available():
#                 raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
#             model, optimizer = amp.initialize(model, optimizer, opt_level=self.args.fp16_opt_level)

#         # Multi-gpu training (should be after apex fp16 initialization)
#         if self.args.n_gpu > 1:
#             model = torch.nn.DataParallel(model)

#         # Distributed training (should be after apex fp16 initialization)
#         if self.args.local_rank != -1:
#             model = torch.nn.parallel.DistributedDataParallel(
#                 model,
#                 device_ids=[self.args.local_rank],
#                 output_device=self.args.local_rank,
#                 find_unused_parameters=True,
#             )

#         # Train
#         if transformers.is_torch_tpu_available():
#             total_train_batch_size = self.args.train_batch_size * xm.xrt_world_size()
#         else:
#             total_train_batch_size = (
#                 self.args.train_batch_size
#                 * self.args.gradient_accumulation_steps
#                 * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1)
#             )
#         logger.info("***** Running training *****")
#         logger.info("  Num examples = %d", self.num_examples(train_dataloader))
#         logger.info("  Num Epochs = %d", num_train_epochs)
#         logger.info("  Instantaneous batch size per device = %d", self.args.per_device_train_batch_size)
#         logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_train_batch_size)
#         logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
#         logger.info("  Total optimization steps = %d", t_total)

#         self.state = TrainerState()
#         self.state.global_step = 0


#         # --- ADDITION: Setup Gradient Histogram Tracking ---
#         self.pre_clip_history = []
#         self.post_clip_history = []
#         self.clip_count_accumulator = 0  # Resets every log
#         self.clip_step_accumulator = 0   # Resets every log
#         self.global_clip_count = 0       # NEVER resets (for final summary)
#         self.global_clip_steps = 0       # NEVER resets (for final summary)
#         # -------------------------------------------------

#         start_time = time.time()
#         self.state.zo_forward_step = 0
#         self.epoch = 0
#         epochs_trained = 0
#         steps_trained_in_current_epoch = 0
#         flip = False

#         if self.args.gradient_checkpointing:
#             model.gradient_checkpointing_enable()

#         # Check if continuing training from a checkpoint
#         if model_path is not None:
#             # set global_step to global_step of last saved checkpoint from model path
#             try:
#                 self.state.global_step = int(model_path.split("-")[-1].split("/")[0])
#                 epochs_trained = self.state.global_step // (len(train_dataloader) // self.args.gradient_accumulation_steps)
#                 steps_trained_in_current_epoch = self.state.global_step % (
#                     len(train_dataloader) // self.args.gradient_accumulation_steps
#                 )

#                 logger.info("  Continuing training from checkpoint, will skip to saved global_step")
#                 logger.info("  Continuing training from epoch %d", epochs_trained)
#                 logger.info("  Continuing training from global step %d", self.state.global_step)
#                 logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
#             except ValueError:
#                 self.state.global_step = 0
#                 logger.info("  Starting fine-tuning.")

#         tr_loss = torch.tensor(0.0).to(self.args.device)
#         logging_loss_scalar = 0.0
#         model.zero_grad()
#         metrics = None

#         # --- ADDED FOR L2 TRACKING: Initial Weights Capture ---
#         logger.info("Capturing initial weights (w_0) for L2 distance tracking...")
#         initial_weights = {n: p.clone().detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
#         # ------------------------------------------------------

#         # # set up dp parameters:
#         # if self.args.dpzero:
#         #     sample_rate= total_train_batch_size / (model.num_k * model.num_labels)
#         #     multiplier = get_noise_multiplier(target_epsilon=self.args.dp_epsilon,
#         #                                       target_delta=self.args.dp_delta,
#         #                                       steps=t_total,
#         #                                       sample_rate=sample_rate,
#         #                                       )
#         #     self.dpzero_gaussian_std = 2 * multiplier * self.args.dpzero_clip_threshold / total_train_batch_size

#         # 1. Safely extract the Dataset Name ("sst-2", "snli", etc.)
#         if hasattr(self, 'model') and hasattr(self.model, 'data_args') and hasattr(self.model.data_args, 'task_name'):
#             dataset_name = self.model.data_args.task_name.lower()
#         elif hasattr(self.args, 'task_name'):
#             dataset_name = self.args.task_name.lower()
#         else:
#             dataset_name = "unknown_dataset"
        

#         for epoch in range(epochs_trained, int(num_train_epochs)):
#             if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
#                 train_dataloader.sampler.set_epoch(epoch)

#             if transformers.is_torch_tpu_available():
#                 parallel_loader = pl.ParallelLoader(train_dataloader, [self.args.device]).per_device_loader(
#                     self.args.device
#                 )
#                 epoch_iterator = tqdm(parallel_loader, desc="Iteration", disable=not self.is_local_process_zero())
#             else:
#                 epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=True)

#             # Reset the past mems state at the beginning of each epoch if necessary.
#             if self.args.past_index >= 0:
#                 self._past = None

#             if (epoch % 20) == 0:
#                 # print(epoch, flip)
#                 flip = not flip

#             for step, inputs in enumerate(epoch_iterator):
#                 if self.args.sync_embedding_layers:
#                     assert model.module.model_type == 'opt', 'did not implement embedding layer synchronization for non-OPT models'
#                     model.module.model.decoder.embed_tokens.weight = model.module.lm_head.weight
                
#                 # Skip past any already trained steps if resuming training
#                 if steps_trained_in_current_epoch > 0:
#                     steps_trained_in_current_epoch -= 1
#                     continue
                    
#                 if self.args.zero_order_optim:
#                     # Get parameters that should be optimized (for layer-wise optimization and prefix-tuning)
#                     self.named_parameters_to_optim = []
#                     for name, param in model.named_parameters():
#                         if self.should_optim(name, param):
#                             self.named_parameters_to_optim.append((name, param))

#                     # get number of zs to sample
#                     num_zs = self.get_num_samples()
#                     if num_zs > 1:
#                         assert self.args.zero_order_use_trainer_optim, 'cannot sample multiple zs without storing intermediate gradient. use trainer.'

#                     for _ in range(num_zs):
#                         # prepare for sampling new zs
#                         random_vector = None
#                         if self.args.efficient_zero_order:
#                             random_seed = np.random.randint(1000000000)

#                         with torch.no_grad():
#                             # first function evaluation
#                             if self.args.efficient_zero_order:
#                                 model = self.efficient_perturb_parameters(model, random_seed)
#                             else:
#                                 model, random_vector = self.perturb_parameters(model)
#                             loss1 = self.zo_forward(model, inputs)

#                             # second function evaluation
#                             if self.args.efficient_zero_order:
#                                 model = self.efficient_perturb_parameters(model, random_seed, scaling_factor=-2)
#                             else:
#                                 model, random_vector = self.perturb_parameters(model, random_vector, scaling_factor=-2)
#                             loss2 = self.zo_forward(model, inputs)

#                         projected_grad = (loss1 - loss2) / (2 * self.args.zero_order_eps)

#                         # DPZero gradient clipping and noise injection

#                         is_track_step = (self.state.global_step % 100 == 0)
#                         if is_track_step:
#                             self.pre_clip_history.append(projected_grad.detach().cpu().numpy().flatten())
#                         if self.args.dpzero:
#                             if self.args.dpzero_clip_threshold > 0 and flip:
#                                 # --- ADDITION: Count how many elements exceed the threshold ---
#                                 num_clipped = (torch.abs(projected_grad) > self.args.dpzero_clip_threshold).sum().item()
                                
#                                 # Temporary trackers (reset for interval logging)
#                                 self.clip_count_accumulator += num_clipped
#                                 self.clip_step_accumulator += 1
                                
#                                 # Global trackers (never reset)
#                                 self.global_clip_count += num_clipped
#                                 self.global_clip_steps += 1

#                                 projected_grad = dpzero_clip(projected_grad, self.args.dpzero_clip_threshold)
#                             # projected_grad += torch.randn(1).item() * self.dpzero_gaussian_std
#                             if is_track_step:
#                                 self.post_clip_history.append(projected_grad.detach().cpu().numpy().flatten())
#                             projected_grad = projected_grad.mean()

#                             # projected_grad += torch.randn(1).item() * self.dpzero_gaussian_std

#                         # scale grad according to accumulation
#                         if self.args.gradient_accumulation_steps > 1:
#                             assert self.args.zero_order_use_trainer_optim, 'grad accumulation not implemented for non-trainer ZO yet'
#                             projected_grad = projected_grad / self.args.gradient_accumulation_steps

#                         # scale grad according to number of zs sampled
#                         if not self.args.scale_lr_with_samples:
#                             projected_grad = projected_grad / float(num_zs)

#                         # store gradient in parameter buffer if using trainer
#                         # o/w, the loop will exit after one round and the update will be applied directly (see below)
#                         if self.args.zero_order_use_trainer_optim:
#                             if self.args.efficient_zero_order:
#                                 torch.manual_seed(random_seed)

#                             for name, param in self.named_parameters_to_optim:
#                                 # recover noise used in perturbations
#                                 if self.args.efficient_zero_order:
#                                     z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
#                                 else:
#                                     z = random_vector[name]

#                                 if param.grad is None:
#                                     param.grad = projected_grad * z
#                                 else:
#                                     param.grad += projected_grad * z

#                         # reset model back to its parameters at start of step
#                         if self.args.efficient_zero_order:
#                             model = self.efficient_perturb_parameters(model, random_seed)
#                         else:
#                             model, random_vector = self.perturb_parameters(model, random_vector)

#                     # apply gradient updates
#                     # if using trainer, follow trainer logic to clip grad and check if parameters should be updated
#                     if self.args.zero_order_use_trainer_optim:
#                         if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
#                             # last step in epoch but step is always smaller than gradient_accumulation_steps
#                             len(epoch_iterator) <= self.args.gradient_accumulation_steps
#                             and (step + 1) == len(epoch_iterator)
#                         ):
#                             # Gradient norm clipping
#                             if self.args.zero_order_clip_grad:
#                                 norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

#                             # Update the parameters and step scheduler
#                             optimizer.step()
#                             scheduler.step()
                        
#                             # logging
#                             if (self.args.logging_steps > 0 and self.state.global_step % self.args.logging_steps == 0) or (
#                                 self.state.global_step == 1 and self.args.logging_first_step
#                             ):
#                                 logs = {}
#                                 logs["loss"] = loss1.item()
#                                 if not self.args.zero_order_clip_grad:
#                                     norm = 0.0
#                                     for _, p in model.named_parameters():
#                                         if p.grad is not None:
#                                             norm += torch.sum(p.grad ** 2)
#                                     norm = torch.sqrt(norm)
#                                 logs["grad_norm"] = norm.item()
#                                 logs["learning_rate"] = (
#                                     scheduler.get_last_lr()[0]
#                                     if version.parse(torch.__version__) >= version.parse("1.4")
#                                     else scheduler.get_lr()[0]
#                                 )
#                                 logs["num_zs"] = num_zs
#                                 logs["global_step"] = self.state.global_step
#                                 logs["zo_forward_step"] = self.state.zo_forward_step
#                                 logs["max_steps"] = self.args.max_steps
#                                 logs["max_zo_forward_steps"] = self.args.max_zo_forward_steps
#                                 logs["time"] = int(time.time() - start_time)
                                
#                                 # --- ADDED FOR L2 TRACKING: Log calculation for ZO Trainer Optim ---
#                                 logs["l2_weight_dist"] = self.get_l2_weight_distance(initial_weights)
#                                 # -------------------------------------------------------------------
#                                 # --- ADDITION: Log the average clipped numbers ---
#                                 if self.clip_step_accumulator > 0:
#                                     logs["avg_clipped_per_iter"] = self.clip_count_accumulator / self.clip_step_accumulator
#                                     # Reset accumulators for the next logging window
#                                     self.clip_count_accumulator = 0
#                                     self.clip_step_accumulator = 0
#                                 # -------------------------------------------------

#                                 logs["clipping"] = flip 
#                                 self.log(logs)
#                                 logger.info(str(logs))
                            
#                             model.zero_grad()
#                             self.state.global_step += 1
#                             self.epoch = epoch + (step + 1) / len(epoch_iterator)
#                     # if not using the trainer, the updates are resampled and directly applied to the parameters
#                     else:
#                         # Efficient mode 
#                         # WARNING: no gradient accumulation when not storing the grad
#                         assert self.args.gradient_accumulation_steps == 1, 'gradient accumulation is not supported for zero-order optimization'
#                         assert self.args.zero_order_sample_scheduler is None
#                         assert not self.args.zero_order_clip_grad, 'gradient clipping not implemented yet for non-trainer ZO'

#                         if self.args.efficient_zero_order:
#                             torch.manual_seed(random_seed)     
#                         for name, param in self.named_parameters_to_optim:
#                             if self.args.efficient_zero_order:
#                                 z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
#                             else:
#                                 z = random_vector[name]
#                             param.data = param.data - self.args.learning_rate * projected_grad * z 

#                         if (self.args.logging_steps > 0 and self.state.global_step % self.args.logging_steps == 0) or (
#                                 self.state.global_step == 1 and self.args.logging_first_step
#                             ):
#                                 logs = {}
#                                 if self.args.dpzero:
#                                     logs["loss"] = loss1.mean().item()
#                                 else:
#                                     logs["loss"] = loss1.item()
#                                 logs["learning_rate"] = self.args.learning_rate
#                                 logs["global_step"] = self.state.global_step
#                                 logs["zo_forward_step"] = self.state.zo_forward_step
#                                 logs["max_steps"] = self.args.max_steps
#                                 logs["max_zo_forward_steps"] = self.args.max_zo_forward_steps
#                                 logs["time"] = int(time.time() - start_time)
                                
#                                 # --- ADDED FOR L2 TRACKING: Log calculation for ZO Non-Trainer Optim ---
#                                 logs["l2_weight_dist"] = self.get_l2_weight_distance(initial_weights)
#                                 # -----------------------------------------------------------------------
                                
#                                 # --- ADDITION: Log the average clipped numbers ---
#                                 if self.clip_step_accumulator > 0:
#                                     logs["avg_clipped_per_iter"] = self.clip_count_accumulator / self.clip_step_accumulator
#                                     # Reset accumulators for the next logging window
#                                     self.clip_count_accumulator = 0
#                                     self.clip_step_accumulator = 0
#                                 # -------------------------------------------------

#                                 logs["clipping"] = flip 
                                
#                                 self.log(logs)
#                                 logger.info(str(logs))


#                         self.state.global_step += 1
#                         self.epoch = epoch + (step + 1) / len(epoch_iterator)
                    
#                     # Debug information
#                     # print("%.5f, %.5f" % (loss1.item(), loss2.item()))
#                     # print("Loss: %.10f, projected_grad: %.5f" % (loss1, projected_grad))

#                 # standard, non-ZO optimization
#                 else:
#                     tr_loss += self.training_step(model, inputs)

#                     if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
#                         # last step in epoch but step is always smaller than gradient_accumulation_steps
#                         len(epoch_iterator) <= self.args.gradient_accumulation_steps
#                         and (step + 1) == len(epoch_iterator)
#                     ):
#                         if self.args.fp16 and _use_native_amp:
#                             self.scaler.unscale_(optimizer)
#                             norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
#                         elif self.args.fp16:
#                             norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), self.args.max_grad_norm)
#                         else:
#                             norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

#                         if self.args.optimizer_variant == 'signgd':
#                             for n,p in model.named_parameters():
#                                 if p.grad is not None:
#                                     p.grad = torch.sign(p.grad)

#                         if transformers.is_torch_tpu_available():
#                             xm.optimizer_step(optimizer)
#                         elif self.args.fp16 and _use_native_amp:
#                             self.scaler.step(optimizer)
#                             self.scaler.update()
#                         else:
#                             optimizer.step()

#                         scheduler.step()
#                         model.zero_grad()
#                         self.state.global_step += 1
#                         self.epoch = epoch + (step + 1) / len(epoch_iterator)

#                         if (self.args.logging_steps > 0 and self.state.global_step % self.args.logging_steps == 0) or (
#                             self.state.global_step == 1 and self.args.logging_first_step
#                         ):
#                             logs = {}
#                             tr_loss_scalar = tr_loss.item()
#                             logs["loss"] = (tr_loss_scalar - logging_loss_scalar) / self.args.logging_steps
#                             logs["norm"] = norm.item()
#                             # backward compatibility for pytorch schedulers
#                             logs["learning_rate"] = (
#                                 scheduler.get_last_lr()[0]
#                                 if version.parse(torch.__version__) >= version.parse("1.4")
#                                 else scheduler.get_lr()[0]
#                             )
#                             logging_loss_scalar = tr_loss_scalar

#                             # --- ADDED FOR L2 TRACKING: Log calculation for Standard FO Optim ---
#                             logs["l2_weight_dist"] = self.get_l2_weight_distance(initial_weights)
#                             # --------------------------------------------------------------------
                            
#                             logs["clipping"] = flip 

#                             self.log(logs)
#                             logger.info(str(logs))

#                 if self.args.max_steps > 0 and self.state.global_step > self.args.max_steps or (self.args.max_zo_forward_steps > 0 and self.state.zo_forward_step > self.args.max_zo_forward_steps):
#                     epoch_iterator.close()
#                     break

#                 if self.args.evaluate_during_training and self.state.global_step % self.args.eval_steps == 0:
#                     output = self.evaluate()
#                     metrics = output.metrics
#                     objective = self.dev_objective(metrics)
#                     if objective > self.objective:
#                         logger.info("Best dev result: {}".format(objective))
#                         self.objective = objective
#                         # self.save_model(self.args.output_dir)

#                         # Now we save this to (CPU) memory instead of disk <-- much faster
#                         self.best_model_ckpt = {k: v.detach().cpu() for k, v in model.state_dict().items()}

            
#             if (epoch % 20 == 0):
#                 # After the training loop of each epoch, evaluate and print accuracy
#                 output = self.evaluate()  # Evaluate the model after each epoch
#                 metrics = output.metrics
#                 eval_acc = metrics.get("accuracy", None)  # Assuming "accuracy" is in the returned metrics

#                 if eval_acc is not None:
#                     print(f"Epoch {epoch + 1} - Accuracy: {eval_acc:.4f}")
            
            
#             if self.args.max_steps > 0 and self.state.global_step > self.args.max_steps or (self.args.max_zo_forward_steps > 0 and self.state.zo_forward_step > self.args.max_zo_forward_steps):
#                 # train_iterator.close()
#                 break
#             if self.args.tpu_metrics_debug or self.args.debug:
#                 # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
#                 xm.master_print(met.metrics_report())

#         if self.args.past_index and hasattr(self, "_past"):
#             # Clean the state at the end of training
#             delattr(self, "_past")


#         # --- ADDITION: Save Histogram Data to disk ---
#         if len(self.pre_clip_history) > 0:
#             try:
#                 # 1. Create the folder
#                 os.makedirs("mixed", exist_ok=True)
                
#                 # 2. Dynamically name the file based on arguments
#                 lr_val = self.args.learning_rate
#                 clip_val = self.args.dpzero_clip_threshold
#                 seed_val = str(getattr(self.args, 'seed', 'unknown'))
#                 file_path = os.path.join("mixed", f"{lr_val}_{clip_val}_histogram_seed_{seed_val}.npz")
                
#                 # 3. Concatenate ALL points into one massive array and save
#                 np.savez(
#                     file_path,
#                     pre_clip=np.concatenate(self.pre_clip_history),
#                     post_clip=np.concatenate(self.post_clip_history) if len(self.post_clip_history) > 0 else []
#                 )
#                 logger.info(f"✅ Saved gradient distributions to {file_path}")
#             except Exception as e:
#                 logger.error(f"Failed to save histogram data: {e}")
#         # ---------------------------------------------

#         # --- ADDITION: Print Global Clipping Summary ---
#         if self.global_clip_steps > 0:
#             global_avg_clipped = self.global_clip_count / self.global_clip_steps
#             # Calculate the total batch size vector length by looking at the first saved gradient
#             total_elements = len(self.pre_clip_history[0]) if len(self.pre_clip_history) > 0 else "Unknown"
            
#             logger.info("\n" + "="*50)
#             logger.info("📉 GLOBAL CLIPPING STATISTICS")
#             logger.info("="*50)
#             logger.info(f"Total ZO Steps Executed: {self.global_clip_steps}")
#             logger.info(f"Threshold Used: {self.args.dpzero_clip_threshold}")
#             logger.info(f"Average elements clipped per step: {global_avg_clipped:.2f} (out of {total_elements})")
#             logger.info("="*50 + "\n")
#         # -----------------------------------------------


#         logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
#         return TrainOutput(self.state.global_step, tr_loss / self.state.global_step, metrics), self.objective


#     """
#     Difference compared to original implementation: return output instead of output.metrics (so there is also the logits)
#     """
#     def evaluate(self, eval_dataset: Optional[Dataset] = None, task = None) -> Dict[str, float]:
#         """
#         Run evaluation and returns metrics.

#         The calling script will be responsible for providing a method to compute metrics, as they are
#         task-dependent (pass it to the init :obj:`compute_metrics` argument).

#         You can also subclass and override this method to inject custom behavior.

#         Args:
#             eval_dataset (:obj:`Dataset`, `optional`):
#                 Pass a dataset if you wish to override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`,
#                 columns not accepted by the ``model.forward()`` method are automatically removed. It must implement
#                 the :obj:`__len__` method.

#         Returns:
#             A dictionary containing the evaluation loss and the potential metrics computed from the predictions.
#         """
#         if eval_dataset is not None and not isinstance(eval_dataset, collections.abc.Sized):
#             raise ValueError("eval_dataset must implement __len__")

#         eval_dataloader = self.get_eval_dataloader(eval_dataset)

#         output = self.prediction_loop(eval_dataloader, description="Evaluation")

#         self.log(output.metrics)
#         logger.info(output.metrics)

#         if self.args.tpu_metrics_debug or self.args.debug:
#             # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
#             xm.master_print(met.metrics_report())

#         # =========================================================
#         # --- SAVE EVALUATION LOGITS TO NPZ ---
#         # =========================================================
#         if output.predictions is not None and output.label_ids is not None and task is not None:
#             preds = np.argmax(output.predictions, axis=-1)
#             labels = output.label_ids
#             correct_mask = (preds == labels).astype(int) 
            
#             npz_dir = os.path.join(os.getcwd(), "mixed")
#             os.makedirs(npz_dir, exist_ok=True)

#             # 1. Safely extract the Dataset Name ("sst-2", "snli", etc.)
#             if hasattr(self, 'model') and hasattr(self.model, 'data_args') and hasattr(self.model.data_args, 'task_name'):
#                 dataset_name = self.model.data_args.task_name.lower()
#             elif hasattr(self.args, 'task_name'):
#                 dataset_name = self.args.task_name.lower()
#             else:
#                 dataset_name = "unknown_dataset"
            
#             lr_str = str(self.args.learning_rate)
#             seed_val = str(getattr(self.args, 'seed', 'unknown'))
#             clip_val = str(getattr(self.args, 'dpzero_clip_threshold', 0)) if getattr(self.args, 'dpzero', False) else "0"
            
#             eval_mask_path = os.path.join(npz_dir, f"{dataset_name}_{task}_logits_lr{lr_str}_clip{clip_val}_seed{seed_val}.npz")
            
#             np.savez(eval_mask_path, test_correct_mask=correct_mask, test_preds=preds, test_labels=labels, logits=output.predictions)
#             logger.info(f"Saved evaluation logits to {eval_mask_path}")

#             preds = np.argmax(output.predictions, axis=-1)
#             labels = output.label_ids
#             # Boolean array of 1s and 0s indicating correct/incorrect predictions
#             correct_mask = (preds == labels).astype(int) 
            
#             npz_dir = os.path.join(os.getcwd(), "mixed")
#             os.makedirs(npz_dir, exist_ok=True)
            
#             seed_val = str(getattr(self.args, 'seed', 'unknown'))
#             lr_str = str(self.args.learning_rate)
#             clip_val = str(getattr(self.args, 'dpzero_clip_threshold', 'no_clip'))
#             eval_mask_path = os.path.join(npz_dir, f"{task}_bool_mask_{lr_str}_{clip_val}_{seed_val}.npz")
            
#             np.savez(eval_mask_path, test_correct_mask=correct_mask, test_preds=preds, test_labels=labels)
#             logger.info(f"Saved boolean tracking mask (1s and 0s) to {eval_mask_path}")

#         return output
