"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import dataclasses
import getpass
import glob
import logging
import math
import os
import pprint
import shutil
import signal
import socket
import uuid
from collections import defaultdict
from typing import Any, Dict, List, Optional

import coolname
import hydra
import numpy as np
import omegaconf
import pynvml
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import DistributedType, ProjectConfiguration
from requests.exceptions import RequestException
from transformers import get_linear_schedule_with_warmup, set_seed

import environment_accel as environment
import wandb
from algos.ppo import AdaptiveKLController, FixedKLController, KLController
from core import nest, record, stats, vtrace
from dataset.imdb_dataloader import get_imdb_data_loaders
from dataset.ni_dataloader import get_ni_data_loaders
from dataset.summarisation_dataloader import get_summarisation_dataloader
from evaluation.reward_functions import make_reward_function
from evaluation.summarisation_metrics import compute_all_summarisation_metrics
from models.model_creation import (construct_device_map,
                                   construct_model_from_class,
                                   tie_frozen_layers)
from utils.core import normalise_seed

FLAGS: Any

local_stats = defaultdict(
    stats.StatMean,
    {
        "mean_reward": stats.StatMean(),
        "mean_kl_reward": stats.StatMean(),
        "mean_response_length": stats.StatMean(),
        "median_response_length": stats.StatMean(),
        "SPS": stats.StatMean(),
        "env_train_steps": stats.StatSum(),
        "rollouts": stats.StatSum(),
        "algo_steps": stats.StatSum(),
        "optimizer_steps": stats.StatSum(),
        "unclipped_grad_norm": stats.StatMean(),
        "num_gradients": stats.StatMean(),
        "policy_loss": stats.StatMean(),
        "baseline_loss": stats.StatMean(),
        "max_baseline_value": stats.StatMean(),
        "mean_baseline_value": stats.StatMean(),
        "min_baseline_value": stats.StatMean(),
        "entropy_loss": stats.StatMean(),
        "max_entropy_value": stats.StatMean(),
        "mean_entropy_value": stats.StatMean(),
        "min_entropy_value": stats.StatMean(),
        "kl_div": stats.StatMean(),
        "kl_coef": stats.StatMean(),
        "mean_kl_div_value": stats.StatMean(),
        "max_kl_div_value": stats.StatMean(),
        "min_kl_div_value": stats.StatMean(),
        "clipped_baseline_fraction": stats.StatMean(),
        "clipped_policy_fraction": stats.StatMean(),
        "running_advantages": stats.StatMean(cumulative=True),
        "sample_advantages": stats.StatMean(),
        "sample_advantages_std": stats.StatMean(),
        "learning_rate": stats.StatMean(),
        "validation/average_reward": stats.StatMean(),
        "validation/median_reward": stats.StatMean(),
        "validation/average_rewards": stats.StatMean(),
        "validation/median_rewards": stats.StatMean(),
        "validation/average_kl_div": stats.StatMean(),
        "validation/median_kl_div": stats.StatMean(),
        "validation/average_kl_rewards": stats.StatMean(),
        "validation/median_kl_rewards": stats.StatMean(),
        "validation/average_auxiliary_rewards": stats.StatMean(),
        "validation/median_auxiliary_rewards": stats.StatMean(),
        "validation/average_prediction_length": stats.StatMean(),
        "validation/median_prediction_length": stats.StatMean(),
        "validation/exact_match": stats.StatMean(),
        "validation/rouge1": stats.StatMean(),
        "validation/rougeL": stats.StatMean(),
        "validation/bleu": stats.StatMean(),
        "test/average_reward": stats.StatMean(),
        "test/median_reward": stats.StatMean(),
        "test/average_rewards": stats.StatMean(),
        "test/median_rewards": stats.StatMean(),
        "test/average_prediction_length": stats.StatMean(),
        "test/median_prediction_length": stats.StatMean(),
        "test/average_kl_div": stats.StatMean(),
        "test/median_kl_div": stats.StatMean(),
        "test/average_kl_rewards": stats.StatMean(),
        "test/median_kl_rewards": stats.StatMean(),
        "test/average_auxiliary_rewards": stats.StatMean(),
        "test/median_auxiliary_rewards": stats.StatMean(),
        "test/exact_match": stats.StatMean(),
        "test/rouge1": stats.StatMean(),
        "test/rougeL": stats.StatMean(),
        "test/bleu": stats.StatMean(),
        "train/average_rewards": stats.StatMean(),
        "train/average_reward": stats.StatMean(),
        "train/median_reward": stats.StatMean(),
        "train/median_rewards": stats.StatMean(),
        "train/average_prediction_length": stats.StatMean(),
        "train/median_prediction_length": stats.StatMean(),
        "train/exact_match": stats.StatMean(),
        "train/rouge1": stats.StatMean(),
        "train/rougeL": stats.StatMean(),
        "train/bleu": stats.StatMean(),
        "train/average_kl_div": stats.StatMean(),
        "train/median_kl_div": stats.StatMean(),
        "train/average_kl_rewards": stats.StatMean(),
        "train/median_kl_rewards": stats.StatMean(),
        "train/average_auxiliary_rewards": stats.StatMean(),
        "train/median_auxiliary_rewards": stats.StatMean(),
    },
)


@dataclasses.dataclass
class LearnerState:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    lr_scheduler: torch.optim.lr_scheduler.LambdaLR
    kl_controller: KLController
    accelerator: Accelerator
    model_version: int = 0
    algo_steps: int = 0

    def state_dict(self):
        return dict(
            model_version=self.model_version,
            algo_steps=self.algo_steps,
        )

    def load_state_dict(self, state):
        for k, v in state.items():
            if k not in ("model", "optimizer", "kl_controller", "lr_scheduler", "accelerator"):
                setattr(self, k, v)


def log_gpu_utilization(prefix=""):
    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError_LibraryNotFound:
        logging.info(f"{prefix}: GPU utilization logging is not supported on this machine.")
        return
    for device_idx in range(torch.cuda.device_count()):
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        logging.info(f"{prefix} cuda:{device_idx}: GPU memory occupied: {info.used//1024**2} MB.")


def get_layer_devices(model: torch.nn.Module) -> List[torch.device]:
    embed_device = next(model.get_base_model_transformer().embed_tokens.parameters()).device
    layer_devices = [
        next(layer.parameters()).device
        for layer in getattr(model.get_base_model_transformer(), model.base_model_layers)
    ]
    return [embed_device] + layer_devices


def compute_baseline_loss(actor_baseline, learner_baseline, target, mask, clip_delta_value=None, stats=None):
    baseline_loss = (target - learner_baseline) ** 2

    if clip_delta_value:
        # Common PPO trick - clip a change in baseline fn
        # (cf PPO2 github.com/Stable-Baselines-Team/stable-baselines)
        learner_baseline_clipped = torch.max(
            torch.min(learner_baseline, actor_baseline + clip_delta_value), actor_baseline - clip_delta_value
        )

        clipped_baseline_loss = (target - learner_baseline_clipped) ** 2

        if stats:
            clipped = (clipped_baseline_loss > baseline_loss).float().mean().item()
            stats["clipped_baseline_fraction"] += clipped

        baseline_loss = torch.max(baseline_loss, clipped_baseline_loss)

    if stats:
        stats["max_baseline_value"] += torch.max(learner_baseline).item()
        stats["min_baseline_value"] += torch.min(learner_baseline).item()
        stats["mean_baseline_value"] += torch.mean(learner_baseline).item()
    return 0.5 * torch.mean(mask * baseline_loss)


def compute_entropy_loss(logits, mask, stats=None):
    policy = F.softmax(logits, dim=-1)
    log_policy = F.log_softmax(logits, dim=-1)
    entropy_per_timestep = torch.sum(-policy * log_policy, dim=-1) * mask
    if stats:
        stats["max_entropy_value"] += torch.max(entropy_per_timestep).item()
        stats["min_entropy_value"] += torch.min(entropy_per_timestep).item()
        stats["mean_entropy_value"] += torch.mean(entropy_per_timestep).item()
    return -torch.mean(entropy_per_timestep)


def compute_kl_loss(learner_logits, reference_logits, response_ids, kl_controller, mask, stats=None):
    """Compute KL divergence between actor and learner logits."""
    kl_reward_value, kl_divergence = environment.calculate_kl_control_reward(
        learner_logits, reference_logits, response_ids, kl_controller, FLAGS
    )
    kl_loss = -torch.mean(kl_reward_value * mask)
    if stats:
        stats["max_kl_div_value"] += torch.max(kl_divergence).item()
        stats["min_kl_div_value"] += torch.min(kl_divergence).item()
        stats["mean_kl_div_value"] += torch.mean(kl_divergence).item()
    return kl_loss


def compute_policy_gradient_loss(
    actor_log_prob,
    learner_log_prob,
    advantages,
    mask,
    normalize_advantages=False,
    clip_delta_policy=None,
    stats=None,
):
    advantages = advantages.detach()
    stats["running_advantages"] += advantages

    adv = advantages

    if normalize_advantages:
        # Common PPO trick (cf PPO2 github.com/Stable-Baselines-Team/stable-baselines)
        if FLAGS.use_global_advantage_norm:
            sample_adv = stats["running_advantages"]
        else:
            sample_adv = adv
        advantages = (adv - sample_adv.mean()) / (1e-8 + sample_adv.std())
        stats["sample_advantages"] += advantages.mean().item()
        stats["sample_advantages_std"] += advantages.std().item()

    if clip_delta_policy:
        # APPO policy loss - clip a change in policy fn
        ratio = torch.exp(learner_log_prob - actor_log_prob)
        policy_loss = ratio * advantages

        clip_high = 1.0 + clip_delta_policy
        clip_low = 1.0 - clip_delta_policy

        clipped_ratio = torch.clamp(ratio, clip_low, clip_high)
        clipped_policy_loss = clipped_ratio * advantages

        if stats:
            clipped_fraction = (clipped_policy_loss < policy_loss).float().mean().item()
            stats["clipped_policy_fraction"] += clipped_fraction
        policy_loss = torch.min(policy_loss, clipped_policy_loss)
    else:
        # IMPALA policy loss
        policy_loss = learner_log_prob * advantages

    return -torch.mean(policy_loss * mask)


def create_optimizer(model):
    return torch.optim.Adam(
        model.parameters(),
        lr=FLAGS.adam_learning_rate,
        betas=(FLAGS.adam_beta1, FLAGS.adam_beta2),
        eps=FLAGS.adam_eps,
        foreach=False,
    )


def create_lr_scheduler(optimizer, num_training_steps):
    return get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )


def create_kl_controller():
    if FLAGS.adap_kl_ctrl:
        return AdaptiveKLController(
            FLAGS.init_kl_coef,
            FLAGS.target,
            FLAGS.horizon,
        )
    else:
        return FixedKLController(FLAGS.init_kl_coef)


def compute_gradients(accelerator, data, learner_state, stats, tokenizer):
    env_outputs = data["env_outputs"]
    actor_outputs = data["actor_outputs"]
    logging_outputs = data["logging_outputs"]

    learner_outputs = learner_state.model(
        env_outputs["full_sequence"], attention_mask=env_outputs["full_mask"]
    )

    gen_len = actor_outputs["baseline"].shape[1]
    learner_outputs = {
        "policy_logits": learner_outputs.logits[:, -gen_len - 1 : -1, :],
        "baseline": learner_outputs.value[:, -gen_len - 1 : -1],
    }

    kl_loss = compute_kl_loss(
        learner_outputs["policy_logits"],
        actor_outputs["frozen_policy_logits"],
        actor_outputs["action"],
        learner_state.kl_controller,
        env_outputs["mask"],
    )
    kl_loss = kl_loss if not FLAGS.kl_reward else 0.0

    # Outputs are [B, T], Functions Expect [T, B]
    learner_outputs = nest.map(lambda t: t.transpose(0, 1), learner_outputs)
    env_outputs = nest.map(lambda t: t.transpose(0, 1), env_outputs)
    actor_outputs = nest.map(lambda t: t.transpose(0, 1), actor_outputs)

    # Use last baseline value (from the value function) to bootstrap.
    bootstrap_value = learner_outputs["baseline"][-1]

    # Move from env_outputs[t] -> action[t] to action[t] -> env_outputs[t].
    # env_outputs = nest.map(lambda t: t[1:], env_outputs)
    # learner_outputs = nest.map(lambda t: t[:-1], learner_outputs)
    # actor_outputs = nest.map(lambda t: t[:-1], actor_outputs)

    rewards = env_outputs["reward"] * FLAGS.reward_scale
    mean_reward = torch.mean(torch.sum(rewards, dim=0)).item()

    mean_kl_reward = torch.mean(torch.sum(env_outputs["kl_rewards"], dim=0)).item()

    if FLAGS.kl_reward:
        rewards = rewards + env_outputs["kl_rewards"]

    if FLAGS.reward_clip:
        rewards = torch.clip(rewards, -FLAGS.reward_clip, FLAGS.reward_clip)

    mean_total_reward = torch.mean(torch.sum(rewards, dim=0)).item()

    mask = env_outputs["mask"]

    discounts = (1 - env_outputs["done"]).float() * FLAGS.discounting

    vtrace_returns = vtrace.from_logits(
        behavior_policy_logits=actor_outputs["policy_logits"],
        target_policy_logits=learner_outputs["policy_logits"],
        actions=actor_outputs["action"],
        discounts=discounts,
        rewards=rewards,
        values=learner_outputs["baseline"],
        bootstrap_value=bootstrap_value,
        lam=FLAGS.lam,
        importance_sampling_correction=FLAGS.importance_sampling_correction,
        use_gae_lambda_advantages=FLAGS.use_gae_lambda_advantages,
    )

    entropy_loss = FLAGS.entropy_cost * compute_entropy_loss(
        learner_outputs["policy_logits"], mask, stats=stats
    )

    pg_loss = compute_policy_gradient_loss(
        vtrace_returns.behavior_action_log_probs,
        vtrace_returns.target_action_log_probs,
        vtrace_returns.pg_advantages,
        mask,
        FLAGS.normalize_advantages,
        FLAGS.appo_clip_policy,
        stats=stats,
    )

    baseline_loss = FLAGS.baseline_cost * compute_baseline_loss(
        actor_outputs["baseline"],
        learner_outputs["baseline"],
        vtrace_returns.vs,
        mask,
        FLAGS.appo_clip_baseline,
        stats=stats,
    )

    total_loss = entropy_loss + pg_loss + baseline_loss + kl_loss
    accelerator.backward(total_loss)

    response_lengths = [len(r) for r in tokenizer(logging_outputs["text_out"])["input_ids"]]

    stats["env_train_steps"] += gen_len * FLAGS.learn_batch_size
    stats["policy_loss"] += pg_loss.item()
    stats["baseline_loss"] += baseline_loss.item()
    stats["entropy_loss"] += entropy_loss.item()
    stats["kl_div"] += torch.mean(torch.sum(env_outputs["kl_div"], dim=0)).item()
    stats["kl_coef"] += learner_state.kl_controller.value
    stats["mean_reward"] += mean_reward
    stats["mean_kl_reward"] += mean_kl_reward
    stats["mean_total_reward"] += mean_total_reward
    stats["mean_response_length"] += np.mean(response_lengths)
    stats["median_response_length"] += np.median(response_lengths)

    stats["learning_rate"] += learner_state.lr_scheduler.get_last_lr()[0]


def step_optimizer(learner_state, accelerator, stats):
    optimizer = learner_state.optimizer
    model = learner_state.model

    if FLAGS.grad_norm_clipping:
        unclipped_grad_norm = nn.utils.clip_grad_norm_(model.parameters(), FLAGS.grad_norm_clipping).item()
    else:
        total_grad_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                total_grad_norm += torch.norm(param.grad).item()

        unclipped_grad_norm = math.sqrt(total_grad_norm)

    optimizer.step()

    if accelerator.sync_gradients:
        learner_state.model_version += 1
        learner_state.lr_scheduler.step()
        logging.info("Optimizer Stepped, Model Version %s", learner_state.model_version)
    else:
        logging.info("Optimizer faux stepped due to accumulation")

    stats["unclipped_grad_norm"] += unclipped_grad_norm
    stats["optimizer_steps"] += 1


def prepare_training_text_table(batch: Dict, learner_state: LearnerState) -> wandb.Table:
    columns = [
        "input",
        "response",
        "label",
        "reward",
        "kl_div",
        "kl_reward",
        "group",
        "model_version",
        "token_predictions",
    ]
    data = [
        batch["logging_outputs"]["text_in"],
        batch["logging_outputs"]["text_out"],
        batch["logging_outputs"]["labels"],
        batch["env_outputs"]["reward"].sum(dim=1).cpu().numpy(),
        batch["env_outputs"]["kl_div"].sum(dim=1).cpu().numpy(),
        batch["env_outputs"]["kl_rewards"].sum(dim=1).cpu().numpy(),
        [FLAGS.group] * len(batch["logging_outputs"]["text_in"]),
        [learner_state.model_version] * len(batch["logging_outputs"]["text_in"]),
        batch["actor_outputs"]["action"].cpu().numpy(),
    ]

    if FLAGS.dataset == "imdb":
        columns.append("sentiment")
        data.append(batch["logging_outputs"]["sentiment"])

    text_table = wandb.Table(columns=columns)

    for row in zip(*data):
        text_table.add_data(*row)

    return text_table


def prepare_eval_text_table(batch: Dict, learner_state: LearnerState) -> wandb.Table:
    columns = [
        "input",
        "response",
        "label",
        "reward",
        "kl_div",
        "kl_reward",
        "group",
        "model_version",
        "token_predictions",
    ]
    if FLAGS.dataset == "imdb":
        columns.append("sentiment")
    text_table = wandb.Table(columns=columns)
    data = [
        batch["inputs"],
        batch["predictions"],
        batch["references"],
        batch["rewards"],
        batch["kl_div"],
        batch["kl_rewards"],
        [FLAGS.group] * len(batch["inputs"]),
        [learner_state.model_version] * len(batch["inputs"]),  # type: ignore
        batch["token_predictions"],
    ]
    if FLAGS.dataset == "imdb":
        data.append(batch["logging_outputs"]["sentiment"])
    for row in zip(*data):
        text_table.add_data(*row)
    return text_table


def log(stats, model_version: Optional[int] = None, text_table: wandb.Table = None):
    stats_values = {}
    for k, v in stats.items():
        stats_values[k] = v.result()
        v.reset()

    if model_version is not None:
        stats_values["model_version"] = model_version

    logging.info(stats_values)
    record.log_to_file(localdir=FLAGS.localdir, **stats_values)

    if text_table is not None:
        stats_values["text_table"] = text_table

    if FLAGS.wandb:
        wandb.log(stats_values, step=stats["algo_steps"].result())


def do_evaluation(
    learner_state: LearnerState,
    env,
    tokenizer,
    step,
    splits: List[str] = ["train", "validation"],
    final: bool = False,
):
    target_eval_datapoints = FLAGS.target_eval_datapoints if final else FLAGS.training_target_eval_datapoints
    target_batches = math.ceil(target_eval_datapoints / FLAGS.batch_size)
    eval_batch = env.run_evaluation(num_evaluation_batches=target_batches, splits=splits)
    all_metrics = {}
    text_tables = {}

    for data_set, batch in eval_batch.items():
        metrics = compute_all_summarisation_metrics(
            **batch,
            tokenizer=tokenizer,
            return_text_table=False,
            return_distributions=False,
        )
        metrics = {f"{data_set}/{k}": v for k, v in metrics.items()}
        all_metrics.update(metrics)

        text_table = prepare_eval_text_table(batch, learner_state)
        text_tables[data_set] = text_table

    for k, v in all_metrics.items():
        local_stats[k] += v

    try:
        wandb.log(text_tables, step=step)
    except RequestException as e:
        logging.warning(
            "Failed to log to WANDB due to error %s with table json:\n%s",
            e,
            [text_table._to_table_json() for text_table in text_tables.values()],
        )


def save_checkpoint(checkpoint_path, accelerator, symlink_path: Optional[str] = None):
    tmp_path = "%s.tmp.%s" % (checkpoint_path, uuid.uuid4())
    accelerator.save_state(tmp_path)
    if os.path.exists(checkpoint_path):
        shutil.rmtree(checkpoint_path)
    os.replace(tmp_path, checkpoint_path)

    logging.info("Checkpoint saved to %s", checkpoint_path)

    if symlink_path:
        record.symlink_path(checkpoint_path, symlink_path, force=True)
        logging.info("Checkpoint symlinked (%s -> %s)", checkpoint_path, symlink_path)

    if FLAGS.checkpoint_limit:
        # List checkpoints in FLAGS.savedir, and delete the oldest ones
        checkpoints = glob.glob(os.path.join(FLAGS.savedir, "checkpoint_v*.tar"))
        checkpoints.sort(
            key=lambda path: int(path.split("/")[-1][len("checkpoint_v") : -len(".tar")].split("_")[0])
        )
        for checkpoint_path in checkpoints[: -FLAGS.checkpoint_limit]:
            logging.info("Deleting checkpoint %s", checkpoint_path)
            os.remove(checkpoint_path)


def calculate_sps(stats, delta, prev_steps):
    env_train_steps = stats["env_train_steps"].result()
    logging.info("calculate_sps %g steps in %g", env_train_steps - prev_steps, delta)
    stats["SPS"] += (env_train_steps - prev_steps) / delta
    return env_train_steps


def uid():
    return "%s:%i:%s" % (socket.gethostname(), os.getpid(), coolname.generate_slug(2))


def train_id():
    entity = FLAGS.entity if FLAGS.entity is not None else getpass.getuser()
    return "%s/%s/%s" % (entity, FLAGS.project, FLAGS.group)


def get_slurm_job_id():
    return os.environ.get("SLURM_ARRAY_JOB_ID")


def get_slurm_task_id():
    return os.environ.get("SLURM_ARRAY_TASK_ID")


omegaconf.OmegaConf.register_new_resolver("uid", uid, use_cache=True)
omegaconf.OmegaConf.register_new_resolver("slurm_job_id", get_slurm_job_id, use_cache=False)
omegaconf.OmegaConf.register_new_resolver("slurm_task_id", get_slurm_task_id, use_cache=False)


# Override config_path via --config_path.
@hydra.main(version_base=None, config_path="conf", config_name="config_accel")
def main(cfg):
    global FLAGS
    FLAGS = cfg

    log_gpu_utilization("Initial State:")

    FLAGS.seed = normalise_seed(FLAGS.seed)
    set_seed(FLAGS.seed)
    logging.info("cluster job id, cluster task id: %s, %s", FLAGS.cluster_job_id, FLAGS.cluster_task_id)

    if FLAGS.wandb:
        wandb.init(
            project=str(FLAGS.project),
            group=FLAGS.group,
            entity=FLAGS.entity,
            name=FLAGS.local_name,
            tags=FLAGS.wandb_tags,
            config=omegaconf.OmegaConf.to_container(FLAGS, resolve=True),
        )

    MAX_MODEL_VERSION = (
        FLAGS.total_steps
        * FLAGS.ppo_epochs
        * (FLAGS.rollout_batch_size / FLAGS.learn_batch_size)
        * (FLAGS.rollout_accumulation_steps / FLAGS.gradient_accumulation_steps)
    )

    original_cwd = hydra.utils.get_original_cwd()
    if not os.path.isabs(FLAGS.savedir):
        FLAGS.savedir = os.path.join(original_cwd, FLAGS.savedir)
    logging.info("flags:\n%s\n", pprint.pformat(dict(FLAGS)))
    logging.info("savedir: %s", FLAGS.savedir)
    if record.symlink_path(FLAGS.savedir, os.path.join(original_cwd, "latest")):
        logging.info("Symlinked savedir as 'latest'")

    project_config = ProjectConfiguration(project_dir=FLAGS.savedir)
    accelerator = Accelerator(
        project_dir=FLAGS.savedir,
        project_config=project_config,
        step_scheduler_with_optimizer=False,
        device_placement=False,
    )

    model, tokenizer, model_ref = construct_model_from_class(FLAGS)

    if FLAGS.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    log_gpu_utilization("Model Loaded:")

    if model_ref:
        if not model_ref.model_parallel:
            # We haven't handled devices through parallelism
            model_ref.to(FLAGS.ref_device)
        log_gpu_utilization("Reference Loaded:")
    torch.cuda.empty_cache()
    log_gpu_utilization("Model Loaded post cache clear:")

    optimizer = create_optimizer(model)
    lr_scheduler = create_lr_scheduler(optimizer, MAX_MODEL_VERSION)
    kl_controller = create_kl_controller()

    if FLAGS.dataset == "ni":
        dataloaders = get_ni_data_loaders(FLAGS, tokenizer)
    elif FLAGS.dataset == "imdb":
        dataloaders = get_imdb_data_loaders(FLAGS, tokenizer)
    elif FLAGS.dataset == "summarisation":
        dataloaders = get_summarisation_dataloader(FLAGS, tokenizer)
    else:
        raise ValueError("Unknown dataset %s" % FLAGS.dataset)

    reward_fn = make_reward_function(FLAGS.reward_function, FLAGS)

    if FLAGS.tie_frozen_layers and FLAGS.freeze_layers > 0:
        tie_frozen_layers(model, reward_fn.model)

    if FLAGS.parallelize:
        device_map = construct_device_map(
            torch.device(FLAGS.device).index,
            torch.device(FLAGS.rf_device).index,
            reward_fn.model,
            FLAGS.rm_split_percentage,
        )
        reward_fn.model.parallelize(device_map)
        reward_fn.device = FLAGS.device  # This is the device inputs are sent to

    model_layer_devices = get_layer_devices(accelerator.unwrap_model(model))
    model_ref_layer_devices = get_layer_devices(accelerator.unwrap_model(model_ref)) if model_ref else None
    reward_model_layer_devices = get_layer_devices(accelerator.unwrap_model(reward_fn.model))
    logging.info("Pre Accel Devices:")
    logging.info("Model layer devices: %s", model_layer_devices)
    logging.info("Model ref layer devices: %s", model_ref_layer_devices)
    logging.info("Reward model layer devices: %s", reward_model_layer_devices)

    log_gpu_utilization("Reward Function Loaded:")
    torch.cuda.empty_cache()
    log_gpu_utilization("Reward Function Loaded post cache clear:")

    accelerator.register_for_checkpointing(lr_scheduler)
    accelerator.register_for_checkpointing(kl_controller)
    optimizer, lr_scheduler, *prepared_dataloaders = accelerator.prepare(
        optimizer, lr_scheduler, *dataloaders.values()
    )
    accelerator._models.append(model)
    if accelerator.distributed_type == DistributedType.MULTI_GPU:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=None, output_device=None)
    for key, value in zip(dataloaders.keys(), prepared_dataloaders):
        dataloaders[key] = value
    log_gpu_utilization("Accelerator prepare:")
    torch.cuda.empty_cache()
    log_gpu_utilization("Accelrator prepare post cache clear:")

    model_layer_devices = get_layer_devices(accelerator.unwrap_model(model))
    model_ref_layer_devices = get_layer_devices(accelerator.unwrap_model(model_ref)) if model_ref else None
    reward_model_layer_devices = get_layer_devices(accelerator.unwrap_model(reward_fn.model))
    logging.info("Post Accel Devices:")
    logging.info("Model layer devices: %s", model_layer_devices)
    logging.info("Model ref layer devices: %s", model_ref_layer_devices)
    logging.info("Reward model layer devices: %s", reward_model_layer_devices)

    env = environment.MockEnv(
        dataloaders, tokenizer, model, model_ref, reward_fn, kl_controller, accelerator, FLAGS
    )
    log_gpu_utilization("Dataloaders Loaded:")

    model_numel = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info("Number of model parameters: %i", model_numel)
    record.write_metadata(
        FLAGS.localdir,
        original_cwd,
        flags=omegaconf.OmegaConf.to_container(FLAGS, resolve=True),
        model_numel=model_numel,
    )

    learner_state = LearnerState(model, optimizer, lr_scheduler, kl_controller, accelerator)
    accelerator.register_for_checkpointing(learner_state)

    logging.info("Using train_id: %s", train_id())

    checkpoint_path = os.path.join(FLAGS.savedir, "checkpoint_latest")

    if os.path.exists(checkpoint_path):
        logging.info("Loading checkpoint: %s" % checkpoint_path)
        with accelerator.main_process_first():
            model_device_map = accelerator.unwrap_model(model).device_map
            accelerator.load_state(checkpoint_path, map_location="cpu")
            accelerator.unwrap_model(model).parallelize(model_device_map)
        logging.info(
            "Resuming Run, starting from model version %s, algo_steps %s",
            learner_state.model_version,
            learner_state.algo_steps,
        )
        local_stats["algo_steps"] += learner_state.algo_steps - local_stats["algo_steps"].result()
        logging.info("LearnerState dict %s", learner_state.state_dict())

    terminate = False
    previous_signal_handler: dict = {}

    def signal_handler(signum, frame):
        nonlocal terminate
        logging.info(
            "Got signal %s, quitting!",
            signal.strsignal(signum) if hasattr(signal, "strsignal") else signum,
        )
        if signum == signal.SIGUSR1:
            logging.info("Got SIGUSR1")
            logging.info(f"accelerator.is_main_process: {accelerator.is_main_process}")
            if accelerator.is_main_process:
                logging.info("requeuing job " + os.environ["SLURM_JOB_ID"])
                os.system("scontrol requeue " + os.environ["SLURM_JOB_ID"])
        terminate = True
        previous_handler = previous_signal_handler.get(signum)
        if previous_handler is not None:
            previous_signal_handler[signum] = None
            signal.signal(signum, previous_handler)

    previous_signal_handler[signal.SIGTERM] = signal.signal(signal.SIGTERM, signal_handler)
    previous_signal_handler[signal.SIGINT] = signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGUSR1, signal_handler)  # register SIGUSR1

    if torch.backends.cudnn.is_available():
        logging.info("Optimising CuDNN kernels")
        torch.backends.cudnn.benchmark = True

    log_gpu_utilization("Spinning Up:")
    torch.cuda.empty_cache()
    log_gpu_utilization("Spinning Up post cache clean:")

    log_count = 0
    mb_queue: list = []
    latest_batch = None
    last_log = learner_state.algo_steps
    last_evaluation = (learner_state.algo_steps // FLAGS.evaluation_steps) * FLAGS.evaluation_steps
    last_checkpoint = (learner_state.algo_steps // FLAGS.checkpoint_steps) * FLAGS.checkpoint_steps

    def generate_rollouts():
        log_gpu_utilization("pre rollout:")
        local_mb_queue = []
        for _ in range(FLAGS.rollout_accumulation_steps):
            batch = env.train_rollout()
            local_stats["rollouts"] += FLAGS.rollout_batch_size
            local_mb_queue.append(batch)

            if terminate:
                break

        mb_queue_indices = np.arange(len(local_mb_queue))
        batch_indices = np.arange(FLAGS.rollout_batch_size)
        L = FLAGS.learn_batch_size
        for _ in range(FLAGS.ppo_epochs):
            np.random.shuffle(mb_queue_indices)
            for idx in mb_queue_indices:
                batch = local_mb_queue[idx]
                np.random.shuffle(batch_indices)
                for i in range(FLAGS.rollout_batch_size // FLAGS.learn_batch_size):
                    mb_queue.append(nest.index(batch, batch_indices[i * L : (i + 1) * L]))

                    if terminate:
                        break

        log_gpu_utilization("post rollout 1:")

        del batch
        latest_batch = mb_queue[0]
        log_gpu_utilization("post rollout 2:")
        torch.cuda.empty_cache()
        log_gpu_utilization("post rollout 3:")

        if (kl_div := local_stats["kl_div"].result()) is not None:
            learner_state.kl_controller.update(kl_div, FLAGS.rollout_batch_size)

        return latest_batch

    def apply_gradients():
        log_gpu_utilization("pre Optim Step:")
        step_optimizer(learner_state, accelerator, local_stats)
        optimizer.zero_grad()
        log_gpu_utilization("post Optim Step:")

    def calculate_gradients():
        log_gpu_utilization("pre learn:")
        compute_gradients(accelerator, mb_queue.pop(0), learner_state, local_stats, tokenizer)
        log_gpu_utilization("post learn:")
        torch.cuda.empty_cache()
        log_gpu_utilization("post learn cache:")

    logging.info("Setup complete, waiting")
    accelerator.wait_for_everyone()
    logging.info("Everyone is here")

    wandb.log({"process_index": accelerator.process_index, "is_main_process": accelerator.is_main_process})

    while not terminate:
        logging.info("Starting Iteration: %s", learner_state.algo_steps)
        logging.info("accelerator.is_main_process: %s", accelerator.is_main_process)
        logging.info("process index: %s", accelerator.process_index)
        logging.info("num processes: %s", accelerator.num_processes)
        logging.info("MB queue size: %s", len(mb_queue))

        if learner_state.algo_steps >= FLAGS.total_steps:
            logging.info("Stopping training after %i algo steps", learner_state.algo_steps)
            break

        # Useful as often model version is more robust than global/algo_steps, so we want to stop here too
        if learner_state.model_version > MAX_MODEL_VERSION:
            logging.info("Stopping training after reaching %i model_version", learner_state.model_version)
            break

        # Do evaluation
        if (learner_state.algo_steps - last_evaluation) >= FLAGS.evaluation_steps:
            logging.info("Evaluating model")
            do_evaluation(learner_state, env, tokenizer, step=learner_state.algo_steps)
            last_evaluation = learner_state.algo_steps

        # Do logging
        if (learner_state.algo_steps - last_log) >= FLAGS.log_steps:
            log_count += 1

            text_table = None
            if log_count % 10 == 0:
                batch = mb_queue[0] if mb_queue else latest_batch
                text_table = prepare_training_text_table(batch, learner_state) if batch else None

            try:
                log(local_stats, model_version=learner_state.model_version, text_table=text_table)
            except RequestException as e:
                logging.warning(
                    "Failed to log to WANDB due to error %s with table json:\n%s",
                    e,
                    text_table._to_table_json(),  # type: ignore
                )
                log(local_stats, model_version=learner_state.model_version)

            del latest_batch
            latest_batch = None
            last_log = learner_state.algo_steps

        # Do checkpointing
        if accelerator.is_main_process and (
            (learner_state.algo_steps - last_checkpoint) >= FLAGS.checkpoint_steps
        ):
            ckpt = "checkpoint_v%d_%d" % (learner_state.model_version, learner_state.algo_steps)
            save_checkpoint(os.path.join(FLAGS.savedir, ckpt), accelerator, symlink_path=checkpoint_path)
            last_checkpoint = learner_state.algo_steps

        # Main Training Loop
        del latest_batch
        latest_batch = generate_rollouts()
        logging.info("MB queue size after rollout generation: %s", len(mb_queue))
        i = 0
        while len(mb_queue) > 0:
            if terminate:
                break
            if (i + 1) % FLAGS.gradient_accumulation_steps != 0:
                with accelerator.no_sync(model):
                    calculate_gradients()
            else:
                calculate_gradients()
                apply_gradients()
            i += 1

        local_stats["algo_steps"] += 1
        learner_state.algo_steps += 1

    # Log to make sure everything works
    log(local_stats, model_version=learner_state.model_version)

    # Final checkpoint and evaluation
    if accelerator.is_main_process:
        ckpt = "checkpoint_v%d_%d" % (learner_state.model_version, learner_state.algo_steps)
        save_checkpoint(os.path.join(FLAGS.savedir, ckpt), accelerator, symlink_path=checkpoint_path)

    if not terminate:
        splits = ["train", "validation", "test"]
        if FLAGS.dataset_structured_subset:
            splits.extend(["full_validation", "ood_validation", "full_test", "ood_test"])
        evaluation_splits = FLAGS.evaluation_splits if FLAGS.evaluation_splits else None
        if evaluation_splits:
            splits = evaluation_splits
        do_evaluation(learner_state, env, tokenizer, step=learner_state.algo_steps, splits=splits, final=True)

        log(local_stats, model_version=learner_state.model_version)

    wandb.finish()
    logging.info("Graceful exit. Bye bye!")


if __name__ == "__main__":
    main()
