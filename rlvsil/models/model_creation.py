"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import copy
import inspect
import logging
from dataclasses import dataclass
from enum import Enum
from packaging import version
from typing import Any, Dict, Optional, Tuple, Union

import torch
import transformers
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import (AutoConfig, AutoModel, AutoTokenizer,
                          LlamaPreTrainedModel, LlamaTokenizer,
                          PreTrainedModel, PreTrainedTokenizer)
from transformers.modeling_outputs import ModelOutput
from transformers.models.llama.modeling_llama import (LlamaModel,
                                                      LlamaPreTrainedModel)
from transformers.models.opt.modeling_opt import OPTModel
from transformers.utils.model_parallel_utils import (assert_device_map,
                                                     get_device_map)

from models.llama_parallel import LlamaParallelModel
from models.opt_parallel import OPTParallelModel
from models.value_functions import ValueHead
from optimum.bettertransformer import BetterTransformer

__all__ = ["CausalLMOutputWithCrossAttentions", "ValueHead", "make_causal_lm_value_model"]


class LLaMATokenizer(LlamaTokenizer):
    def __init__(
        self,
        vocab_file,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token=None,
        sp_model_kwargs: Union[Dict[str, Any], None] = None,
        add_bos_token=True,
        add_eos_token=False,
        clean_up_tokenization_spaces=False,
        **kwargs,
    ):
        super().__init__(
            vocab_file,
            unk_token,
            bos_token,
            eos_token,
            pad_token,
            sp_model_kwargs,
            add_bos_token,
            add_eos_token,
            clean_up_tokenization_spaces,
            **kwargs,
        )


transformers.LLaMATokenizer = LLaMATokenizer


@dataclass
class CausalLMOutputWithCrossAttentions(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    value: Optional[torch.FloatTensor] = None


@dataclass
class RFModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class ModelPurpose(Enum):
    """Enum for RL, SL or RF model purposes."""

    RL = "rl"
    SL = "sl"
    RF = "rf"


def __init__(
    self,
    config,
    transformer_class: type,
    super_class: type,
    is_opt: bool,
    model_purpose: ModelPurpose,
    value_normalisation: float,
    value_normalisation_std: float,
):
    super(super_class, self).__init__(config)  # type: ignore

    self.super_class = super_class
    self.model_purpose = model_purpose

    config.num_labels = 1

    self.is_opt = is_opt
    if self.is_opt or super_class == LlamaPreTrainedModel:
        self.model = transformer_class(config)
        self.base_model_prefix = "model"
        self.base_model_layers = "layers"
    else:
        self.transformer = transformer_class(config)
        self.base_model_layers = "h"

    self.rf_batched = False  # disabling for now as it's broken

    # OPT doesn't use n_embd but rather word_embed_proj_dim, so catch that, and also for GPT Neo
    n_embd = (
        config.n_embd
        if getattr(config, "n_embd", None) is not None
        else config.word_embed_proj_dim
        if getattr(config, "word_embed_proj_dim", None) is not None
        else config.hidden_size
    )

    self.v_head = ValueHead(n_embd, value_normalisation, value_normalisation_std, config)
    self.lm_head = nn.Linear(n_embd, config.vocab_size, bias=False)

    # Model parallel
    self.model_parallel = False
    self.device_map = None

    self.post_init()


def get_base_model(self) -> nn.Module:
    return getattr(self, self.base_model_prefix)


def get_base_model_transformer(self) -> nn.Module:
    if self.is_opt:
        return self.get_base_model().decoder
    return self.get_base_model()


def remove_unneeded_args(self, kwargs):
    signature = inspect.signature(self.get_base_model().forward)
    self._signature_columns = list(signature.parameters.keys())
    model_kwargs = kwargs
    extra_kwargs = {}
    for key in list(model_kwargs.keys()):
        if key not in self._signature_columns:
            extra_kwargs[key] = model_kwargs[key]
            del model_kwargs[key]

    return model_kwargs, extra_kwargs


def forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.LongTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    *args,
    **kwargs,
):
    model_kwargs, extra_kwargs = self.remove_unneeded_args(kwargs)
    transformer_outputs = self.get_base_model()(
        input_ids=input_ids,
        attention_mask=attention_mask,
        *args,
        **model_kwargs,
    )

    hidden_states = transformer_outputs[0]

    # Set device for model parallelism
    if self.model_parallel:
        device = self.get_base_model_transformer().first_device
        torch.cuda.set_device(device)
        hidden_states = hidden_states.to(self.lm_head.weight.device)

    value = self.v_head(hidden_states).squeeze(-1)
    lm_logits = self.lm_head(hidden_states) if self.model_purpose != ModelPurpose.RF else None

    # Language Modelling loss calculation (for SL, not RL)
    loss = None
    if labels is not None and self.model_purpose == ModelPurpose.SL:
        # Shift so that tokens < n predict n
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    return CausalLMOutputWithCrossAttentions(
        loss=loss,
        logits=lm_logits,
        past_key_values=transformer_outputs.past_key_values,
        hidden_states=transformer_outputs.hidden_states,
        attentions=transformer_outputs.attentions,
        cross_attentions=getattr(transformer_outputs, "cross_attentions", None),
        value=value,
    )


def calculate_reward(self, input_ids, attention_mask) -> torch.Tensor:
    """Calculate scalar reward (i.e. value of eos token) across a batch of inputs."""
    hidden_states = self.get_base_model()(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )[0]

    if self.model_parallel:
        device = self.get_base_model_transformer().first_device
        torch.cuda.set_device(device)
        hidden_states = hidden_states.to(self.v_head.linear.weight.device)

    per_token_rewards = self.v_head(hidden_states)

    batch_size = input_ids.shape[0]

    sequence_lengths = torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1
    return per_token_rewards[torch.arange(batch_size, device=per_token_rewards.device), sequence_lengths]


def rf_forward(
    self,
    input_0_input_ids,
    input_0_attention_mask,
    input_1_input_ids,
    input_1_attention_mask,
    labels=None,
    *args,
    **kwargs,
) -> RFModelOutput:
    """Calculate self.rf_model on both input pairs, compute logits and return results."""
    if False:  # Ignore for now as it doesn't work
        try:
            rewards_0_1 = self.calculate_reward(
                torch.cat([input_0_input_ids, input_1_input_ids], dim=0),
                torch.cat([input_0_attention_mask, input_1_attention_mask], dim=0),
            )
            logits = torch.reshape(rewards_0_1, (2, input_0_input_ids.shape[0])).T
        except RuntimeError as e:
            if "CUDA" in str(e):
                # We OOMed on the above calculation, lets try again one by one
                logging.info(e)
                self.rf_batched = False
                torch.cuda.empty_cache()
            else:
                raise

    if not self.rf_batched:
        rewards_0 = self.calculate_reward(input_0_input_ids, input_0_attention_mask)
        rewards_1 = self.calculate_reward(input_1_input_ids, input_1_attention_mask)

        logits = torch.stack([rewards_0, rewards_1], dim=1)

    loss = None
    if labels is not None:
        loss = nn.CrossEntropyLoss()(logits, labels)

    return RFModelOutput(
        loss=loss,
        logits=logits,
    )


def get_output_embeddings(self):
    return self.lm_head


def set_output_embeddings(self, new_embeddings):
    self.lm_head = new_embeddings


def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
    token_type_ids = kwargs.get("token_type_ids", None)
    # only last token for inputs_ids if past is defined in kwargs
    if past:
        input_ids = input_ids[:, -1].unsqueeze(-1)
        if token_type_ids is not None:
            token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

    attention_mask = kwargs.get("attention_mask", None)
    position_ids = kwargs.get("position_ids", None)

    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past:
            position_ids = position_ids[:, -1].unsqueeze(-1)
    else:
        position_ids = None

    return {
        "input_ids": input_ids,
        "past_key_values": past,
        "use_cache": kwargs.get("use_cache"),
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    }


def _init_weights(self, module):
    """Initialize the weights."""
    if module._get_name() == "ValueHead":
        module.weight.data.normal_(mean=0.0, std=1.0 / (module.linear.in_features + 1))
        if module.bias is not None:
            module.bias.data.zero_()
    else:
        super(self.super_class, self)._init_weights(module)


def parallelize(self, device_map=None):
    device_map_len = len(getattr(self.get_base_model_transformer(), self.base_model_layers))
    self.device_map = (
        get_device_map(
            device_map_len,
            range(torch.cuda.device_count()),
        )
        if device_map is None
        else device_map
    )
    assert_device_map(self.device_map, device_map_len)
    self.get_base_model().parallelize(self.device_map)
    self.lm_head = self.lm_head.to(self.get_base_model_transformer().first_device)
    self.v_head = self.v_head.to(self.get_base_model_transformer().first_device)
    self.model_parallel = True


def deparallelize(self):
    self.get_base_model().deparallelize()
    setattr(self, self.base_model_prefix, self.get_base_model().to("cpu"))
    self.lm_head = self.lm_head.to("cpu")
    self.v_head = self.v_head.to("cpu")
    self.model_parallel = False
    self.device_map = None
    torch.cuda.empty_cache()


_keys_to_ignore_on_load_missing = [r"attn.masked_bias", r"attn.bias", r"lm_head.weight", r"v_head"]


def make_causal_lm_value_model(
    transformer_class: type,
    pretrained_model_class: type,
    is_opt: bool,
    model_checkpoint: str,
    model_purpose: str,
    value_normalisation: float,
    value_normalisation_std: float,
    *args,
    **kwargs,
):
    """Make a new class and instantiate it, for a causal LM with value head.

    For example:

    >>> config = AutoConfig.from_pretrained('gpt2')
    >>> model = make_causal_lm_value_model(GPT2Model, GPT2PretrainedModel, "gpt2", config=config)
    >>> type(model)
    <class 'models.model_creation.GPT2CausalLMValueModel'>
    >>> model.v_head
    <class 'models.value_functions.ValueHead'>

    """
    new_class_name = (
        transformer_class.__name__.replace("Model", "").replace("Decoder", "") + "CausalLMValueModel"
    )
    NewClass = type(
        new_class_name,
        (pretrained_model_class,),
        {
            "__init__": __init__,
            "calculate_reward": calculate_reward,
            "forward": rf_forward if model_purpose is ModelPurpose.RF else forward,
            "remove_unneeded_args": remove_unneeded_args,
            "get_output_embeddings": get_output_embeddings,
            "set_output_embeddings": set_output_embeddings,
            "prepare_inputs_for_generation": prepare_inputs_for_generation,
            "_keys_to_ignore_on_load_missing": _keys_to_ignore_on_load_missing,
            "get_base_model": get_base_model,
            "get_base_model_transformer": get_base_model_transformer,
            "parallelize": parallelize,
            "deparallelize": deparallelize,
        },
    )
    return NewClass.from_pretrained(  # type: ignore
        model_checkpoint,
        transformer_class,
        pretrained_model_class,
        is_opt,
        model_purpose,
        value_normalisation,
        value_normalisation_std,
        use_auth_token=True,
        *args,
        **kwargs,
    )


def update_model_config(config, args):
    for key in config.__dict__:
        if v := getattr(args, key, None):
            setattr(config, key, v)
    return config


def get_model_classes(model_config):
    is_opt = False
    transformer_class = AutoModel._model_mapping[type(model_config)]
    pretrained_model_cls = transformer_class.__bases__[0]
    if transformer_class == OPTModel:
        is_opt = True
        transformer_class = OPTParallelModel
    if transformer_class == LlamaModel:
        transformer_class = LlamaParallelModel
    return transformer_class, pretrained_model_cls, is_opt


def get_model_purpose(args):
    if (rl_training := getattr(args, "rl_training", None)) is not None:
        return ModelPurpose.RL if rl_training else ModelPurpose.SL
    else:
        return ModelPurpose.RF


def tie_frozen_layers(model: PreTrainedModel, model_tie: PreTrainedModel):
    """Tie weights of model_tie to model for frozen layers."""
    for param, param_ref in zip(model.get_base_model().parameters(), model_tie.get_base_model().parameters()):
        if not param.requires_grad and not param_ref.requires_grad:
            param_ref.data = param.data
            param_ref.requires_grad = False


def construct_device_map(
    device_1: int, device_2: int, model: PreTrainedModel, device_1_layer_percentage: float
) -> Dict[int, list]:
    transformer = model.get_base_model_transformer()
    layers = getattr(transformer, model.base_model_layers)
    device_1_layers = int(len(layers) * device_1_layer_percentage)
    if device_1 == device_2 or device_1_layers == len(layers):
        device_map = {device_1: list(range(len(layers)))}
    else:
        device_map = {
            device_1: list(range(device_1_layers)),
            device_2: list(range(device_1_layers, len(layers))),
        }
    return device_map


def freeze_layers(model, freeze_layers: float, model_purpose: ModelPurpose, freeze_lm_head: bool):
    """Freeze layers of model."""
    transformer = model.get_base_model_transformer()
    layers = getattr(transformer, model.base_model_layers)
    if layers:
        freeze_until = int(len(layers) * freeze_layers)
        logging.info(f"Freezing {freeze_until} layers")
        for param in layers[:freeze_until].parameters():
            param.requires_grad = False

    if (embed_positions := getattr(transformer, "embed_positions", None)) is not None:
        for parameter in embed_positions.parameters():
            parameter.requires_grad = False

    if model_purpose is ModelPurpose.RF or freeze_lm_head:
        for parameter in transformer.embed_tokens.parameters():
            parameter.requires_grad = False
        for parameter in model.lm_head.parameters():
            parameter.requires_grad = False


def construct_model_from_class(
    args,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer, Optional[PreTrainedModel]]:
    model_config = AutoConfig.from_pretrained(args.model_name, use_auth_token=True)
    model_config = update_model_config(model_config, args)
    model_config.value_head_activation = args.value_head_activation

    transformer_class, pretrained_model_cls, is_opt = get_model_classes(model_config)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=not is_opt, use_auth_token=True)
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if not model_config.pad_token_id:
        model_config.pad_token_id = tokenizer.eos_token_id

    model_purpose = get_model_purpose(args)

    model = make_causal_lm_value_model(
        transformer_class,
        pretrained_model_cls,
        is_opt,
        args.model_name,
        model_purpose=model_purpose,
        config=model_config,
        value_normalisation=args.value_normalisation,
        value_normalisation_std=args.value_normalisation_std,
    )
    if args.bettertransformer:
        model = BetterTransformer.transform(model)

    if args.torchcompile and version.parse(torch.__version__) > version.parse("2.0.0"):
        model = torch.compile(model)

    if args.freeze_layers > 0:
        freeze_layers(model, args.freeze_layers, model_purpose, args.freeze_lm_head)

    model_ref = None

    if model_purpose == ModelPurpose.RL:
        if args.model_name != args.base_model_name:
            model_ref = model.from_pretrained(
                args.base_model_name,
                transformer_class,
                pretrained_model_cls,
                is_opt,
                model_purpose,
                config=model_config,
                value_normalisation=args.value_normalisation,
                value_normalisation_std=args.value_normalisation_std,
                use_auth_token=True,
            )
            if args.bettertransformer:
                model_ref = BetterTransformer.transform(model_ref)
        else:
            model_ref = copy.deepcopy(model)

        freeze_layers(model_ref, 1.0, model_purpose, True)

    if args.parallelize:
        device_map = construct_device_map(
            torch.device(args.device).index,
            torch.device(args.policy_head_device).index,
            model,
            args.policy_split_percentage,
        )
        model.parallelize(device_map)

    if model_ref is not None:
        if args.tie_frozen_layers and args.freeze_layers > 0:
            logging.info("tieing layers")
            tie_frozen_layers(model, model_ref)
            torch.cuda.empty_cache()

        if args.parallelize:
            ref_device_map = construct_device_map(
                torch.device(args.device).index,
                torch.device(args.ref_device).index,
                model_ref,
                args.ref_split_percentage,
            )
            model_ref.parallelize(ref_device_map)

    torch.cuda.empty_cache()

    return model, tokenizer, model_ref
