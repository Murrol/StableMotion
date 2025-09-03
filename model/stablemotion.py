# Copyright 2024 Stability AI and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, Optional, Union, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import (
    Attention,
    AttentionProcessor,
    StableAudioAttnProcessor2_0,
)
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import is_torch_version, logging
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.embeddings import get_1d_rotary_pos_embed
from diffusers.models.embeddings import Timesteps, TimestepEmbedding, LabelEmbedding

from einops import rearrange

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class AdaLayerNormSingle(nn.Module):
    r"""
    Adaptive LayerNorm (adaLN-single).

    Based on PixArt-Alpha (https://arxiv.org/abs/2310.00426, Sec. 2.3).

    Args:
        embedding_dim (int): Size of the embedding vector.
        use_additional_conditions (bool): Whether to use extra conditioning (e.g., class/mode).
        mode_indicator_dim (int): Factor to split `embedding_dim` when building mode embeddings.
    """

    def __init__(self, embedding_dim: int, use_additional_conditions: bool = False, mode_indicator_dim: int = 2):
        super().__init__()

        self.emb = CombinedTimestepModeEmbeddings(
            embedding_dim, 
            size_emb_dim=embedding_dim // mode_indicator_dim, 
            use_additional_conditions=use_additional_conditions,
        )

        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, bias=True)

    def forward(
        self,
        timestep: torch.Tensor,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        batch_size: Optional[int] = None,
        hidden_dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Embed timestep (plus optional extra conditions).
        embedded_timestep = self.emb(timestep, **added_cond_kwargs, batch_size=batch_size, hidden_dtype=hidden_dtype)
        # Project to modulation parameters.
        return self.linear(self.silu(embedded_timestep)), embedded_timestep

class CombinedTimestepModeEmbeddings(nn.Module):
    """
    Combine time embeddings with optional mode embeddings.

    Based on PixArt-Alpha:
    https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L164C9-L168C29

    Args:
        embedding_dim (int): Output embedding dimension.
        size_emb_dim (int): Dimension used for mode embeddings.
        use_additional_conditions (bool): If True, add mode embeddings on top of time embeddings.
        vs_modes (int): Number of possible mode labels (for LabelEmbedding).
    """

    def __init__(self, 
                 embedding_dim, 
                 size_emb_dim, 
                 use_additional_conditions: bool = False, 
                 vs_modes: int = 2, 
                 ):
        super().__init__()

        # Time embedding
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

        # Optional mode embedding
        self.use_additional_conditions = use_additional_conditions
        if use_additional_conditions:
            self.vis_mode_embedder = LabelEmbedding(vs_modes, size_emb_dim, dropout_prob=0.)

    def forward(self, timestep, batch_size, hidden_dtype, class_cond):
        # Time embedding
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_dtype))  # (B, D)

        # If extra conditions are used, add them to the time embedding
        if self.use_additional_conditions:
            vis_emb = self.vis_mode_embedder(class_cond)
            conditioning = timesteps_emb.unsqueeze(1) + rearrange(vis_emb, 'b l h d -> b l (h d)')
        else:
            conditioning = timesteps_emb.unsqueeze(1)

        return conditioning

@maybe_allow_in_graph
class StableMotionDiTBlock(nn.Module):
    r"""
    Transformer block adapted from Stable Audio (https://github.com/Stability-AI/stable-audio-tools).

    __init__ Args:
        dim (int): Model width; channels of input and output tokens.
        num_attention_heads (int): Number of self-attention heads.
        attention_head_dim (int): Size of each attention head.
        dropout (float, optional): Dropout used in attention/MLP. Default: 0.0.
        upcast_attention (bool, optional): Compute attention in float32 (helps stability in AMP).
        norm_eps (float, optional): Epsilon for LayerNorm. Default: 1e-5.
        ff_inner_dim (int, optional): Hidden size of the MLP. If None, uses module default.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        upcast_attention: bool = False,
        norm_eps: float = 1e-5,
        ff_inner_dim: Optional[int] = None,
    ):
        super().__init__()
        # Block 1: Self-Attention (norm -> attention -> residual)
        self.norm1 = nn.LayerNorm(dim, eps=norm_eps, elementwise_affine=False)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=False,
            upcast_attention=upcast_attention,
            out_bias=False,
            processor=StableAudioAttnProcessor2_0(),
        )

        # Block 2: Feed-Forward (norm -> MLP -> residual)
        self.norm2 = nn.LayerNorm(dim, eps=norm_eps, elementwise_affine=False)
        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn="swiglu",
            final_dropout=False,
            inner_dim=ff_inner_dim,
            bias=True,
        )

        # Time-conditioned scale/shift/gate for both blocks (6 slots total).
        self.scale_shift_table = nn.Parameter(torch.randn(1, 1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        rotary_embedding: Optional[torch.FloatTensor] = None,
        time_hidden_states: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        # Norm-first design; apply per-block scale/shift/gate from time_hidden_states.

        # 1) Self-Attention
        batch_size, seq_len = hidden_states.shape[:2]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table + time_hidden_states.reshape(batch_size, seq_len, 6, -1)
        ).chunk(6, dim=-2)

        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_msa.squeeze(-2)) + shift_msa.squeeze(-2)

        attn_output = self.attn1(
            norm_hidden_states,
            attention_mask=attention_mask,
            rotary_emb=rotary_embedding,
        )

        attn_output = gate_msa.squeeze(-2) * attn_output
        hidden_states = attn_output + hidden_states  # residual

        # 2) Feed-Forward
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp.squeeze(-2)) + shift_mlp.squeeze(-2)

        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.squeeze(-2) * ff_output

        hidden_states = ff_output + hidden_states  # residual

        return hidden_states
    

class StableMotionDiTModel(ModelMixin, ConfigMixin):
    """
    The Diffusion Transformer model adapted from Stable Audio.

    Reference: https://github.com/Stability-AI/stable-audio-tools

    Parameters:
        in_channels (`int`, *optional*, defaults to 64): The number of channels in the input.
        num_layers (`int`, *optional*, defaults to 24): The number of layers of Transformer blocks to use.
        attention_head_dim (`int`, *optional*, defaults to 64): The number of channels in each head.
        num_attention_heads (`int`, *optional*, defaults to 24): The number of heads to use for the query states.
        out_channels (`int`, defaults to 64): Number of output channels.
        time_proj_dim ( `int`, *optional*, defaults to 256): Dimension of the timestep inner projection.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: int = 64,
        num_layers: int = 24,
        attention_head_dim: int = 64,
        num_attention_heads: int = 24,
        out_channels: int = 64,
        time_proj_dim: int = 256,
        class_cond: bool=False,
        zero_init: bool=False,
        cond_index: list=[0, -1], #### HardCode inpaint_cond is full vector mask, we do not do joint inpainting
    ):
        super().__init__()

        self.class_cond = class_cond
        self.cond_index = cond_index
        self.mode_indicator_dim = len(self.cond_index)
        self.dropout = 0.
        self.zero_init = zero_init

        self.out_channels = out_channels
        self.in_channels = in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        self.rotary_embed_dim = attention_head_dim // 2

        self.transformer_blocks = nn.ModuleList(
            [
                StableMotionDiTBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    dropout=self.dropout,
                )
                for i in range(num_layers)
            ]
        )
        self.timestep_proj = nn.Sequential(
            nn.Linear(time_proj_dim, self.inner_dim, bias=True),
            nn.SiLU(),
            nn.Linear(self.inner_dim, self.inner_dim, bias=True),
        )

        self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.scale_shift_table = nn.Parameter(torch.randn(1, 1, 2, self.inner_dim) / self.inner_dim**0.5)

        self.adaln_single = AdaLayerNormSingle(
            self.inner_dim, 
            use_additional_conditions=self.class_cond,
            mode_indicator_dim=self.mode_indicator_dim,
        )
        self.in_channels += self.mode_indicator_dim
        
        self.preprocess_conv = nn.Conv1d(self.in_channels, self.in_channels, 1, bias=False) # Why 1*1 conv, same as linear, feature fusion?
        self.proj_in = nn.Linear(self.in_channels, self.inner_dim, bias=False)
        self.proj_out = nn.Linear(self.inner_dim, self.out_channels, bias=False)
        self.postprocess_conv = nn.Conv1d(self.out_channels, self.out_channels, 1, bias=False)
        self.gradient_checkpointing = False
        if self.zero_init:
            self.init_weights()

    def zero_module(self, module, mode='zero'):
        for p in module.parameters():
            if p.requires_grad:
                if mode == 'zero':
                    nn.init.zeros_(p)
                else:
                    raise NotImplementedError
        return module
    
    def init_weights(self):
        self.zero_module(nn.ModuleList([
            self.proj_out,
            self.postprocess_conv,
            ]))

    
    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    # Copied from diffusers.models.transformers.hunyuan_transformer_2d.HunyuanDiT2DModel.set_default_attn_processor with Hunyuan->StableAudio
    def set_default_attn_processor(self):
        """
        Disables custom attention processors and sets the default attention implementation.
        """
        self.set_attn_processor(StableAudioAttnProcessor2_0())

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        timestep: torch.LongTensor = None,
        global_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.LongTensor] = None,
        inpaint_cond: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        """
        The [`StableAudioDiTModel`] forward method.

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, in_channels, sequence_len)`):
                Input `hidden_states`.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, encoder_sequence_len, cross_attention_input_dim)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            global_hidden_states (`torch.FloatTensor` of shape `(batch size, global_sequence_len, global_states_input_dim)`):
               Global embeddings that will be prepended to the hidden states.
            rotary_embedding (`torch.Tensor`):
                The rotary embeddings to apply on query and key tensors during attention calculation.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_len)`, *optional*):
                Mask to avoid performing attention on padding token indices, formed by concatenating the attention
                masks
                    for the two text encoders together. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            encoder_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_len)`, *optional*):
                Mask to avoid performing attention on padding token cross-attention indices, formed by concatenating
                the attention masks
                    for the two text encoders together. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """

        if inpaint_cond is not None:
            inpaint_cond = inpaint_cond[:, self.cond_index] # Process inpaint_cond to get mode indicators, i.e. two modes

        hidden_states = torch.concat((hidden_states, inpaint_cond), dim=1) # (batch_size, dim+mode_indicator_dim, sequence_length)
        hidden_states = self.preprocess_conv(hidden_states) + hidden_states
        # (batch_size, dim, sequence_length) -> (batch_size, sequence_length, dim)
        hidden_states = hidden_states.transpose(1, 2)

        hidden_states = self.proj_in(hidden_states)
        
        batch_size = hidden_states.shape[0]
        added_cond_kwargs = {'class_cond': inpaint_cond.transpose(1, 2).long()} if self.class_cond else {}
        time_hidden_states, embedded_timestep = self.adaln_single(
            timestep, added_cond_kwargs, batch_size=batch_size, hidden_dtype=hidden_states.dtype, 
        )
        
        # Add pos embedding
        rotary_embedding = get_1d_rotary_pos_embed(
            self.rotary_embed_dim,
            hidden_states.shape[-2],
            use_real=True,
            repeat_interleave_real=False,
        )

        for block in self.transformer_blocks:
            if self.training and self.gradient_checkpointing:
                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    attention_mask,
                    None, #cross_attention_hidden_states,
                    None, #encoder_attention_mask,
                    rotary_embedding,
                    time_hidden_states
                    **ckpt_kwargs,
                )

            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=None, #cross_attention_hidden_states,
                    encoder_attention_mask=None, #encoder_attention_mask,
                    rotary_embedding=rotary_embedding,
                    time_hidden_states=time_hidden_states,
                )

        hidden_states = self.proj_out(hidden_states)

        # (batch_size, sequence_length, dim) -> (batch_size, dim, sequence_length)
        # remove prepend length that has been added by global hidden states
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.postprocess_conv(hidden_states) + hidden_states

        return hidden_states
    
class StableMotionDiTModel(ModelMixin, ConfigMixin):
    """
    Diffusion Transformer adapted from Stable Audio.

    Reference: https://github.com/Stability-AI/stable-audio-tools

    __init__ Args:
        in_channels (int, optional): Input channel count. Default: 64.
        num_layers (int, optional): Number of Transformer blocks. Default: 24.
        attention_head_dim (int, optional): Dimensionality per attention head. Default: 64.
        num_attention_heads (int, optional): Number of self-attention heads. Default: 24.
        out_channels (int, optional): Output channel count. Default: 64.
        time_proj_dim (int, optional): Inner dim for timestep projection MLP. Default: 256.
        class_cond (bool, optional): If True, enable class/mode conditioning in AdaLN. Default: False.
        zero_init (bool, optional): If True, zero-init selected output layers (helps stable starts). Default: False.
        cond_index (list, optional): Indices selecting mode channels from `inpaint_cond`. The length defines the
            number of mode indicators appended to inputs (i.e., `mode_indicator_dim`). Default: [0, -1].

    Notes:
        - `mode_indicator_dim = len(cond_index)`; these extra channels are concatenated to `hidden_states`.
        - Rotary embeddings are used for 1D sequences (`rotary_embed_dim = attention_head_dim // 2`).
        - `adaln_single` produces time/mode conditioning for each block; weights can be zero-initialized with `zero_init`.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: int = 64,
        num_layers: int = 24,
        attention_head_dim: int = 64,
        num_attention_heads: int = 24,
        out_channels: int = 64,
        time_proj_dim: int = 256,
        class_cond: bool=False,
        zero_init: bool=False,
        cond_index: list=[0, -1], #### HardCode inpaint_cond is full vector mask, we do not do joint inpainting
    ):
        super().__init__()

        self.class_cond = class_cond
        self.cond_index = cond_index
        self.mode_indicator_dim = len(self.cond_index)
        self.dropout = 0.
        self.zero_init = zero_init

        self.out_channels = out_channels
        self.in_channels = in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        self.rotary_embed_dim = attention_head_dim // 2

        # Stack of DiT blocks
        self.transformer_blocks = nn.ModuleList(
            [
                StableMotionDiTBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    dropout=self.dropout,
                )
                for i in range(num_layers)
            ]
        )

        # AdaLN head for time/mode conditioning
        self.adaln_single = AdaLayerNormSingle(
            self.inner_dim, 
            use_additional_conditions=self.class_cond,
            mode_indicator_dim=self.mode_indicator_dim,
        )

        # Append mode indicators to input channels
        self.in_channels += self.mode_indicator_dim
        
        # Lightweight 1x1 convs for pre/post fusion (equivalent to per-channel linear)
        self.preprocess_conv = nn.Conv1d(self.in_channels, self.in_channels, 1, bias=False)
        self.proj_in = nn.Linear(self.in_channels, self.inner_dim, bias=False)
        self.proj_out = nn.Linear(self.inner_dim, self.out_channels, bias=False)
        self.postprocess_conv = nn.Conv1d(self.out_channels, self.out_channels, 1, bias=False)

        # Training utilities
        self.gradient_checkpointing = False
        if self.zero_init:
            self.init_weights()

    def zero_module(self, module, mode='zero'):
        # Utility: zero-initialize a module's parameters (opt-in via `zero_init`).
        for p in module.parameters():
            if p.requires_grad:
                if mode == 'zero':
                    nn.init.zeros_(p)
                else:
                    raise NotImplementedError
        return module
    
    def init_weights(self):
        # Zero selected outputs for smoother early training (optional).
        self.zero_module(nn.ModuleList([
            self.proj_out,
            self.postprocess_conv,
            ]))

    
    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    # Copied from diffusers.models.transformers.hunyuan_transformer_2d.HunyuanDiT2DModel.set_default_attn_processor with Hunyuan->StableAudio
    def set_default_attn_processor(self):
        """
        Disables custom attention processors and sets the default attention implementation.
        """
        self.set_attn_processor(StableAudioAttnProcessor2_0())

    def _set_gradient_checkpointing(self, module, value=False):
        # Enable/disable grad checkpointing recursively where supported.
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        timestep: torch.LongTensor = None,
        global_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.LongTensor] = None,
        inpaint_cond: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        """
        Forward pass.

        Args:
            hidden_states (FloatTensor): Shape (B, in_channels, T). Input sequence features.
            timestep (LongTensor): Diffusion step index.
            global_hidden_states (FloatTensor, optional): Prepended global tokens (unused here).
            attention_mask (LongTensor, optional): Attention mask over sequence tokens.
            inpaint_cond (LongTensor, optional): Full vector mask; indexed by `cond_index` to form mode indicators.

        Returns:
            FloatTensor: Output features of shape (B, out_channels, T).
        """

        if inpaint_cond is not None:
            inpaint_cond = inpaint_cond[:, self.cond_index] # Process inpaint_cond to get mode indicators, i.e. two modes

        # Concatenate mode indicators and shallow fuse
        hidden_states = torch.concat((hidden_states, inpaint_cond), dim=1) # (batch_size, dim+mode_indicator_dim, sequence_length)
        hidden_states = self.preprocess_conv(hidden_states) + hidden_states

        # (B, C, T) -> (B, T, C)
        hidden_states = hidden_states.transpose(1, 2)

        # Project to model width
        hidden_states = self.proj_in(hidden_states)
        
        batch_size = hidden_states.shape[0]
        added_cond_kwargs = {'class_cond': inpaint_cond.transpose(1, 2).long()} if self.class_cond else {}

        # AdaLN: get time/mode conditioning and cache the raw embedding if needed
        time_hidden_states, embedded_timestep = self.adaln_single(
            timestep, added_cond_kwargs, batch_size=batch_size, hidden_dtype=hidden_states.dtype, 
        )
        
        # Rotary positional embedding for 1D sequence
        rotary_embedding = get_1d_rotary_pos_embed(
            self.rotary_embed_dim,
            hidden_states.shape[-2],
            use_real=True,
            repeat_interleave_real=False,
        )

        # Transformer stack
        for block in self.transformer_blocks:
            if self.training and self.gradient_checkpointing:
                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)
                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    attention_mask,
                    None, #cross_attention_hidden_states,
                    None, #encoder_attention_mask,
                    rotary_embedding,
                    time_hidden_states
                    **ckpt_kwargs,
                )
            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=None, #cross_attention_hidden_states,
                    encoder_attention_mask=None, #encoder_attention_mask,
                    rotary_embedding=rotary_embedding,
                    time_hidden_states=time_hidden_states,
                )

        # Project back to output channels
        hidden_states = self.proj_out(hidden_states)

        # (B, T, C) -> (B, C, T), then shallow fuse
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.postprocess_conv(hidden_states) + hidden_states

        return hidden_states