"""Paper-guided sequential MoT core for MemoryWAM.

This is an independent reimplementation on FastWAM, not official MemoryWAM
source.  FastWAM supplies the video/action DiT blocks.  The paper specifies
the hybrid retention policy and shared 3-D positional frame.  Same-frame
clean/gist mutual visibility, the gist marker, and action coordinates follow
the approved implementation inferences recorded in ``docs/upstream_architecture.md``.

The public training boundary consumes already embedded per-frame tokens.  It
is deliberately below the VAE and expert input/output heads: episode-level
VAE packing and loss assembly are separate integration work.  This module
does implement the stateful attention semantics used by that future path.
"""

from __future__ import annotations

from dataclasses import dataclass
import weakref
from collections.abc import Sequence

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .memory_cache import (
    CacheSnapshot,
    MemoryWAMLayerKVCache,
    NUM_ACTION_TOKENS,
    NUM_ANCHOR_FRAMES,
    NUM_GIST_TOKENS,
    NUM_RECENT_FRAMES,
    SharedGistTokens,
    action_coordinates,
    gist_coordinates,
    video_coordinates,
)
from .wan_video_dit import flash_attention, modulate, rope_apply


def rope_frequencies_from_coordinates(
    rope_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    coordinates: torch.Tensor,
) -> torch.Tensor:
    """Index Wan's 3-D RoPE tables with absolute ``(frame, row, column)``."""

    if len(rope_cache) != 3:
        raise ValueError("`rope_cache` must contain frame, row, and column tables.")
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError(
            f"`coordinates` must have shape [S,3], got {tuple(coordinates.shape)}."
        )
    if coordinates.dtype not in (torch.int32, torch.int64):
        raise TypeError("`coordinates` must use an integer dtype.")
    if coordinates.numel() and int(coordinates.min().item()) < 0:
        raise ValueError("3-D RoPE coordinates must be non-negative.")

    parts = []
    for axis, table in enumerate(rope_cache):
        positions = coordinates[:, axis].to(device=table.device, dtype=torch.long)
        if positions.numel() and int(positions.max().item()) >= table.shape[0]:
            raise ValueError(
                f"RoPE axis {axis} coordinate {int(positions.max().item())} exceeds "
                f"the cached range {table.shape[0]}."
            )
        parts.append(table.index_select(0, positions))
    return torch.cat(parts, dim=-1).unsqueeze(1)


def sample_conditioning_noise_ratio(
    batch_size: int,
    *,
    device: torch.device | str,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample the paper's uniform conditioning corruption ratio in ``[0,1]``."""

    if batch_size <= 0:
        raise ValueError(f"`batch_size` must be positive, got {batch_size}.")
    return torch.rand(
        (batch_size,), device=device, dtype=torch.float32, generator=generator
    )


def corrupt_conditioning_latents(
    clean_latents: torch.Tensor,
    ratio: torch.Tensor,
    *,
    num_train_timesteps: int = 1000,
    noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mix conditioning latents with noise and return their modulation timestep.

    The returned timestep is ``ratio * num_train_timesteps``.  Passing the
    actual continuous ratio through this timestep is the approved
    implementation inference; a binary clean/noisy indicator is not used.
    """

    if clean_latents.ndim < 2:
        raise ValueError("`clean_latents` must have a batch dimension.")
    if ratio.shape != (clean_latents.shape[0],):
        raise ValueError(
            "`ratio` must have shape [B], got "
            f"{tuple(ratio.shape)} for batch {clean_latents.shape[0]}."
        )
    ratio_float = ratio.to(device=clean_latents.device, dtype=torch.float32)
    if not bool(torch.isfinite(ratio_float).all()) or bool(
        ((ratio_float < 0) | (ratio_float > 1)).any()
    ):
        raise ValueError("`ratio` must be finite and lie in [0,1].")
    if num_train_timesteps <= 0:
        raise ValueError("`num_train_timesteps` must be positive.")
    if noise is None:
        noise = torch.randn_like(clean_latents)
    if noise.shape != clean_latents.shape:
        raise ValueError("`noise` must match `clean_latents` exactly.")

    mix = ratio_float.to(dtype=clean_latents.dtype).view(
        -1, *([1] * (clean_latents.ndim - 1))
    )
    corrupted = (1 - mix) * clean_latents + mix * noise
    timestep = ratio_float.mul(float(num_train_timesteps)).to(clean_latents.dtype)
    return corrupted, timestep


def build_current_frame_attention_mask(
    *,
    prefix_tokens: int,
    clean_video_tokens: int,
    gist_tokens: int,
    noisy_video_tokens: int,
    action_tokens: int,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Build the exact current-frame query mask used by cached and training paths.

    Clean and gist queries share the historical prefix and see one another.

    The noisy-video target is the same frame as the clean conditioning block,
    exactly as released FastWAM pairs ``latents_noisy`` and ``latents_cond``
    from one ``input_latents`` tensor.  It therefore sees only the retained
    history and its own block; an edge into current clean/gist would expose the
    frame it must denoise, and a video rollout has no such block at inference.

    The action target does see current clean/gist K/V, matching both FastWAM's
    ``action -> cond`` edge and the deployed policy, which conditions on the
    observation it has just received.  Neither target branch sees the other and
    no historical action key exists.
    """

    counts = (
        prefix_tokens,
        clean_video_tokens,
        gist_tokens,
        noisy_video_tokens,
        action_tokens,
    )
    if any(
        isinstance(count, bool) or not isinstance(count, int) or count < 0
        for count in counts
    ):
        raise ValueError(f"Token counts must be non-negative integers, got {counts}.")
    if clean_video_tokens <= 0:
        raise ValueError("A frame must contain at least one clean video token.")
    if gist_tokens != NUM_GIST_TOKENS:
        raise ValueError(
            f"MemoryWAM requires {NUM_GIST_TOKENS} gist tokens, got {gist_tokens}."
        )
    if action_tokens not in (0, NUM_ACTION_TOKENS):
        raise ValueError(
            f"Action blocks must be empty or contain {NUM_ACTION_TOKENS} tokens."
        )

    prefix_end = prefix_tokens
    clean_end = prefix_end + clean_video_tokens
    gist_end = clean_end + gist_tokens
    noisy_end = gist_end + noisy_video_tokens
    key_count = noisy_end + action_tokens
    query_count = sum(counts[1:])
    mask = torch.zeros((query_count, key_count), dtype=torch.bool, device=device)

    clean_gist_queries = clean_video_tokens + gist_tokens
    mask[:clean_gist_queries, :gist_end] = True
    if noisy_video_tokens:
        noisy_rows = slice(clean_gist_queries, clean_gist_queries + noisy_video_tokens)
        mask[noisy_rows, :prefix_end] = True
        mask[noisy_rows, gist_end:noisy_end] = True
    if action_tokens:
        action_rows = slice(clean_gist_queries + noisy_video_tokens, query_count)
        mask[action_rows, :gist_end] = True
        mask[action_rows, noisy_end:key_count] = True
    return mask


def _split_modulation(block: nn.Module, t_mod: torch.Tensor):
    has_sequence_axis = t_mod.ndim == 4
    chunk_dim = 2 if has_sequence_axis else 1
    values = (
        block.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
    ).chunk(6, dim=chunk_dim)
    if has_sequence_axis:
        values = tuple(value.squeeze(2) for value in values)
    return values


def _expand_context_mask(mask: torch.Tensor, query_tokens: int) -> torch.Tensor:
    if mask.ndim == 2:
        mask = mask.unsqueeze(1).expand(-1, query_tokens, -1)
    elif mask.ndim == 3:
        if mask.shape[1] == 1:
            mask = mask.expand(-1, query_tokens, -1)
        elif mask.shape[1] != query_tokens:
            raise ValueError(
                f"Context mask has {mask.shape[1]} query rows, expected {query_tokens}."
            )
    else:
        raise ValueError("Context masks must have shape [B,L] or [B,S,L].")
    return mask.unsqueeze(1)


@dataclass(frozen=True)
class _AttentionIO:
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    residual: torch.Tensor
    gate_msa: torch.Tensor
    shift_mlp: torch.Tensor
    scale_mlp: torch.Tensor
    gate_mlp: torch.Tensor


class MemoryWAMPairedLayer(nn.Module):
    """One FSDP/checkpoint boundary owning a video/action block pair."""

    def __init__(self, video_block: nn.Module, action_block: nn.Module):
        super().__init__()
        if int(video_block.num_heads) != int(action_block.num_heads):
            raise ValueError(
                "Paired blocks must use the same number of attention heads."
            )
        if int(video_block.attn_head_dim) != int(action_block.attn_head_dim):
            raise ValueError(
                "Paired blocks must use the same attention head dimension."
            )
        self.video_block = video_block
        self.action_block = action_block
        self.num_heads = int(video_block.num_heads)
        self.attention_dim = self.num_heads * int(video_block.attn_head_dim)

    @staticmethod
    def _attention_io(
        block: nn.Module,
        tokens: torch.Tensor,
        freqs: torch.Tensor,
        t_mod: torch.Tensor,
    ) -> _AttentionIO:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            _split_modulation(block, t_mod)
        )
        attention_input = modulate(block.norm1(tokens), shift_msa, scale_msa)
        query = block.self_attn.norm_q(block.self_attn.q(attention_input))
        key = block.self_attn.norm_k(block.self_attn.k(attention_input))
        value = block.self_attn.v(attention_input)
        return _AttentionIO(
            query=rope_apply(query, freqs, block.num_heads),
            key=rope_apply(key, freqs, block.num_heads),
            value=value,
            residual=tokens,
            gate_msa=gate_msa,
            shift_mlp=shift_mlp,
            scale_mlp=scale_mlp,
            gate_mlp=gate_mlp,
        )

    @staticmethod
    def _post(
        block: nn.Module,
        state: _AttentionIO,
        attention_output: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        tokens = block.gate(
            state.residual,
            state.gate_msa,
            block.self_attn.o(attention_output),
        )
        tokens = tokens + block.cross_attn(
            block.norm3(tokens),
            context,
            ctx_mask=_expand_context_mask(context_mask, tokens.shape[1]),
        )
        mlp_input = modulate(block.norm2(tokens), state.shift_mlp, state.scale_mlp)
        return block.gate(tokens, state.gate_mlp, block.ffn(mlp_input))

    def forward(
        self,
        clean_video_tokens: torch.Tensor,
        gist_tokens: torch.Tensor,
        noisy_video_tokens: torch.Tensor,
        action_tokens: torch.Tensor,
        clean_video_freqs: torch.Tensor,
        gist_freqs: torch.Tensor,
        noisy_video_freqs: torch.Tensor,
        action_freqs: torch.Tensor,
        clean_video_t_mod: torch.Tensor,
        gist_t_mod: torch.Tensor,
        noisy_video_t_mod: torch.Tensor,
        action_t_mod: torch.Tensor,
        video_context: torch.Tensor,
        video_context_mask: torch.Tensor,
        action_context: torch.Tensor,
        action_context_mask: torch.Tensor,
        prefix_key: torch.Tensor,
        prefix_value: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Process one frame without mutating cache state.

        Cache insertion happens in :class:`MemoryWAMSequentialCore` after this
        pure tensor call, which makes activation recomputation safe.
        """

        if clean_video_tokens.shape[1] <= 0:
            raise ValueError("`clean_video_tokens` cannot be empty.")
        if gist_tokens.shape[1] != NUM_GIST_TOKENS:
            raise ValueError(f"Expected {NUM_GIST_TOKENS} gist tokens.")
        if action_tokens.shape[1] not in (0, NUM_ACTION_TOKENS):
            raise ValueError(f"Expected zero or {NUM_ACTION_TOKENS} action tokens.")
        if prefix_key.shape != prefix_value.shape:
            raise ValueError("Prefix K/V shapes must match.")
        if prefix_key.ndim != 3 or prefix_key.shape[2] != self.attention_dim:
            raise ValueError(
                "Prefix K/V must have shape [B,P,H*Dh] with attention width "
                f"{self.attention_dim}."
            )

        clean = self._attention_io(
            self.video_block, clean_video_tokens, clean_video_freqs, clean_video_t_mod
        )
        gist = self._attention_io(self.video_block, gist_tokens, gist_freqs, gist_t_mod)
        noisy = (
            self._attention_io(
                self.video_block,
                noisy_video_tokens,
                noisy_video_freqs,
                noisy_video_t_mod,
            )
            if noisy_video_tokens.shape[1]
            else None
        )
        action = (
            self._attention_io(
                self.action_block, action_tokens, action_freqs, action_t_mod
            )
            if action_tokens.shape[1]
            else None
        )

        states = [clean, gist]
        if noisy is not None:
            states.append(noisy)
        if action is not None:
            states.append(action)
        query = torch.cat([state.query for state in states], dim=1)
        key = torch.cat(
            [prefix_key, clean.key, gist.key]
            + ([] if noisy is None else [noisy.key])
            + ([] if action is None else [action.key]),
            dim=1,
        )
        value = torch.cat(
            [prefix_value, clean.value, gist.value]
            + ([] if noisy is None else [noisy.value])
            + ([] if action is None else [action.value]),
            dim=1,
        )
        attention_mask = build_current_frame_attention_mask(
            prefix_tokens=prefix_key.shape[1],
            clean_video_tokens=clean_video_tokens.shape[1],
            gist_tokens=gist_tokens.shape[1],
            noisy_video_tokens=noisy_video_tokens.shape[1],
            action_tokens=action_tokens.shape[1],
            device=query.device,
        )
        attended = flash_attention(
            q=query,
            k=key,
            v=value,
            num_heads=self.num_heads,
            ctx_mask=attention_mask,
        )

        cursor = 0
        clean_end = cursor + clean_video_tokens.shape[1]
        clean_out = self._post(
            self.video_block,
            clean,
            attended[:, cursor:clean_end],
            video_context,
            video_context_mask,
        )
        cursor = clean_end
        gist_end = cursor + gist_tokens.shape[1]
        gist_out = self._post(
            self.video_block,
            gist,
            attended[:, cursor:gist_end],
            video_context,
            video_context_mask,
        )
        cursor = gist_end
        if noisy is None:
            noisy_out = noisy_video_tokens
        else:
            noisy_end = cursor + noisy_video_tokens.shape[1]
            noisy_out = self._post(
                self.video_block,
                noisy,
                attended[:, cursor:noisy_end],
                video_context,
                video_context_mask,
            )
            cursor = noisy_end
        if action is None:
            action_out = action_tokens
        else:
            action_out = self._post(
                self.action_block,
                action,
                attended[:, cursor:],
                action_context,
                action_context_mask,
            )
        return (
            clean_out,
            gist_out,
            noisy_out,
            action_out,
            clean.key,
            clean.value,
            gist.key,
            gist.value,
        )

    def forward_dense_joint(
        self,
        video_tokens: torch.Tensor,
        action_tokens: torch.Tensor,
        video_freqs: torch.Tensor,
        action_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        action_t_mod: torch.Tensor,
        video_context: torch.Tensor,
        video_context_mask: torch.Tensor,
        action_context: torch.Tensor,
        action_context_mask: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compatibility path for inherited bounded-clip FastWAM methods."""

        video = self._attention_io(
            self.video_block, video_tokens, video_freqs, video_t_mod
        )
        action = self._attention_io(
            self.action_block, action_tokens, action_freqs, action_t_mod
        )
        attended = flash_attention(
            q=torch.cat([video.query, action.query], dim=1),
            k=torch.cat([video.key, action.key], dim=1),
            v=torch.cat([video.value, action.value], dim=1),
            num_heads=self.num_heads,
            ctx_mask=attention_mask.to(video_tokens.device),
        )
        split = video_tokens.shape[1]
        return (
            self._post(
                self.video_block,
                video,
                attended[:, :split],
                video_context,
                video_context_mask,
            ),
            self._post(
                self.action_block,
                action,
                attended[:, split:],
                action_context,
                action_context_mask,
            ),
        )

    def prefill_dense_video(
        self,
        video_tokens: torch.Tensor,
        video_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        video_context: torch.Tensor,
        video_context_mask: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        video = self._attention_io(
            self.video_block, video_tokens, video_freqs, video_t_mod
        )
        attended = flash_attention(
            q=video.query,
            k=video.key,
            v=video.value,
            num_heads=self.num_heads,
            ctx_mask=attention_mask.to(video_tokens.device),
        )
        return (
            self._post(
                self.video_block,
                video,
                attended,
                video_context,
                video_context_mask,
            ),
            video.key,
            video.value,
        )

    def forward_action_from_prefix(
        self,
        action_tokens: torch.Tensor,
        action_freqs: torch.Tensor,
        action_t_mod: torch.Tensor,
        action_context: torch.Tensor,
        action_context_mask: torch.Tensor,
        prefix_key: torch.Tensor,
        prefix_value: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        action = self._attention_io(
            self.action_block, action_tokens, action_freqs, action_t_mod
        )
        if attention_mask is None:
            attention_mask = torch.ones(
                (action_tokens.shape[1], prefix_key.shape[1] + action_tokens.shape[1]),
                dtype=torch.bool,
                device=action_tokens.device,
            )
        attended = flash_attention(
            q=action.query,
            k=torch.cat([prefix_key, action.key], dim=1),
            v=torch.cat([prefix_value, action.value], dim=1),
            num_heads=self.num_heads,
            ctx_mask=attention_mask.to(action_tokens.device),
        )
        return self._post(
            self.action_block,
            action,
            attended,
            action_context,
            action_context_mask,
        )


class _ExpertBlockView(Sequence[nn.Module]):
    """Unregistered compatibility view over blocks owned by paired layers."""

    def __init__(self, core: "MemoryWAMSequentialCore", branch: str):
        self._core_ref = weakref.ref(core)
        self._branch = branch

    def _blocks(self) -> list[nn.Module]:
        core = self._core_ref()
        if core is None:
            return []
        attribute = f"{self._branch}_block"
        return [getattr(layer, attribute) for layer in core.layers]

    def __len__(self) -> int:
        return len(self._blocks())

    def __getitem__(self, index):
        return self._blocks()[index]


@dataclass
class _ReferenceFrame:
    full_key: torch.Tensor
    full_value: torch.Tensor
    gist_key: torch.Tensor
    gist_value: torch.Tensor


def _reference_snapshot(
    history: list[_ReferenceFrame],
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if not history:
        return None, None
    retained_full = set(range(min(NUM_ANCHOR_FRAMES, len(history))))
    retained_full.update(
        range(max(NUM_ANCHOR_FRAMES, len(history) - NUM_RECENT_FRAMES), len(history))
    )
    keys: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    for index, frame in enumerate(history):
        if index in retained_full:
            keys.append(frame.full_key)
            values.append(frame.full_value)
        keys.append(frame.gist_key)
        values.append(frame.gist_value)
    return torch.cat(keys, dim=1), torch.cat(values, dim=1)


class MemoryWAMSequentialCore(nn.Module):
    """Stateful per-frame MemoryWAM core with full-episode autograd semantics."""

    def __init__(
        self,
        *,
        video_blocks: Sequence[nn.Module],
        action_blocks: Sequence[nn.Module],
        video_rope_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        video_hidden_dim: int,
        action_hidden_dim: int,
        activation_checkpointing: bool = True,
    ):
        super().__init__()
        if len(video_blocks) == 0 or len(video_blocks) != len(action_blocks):
            raise ValueError(
                "Video/action block lists must have the same positive length."
            )
        if len(video_rope_cache) != 3:
            raise ValueError("`video_rope_cache` must contain three RoPE tables.")
        self.layers = nn.ModuleList(
            [
                MemoryWAMPairedLayer(video_block, action_block)
                for video_block, action_block in zip(video_blocks, action_blocks)
            ]
        )
        self.video_hidden_dim = int(video_hidden_dim)
        self.action_hidden_dim = int(action_hidden_dim)
        if self.video_hidden_dim <= 0 or self.action_hidden_dim <= 0:
            raise ValueError("Expert hidden dimensions must be positive.")
        for layer in self.layers:
            if int(layer.video_block.hidden_dim) != self.video_hidden_dim:
                raise ValueError("Every video block must match `video_hidden_dim`.")
            if int(layer.action_block.hidden_dim) != self.action_hidden_dim:
                raise ValueError("Every action block must match `action_hidden_dim`.")
            if layer.num_heads != self.layers[0].num_heads:
                raise ValueError(
                    "Every paired layer must use the same number of heads."
                )
            if layer.attention_dim != self.layers[0].attention_dim:
                raise ValueError(
                    "Every paired layer must use the same attention width."
                )
        reference_parameter = next(self.layers[0].video_block.parameters(), None)
        gist_kwargs = {}
        if reference_parameter is not None:
            gist_kwargs = {
                "device": reference_parameter.device,
                "dtype": reference_parameter.dtype,
            }
        self.shared_gist_tokens = SharedGistTokens(self.video_hidden_dim, **gist_kwargs)
        self.activation_checkpointing = bool(activation_checkpointing)
        self.num_layers = len(self.layers)
        self.num_heads = self.layers[0].num_heads
        self.attn_head_dim = self.layers[0].attention_dim // self.num_heads
        for name, table in zip(("frame", "row", "column"), video_rope_cache):
            self.register_buffer(f"rope_{name}", table, persistent=False)
        self._layer_caches = [MemoryWAMLayerKVCache() for _ in self.layers]
        self._last_frame_id: int | None = None

    @classmethod
    def from_experts(
        cls,
        video_expert: nn.Module,
        action_expert: nn.Module,
        *,
        activation_checkpointing: bool = True,
    ) -> "MemoryWAMSequentialCore":
        """Move actual DiT blocks into paired ownership and install proxy views."""

        video_blocks = list(video_expert.blocks)
        action_blocks = list(action_expert.blocks)
        if len(video_blocks) == 0 or len(video_blocks) != len(action_blocks):
            raise ValueError("Experts must expose equal, non-empty `blocks` sequences.")
        if not hasattr(video_expert, "freqs") or len(video_expert.freqs) != 3:
            raise ValueError("Video expert must expose Wan 3-D RoPE tables in `freqs`.")
        video_hidden_dim = int(video_expert.hidden_dim)
        action_hidden_dim = int(action_expert.hidden_dim)

        core = cls(
            video_blocks=video_blocks,
            action_blocks=action_blocks,
            video_rope_cache=video_expert.freqs,
            video_hidden_dim=video_hidden_dim,
            action_hidden_dim=action_hidden_dim,
            activation_checkpointing=activation_checkpointing,
        )
        delattr(video_expert, "blocks")
        delattr(action_expert, "blocks")
        object.__setattr__(video_expert, "blocks", _ExpertBlockView(core, "video"))
        object.__setattr__(action_expert, "blocks", _ExpertBlockView(core, "action"))
        return core

    @property
    def rope_cache(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.rope_frame, self.rope_row, self.rope_column

    @property
    def frame_ids(self) -> tuple[int, ...]:
        return self._layer_caches[0].frame_ids

    def reset(self) -> None:
        for cache in self._layer_caches:
            cache.reset()
        self._last_frame_id = None

    def _prefix_tensors(
        self,
        layer_index: int,
        reference_history: list[list[_ReferenceFrame]] | None,
        like: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if reference_history is None:
            snapshot: CacheSnapshot = self._layer_caches[layer_index].snapshot()
            key, value = snapshot.key, snapshot.value
        else:
            key, value = _reference_snapshot(reference_history[layer_index])
        if key is None:
            width = self.layers[layer_index].attention_dim
            empty = like.new_empty((like.shape[0], 0, width))
            return empty, empty
        return key, value

    @staticmethod
    def _gist_modulation(clean_modulation: torch.Tensor) -> torch.Tensor:
        if clean_modulation.ndim == 3:
            return clean_modulation
        if clean_modulation.ndim != 4:
            raise ValueError(
                "Timestep modulation must have shape [B,6,D] or [B,S,6,D]."
            )
        return clean_modulation[:, :1].expand(-1, NUM_GIST_TOKENS, -1, -1)

    def _frequencies(
        self, frame_id: int, grid_height: int, grid_width: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        video = rope_frequencies_from_coordinates(
            self.rope_cache,
            video_coordinates(frame_id, grid_height, grid_width, device=device),
        )
        gist = rope_frequencies_from_coordinates(
            self.rope_cache,
            gist_coordinates(frame_id, grid_height, grid_width, device=device),
        )
        action = rope_frequencies_from_coordinates(
            self.rope_cache,
            action_coordinates(frame_id, grid_height, grid_width, device=device),
        )
        return video, gist, action

    def _call_layer(self, layer: MemoryWAMPairedLayer, *inputs):
        if self.training and self.activation_checkpointing:
            return checkpoint(layer, *inputs, use_reentrant=False)
        return layer(*inputs)

    def _forward_frame(
        self,
        *,
        frame_id: int,
        clean_video_tokens: torch.Tensor,
        noisy_video_tokens: torch.Tensor,
        action_tokens: torch.Tensor,
        clean_video_t_mod: torch.Tensor,
        noisy_video_t_mod: torch.Tensor,
        action_t_mod: torch.Tensor,
        video_context: torch.Tensor,
        video_context_mask: torch.Tensor,
        action_context: torch.Tensor,
        action_context_mask: torch.Tensor,
        grid_height: int,
        grid_width: int,
        reference_history: list[list[_ReferenceFrame]] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if clean_video_tokens.shape[1] != grid_height * grid_width:
            raise ValueError(
                "Clean video token count must equal `grid_height * grid_width`."
            )
        if noisy_video_tokens.shape != clean_video_tokens.shape:
            raise ValueError("Noisy and clean video tokens must have identical shapes.")
        if action_tokens.shape[1] != NUM_ACTION_TOKENS:
            raise ValueError(
                f"MemoryWAM requires {NUM_ACTION_TOKENS} action tokens per frame."
            )
        video_freqs, gist_freqs, action_freqs = self._frequencies(
            frame_id, grid_height, grid_width, clean_video_tokens.device
        )
        gist_tokens = self.shared_gist_tokens(clean_video_tokens.shape[0])[:, 0].to(
            device=clean_video_tokens.device, dtype=clean_video_tokens.dtype
        )
        gist_t_mod = self._gist_modulation(clean_video_t_mod)

        clean = clean_video_tokens
        noisy = noisy_video_tokens
        action = action_tokens
        gist = gist_tokens
        for layer_index, layer in enumerate(self.layers):
            prefix_key, prefix_value = self._prefix_tensors(
                layer_index, reference_history, clean
            )
            (
                clean,
                gist,
                noisy,
                action,
                clean_key,
                clean_value,
                gist_key,
                gist_value,
            ) = self._call_layer(
                layer,
                clean,
                gist,
                noisy,
                action,
                video_freqs,
                gist_freqs,
                video_freqs,
                action_freqs,
                clean_video_t_mod,
                gist_t_mod,
                noisy_video_t_mod,
                action_t_mod,
                video_context,
                video_context_mask,
                action_context,
                action_context_mask,
                prefix_key,
                prefix_value,
            )
            if reference_history is None:
                self._layer_caches[layer_index].append_frame(
                    frame_id,
                    full_key=clean_key,
                    full_value=clean_value,
                    gist_key=gist_key,
                    gist_value=gist_value,
                )
            else:
                reference_history[layer_index].append(
                    _ReferenceFrame(
                        full_key=clean_key,
                        full_value=clean_value,
                        gist_key=gist_key,
                        gist_value=gist_value,
                    )
                )
        return clean, gist, noisy, action

    def forward_frame(self, **kwargs):
        frame_id = kwargs["frame_id"]
        if isinstance(frame_id, bool) or not isinstance(frame_id, int) or frame_id < 0:
            raise ValueError(
                f"Frame IDs must be non-negative integers, got {frame_id!r}."
            )
        if self._last_frame_id is not None and frame_id <= self._last_frame_id:
            raise ValueError(
                f"Frame IDs must increase strictly, got {frame_id} after {self._last_frame_id}."
            )
        output = self._forward_frame(reference_history=None, **kwargs)
        self._last_frame_id = frame_id
        return output

    def forward_episode(
        self,
        *,
        frame_ids: Sequence[int],
        clean_video_frames: Sequence[torch.Tensor],
        noisy_video_frames: Sequence[torch.Tensor],
        action_frames: Sequence[torch.Tensor],
        clean_video_t_mods: Sequence[torch.Tensor],
        noisy_video_t_mods: Sequence[torch.Tensor],
        action_t_mods: Sequence[torch.Tensor],
        video_context: torch.Tensor,
        video_context_mask: torch.Tensor,
        action_context: torch.Tensor,
        action_context_mask: torch.Tensor,
        grid_height: int,
        grid_width: int,
        reference: bool = False,
    ) -> tuple[
        list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]
    ]:
        """Run an episode sequentially without detaching any frame boundary."""

        frame_ids = tuple(frame_ids)
        sequences = (
            clean_video_frames,
            noisy_video_frames,
            action_frames,
            clean_video_t_mods,
            noisy_video_t_mods,
            action_t_mods,
        )
        lengths = {len(frame_ids), *(len(sequence) for sequence in sequences)}
        if lengths != {len(frame_ids)} or not frame_ids:
            raise ValueError(
                "Every episode sequence must have the same positive length."
            )
        if any(
            isinstance(frame_id, bool) or not isinstance(frame_id, int) or frame_id < 0
            for frame_id in frame_ids
        ):
            raise ValueError("`frame_ids` must contain non-negative integers.")
        if any(
            current <= previous for previous, current in zip(frame_ids, frame_ids[1:])
        ):
            raise ValueError("`frame_ids` must increase strictly.")

        if reference:
            histories: list[list[_ReferenceFrame]] | None = [
                [] for _ in range(self.num_layers)
            ]
        else:
            self.reset()
            histories = None
        outputs = ([], [], [], [])
        for index, frame_id in enumerate(frame_ids):
            frame_output = self._forward_frame(
                frame_id=frame_id,
                clean_video_tokens=clean_video_frames[index],
                noisy_video_tokens=noisy_video_frames[index],
                action_tokens=action_frames[index],
                clean_video_t_mod=clean_video_t_mods[index],
                noisy_video_t_mod=noisy_video_t_mods[index],
                action_t_mod=action_t_mods[index],
                video_context=video_context,
                video_context_mask=video_context_mask,
                action_context=action_context,
                action_context_mask=action_context_mask,
                grid_height=grid_height,
                grid_width=grid_width,
                reference_history=histories,
            )
            for branch, value in zip(outputs, frame_output):
                branch.append(value)
        if not reference:
            self._last_frame_id = frame_ids[-1]
        return outputs

    def forward_joint_core(
        self,
        video_tokens: torch.Tensor,
        action_tokens: torch.Tensor,
        video_freqs: torch.Tensor,
        action_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        action_t_mod: torch.Tensor,
        video_context: torch.Tensor,
        video_context_mask: torch.Tensor,
        action_context: torch.Tensor,
        action_context_mask: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Retain inherited bounded-clip behavior for existing smoke paths."""

        video, action = video_tokens, action_tokens
        for layer in self.layers:
            video, action = layer.forward_dense_joint(
                video,
                action,
                video_freqs,
                action_freqs,
                video_t_mod,
                action_t_mod,
                video_context,
                video_context_mask,
                action_context,
                action_context_mask,
                attention_mask,
            )
        return video, action

    def prefill_video_cache_tensor(
        self,
        video_tokens: torch.Tensor,
        video_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        video_context: torch.Tensor,
        video_context_mask: torch.Tensor,
        video_attention_mask: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Compatibility cache for inherited one-image inference only."""

        keys: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        video = video_tokens
        for layer in self.layers:
            video, key, value = layer.prefill_dense_video(
                video,
                video_freqs,
                video_t_mod,
                video_context,
                video_context_mask,
                video_attention_mask,
            )
            keys.append(key)
            values.append(value)
        return keys, values

    def forward_action_with_video_cache_tensor(
        self,
        action_tokens: torch.Tensor,
        action_freqs: torch.Tensor,
        action_t_mod: torch.Tensor,
        action_context: torch.Tensor,
        action_context_mask: torch.Tensor,
        video_cache_k: list[torch.Tensor],
        video_cache_v: list[torch.Tensor],
        action_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        action = action_tokens
        for index, layer in enumerate(self.layers):
            action = layer.forward_action_from_prefix(
                action,
                action_freqs,
                action_t_mod,
                action_context,
                action_context_mask,
                video_cache_k[index],
                video_cache_v[index],
                action_attention_mask,
            )
        return action


__all__ = [
    "MemoryWAMPairedLayer",
    "MemoryWAMSequentialCore",
    "build_current_frame_attention_mask",
    "corrupt_conditioning_latents",
    "rope_frequencies_from_coordinates",
    "sample_conditioning_noise_ratio",
]
