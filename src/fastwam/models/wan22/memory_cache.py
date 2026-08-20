"""MemoryWAM's model-independent hybrid-memory cache contract.

The retention policy is specified by the MemoryWAM paper: the first two
frames remain as full-token anchors, the four latest non-anchor frames remain
as full-token short-term memory, and eight gist tokens from every frame remain
for the episode.  The following details are approved implementation
inferences for this reimplementation:

* one shared bank of eight learnable gist inputs is reused for every frame;
* clean-frame and same-frame gist queries are mutually visible;
* current action queries see the hybrid video cache and one another, while
  actions are never inserted into persistent memory;
* gist and action tokens use the video expert's 3D coordinate basis, with the
  one-past-grid marker described by :func:`gist_coordinates` and
  :func:`action_coordinates`.

The cache holds one transformer's layer of K/V tensors.  Integration code is
expected to create one instance per paired video/action layer.  Tensor inputs
keep their autograd history; this module neither clones nor detaches them.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

import torch
import torch.nn as nn


NUM_ANCHOR_FRAMES = 2
NUM_RECENT_FRAMES = 4
NUM_GIST_TOKENS = 8
NUM_ACTION_TOKENS = 16


class MemoryTokenKind(str, Enum):
    """Token blocks represented by the hybrid-memory mask."""

    VIDEO = "video"
    GIST = "gist"
    ACTION = "action"


@dataclass(frozen=True)
class TokenSpan:
    """A contiguous token block in a flattened training sequence."""

    frame_id: int
    kind: MemoryTokenKind
    start: int
    stop: int

    @property
    def token_slice(self) -> slice:
        return slice(self.start, self.stop)

    @property
    def token_count(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True)
class CacheSnapshot:
    """Materialized persistent K/V in chronological block order."""

    key: torch.Tensor | None
    value: torch.Tensor | None
    segments: tuple[TokenSpan, ...]

    @property
    def token_count(self) -> int:
        return sum(segment.token_count for segment in self.segments)


@dataclass
class _CachedFrame:
    frame_id: int
    is_anchor: bool
    full_key: torch.Tensor | None
    full_value: torch.Tensor | None
    gist_key: torch.Tensor
    gist_value: torch.Tensor


class SharedGistTokens(nn.Module):
    """One learned eight-token input bank shared by all episode frames."""

    def __init__(self, hidden_dim: int, *, device=None, dtype=None):
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError(f"`hidden_dim` must be positive, got {hidden_dim}.")
        self.weight = nn.Parameter(
            torch.empty(NUM_GIST_TOKENS, hidden_dim, device=device, dtype=dtype)
        )
        nn.init.normal_(self.weight, std=hidden_dim**-0.5)

    def forward(self, batch_size: int, num_frames: int = 1) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError(f"`batch_size` must be positive, got {batch_size}.")
        if num_frames <= 0:
            raise ValueError(f"`num_frames` must be positive, got {num_frames}.")
        return self.weight.view(1, 1, NUM_GIST_TOKENS, -1).expand(
            batch_size, num_frames, -1, -1
        )


class MemoryWAMLayerKVCache:
    """Persistent hybrid K/V state for one transformer layer.

    K/V tensors use their penultimate dimension as the token dimension.  This
    accepts the FastWAM forms ``[B, S, D]`` and ``[B, H, S, D]`` without
    coupling the cache contract to one attention implementation.
    """

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        """Clear every tensor and restart the per-episode frame ordering."""

        self._frames: list[_CachedFrame] = []
        self._last_frame_id: int | None = None

    def __len__(self) -> int:
        return len(self._frames)

    @property
    def frame_ids(self) -> tuple[int, ...]:
        return tuple(frame.frame_id for frame in self._frames)

    @property
    def full_frame_ids(self) -> tuple[int, ...]:
        return tuple(
            frame.frame_id for frame in self._frames if frame.full_key is not None
        )

    @property
    def gist_frame_ids(self) -> tuple[int, ...]:
        return tuple(frame.frame_id for frame in self._frames)

    def append_frame(
        self,
        frame_id: int,
        *,
        full_key: torch.Tensor,
        full_value: torch.Tensor,
        gist_key: torch.Tensor,
        gist_value: torch.Tensor,
    ) -> None:
        """Insert one clean frame and evict only obsolete full-frame K/V."""

        frame_id = _validate_frame_id(frame_id)
        if self._last_frame_id is not None and frame_id <= self._last_frame_id:
            raise ValueError(
                "Frame IDs must increase strictly within an episode, got "
                f"{frame_id} after {self._last_frame_id}."
            )

        _validate_kv_pair("full", full_key, full_value)
        _validate_kv_pair("gist", gist_key, gist_value, expected_tokens=NUM_GIST_TOKENS)
        if _non_sequence_shape(full_key) != _non_sequence_shape(gist_key):
            raise ValueError(
                "Full-frame and gist K/V must match outside the token dimension, "
                f"got {tuple(full_key.shape)} and {tuple(gist_key.shape)}."
            )

        self._frames.append(
            _CachedFrame(
                frame_id=frame_id,
                is_anchor=len(self._frames) < NUM_ANCHOR_FRAMES,
                full_key=full_key,
                full_value=full_value,
                gist_key=gist_key,
                gist_value=gist_value,
            )
        )
        self._last_frame_id = frame_id
        self._evict_old_full_frames()

    def snapshot(self) -> CacheSnapshot:
        """Concatenate retained video/gist K/V without any action K/V."""

        keys: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        for frame in self._frames:
            if frame.full_key is not None:
                keys.append(frame.full_key)
                assert frame.full_value is not None
                values.append(frame.full_value)
            keys.append(frame.gist_key)
            values.append(frame.gist_value)

        if not keys:
            return CacheSnapshot(key=None, value=None, segments=())
        return CacheSnapshot(
            key=torch.cat(keys, dim=-2),
            value=torch.cat(values, dim=-2),
            segments=self._segments(),
        )

    def coordinates(
        self,
        grid_height: int,
        grid_width: int,
        *,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Return 3D positions aligned with :meth:`snapshot`."""

        blocks: list[torch.Tensor] = []
        for segment in self._segments():
            if segment.kind is MemoryTokenKind.VIDEO:
                block = video_coordinates(
                    segment.frame_id, grid_height, grid_width, device=device
                )
            else:
                block = gist_coordinates(
                    segment.frame_id, grid_height, grid_width, device=device
                )
            if block.shape[0] != segment.token_count:
                raise ValueError(
                    "Cached full-frame token count does not match the supplied grid: "
                    f"frame {segment.frame_id} has {segment.token_count} tokens, "
                    f"grid has {block.shape[0]}."
                )
            blocks.append(block)
        if not blocks:
            return torch.empty((0, 3), dtype=torch.long, device=device)
        return torch.cat(blocks, dim=0)

    def _evict_old_full_frames(self) -> None:
        recent_non_anchors = [
            frame
            for frame in self._frames
            if not frame.is_anchor and frame.full_key is not None
        ]
        for frame in recent_non_anchors[:-NUM_RECENT_FRAMES]:
            frame.full_key = None
            frame.full_value = None

    def _segments(self) -> tuple[TokenSpan, ...]:
        segments: list[TokenSpan] = []
        cursor = 0
        for frame in self._frames:
            if frame.full_key is not None:
                token_count = int(frame.full_key.shape[-2])
                segments.append(
                    TokenSpan(
                        frame_id=frame.frame_id,
                        kind=MemoryTokenKind.VIDEO,
                        start=cursor,
                        stop=cursor + token_count,
                    )
                )
                cursor += token_count

            token_count = int(frame.gist_key.shape[-2])
            segments.append(
                TokenSpan(
                    frame_id=frame.frame_id,
                    kind=MemoryTokenKind.GIST,
                    start=cursor,
                    stop=cursor + token_count,
                )
            )
            cursor += token_count
        return tuple(segments)


class MemoryWAMTrainingLayout:
    """Flattened per-frame clean/gist/action sequence and its exact mask."""

    def __init__(self, frame_ids: Iterable[int], video_tokens_per_frame: int):
        ids = tuple(_validate_frame_id(frame_id) for frame_id in frame_ids)
        if not ids:
            raise ValueError("`frame_ids` must contain at least one frame.")
        if any(current <= previous for previous, current in zip(ids, ids[1:])):
            raise ValueError(f"`frame_ids` must be strictly increasing, got {ids}.")
        if video_tokens_per_frame <= 0:
            raise ValueError(
                "`video_tokens_per_frame` must be positive, got "
                f"{video_tokens_per_frame}."
            )

        self.frame_ids = ids
        self.video_tokens_per_frame = int(video_tokens_per_frame)
        spans: list[TokenSpan] = []
        lookup: dict[tuple[int, MemoryTokenKind], TokenSpan] = {}
        cursor = 0
        counts = (
            (MemoryTokenKind.VIDEO, self.video_tokens_per_frame),
            (MemoryTokenKind.GIST, NUM_GIST_TOKENS),
            (MemoryTokenKind.ACTION, NUM_ACTION_TOKENS),
        )
        for frame_id in ids:
            for kind, count in counts:
                span = TokenSpan(frame_id, kind, cursor, cursor + count)
                spans.append(span)
                lookup[(frame_id, kind)] = span
                cursor += count
        self.spans = tuple(spans)
        self.token_count = cursor
        self._lookup = lookup

    def span(self, frame_id: int, kind: MemoryTokenKind) -> TokenSpan:
        return self._lookup[(frame_id, kind)]

    def attention_mask(
        self, *, device: torch.device | str | None = None
    ) -> torch.Tensor:
        """Build the paper-reconstructed training visibility matrix.

        Each prefix is viewed after its current clean frame and gist have been
        inserted into the inference cache.  Clean and gist queries see exactly
        that hybrid video snapshot.  Current action queries additionally see
        their same-frame action block.  Therefore the matrix has no future
        frame edges and no historical action edges.
        """

        mask = torch.zeros(
            (self.token_count, self.token_count), dtype=torch.bool, device=device
        )
        for frame_index, frame_id in enumerate(self.frame_ids):
            prefix = self.frame_ids[: frame_index + 1]
            retained_full = set(prefix[:NUM_ANCHOR_FRAMES])
            retained_full.update(prefix[NUM_ANCHOR_FRAMES:][-NUM_RECENT_FRAMES:])

            key_spans = [
                self.span(prefix_frame_id, MemoryTokenKind.GIST)
                for prefix_frame_id in prefix
            ]
            key_spans.extend(
                self.span(prefix_frame_id, MemoryTokenKind.VIDEO)
                for prefix_frame_id in prefix
                if prefix_frame_id in retained_full
            )

            for query_kind in (MemoryTokenKind.VIDEO, MemoryTokenKind.GIST):
                query = self.span(frame_id, query_kind).token_slice
                for key in key_spans:
                    mask[query, key.token_slice] = True

            action = self.span(frame_id, MemoryTokenKind.ACTION)
            for key in key_spans:
                mask[action.token_slice, key.token_slice] = True
            mask[action.token_slice, action.token_slice] = True
        return mask

    def coordinates(
        self,
        grid_height: int,
        grid_width: int,
        *,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        _validate_grid(grid_height, grid_width)
        grid_tokens = grid_height * grid_width
        if self.video_tokens_per_frame != grid_tokens:
            raise ValueError(
                "Training video-token count does not match the supplied grid: "
                f"layout has {self.video_tokens_per_frame}, grid has {grid_tokens}."
            )
        blocks: list[torch.Tensor] = []
        for span in self.spans:
            if span.kind is MemoryTokenKind.VIDEO:
                block = video_coordinates(
                    span.frame_id, grid_height, grid_width, device=device
                )
            elif span.kind is MemoryTokenKind.GIST:
                block = gist_coordinates(
                    span.frame_id, grid_height, grid_width, device=device
                )
            else:
                block = action_coordinates(
                    span.frame_id, grid_height, grid_width, device=device
                )
            blocks.append(block)
        return torch.cat(blocks, dim=0)


def video_coordinates(
    frame_id: int,
    grid_height: int,
    grid_width: int,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Raster-ordered ``(frame, row, column)`` positions for video tokens."""

    frame_id = _validate_frame_id(frame_id)
    _validate_grid(grid_height, grid_width)
    rows = torch.arange(grid_height, dtype=torch.long, device=device).repeat_interleave(
        grid_width
    )
    columns = torch.arange(grid_width, dtype=torch.long, device=device).repeat(
        grid_height
    )
    frames = torch.full_like(rows, frame_id)
    return torch.stack((frames, rows, columns), dim=-1)


def gist_coordinates(
    frame_id: int,
    grid_height: int,
    grid_width: int,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Eight positions at the associated frame's one-past-grid marker."""

    frame_id = _validate_frame_id(frame_id)
    _validate_grid(grid_height, grid_width)
    return torch.tensor(
        (frame_id, grid_height, grid_width), dtype=torch.long, device=device
    ).expand(NUM_GIST_TOKENS, -1)


def action_coordinates(
    frame_id: int,
    grid_height: int,
    grid_width: int,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Current-frame action positions at sentinel row and sub-step columns."""

    frame_id = _validate_frame_id(frame_id)
    _validate_grid(grid_height, grid_width)
    columns = torch.arange(NUM_ACTION_TOKENS, dtype=torch.long, device=device)
    frames = torch.full_like(columns, frame_id)
    rows = torch.full_like(columns, grid_height)
    return torch.stack((frames, rows, columns), dim=-1)


def _validate_frame_id(frame_id: int) -> int:
    if isinstance(frame_id, bool) or not isinstance(frame_id, int) or frame_id < 0:
        raise ValueError(f"Frame IDs must be non-negative integers, got {frame_id!r}.")
    return frame_id


def _validate_grid(grid_height: int, grid_width: int) -> None:
    if grid_height <= 0 or grid_width <= 0:
        raise ValueError(
            f"Grid dimensions must be positive, got {(grid_height, grid_width)}."
        )


def _validate_kv_pair(
    name: str,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    expected_tokens: int | None = None,
) -> None:
    if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} K/V must both be torch tensors.")
    if key.ndim < 2 or value.ndim < 2:
        raise ValueError(f"{name} K/V must have at least two dimensions.")
    if key.shape != value.shape:
        raise ValueError(
            f"{name} K/V shapes must match, got {tuple(key.shape)} and "
            f"{tuple(value.shape)}."
        )
    token_count = int(key.shape[-2])
    if token_count <= 0:
        raise ValueError(f"{name} K/V must contain at least one token.")
    if expected_tokens is not None and token_count != expected_tokens:
        raise ValueError(
            f"{name} K/V must contain {expected_tokens} tokens, got {token_count}."
        )


def _non_sequence_shape(tensor: torch.Tensor) -> tuple[int, ...]:
    return tuple(tensor.shape[:-2]) + tuple(tensor.shape[-1:])


__all__ = [
    "CacheSnapshot",
    "MemoryTokenKind",
    "MemoryWAMLayerKVCache",
    "MemoryWAMTrainingLayout",
    "NUM_ACTION_TOKENS",
    "NUM_ANCHOR_FRAMES",
    "NUM_GIST_TOKENS",
    "NUM_RECENT_FRAMES",
    "SharedGistTokens",
    "TokenSpan",
    "action_coordinates",
    "gist_coordinates",
    "video_coordinates",
]
