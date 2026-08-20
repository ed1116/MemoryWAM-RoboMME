from __future__ import annotations

import pytest
import torch

from fastwam.models.wan22.memory_cache import (
    MemoryTokenKind,
    MemoryWAMLayerKVCache,
    MemoryWAMTrainingLayout,
    SharedGistTokens,
    action_coordinates,
    gist_coordinates,
    video_coordinates,
)


def _append(cache: MemoryWAMLayerKVCache, frame_id: int, video_tokens: int = 3):
    full_key = torch.full((1, video_tokens, 2), float(frame_id))
    full_value = full_key + 0.25
    gist_key = torch.full((1, 8, 2), float(frame_id) + 0.5)
    gist_value = gist_key + 0.25
    cache.append_frame(
        frame_id,
        full_key=full_key,
        full_value=full_value,
        gist_key=gist_key,
        gist_value=gist_value,
    )


def _training_columns_for_snapshot(layout, snapshot):
    columns = []
    for segment in snapshot.segments:
        span = layout.span(segment.frame_id, segment.kind)
        columns.extend(range(span.start, span.stop))
    return columns


def test_eviction_preserves_two_anchors_four_recent_frames_and_every_gist():
    cache = MemoryWAMLayerKVCache()
    for frame_id in range(10):
        _append(cache, frame_id)

    assert cache.full_frame_ids == (0, 1, 6, 7, 8, 9)
    assert cache.gist_frame_ids == tuple(range(10))
    snapshot = cache.snapshot()
    assert snapshot.key is not None
    assert snapshot.value is not None
    assert snapshot.key.shape[-2] == 6 * 3 + 10 * 8
    assert all(
        segment.kind is not MemoryTokenKind.ACTION for segment in snapshot.segments
    )


def test_shared_gist_bank_is_reused_and_accumulates_gradients():
    bank = SharedGistTokens(hidden_dim=4)
    tokens = bank(batch_size=2, num_frames=3)

    assert tokens.shape == (2, 3, 8, 4)
    torch.testing.assert_close(tokens[0, 0], tokens[1, 2])
    tokens.sum().backward()
    torch.testing.assert_close(bank.weight.grad, torch.full_like(bank.weight, 6.0))


def test_attention_mask_has_no_future_or_historical_action_edges():
    layout = MemoryWAMTrainingLayout(range(8), video_tokens_per_frame=3)
    mask = layout.attention_mask()

    frame_id = 7
    for query_kind in MemoryTokenKind:
        query_row = layout.span(frame_id, query_kind).start
        for future_kind in MemoryTokenKind:
            # The sequence has no later frame after frame 7, so inspect frame 3 below.
            assert (
                mask[
                    layout.span(3, query_kind).start,
                    layout.span(4, future_kind).token_slice,
                ].sum()
                == 0
            )
        for past_frame in range(frame_id):
            assert (
                mask[
                    query_row,
                    layout.span(past_frame, MemoryTokenKind.ACTION).token_slice,
                ].sum()
                == 0
            )

    query_row = layout.span(frame_id, MemoryTokenKind.VIDEO).start
    assert mask[query_row, layout.span(0, MemoryTokenKind.VIDEO).token_slice].all()
    assert mask[query_row, layout.span(1, MemoryTokenKind.VIDEO).token_slice].all()
    assert not mask[query_row, layout.span(2, MemoryTokenKind.VIDEO).token_slice].any()
    assert mask[query_row, layout.span(2, MemoryTokenKind.GIST).token_slice].all()
    assert not mask[query_row, layout.span(3, MemoryTokenKind.VIDEO).token_slice].any()
    for recent_frame in (4, 5, 6, 7):
        assert mask[
            query_row, layout.span(recent_frame, MemoryTokenKind.VIDEO).token_slice
        ].all()

    action = layout.span(frame_id, MemoryTokenKind.ACTION)
    assert mask[action.token_slice, action.token_slice].all()
    assert not mask[
        layout.span(frame_id, MemoryTokenKind.GIST).token_slice, action.token_slice
    ].any()


def test_training_mask_matches_incremental_inference_cache_oracle():
    frame_ids = (10, 20, 30, 40, 50, 60, 70, 80)
    layout = MemoryWAMTrainingLayout(frame_ids, video_tokens_per_frame=3)
    mask = layout.attention_mask()
    cache = MemoryWAMLayerKVCache()

    for frame_id in frame_ids:
        _append(cache, frame_id)
        persistent_columns = _training_columns_for_snapshot(layout, cache.snapshot())
        expected = torch.zeros(layout.token_count, dtype=torch.bool)
        expected[persistent_columns] = True

        for kind in (MemoryTokenKind.VIDEO, MemoryTokenKind.GIST):
            query_rows = layout.span(frame_id, kind).token_slice
            torch.testing.assert_close(
                mask[query_rows],
                expected.expand(query_rows.stop - query_rows.start, -1),
            )

        action = layout.span(frame_id, MemoryTokenKind.ACTION)
        expected_action = expected.clone()
        expected_action[action.token_slice] = True
        torch.testing.assert_close(
            mask[action.token_slice],
            expected_action.expand(action.token_count, -1),
        )


def test_reset_clears_state_and_frame_ids_are_strictly_monotonic():
    cache = MemoryWAMLayerKVCache()
    _append(cache, 5)
    with pytest.raises(ValueError, match="increase strictly"):
        _append(cache, 5)
    with pytest.raises(ValueError, match="increase strictly"):
        _append(cache, 4)

    cache.reset()
    assert len(cache) == 0
    assert cache.frame_ids == ()
    assert cache.snapshot().key is None
    _append(cache, 0)
    assert cache.full_frame_ids == (0,)


def test_video_gist_and_action_coordinates_share_absolute_3d_basis():
    video = video_coordinates(frame_id=12, grid_height=2, grid_width=3)
    gist = gist_coordinates(frame_id=12, grid_height=2, grid_width=3)
    action = action_coordinates(frame_id=12, grid_height=2, grid_width=3)

    torch.testing.assert_close(
        video,
        torch.tensor(
            [
                [12, 0, 0],
                [12, 0, 1],
                [12, 0, 2],
                [12, 1, 0],
                [12, 1, 1],
                [12, 1, 2],
            ]
        ),
    )
    torch.testing.assert_close(gist, torch.tensor([[12, 2, 3]]).expand(8, -1))
    torch.testing.assert_close(action[:, 0], torch.full((16,), 12))
    torch.testing.assert_close(action[:, 1], torch.full((16,), 2))
    torch.testing.assert_close(action[:, 2], torch.arange(16))

    layout = MemoryWAMTrainingLayout((12,), video_tokens_per_frame=6)
    assert layout.coordinates(grid_height=2, grid_width=3).shape == (30, 3)
    with pytest.raises(ValueError, match="does not match"):
        layout.coordinates(grid_height=2, grid_width=2)


def test_long_history_grows_only_by_gists_beyond_bounded_full_cache():
    cache = MemoryWAMLayerKVCache()
    frame_ids = tuple(range(1_000, 2_600))
    for frame_id in frame_ids:
        _append(cache, frame_id, video_tokens=1)

    assert cache.full_frame_ids == frame_ids[:2] + frame_ids[-4:]
    assert cache.gist_frame_ids == frame_ids
    snapshot = cache.snapshot()
    assert snapshot.token_count == 6 + 1_600 * 8
    assert snapshot.key is not None
    assert snapshot.key.shape == (1, snapshot.token_count, 2)
