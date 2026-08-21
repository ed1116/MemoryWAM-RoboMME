from __future__ import annotations

import copy

import torch
import torch.nn as nn

from fastwam.models.wan22.memory_mot import (
    MemoryWAMSequentialCore,
    build_current_frame_attention_mask,
    corrupt_conditioning_latents,
    rope_frequencies_from_coordinates,
    sample_conditioning_noise_ratio,
)
from fastwam.models.wan22.mot import MoT
from fastwam.models.wan22.wan_video_dit import (
    DiTBlock,
    precompute_freqs_cis,
    precompute_freqs_cis_3d,
)


VIDEO_DIM = 8
ACTION_DIM = 6
HEAD_DIM = 6
NUM_HEADS = 2
VIDEO_TOKENS = 4
ACTION_TOKENS = 16


class _TinyExpert(nn.Module):
    def __init__(self, hidden_dim: int, *, is_video: bool):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = NUM_HEADS
        self.attn_head_dim = HEAD_DIM
        self.use_gradient_checkpointing = False
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_dim=hidden_dim,
                    attn_head_dim=HEAD_DIM,
                    num_heads=NUM_HEADS,
                    ffn_dim=hidden_dim * 2,
                    eps=1e-6,
                )
                for _ in range(2)
            ]
        )
        self.freqs = (
            precompute_freqs_cis_3d(HEAD_DIM, end=64)
            if is_video
            else precompute_freqs_cis(HEAD_DIM, end=64)
        )


def _make_core(*, seed: int = 0, activation_checkpointing: bool = False):
    torch.manual_seed(seed)
    return MemoryWAMSequentialCore(
        video_blocks=[
            DiTBlock(VIDEO_DIM, HEAD_DIM, NUM_HEADS, VIDEO_DIM * 2, 1e-6)
            for _ in range(2)
        ],
        action_blocks=[
            DiTBlock(ACTION_DIM, HEAD_DIM, NUM_HEADS, ACTION_DIM * 2, 1e-6)
            for _ in range(2)
        ],
        video_rope_cache=precompute_freqs_cis_3d(HEAD_DIM, end=64),
        video_hidden_dim=VIDEO_DIM,
        action_hidden_dim=ACTION_DIM,
        activation_checkpointing=activation_checkpointing,
    )


def _episode_inputs(frame_ids, *, requires_grad: bool = False):
    frame_ids = tuple(frame_ids)
    generator = torch.Generator().manual_seed(987)

    def randn(*shape, grad=False):
        return torch.randn(*shape, generator=generator).requires_grad_(grad)

    batch_size = 1
    return {
        "frame_ids": frame_ids,
        "clean_video_frames": [
            randn(batch_size, VIDEO_TOKENS, VIDEO_DIM, grad=requires_grad)
            for _ in frame_ids
        ],
        "noisy_video_frames": [
            randn(batch_size, VIDEO_TOKENS, VIDEO_DIM, grad=requires_grad)
            for _ in frame_ids
        ],
        "action_frames": [
            randn(batch_size, ACTION_TOKENS, ACTION_DIM, grad=requires_grad)
            for _ in frame_ids
        ],
        "clean_video_t_mods": [randn(batch_size, 6, VIDEO_DIM) for _ in frame_ids],
        "noisy_video_t_mods": [randn(batch_size, 6, VIDEO_DIM) for _ in frame_ids],
        "action_t_mods": [randn(batch_size, 6, ACTION_DIM) for _ in frame_ids],
        "video_context": randn(batch_size, 3, VIDEO_DIM),
        "video_context_mask": torch.tensor([[True, True, False]]),
        "action_context": randn(batch_size, 3, ACTION_DIM),
        "action_context_mask": torch.tensor([[True, True, False]]),
        "grid_height": 2,
        "grid_width": 2,
    }


def _frame_inputs(episode, index: int):
    return {
        "frame_id": episode["frame_ids"][index],
        "clean_video_tokens": episode["clean_video_frames"][index],
        "noisy_video_tokens": episode["noisy_video_frames"][index],
        "action_tokens": episode["action_frames"][index],
        "clean_video_t_mod": episode["clean_video_t_mods"][index],
        "noisy_video_t_mod": episode["noisy_video_t_mods"][index],
        "action_t_mod": episode["action_t_mods"][index],
        "video_context": episode["video_context"],
        "video_context_mask": episode["video_context_mask"],
        "action_context": episode["action_context"],
        "action_context_mask": episode["action_context_mask"],
        "grid_height": episode["grid_height"],
        "grid_width": episode["grid_width"],
    }


def test_current_frame_mask_separates_clean_noisy_and_action_branches():
    mask = build_current_frame_attention_mask(
        prefix_tokens=2,
        clean_video_tokens=2,
        gist_tokens=8,
        noisy_video_tokens=2,
        action_tokens=16,
    )
    # keys: prefix[0,2) clean[2,4) gist[4,12) noisy[12,14) action[14,30)
    # queries: clean[0,2) gist[2,10) noisy[10,12) action[12,28)
    expected = torch.zeros_like(mask)
    expected[:10, :12] = True
    expected[10:12, :2] = True
    expected[10:12, 12:14] = True
    expected[12:, :12] = True
    expected[12:, 14:] = True

    torch.testing.assert_close(mask, expected)


def test_conditioning_corruption_uses_continuous_ratio_and_is_seedable():
    clean = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)
    noise = torch.full_like(clean, -2.0)
    corrupted, timestep = corrupt_conditioning_latents(
        clean,
        torch.tensor([0.0, 1.0]),
        noise=noise,
    )

    torch.testing.assert_close(corrupted[0], clean[0])
    torch.testing.assert_close(corrupted[1], noise[1])
    torch.testing.assert_close(timestep, torch.tensor([0.0, 1000.0]))
    first = sample_conditioning_noise_ratio(
        4, device="cpu", generator=torch.Generator().manual_seed(11)
    )
    second = sample_conditioning_noise_ratio(
        4, device="cpu", generator=torch.Generator().manual_seed(11)
    )
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_fastwam_compatibility_paths_match_and_transfer_ownership():
    torch.manual_seed(4)
    video_expert = _TinyExpert(VIDEO_DIM, is_video=True).double()
    action_expert = _TinyExpert(ACTION_DIM, is_video=False).double()
    core_video_expert = copy.deepcopy(video_expert)
    core_action_expert = copy.deepcopy(action_expert)
    fastwam_mot = MoT({"video": video_expert, "action": action_expert})
    core = MemoryWAMSequentialCore.from_experts(
        core_video_expert,
        core_action_expert,
        activation_checkpointing=False,
    )

    batch_size = 2
    video_tokens = torch.randn(batch_size, 3, VIDEO_DIM, dtype=torch.float64)
    action_tokens = torch.randn(batch_size, 4, ACTION_DIM, dtype=torch.float64)
    video_freqs = rope_frequencies_from_coordinates(
        core.rope_cache,
        torch.tensor([[1, 0, 0], [1, 0, 1], [1, 0, 2]]),
    )
    action_freqs = action_expert.freqs[:4].view(4, 1, -1)
    video_t_mod = torch.randn(batch_size, 6, VIDEO_DIM, dtype=torch.float64)
    action_t_mod = torch.randn(batch_size, 6, ACTION_DIM, dtype=torch.float64)
    video_context = torch.randn(batch_size, 3, VIDEO_DIM, dtype=torch.float64)
    action_context = torch.randn(batch_size, 3, ACTION_DIM, dtype=torch.float64)
    video_context_mask = torch.ones(batch_size, 3, 3, dtype=torch.bool)
    action_context_mask = torch.ones(batch_size, 4, 3, dtype=torch.bool)
    attention_mask = torch.zeros(7, 7, dtype=torch.bool)
    attention_mask[:3, :3] = True
    attention_mask[3:, :] = True
    kwargs = {
        "video_tokens": video_tokens,
        "action_tokens": action_tokens,
        "video_freqs": video_freqs,
        "action_freqs": action_freqs,
        "video_t_mod": video_t_mod,
        "action_t_mod": action_t_mod,
        "video_context": video_context,
        "video_context_mask": video_context_mask,
        "action_context": action_context,
        "action_context_mask": action_context_mask,
        "attention_mask": attention_mask,
    }

    expected = fastwam_mot.forward_joint_core(**kwargs)
    actual = core.forward_joint_core(**kwargs)

    for actual_branch, expected_branch in zip(actual, expected):
        torch.testing.assert_close(actual_branch, expected_branch)

    expected_k, expected_v = fastwam_mot.prefill_video_cache_tensor(
        video_tokens=video_tokens,
        video_freqs=video_freqs,
        video_t_mod=video_t_mod,
        video_context=video_context,
        video_context_mask=video_context_mask,
        video_attention_mask=attention_mask[:3, :3],
    )
    actual_k, actual_v = core.prefill_video_cache_tensor(
        video_tokens=video_tokens,
        video_freqs=video_freqs,
        video_t_mod=video_t_mod,
        video_context=video_context,
        video_context_mask=video_context_mask,
        video_attention_mask=attention_mask[:3, :3],
    )
    for actual_cache, expected_cache in zip(
        actual_k + actual_v, expected_k + expected_v
    ):
        torch.testing.assert_close(actual_cache, expected_cache)
    expected_action = fastwam_mot.forward_action_with_video_cache_tensor(
        action_tokens=action_tokens,
        action_freqs=action_freqs,
        action_t_mod=action_t_mod,
        action_context=action_context,
        action_context_mask=action_context_mask,
        video_cache_k=expected_k,
        video_cache_v=expected_v,
        action_attention_mask=attention_mask[3:],
    )
    actual_action = core.forward_action_with_video_cache_tensor(
        action_tokens=action_tokens,
        action_freqs=action_freqs,
        action_t_mod=action_t_mod,
        action_context=action_context,
        action_context_mask=action_context_mask,
        video_cache_k=actual_k,
        video_cache_v=actual_v,
        action_attention_mask=attention_mask[3:],
    )
    torch.testing.assert_close(actual_action, expected_action)
    assert core_video_expert.blocks[0] is core.layers[0].video_block
    assert core_action_expert.blocks[1] is core.layers[1].action_block
    assert core.shared_gist_tokens.weight.dtype == torch.float64


def test_cached_episode_matches_independent_history_reference_through_eviction():
    core = _make_core(seed=3)
    core.eval()
    episode = _episode_inputs(range(7))

    with torch.no_grad():
        cached = core.forward_episode(**episode)
        reference = core.forward_episode(**episode, reference=True)

    for cached_branch, reference_branch in zip(cached, reference):
        for cached_frame, reference_frame in zip(cached_branch, reference_branch):
            torch.testing.assert_close(cached_frame, reference_frame, rtol=0, atol=0)
    assert core.frame_ids == tuple(range(7))
    assert all(
        cache.full_frame_ids == (0, 1, 3, 4, 5, 6) for cache in core._layer_caches
    )


def test_reset_matches_a_fresh_core():
    core = _make_core(seed=5)
    fresh = _make_core(seed=6)
    fresh.load_state_dict(core.state_dict())
    core.eval()
    fresh.eval()

    with torch.no_grad():
        core.forward_episode(**_episode_inputs((10, 11, 12)))
        core.reset()
        assert core.frame_ids == ()
        episode = _episode_inputs((0,))
        after_reset = core.forward_frame(**_frame_inputs(episode, 0))
        from_fresh = fresh.forward_frame(**_frame_inputs(episode, 0))

    for actual, expected in zip(after_reset, from_fresh):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_checkpointed_episode_backpropagates_across_cache_boundaries():
    core = _make_core(seed=7, activation_checkpointing=True)
    core.train()
    episode = _episode_inputs(range(7), requires_grad=True)

    outputs = core.forward_episode(**episode)
    outputs[3][-1].square().mean().backward()

    evicted_clean_grad = episode["clean_video_frames"][2].grad
    gist_grad = core.shared_gist_tokens.weight.grad
    assert evicted_clean_grad is not None
    assert gist_grad is not None
    assert bool(torch.isfinite(evicted_clean_grad).all())
    assert bool(torch.isfinite(gist_grad).all())
    assert int(torch.count_nonzero(evicted_clean_grad)) > 0
    assert int(torch.count_nonzero(gist_grad)) > 0
    assert core.frame_ids == tuple(range(7))
    assert all(2 not in cache.full_frame_ids for cache in core._layer_caches)


def test_noisy_video_target_never_sees_its_own_frame_conditioning():
    """The noisy target is the same frame as the clean block, so it must not read it.

    Released FastWAM pairs ``latents_noisy`` and ``latents_cond`` from one
    ``input_latents`` tensor and leaves the noisy->cond block of its
    teacher-forcing mask False.  A video rollout also has no clean block for the
    frame it is generating, so this edge would break inference equivalence too.
    """

    prefix, clean, gist, noisy, action = 3, 4, 8, 4, 16
    mask = build_current_frame_attention_mask(
        prefix_tokens=prefix,
        clean_video_tokens=clean,
        gist_tokens=gist,
        noisy_video_tokens=noisy,
        action_tokens=action,
    )
    prefix_keys = slice(0, prefix)
    clean_keys = slice(prefix, prefix + clean)
    gist_keys = slice(prefix + clean, prefix + clean + gist)
    noisy_keys = slice(prefix + clean + gist, prefix + clean + gist + noisy)
    noisy_queries = slice(clean + gist, clean + gist + noisy)
    action_queries = slice(clean + gist + noisy, clean + gist + noisy + action)

    assert not mask[noisy_queries, clean_keys].any()
    assert not mask[noisy_queries, gist_keys].any()
    assert mask[noisy_queries, prefix_keys].all()
    assert mask[noisy_queries, noisy_keys].all()

    # The action target keeps the observation it conditions on at deployment.
    assert mask[action_queries, clean_keys].all()
    assert mask[action_queries, gist_keys].all()
    assert not mask[action_queries, noisy_keys].any()


def test_noisy_video_target_has_no_gradient_path_to_its_own_clean_frame():
    """Gradient presence is exact, so it proves the edge rather than measuring it."""

    core = _make_core(seed=11)
    episode = _episode_inputs((0, 1), requires_grad=True)
    noisy_outputs = core.forward_episode(**episode)[2]
    first_clean = episode["clean_video_frames"][0]

    # Frame 0's noisy target has no history, so the only possible path to any
    # clean tensor would be the same-frame conditioning edge.
    (own_frame,) = torch.autograd.grad(
        noisy_outputs[0].sum(), first_clean, retain_graph=True, allow_unused=True
    )
    assert own_frame is None or bool((own_frame == 0).all())

    # Frame 1 must still reach clean frame 0 through the retained history.
    (through_history,) = torch.autograd.grad(
        noisy_outputs[1].sum(), first_clean, retain_graph=True, allow_unused=True
    )
    assert through_history is not None and bool((through_history != 0).any())

    # The action target keeps its direct path to the same frame's observation.
    action_outputs = core.forward_episode(**episode)[3]
    (action_to_clean,) = torch.autograd.grad(
        action_outputs[0].sum(), first_clean, allow_unused=True
    )
    assert action_to_clean is not None and bool((action_to_clean != 0).any())
