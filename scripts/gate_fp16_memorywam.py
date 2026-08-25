#!/usr/bin/env python3
"""Measure the MemoryWAM sequential core under FP16 autocast on one GPU.

Implements the precision half of the project plan's FP16 fit gate: hardware and
kernel support, forward and short-sequence stability, one loss-scaled training
step with finite-gradient checks, and per-layer peak memory. FSDP and the full
pretrained model are deliberately out of scope; this measures the paired-layer
boundary at the real RoboMME token geometry so 30-layer cost can be projected.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from fastwam.models.wan22.memory_cache import NUM_ACTION_TOKENS, NUM_GIST_TOKENS
from fastwam.models.wan22.memory_mot import MemoryWAMSequentialCore
from fastwam.models.wan22.wan_video_dit import DiTBlock, precompute_freqs_cis_3d

# Wan2.2-TI2V-5B video expert and the FastWAM action expert, per
# configs/model/memorywam_robomme.yaml.
VIDEO_HIDDEN, VIDEO_FFN = 3072, 14336
ACTION_HIDDEN, ACTION_FFN = 1024, 4096
NUM_HEADS, HEAD_DIM, FULL_DEPTH = 24, 128, 30
# 224x448 mosaic -> VAE /16 -> 14x28 latent -> patch 2 -> 7x14 grid.
GRID_H, GRID_W = 7, 14


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def build_core(layers: int, device: str) -> MemoryWAMSequentialCore:
    core = MemoryWAMSequentialCore(
        video_blocks=[
            DiTBlock(VIDEO_HIDDEN, HEAD_DIM, NUM_HEADS, VIDEO_FFN, 1e-6)
            for _ in range(layers)
        ],
        action_blocks=[
            DiTBlock(ACTION_HIDDEN, HEAD_DIM, NUM_HEADS, ACTION_FFN, 1e-6)
            for _ in range(layers)
        ],
        video_rope_cache=precompute_freqs_cis_3d(HEAD_DIM, end=1024),
        video_hidden_dim=VIDEO_HIDDEN,
        action_hidden_dim=ACTION_HIDDEN,
        activation_checkpointing=True,
    )
    # Parameters stay FP32. GradScaler refuses to unscale FP16 gradients, so the
    # paper's "FP16 with dynamic loss scaling" means autocast over FP32 master
    # weights, not half-precision parameters.
    return core.to(device=device)


def episode_inputs(frames: int, device: str, dtype: torch.dtype = torch.float32) -> dict:
    tokens = GRID_H * GRID_W
    generator = torch.Generator(device="cpu").manual_seed(0)

    def randn(*shape):
        return torch.randn(*shape, generator=generator).to(device=device, dtype=dtype)

    return {
        "frame_ids": tuple(range(frames)),
        "clean_video_frames": [randn(1, tokens, VIDEO_HIDDEN) for _ in range(frames)],
        "noisy_video_frames": [randn(1, tokens, VIDEO_HIDDEN) for _ in range(frames)],
        "action_frames": [
            randn(1, NUM_ACTION_TOKENS, ACTION_HIDDEN) for _ in range(frames)
        ],
        "clean_video_t_mods": [randn(1, 6, VIDEO_HIDDEN) for _ in range(frames)],
        "noisy_video_t_mods": [randn(1, 6, VIDEO_HIDDEN) for _ in range(frames)],
        "action_t_mods": [randn(1, 6, ACTION_HIDDEN) for _ in range(frames)],
        "video_context": randn(1, 8, VIDEO_HIDDEN),
        "video_context_mask": torch.ones(1, 8, dtype=torch.bool, device=device),
        "action_context": randn(1, 8, ACTION_HIDDEN),
        "action_context_mask": torch.ones(1, 8, dtype=torch.bool, device=device),
        "grid_height": GRID_H,
        "grid_width": GRID_W,
    }


def usable_sdpa_backends(device: str) -> list[str]:
    """Report the attention kernels that actually execute on this architecture."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    query = torch.randn(1, NUM_HEADS, 122, HEAD_DIM, device=device, dtype=torch.float16)
    usable = []
    for name, backend in (
        ("flash", SDPBackend.FLASH_ATTENTION),
        ("mem_efficient", SDPBackend.EFFICIENT_ATTENTION),
        ("math", SDPBackend.MATH),
    ):
        try:
            with sdpa_kernel(backend), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                torch.nn.functional.scaled_dot_product_attention(query, query, query)
            usable.append(name)
        except RuntimeError:
            continue
    return usable


def matmul_slowdown(device: str) -> float:
    """BF16 relative to FP16, to show whether BF16 is emulated rather than native."""
    timings = {}
    for dtype in (torch.float16, torch.bfloat16):
        left = torch.randn(4096, 4096, device=device, dtype=dtype)
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(20):
            left @ left
        torch.cuda.synchronize(device)
        timings[dtype] = time.perf_counter() - start
    return round(timings[torch.bfloat16] / timings[torch.float16], 2)


def finite(tensors) -> bool:
    return all(bool(torch.isfinite(t).all()) for t in tensors)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("this gate requires a CUDA device")
    device = args.device
    capability = torch.cuda.get_device_capability(device)
    report = {
        "schema_version": "memorywam.fp16-gate/v1",
        "device_name": torch.cuda.get_device_name(device),
        "compute_capability": "%d.%d" % capability,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "nccl": ".".join(str(v) for v in torch.cuda.nccl.version()),
        # is_bf16_supported() defaults to counting emulation, and the sdp_*_enabled
        # flags are preferences rather than availability. Report what actually runs.
        "bf16_supported_native": bool(
            torch.cuda.is_bf16_supported(including_emulation=False)
        ),
        "bf16_supported_emulated": bool(torch.cuda.is_bf16_supported()),
        "bf16_vs_fp16_matmul_slowdown": matmul_slowdown(device),
        "usable_sdpa_backends": usable_sdpa_backends(device),
        "layers_measured": args.layers,
        "full_depth": FULL_DEPTH,
        "frames": args.frames,
        "grid": [GRID_H, GRID_W],
        "tokens_per_frame": {
            "video": GRID_H * GRID_W,
            "gist": NUM_GIST_TOKENS,
            "action": NUM_ACTION_TOKENS,
        },
    }

    core = build_core(args.layers, device)
    report["rope_dtype"] = str(core.rope_frame.dtype)
    report["rope_is_complex_after_device_move"] = bool(core.rope_frame.is_complex())
    parameters = sum(p.numel() for p in core.parameters())
    report["parameters_measured"] = parameters
    report["parameters_projected_full_depth"] = int(
        parameters / args.layers * FULL_DEPTH
    )

    # 1) Inference in FP16: one frame, then a short sequence through eviction.
    torch.cuda.reset_peak_memory_stats(device)
    core.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        single = episode_inputs(1, device)
        outputs = core.forward_episode(**single)
        report["single_frame_finite"] = finite(
            [t for branch in outputs for t in branch]
        )
        report["autocast_activation_dtype"] = str(outputs[2][0].dtype)
        core.reset()
        sequence = episode_inputs(args.frames, device)
        outputs = core.forward_episode(**sequence)
        report["sequence_finite"] = finite([t for branch in outputs for t in branch])
    core.reset()
    report["inference_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
    report["inference_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)

    # 2) One training step with dynamic loss scaling and finite-gradient checks.
    core.train()
    core.reset()
    torch.cuda.reset_peak_memory_stats(device)
    scaler = torch.amp.GradScaler("cuda")
    optimizer = torch.optim.AdamW(
        core.parameters(), lr=2e-4, weight_decay=0.01, betas=(0.9, 0.95)
    )
    train_inputs = episode_inputs(args.frames, device)
    before_forward = torch.cuda.memory_allocated(device)
    with torch.autocast("cuda", dtype=torch.float16):
        _, _, noisy, action = core.forward_episode(**train_inputs)
    # Graph retained for backward. Measured directly rather than derived from
    # peak minus optimizer state, which would absorb the AdamW step transient.
    report["activation_bytes"] = torch.cuda.memory_allocated(device) - before_forward
    loss = torch.stack(
        [t.float().pow(2).mean() for t in noisy]
        + [t.float().pow(2).mean() for t in action]
    ).mean()
    report["loss"] = float(loss.detach())
    report["loss_finite"] = bool(torch.isfinite(loss))
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(core.parameters(), 1.0)
    report["initial_loss_scale"] = float(scaler.get_scale())
    report["grad_norm"] = float(grad_norm)
    report["grad_norm_finite"] = bool(torch.isfinite(grad_norm))
    trainable = [p for p in core.parameters() if p.requires_grad]
    report["parameters_with_grad"] = sum(1 for p in trainable if p.grad is not None)
    report["parameters_total"] = len(trainable)
    report["all_grads_finite"] = all(
        bool(torch.isfinite(p.grad).all()) for p in trainable if p.grad is not None
    )
    scaler.step(optimizer)
    scaler.update()
    report["loss_scale_after_step"] = float(scaler.get_scale())
    report["step_applied"] = report["loss_scale_after_step"] >= report[
        "initial_loss_scale"
    ]
    report["train_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
    report["train_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
    # Project the two terms separately: optimizer state shards under FSDP,
    # activations do not. A single linear projection of peak memory conflates
    # them and badly overestimates the sharded per-GPU cost.
    total_memory = torch.cuda.get_device_properties(device).total_memory
    full_parameters = report["parameters_projected_full_depth"]
    state_bytes = full_parameters * 16  # FP32 params + grads + AdamW m and v
    act_per_layer_frame = (
        report["activation_bytes"] / args.layers / args.frames
    )
    report["device_total_memory_bytes"] = total_memory
    report["optimizer_state_projected_bytes"] = state_bytes
    report["activation_bytes_per_layer_frame"] = act_per_layer_frame
    report["activation_projected_bytes_per_frame"] = act_per_layer_frame * FULL_DEPTH
    report["per_gpu_budget"] = {
        str(gpus): {
            str(frames): int(state_bytes / gpus + act_per_layer_frame * FULL_DEPTH * frames)
            for frames in (8, 27, 60, 88)  # min, median, p90, max RoboMME episode
        }
        for gpus in (1, 2, 4, 8)
    }
    report["fits_on_four_gpus_at_median_episode"] = (
        state_bytes / 4 + act_per_layer_frame * FULL_DEPTH * 27
    ) <= total_memory
    report["passed"] = all(
        (
            report["single_frame_finite"],
            report["sequence_finite"],
            report["loss_finite"],
            report["grad_norm_finite"],
            report["all_grads_finite"],
            report["step_applied"],
        )
    )

    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
