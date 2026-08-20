# Upstream architecture and MemoryWAM reimplementation map

This repository is an independent, paper-guided reimplementation. It is not
the official MemoryWAM training or inference code. The labels below are
normative:

- **Inherited**: preserve FastWAM behavior unless a MemoryWAM requirement
  conflicts with it.
- **Paper**: required by the MemoryWAM paper.
- **Inference**: an explicit project decision where the paper is incomplete.
- **RoboMME**: an embodiment or benchmark adaptation, not a paper claim.

## Pinned evidence

| Source | Revision | Evidence used here |
| --- | --- | --- |
| FastWAM | `7faa71108368fbb3b6885649f112af607427a2d4` | The inherited source tree and symbols listed below. |
| MemoryWAM reference | `dc82a9d760e751ae88f2e0ae521af7c7ed597dfb` | `README.md`, figures, and the paper link. The pinned tree contains only `README.md`, `LICENSE.txt`, and assets; its TODO still says training and inference code are unreleased. |
| MemoryWAM paper | arXiv `2606.20562` | Sections 3.2, 3.3, 4.1, and Appendix A.1. |

The reference remote is documentation evidence only. No implementation is
copied or inferred to be official.

## FastWAM preservation map

| Area | Exact inherited path and symbols | What remains intact | MemoryWAM integration boundary |
| --- | --- | --- | --- |
| Causal video VAE | `src/fastwam/models/wan22/wan_video_vae.py`: `WanVideoVAE38`, `VideoVAE38_`, `encode`; `src/fastwam/models/wan22/fastwam.py`: `FastWAM._encode_video_latents`, `_encode_input_image_latents_tensor` | Wan2.2 48-channel causal latents, spatial downsample 16, temporal downsample 4, and frozen VAE weights. | Feed clean observation latents into the persistent video cache. Full-prefix causal encoding is the correctness reference; a streaming shortcut is allowed only after equivalence is measured. |
| Wan video DiT | `src/fastwam/models/wan22/wan_video_dit.py`: `WanVideoDiT`, `DiTBlock`, `precompute_freqs_cis_3d`, `get_freqs`, `prepare`, `post`, `build_video_to_video_mask` | Wan2.2 patch embedding, 3D RoPE, text cross-attention, timestep modulation, blocks, and video head. | Add per-frame gist tokens, absolute temporal coordinates, and the hybrid-memory visibility mask without replacing the pretrained expert. |
| Paired MoT experts | `src/fastwam/models/wan22/mot.py`: `MoT`, `_forward_joint_layer`, `forward_joint_core`, `prefill_video_cache_tensor`, `forward_action_with_video_cache_tensor` | Separate video/action parameters with joint mixed self-attention and per-expert cross-attention/FFNs. | Make the layer boundary own persistent anchor/recent/gist K/V and expose an FSDP/checkpoint-safe paired video/action layer. |
| Action DiT | `src/fastwam/models/wan22/action_dit.py`: `ActionDiT`, `prepare`, `post`, `get_freqs`; `src/fastwam/models/wan22/fastwam.py`: `FastWAM.from_wan22_pretrained` | The 1B action expert, action encoder/head, flow timestep modulation, and the requirement that depth, heads, and head dimension match the video expert. | Replace its independent 1D sequence RoPE with coordinates in the video expert's 3D RoPE basis and reconstruct the paper's action-token visibility rather than selecting it implicitly. |
| Model composition and losses | `src/fastwam/models/wan22/fastwam.py`: `FastWAM`, `build_inputs`, `_joint_denoise_core`, `training_loss`; `src/fastwam/models/wan22/fastwam_idm.py`: `FastWAMIDM`, `_teacher_forcing_training_denoise_core`, `training_loss` | Joint video/action flow-matching supervision, frozen VAE/text encoder, proprio token projection, and teacher-forced clean-video conditioning as the closest released starting point. | Train an episode-autoregressive clean/noisy/gist sequence whose mask matches deployment; do not use a clip-local full-conditioning cache as a substitute. |
| Flow scheduler | `src/fastwam/models/wan22/schedulers/scheduler_continuous.py`: `WanContinuousFlowMatchScheduler` | Continuous interpolation, velocity target, scheduler weighting, and Euler inference interface. | Implement the paper's shifted-logit-normal timestep draw and preserve shifts 5.0 (video) and 1.0 (action). |
| Training entry point | `scripts/train.py`: `main`; `src/fastwam/runtime.py`: `create_fastwam_idm`, `build_datasets`, `run_training`; `src/fastwam/trainer.py`: `Wan22Trainer`, `train` | Hydra construction, AdamW loop, gradient accumulation/clipping, validation hooks, and reproducible dataloader resume. | Add RoboMME episode batches and FP16 FSDP. Existing launchers under `scripts/train_zero{1,2}.sh` are Accelerate/DeepSpeed ZeRO launchers, not the paper's FSDP setup. |
| Checkpointing | `src/fastwam/models/wan22/fastwam.py`: `save_checkpoint`, `load_checkpoint`; `src/fastwam/trainer.py`: `_save_weights_checkpoint`, `save_checkpoint`, `load_training_state` | Separate portable weights and resumable optimizer/scheduler/dataloader state are useful interfaces. | Save/load FSDP-safe full model state, including gist parameters and any non-parameter cache metadata needed to reproduce reset/prefill behavior. Runtime KV cache itself is episode state, not checkpoint state. |
| Inference | `src/fastwam/models/wan22/fastwam.py`: `infer_action`, `_denoise_action_with_video_cache`; `src/fastwam/models/wan22/mot.py`: the two tensor cache methods above | Prefill video K/V once, then reuse it across every action denoising step; video generation can be bypassed. | Make prefill incremental and stateful across policy calls, retaining hybrid K/V per layer and clearing it only on `reset()`. |
| Model downloads | `src/fastwam/models/wan22/helpers/loader.py`: `load_wan22_ti2v_5b_components`, `_resolve_configs`, `_load_registered_model`; `src/fastwam/models/wan22/helpers/io.py`: `ModelConfig.download_if_necessary`, `download` | Registered Wan component loading and state-dict conversion. | Require Hugging Face first, an explicit model revision, and the existing HF cache. `ModelConfig` currently defaults to ModelScope and materializes files under `./checkpoints`, so it is not sufficient unchanged. |

## Why the released classes are not MemoryWAM

`FastWAM`, `FastWAMJoint`, and `FastWAMIDM` operate on a bounded clip or one
input image. Their action-only cache is created inside one inference call and
contains full video tokens; there are no learnable gist tokens, task-onset
anchors, recent-frame eviction, absolute episode frame IDs, or persistent
`reset()` state. `FastWAMIDM` is useful for clean-video teacher forcing, but its
mask exposes a full conditional clip and its optional conditioning corruption
does not implement the paper's per-frame hybrid cache.

The current `ActionDiT.get_freqs` is 1D, whereas the paper places both experts
in one 3D positional frame. The current `MoT` also reaches directly into
`expert.blocks[layer_idx]` and explicitly rejects expert gradient
checkpointing. Consequently, wrapping the existing `DiTBlock` objects alone
cannot provide the paper's per-block activation checkpointing or a clean FSDP
unit; a paired video/action layer boundary is required. Finally, the current
main-rank `mot.state_dict()` weight save is not, by itself, an FSDP state-dict
protocol.

## Required target behavior

| Requirement | Provenance | Target in this repository |
| --- | --- | --- |
| Wan2.2-TI2V-5B video expert: hidden 3072, FFN 14336, 24 heads x 128, 30 blocks, patch `1x2x2` | Paper and FastWAM | Preserve pretrained topology and initialization. |
| Separate action expert: hidden 1024, FFN 4096, 24 heads x 128, 30 blocks | Paper and FastWAM | Preserve the interpolated FastWAM action backbone; change RoboMME input/output width to 8. |
| Video prediction during training; no video generation during policy inference | Paper | Keep dense video flow loss and use action-only closed-loop inference. |
| Per-layer hybrid video K/V | Paper | Two full-frame initial anchors, four full-frame recent non-anchor frames, and eight persistent gist tokens for every frame; evict only an old frame's full K/V. |
| Gist construction and visibility | Paper | Attach learnable gist queries to each clean frame. Each gist attends to that frame and permitted history; later video/action queries use the gist after the frame's full K/V is evicted. |
| Train/deploy attention equivalence | Paper | Build the per-frame autoregressive clean/noisy-video/gist/action mask so training visibility reproduces the inference cache exactly. |
| Joint 3D positional frame | Paper | Use absolute episode frame coordinates for video, gist, and action keys/queries. |
| Flow training | Paper | 1000 timesteps, shifted-logit-normal draw, shifts 5.0/1.0, equal video/action loss weights, conditioning-frame Gaussian mixing with probability 1.0, AdamW `lr=2e-4`, weight decay `0.01`, betas `(0.9, 0.95)`, and grad clip 1.0. |
| Distributed precision | Paper plus hardware adaptation | Paper used BF16 FSDP and per-block checkpointing. RoboMME uses FP16 with dynamic loss scaling because Quadro RTX 8000 lacks BF16 support; this is not a parity claim. |
| RoboMME observation/action contract | RoboMME | Horizontal front+wrist `224x448` mosaic, producing a `14x28` VAE grid and `7x14=98` DiT tokens per latent frame; 8-D absolute action/state and horizon 16. |
| Closed-loop update | Paper and RoboMME | After a 16-action chunk, ingest sub-step observations `{3, 7, 11, 15}` and update memory without generating video. |

## Approved paper-consistent inferences

These choices are intentional project decisions, not claims about unreleased
official code:

1. Place gist tokens at the associated frame's temporal coordinate and use the
   one-past-grid spatial marker `(h=H, w=W)`.
2. Place each action token at the current frame coordinate, use the sentinel
   spatial row, and encode action sub-step `0..15` along the other spatial
   coordinate so action queries share the video 3D RoPE basis.
3. Use the conditioning-noise mixing ratio as the conditioning video's
   timestep input/modulation, in addition to using it for linear Gaussian
   mixing.
4. Use shifted-logit-normal base parameters `mu=0`, `sigma=1`, followed by the
   paper's branch shifts.
5. Treat full-prefix causal-VAE encoding as the correctness oracle. Any local
   or cached encoding must match it before use.
6. Backpropagate through the full training episode. If measured OOM makes that
   impossible, truncation or cache detachment is a new architectural decision
   and must be approved rather than enabled silently.

For RoboMME, the first two task-visible observations are the anchors, including
demonstration history when present. No unavailable oracle event boundary is
introduced.

## Semantics still unresolved by released evidence

Do not choose these silently during implementation:

- The paper gives relationships and a figure, but not the complete Boolean
  matrix for every clean-video, noisy-video, gist, and current-action
  query/key pair. In particular, same-frame noisy-video-to-gist visibility and
  gist visibility to current action tokens must be fixed by a documented mask
  reconstruction and unit tests before model training.
- The paper does not state whether the eight learnable gist embeddings are one
  shared parameter bank reused at every frame or independently parameterized
  by frame. An unbounded per-frame parameter table would conflict with
  arbitrary episode lengths, but the final choice still needs to be recorded.
- Full-prefix VAE is selected as the oracle, not as the required fast path.
  Incremental VAE feature-cache semantics and the batching of the four raw
  sub-step mosaics remain an empirical equivalence question.
- Full-episode gradients are selected. No fallback truncation length,
  detachment boundary, or recomputation policy is authorized unless an OOM is
  measured and the replacement is explicitly decided.

Every implementation commit should update this map when an open item becomes
a measured fact or an approved inference.
