# Provenance and implementation boundary

## Source revisions

| Component | Source | Revision | Role |
| --- | --- | --- | --- |
| FastWAM | `git@github.com:yuantianyuan01/FastWAM.git` | `7faa71108368fbb3b6885649f112af607427a2d4` | Wan world-action model foundation |
| MemoryWAM reference | `git@github.com:yangsizhe/MemoryWAM.git` | `dc82a9d760e751ae88f2e0ae521af7c7ed597dfb` | Paper/project documentation only |
| RoboMME policy learning | `git@github.com:RoboMME/robomme_policy_learning.git` | `ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b` | Clean policy/serving contract reference |
| RoboMME benchmark | `git@github.com:RoboMME/robomme_benchmark.git` | `856bc3a189d4172f3f47dbee4424d585f8d78db3` | Simulator interface reference |

## Status

This is an independent reimplementation, not official MemoryWAM source. The
MemoryWAM paper and public project repository specify the target behavior;
FastWAM supplies the released Wan2.2 world-action implementation.

## Feature provenance categories

| Feature | Category | Initial source |
| --- | --- | --- |
| Wan2.2 video VAE/DiT and action expert | Inherited | FastWAM |
| Joint video/action flow matching | Inherited and paper-aligned | FastWAM and MemoryWAM paper |
| Two full-frame anchors | Paper-specified | MemoryWAM paper |
| Four recent full frames | Paper-specified | MemoryWAM paper |
| Eight retained gist K/V tokens per frame | Paper-specified | MemoryWAM paper |
| One shared eight-token learned gist input bank | Implementation inference | Fixed for unbounded episode length; verified for reuse and gradients |
| Exact clean/gist/noisy/action visibility matrix | Implementation inference constrained by paper and released FastWAM | Direct mask tests, a noisy-target gradient-path test, and cached/reference-equivalence tests |
| Disjoint video/gist/action rows in the shared 3-D basis | Implementation inference | Coordinate-collision test across grids narrower and wider than the action horizon |
| Paired-layer cache/checkpoint boundary | Implementation inference | CPU dense FastWAM equivalence and checkpointed cross-frame-gradient tests |
| Shifted-logit-normal base draw (`mu=0`, `sigma=1`) before branch shift | Implementation inference constrained by paper | Seeded scheduler reference test; training selection not wired yet |
| Two-camera 224x448 mosaic and 8-D action horizon 16 | RoboMME adaptation | Project plan |
| FP16 with dynamic loss scaling | Hardware adaptation | Project plan; paper used BF16 |

The raw HDF5 dataset is read from
`/data/ed1116/Datasets/robomme_data_h5`. Processed artifacts, checkpoints, and
runs live under `/data/ed1116/robomme` and are never committed.

## Environments

The repository has no in-tree virtual environment. The interpreter lives
outside Git at `/data/ed1116/robomme/envs/memorywam`, which also provides
`ruff`. Run the repository-owned suite from the repository root:

```bash
/data/ed1116/robomme/envs/memorywam/bin/python -m pytest -q tests
```

Use `tests` explicitly. A bare `pytest` at the root also collects the vendored
RoboTwin tests under `third_party`, whose collection fails only because the
optional `openai` and `sapien` packages are absent.

## Evaluation rollout budget

Simulator evaluation uses **10 rollouts per task** (160 per method), not the
official 50. This is an evaluation-time budget only: it must never influence
training data, hyperparameters, checkpoint selection, or VQA corpus
construction, all of which use the full 100 demonstrations per task with the
canonical 90/10 episode split.
