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
| Eight learned gist tokens per older frame | Paper-specified | MemoryWAM paper |
| Exact integration points and cache containers | Implementation inference | Must be verified against uncached reference tests |
| Two-camera 224x448 mosaic and 8-D action horizon 16 | RoboMME adaptation | Project plan |
| FP16 with dynamic loss scaling | Hardware adaptation | Project plan; paper used BF16 |

The raw HDF5 dataset is read from
`/data/ed1116/Datasets/robomme_data_h5`. Processed artifacts, checkpoints, and
runs live under `/data/ed1116/robomme` and are never committed.
