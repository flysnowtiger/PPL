# Anonymous Video-based Person Re-identification Implementation

This repository contains an anonymized implementation submitted for double-blind review. Author identities, institutional paths, datasets, pretrained weights, checkpoints, and experiment logs are intentionally omitted.

> **Review-stage release:** The core `ProgressivePartialLearning.forward` implementation is withheld during anonymous review. The complete runnable implementation will be released after acceptance.

## Environment

The code requires Python 3.9+ and a CUDA-compatible PyTorch installation.

```bash
python -m pip install -r requirements.txt
```

## Data

Set `DATASETS.ROOT_DIR` in `configs/resnet50.yml` to the directory containing the dataset folders. The expected layout is:

```text
data/
|-- MARS/
|-- DukeMTMC-VideoReID/
`-- LS-VID/
```

The datasets are not redistributed by this repository. Please obtain them from their official sources and follow their respective terms of use.

## Pretrained backbone

Place the ImageNet-pretrained ResNet-50 checkpoint at:

```text
pretrained/resnet50-19c8e357.pth
```

Alternatively, set `MODEL.PRETRAIN_DIR` to the directory containing that file.

## Training

```bash
python Main.py --config-file configs/resnet50.yml \
  DATASETS.NAME mars \
  DATASETS.ROOT_DIR /path/to/data \
  MODEL.PRETRAIN_DIR /path/to/pretrained \
  OUTPUT_DIR ./outputs
```

The sequence length and GPU identifiers can be overridden through the configuration:

```bash
python Main.py --config-file configs/resnet50.yml \
  DATASETS.SEQ_LEN 8 \
  RUNTIME.GPU_IDS "[0]"
```

## Evaluation

```bash
python Main.py --config-file configs/resnet50.yml \
  TEST.ONLY_TEST True \
  TEST.WEIGHT /path/to/checkpoint.pth \
  DATASETS.ROOT_DIR /path/to/data \
  MODEL.PRETRAIN_DIR /path/to/pretrained \
  RUNTIME.GPU_IDS "[0]"
```

Evaluation first reports begin-interval sampling results and then dense full-video results. The output also includes model complexity, inference latency, throughput, and peak GPU memory.

## Anonymous-review notice

Please do not include identifying information in issues, pull requests, commit metadata, checkpoint metadata, or experiment logs while the submission is under double-blind review. Third-party references retained in source comments indicate upstream implementations and are not author-identifying information for this submission.
