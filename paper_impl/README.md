# Stability-Aware Dynamic BiFPN — canonical paper implementation

This directory is the reviewer-facing implementation of the proposed **Stability-Aware Dynamic BiFPN–EfficientDet** method. Legacy research files at repository root are retained for provenance; this directory is the recommended reproducible entry point.

## Scope
The method modifies **cross-scale feature aggregation** in an EfficientDet-style detector. It does **not** implement a DWI-FLAIR cross-modal fusion network. The detector consumes one three-channel image tensor and operates across P3-P7.

Implemented components:
1. EfficientNet hierarchical feature extraction.
2. Polarity-aware pre-normalization modulation.
3. Softplus-positive weights.
4. Epsilon-stabilized positive normalization.
5. Sequential TD -> BU+ -> BU- propagation.
6. Shared BU+/BU- fusion and convolution parameters.
7. Depthwise-separable refinement.
8. Training-only EMA topology-aware distillation.
9. L2-normalized intermediate/final feature consistency.
10. Polarity magnitude regularization.
11. Detection + distillation + polarity objective.
12. Explicit VOC-style AP@0.50 and COCO-style mAP@[0.50:0.95] evaluators.

See `METHOD_TO_CODE.md` and `REPRODUCIBILITY.md`.

## Installation
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r paper_impl/requirements.txt
```

## Tests
```bash
python -m paper_impl.smoke_test
pytest -q paper_impl/tests
```

## Prepared ISLES detector data
No undocumented MRI preprocessing is performed. Detector-side preprocessing is explicit: RGB conversion, deterministic square resize, tensor conversion, and configurable normalization. Horizontal flipping is disabled by default.

Example:
```text
data/isles/
  images/
  annotations/
  train_ids.txt
  eval_ids.txt
```

Audit a split before training:
```bash
python -m paper_impl.audit_dataset --train-ids train_ids.txt --eval-ids eval_ids.txt --annotation-dir data/isles/annotations
```

Train D0:
```bash
python -m paper_impl.train --config paper_impl/configs/isles_d0.yaml
```

Evaluate with an explicit metric:
```bash
python -m paper_impl.evaluate --config eval_isles.yaml --checkpoint runs/.../checkpoint_epoch_070.pth --metric coco
python -m paper_impl.evaluate --config eval_isles.yaml --checkpoint runs/.../checkpoint_epoch_070.pth --metric voc50
```

Never label `voc50` output as COCO mAP.

## COCO pre-validation
```bash
python -m paper_impl.train --config paper_impl/configs/coco_d0.yaml
```
The COCO stage characterizes general detector behavior, not clinical validity.

## EMA teacher versus ordinary model EMA
`paper_impl.ema.EMATeacher` is a training-only **feature teacher**, not an evaluation/checkpoint moving average. It produces hierarchical targets and is discarded at deployment.

## Profiling and deployment
```bash
python -m paper_impl.profile_model --config paper_impl/configs/isles_d0.yaml
python -m paper_impl.export_onnx --config ... --checkpoint ... --output dynamic_bifpn.onnx
python -m paper_impl.benchmark_torch --config ... --checkpoint ... --fp16
```
PyTorch timing is labeled as PyTorch timing and must not be reported as TensorRT latency.

## Historical-result boundary
This package implements the manuscript method. It does not automatically establish provenance for historical table values. A result should be described as reproduced by this package only after the corresponding data split, checkpoint, evaluator, and runtime logs are archived.


## Executable controlled ablations

The repository now exposes a prospective **A0-A8** mechanism-isolation matrix
under `paper_impl/configs/ablations/`.  Each successive configuration changes
exactly one mechanism:

```text
A0  fixed-fusion TD->BU reference
A1  + learnable adaptive edge weights
A2  + polarity-aware modulation
A3  + Softplus-positive re-parameterization
A4  + epsilon-stabilized positive normalization
A5  + complementary BU- pass with independent parameters
A6  + shared BU+/BU- parameters
A7  + polarity-magnitude regularization
A8  + EMA topology-aware distillation (canonical full model)
```

Validate the matrix:

```bash
python -m paper_impl.ablation --validate
```

Train a selected variant after setting dataset paths in its YAML:

```bash
python -m paper_impl.train \
  --config paper_impl/configs/ablations/coco_d0_a8_full_with_ema.yaml
```

See `ABLATION_PROTOCOL.md` for the exact switch matrix and claim boundary.
The historical numbers in manuscript Table 3 are not automatically attributed
to A0-A8; they become reproduced results only after controlled re-execution and
archival of the corresponding run artifacts.
