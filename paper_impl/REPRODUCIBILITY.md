# Reproducibility and claim-control checklist

## Input provenance
The package accepts prepared three-channel detector images. It does not reconstruct upstream DWI/FLAIR preparation and must not be described as an explicit cross-modal fusion implementation.

## Split provenance
Use explicit split files and archive them with each run. If patient IDs are unavailable, describe the split as image-level and do not claim patient-level independence. `audit_dataset.py` verifies image-ID disjointness and annotation availability.

## Metric provenance
- `--metric voc50`: single-threshold AP@0.50.
- `--metric coco`: COCO mAP@[0.50:0.95] through pycocotools.
Never relabel one as the other or rank values from incompatible definitions.

## EMA topology-aware teacher
The teacher is initialized from the student, gradients are disabled, it stays in evaluation mode, and it is updated after each optimizer step. It is used for feature targets during training and discarded at inference.

## Feature distillation
The default objective uses the last Dynamic BiFPN cell. BU+ is the intermediate state and BU- the final state. Each sample/level feature is flattened and L2-normalized before MSE comparison.

## Polarity interpretation
BU- is not subtractive. Polarity acts on scalar fusion parameters before softplus. Effective post-softplus coefficients remain non-negative. Epsilon-stabilized normalization yields a coefficient sum slightly below one.

## Runtime provenance
`benchmark_torch.py` reports PyTorch CUDA timing only. TensorRT numbers require a separately documented TensorRT build/runtime and must not be inferred from PyTorch timing. `export_onnx.py` exports raw heads for downstream engine construction.

## Historical-result boundary
The repository contains legacy research files and historical values. `paper_impl/` is the canonical executable specification going forward. Historical table values are not retroactively attributed to this implementation unless rerun and archived.
