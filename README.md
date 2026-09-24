# Stability-Aware Dynamic BiFPN for MRI-Based Ischemic Stroke Lesion Detection

This repository accompanies the manuscript on **Stability-Aware Dynamic BiFPN–EfficientDet** for regulated multi-scale feature aggregation.

## Canonical implementation

The reviewer-facing, reproducible implementation is in:

- **`paper_impl/`** — executable paper method
- **`paper_impl/METHOD_TO_CODE.md`** — manuscript-to-code traceability
- **`paper_impl/REPRODUCIBILITY.md`** — claim-control and reproducibility checklist
- **`paper_impl/configs/`** — explicit COCO and ISLES experiment configurations
- **`paper_impl/tests/`** — unit tests for fusion, EMA, anchors, and distillation

The older Python files in the repository root are retained as **legacy research code/provenance** and should not be treated as the canonical implementation of the final manuscript.

## What the method implements

The paper implementation keeps the EfficientNet backbone and EfficientDet-style classification/regression heads conceptually fixed while replacing conventional pyramid aggregation with a Dynamic BiFPN containing:

- polarity-aware pre-normalization modulation;
- softplus-positive fusion weights;
- epsilon-stabilized positive normalization;
- TD -> BU+ -> BU- propagation;
- shared BU+/BU- fusion/refinement parameters;
- depthwise-separable refinement;
- training-only EMA topology-aware feature distillation;
- polarity-magnitude regularization.

The method is a **cross-scale feature aggregation** architecture. It is not presented as an explicit DWI-FLAIR cross-modal fusion network.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r paper_impl/requirements.txt

python -m paper_impl.smoke_test
pytest -q paper_impl/tests
```

For training and evaluation instructions, see [paper_impl/README.md](paper_impl/README.md).

## Metric semantics

The repository intentionally separates:

- single-threshold **AP@0.50** (`--metric voc50`), and
- COCO-style **mAP@[0.50:0.95]** (`--metric coco`).

These metrics must not be conflated.

## Reproducibility boundary

The canonical code implements the final method, but it does not retroactively establish provenance for historical manuscript values. Historical results should only be attributed to this implementation after controlled re-execution with the corresponding split, checkpoint, evaluator, and runtime records.

## License / attribution

Legacy root files retain their original source headers and licensing notices where present. The new `paper_impl/` code is written specifically for this research repository. Add a repository-level license before third-party redistribution if required by your institution or journal.
