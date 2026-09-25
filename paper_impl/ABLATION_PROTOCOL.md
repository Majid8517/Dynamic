# Executable controlled ablation protocol

The public implementation exposes a **prospective A0-A8 mechanism-isolation
matrix** for controlled re-execution on COCO-2017 with EfficientDet-D0 settings.
Each successive configuration changes exactly one mechanism while keeping the
host backbone family, prediction-head design, optimizer, training duration,
input size, seed, and evaluator definition fixed.

| ID | Increment from previous configuration | AW | Polarity | Softplus | PN | Dual BU | WS | Pol. reg. | EMA |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 | Fixed-fusion TD->BU reference | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| A1 | Learnable adaptive edge weights | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| A2 | Polarity-aware modulation | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| A3 | Softplus-positive re-parameterization | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| A4 | Epsilon-stabilized positive normalization | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 |
| A5 | Complementary BU- pass, independent parameters | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 0 |
| A6 | Shared BU+/BU- fusion/refinement parameters | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 |
| A7 | Polarity-magnitude regularization | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 |
| A8 | EMA topology-aware distillation | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |

**PN** means epsilon-stabilized positive normalization. **WS** means
BU+/BU- weight/refinement sharing. A8 is the canonical full paper model.

Validate the matrix without training:

```bash
python -m paper_impl.ablation --validate
```

Run any configuration after replacing the COCO paths in the YAML file:

```bash
python -m paper_impl.train \
  --config paper_impl/configs/ablations/coco_d0_a4_positive_normalization.yaml
```

Evaluate the resulting checkpoint using the same COCO-style evaluator:

```bash
python -m paper_impl.evaluate \
  --config paper_impl/configs/ablations/coco_d0_a4_positive_normalization.yaml \
  --checkpoint runs/ablations/.../checkpoint_epoch_030.pth \
  --metric coco
```

## Relationship to historical Table 3

The numerical values currently retained in manuscript Table 3 are **historical
records**.  They are not automatically assigned to A0-A8.  The A0-A8 matrix is
the public executable protocol for future controlled re-execution.  Historical
values should be replaced or relabeled as reproduced only after the relevant
configuration has been rerun with archived resolved YAML, checkpoint, raw
evaluator output, Git commit, software/hardware information, and runtime logs.

This separation prevents circular provenance: executable availability supports
method traceability, while numerical reproduction requires an actual run.
