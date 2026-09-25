from pathlib import Path
import torch

from paper_impl.ablation import ABLATION_CONFIGS, EXPECTED, flag_tuple, load_variant, validate_matrix
from paper_impl.dynamic_bifpn import DynamicBiFPN, PolarityAwareFusion

def test_ablation_matrix_changes_one_mechanism_per_step():
    validate_matrix()
    assert list(ABLATION_CONFIGS) == [f"A{i}" for i in range(9)]
    for key, expected in EXPECTED.items():
        assert flag_tuple(load_variant(key)) == expected

def test_fixed_fusion_reference_is_unweighted_sum():
    m = PolarityAwareFusion(
        2, adaptive_weights=False, use_polarity=False,
        use_softplus=False, use_positive_normalization=False,
    )
    x = torch.ones(1, 1, 4, 4)
    y = 2 * torch.ones_like(x)
    out = m([x, y], polarity=0)
    assert torch.allclose(out, x + y)

def test_softplus_variant_is_positive_without_normalization():
    m = PolarityAwareFusion(
        3, adaptive_weights=True, use_polarity=True,
        use_softplus=True, use_positive_normalization=False,
    )
    with torch.no_grad():
        m.edge_weights.copy_(torch.tensor([-3.0, 0.0, 2.0]))
    w = m.effective_weights(+1)
    assert torch.all(w > 0)
    assert not torch.isclose(w.sum(), torch.tensor(1.0))

def test_positive_normalization_sums_slightly_below_one():
    m = PolarityAwareFusion(
        3, epsilon=1e-4, adaptive_weights=True, use_polarity=True,
        use_softplus=True, use_positive_normalization=True,
    )
    w = m.effective_weights(+1)
    assert torch.all(w >= 0)
    assert 0.99 < float(w.sum()) < 1.0

def _features():
    return [
        torch.randn(2, 8, 16, 16),
        torch.randn(2, 16, 8, 8),
        torch.randn(2, 32, 4, 4),
    ]

def test_single_bottom_up_variant_has_no_minus_modules():
    fpn = DynamicBiFPN(
        [8, 16, 32], 16, repeats=1,
        adaptive_weights=True, use_polarity=True, use_softplus=True,
        use_positive_normalization=True, dual_bottom_up=False,
        share_bottom_up=False, use_polarity_regularization=False,
    )
    out, states = fpn(_features(), return_states=True)
    cell = fpn.cells[0]
    assert cell.bu_minus_fusions is None
    assert cell.bu_minus_refine is None
    assert all(a is b for a,b in zip(states[0].bu_plus, states[0].bu_minus))
    assert len(out) == 5

def test_unshared_and_shared_dual_bottom_up_are_structurally_distinct():
    unshared = DynamicBiFPN(
        [8,16,32],16,repeats=1,dual_bottom_up=True,share_bottom_up=False,
        use_polarity_regularization=False,
    )
    shared = DynamicBiFPN(
        [8,16,32],16,repeats=1,dual_bottom_up=True,share_bottom_up=True,
        use_polarity_regularization=False,
    )
    assert unshared.cells[0].bu_minus_fusions is not None
    assert unshared.cells[0].bu_minus_refine is not None
    assert shared.cells[0].bu_minus_fusions is None
    assert shared.cells[0].bu_minus_refine is None
    assert sum(p.numel() for p in unshared.parameters()) > sum(p.numel() for p in shared.parameters())

def test_a8_is_canonical_full_switch_state():
    cfg = load_variant("A8")
    d = cfg.model.dynamic
    assert all([
        d.adaptive_weights, d.use_polarity, d.use_softplus,
        d.use_positive_normalization, d.dual_bottom_up,
        d.share_bottom_up, d.use_polarity_regularization,
        d.use_ema_distillation,
    ])
