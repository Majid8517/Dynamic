"""Canonical P7 is opt-in; legacy topology and state_dict keys remain unchanged."""
import torch
from paper_impl.dynamic_bifpn import DynamicBiFPNCell, DynamicBiFPN
from paper_impl.config import DynamicBiFPNConfig

def _cell(*, canonical, dual, shared):
    return DynamicBiFPNCell(
        8, epsilon=1e-4, alpha_max=0.95, alpha_init=0.02,
        adaptive_weights=True, use_polarity=True,
        use_softplus=True, use_positive_normalization=True,
        dual_bottom_up=dual, share_bottom_up=shared,
        p7_canonical_fusion=canonical,
    ).eval()

def _pyramid():
    torch.manual_seed(11)
    return [torch.randn(1, 8, s, s) for s in (32, 16, 8, 4, 2)]

def test_default_preserves_legacy_3_input_p7_and_checkpoint_keys():
    assert DynamicBiFPNConfig().p7_canonical_fusion is False
    old = _cell(canonical=False, dual=True, shared=True)
    assert old.bu_fusions[3].num_inputs == 3
    assert tuple(old.bu_fusions[3].edge_weights.shape) == (3,)
    state = old.state_dict()
    copy = _cell(canonical=False, dual=True, shared=True)
    copy.load_state_dict(state, strict=True)
    with torch.inference_mode():
        a = old(_pyramid())
        b = copy(_pyramid())
    assert all(torch.equal(x, y) for x, y in zip(a, b))

def test_corrected_p7_two_distinct_inputs_on_both_passes():
    for dual,shared in [(False,False),(True,False),(True,True)]:
        cell = _cell(canonical=True, dual=dual, shared=shared)
        assert [m.num_inputs for m in cell.bu_fusions] == [3,3,3,2]
        if dual and not shared:
            assert [m.num_inputs for m in cell.bu_minus_fusions] == [3,3,3,2]
        got=[]
        def watch(module, args):
            feats = args[0]
            got.append(len(feats))
            assert not torch.equal(feats[0], feats[-1]), "P7 must fuse distinct inputs"
        handles=[cell.bu_fusions[3].register_forward_pre_hook(watch)]
        if dual and not shared:
            handles.append(cell.bu_minus_fusions[3].register_forward_pre_hook(watch))
        with torch.inference_mode():
            out,state=cell(_pyramid(),return_state=True)
        for handle in handles: handle.remove()
        assert [x.shape[-2:] for x in out] == [(32,32),(16,16),(8,8),(4,4),(2,2)]
        assert got == ([2,2] if dual and shared else ([2,2] if dual else [2]))

def test_canonical_p7_gradients_and_parameter_sharing():
    cell=_cell(canonical=True,dual=True,shared=True).train()
    out,state=cell(_pyramid(),return_state=True)
    sum(x.square().mean() for x in out).backward()
    assert cell.bu_minus_fusions is None
    assert cell.bu_fusions[3].edge_weights.grad is not None
    assert cell.bu_fusions[3].edge_weights.grad.abs().sum().item()>0

def test_corrected_two_input_fusion_vs_legacy_at_equivalent_init():
    # Original P7: [p7, p7, propagated] with weights 1:1:1.
    # Canonical P7: [p7, propagated] with weights 2:1.
    # Compare fusion coefficients, not random refinement layers.
    a=_cell(canonical=False,dual=False,shared=False)
    b=_cell(canonical=True,dual=False,shared=False)
    with torch.no_grad():
        # Use linear normalized weights, avoiding nonlinear softplus ambiguity.
        a.bu_fusions[3].edge_weights.fill_(1)
        b.bu_fusions[3].edge_weights.copy_(torch.tensor([2.0,1.0]))
        a.bu_fusions[3].use_softplus=False
        b.bu_fusions[3].use_softplus=False
        a.bu_fusions[3].use_polarity=False
        b.bu_fusions[3].use_polarity=False
        original=a.bu_fusions[3].effective_weights()
        corrected=b.bu_fusions[3].effective_weights()
        assert torch.allclose(torch.stack([original[:2].sum(),original[-1]]),corrected,rtol=1e-5,atol=1e-5)
