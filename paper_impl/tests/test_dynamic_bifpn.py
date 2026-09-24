import torch
from paper_impl.dynamic_bifpn import DynamicBiFPN, PolarityAwareFusion

def test_positive_epsilon_normalization():
    m=PolarityAwareFusion(3,epsilon=1e-4,alpha_max=0.95,alpha_init=0.02)
    for state in (-1,0,1):
        w=m.normalized_weights(state)
        assert torch.all(w>=0)
        assert float(w.sum().detach())<1.0
        assert float(w.sum().detach())>0.99

def test_alpha_bounds_and_initialization():
    m=PolarityAwareFusion(4,alpha_max=0.95,alpha_init=0.02)
    a=m.effective_alpha()
    assert torch.all(a>=0) and torch.all(a<0.95)
    assert torch.allclose(a,torch.full_like(a,0.02),atol=1e-5)

def test_td_bu_shapes_and_backward():
    fpn=DynamicBiFPN([24,40,112],32,repeats=2)
    feats=[torch.randn(2,24,32,32),torch.randn(2,40,16,16),torch.randn(2,112,8,8)]
    out,states=fpn(feats,return_states=True)
    assert len(out)==5 and len(states)==2
    assert [x.shape[-2:] for x in out]==[(32,32),(16,16),(8,8),(4,4),(2,2)]
    sum(x.mean() for x in out).backward()
    assert any(p.grad is not None for p in fpn.parameters())

def test_bu_parameter_sharing_is_structural():
    fpn=DynamicBiFPN([24,40,112],32,repeats=1)
    cell=fpn.cells[0]
    assert len(cell.bu_fusions)==4 and len(cell.bu_refine)==4
