import torch
from paper_impl.losses import topology_distillation_loss
from paper_impl.dynamic_bifpn import DynamicBiFPN

def test_topology_distillation_is_finite():
    fpn=DynamicBiFPN([8,16,32],16,repeats=1)
    feats=[torch.randn(2,8,16,16),torch.randn(2,16,8,8),torch.randn(2,32,4,4)]
    _,s=fpn(feats,return_states=True)
    _,t=fpn(feats,return_states=True)
    loss=topology_distillation_loss(s,t)
    assert torch.isfinite(loss)
