"""Fast differentiability/shape smoke test for the method core.
This does not claim to reproduce manuscript accuracy.
"""
import torch
from .dynamic_bifpn import DynamicBiFPN

def main():
    torch.manual_seed(0)
    fpn=DynamicBiFPN([24,40,112],out_channels=32,repeats=2)
    feats=[torch.randn(2,24,32,32,requires_grad=True),torch.randn(2,40,16,16,requires_grad=True),torch.randn(2,112,8,8,requires_grad=True)]
    out,states=fpn(feats,return_states=True)
    assert [tuple(x.shape[-2:]) for x in out]==[(32,32),(16,16),(8,8),(4,4),(2,2)]
    loss=sum(x.square().mean() for x in out)+1e-3*fpn.polarity_regularizer(); loss.backward()
    assert torch.isfinite(loss); print("DynamicBiFPN smoke test passed",float(loss.detach()))

if __name__=="__main__": main()
