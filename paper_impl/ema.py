from __future__ import annotations
import copy
import torch
import torch.nn as nn

class EMATeacher(nn.Module):
    """Training-only EMA feature teacher, distinct from checkpoint/eval EMA."""
    def __init__(self, student: nn.Module, rho: float = 0.999):
        super().__init__()
        if not (0.0 <= rho < 1.0):
            raise ValueError("rho must be in [0, 1)")
        self.rho = float(rho)
        self.model = copy.deepcopy(student).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def initialize_from(self, student):
        self.model.load_state_dict(student.state_dict(), strict=True)
        self.model.eval()

    @torch.no_grad()
    def update(self, student):
        ss = student.state_dict()
        ts = self.model.state_dict()
        for name, t in ts.items():
            s = ss[name].detach()
            if torch.is_floating_point(t):
                t.mul_(self.rho).add_(s.to(dtype=t.dtype), alpha=1.0 - self.rho)
            else:
                t.copy_(s)
        self.model.eval()

    def train(self, mode=True):
        super().train(False)
        self.model.eval()
        return self
