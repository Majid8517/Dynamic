import torch
import torch.nn as nn
from paper_impl.ema import EMATeacher

def test_teacher_no_grad_and_update():
    student=nn.Sequential(nn.Linear(4,4),nn.ReLU(),nn.Linear(4,2))
    teacher=EMATeacher(student,rho=0.5)
    assert all(not p.requires_grad for p in teacher.model.parameters())
    before=[p.detach().clone() for p in teacher.model.parameters()]
    with torch.no_grad():
        for p in student.parameters(): p.add_(1.0)
    teacher.update(student)
    for b,a,s in zip(before,teacher.model.parameters(),student.parameters()):
        assert torch.allclose(a,0.5*b+0.5*s.detach())
