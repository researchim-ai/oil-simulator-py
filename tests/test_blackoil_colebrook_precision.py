"""Colebrook closure and derivative precision after reducing redundant Newton work."""
import math

import pytest
import torch
from surface.blackoil_pipe import _darcy


def _ten_iteration_reference(reynolds, roughness):
    r=reynolds.clamp_min(1e-15);rt=r.clamp_min(2000.);u=torch.full_like(r,7.)
    for _ in range(10):
        arg=roughness/3.7+2.51*u/rt
        u=u-(u+2*torch.log10(arg))/(1+2/math.log(10)*(2.51/rt)/arg)
    b=((r-2000)/2000).clamp(0,1);b=b.square()*(3-2*b)
    return (1-b)*64/r+b*u.reciprocal().square()


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_values_and_first_second_log_sensitivities_match_reference(device):
    if device=='cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    re=torch.cat([torch.logspace(-10,100,170,dtype=torch.float64),
                  torch.tensor([1999.,2000.,2001.,3999.,4000.,4001.],dtype=torch.float64)]).to(device)
    rough=torch.tensor([0.,1e-12,1e-8,1e-4,.01,.2,1.],dtype=torch.float64,device=device)
    re,rough=torch.meshgrid(re,rough,indexing='ij')
    re,rough=re.clone().requires_grad_(),rough.clone().requires_grad_()
    results=[]
    for fn in (_ten_iteration_reference,_darcy):
        value=fn(re,rough)
        grad=torch.autograd.grad(value.sum(),(re,rough),create_graph=True)
        second=torch.autograd.grad((grad[0]*re+grad[1]*rough).sum(),(re,rough))
        results.append([value,grad[0]*re,grad[1]*rough,second[0]*re,second[1]*rough])
    for a,b in zip(*results):
        assert bool(torch.isfinite(b).all())
        torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-16)


def test_turbulent_solution_satisfies_independent_colebrook_equation_and_implicit_derivatives():
    re=torch.logspace(math.log10(4000),14,120,dtype=torch.float64)[:,None].requires_grad_()
    rough=torch.tensor([0.,1e-8,1e-4,.01,.2],dtype=torch.float64)[None,:].requires_grad_()
    value=_darcy(re,rough);u=value.rsqrt();arg=rough/3.7+2.51*u/re
    torch.testing.assert_close(u+2*torch.log10(arg),torch.zeros_like(value),rtol=0,atol=3e-14)
    fu=1+2/math.log(10)*(2.51/re)/arg
    du_re=(2/math.log(10)*2.51*u/re.square()/arg)/fu
    du_rough=-(2/math.log(10)/3.7/arg)/fu
    dr,de=torch.autograd.grad(value.sum(),(re,rough))
    torch.testing.assert_close(dr,(-2*u.pow(-3)*du_re).sum(1,keepdim=True),rtol=1e-12,atol=1e-18)
    torch.testing.assert_close(de,(-2*u.pow(-3)*du_rough).sum(0,keepdim=True),rtol=1e-12,atol=1e-15)
