"""Capture preparation must preserve literal dtype, derivatives and ownership."""

import pytest
import torch
from check_pr_cuda_runtime_graph import native_literal_factories


@pytest.mark.parametrize("literal", [True, 2, 1.25, [1.0], [2.0, 2.0]])
@pytest.mark.parametrize("explicit_dtype", [False, True])
def test_literal_factory_equivalence_and_independent_storage(literal, explicit_dtype):
    class Formula(torch.nn.Module):
        def forward(self, x):
            kw = {"dtype": torch.float32} if explicit_dtype else {}
            a = x.new_tensor(literal, **kw)
            return x * a.sum(), a

    module = torch.fx.symbolic_trace(Formula())
    x = torch.tensor([0.3, 0.7], dtype=torch.float64)
    reference, literal_reference = module(x)
    tangent = torch.func.jacrev(lambda z: module(z)[0])(x)
    assert native_literal_factories(module) == 1
    result, first = module(x)
    torch.testing.assert_close(result, reference, rtol=0, atol=0)
    torch.testing.assert_close(first, literal_reference, rtol=0, atol=0)
    torch.testing.assert_close(torch.func.jacrev(lambda z: module(z)[0])(x), tangent)
    first.fill_(0)
    _, second = module(x)
    torch.testing.assert_close(second, literal_reference, rtol=0, atol=0)
    assert first.data_ptr() != second.data_ptr()


@pytest.mark.parametrize("literal", [[1.0, 2.0], [True, 1], [[1.0]], [], [-0.0, 0.0]])
def test_unqualified_factories_remain_untouched(literal):
    class Formula(torch.nn.Module):
        def forward(self, x):
            return x.new_tensor(literal)

    module = torch.fx.symbolic_trace(Formula())
    assert native_literal_factories(module) == 0
