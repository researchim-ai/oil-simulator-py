"""Large derivative assembly preserves dense reference, modes and shared owners."""

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from simulator.surface_well_blocks import assemble_surface_blocks


@pytest.mark.parametrize("nb", [2, 3])
def test_chunked_product_matches_full_reference_and_bounds_temporaries(nb):
    nw, nconn = 7, 31
    rng = torch.Generator().manual_seed(6)
    rand = lambda *size: torch.randn(*size, dtype=torch.float64, generator=rng)
    K, dq, Cv, weights = rand(nw, nw, 3), rand(nconn, 3, nb), rand(nconn, 3), rand(nconn, 3)
    owners = torch.arange(nconn) % nw
    rate = torch.tensor([True, False, False, True, False, False, True])
    surface = torch.tensor([False, True, False, False, True, False, False])
    expected = assemble_surface_blocks(K, owners, dq, Cv, weights, rate, surface)
    budget = nw * 3 * nb * 2
    large_outputs = []
    input_storage = {
        x.untyped_storage().data_ptr() for x in (K, dq, Cv, weights, owners, rate, surface)
    }
    output_storage = set()

    class Budget(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            values = out if isinstance(out, (tuple, list)) else (out,)
            for value in values:
                if (
                    torch.is_tensor(value)
                    and value.numel() > budget
                    and value.untyped_storage().data_ptr() not in input_storage
                ):
                    # B is the explicit persistent result. No dense Brate or
                    # full cross-component multiplication is allowed here.
                    assert tuple(value.shape) == (nw, nconn, nb), (func, value.shape)
                    output_storage.add(value.untyped_storage().data_ptr())
                    assert len(output_storage) <= 1
                    large_outputs.append(str(func))
            return out

    with Budget():
        actual = assemble_surface_blocks(
            K, owners, dq, Cv, weights, rate, surface, maximum_product_entries=budget
        )
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)
    assert large_outputs
