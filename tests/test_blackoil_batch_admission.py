"""Device-batched admission preserves scalar PVTO domain/continuity checks."""

import pytest
import torch
from test_surface_blackoil_properties import DENSITIES, model, pvt_data

from surface.blackoil_batch_admission import validate_inputs
from surface.blackoil_properties import BlackOilMixture
from surface.liquid import PropertyModelError


@pytest.mark.parametrize(
    "rates",
    [
        [[0.85, 2, 0.06], [0.85, 0.5, 0.144], [0, 0.1, 0], [0, 0, 0.03]],
        [[0.85, 2, 0.024], [0.1, 0.2, 0.0001]],
    ],
)
def test_valid_mixed_regimes_match_scalar_admission_exactly(rates):
    m = model()
    p = [2.4e6] * len(rates)
    a, b = validate_inputs(m, p, rates, len(rates))
    for i in range(len(rates)):
        torch.testing.assert_close(b[i], m.validate(p[i], rates[i]), rtol=0, atol=0)
    torch.testing.assert_close(a, torch.tensor(p, dtype=torch.float64))


@pytest.mark.parametrize(
    "mutation", ["pressure", "nan", "negative", "zero", "coverage", "Bo", "mu_o"]
)
def test_invalid_one_row_cannot_be_hidden_by_valid_rows(mutation):
    data = pvt_data()
    p = [2.4e6] * 3
    rates = [[0.85, 2.0, 0.06] for _ in p]
    if mutation == "pressure":
        p[1] = 1e9
    elif mutation == "nan":
        rates[1][0] = float("nan")
    elif mutation == "negative":
        rates[1][1] = -0.1
    elif mutation == "zero":
        rates[1] = [0, 0, 0]
    elif mutation == "coverage":
        # The valid model can have a narrower Rs range; demand a ratio below it.
        del data["extended"]["oil"]["tables"]["0.0"]
        data["extended"]["oil"]["rs_values"] = [20.0, 80.0]
        rates[1] = [0.85, 2.0, 0.001]
    else:
        col = "Bo" if mutation == "Bo" else "mu_o_cP"
        for table in data["extended"]["oil"]["tables"].values():
            table[col] = [v * 1.01 for v in table[col]]
    m = BlackOilMixture(data, DENSITIES)
    with pytest.raises((ValueError, PropertyModelError)):
        m.validate(p[1], rates[1])
    with pytest.raises((ValueError, PropertyModelError)):
        validate_inputs(m, p, rates, 3)


def test_no_per_stream_validate_calls_and_single_host_status_packet(monkeypatch):
    m = model()

    def forbidden(*args, **kwargs):
        raise AssertionError("Scalar admission is forbidden on the batched path")

    monkeypatch.setattr(m, "validate", forbidden)
    for name in ("__bool__", "__float__", "__int__"):
        monkeypatch.setattr(torch.Tensor, name, forbidden)
    original = torch.Tensor.tolist
    reads = []

    def read(value, *args, **kwargs):
        reads.append(tuple(value.shape))
        return original(value, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "tolist", read)
    p, r = validate_inputs(m, [2.4e6] * 512, [[0.85, 2, 0.06]] * 512, 512)
    assert r.shape == (512, 3) and p.shape == (512,)
    assert reads == [(5,)]


def test_zero_gor_pvto_is_still_required_for_pure_water_sensitivities():
    data = pvt_data()
    del data["extended"]
    m = BlackOilMixture(data, DENSITIES)
    with pytest.raises(PropertyModelError, match="Pure-water"):
        validate_inputs(m, [2e6], [[0, 1, 0]], 1)
    _, masses = validate_inputs(m, [2e6], [[0, 0, 0.03]], 1)
    assert float(masses[0, 2]) == 0.03


@pytest.mark.parametrize("pressure", [True, complex(2e6, 1), "2000000"])
def test_nonreal_or_boolean_host_pressures_are_not_coerced(pressure):
    with pytest.raises(ValueError):
        validate_inputs(model(), [pressure], [[0.85, 2, 0.06]], 1)


def test_flat_rs_and_no_oil_do_not_evaluate_a_singular_bubble_point():
    data = pvt_data()
    data["Rs_m3m3"] = [0.0, 0.0, 0.0]
    m = BlackOilMixture(data, DENSITIES)
    p, r = validate_inputs(m, [2e6, 2e6], [[0, 1, 0], [0, 0, 0.03]], 2)
    assert torch.isfinite(p).all() and torch.isfinite(r).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_device_local_tensor_inputs_do_not_require_host_values():
    m = model().to("cuda")
    p = torch.full((128,), 2.4e6, dtype=torch.float64, device="cuda")
    rates = torch.tensor([[0.85, 2, 0.06]], dtype=torch.float64, device="cuda").expand(128, -1)
    a, b = validate_inputs(m, p, rates, 128)
    assert a.device == p.device and b.device == rates.device
    torch.testing.assert_close(a, p)
    torch.testing.assert_close(b, rates)
