"""Benchmark-only switches for exact residual preparation before construction."""

from contextlib import contextmanager


@contextmanager
def operator_preparation(enabled):
    from surface.pr_equilibrium_transport import PREquilibriumTransportNetwork
    from surface.pr_gravity_transport import PRGravityTransportNetwork
    from surface.stratified_pr import StratifiedPRColumn
    from surface.stratified_pr_flow import StratifiedPRFlow
    from surface.stratified_pr_network import StratifiedPRNetwork

    if type(enabled) is not bool:
        raise TypeError("Preparation selection must be boolean")
    settings = (
        (StratifiedPRColumn, "_prepare_geometry"),
        (StratifiedPRFlow, "_prune_unused"),
        (StratifiedPRNetwork, "_prepare_residual"),
        (StratifiedPRNetwork, "_batch_network_eos"),
        (PRGravityTransportNetwork, "_prepare_residual"),
        (PREquilibriumTransportNetwork, "_prepare_endpoint"),
    )
    previous = [(cls, name, getattr(cls, name)) for cls, name in settings]
    try:
        for cls, name in settings:
            setattr(cls, name, enabled)
        yield
    finally:
        for cls, name, value in previous:
            setattr(cls, name, value)
