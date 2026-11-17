"""Validate a numerical BHP guess without changing a physical flow equation."""
import numpy as np

try:
    from ..surface import ComponentStream
    from ..surface.blackoil_pipe import ChokedFlowError, PipePressureCoverageError
    from ..surface.liquid import HydraulicConvergenceError
except ImportError:
    from surface import ComponentStream
    from surface.blackoil_pipe import ChokedFlowError, PipePressureCoverageError
    from surface.liquid import HydraulicConvergenceError


def initial_network_seed(wells, v, d, proposed, fallback):
    """Only an admissible network guess is used; invalid property data propagates.

    Cache the exact source packet and network object for the first assembly.
    A changed flow or replacement network cannot consume this cached result.
    """
    if wells.surface_pressure_control is not None:
        return fallback, None
    q, *_ = wells._production_connection_flows(v, d, proposed)
    rates = q.new_zeros((wells.n_w, 3)).index_add_(0, wells.conn_well, q).detach().cpu().numpy()
    if not np.isfinite(rates).all() or (rates < 0).any() or (rates.sum(1) <= 0).any():
        return fallback, None
    sources = {}
    for node, rate in zip(wells.surface_nodes, rates):
        stream = ComponentStream.from_mass_rates(*rate, wells.surface_network.model.densities)
        sources[node] = ComponentStream.mix([sources.get(node, ComponentStream()), stream])
    try:
        mode = "operator" if wells.sensitivity_backend == "tree_operator" else "dense"
        result = wells.surface_network.solve(sources, sensitivity_mode=mode)
    except (ChokedFlowError, PipePressureCoverageError, HydraulicConvergenceError):
        return fallback, None
    return proposed, (wells.surface_network, rates.copy(), result)
