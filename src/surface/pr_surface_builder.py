"""Compile pipes, finite gravitational vessels and equipment into one Newton.

Column terminals are geometry, never synthetic mixing volumes. Every native
column aperture occupies the actual first/last finite pipe face. Phase births,
slip, energy and topology recovery belong to the existing implicit PR solver.
"""

from dataclasses import dataclass

from .liquid import _name, _number
from .pr_gravity_transport import PRColumnPort, PRGravityTransportNetwork
from .pr_transport_builder import build_equilibrium_transport, initialize_grouped
from .stratified_pr import StratifiedPRState


@dataclass(frozen=True)
class PRColumnTerminal:
    name: str
    column: str
    height_m: float

    def __post_init__(self):
        _name(self.name)
        _name(self.column)
        _number(self.height_m, "column terminal height", nonnegative=True)


def build_pr_surface_network(
    fluid,
    vessels,
    pipes,
    *,
    columns=None,
    column_terminals=(),
    column_walls=None,
    links=(),
    heat_links=(),
    column_heat_links=(),
    resolved_separators=(),
    resolved_nozzles=(),
    maximum_transverse_head_fraction=1e-3,
    **solver,
):
    """Build arbitrary finite branches/loops with direct resolved apertures.

    Source/target pipe names may reference finite vessels or declared column
    terminals. All equipment accepted by PREquilibriumTransportNetwork remains
    available through solver arguments. Runtime persistence uses the existing
    pr_gravity_transport_v1 schema, not a second private representation.
    """
    columns = dict(columns or {})
    terminals = tuple(column_terminals)
    pipes = tuple(pipes)
    vessels = tuple(vessels)
    if any(not isinstance(p, PRColumnTerminal) for p in terminals):
        raise TypeError("Expected PRColumnTerminal")
    by_name = {p.name: p for p in terminals}
    if len(by_name) != len(terminals):
        raise ValueError("Unique column terminal names required")
    if set(by_name) & set(columns):
        raise ValueError("Column terminals and finite columns need distinct names")
    elevations = {}
    for p in terminals:
        if p.column not in columns:
            raise ValueError("Unknown terminal column")
        column = columns[p.column]
        if p.height_m > column.geometry.height_m:
            raise ValueError("Column terminal above vessel")
        elevations[p.name] = column.elevation_m + p.height_m
    gravity = solver.setdefault(
        "gravity_m_s2", next(iter(columns.values())).gravity_m_s2 if columns else 0.0
    )
    if any(c.gravity_m_s2 != gravity for c in columns.values()):
        raise ValueError("Column and pipe gravity must agree")
    referenced = {name for pipe in pipes for name in (pipe.source, pipe.target) if name in by_name}
    if referenced != set(by_name):
        raise ValueError("Every declared column terminal must connect to a pipe")
    from .pr_resolved_separator import PRResolvedSeparator, join_pr_transport_graphs

    # Equipment may join bodies from different physical blocks. Validate it
    # only after all bodies exist; adding it to the first subgraph would
    # incorrectly reject real cross-block endpoints.
    equipment = {
        key: solver.pop(key, ())
        for key in ("pumps", "pump_connections", "compressors", "restrictions", "drives")
    }
    blocks = []
    if vessels or pipes:
        blocks.append(
            build_equilibrium_transport(
                fluid, vessels, pipes, boundary_elevations_m=elevations, **solver
            )
        )
    topology = {
        "model",
        "fluid",
        "nodes",
        "links",
        "heat_links",
        "pumps",
        "compressors",
        "pump_connections",
        "restrictions",
        "drives",
        "pipe_paths",
    }
    for separator in resolved_separators:
        if not isinstance(separator, PRResolvedSeparator):
            raise TypeError("Expected PRResolvedSeparator")
        policy = (
            {k: v for k, v in blocks[0].to_dict().items() if k not in topology}
            if blocks
            else solver
        )
        blocks.append(separator.build(fluid, **policy))
    if not blocks:
        raise ValueError("At least one finite vessel, pipe or resolved separator required")
    transport = join_pr_transport_graphs(
        blocks, links=links, heat_links=heat_links, nozzles=resolved_nozzles, **equipment
    )
    ports = []
    for pipe in pipes:
        path = transport.pipe_paths[pipe.name]
        for side, terminal, node in (
            ("left", pipe.source, path[0]),
            ("right", pipe.target, path[-1]),
        ):
            if terminal not in by_name:
                continue
            if (
                getattr(pipe, "source_height_m" if side == "left" else "target_height_m")
                is not None
            ):
                raise ValueError("Column height belongs to its terminal, not a kinematic pickup")
            endpoint = by_name[terminal]
            cell = transport.nodes[transport.names.index(node)]
            ports.append(
                PRColumnPort(
                    f"{pipe.name}__column_{side}",
                    endpoint.column,
                    node,
                    endpoint.height_m,
                    cell.face_diameter(side),
                    pipe_port=side,
                )
            )
    return PRGravityTransportNetwork(
        transport,
        columns,
        ports,
        walls=column_walls,
        maximum_transverse_head_fraction=maximum_transverse_head_fraction,
        heat_links=column_heat_links,
    )


def initialize_pr_surface_network(model, transport_initial, column_initial, wall_temperatures=None):
    """Initialize physical inventories once, accepting grouped pipe fields.

    A column may be supplied as a verified profile checkpoint or explicit PTz
    initialization arguments. There are no synthetic terminal stocks to fill.
    """
    if not isinstance(model, PRGravityTransportNetwork):
        raise TypeError("Expected the compiled gravitational PR surface network")
    if set(column_initial) != set(model.columns):
        raise ValueError("Initial profiles must cover every finite column")
    profiles = {}
    for name, spec in column_initial.items():
        if isinstance(spec, StratifiedPRState):
            model.columns[name].inspect(spec)
            profiles[name] = spec
        else:
            profiles[name] = StratifiedPRState.from_dict(
                model.columns[name].initialize(**spec)["state"]
            )
    transport = initialize_grouped(model.transport, transport_initial)
    return model.initialize(transport, profiles, wall_temperatures)
