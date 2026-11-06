"""Explicit physical plant/checkpoint dispatch for shared surface controls."""

from dataclasses import replace

from .pr_gravity_transport import PRGravityTransportNetwork, PRGravityTransportState


def physical_plant_from_dict(data, *, device="cpu"):
    if data.get("model") in ("pr_profile_graph_v1", "pr_profile_graph_v2"):
        from .pr_profile_graph import PRProfileGraphNetwork

        return PRProfileGraphNetwork.from_dict(data, device=device)
    return PRGravityTransportNetwork.from_dict(data, device=device)


def physical_state_from_dict(data):
    if "profiles" in data:
        from .pr_profile_graph import ProfileGraphState

        return ProfileGraphState.from_dict(data)
    return PRGravityTransportState.from_dict(data)


def align_physical_clock(state, time_s):
    """Apply the accepted report endpoint rounding to every explicit clock."""
    state = replace(state, transport=replace(state.transport, time_s=time_s))
    if hasattr(state, "profiles"):
        state = replace(
            state, profiles=tuple((n, replace(s, time_s=time_s)) for n, s in state.profiles)
        )
    return state
