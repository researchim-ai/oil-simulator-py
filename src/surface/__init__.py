"""Physical surface networks, separate from the reservoir's tabulated VFP network."""

from .control import PressureControlResult, PumpPressureControl
from .curves import EquipmentCurve
from .dispatch import FacilityCapacityControl
from .facility import FacilityLimit, SurfaceFacility, evaluate_limits
from .gas import (
    GasCompressor,
    GasCooler,
    GasExportSystem,
    GasLetdown,
    GasState,
    GasTrain,
    IdealGas,
    IsothermalGasPipe,
    intercooled_train,
    mix_gas_streams,
)
from .liquid import (
    ConservationError,
    EquipmentOperatingError,
    HydraulicConvergenceError,
    InvalidPressureError,
    Liquid,
    LiquidNetwork,
    NetworkResult,
    Node,
    Pipe,
    PropertyModelError,
    Pump,
    ThrottleValve,
)
from .operation import FacilityPeriod, facility_fingerprint, run_facility_schedule
from .process import (
    BlackOilSeparator,
    ComponentStream,
    StockDensities,
    component_balance,
    run_separator_train,
)
from .station import PumpStationControl, PumpStationOperatingError, PumpStationResult

__all__ = [
    "BlackOilSeparator",
    "ComponentStream",
    "ConservationError",
    "EquipmentCurve",
    "EquipmentOperatingError",
    "FacilityCapacityControl",
    "FacilityLimit",
    "FacilityPeriod",
    "GasCompressor",
    "GasCooler",
    "GasExportSystem",
    "GasLetdown",
    "GasState",
    "GasTrain",
    "HydraulicConvergenceError",
    "IdealGas",
    "InvalidPressureError",
    "IsothermalGasPipe",
    "Liquid",
    "LiquidNetwork",
    "NetworkResult",
    "Node",
    "Pipe",
    "PressureControlResult",
    "PropertyModelError",
    "Pump",
    "PumpPressureControl",
    "PumpStationControl",
    "PumpStationOperatingError",
    "PumpStationResult",
    "StockDensities",
    "SurfaceFacility",
    "ThrottleValve",
    "component_balance",
    "evaluate_limits",
    "facility_fingerprint",
    "intercooled_train",
    "mix_gas_streams",
    "run_facility_schedule",
    "run_separator_train",
]

# These CPU PVT tools use torch autograd. Load them on demand so importing the
# standalone liquid/gas equipment does not acquire the extra startup cost.
_LAZY_MODELS = {
    "BlackOilMixture": "blackoil_properties",
    "ThermalLetdown": "thermal_equipment",
    "EquilibriumNozzle": "thermal_equipment",
    "PengRobinsonFluid": "peng_robinson",
    "PengRobinsonMixture": "peng_robinson_mixture",
    "MixtureLetdown": "mixture_equipment",
    "GasExportValve": "gas_valve",
    "ideal_gas_nozzle_flow": "gas_valve",
    "BlackOilPipe": "blackoil_pipe",
    "BlackOilLiquidValve": "blackoil_valve",
    "ProductionValve": "blackoil_network",
    "BlackOilPressureControl": "blackoil_control",
    "ChokedFlowError": "blackoil_pipe",
    "BlackOilProductionNetwork": "blackoil_network",
    "ProductionNode": "blackoil_network",
    "ProductionPipe": "blackoil_network",
    "GatheringFacility": "gathering",
    "HomogeneousFluid": "multiphase",
    "HomogeneousPipe": "multiphase",
}
__all__ += list(_LAZY_MODELS)


def __getattr__(name):
    if name not in _LAZY_MODELS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(f".{_LAZY_MODELS[name]}", __name__), name)
    globals()[name] = value
    return value
