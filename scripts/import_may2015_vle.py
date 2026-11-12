"""Extract, never fit, methane/ethane measurements from NIST ThermoML."""

import argparse
import hashlib
import json
import math
from pathlib import Path

DOI = "10.1021/acs.jced.5b00610"
URL = "https://trc.nist.gov/ThermoML/" + DOI + ".json"


def extract(path, xml=None):
    path = Path(path)
    raw = json.loads(path.read_bytes())
    if raw["Citation"]["sDOI"] != DOI:
        raise ValueError("Wrong experimental article")
    compounds = {c["RegNum"]["nOrgNum"]: c["sFormulaMolec"] for c in raw["Compound"]}
    if compounds[1] != "CH4" or compounds[2] != "C2H6":
        raise ValueError("Methane/ethane compound identities differ")
    datasets = {d["nPureOrMixtureDataNumber"]: d for d in raw["PureOrMixtureData"]}
    pressure, vapor = datasets[1], datasets[2]
    for data in (pressure, vapor):
        if {c["RegNum"]["nOrgNum"] for c in data["Component"]} != {1, 2}:
            raise ValueError("Wrong binary experimental dataset")
        variables = {v["nVarNumber"]: v for v in data["Variable"]}
        if (
            set(variables) != {1, 2}
            or variables[1]["VariableID"]["VariableType"].get("eTemperature") != "Temperature, K"
            or variables[2]["VariableID"]["VariableType"].get("eComponentComposition")
            != "Mole fraction"
            or variables[2]["VariableID"]["RegNum"]["nOrgNum"] != 1
            or any(v["VarPhaseID"]["eVarPhase"] != "Liquid" for v in variables.values())
        ):
            raise ValueError("Variables must be liquid temperature and methane mole fraction")
        if len(data["Property"]) != 1 or data["Property"][0]["nPropNumber"] != 1:
            raise ValueError("Single property required")
        uncertainty = data["Property"][0]["CombinedUncertainty"]
        if uncertainty["nCombUncertLevOfConfid"] != 95 or uncertainty["nCombUncertAssessNum"] != 1:
            raise ValueError("Explicit archived 95% expanded uncertainty required")
    pp, yp = pressure["Property"][0], vapor["Property"][0]
    if (
        pp["PropPhaseID"]["ePropPhase"] != "Liquid"
        or pp["Property-MethodID"]["PropertyGroup"]["VaporPBoilingTAzeotropTandP"]["ePropName"]
        != "Vapor or sublimation pressure, kPa"
        or yp["PropPhaseID"]["ePropPhase"] != "Gas"
        or yp["Property-MethodID"]["PropertyGroup"]["CompositionAtPhaseEquilibrium"]["ePropName"]
        != "Mole fraction"
        or yp["Property-MethodID"]["RegNum"]["nOrgNum"] != 2
    ):
        raise ValueError("Expected pressure in kPa and vapor ETHANE mole fraction")

    def indexed(data):
        out = {}
        for index, row in enumerate(data["NumValues"]):
            values = {v["nVarNumber"]: v["nVarValue"] for v in row["VariableValue"]}
            key = values[1], values[2]
            if key in out or set(values) != {1, 2} or len(row["PropertyValue"]) != 1:
                raise ValueError("Duplicate or incomplete experimental coordinates")
            prop = row["PropertyValue"][0]
            u = prop["CombinedUncertainty"]
            if prop["nPropNumber"] != 1 or u["nCombUncertAssessNum"] != 1:
                raise ValueError("Mismatched uncertainty/property assessment")
            value, bound = prop["nPropValue"], u["nCombExpandUncertValue"]
            if not all(math.isfinite(v) for v in (*key, value, bound)) or bound <= 0:
                raise ValueError("Nonfinite measurement or nonpositive uncertainty")
            out[key] = index, value, bound
        return out

    p, y = indexed(pressure), indexed(vapor)
    if p.keys() != y.keys() or len(p) != 17:
        raise ValueError("Exactly 17 matched points required; no partial or positional join")
    points = []
    for (t, x), (pi, value, bound) in p.items():
        yi, fraction, u = y[t, x]
        if not (t > 0 and 0 < x < 1 and value > 0 and 0 < fraction < 1):
            raise ValueError("Measurements outside physical domain")
        points.append(
            {
                "pressure_dataset": 1,
                "pressure_index": pi,
                "vapor_dataset": 2,
                "vapor_index": yi,
                "temperature_k": t,
                "liquid_methane_mole_fraction": x,
                "pressure_pa": value * 1000,
                "pressure_expanded_uncertainty_pa": bound * 1000,
                "vapor_ethane_mole_fraction": fraction,
                "vapor_expanded_uncertainty": u,
            }
        )
    if (
        xml is not None
        and hashlib.md5(Path(xml).read_bytes()).hexdigest() != raw["THERMOML_MD5_CHECKSUM"]
    ):
        raise ValueError("Original XML differs from the JSON's ThermoML checksum")
    return {
        "format": "may2015_methane_ethane_measurements_v1",
        "source_url": URL,
        "doi": DOI,
        "archive_json_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "archive_xml_md5": raw["THERMOML_MD5_CHECKSUM"],
        "uncertainty_confidence_percent": 95,
        "uncertainty_kind": "NIST/TRC combined expanded marginal property uncertainty; covariance not supplied",
        "selection": "All 17 points in binary datasets 1 and 2, joined by exact (T,x_liquid_CH4); no selection by model errors",
        "points": points,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--xml", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = extract(args.source, args.xml)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {"points": len(result["points"]), "source_sha256": result["archive_json_sha256"]}
        )
    )
