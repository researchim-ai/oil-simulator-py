"""Bounded, two-continuous-layer gas/liquid force balance in a circular pipe.

See docs/PR_STRATIFIED_DRIFT.md for equations, constitutive choices, and limits.
This module does not select regimes or integrate itself into the PR graph.
"""

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True, kw_only=True)
class StratifiedGasLiquidClosure:
    """Explicit stratified branch with declared gas/liquid dynamic viscosities.

    Public velocities uses +g*dz/L (uphill). Diagnostic parameters instead
    uses actual signed gravitational ACCELERATION along x, -g*dz/L. No phase identity is inferred
    beyond density sorting: the caller must establish that these are gas/liquid.
    """

    interfacial_friction_factor: float  # Fanning convention, tau=f*rho*s*abs(s)/2
    wall_roughness_m: float
    gas_dynamic_viscosity_pa_s: float
    liquid_dynamic_viscosity_pa_s: float
    minimum_gas_volume_fraction: float = 0.02
    maximum_gas_volume_fraction: float = 0.98
    maximum_gas_liquid_density_ratio: float = 0.2
    maximum_inclination_deg: float = 5.0
    maximum_slip_m_s: float = 100.0
    maximum_phase_speed_m_s: float = 100.0
    model: str = "stratified_gas_liquid"

    def __post_init__(self):
        for name in (
            "interfacial_friction_factor",
            "gas_dynamic_viscosity_pa_s",
            "liquid_dynamic_viscosity_pa_s",
            "maximum_slip_m_s",
            "maximum_phase_speed_m_s",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Positive finite {name} required")
        if not math.isfinite(self.wall_roughness_m) or self.wall_roughness_m < 0:
            raise ValueError("Nonnegative finite wall roughness required")
        if (
            not 1e-4
            <= self.minimum_gas_volume_fraction
            < self.maximum_gas_volume_fraction
            <= 1 - 1e-4
        ):
            raise ValueError("Strict interior gas fraction envelope required")
        if not 0 < self.maximum_gas_liquid_density_ratio < 1:
            raise ValueError("Gas/liquid density contrast required")
        if not 0 < self.maximum_inclination_deg <= 10:
            raise ValueError("Near-horizontal inclination envelope must be <= 10 degrees")
        if self.model != "stratified_gas_liquid":
            raise ValueError("Unknown stratified model")

    @staticmethod
    def _segment(alpha_l, diameter):
        # alpha_l=(theta-sin(theta)*cos(theta))/pi. Detached bisection then
        # two live Newton corrections recover first and second implicit AD.
        with torch.no_grad():
            a = alpha_l.detach()
            lo, hi = a * 0, a * 0 + math.pi
            for _ in range(60):
                theta = (lo + hi) / 2
                lower = (theta - theta.sin() * theta.cos()) / math.pi < a
                lo, hi = torch.where(lower, theta, lo), torch.where(lower, hi, theta)
            theta = ((lo + hi) / 2).detach()
        for _ in range(2):
            theta = theta - ((theta - theta.sin() * theta.cos()) / math.pi - alpha_l) / (
                2 * theta.sin().square() / math.pi
            )
        area = math.pi * diameter.square() / 4
        al, ag = area * alpha_l, area * (1 - alpha_l)
        sl, sg, si = diameter * theta, diameter * (math.pi - theta), diameter * theta.sin()
        # Liquid free surface excluded; gas sees both wall and moving interface.
        dl, dg = 4 * al / sl, 4 * ag / (sg + si)
        return al, ag, sl, sg, si, dl, dg, diameter * (1 - theta.cos()) / 2

    def _wall(self, v, rho, mu, dh):
        """Signed shear and exact d(tau)/dv, including the laminar stop limit.

        Max of laminar shear and Haaland turbulent shear is a declared
        engineering envelope, not Taitel--Dukler's original Blasius fit.
        """
        speed = v.abs()
        re = rho * speed * dh / mu
        ret = re.clamp_min(2300)
        b = (self.wall_roughness_m / (3.7 * dh)).pow(1.11) + 6.9 / ret
        q = -1.8 * torch.log10(b)
        fd = q.reciprocal().square()
        laminar = 8 * mu / dh
        turbulent = rho * fd * speed / 8
        tau = v * torch.maximum(laminar, turbulent)
        dq_dre = 1.8 * 6.9 / (math.log(10) * b * ret.square())
        df_dre = -2 * dq_dre / q.pow(3)
        df_dre = torch.where(re > 2300, df_dre, 0.0)
        slope_t = rho * (2 * fd * speed + speed.square() * df_dre * rho * dh / mu) / 8
        return tau, torch.where(laminar >= turbulent, laminar, slope_t)

    def _balance(self, slip, args):
        u, rg, rl, mg, ml, mug, mul, gx, al, ag, sl, sg, si, dl, dg = args
        vg, vl = u + ml * slip, u - mg * slip
        tg, dtg = self._wall(vg, rg, mug, dg)
        tl, dtl = self._wall(vl, rl, mul, dl)
        # Viscous gas-side coupling makes slope strictly positive even at stop.
        ci = 8 * mug / dg
        ti = ci * slip + 0.5 * self.interfacial_friction_factor * rg * slip * slip.abs()
        dti = ci + self.interfacial_friction_factor * rg * slip.abs()
        pg = rg * gx - (tg * sg + ti * si) / ag
        pl = rl * gx - (tl * sl - ti * si) / al
        residual = pl - pg
        slope = dtl * mg * sl / al
        slope = slope + dtg * ml * sg / ag + dti * si * (1 / al + 1 / ag)
        return residual, slope, vg, vl, pg, pl, tg, tl, ti

    def parameters(
        self,
        phase_volumes,
        phase_densities,
        bulk_velocity,
        *,
        diameter_m,
        axial_gravity_m_s2,
        gravity_m_s2,
        phase_viscosities_pa_s=None,
        **context,
    ):
        """Torch scalars in SI; diagnostics and rejection predicates stay on device.

        Ordinary rejected states produce diagnostic candidates, never an
        accepted fallback; extreme overflow is not regularized. Geometry scalars may be live tensors. No .item()/CPU.
        """
        if phase_volumes.shape != (2,) or phase_densities.shape != (2,):
            raise ValueError("Exactly two continuous gas/liquid phases required")
        ref = phase_volumes
        cast = lambda x: torch.as_tensor(x, dtype=ref.dtype, device=ref.device)
        rho = phase_densities
        if phase_viscosities_pa_s is None:
            gas_role = (torch.arange(2, device=ref.device) == rho.argmin()).to(ref.dtype)
            mu = (
                gas_role * self.gas_dynamic_viscosity_pa_s
                + (1 - gas_role) * self.liquid_dynamic_viscosity_pa_s
            )
        else:
            mu = cast(phase_viscosities_pa_s)
        if mu.shape != (2,):
            raise ValueError("Two dynamic viscosities in phase order required")
        u, d, gx, g = map(cast, (bulk_velocity, diameter_m, axial_gravity_m_s2, gravity_m_s2))
        finite = (
            torch.isfinite(ref).all()
            & torch.isfinite(rho).all()
            & torch.isfinite(mu).all()
            & torch.isfinite(u)
            & torch.isfinite(d)
            & torch.isfinite(gx)
            & torch.isfinite(g)
        )
        positive = (ref > 0).all() & (rho > 0).all() & (mu > 0).all() & (d > 0) & (g > 0)
        # Safe internal arguments for rejected line-search candidates.
        volumes = torch.where(torch.isfinite(ref) & (ref > 0), ref, torch.ones_like(ref))
        volumes = volumes / volumes.max()
        densities = torch.where(torch.isfinite(rho) & (rho > 0), rho, torch.ones_like(rho))
        viscosity = torch.where(torch.isfinite(mu) & (mu > 0), mu, torch.full_like(mu, 1e-3))
        u = torch.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
        d = torch.where(torch.isfinite(d) & (d > 0), d, torch.ones_like(d))
        gx = torch.nan_to_num(gx, nan=0.0, posinf=0.0, neginf=0.0)
        g = torch.where(torch.isfinite(g) & (g > 0), g, torch.ones_like(g))
        gas = (torch.arange(2, device=ref.device) == densities.argmin()).to(ref.dtype)
        alpha = volumes / volumes.sum()
        ag_raw = (alpha * gas).sum()
        ag_fraction = ag_raw.clamp(1e-4, 1 - 1e-4)
        rg, rl = (densities * gas).sum(), (densities * (1 - gas)).sum()
        mug, mul = (viscosity * gas).sum(), (viscosity * (1 - gas)).sum()
        mg = ag_fraction * rg / (ag_fraction * rg + (1 - ag_fraction) * rl)
        ml = 1 - mg
        geom = self._segment(1 - ag_fraction, d)
        al, ag, sl, sg, si, dl, dg, height = geom
        args = (u, rg, rl, mg, ml, mug, mul, gx, *geom[:7])
        with torch.no_grad():
            aa = tuple(x.detach() for x in args)
            lo, hi = u.detach() * 0 - self.maximum_slip_m_s, u.detach() * 0 + self.maximum_slip_m_s
            bracketed = (self._balance(lo, aa)[0] <= 0) & (self._balance(hi, aa)[0] >= 0)
            for _ in range(64):
                s = (lo + hi) / 2
                lower = self._balance(s, aa)[0] <= 0
                lo, hi = torch.where(lower, s, lo), torch.where(lower, hi, s)
            s = ((lo + hi) / 2).detach()
        # Detached tangent reattachment is deliberately NOT used: two live
        # Newton corrections give second derivatives as well as first order AD.
        for _ in range(2):
            residual, slope, *_ = self._balance(s, args)
            s = (s - residual / slope).clamp(-self.maximum_slip_m_s, self.maximum_slip_m_s)
        residual, slope, vg, vl, pg, pl, tg, tl, ti = self._balance(s, args)
        velocities = gas * vg + (1 - gas) * vl
        normal_g = (g.square() - gx.square()).clamp_min(0).sqrt()
        # Rigid-lid, inviscid, long-wave two-layer screen; NOT a finite-wave map.
        kh_critical = (rl - rg) * normal_g * (ag / rg + al / rl) / si
        kh_margin = kh_critical - s.square()
        inclination_valid = gx.abs() <= g * math.sin(math.radians(self.maximum_inclination_deg))
        state_valid = (
            finite
            & positive
            & (ag_raw >= self.minimum_gas_volume_fraction)
            & (ag_raw <= self.maximum_gas_volume_fraction)
            & (rg / rl <= self.maximum_gas_liquid_density_ratio)
            & (self.wall_roughness_m / torch.minimum(dl, dg) <= 0.05)
        )
        tol = 128 * torch.finfo(ref.dtype).eps * (1 + pg.abs() + pl.abs())
        root_valid = (
            bracketed
            & (residual.abs() <= tol)
            & (slope > 0)
            & torch.isfinite(velocities).all()
            & (velocities.abs().max() <= self.maximum_phase_speed_m_s)
        )
        return {
            "velocities": velocities,
            "valid": state_valid & inclination_valid & root_valid & (kh_margin > 0),
            "state_valid": state_valid,
            "inclination_valid": inclination_valid,
            "root_valid": root_valid,
            "bracketed": bracketed,
            "long_wave_stable": kh_margin > 0,
            "kh_margin_m2_s2": kh_margin,
            "slip_m_s": s,
            "force_residual_pa_m": residual,
            "force_slope_pa_s_m2": slope,
            "pressure_gradient_pa_m": (pg + pl) / 2,
            "gas_pressure_gradient_pa_m": pg,
            "liquid_pressure_gradient_pa_m": pl,
            "gas_wall_shear_pa": tg,
            "liquid_wall_shear_pa": tl,
            "interfacial_shear_pa": ti,
            "liquid_height_m": height,
            "gas_area_m2": ag,
            "liquid_area_m2": al,
            "gas_wall_perimeter_m": sg,
            "liquid_wall_perimeter_m": sl,
            "interface_perimeter_m": si,
            "volume_flux_m_s": ag_raw * vg + (1 - ag_raw) * vl,
        }

    def velocities(self, phase_volumes, phase_densities, bulk_velocity, **context):
        """Core-compatible API: axial_gravity_m_s2 is positive uphill (+g*dz/L)."""
        if phase_volumes.shape == (1,) and phase_densities.shape == (1,):
            return bulk_velocity.expand_as(phase_volumes), bulk_velocity.new_ones(
                (), dtype=torch.bool
            )
        # Public graph convention: positive uphill; parameters uses actual gx.
        context = {**context, "axial_gravity_m_s2": -context["axial_gravity_m_s2"]}
        result = self.parameters(phase_volumes, phase_densities, bulk_velocity, **context)
        return result["velocities"], result["valid"]
