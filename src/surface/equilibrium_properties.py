"""Component-based caloric Peng--Robinson equilibrium on Torch CPU/CUDA.

PT Gibbs minimization with numerical multistart tangent-plane stability and
up to three phases. This is an EOS model, not a claim of accurate water/brine
or a global stability proof. Kij and ideal heat capacities must be supplied.
"""

import math
from dataclasses import asdict, dataclass

import torch

from .equilibrium_cache import FlashCache, cached_equilibrium
from .equilibrium_stability_batch import stability_candidates
from .liquid import HydraulicConvergenceError, PropertyModelError, _name, _number
from .operation import _digest
from .peng_robinson import PengRobinsonFluid, R
from .peng_robinson_mixture import PengRobinsonMixture


def _gibbs_logit_gradient(u, weighted_potential, probabilities, dim):
    """Exact Gibbs--Duhem softmax/tanh gradient; EOS need not build an AD tape."""
    return (weighted_potential - probabilities * weighted_potential.sum(dim, keepdim=True)) * (
        1 - torch.tanh(u / 80).square()
    )


@dataclass(frozen=True)
class EquilibriumComponent(PengRobinsonFluid):
    """PR caloric data with an explicit fluid-only temperature domain.

    The legacy pure-fluid saturation helper imposes T>=0.5 Tc; that is
    an algorithmic bracket restriction, not a restriction of the PR EOS.
    The new equilibrium solver supports declared lower liquid temperatures
    (e.g. ambient water). Solids, association and electrolytes are absent.
    """

    def __post_init__(self):
        _name(self.name)
        for key in (
            "critical_temperature_k",
            "critical_pressure_pa",
            "molar_mass_kg_mol",
            "ideal_cp_j_mol_k",
            "min_temperature_k",
            "max_temperature_k",
            "reference_temperature_k",
            "reference_pressure_pa",
        ):
            _number(getattr(self, key), key, positive=True)
        _number(self.acentric_factor, "acentric_factor")
        if self.ideal_cp_j_mol_k <= R:
            raise PropertyModelError("Ideal Cp must exceed R")
        if not self.min_temperature_k < self.max_temperature_k:
            raise PropertyModelError("Invalid declared caloric temperature interval")
        if not self.min_temperature_k <= self.reference_temperature_k <= self.max_temperature_k:
            raise PropertyModelError("Reference temperature outside declared interval")


class EquilibriumFluid:
    """Immutable property snapshot, physically identical across devices.

    Absent components remain zero and retain their positions in the registry.
    No compositional inference from oil/water/gas labels is performed.
    """

    def __init__(
        self,
        components,
        binary_interactions,
        *,
        device="cpu",
        min_pressure_pa=100.0,
        max_pressure_pa=1e9,
        max_phases=3,
    ):
        reference = PengRobinsonMixture(tuple(components), tuple(map(tuple, binary_interactions)))
        self.components = reference.components
        self.binary_interactions = reference.binary_interactions
        self.device = torch.device(device)
        self.min_pressure_pa = _number(min_pressure_pa, "min_pressure_pa", positive=True)
        self.max_pressure_pa = _number(max_pressure_pa, "max_pressure_pa", positive=True)
        if self.max_pressure_pa <= self.min_pressure_pa:
            raise ValueError("Invalid equilibrium pressure domain")
        if isinstance(max_phases, bool) or max_phases not in (1, 2, 3):
            raise ValueError("max_phases must be 1, 2 or 3")
        self.max_phases = max_phases
        self.min_temperature_k = max(c.min_temperature_k for c in self.components)
        self.max_temperature_k = min(c.max_temperature_k for c in self.components)
        if self.min_temperature_k >= self.max_temperature_k:
            raise PropertyModelError("No common caloric temperature interval")
        self.names = tuple(c.name for c in self.components)
        self.molar_masses = self.tensor([c.molar_mass_kg_mol for c in self.components])
        self.tc = self.tensor([c.critical_temperature_k for c in self.components])
        self.pc = self.tensor([c.critical_pressure_pa for c in self.components])
        self.cp = self.tensor([c.ideal_cp_j_mol_k for c in self.components])
        self.kappa = self.tensor(
            [
                0.37464 + 1.54226 * c.acentric_factor - 0.26992 * c.acentric_factor**2
                for c in self.components
            ]
        )
        self.b = self.tensor([c.b for c in self.components])
        self.a0 = self.tensor([c.attraction_coefficient for c in self.components])
        self.kij = self.tensor(self.binary_interactions)
        self.tref = self.components[0].reference_temperature_k
        self.pref = self.components[0].reference_pressure_pa
        self.model_id = _digest({"model": "caloric_pr_multiphase_v1", **self.to_dict()})

    def tensor(self, value):
        return torch.as_tensor(value, dtype=torch.float64, device=self.device)

    def to_dict(self):
        return {
            "components": [asdict(c) for c in self.components],
            "binary_interactions": [list(r) for r in self.binary_interactions],
            "min_pressure_pa": self.min_pressure_pa,
            "max_pressure_pa": self.max_pressure_pa,
            "max_phases": self.max_phases,
        }

    @classmethod
    def from_dict(cls, data, *, device="cpu"):
        data = dict(data)
        data["components"] = tuple(EquilibriumComponent(**c) for c in data["components"])
        return cls(**data, device=device)

    def to(self, device):
        return self.from_dict(self.to_dict(), device=device)

    def _validate(self, p, t, z):
        if p.ndim or t.ndim or z.shape != (len(self.components),):
            raise PropertyModelError(
                "Expected scalar pressure/temperature and registry composition"
            )
        good = (
            torch.isfinite(p)
            & torch.isfinite(t)
            & torch.isfinite(z).all()
            & (p >= self.min_pressure_pa)
            & (p <= self.max_pressure_pa)
            & (t >= self.min_temperature_k)
            & (t <= self.max_temperature_k)
            & (z >= 0).all()
            & ((z.sum() - 1).abs() <= 1e-10)
        )
        alpha = 1 + self.kappa * (1 - torch.sqrt(t / self.tc))
        if not bool((good & (alpha > 0).all()).detach().cpu()):
            raise PropertyModelError(
                "Equilibrium pressure, temperature or composition outside domain"
            )

    def _roots(self, A, B, derivative_order=1):
        """Both mechanically stable cubic roots, no host/numpy root solve."""
        with torch.no_grad():
            # no_grad disables reverse AD only. Forward AD (jacfwd/JVP)
            # must not differentiate acos/cuberoot at a degenerate auxiliary
            # root. Physical dZ is supplied by the implicit correction below.
            aa, bb = A.detach(), B.detach()
            a, b, c = bb - 1, aa - 3 * bb * bb - 2 * bb, -aa * bb + bb * bb + bb**3
            q = (a * a - 3 * b) / 9
            r = (2 * a**3 - 9 * a * b + 27 * c) / 54
            disc = r * r - q**3
            sqrtq = q.clamp_min(1e-300).sqrt()
            angle = torch.acos((r / sqrtq**3).clamp(-1, 1))
            angles = (angle[..., None] + self.tensor([0, 2 * math.pi, 4 * math.pi])) / 3
            three = -2 * sqrtq[..., None] * torch.cos(angles) - a[..., None] / 3
            d = disc.clamp_min(0).sqrt()
            cb = lambda v: v.sign() * v.abs().pow(1 / 3)
            single = cb(-r + d) + cb(-r - d) - a / 3
            roots = (
                torch.where(
                    ((disc < 0) & (q > 0))[..., None], three, single[..., None].expand_as(three)
                )
                .sort(-1)
                .values
            )
        # Each differentiable Newton correction doubles the accurate derivative
        # order around the detached exact root. Two corrections are needed when
        # differentiating a sonic thermodynamic tangent (second derivatives).
        for _ in range(derivative_order):
            f = (
                roots**3
                + (B[..., None] - 1) * roots**2
                + (A - 3 * B * B - 2 * B)[..., None] * roots
                + (-A * B + B * B + B**3)[..., None]
            )
            fz = 3 * roots**2 + 2 * (B[..., None] - 1) * roots + (A - 3 * B * B - 2 * B)[..., None]
            safe = torch.where(fz.abs() > 1e-14, fz, torch.ones_like(fz))
            roots = roots - f / safe

        return roots, fz

    def tensor_homogeneous(self, p, t, x, phase="stable", *, derivative_order=1):
        """Local differentiable EOS; callers validate domain before solving."""
        if derivative_order not in (1, 2):
            raise ValueError("EOS derivative order must be one or two")
        if phase not in ("stable", "liquid", "vapor"):
            raise ValueError("Unknown cubic-root selection")
        rt = torch.sqrt(t / self.tc)
        m = 1 + self.kappa * (1 - rt)
        ai = self.a0 * m * m
        ait = -self.a0 * self.kappa * m / (self.tc * rt)
        aitt = self.a0 * self.kappa * (1 + self.kappa) / (2 * self.tc**2 * rt**3)
        aij = torch.sqrt(ai[..., :, None] * ai[..., None, :]) * (1 - self.kij)
        aijt = (
            0.5
            * aij
            * (ait[..., :, None] / ai[..., :, None] + ait[..., None, :] / ai[..., None, :])
        )
        ax = x @ aij
        a = (ax * x).sum(-1)
        at = ((x @ aijt) * x).sum(-1)
        ar = ait / ai
        aijtt = aij * (
            0.5 * (aitt[..., :, None] / ai[..., :, None] + aitt[..., None, :] / ai[..., None, :])
            - 0.25 * (ar[..., :, None] - ar[..., None, :]) ** 2
        )
        att = ((x @ aijtt) * x).sum(-1)
        b = (x * self.b).sum(-1)
        A, B = a * p / (R * t) ** 2, b * p / (R * t)
        roots, fz = self._roots(A, B, derivative_order)
        z = roots
        ratio = torch.log1p(
            2 * math.sqrt(2) * B[..., None] / (z + (1 - math.sqrt(2)) * B[..., None])
        )
        bi = self.b / b[..., None]
        ph = (
            bi[..., None, :] * (z[..., :, None] - 1)
            - torch.log((z - B[..., None]).clamp_min(1e-300))[..., :, None]
            - (a / (2 * math.sqrt(2) * b * R * t))[..., None, None]
            * (2 * ax / a[..., None] - bi)[..., None, :]
            * ratio[..., :, None]
        )
        vol = z * R * t / p
        dpdv = (
            -R * t / (vol - b[..., None]) ** 2
            + a[..., None]
            * (2 * vol + 2 * b[..., None])
            / (vol * vol + 2 * b[..., None] * vol - b[..., None] ** 2) ** 2
        )
        ideal_cp = (x * self.cp).sum(-1)
        cv = (ideal_cp - R)[..., None] + (t * att / (2 * math.sqrt(2) * b))[..., None] * ratio
        dpdt = R / (vol - b[..., None]) - at[..., None] / (
            vol**2 + 2 * b[..., None] * vol - b[..., None] ** 2
        )
        cp = cv - t * dpdt**2 / dpdv
        valid = (
            (z > B[..., None])
            & (dpdv < 0)
            & (fz.abs() > 1e-12)
            & torch.isfinite(ph).all(-1)
            & torch.isfinite(cp)
            & torch.isfinite(cv)
            & (cp > 0)
            & (cv > 0)
        )
        gibbs = (x[..., None, :] * ph).sum(-1)
        if phase == "stable":
            index = torch.where(valid, gibbs, torch.inf).argmin(-1)
        else:
            index = torch.where(valid, z if phase == "liquid" else -z, torch.inf).argmin(-1)
        select = lambda a: torch.gather(a, -1, index[..., None]).squeeze(-1)
        selected_phi = torch.gather(
            ph, -2, index[..., None, None].expand(*index.shape, 1, len(self.components))
        ).squeeze(-2)
        selected_z = select(z)
        selected_g = select(gibbs)
        degenerate = (
            valid
            & ((z - selected_z[..., None]).abs() > 1e-7)
            & ((gibbs - selected_g[..., None]).abs() < 1e-9)
        ).any(-1)
        z, ratio, vol = select(z), select(ratio), select(vol)
        ideal_cp = (x * self.cp).sum(-1)
        h = (
            ideal_cp * (t - self.tref)
            + R * t * (z - 1)
            + (t * at - a) / (2 * math.sqrt(2) * b) * ratio
        )
        s = (
            ideal_cp * torch.log(t / self.tref)
            - R * torch.log(p / self.pref)
            # Inactive components stay exactly zero on a fixed support.
            # xlogy(0, 0) has the right value but its forward derivative
            # produces 0/0 even for a zero composition tangent. Keep the
            # inactive branch constant, with a finite log in both branches.
            - R * torch.where(x > 0, x * x.clamp_min(1e-300).log(), 0.0).sum(-1)
            + R * torch.log(z - B)
            + at / (2 * math.sqrt(2) * b) * ratio
        )
        mw = (x * self.molar_masses).sum(-1)
        return {
            "mole_fractions": x,
            "compressibility_factor": z,
            "log_fugacity_coefficients": selected_phi,
            "molar_volume_m3_mol": vol,
            "molar_enthalpy_j_mol": h,
            "molar_entropy_j_mol_k": s,
            "molar_internal_energy_j_mol": h - p * vol,
            "molar_mass_kg_mol": mw,
            "density_kg_m3": mw / vol,
            "homogeneous_molar_cp_j_mol_k": select(cp),
            "homogeneous_molar_cv_j_mol_k": select(cv),
            "frozen_sound_speed_m_s": torch.sqrt(
                (
                    -vol.square() / mw * (select(dpdv) - t * select(dpdt).square() / select(cv))
                ).clamp_min(1e-30)
            ),
            "valid": valid.any(-1),
            "coexistence_degenerate": degenerate,
        }

    def homogeneous(self, pressure_pa, temperature_k, mole_fractions, phase="stable"):
        p, t, z = map(self.tensor, (pressure_pa, temperature_k, mole_fractions))
        self._validate(p, t, z)
        result = self.tensor_homogeneous(p, t, z, phase)
        if not bool(result["valid"].detach().cpu()):
            raise HydraulicConvergenceError("No nonsingular mechanically stable PR root")
        return result

    def homogeneous_stability_mask(
        self,
        pressures,
        temperatures,
        compositions,
        log_fugacity_coefficients,
        *,
        maximum_seed_rows=4096,
    ):
        """Confirm stable homogeneous states in bounded spatial batches.

        False means "use the full individual flash", never "unstable" or
        "ignore this point". Every original TPD start must resolve before a
        point can be confirmed. Supports differ? Keep the scalar reference.
        """
        if type(maximum_seed_rows) is not int or maximum_seed_rows < 1:
            raise ValueError("Positive stability batch budget required")
        p, t = torch.broadcast_tensors(self.tensor(pressures), self.tensor(temperatures))
        x, phi = self.tensor(compositions), self.tensor(log_fugacity_coefficients)
        if p.ndim != 1 or x.shape != (len(p), len(self.components)) or phi.shape != x.shape:
            raise ValueError("One pressure/temperature and component row per point required")
        confirmed = torch.zeros_like(p, dtype=torch.bool)
        if not len(p):
            return confirmed
        with torch.no_grad():
            valid = (
                torch.isfinite(p).all()
                & torch.isfinite(t).all()
                & torch.isfinite(x).all()
                & torch.isfinite(phi).all()
                & (p >= self.min_pressure_pa).all()
                & (p <= self.max_pressure_pa).all()
                & (t >= self.min_temperature_k).all()
                & (t <= self.max_temperature_k).all()
                & (x >= 0).all()
                & ((x.sum(-1) - 1).abs() <= 1e-10).all()
                & ((1 + self.kappa * (1 - torch.sqrt(t[:, None] / self.tc))) > 0).all()
            )
            if not bool(valid.cpu()):
                raise PropertyModelError("Spatial stability states outside EOS domain")
            support = x > 0
            if not bool((support == support[0]).all().cpu()):
                return confirmed
            active = torch.nonzero(support[0]).flatten()
            n = len(active)
            seeds = 3 + 2 * n
            if seeds > maximum_seed_rows:
                return confirmed
            cache = getattr(self, "_active_flash_cache", None)
            width = max(1, maximum_seed_rows // seeds)
            omega = self.tensor([c.acentric_factor for c in self.components])
            eye = torch.eye(n, device=self.device, dtype=torch.float64)
            for begin in range(0, len(p), width):
                end = min(len(p), begin + width)
                pp, tt, xx, ph = p[begin:end], t[begin:end], x[begin:end], phi[begin:end]
                feed = torch.vmap(self.tensor_homogeneous)(pp, tt, xx)
                matches = (
                    (ph[:, active] - feed["log_fugacity_coefficients"][:, active]).abs().amax(-1)
                    <= 2e-7
                ) & feed["valid"]
                if n == 1:
                    confirmed[begin:end] = matches
                    continue
                xa = xx[:, active]
                base = torch.log(xa)
                lk = torch.log(self.pc[active] / pp[:, None]) + 5.373 * (1 + omega[active]) * (
                    1 - self.tc[active] / tt[:, None]
                )
                starts = torch.cat(
                    (
                        torch.stack((base, base + lk, base - lk), 1),
                        torch.log(0.98 * eye[None, :, :] + 0.02 * xa[:, None, :]),
                        torch.log(0.5 * eye[None, :, :] + 0.5 * xa[:, None, :]),
                    ),
                    1,
                )
                bounded = 80 * torch.tanh(starts / 80)
                points = (bounded[:, :, :-1] - bounded[:, :, -1:]).reshape(-1, n - 1)
                mu = base + feed["log_fugacity_coefficients"][:, active]
                try:
                    value, _, _, accepted = stability_candidates(
                        self,
                        pp[:, None].expand(-1, seeds).reshape(-1),
                        tt[:, None].expand(-1, seeds).reshape(-1),
                        mu[:, None, :].expand(-1, seeds, -1).reshape(-1, n),
                        active,
                        points,
                    )
                    resolved = accepted.reshape(end - begin, seeds).all(-1)
                    stable = value.reshape(end - begin, seeds).amin(-1) >= -1e-8
                    confirmed[begin:end] = matches & resolved & stable
                except torch.linalg.LinAlgError:
                    accepted = torch.zeros(len(points), device=self.device, dtype=torch.bool)
                if isinstance(cache, FlashCache):
                    counts = (
                        torch.stack((accepted.sum(), confirmed[begin:end].sum())).cpu().tolist()
                    )
                    cache.stability_seeds += len(points)
                    cache.stability_batched_seeds += int(counts[0])
                    cache.spatial_stability_batches += 1
                    cache.spatial_stability_points += end - begin
                    cache.spatial_stability_confirmed += int(counts[1])
        return confirmed

    def _stability_predictors(self, p, t, z, chemical, active, starts):
        """All original starts, with independent batched tangents and fallback."""
        bounded = 80 * torch.tanh(torch.stack(starts) / 80)
        point = (bounded[:, :-1] - bounded[:, -1:]).detach()
        value, w, q, accepted = stability_candidates(
            self,
            p.expand(len(starts)),
            t.expand(len(starts)),
            chemical[active].expand(len(starts), -1),
            active,
            point,
        )
        mask = accepted.cpu().tolist()
        cache = getattr(self, "_active_flash_cache", None)
        if isinstance(cache, FlashCache):
            cache.stability_seeds += len(mask)
            cache.stability_batched_seeds += sum(mask)
        return [
            (value[i].detach(), w[i].detach(), {k: v[i].detach() for k, v in q.items()})
            if yes
            else None
            for i, yes in enumerate(mask)
        ]

    def _stability(self, p, t, z, chemical, active, *, _accelerated=True):
        """Numerical multistart TPD minimization, never a global certificate."""
        za = z[active]
        n = len(za)
        if n == 1:
            q = self.tensor_homogeneous(p, t, z)
            return self.tensor(0.0), z, q
        lk = torch.log(self.pc / p) + 5.373 * (
            1 + self.tensor([c.acentric_factor for c in self.components])
        ) * (1 - self.tc / t)
        base = torch.log(za)
        starts = [base, base + lk[active], base - lk[active]]
        starts.extend(
            torch.log(0.98 * torch.eye(n, device=self.device, dtype=torch.float64) + 0.02 * za)
        )
        # Additional deterministic mixed seeds cover interior liquid minima.
        starts.extend(
            torch.log(0.5 * torch.eye(n, device=self.device, dtype=torch.float64) + 0.5 * za)
        )
        try:
            accelerated = (
                self._stability_predictors(p, t, z, chemical, active, starts)
                if _accelerated
                else [None] * len(starts)
            )
        except torch.linalg.LinAlgError:
            # A failed predictor is not a failed phase test: every original
            # seed still has the independent reference minimizer below.
            accelerated = [None] * len(starts)
        minima = []
        for seed, predicted in zip(starts, accelerated):
            if predicted is not None:
                minima.append(predicted)
                continue
            # Independent histories avoid one converged/boundary seed spoiling
            # the curvature model for all other composition searches.
            u = seed.detach().clone().requires_grad_(True)
            optimizer = torch.optim.LBFGS(
                [u],
                lr=1,
                max_iter=150,
                max_eval=220,
                tolerance_grad=1e-10,
                tolerance_change=1e-15,
                line_search_fn="strong_wolfe",
            )

            def values(v=None, u=u):
                v = u if v is None else v
                wa = torch.softmax(v, -1)
                w = torch.zeros_like(z).index_copy(0, active, wa)
                q = self.tensor_homogeneous(p, t, w)
                delta = (
                    torch.log_softmax(v, -1)
                    + q["log_fugacity_coefficients"][active]
                    - chemical[active]
                )
                return wa, w, q, delta

            def closure(optimizer=optimizer, values=values, u=u):
                optimizer.zero_grad()
                # Bound only the optimizer's trial coordinates, not the final
                # equilibrium composition. Newton below uses unbounded logits.
                # This is exactly the previous detached-delta objective's
                # softmax/tanh gradient, without recording the entire cubic
                # EOS graph only to throw its derivatives away.
                with torch.no_grad():
                    wa, _, _, delta = values(80 * torch.tanh(u / 80))
                    weighted = wa * delta
                    u.grad = _gibbs_logit_gradient(u, weighted, wa, -1)
                    return weighted.sum()

            with torch.enable_grad():
                optimizer.step(closure)
            # Polish interior stationary points when Gibbs changes fall below
            # LBFGS' floating-point line-search resolution before the gradient.
            bounded = 80 * torch.tanh(u.detach() / 80)
            v = bounded[:-1] - bounded[-1]

            def stationarity(v):
                _, _, _, delta = values(torch.cat((v, v.new_zeros(1))))
                # Unweighted chemical-potential differences remain conditioned
                # when a soluble trace species has a fraction near 1e-20.
                return delta[:-1] - delta[-1]

            for _ in range(16):
                r = stationarity(v)
                norm = float(r.abs().max().detach().cpu())
                if norm < 1e-9:
                    break
                with torch.enable_grad():
                    jac = torch.autograd.functional.jacobian(stationarity, v, vectorize=True)
                step, info = torch.linalg.solve_ex(jac, -r)
                if int(info.detach().cpu()) or not bool(torch.isfinite(step).all().detach().cpu()):
                    break
                for power in range(12):
                    trial = (v + step * 2.0 ** (-power)).detach()
                    if float(stationarity(trial).abs().max().detach().cpu()) < norm:
                        v = trial
                        break
                else:
                    break
            with torch.no_grad():
                wa, w, q, delta = values(torch.cat((v, v.new_zeros(1))))
                tpd = (wa * delta).sum()
                grad = wa * (delta - tpd)
                resolved = (grad.abs().max() < 2e-7) & q["valid"] & torch.isfinite(tpd)
                if not bool(resolved.cpu()):
                    raise HydraulicConvergenceError(
                        f"Unresolved PR stability seed at P={float(p.cpu())}, T={float(t.cpu())}: "
                        f"gradient={float(grad.abs().max().cpu())}, composition={w.cpu().tolist()}"
                    )
                minima.append((tpd.detach(), w.detach(), {k: v.detach() for k, v in q.items()}))
        return min(minima, key=lambda entry: float(entry[0].cpu()))

    def _allocation(self, p, t, z, active, allocation):
        """Conservative Gibbs minimization then equal-fugacity Newton polish."""
        za = z[active]
        u = torch.log(allocation / za).detach().clone().requires_grad_(True)

        def decode(v):
            loga = torch.log_softmax(v, dim=0) + torch.log(za)
            a = torch.exp(loga)
            logbeta = torch.logsumexp(loga, -1)
            beta = torch.exp(logbeta)
            logxa = loga - logbeta[:, None]
            xa = torch.exp(logxa)
            x = torch.zeros((len(a), len(z)), dtype=z.dtype, device=z.device).index_copy(
                -1, active, xa
            )
            q = self.tensor_homogeneous(p, t, x)
            mu = logxa + q["log_fugacity_coefficients"][:, active]
            return a, beta, x, q, mu

        optimizer = torch.optim.LBFGS(
            [u],
            lr=1,
            max_iter=250,
            max_eval=350,
            tolerance_grad=1e-12,
            tolerance_change=1e-15,
            line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad()
            with torch.no_grad():
                bounded = 80 * torch.tanh(u / 80)
                a, _, _, _, mu = decode(bounded)
                weighted = a * mu
                probabilities = torch.log_softmax(bounded, dim=0).exp()
                u.grad = _gibbs_logit_gradient(u, weighted, probabilities, 0)
                return weighted.sum()

        with torch.enable_grad():
            optimizer.step(closure)
        # Fix the redundant softmax reference phase before Newton.
        bounded = 80 * torch.tanh(u.detach() / 80)
        v = bounded[:-1] - bounded[-1]

        def residual(v):
            mu = decode(torch.cat((v, v.new_zeros((1, v.shape[-1]))), dim=0))[-1]
            return (mu[:-1] - mu[-1]).reshape(-1)

        for _ in range(15):
            r = residual(v)
            norm = float(r.abs().max().detach().cpu())
            if norm < 2e-9:
                break
            with torch.enable_grad():
                jac = torch.autograd.functional.jacobian(residual, v, vectorize=True).reshape(
                    r.numel(), r.numel()
                )
            try:
                step = torch.linalg.solve(jac, -r).reshape_as(v)
            except RuntimeError as exc:
                raise HydraulicConvergenceError("Singular phase-equilibrium tangent") from exc
            accepted = False
            for power in range(16):
                trial = (v + step * 2.0 ** (-power)).detach()
                newnorm = float(residual(trial).abs().max().detach().cpu())
                if math.isfinite(newnorm) and newnorm < norm:
                    v, accepted = trial, True
                    break
            if not accepted:
                raise HydraulicConvergenceError("Phase-equilibrium Newton line search failed")
        a, beta, x, q, mu = decode(torch.cat((v, v.new_zeros((1, v.shape[-1]))), dim=0))
        error = (mu[:-1] - mu[-1]).abs().max()
        if not bool(((error < 1e-8) & q["valid"].all() & (beta > 1e-12).all()).detach().cpu()):
            raise HydraulicConvergenceError("Unresolved/vanishing phase split")
        return (
            a.detach(),
            beta.detach(),
            x.detach(),
            {k: v.detach() for k, v in q.items()},
            mu[-1].detach(),
            error.detach(),
        )

    @cached_equilibrium
    def flash_pt(
        self, pressure_pa, temperature_k, mole_fractions, *, _allow_pure_coexistence=False
    ):
        p, t, z = (self.tensor(v).detach() for v in (pressure_pa, temperature_k, mole_fractions))
        self._validate(p, t, z)
        active = torch.nonzero(z > 0).flatten()
        feed = self.tensor_homogeneous(p, t, z)
        if not bool(feed["valid"].cpu()):
            raise HydraulicConvergenceError("Invalid feed EOS root")
        if (
            len(active) == 1
            and bool(feed["coexistence_degenerate"].cpu())
            and not _allow_pure_coexistence
        ):
            raise PropertyModelError("Pure-fluid PT coexistence needs energy or phase fraction")
        chemical = torch.log(z.clamp_min(1e-300)) + feed["log_fugacity_coefficients"]
        allocation = z[active][None, :]
        beta, x = self.tensor([1.0]), z[None, :]
        q = {k: v[None] for k, v in feed.items()}
        fugacity_error = self.tensor(0.0)
        initial_g = (z[active] * chemical[active]).sum()
        for phase_count in range(1, self.max_phases + 1):
            minimum, trial, _ = self._stability(p, t, z, chemical, active)
            if float(minimum.cpu()) >= -1e-8:
                break
            if phase_count == self.max_phases:
                raise PropertyModelError("Equilibrium requires additional phases beyond max_phases")
            # Seed a new phase without ever changing component totals.
            fraction = torch.minimum(
                self.tensor(0.15), 0.5 * (z[active] / trial[active].clamp_min(1e-300)).min()
            )
            new = fraction * trial[active]
            allocation = torch.cat((allocation * (1 - new / z[active]), new[None, :]))
            allocation, beta, x, q, mu, fugacity_error = self._allocation(
                p, t, z, active, allocation
            )
            chemical = torch.zeros_like(z).index_copy(0, active, mu)
        order = q["molar_volume_m3_mol"].argsort()
        beta, x = beta[order], x[order]
        q = {k: v[order] for k, v in q.items()}
        balance = (beta[:, None] * x).sum(0) - z
        avg = lambda key: (beta * q[key]).sum()
        final_g = (
            beta
            * (
                x[:, active] * (torch.log(x[:, active]) + q["log_fugacity_coefficients"][:, active])
            ).sum(-1)
        ).sum()
        if not bool(
            (
                (balance.abs().max() < 1e-10)
                & (final_g <= initial_g + 1e-10)
                & torch.isfinite(final_g)
            ).cpu()
        ):
            raise HydraulicConvergenceError("Flash component/Gibbs audit failed")
        phases = [
            {**{k: v[i] for k, v in q.items() if k != "valid"}, "molar_fraction": beta[i]}
            for i in range(len(beta))
        ]
        return {
            "pressure_pa": p,
            "temperature_k": t,
            "mole_fractions": z,
            "phase_count": len(beta),
            "phases": phases,
            "molar_enthalpy_j_mol": avg("molar_enthalpy_j_mol"),
            "molar_internal_energy_j_mol": avg("molar_internal_energy_j_mol"),
            "molar_entropy_j_mol_k": avg("molar_entropy_j_mol_k"),
            "molar_volume_m3_mol": avg("molar_volume_m3_mol"),
            "density_kg_m3": (z * self.molar_masses).sum() / avg("molar_volume_m3_mol"),
            "component_balance_error": balance.abs().max(),
            "log_fugacity_error": fugacity_error,
            "minimum_tpd": minimum,
            "gibbs_change_rt": final_g - initial_g,
            "stability_search": "deterministic multistart TPD; not a global certificate",
        }

    def _flash_property(self, p, target, z, key, temperature_guess_k=None):
        """Cheap homogeneous caloric predictor, accepted ONLY by full PT flash.

        No phase search is skipped in an accepted result. The original
        bracketed multiphase/latent-heat solve remains the fallback.
        """
        target = _number(target, key)
        guess = (
            0.5 * (self.min_temperature_k + self.max_temperature_k)
            if temperature_guess_k is None
            else _number(temperature_guess_k, "temperature_guess_k", positive=True)
        )
        if not self.min_temperature_k <= guess <= self.max_temperature_k:
            raise PropertyModelError("Temperature guess outside caloric domain")
        pp, zz = self.tensor(p).detach(), self.tensor(z).detach()
        self._validate(pp, self.tensor(guess), zz)
        tolerance = 1e-7 + abs(target) * 1e-10
        previous = math.inf
        for _ in range(12):
            with torch.no_grad():
                q = self.tensor_homogeneous(pp, self.tensor(guess), zz)
                derivative = q["homogeneous_molar_cp_j_mol_k"]
                if key == "molar_entropy_j_mol_k":
                    derivative = derivative / guess
                packet = torch.stack(
                    (
                        q[key],
                        derivative,
                        q["valid"].to(torch.float64),
                        q["coexistence_degenerate"].to(torch.float64),
                    )
                )
                value, slope, valid, degenerate = packet.cpu().tolist()
            error = value - target
            if (
                not valid
                or degenerate
                or not math.isfinite(error)
                or not math.isfinite(slope)
                or slope <= 0
            ):
                break
            if abs(error) <= 0.25 * tolerance:
                try:
                    result = self.flash_pt(pp, guess, zz)
                except (HydraulicConvergenceError, PropertyModelError):
                    break
                defect = float(result[key].detach().cpu()) - target
                if result["phase_count"] == 1 and abs(defect) <= tolerance:
                    cache = getattr(self, "_active_flash_cache", None)
                    if isinstance(cache, FlashCache):
                        cache.caloric_fast_accepts += 1
                    return dict(result, property_residual=self.tensor(defect))
                break
            if abs(error) >= previous:
                break
            previous = abs(error)
            change = max(-50.0, min(50.0, -error / slope))
            trial = max(self.min_temperature_k, min(self.max_temperature_k, guess + change))
            if trial == guess:
                break
            guess = trial
        cache = getattr(self, "_active_flash_cache", None)
        if isinstance(cache, FlashCache):
            cache.caloric_bracket_fallbacks += 1
        return self._flash_property_bracketed(p, target, z, key, temperature_guess_k)

    def _flash_property_bracketed(self, p, target, z, key, temperature_guess_k=None):
        p = self.tensor(p)
        target = _number(target, key)
        low, high = self.min_temperature_k, self.max_temperature_k
        if temperature_guess_k is not None:
            guess = _number(temperature_guess_k, "temperature_guess_k", positive=True)
            if not low <= guess <= high:
                raise PropertyModelError("Temperature guess outside caloric domain")
        else:
            guess = 0.5 * (low + high)
        cache = {}

        def evaluate(t):
            if t not in cache:
                cache[t] = self.flash_pt(p, t, z, _allow_pure_coexistence=True)
            return float(cache[t][key].cpu()) - target

        # Expand locally first to avoid remote, unnecessary phase searches.
        span = max(1.0, 0.01 * guess)
        while True:
            lo, hi = max(low, guess - span), min(high, guess + span)
            fl, fh = evaluate(lo), evaluate(hi)
            if fl * fh <= 0:
                break
            if lo == low and hi == high:
                raise PropertyModelError("Thermal flash target outside declared domain")
            span *= 2
        tolerance = 1e-7 + abs(target) * 1e-10
        for _ in range(70):
            # Safeguarded secant; bisection limits pathological bracket motion.
            f = 0.5 if fh == fl else min(0.9, max(0.1, -fl / (fh - fl)))
            t = lo + f * (hi - lo)
            error = evaluate(t)
            if hi - lo < 1e-7 and int((self.tensor(z) > 0).sum().cpu()) == 1:
                liquid = self.tensor_homogeneous(p, self.tensor(t), self.tensor(z), "liquid")
                vapor = self.tensor_homogeneous(p, self.tensor(t), self.tensor(z), "vapor")
                active = torch.nonzero(self.tensor(z) > 0).flatten()
                gap = float((vapor[key] - liquid[key]).cpu())
                fugacity = (
                    (
                        liquid["log_fugacity_coefficients"][active]
                        - vapor["log_fugacity_coefficients"][active]
                    )
                    .abs()
                    .max()
                )
                if gap > 1e-8 and float(fugacity.cpu()) < 1e-8:
                    beta = (target - float(liquid[key].cpu())) / gap
                    if 0 <= beta <= 1:
                        result = dict(cache[t])
                        result["phase_count"] = 2
                        result["phases"] = [
                            dict(q, molar_fraction=self.tensor(b))
                            for q, b in ((liquid, 1 - beta), (vapor, beta))
                        ]
                        for name in (
                            "molar_enthalpy_j_mol",
                            "molar_entropy_j_mol_k",
                            "molar_internal_energy_j_mol",
                            "molar_volume_m3_mol",
                        ):
                            result[name] = (1 - beta) * liquid[name] + beta * vapor[name]
                        result["density_kg_m3"] = (
                            self.tensor(z) * self.molar_masses
                        ).sum() / result["molar_volume_m3_mol"]
                        result["log_fugacity_error"] = fugacity
                        result["property_residual"] = result[key] - target
                        return result
            if abs(error) <= tolerance:
                result = dict(cache[t], property_residual=self.tensor(error))
                return result
            if error * fl > 0:
                lo, fl = t, error
            else:
                hi, fh = t, error
        raise HydraulicConvergenceError("Thermal flash failed to converge")

    @cached_equilibrium
    def flash_ph(self, pressure_pa, molar_enthalpy_j_mol, mole_fractions, temperature_guess_k=None):
        return self._flash_property(
            pressure_pa,
            molar_enthalpy_j_mol,
            mole_fractions,
            "molar_enthalpy_j_mol",
            temperature_guess_k,
        )

    @cached_equilibrium
    def flash_ps(
        self, pressure_pa, molar_entropy_j_mol_k, mole_fractions, temperature_guess_k=None
    ):
        return self._flash_property(
            pressure_pa,
            molar_entropy_j_mol_k,
            mole_fractions,
            "molar_entropy_j_mol_k",
            temperature_guess_k,
        )

    def equilibrium_derivatives(
        self, pressure_pa, temperature_k, mole_fractions, *, equilibrium=None
    ):
        """Implicit first derivatives on a fixed, nonsingular phase branch.

        Composition columns are active component mole amounts normalized to
        sum one at the evaluation point. Exact-zero species are not perturbed;
        appearance/critical boundaries need a new active-set flash.
        """
        result = (
            self.flash_pt(pressure_pa, temperature_k, mole_fractions)
            if equilibrium is None
            else equilibrium
        )
        p, t, z = (result[k] for k in ("pressure_pa", "temperature_k", "mole_fractions"))
        active = torch.nonzero(z > 0).flatten()
        y0 = torch.cat((p.reshape(1), t.reshape(1), z[active]))
        keys = (
            "molar_enthalpy_j_mol",
            "molar_entropy_j_mol_k",
            "molar_volume_m3_mol",
            "molar_internal_energy_j_mol",
        )
        count = result["phase_count"]

        def decode(v, y):
            pp, tt = y[:2]
            za = y[2:] / y[2:].sum()
            if count == 1:
                beta = y.new_ones(1)
                xa = za[None, :]
            else:
                logits = torch.cat(
                    (v.reshape(count - 1, len(active)), v.new_zeros((1, len(active))))
                )
                loga = torch.log_softmax(logits, 0) + torch.log(za)
                lb = torch.logsumexp(loga, -1)
                beta = torch.exp(lb)
                xa = torch.exp(loga - lb[:, None])
            x = y.new_zeros((count, len(z))).index_copy(-1, active, xa)
            q = self.tensor_homogeneous(pp, tt, x)
            values = torch.stack([(beta * q[k]).sum() for k in keys])
            mu = torch.log(xa) + q["log_fugacity_coefficients"][:, active]
            f = (mu[:-1] - mu[-1]).reshape(-1)
            return values, f

        v0 = y0.new_empty(0)
        if count > 1:
            allocation = torch.stack(
                [q["molar_fraction"] * q["mole_fractions"][active] for q in result["phases"]]
            )
            logs = torch.log(allocation)
            v0 = (logs[:-1] - logs[-1]).reshape(-1)
        with torch.enable_grad():
            gy = torch.autograd.functional.jacobian(lambda y: decode(v0, y)[0], y0, vectorize=True)
            if count > 1:
                fv = torch.autograd.functional.jacobian(
                    lambda v: decode(v, y0)[1], v0, vectorize=True
                )
                fy = torch.autograd.functional.jacobian(
                    lambda y: decode(v0, y)[1], y0, vectorize=True
                )
                gv = torch.autograd.functional.jacobian(
                    lambda v: decode(v, y0)[0], v0, vectorize=True
                )
                try:
                    gy = gy - gv @ torch.linalg.solve(fv, fy)
                except RuntimeError as exc:
                    raise HydraulicConvergenceError(
                        "Singular equilibrium-property tangent"
                    ) from exc
        if not bool(torch.isfinite(gy).all().detach().cpu()):
            raise HydraulicConvergenceError("Nonfinite equilibrium-property tangent")
        return {
            "property_names": keys,
            "columns": (
                "pressure_pa",
                "temperature_k",
                *(self.names[i] for i in active.cpu().tolist()),
            ),
            "jacobian": gy.detach(),
            "phase_count": count,
        }

    def flash_uv(
        self,
        component_moles,
        internal_energy_j,
        volume_m3,
        *,
        pressure_guess_pa,
        temperature_guess_k,
    ):
        """Rigid equilibrium inventory: recover P,T from conserved N,U,V.

        Bounded Newton uses implicit equilibrium derivatives; no SciPy or
        conversion of U to H at an invented pressure. Guesses are mandatory.
        """
        n = self.tensor(component_moles)
        if n.shape != (len(self.components),) or not bool(
            (torch.isfinite(n).all() & (n >= 0).all() & (n.sum() > 0)).cpu()
        ):
            raise PropertyModelError("UV flash needs finite nonnegative component inventories")
        energy = _number(internal_energy_j, "internal_energy_j")
        volume = _number(volume_m3, "volume_m3", positive=True)
        total = n.sum()
        z = n / total
        target_u = energy / float(total.cpu())
        target_v = volume / float(total.cpu())
        p = _number(pressure_guess_pa, "pressure_guess_pa", positive=True)
        t = _number(temperature_guess_k, "temperature_guess_k", positive=True)
        uscale = max(1.0, abs(target_u), R * t)

        def residual(q):
            return torch.stack(
                (
                    (q["molar_internal_energy_j_mol"] - target_u) / uscale,
                    torch.log(q["molar_volume_m3_mol"] / target_v),
                )
            )

        if int((z > 0).sum().cpu()) == 1:
            return self._pure_flash_uv(n, energy, volume, p, t)
        q = self.flash_pt(p, t, z)
        for iteration in range(25):
            r = residual(q)
            norm = float(r.abs().max().cpu())
            if norm < 2e-10:
                return {
                    **q,
                    "component_moles": n,
                    "internal_energy_j": total * q["molar_internal_energy_j_mol"],
                    "volume_m3": total * q["molar_volume_m3_mol"],
                    "energy_residual_j": total * q["molar_internal_energy_j_mol"] - energy,
                    "volume_residual_m3": total * q["molar_volume_m3_mol"] - volume,
                    "nonlinear_iterations": iteration,
                }
            jac = self.equilibrium_derivatives(p, t, z, equilibrium=q)["jacobian"]
            J = torch.stack((jac[3, :2] / uscale, jac[2, :2] / q["molar_volume_m3_mol"]))
            J[:, 0] *= p
            try:
                step = torch.linalg.solve(J, -r)
            except RuntimeError as exc:
                raise HydraulicConvergenceError("Singular UV flash") from exc
            # Limit trial motion; every accepted state receives a full PT audit.
            factor = min(
                1.0,
                1.0 / max(1.0, abs(float(step[0].cpu()))),
                20.0 / max(20.0, abs(float(step[1].cpu()))),
            )
            accepted = False
            for power in range(16):
                a = factor * 2.0 ** (-power)
                pp, tt = p * math.exp(a * float(step[0].cpu())), t + a * float(step[1].cpu())
                if not (
                    self.min_pressure_pa <= pp <= self.max_pressure_pa
                    and self.min_temperature_k <= tt <= self.max_temperature_k
                ):
                    continue
                try:
                    trial = self.flash_pt(pp, tt, z)
                except (HydraulicConvergenceError, PropertyModelError):
                    continue
                if float(residual(trial).abs().max().cpu()) < norm:
                    p, t, q, accepted = pp, tt, trial, True
                    break
            if not accepted:
                raise HydraulicConvergenceError("UV flash line search failed")
        raise HydraulicConvergenceError("UV flash iteration budget exhausted")

    def _polish_pure_uv_coexistence(self, n, energy, volume, guess):
        """Joint P/T/beta refinement avoids the accuracy floor of nested PH.

        Called on an already identified two-phase branch near the UV solution.
        No finite-difference or host EOS solver is used.
        """
        from .torch_linear import TorchNewton

        total = n.sum()
        z = n / total
        active = torch.nonzero(z > 0).flatten()
        target_u = energy / float(total.cpu())
        target_v = volume / float(total.cpu())
        uscale = max(1.0, abs(target_u), R * float(guess["temperature_k"].cpu()))

        def evaluate(y):
            p, t, beta = y[0].exp(), y[1] * self.tref, y[2]
            liquid = self.tensor_homogeneous(p, t, z, "liquid")
            vapor = self.tensor_homogeneous(p, t, z, "vapor")
            u = (1 - beta) * liquid["molar_internal_energy_j_mol"] + beta * vapor[
                "molar_internal_energy_j_mol"
            ]
            v = (1 - beta) * liquid["molar_volume_m3_mol"] + beta * vapor["molar_volume_m3_mol"]
            mu = (
                liquid["log_fugacity_coefficients"][active]
                - vapor["log_fugacity_coefficients"][active]
            ).sum()
            return torch.stack((mu, (u - target_u) / uscale, torch.log(v / target_v))), (
                p,
                t,
                beta,
                liquid,
                vapor,
                u,
                v,
            )

        initial = torch.stack(
            (
                guess["pressure_pa"].log(),
                guess["temperature_k"] / self.tref,
                guess["phases"][1]["molar_fraction"],
            )
        )
        low = self.tensor([math.log(self.min_pressure_pa), self.min_temperature_k / self.tref, 0.0])
        high = self.tensor(
            [math.log(self.max_pressure_pa), self.max_temperature_k / self.tref, 1.0]
        )
        solver = TorchNewton(self.device, maximum_iterations=15)
        solver.host_convergence = True
        with torch.no_grad():
            y, _, ok, count = solver.solve(
                lambda y: evaluate(y)[0], initial, low, high, tolerance=2e-13
            )
            residual, data = evaluate(y)
            pp, tt, beta, liquid, vapor, u, v = data
            if (
                not ok
                or not bool((liquid["valid"] & vapor["valid"]).cpu())
                or float(residual.abs().max().cpu()) > 1e-11
                or float((vapor["molar_volume_m3_mol"] - liquid["molar_volume_m3_mol"]).cpu())
                <= 1e-8 * target_v
            ):
                raise HydraulicConvergenceError("Pure-fluid joint UV refinement failed")
            result = {
                **guess,
                "pressure_pa": pp,
                "temperature_k": tt,
                "phase_count": 2,
                "phases": [dict(liquid, molar_fraction=1 - beta), dict(vapor, molar_fraction=beta)],
                "component_moles": n,
                "internal_energy_j": total * u,
                "volume_m3": total * v,
                "energy_residual_j": total * u - energy,
                "volume_residual_m3": total * v - volume,
                "log_fugacity_error": residual[0].abs(),
                "nonlinear_iterations": count,
            }
            for key in (
                "molar_internal_energy_j_mol",
                "molar_volume_m3_mol",
                "molar_entropy_j_mol_k",
                "molar_enthalpy_j_mol",
            ):
                result[key] = (1 - beta) * liquid[key] + beta * vapor[key]
            edge = 16 * torch.finfo(beta.dtype).eps
            if float(beta.cpu()) <= edge or float((1 - beta).cpu()) <= edge:
                remaining = liquid if float(beta.cpu()) <= edge else vapor
                ru = remaining["molar_internal_energy_j_mol"]
                rv = remaining["molar_volume_m3_mol"]
                if (
                    abs(float(((ru - target_u) / uscale).cpu())) < 2e-12
                    and abs(float(torch.log(rv / target_v).cpu())) < 2e-12
                ):
                    result["phase_count"] = 1
                    result["phases"] = [dict(remaining, molar_fraction=beta.new_ones(()))]
                    for key in (
                        "molar_internal_energy_j_mol",
                        "molar_volume_m3_mol",
                        "molar_entropy_j_mol_k",
                        "molar_enthalpy_j_mol",
                    ):
                        result[key] = remaining[key]
                    result.update(
                        internal_energy_j=total * ru,
                        volume_m3=total * rv,
                        energy_residual_j=total * ru - energy,
                        volume_residual_m3=total * rv - volume,
                    )
                    v = rv
            result["density_kg_m3"] = (z * self.molar_masses).sum() / v
            # The input PH property is no longer the scalar refinement equation.
            result.pop("property_residual", None)
            return result

    def _pure_flash_uv(self, n, energy, volume, p, t):
        """Follow PH with H=U+PV, including pure-fluid latent heat plateaus."""
        total = n.sum()
        z = n / total
        target_u = energy / float(total.cpu())
        target_v = volume / float(total.cpu())

        def evaluate(pp, tt):
            return self.flash_ph(pp, target_u + pp * target_v, z, temperature_guess_k=tt)

        q = evaluate(p, t)
        for iteration in range(30):
            v = q["molar_volume_m3_mol"]
            r = float(torch.log(v / target_v).cpu())
            if q["phase_count"] == 2 and abs(r) < 1e-3:
                try:
                    return self._polish_pure_uv_coexistence(n, energy, volume, q)
                except HydraulicConvergenceError:
                    # A nearby active-set boundary may require the outer PH path.
                    pass
            if abs(r) < 2e-10:
                return {
                    **q,
                    "component_moles": n,
                    "internal_energy_j": total * q["molar_internal_energy_j_mol"],
                    "volume_m3": total * v,
                    "energy_residual_j": total * q["molar_internal_energy_j_mol"] - energy,
                    "volume_residual_m3": total * v - volume,
                    "nonlinear_iterations": iteration,
                }
            t = float(q["temperature_k"].cpu())
            if q["phase_count"] == 1:
                jac = self.equilibrium_derivatives(p, t, z, equilibrium=q)["jacobian"]
                slope = jac[2, 0] + jac[2, 1] * (target_v - jac[0, 0]) / jac[0, 1]
            else:
                beta = q["phases"][1]["molar_fraction"]
                y0 = torch.stack((self.tensor(p), self.tensor(t), beta))
                active = torch.nonzero(z > 0).flatten()

                def values(y, active=active):
                    pp, tt, b = y.unbind()
                    l = self.tensor_homogeneous(pp, tt, z, "liquid")
                    g = self.tensor_homogeneous(pp, tt, z, "vapor")
                    h = (1 - b) * l["molar_enthalpy_j_mol"] + b * g["molar_enthalpy_j_mol"]
                    vv = (1 - b) * l["molar_volume_m3_mol"] + b * g["molar_volume_m3_mol"]
                    mu = (
                        l["log_fugacity_coefficients"][active]
                        - g["log_fugacity_coefficients"][active]
                    ).sum()
                    return torch.stack((mu, h - target_u - pp * target_v, vv))

                with torch.enable_grad():
                    jac = torch.autograd.functional.jacobian(values, y0, vectorize=True)
                motion = torch.linalg.solve(jac[:2, 1:], -jac[:2, 0])
                slope = jac[2, 0] + jac[2, 1:] @ motion
            tangent = float((p / v * slope).detach().cpu())
            if not math.isfinite(tangent) or abs(tangent) < 1e-12:
                raise HydraulicConvergenceError("Singular pure-fluid UV tangent")
            step = max(-1.0, min(1.0, -r / tangent))
            for power in range(16):
                pp = p * math.exp(step * 2.0 ** (-power))
                if not self.min_pressure_pa <= pp <= self.max_pressure_pa:
                    continue
                try:
                    trial = evaluate(pp, t)
                except (HydraulicConvergenceError, PropertyModelError):
                    continue
                if abs(float(torch.log(trial["molar_volume_m3_mol"] / target_v).cpu())) < abs(r):
                    p, q = pp, trial
                    break
            else:
                raise HydraulicConvergenceError("Pure-fluid UV line search failed")
        raise HydraulicConvergenceError("Pure-fluid UV iteration budget exhausted")
