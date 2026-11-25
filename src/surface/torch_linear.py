"""Float64 PyTorch sparse operators and bounded Newton; no SciPy execution.

Topology is compiled once on the host. Tensor arithmetic, Krylov vectors and
Jacobians stay on the requested device. No implicit CPU or dense large fallback.
"""

import math

import torch


def checked_device(device):
    device = torch.device(device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Surface backend supports cpu or cuda")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; CPU fallback is disabled")
        index = torch.cuda.current_device() if device.index is None else device.index
        if index >= torch.cuda.device_count():
            raise ValueError("CUDA device index is unavailable")
        device = torch.device("cuda", index)
    return device


class TorchCOO:
    """Row-sorted gather/segmented reduction, without CUDA scatter atomics."""

    def __init__(self, rows, cols, values, size, device=None):
        self.size = int(size)
        self.device = checked_device(device or values.device)
        order = sorted(range(len(rows)), key=lambda i: (rows[i], cols[i]))
        self.rows = torch.tensor([rows[i] for i in order], dtype=torch.long, device=self.device)
        self.cols = torch.tensor([cols[i] for i in order], dtype=torch.long, device=self.device)
        self.order = torch.tensor(order, dtype=torch.long, device=self.device)
        self.values = values.index_select(-1, self.order)
        counts = [0] * size
        for r in rows:
            counts[r] += 1
        self.counts = torch.tensor(counts, dtype=torch.long, device=self.device)
        self.diagonal_mask = self.rows == self.cols
        self.groups = []
        if self.device.type == "cuda":
            # segment_reduce assigns oversized work groups to degree-1/2 rows.
            # Compile the SAME local reductions by degree for coalesced gathers.
            offsets = [0]
            by_degree = {}
            for row, count in enumerate(counts):
                offsets.append(offsets[-1] + count)
                by_degree.setdefault(count, []).append(row)
            permutation = []
            for degree, rows_at_degree in sorted(by_degree.items()):
                permutation.extend(rows_at_degree)
                indices = torch.tensor(
                    [list(range(offsets[r], offsets[r + 1])) for r in rows_at_degree],
                    dtype=torch.long,
                    device=self.device,
                ).reshape(len(rows_at_degree), degree)
                self.groups.append(indices)
            inverse = [0] * self.size
            for index, row in enumerate(permutation):
                inverse[row] = index
            self.inverse = torch.tensor(inverse, dtype=torch.long, device=self.device)

    def reduce(self, products):
        if self.device.type == "cuda":
            pieces = [products[..., indices].sum(-1) for indices in self.groups]
            return torch.cat(pieces, dim=-1).index_select(-1, self.inverse)
        counts = self.counts.expand(*products.shape[:-1], self.size)
        return torch.segment_reduce(products, "sum", lengths=counts, axis=-1, unsafe=True)

    def mv(self, x, values=None):
        v = self.values if values is None else values
        products = v * x.index_select(-1, self.cols)
        return self.reduce(products)

    def row_scale(self):
        return self.reduce(self.values.abs()).clamp_min(1e-12)

    def diagonal(self):
        return self.reduce(torch.where(self.diagonal_mask, self.values, 0.0))


class BlockJacobi:
    """Local physical-block inverses; topology on host, factors on device.

    This is a preconditioner, never a replacement for the global sparse solve.
    Singular local factors revert to diagonal scaling, and the owning Newton
    still checks the full unpreconditioned residual.
    """

    def __init__(self, rows, cols, blocks, size, device):
        owners, local, groups = {}, {}, {}
        for block in blocks:
            indices = list(block)
            if not indices or len(indices) > 256:
                raise ValueError("Physical preconditioner blocks require 1..256 unknowns")
            group = groups.setdefault(len(indices), {"blocks": [], "entries": []})
            slot = len(group["blocks"])
            group["blocks"].append(indices)
            for j, index in enumerate(indices):
                if index in owners or not 0 <= index < size:
                    raise ValueError("Preconditioner blocks must partition the unknowns")
                owners[index] = (len(indices), slot)
                local[index] = j
        if len(owners) != size:
            raise ValueError("Preconditioner blocks must cover all unknowns")
        for entry, (r, c) in enumerate(zip(rows, cols)):
            if owners[r] == owners[c]:
                degree, slot = owners[r]
                groups[degree]["entries"].append((slot, local[r], local[c], entry))
        self.size, self.groups = size, []
        for degree, group in sorted(groups.items()):
            idx = torch.tensor(group["blocks"], dtype=torch.long, device=device)
            entries = group["entries"]
            coordinates = tuple(
                torch.tensor([e[j] for e in entries], dtype=torch.long, device=device)
                for j in range(4)
            )
            self.groups.append((degree, idx, coordinates))
        self.factors = []

    def update(self, values):
        self.factors = []
        for degree, idx, (slot, r, c, entry) in self.groups:
            matrix = values.new_zeros((len(idx), degree, degree)).index_put(
                (slot, r, c), values[entry]
            )
            scale = matrix.abs().amax(-1).clamp_min(1e-30)
            inverse, info = torch.linalg.inv_ex(matrix / scale[..., None], check_errors=False)
            inverse = inverse / scale[:, None, :]
            diagonal = matrix.diagonal(dim1=-2, dim2=-1)
            safe = torch.where(diagonal.abs() > 1e-12, diagonal, scale)
            fallback = torch.diag_embed(safe.reciprocal())
            valid = (info == 0) & torch.isfinite(inverse).all(dim=(-2, -1))
            self.factors.append(torch.where(valid[:, None, None], inverse, fallback))

    def __call__(self, rhs):
        result = torch.zeros_like(rhs)
        for (_degree, idx, _coordinates), inverse in zip(self.groups, self.factors):
            solved = (inverse @ rhs[idx][..., None]).squeeze(-1)
            result = result.index_copy(0, idx.flatten(), solved.flatten())
        return result


def gmres_device(operator, b, diagonal, restart=24, cycles=2, *, preconditioner=None):
    """Fixed-work left-preconditioned GMRES; no host scalar convergence reads."""
    n = b.numel()
    k = min(restart, n)
    safe = torch.where(diagonal.abs() > 1e-12, diagonal, torch.where(diagonal < 0, -1e-12, 1e-12))
    precondition = (lambda v: v / safe) if preconditioner is None else preconditioner
    x = torch.zeros_like(b)
    for _ in range(cycles):
        r = precondition(b - operator(x))
        beta = torch.linalg.vector_norm(r)
        basis = [r / beta.clamp_min(1e-30)]
        H = torch.zeros((k + 1, k), dtype=b.dtype, device=b.device)
        g = torch.zeros(k + 1, dtype=b.dtype, device=b.device)
        g[0] = beta
        cs = []
        sn = []
        for j in range(k):
            w = precondition(operator(basis[j]))
            for _ in range(2):
                for i in range(j + 1):
                    h = torch.dot(basis[i], w)
                    H[i, j] += h
                    w = w - h * basis[i]
            H[j + 1, j] = torch.linalg.vector_norm(w)
            basis.append(w / H[j + 1, j].clamp_min(1e-30))
            for i in range(j):
                a = H[i, j].clone()
                c = H[i + 1, j].clone()
                H[i, j] = cs[i] * a + sn[i] * c
                H[i + 1, j] = -sn[i] * a + cs[i] * c
            radius = torch.hypot(H[j, j], H[j + 1, j])
            cosine = torch.where(radius > 1e-30, H[j, j] / radius.clamp_min(1e-30), 1.0)
            sine = H[j + 1, j] / radius.clamp_min(1e-30)
            cs.append(cosine)
            sn.append(sine)
            H[j, j] = radius
            H[j + 1, j].zero_()
            old = g[j].clone()
            g[j] = cosine * old
            g[j + 1] = -sine * old
        upper = H[:k].clone()
        d = upper.diagonal()
        upper = (
            upper - torch.diag(d) + torch.diag(torch.where(d.abs() > 1e-25, d, torch.ones_like(d)))
        )
        y = torch.linalg.solve_triangular(upper, g[:k, None], upper=True).squeeze(-1)
        x = x + torch.stack(basis[:k], dim=1) @ y
    return x


class ColoredJacobian:
    def __init__(self, size, rows, cols, device, *, chunk_size=64, maximum_seed_entries=1_048_576):
        if type(chunk_size) is not int or chunk_size < 1:
            raise ValueError("Positive Jacobian direction batch size required")
        if type(maximum_seed_entries) is not int or maximum_seed_entries < 1:
            raise ValueError("Positive Jacobian seed entry budget required")
        self.size = size
        self.device = checked_device(device)
        rows, cols = list(rows), list(cols)
        if (
            size < 1
            or len(rows) != len(cols)
            or any(not (0 <= r < size and 0 <= c < size) for r, c in zip(rows, cols))
        ):
            raise ValueError("Square Jacobian graph and matching in-range coordinates required")
        self.maximum_seed_entries = maximum_seed_entries
        self.chunk_size = chunk_size = min(chunk_size, max(1, maximum_seed_entries // size))
        row_cols = [set() for _ in range(size)]
        col_rows = [set() for _ in range(size)]
        for r, c in zip(rows, cols):
            row_cols[r].add(c)
            col_rows[c].add(r)
        colors = [-1] * size
        for col in sorted(range(size), key=lambda c: len(col_rows[c]), reverse=True):
            forbidden = {colors[c] for r in col_rows[col] for c in row_cols[r] if colors[c] >= 0}
            color = 0
            while color in forbidden:
                color += 1
            colors[col] = color
        self.rows = list(rows)
        self.cols = list(cols)
        self.color = torch.tensor(colors, dtype=torch.long, device=self.device)
        self.color_count = max(colors) + 1
        self.r = torch.tensor(rows, dtype=torch.long, device=self.device)
        self.c = torch.tensor(cols, dtype=torch.long, device=self.device)
        # Compile sparse extraction once; no color-by-size seed or compressed
        # derivative tensor survives between calls (or across all colors).
        packs = {}
        for entry, (r, c) in enumerate(zip(rows, cols)):
            start = colors[c] // chunk_size * chunk_size
            packs.setdefault(start, []).append((colors[c] - start, r, entry))
        self.packs = tuple(
            (
                torch.arange(start, min(start + chunk_size, self.color_count), device=self.device),
                *(
                    torch.tensor([e[j] for e in entries], dtype=torch.long, device=self.device)
                    for j in range(2)
                ),
            )
            for start, entries in sorted(packs.items())
        )
        ordering = [e[2] for _start, entries in sorted(packs.items()) for e in entries]
        inverse = [0] * len(rows)
        for position, entry in enumerate(ordering):
            inverse[entry] = position
        self.extraction_order = torch.tensor(inverse, dtype=torch.long, device=self.device)

    def values(self, function, x):
        values = []
        for directions, local_color, row in self.packs:
            seeds = (directions[:, None] == self.color[None, :]).to(x.dtype)
            compressed = torch.vmap(lambda seed: torch.func.jvp(function, (x,), (seed,))[1])(seeds)
            values.append(compressed[local_color, row])
        if not values:
            return x.new_empty(0)
        return torch.cat(values).index_select(0, self.extraction_order)


class ReverseColoredJacobian:
    """Exact sparse VJP recovery using the existing column-color planner on J.T.

    Rows with disjoint column support share one reverse direction. One primal
    tape is built per values() call; bounded batches reuse it without replaying
    the physical residual. Coordinates retain the caller's order, including
    duplicates. The supplied pattern must cover every structural dependency.
    """

    def __init__(self, size, rows, cols, device, **kwargs):
        self.plan = ColoredJacobian(size, cols, rows, device, **kwargs)
        self.color_count = self.plan.color_count

    def values(self, function, x):
        plan = self.plan
        if not plan.packs:
            return x.new_empty(0)
        _, pullback = torch.func.vjp(function, x)
        values = []
        for directions, local_color, column in plan.packs:
            seeds = (directions[:, None] == plan.color[None, :]).to(x.dtype)
            compressed = torch.vmap(lambda seed: pullback(seed)[0])(seeds)
            values.append(compressed[local_color, column])
        return torch.cat(values).index_select(0, plan.extraction_order)

    def dense(self, function, x):
        values = self.values(function, x)
        result = x.new_zeros((self.plan.size, self.plan.size))
        # A duplicate coordinate describes one derivative, not additive terms.
        result[self.plan.c, self.plan.r] = values
        return result


def sparse_jacobian_plan(size, rows, cols, device, *, reverse=False, **kwargs):
    """Select compressed AD orientation from exact structural support.

    A dense balance row makes every forward column conflict, even if the
    remaining network is local. Its width is a lower bound on forward colors:
    a cheaper reverse plan can then be selected without building the expensive
    forward conflict graph at all. No numerical entry is dropped or thresholded.
    """
    rows, cols = list(rows), list(cols)
    if not reverse:
        return ColoredJacobian(size, rows, cols, device, **kwargs)
    widths, heights = {}, {}
    for row, col in set(zip(rows, cols)):
        widths[row] = widths.get(row, 0) + 1
        heights[col] = heights.get(col, 0) + 1
    forward_bound = max(widths.values(), default=0)
    reverse_bound = max(heights.values(), default=0)
    # Try the less conflicted orientation first. A global sensor column is
    # the transpose of a global balance row: do not merely move the quadratic
    # planning work from one of these topologies to the other.
    forward = None
    if reverse_bound > forward_bound:
        forward = ColoredJacobian(size, rows, cols, device, **kwargs)
        if forward.color_count < 0.75 * reverse_bound:
            return forward
    candidate = ReverseColoredJacobian(size, rows, cols, device, **kwargs)
    if candidate.color_count < 0.75 * forward_bound:
        return candidate
    if forward is None:
        forward = ColoredJacobian(size, rows, cols, device, **kwargs)
    return candidate if candidate.color_count < 0.75 * forward.color_count else forward


class TorchNewton:
    def __init__(self, device="cpu", maximum_iterations=16, dense_limit=128):
        self.device = checked_device(device)
        self.maximum_iterations = maximum_iterations
        self.dense_limit = dense_limit
        self.coloring = None
        self.matrix = None
        self.block_preconditioner = None
        self.host_convergence = False
        self.status_reads = 0
        self.line_scales = torch.tensor(
            (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125),
            dtype=torch.float64,
            device=self.device,
        )

    def solve(
        self,
        function,
        x,
        lower,
        upper,
        pattern=None,
        tolerance=2e-11,
        *,
        blocks=None,
        maximum_update=None,
        stagnation_tolerance=None,
    ):
        if stagnation_tolerance is not None and (
            not math.isfinite(stagnation_tolerance) or stagnation_tolerance < tolerance
        ):
            raise ValueError("Stagnation tolerance must be finite and no smaller than target")
        x = torch.maximum(lower, torch.minimum(upper, x)).clone()
        evaluations = 0
        best_acceptable = math.inf
        best_acceptable_x = None
        stagnant = 0
        self.stop_reason = "iteration_budget"
        for _ in range(self.maximum_iterations):
            r = function(x)
            evaluations += 1
            # CPU checks require no device synchronization. CUDA keeps a fixed
            # iteration budget; accepted status is read once at the API boundary.
            if (
                self.device.type == "cpu"
                or self.host_convergence
                or stagnation_tolerance is not None
            ):
                if self.device.type == "cuda":
                    norm = float(r.abs().max().detach().cpu())
                    self.status_reads += 1
                else:
                    norm = float(r.abs().max().detach())
                if norm <= tolerance:
                    self.stop_reason = "target_tolerance"
                    break
                if stagnation_tolerance is not None and norm < stagnation_tolerance:
                    previous_best = best_acceptable
                    if norm < best_acceptable:
                        best_acceptable = norm
                        best_acceptable_x = x.clone()
                    stagnant = 0 if norm < 0.99 * previous_best else stagnant + 1
                    if stagnant >= 4:
                        # Keep the best infinity-norm candidate, not the last
                        # rounded iterate. This only shortens iterations already
                        # inside the explicit caller's strict acceptance budget.
                        x = best_acceptable_x
                        self.stop_reason = "stagnated_within_tolerance"
                        break
                else:
                    best_acceptable = math.inf
                    best_acceptable_x = None
                    stagnant = 0
            if x.numel() <= self.dense_limit:
                J = torch.func.jacfwd(function)(x)
                step, info = torch.linalg.solve_ex(J, -r, check_errors=False)
                # A regularized least-squares direction is explicit device math,
                # never a SciPy fallback for singular active-set Jacobians.
                normal = J.T @ J + torch.eye(x.numel(), dtype=x.dtype, device=x.device) * 1e-12
                lm = torch.linalg.solve_ex(normal, -J.T @ r, check_errors=False)[0]
                step = torch.where((info == 0) & torch.isfinite(step).all(), step, lm)
            else:
                if pattern is None:
                    raise ValueError("Sparse pattern required above dense limit")
                if self.coloring is None:
                    self.coloring = ColoredJacobian(x.numel(), *pattern, self.device)
                values = self.coloring.values(function, x)
                if self.matrix is None:
                    self.matrix = TorchCOO(*pattern, values, x.numel(), self.device)
                else:
                    self.matrix.values = values.index_select(-1, self.matrix.order)
                diagonal = self.matrix.diagonal()
                preconditioner = torch.where(
                    diagonal.abs() > 1e-12, diagonal, self.matrix.row_scale()
                )
                if blocks is not None:
                    if self.block_preconditioner is None:
                        self.block_preconditioner = BlockJacobi(
                            *pattern, blocks, x.numel(), self.device
                        )
                    self.block_preconditioner.update(values)
                step = gmres_device(
                    self.matrix.mv,
                    -r,
                    preconditioner,
                    preconditioner=self.block_preconditioner if blocks is not None else None,
                )
            if maximum_update is not None:
                step = torch.maximum(-maximum_update, torch.minimum(maximum_update, step))
            best = x
            best_norm = torch.dot(r, r)
            scales = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125)
            if self.device.type == "cuda":
                # Evaluate the SAME candidates together. This removes six
                # serial streams of tiny kernel launches without weakening
                # the line search or reading a scalar on the host.
                fractions = self.line_scales
                candidates = torch.maximum(
                    lower, torch.minimum(upper, x + fractions[:, None] * step)
                )
                values = torch.vmap(function)(candidates)
                norms = values.square().sum(-1)
                norms = torch.where(torch.isfinite(norms), norms, math.inf)
                index = norms.argmin()
                take = norms.index_select(0, index.reshape(1)).squeeze(0) < best_norm
                candidate = candidates.index_select(0, index.reshape(1)).squeeze(0)
                best = torch.where(take, candidate, best)
                best_norm = torch.where(
                    take, norms.index_select(0, index.reshape(1)).squeeze(0), best_norm
                )
                evaluations += len(scales)
            else:
                for scale in scales:
                    candidate = torch.maximum(lower, torch.minimum(upper, x + scale * step))
                    value = function(candidate)
                    evaluations += 1
                    norm = torch.dot(value, value)
                    take = torch.isfinite(norm) & (norm < best_norm)
                    best = torch.where(take, candidate, best)
                    best_norm = torch.where(take, norm, best_norm)
            if self.device.type == "cpu" or self.host_convergence:
                # Six trial lengths can all overshoot a positive log-stock
                # root. Refine only on stagnation, rather than spending the
                # remaining Newton budget at an unchanged iterate.
                if self.device.type == "cuda":
                    self.status_reads += 1
                if bool((best_norm >= torch.dot(r, r)).detach().cpu()):
                    for power in range(6, 25):
                        candidate = torch.maximum(
                            lower, torch.minimum(upper, x + 2.0 ** (-power) * step)
                        )
                        value = function(candidate)
                        evaluations += 1
                        norm = torch.dot(value, value)
                        take = torch.isfinite(norm) & (norm < best_norm)
                        best = torch.where(take, candidate, best)
                        best_norm = torch.where(take, norm, best_norm)
                    if self.device.type == "cuda":
                        self.status_reads += 1
                    if bool((best_norm >= torch.dot(r, r)).detach().cpu()):
                        self.stop_reason = "line_search_stalled"
                        break
            x = torch.where((r.abs().max() <= tolerance), x, best)
        residual = function(x)
        evaluations += 1
        success = (
            torch.isfinite(x).all()
            & torch.isfinite(residual).all()
            & (residual.abs().max() <= 1e-8)
        )
        return x, residual, success, evaluations


class TorchIncidence:
    """O(E+N) local sums grouped by degree, with native forward AD support.

    Unlike CUDA index_add this forward operator needs no scatter atomics.
    Unlike a global prefix sum it never subtracts unrelated large fluxes.
    """

    def __init__(self, size, starts, ends, device):
        self.device = checked_device(device)
        adjacency = [[] for _ in range(size)]
        for edge, (start, end) in enumerate(zip(starts, ends)):
            adjacency[start].append((edge, 1.0))
            adjacency[end].append((edge, -1.0))
        by_degree = {}
        for row, edges in enumerate(adjacency):
            by_degree.setdefault(len(edges), []).append((row, edges))
        permutation = []
        self.groups = []
        for degree, rows in sorted(by_degree.items()):
            permutation.extend(row for row, _ in rows)
            indices = torch.tensor(
                [[i for i, _ in edges] for _, edges in rows], dtype=torch.long, device=self.device
            ).reshape(len(rows), degree)
            signs = torch.tensor(
                [[v for _, v in edges] for _, edges in rows],
                dtype=torch.float64,
                device=self.device,
            ).reshape(len(rows), degree)
            self.groups.append((indices, signs))
        inverse = [0] * size
        for i, row in enumerate(permutation):
            inverse[row] = i
        self.inverse = torch.tensor(inverse, dtype=torch.long, device=self.device)

    def mv(self, rates):
        pieces = [(rates[indices] * signs).sum(-1) for indices, signs in self.groups]
        return torch.cat(pieces).index_select(0, self.inverse)


def bisect_scalar(function, left, right, xtol=1e-12, rtol=1e-11, max_iterations=100):
    """Host event clock only; each function evaluation uses the Torch solver."""
    import math

    a, b = float(left), float(right)
    fa, fb = float(function(a)), float(function(b))
    if not math.isfinite(fa) or not math.isfinite(fb) or fa * fb > 0:
        raise ValueError("Event root is not bracketed")
    if fa == 0:
        return a
    if fb == 0:
        return b
    for _ in range(max_iterations):
        mid = (a + b) / 2
        fm = float(function(mid))
        if not math.isfinite(fm):
            raise RuntimeError("Nonfinite event function")
        if fm == 0 or b - a <= xtol + rtol * abs(mid):
            return mid
        if fa * fm > 0:
            a, fa = mid, fm
        else:
            b, fb = mid, fm
    raise RuntimeError("Event root iteration limit")
