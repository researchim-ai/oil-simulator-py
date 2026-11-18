"""One bounded, solver-owned CPR graph snapshot, refreshed before every matrix.

No equation or convergence tolerance is cached. Structural compatibility is
checked before reuse; all numerical inputs, including coarse solves and
Chebyshev coefficients, are copied into graph-owned buffers. CPU/offloaded
hierarchies and over-budget inputs use the original CPR path.
"""
import copy
import math
import weakref

import torch

from .cuda_graphs import capture


class CPRReplayWorkspace:
    def __init__(self, budget_bytes):
        if type(budget_bytes) is not int or budget_bytes <= 0:
            raise ValueError('Positive integer CPR graph input budget required')
        self.budget_bytes = budget_bytes
        self.captures = self.refreshes = self.replays = self.fallbacks = 0
        self.reason = None
        self.busy = False
        self.reset()

    def reset(self, *, keep_pool=False):
        self.graph = self.snapshot = self.rhs = self.output = None
        if not keep_pool:
            self.pool = None
        self.buffers = {}
        self.signature = None
        self.bound = None
        self.ready = False
        self.buffer_bytes = 0
        self.owner_stream = None

    @staticmethod
    def _collect(cpr):
        from .cpr_gpu import ClassicalAMG

        amg = cpr.amg
        if (not cpr.J_vals.is_cuda or cpr._J_bsr is None
                or type(amg) is not ClassicalAMG or amg.debug_cycles):
            return None
        device = cpr.J_vals.device
        data = {'J': cpr.J_vals, 'w': cpr.w, 'inv': cpr.inv_diag, 'scale': cpr.p_scale}
        indices = set()
        def sparse(prefix, matrix):
            if matrix.device != device or matrix.layout not in (torch.sparse_csr, torch.sparse_bsr):
                raise ValueError('CPR replay requires device-local CSR/BSR storage')
            data[prefix+'.crow'], data[prefix+'.col'] = matrix.crow_indices(), matrix.col_indices()
            indices.update((prefix+'.crow', prefix+'.col'))
            if prefix != 'J':
                data[prefix+'.values'] = matrix.values()
        sparse('J', cpr._J_bsr)
        metadata = []
        for i, level in enumerate(amg.levels):
            prefix = str(i)
            if level['A'].device != device:
                return None
            sparse(prefix+'.A', level['A'])
            coarse = bool(level['is_coarsest'])
            cheb = bool(level.get('use_chebyshev', False))
            lo, hi = level.get('lambda_min'), level.get('lambda_max')
            if cheb and any(v is not None and not math.isfinite(v) for v in (lo, hi)):
                # Never turn an invalid production smoother into a different,
                # apparently successful Jacobi solve in the replay snapshot.
                return None
            valid = (cheb and lo is not None and hi is not None and
                     math.isfinite(lo) and math.isfinite(hi) and hi > max(lo, 1e-8)
                     and (hi+lo)/2 > 0 and (hi-lo)/2 > 0)
            metadata.append((level['n'], coarse, cheb, valid, str(level['A'].dtype)))
            data[prefix+'.relax'] = level['inv_relax']
            if valid:
                d, c = (hi+lo)/2, (hi-lo)/2
                data[prefix+'.cheb_inv'] = level['inv_diag_cheb']
                data[prefix+'.coeff'] = torch.tensor([1./d, (c/(2*d))**2],
                                                      dtype=level['A'].dtype, device=device)
            if coarse:
                if level['n'] <= 2000:
                    data[prefix+'.dense'] = amg._coarse_dense_matrix(level)
            else:
                sparse(prefix+'.P', level['P'])
                sparse(prefix+'.PT', amg._transpose_csr(level['P']))
        if amg.Dhalf_inv is not None:
            data['equilibration'] = amg.Dhalf_inv
        if any(t.device != device or t.layout != torch.strided for t in data.values()):
            return None
        signature = (cpr.n_cells, cpr.nb, cpr.smoother_sweeps, tuple(metadata),
                     tuple((k, tuple(t.shape), t.dtype, t.device) for k,t in sorted(data.items())))
        return signature, data, indices

    def _build_snapshot(self, cpr):
        b = self.buffers
        snapshot = copy.copy(cpr)
        snapshot.replay_workspace = None
        snapshot._graph_pool = None
        snapshot._graph = False
        snapshot.J_vals, snapshot.w = b['J'], b['w']
        snapshot.inv_diag, snapshot.p_scale = b['inv'], b['scale']
        snapshot.rows = snapshot.col = None
        snapshot._J_bsr = torch.sparse_bsr_tensor(b['J.crow'], b['J.col'], b['J'],
                                                 size=cpr._J_bsr.shape)
        amg = copy.copy(cpr.amg)
        amg.levels = []
        amg._row_idx_cache = {}
        amg._transpose_csr_cache = {}
        amg.Dhalf_inv = b.get('equilibration')
        transposes = {}
        def csr(prefix, shape):
            return torch.sparse_csr_tensor(b[prefix+'.crow'], b[prefix+'.col'],
                                           b[prefix+'.values'], size=shape)
        for i, source in enumerate(cpr.amg.levels):
            prefix = str(i)
            level = {k:source[k] for k in ('n', 'is_coarsest')}
            level['A'] = csr(prefix+'.A', source['A'].shape)
            level['device'], level['dtype'] = source['A'].device, source['A'].dtype
            level['inv_relax'] = b[prefix+'.relax']
            # Invalid Chebyshev coefficients take the same Jacobi fallback.
            level['use_chebyshev'] = prefix+'.coeff' in b
            if level['use_chebyshev']:
                level['lambda_min'], level['lambda_max'] = source['lambda_min'], source['lambda_max']
                level['inv_diag_cheb'] = b[prefix+'.cheb_inv']
                level['_graph_chebyshev_coefficients'] = b[prefix+'.coeff']
            if prefix+'.dense' in b:
                level['_dense'], level['_dense_of'] = b[prefix+'.dense'], level['A']
                level['_dense_version'] = level['A']._version
            if not level['is_coarsest']:
                level['P'] = csr(prefix+'.P', source['P'].shape)
                transposes[id(level['P'])] = csr(prefix+'.PT', tuple(reversed(source['P'].shape)))
            amg.levels.append(level)
        # Fixed transpose buffers are updated explicitly, never lazily replaced.
        amg._transpose_csr = lambda matrix: transposes[id(matrix)]
        snapshot.amg = amg
        return snapshot

    def _prepare(self, cpr, rhs):
        collected = self._collect(cpr)
        if collected is None:
            self.ready = False
            self.reason = 'unsupported_hierarchy'
            self.fallbacks += 1
            self.bound = weakref.ref(cpr)
            return
        signature, data, indices = collected
        size = sum(t.numel()*t.element_size() for t in data.values()) + rhs.numel()*rhs.element_size()
        if size > self.budget_bytes:
            self.ready = False
            self.reason = 'input_budget'
            self.fallbacks += 1
            self.bound = weakref.ref(cpr)
            return
        compatible = signature == self.signature
        if compatible:
            # One host decision for all sparse structures. Values are NOT keys.
            checks = [torch.all(self.buffers[k] == data[k]) for k in sorted(indices)]
            compatible = bool(torch.stack(checks).all())
        if compatible:
            for key, value in data.items():
                if key not in indices:
                    self.buffers[key].copy_(value)
            # Dense coarse storage is already refreshed; preserve its identity.
            for level in self.snapshot.amg.levels:
                if '_dense' in level:
                    level['_dense_version'] = level['A']._version
            self.refreshes += 1
        else:
            # Retain one stream and registered graph pool across structural
            # changes. A new pool/stream on every capture strands allocator
            # reservations on long runs with changing AMG sparsity.
            self.reset(keep_pool=True)
            self.buffers = {k:t.detach().clone(memory_format=torch.contiguous_format) for k,t in data.items()}
            self.signature = signature
            self.buffer_bytes = size
            self.rhs = rhs.clone()
            self.snapshot = self._build_snapshot(cpr)
            self.snapshot._matvec_eager(self.rhs)  # warm libraries outside capture
            cpr.graph_eager_calls += 1
            if self.pool is None:
                self.pool = {'handle':torch.cuda.graph_pool_handle(),
                             'stream':torch.cuda.Stream(device=rhs.device), 'anchor':None}
            self.graph, self.output = capture(lambda: self.snapshot._matvec_eager(self.rhs), rhs.device,
                                              pool=self.pool['handle'], stream=self.pool['stream'])
            if self.pool['anchor'] is None:
                # Keep the pool registered between graph lifetimes. This first
                # graph is never replayed after replacement.
                self.pool['anchor'] = self.graph
            self.owner_stream = (rhs.device, torch.cuda.current_stream(rhs.device).cuda_stream)
            self.captures += 1
            cpr.graph_captures += 1
        self.ready = True
        self.reason = None
        self.bound = weakref.ref(cpr)

    def apply(self, cpr, rhs):
        if (self.busy or not rhs.is_cuda or rhs.device != cpr.J_vals.device or rhs.dtype != cpr.J_vals.dtype
                or tuple(rhs.shape) != (cpr.n_cells*cpr.nb,)):
            return None
        stream = (rhs.device, torch.cuda.current_stream(rhs.device).cuda_stream)
        if self.owner_stream is not None and self.owner_stream != stream:
            return None
        self.busy = True
        try:
            if self.bound is None or self.bound() is not cpr:
                self._prepare(cpr, rhs)
            if not self.ready:
                return None
            self.rhs.copy_(rhs)
            self.graph.replay()
            self.replays += 1
            cpr.graph_replays += 1
            return self.output.clone()
        finally:
            self.busy = False

    def reject(self, cpr, error):
        self.reset()
        self.fallbacks += 1
        self.reason = type(error).__name__ + ': ' + str(error).splitlines()[0][:160]
        self.bound = weakref.ref(cpr)

    def report(self):
        return {'captures':self.captures, 'refreshes':self.refreshes, 'replays':self.replays,
                'fallbacks':self.fallbacks, 'input_bytes':self.buffer_bytes,
                'input_budget_bytes':self.budget_bytes, 'reason':self.reason}
