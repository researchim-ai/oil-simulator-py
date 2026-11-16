"""Bound the cross-well derivative product while retaining the current dense B/D.

This does not remove the output matrices or the well-count admission limit.
The large path never forms a wells x all-perforations x components x variables
product or a second all-perforations rate-control B matrix.
"""

import torch


def assemble_surface_blocks(
    K,
    conn_well,
    dq,
    Cv,
    control_weights,
    rate_mask,
    surface_mask,
    *,
    maximum_product_entries=1_048_576,
    identity=None,
):
    nw, nconn, nb = K.shape[0], dq.shape[0], dq.shape[-1]
    if type(maximum_product_entries) is not int or maximum_product_entries < nw * 3 * nb:
        raise ValueError("Surface product budget must fit one connection")
    eye = torch.eye(nw, dtype=dq.dtype, device=dq.device) if identity is None else identity
    columns = torch.arange(nconn, device=dq.device)
    if nw * nconn * 3 * nb <= maximum_product_entries:
        B_rate = (control_weights[:, :, None] * dq).sum(1)
        Drate = torch.diag(dq.new_zeros(nw).index_add_(0, conn_well, (control_weights * Cv).sum(1)))
        gradient = K[:, conn_well, :]
        Bnet = -(gradient[:, :, :, None] * dq[None, :, :, :]).sum(2)
        Dnet = eye.clone().index_add_(1, conn_well, -(gradient * Cv[None, :, :]).sum(2))
        Brate = torch.zeros_like(Bnet)
        Brate[conn_well, columns] = B_rate
        B = torch.where(
            rate_mask[:, None, None], Brate, torch.where(surface_mask[:, None, None], Bnet, 0.0)
        )
    else:
        chunk = maximum_product_entries // (nw * 3 * nb)
        B = dq.new_zeros(nw, nconn, nb)
        B_rate = dq.new_empty(nconn, nb)
        rate_diagonal = dq.new_zeros(nw)
        Dnet = eye.clone()
        active_surface = surface_mask & ~rate_mask
        for start in range(0, nconn, chunk):
            end = min(start + chunk, nconn)
            owners = conn_well[start:end]
            B_rate[start:end] = (control_weights[start:end, :, None] * dq[start:end]).sum(1)
            rate_diagonal.index_add_(0, owners, (control_weights[start:end] * Cv[start:end]).sum(1))
            gradient = K[:, owners, :]
            local = -(gradient[:, :, :, None] * dq[None, start:end, :, :]).sum(2)
            B[:, start:end] = torch.where(active_surface[:, None, None], local, 0.0)
            B[owners, columns[start:end]] += torch.where(
                rate_mask[owners, None], B_rate[start:end], 0.0
            )
            Dnet.index_add_(1, owners, -(gradient * Cv[None, start:end, :]).sum(2))
        Drate = torch.diag(rate_diagonal)
    D = torch.where(rate_mask[:, None], Drate, torch.where(surface_mask[:, None], Dnet, eye))
    return B, D, B_rate
