from __future__ import annotations

import logging

import torch
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING
from act.back_end.core import Bounds
from act.back_end.solver.solver_base import Solver, SolverCaps

if TYPE_CHECKING:
    from act.back_end.solver.solver_base import BatchLPProblem, BatchLPSolution

logger = logging.getLogger(__name__)

try:
    from act.back_end.solver.solver_gurobi import GurobiSolver, is_gurobi_available

    _HAS_GUROBI = is_gurobi_available()
except ImportError:
    _HAS_GUROBI = False

try:
    import numpy as np
    import scipy.sparse as sp
    from scipy.optimize import linprog

    _HAS_SCIPY = True
except ImportError:
    np = None
    sp = None
    _HAS_SCIPY = False


# ============================================================================
# 1. HZono dataclass
# ============================================================================


@dataclass
class HZono:
    """Z = {c + Gc @ xi_c + Gb @ xi_b | Ac @ xi_c + Ab @ xi_b = b,
    xi_c in [-1,1]^ng, xi_b in {-1,1}^nb}"""

    c: torch.Tensor  # (n, 1)
    Gc: torch.Tensor  # (n, ng)
    Gb: torch.Tensor  # (n, nb)
    Ac: torch.Tensor  # (nc, ng)
    Ab: torch.Tensor  # (nc, nb)
    b: torch.Tensor  # (nc, 1)
    eq_mask: Optional[torch.Tensor] = None
    col_ids: Optional[torch.Tensor] = None
    bcol_ids: Optional[torch.Tensor] = None


@dataclass
class SparseHZono:
    c: "np.ndarray"
    Gc: "sp.csr_matrix"
    Gb: "sp.csr_matrix"
    Ac: "sp.csr_matrix"
    Ab: "sp.csr_matrix"
    b: "np.ndarray"
    Auc: Optional["sp.csr_matrix"] = None
    Aub: Optional["sp.csr_matrix"] = None
    ub: Optional["np.ndarray"] = None
    frame_id: Optional[int] = None

    def __post_init__(self) -> None:
        _require_sparse()
        self.c = np.asarray(self.c, dtype=np.float64).reshape(-1)
        self.b = np.asarray(self.b, dtype=np.float64).reshape(-1)
        self.Gc = _as_csr(self.Gc)
        self.Gb = _as_csr(self.Gb)
        self.Ac = _as_csr(self.Ac)
        self.Ab = _as_csr(self.Ab)

        n_out = int(self.c.size)
        n_cont = int(self.Gc.shape[1])
        n_bin = int(self.Gb.shape[1])
        if self.Gc.shape[0] != n_out or self.Gb.shape[0] != n_out:
            raise ValueError(
                "SparseHZono value shape mismatch: "
                f"c={n_out}, Gc={self.Gc.shape}, Gb={self.Gb.shape}"
            )
        if self.Ac.shape[1] != n_cont or self.Ab.shape[1] != n_bin:
            raise ValueError(
                "SparseHZono equality column mismatch: "
                f"Gc_cols={n_cont}, Gb_cols={n_bin}, Ac={self.Ac.shape}, Ab={self.Ab.shape}"
            )
        if self.Ac.shape[0] != self.Ab.shape[0] or self.Ac.shape[0] != self.b.size:
            raise ValueError(
                "SparseHZono equality row mismatch: "
                f"Ac={self.Ac.shape}, Ab={self.Ab.shape}, b={self.b.size}"
            )

        if self.Auc is None and self.Aub is None and self.ub is None:
            self.Auc = sparse_empty(0, n_cont)
            self.Aub = sparse_empty(0, n_bin)
            self.ub = np.zeros(0, dtype=np.float64)
        elif self.Auc is None or self.Aub is None or self.ub is None:
            raise ValueError("upper constraints require Auc, Aub, and ub together")
        else:
            self.Auc = _as_csr(self.Auc, shape=(self.Auc.shape[0], n_cont))
            self.Aub = _as_csr(self.Aub, shape=(self.Aub.shape[0], n_bin))
            self.ub = np.asarray(self.ub, dtype=np.float64).reshape(-1)
            if self.Auc.shape[0] != self.Aub.shape[0] or self.Auc.shape[0] != self.ub.size:
                raise ValueError(
                    "SparseHZono upper row mismatch: "
                    f"Auc={self.Auc.shape}, Aub={self.Aub.shape}, ub={self.ub.size}"
                )

    @property
    def n_out(self) -> int:
        return int(self.c.size)

    @property
    def n_cont(self) -> int:
        return int(self.Gc.shape[1])

    @property
    def n_bin(self) -> int:
        return int(self.Gb.shape[1])

    @property
    def n_eq(self) -> int:
        return int(self.Ac.shape[0])

    @property
    def n_ineq(self) -> int:
        return int(self.Auc.shape[0])


_NEXT_COL_ID = [-1]


def hz_fresh_col_ids(k: int, device=None) -> torch.Tensor:
    k = int(k)
    start = _NEXT_COL_ID[0]
    _NEXT_COL_ID[0] = start - k
    return torch.arange(start, start - k, -1, dtype=torch.long, device=device)


# ============================================================================
# 2. Algebraic operations
# ============================================================================


def hz_multiply(hz: HZono, R: torch.Tensor) -> HZono:
    R = R.to(dtype=hz.c.dtype, device=hz.c.device)
    return HZono(
        c=R @ hz.c,
        Gc=R @ hz.Gc,
        Gb=R @ hz.Gb,
        Ac=hz.Ac.clone(),
        Ab=hz.Ab.clone(),
        b=hz.b.clone(),
        eq_mask=_clone_ids(hz.eq_mask),
        col_ids=_clone_ids(hz.col_ids),
        bcol_ids=_clone_ids(hz.bcol_ids),
    )


def hz_add_const(hz: HZono, v: torch.Tensor) -> HZono:
    v = v.to(dtype=hz.c.dtype, device=hz.c.device)
    if v.ndim == 1:
        v = v.view(-1, 1)
    return HZono(
        c=hz.c + v,
        Gc=hz.Gc.clone(),
        Gb=hz.Gb.clone(),
        Ac=hz.Ac.clone(),
        Ab=hz.Ab.clone(),
        b=hz.b.clone(),
        eq_mask=_clone_ids(hz.eq_mask),
        col_ids=_clone_ids(hz.col_ids),
        bcol_ids=_clone_ids(hz.bcol_ids),
    )


def hz_minkowski_sum(hz1: HZono, hz2: HZono) -> HZono:
    dtype, device = hz1.c.dtype, hz1.c.device

    new_c = hz1.c + hz2.c.to(dtype=dtype, device=device)
    new_Gc = torch.cat([hz1.Gc, hz2.Gc.to(dtype=dtype, device=device)], dim=1)
    new_Gb = torch.cat([hz1.Gb, hz2.Gb.to(dtype=dtype, device=device)], dim=1)

    nc1, nc2 = hz1.Ac.shape[0], hz2.Ac.shape[0]
    ng1, ng2 = hz1.Gc.shape[1], hz2.Gc.shape[1]
    nb1, nb2 = hz1.Gb.shape[1], hz2.Gb.shape[1]

    Ac_top = torch.cat(
        [hz1.Ac, torch.zeros((nc1, ng2), dtype=dtype, device=device)], dim=1
    )
    Ac_bot = torch.cat(
        [
            torch.zeros((nc2, ng1), dtype=dtype, device=device),
            hz2.Ac.to(dtype=dtype, device=device),
        ],
        dim=1,
    )
    new_Ac = torch.cat([Ac_top, Ac_bot], dim=0)

    Ab_top = torch.cat(
        [hz1.Ab, torch.zeros((nc1, nb2), dtype=dtype, device=device)], dim=1
    )
    Ab_bot = torch.cat(
        [
            torch.zeros((nc2, nb1), dtype=dtype, device=device),
            hz2.Ab.to(dtype=dtype, device=device),
        ],
        dim=1,
    )
    new_Ab = torch.cat([Ab_top, Ab_bot], dim=0)

    new_b = torch.cat([hz1.b, hz2.b.to(dtype=dtype, device=device)], dim=0)
    if hz1.eq_mask is None and hz2.eq_mask is None:
        new_eq_mask = None
    else:
        m1 = hz1.eq_mask if hz1.eq_mask is not None else torch.ones(
            nc1, dtype=torch.bool, device=device
        )
        m2 = hz2.eq_mask if hz2.eq_mask is not None else torch.ones(
            nc2, dtype=torch.bool, device=device
        )
        new_eq_mask = torch.cat([m1.to(device), m2.to(device)], dim=0)
    new_col_ids = None
    if hz1.col_ids is not None and hz2.col_ids is not None:
        new_col_ids = torch.cat([hz1.col_ids.to(device), hz2.col_ids.to(device)])
    new_bcol_ids = None
    if hz1.bcol_ids is not None and hz2.bcol_ids is not None:
        new_bcol_ids = torch.cat([hz1.bcol_ids.to(device), hz2.bcol_ids.to(device)])
    return HZono(
        c=new_c,
        Gc=new_Gc,
        Gb=new_Gb,
        Ac=new_Ac,
        Ab=new_Ab,
        b=new_b,
        eq_mask=new_eq_mask,
        col_ids=new_col_ids,
        bcol_ids=new_bcol_ids,
    )


def hz_from_bounds(
    bounds: Bounds,
    dtype,
    device,
    *,
    track_ids: bool = False,
    col_ids: Optional[torch.Tensor] = None,
) -> HZono:
    """Convert an interval box to an HZ, optionally seeding generator ids."""
    lb = bounds.lb.flatten().to(dtype=dtype, device=device)
    ub = bounds.ub.flatten().to(dtype=dtype, device=device)
    n = lb.shape[0]
    c = ((lb + ub) / 2.0).view(-1, 1)
    rad = (ub - lb) / 2.0
    nz = rad > 0
    ng = int(nz.sum().item())
    idx = torch.where(nz)[0]
    Gc = torch.zeros((n, ng), dtype=dtype, device=device)
    if ng:
        Gc[idx, torch.arange(ng, device=device)] = rad[idx]
    ids = None
    if col_ids is not None:
        full_ids = col_ids.to(device=device)
        if full_ids.numel() == n:
            ids = full_ids[idx]
        elif full_ids.numel() == ng:
            ids = full_ids
        else:
            ids = None
    elif track_ids:
        full_ids = hz_fresh_col_ids(n, device=device)
        ids = full_ids[idx]
    hz = HZono(
        c=c,
        Gc=Gc,
        Gb=torch.zeros((n, 0), dtype=dtype, device=device),
        Ac=torch.zeros((0, ng), dtype=dtype, device=device),
        Ab=torch.zeros((0, 0), dtype=dtype, device=device),
        b=torch.zeros((0, 1), dtype=dtype, device=device),
        col_ids=ids,
        bcol_ids=torch.zeros(0, dtype=torch.long, device=device)
        if ids is not None
        else None,
    )
    if col_ids is not None and col_ids.numel() == n:
        hz.full_col_ids = col_ids.to(device=device)
    elif track_ids:
        hz.full_col_ids = full_ids
    return hz


def _require_sparse() -> None:
    if not _HAS_SCIPY:
        raise RuntimeError("Sparse HybridZ requires scipy")


def _as_csr(mat, *, shape=None):
    _require_sparse()
    out = mat if sp.issparse(mat) else sp.csr_matrix(mat, dtype=np.float64)
    out = out.tocsr().astype(np.float64)
    if shape is not None and out.shape != shape:
        if out.shape[0] != shape[0] or out.shape[1] > shape[1]:
            raise ValueError(f"CSR shape mismatch: {out.shape} vs {shape}")
        out = sp.hstack(
            [out, sp.csr_matrix((out.shape[0], shape[1] - out.shape[1]))],
            format="csr",
        )
    out.eliminate_zeros()
    return out


def _torch_to_csr(t: torch.Tensor):
    arr = t.detach().cpu().numpy().astype(np.float64)
    return sp.csr_matrix(arr)


def sparse_empty(rows: int, cols: int):
    _require_sparse()
    return sp.csr_matrix((int(rows), int(cols)), dtype=np.float64)


def sparse_pad_cols(mat, cols: int):
    mat = _as_csr(mat)
    cols = int(cols)
    if mat.shape[1] == cols:
        return mat
    if mat.shape[1] > cols:
        raise ValueError(f"cannot shrink sparse matrix from {mat.shape[1]} to {cols}")
    return sp.hstack([mat, sparse_empty(mat.shape[0], cols - mat.shape[1])], format="csr")


def sparse_hz_pad_frame(hz: SparseHZono, n_cont: int, n_bin: int) -> SparseHZono:
    return SparseHZono(
        c=hz.c,
        Gc=sparse_pad_cols(hz.Gc, n_cont),
        Gb=sparse_pad_cols(hz.Gb, n_bin),
        Ac=sparse_pad_cols(hz.Ac, n_cont),
        Ab=sparse_pad_cols(hz.Ab, n_bin),
        b=hz.b,
        Auc=sparse_pad_cols(hz.Auc, n_cont),
        Aub=sparse_pad_cols(hz.Aub, n_bin),
        ub=hz.ub,
        frame_id=hz.frame_id,
    )


def sparse_hz_from_bounds(
    bounds: Bounds,
    *,
    frame_id: Optional[int] = None,
    drop_zero_radius: bool = True,
) -> SparseHZono:
    _require_sparse()
    lb = bounds.lb.detach().cpu().numpy().astype(np.float64).reshape(-1)
    ub = bounds.ub.detach().cpu().numpy().astype(np.float64).reshape(-1)
    center = (lb + ub) * 0.5
    rad = (ub - lb) * 0.5
    rows = (
        np.nonzero(np.abs(rad) > 1e-12)[0].astype(np.int32)
        if drop_zero_radius
        else np.arange(rad.size, dtype=np.int32)
    )
    cols = np.arange(rows.size, dtype=np.int32)
    Gc = sp.csr_matrix(
        (rad[rows], (rows, cols)),
        shape=(rad.size, rows.size),
        dtype=np.float64,
    )
    return SparseHZono(
        c=center,
        Gc=Gc,
        Gb=sparse_empty(rad.size, 0),
        Ac=sparse_empty(0, rows.size),
        Ab=sparse_empty(0, 0),
        b=np.zeros(0, dtype=np.float64),
        Auc=sparse_empty(0, rows.size),
        Aub=sparse_empty(0, 0),
        ub=np.zeros(0, dtype=np.float64),
        frame_id=frame_id,
    )


def sparse_hz_linear(hz: SparseHZono, W, bias=None) -> SparseHZono:
    Wsp = _as_csr(W)
    if Wsp.shape[1] != hz.n_out:
        raise ValueError(f"linear shape mismatch: W={Wsp.shape}, n_out={hz.n_out}")
    b = (
        np.zeros(Wsp.shape[0], dtype=np.float64)
        if bias is None
        else np.asarray(bias, dtype=np.float64).reshape(-1)
    )
    if b.size != Wsp.shape[0]:
        raise ValueError(f"bias shape mismatch: bias={b.size}, rows={Wsp.shape[0]}")
    Gc = (Wsp @ hz.Gc).tocsr()
    Gb = (Wsp @ hz.Gb).tocsr() if hz.n_bin else sparse_empty(Wsp.shape[0], 0)
    Gc.eliminate_zeros()
    Gb.eliminate_zeros()
    return SparseHZono(
        c=np.asarray(Wsp @ hz.c).reshape(-1) + b,
        Gc=Gc,
        Gb=Gb,
        Ac=hz.Ac,
        Ab=hz.Ab,
        b=hz.b,
        Auc=hz.Auc,
        Aub=hz.Aub,
        ub=hz.ub,
        frame_id=hz.frame_id,
    )


def sparse_hz_add_const(hz: SparseHZono, bias) -> SparseHZono:
    b = np.asarray(
        bias.detach().cpu().double().numpy() if isinstance(bias, torch.Tensor) else bias,
        dtype=np.float64,
    ).reshape(-1)
    if b.size == 1:
        b = np.full(hz.n_out, float(b[0]), dtype=np.float64)
    if b.size != hz.n_out:
        raise ValueError(f"bias shape mismatch: bias={b.size}, n_out={hz.n_out}")
    return SparseHZono(
        c=hz.c + b,
        Gc=hz.Gc,
        Gb=hz.Gb,
        Ac=hz.Ac,
        Ab=hz.Ab,
        b=hz.b,
        Auc=hz.Auc,
        Aub=hz.Aub,
        ub=hz.ub,
        frame_id=hz.frame_id,
    )


def sparse_hz_scale(hz: SparseHZono, scale) -> SparseHZono:
    s = np.asarray(
        scale.detach().cpu().double().numpy() if isinstance(scale, torch.Tensor) else scale,
        dtype=np.float64,
    ).reshape(-1)
    if s.size == 1:
        s = np.full(hz.n_out, float(s[0]), dtype=np.float64)
    if s.size != hz.n_out:
        raise ValueError(f"scale shape mismatch: scale={s.size}, n_out={hz.n_out}")
    D = sp.diags(s, offsets=0, shape=(hz.n_out, hz.n_out), format="csr")
    return sparse_hz_linear(hz, D, None)


def sparse_hz_gather_rows(hz: SparseHZono, rows) -> SparseHZono:
    idx = np.asarray(rows, dtype=np.int64).reshape(-1)
    return SparseHZono(
        c=hz.c[idx],
        Gc=hz.Gc[idx].tocsr(),
        Gb=hz.Gb[idx].tocsr() if hz.n_bin else sparse_empty(idx.size, 0),
        Ac=hz.Ac,
        Ab=hz.Ab,
        b=hz.b,
        Auc=hz.Auc,
        Aub=hz.Aub,
        ub=hz.ub,
        frame_id=hz.frame_id,
    )


def sparse_hz_reduce_sum_rows(hz: SparseHZono, rows, n_out: int) -> SparseHZono:
    rows = np.asarray(rows, dtype=np.int64).reshape(-1)
    if rows.size != hz.n_out:
        raise ValueError(f"reduce rows mismatch: rows={rows.size}, n_out={hz.n_out}")
    src = np.arange(rows.size, dtype=np.int64)
    R = sp.csr_matrix(
        (np.ones(rows.size, dtype=np.float64), (rows, src)),
        shape=(int(n_out), hz.n_out),
    )
    return sparse_hz_linear(hz, R, None)


def _sparse_same_frame(parts) -> bool:
    frames = [p.frame_id for p in parts]
    return all(f is not None for f in frames) and all(f == frames[0] for f in frames)


def _sparse_vstack(mats, cols: int):
    mats = [sparse_pad_cols(m, cols) for m in mats if m.shape[0]]
    return sp.vstack(mats, format="csr") if mats else sparse_empty(0, cols)


def _sparse_concat_arrays(arrs):
    arrs = [np.asarray(a, dtype=np.float64).reshape(-1) for a in arrs if np.asarray(a).size]
    return np.concatenate(arrs) if arrs else np.zeros(0, dtype=np.float64)


def sparse_hz_concat(parts) -> SparseHZono:
    parts = list(parts)
    if not parts:
        raise ValueError("sparse_hz_concat requires at least one part")
    if not _sparse_same_frame(parts):
        raise ValueError("sparse concat requires one shared frame")
    n_cont = max(p.n_cont for p in parts)
    n_bin = max(p.n_bin for p in parts)
    padded = [sparse_hz_pad_frame(p, n_cont, n_bin) for p in parts]
    return SparseHZono(
        c=np.concatenate([p.c for p in padded]),
        Gc=sp.vstack([p.Gc for p in padded], format="csr"),
        Gb=sp.vstack([p.Gb for p in padded], format="csr"),
        Ac=_sparse_vstack([p.Ac for p in padded], n_cont),
        Ab=_sparse_vstack([p.Ab for p in padded], n_bin),
        b=_sparse_concat_arrays([p.b for p in padded]),
        Auc=_sparse_vstack([p.Auc for p in padded], n_cont),
        Aub=_sparse_vstack([p.Aub for p in padded], n_bin),
        ub=_sparse_concat_arrays([p.ub for p in padded]),
        frame_id=padded[0].frame_id,
    )


def sparse_hz_add_same_frame(x: SparseHZono, y: SparseHZono) -> SparseHZono:
    if not _sparse_same_frame([x, y]):
        raise ValueError("sparse add requires one shared frame")
    if x.n_out != y.n_out:
        raise ValueError(f"sparse add shape mismatch: {x.n_out} vs {y.n_out}")
    n_cont = max(x.n_cont, y.n_cont)
    n_bin = max(x.n_bin, y.n_bin)
    xp = sparse_hz_pad_frame(x, n_cont, n_bin)
    yp = sparse_hz_pad_frame(y, n_cont, n_bin)
    Gc = (xp.Gc + yp.Gc).tocsr()
    Gb = (xp.Gb + yp.Gb).tocsr()
    Gc.eliminate_zeros()
    Gb.eliminate_zeros()
    return SparseHZono(
        c=xp.c + yp.c,
        Gc=Gc,
        Gb=Gb,
        Ac=_sparse_vstack([xp.Ac, yp.Ac], n_cont),
        Ab=_sparse_vstack([xp.Ab, yp.Ab], n_bin),
        b=_sparse_concat_arrays([xp.b, yp.b]),
        Auc=_sparse_vstack([xp.Auc, yp.Auc], n_cont),
        Aub=_sparse_vstack([xp.Aub, yp.Aub], n_bin),
        ub=_sparse_concat_arrays([xp.ub, yp.ub]),
        frame_id=xp.frame_id,
    )


def sparse_hz_sub_same_frame(x: SparseHZono, y: SparseHZono) -> SparseHZono:
    return sparse_hz_add_same_frame(x, sparse_hz_scale(y, -1.0))


def sparse_hz_is_point(hz: SparseHZono, tol: float = 1e-12) -> bool:
    return (
        (hz.Gc.nnz == 0 or bool(np.all(np.abs(hz.Gc.data) <= tol)))
        and (hz.Gb.nnz == 0 or bool(np.all(np.abs(hz.Gb.data) <= tol)))
    )


def sparse_hz_fast_bounds(hz: SparseHZono) -> Bounds:
    abs_gc = np.asarray(np.abs(hz.Gc).sum(axis=1)).reshape(-1)
    abs_gb = np.asarray(np.abs(hz.Gb).sum(axis=1)).reshape(-1) if hz.n_bin else 0.0
    rad = abs_gc + abs_gb
    return Bounds(
        lb=torch.from_numpy(hz.c - rad).reshape(1, -1),
        ub=torch.from_numpy(hz.c + rad).reshape(1, -1),
    )


def _clone_ids(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    return None if t is None else t.clone()


def _align(
    ids_x: torch.Tensor,
    ids_y: torch.Tensor,
    Gx: torch.Tensor,
    Gy: torch.Tensor,
):
    """Merge two generator matrices by column id, preserving shared factors."""
    n = Gx.shape[0]
    dtype, device = Gx.dtype, Gx.device
    pos: dict[int, int] = {}
    merged_ids: list[int] = []
    for idv in ids_x.tolist():
        if idv not in pos:
            pos[idv] = len(merged_ids)
            merged_ids.append(idv)
    for idv in ids_y.tolist():
        if idv not in pos:
            pos[idv] = len(merged_ids)
            merged_ids.append(idv)
    x_map = torch.tensor([pos[v] for v in ids_x.tolist()], dtype=torch.long, device=device)
    y_map = torch.tensor([pos[v] for v in ids_y.tolist()], dtype=torch.long, device=device)
    G = torch.zeros(n, len(merged_ids), dtype=dtype, device=device)
    if Gx.shape[1]:
        G.index_add_(1, x_map, Gx)
    if Gy.shape[1]:
        G.index_add_(1, y_map, Gy.to(dtype=dtype, device=device))
    return G, torch.tensor(merged_ids, dtype=torch.long, device=device), x_map, y_map


def _scatter_cols(A: torch.Tensor, col_map: torch.Tensor, n_merged: int) -> torch.Tensor:
    """Lift constraints into the merged generator-column space."""
    out = A.new_zeros(A.shape[0], n_merged)
    if A.shape[1]:
        out[:, col_map] = A
    return out


def _constraint_mask(hz: HZono) -> torch.Tensor:
    if hz.eq_mask is not None:
        return hz.eq_mask
    return torch.ones(int(hz.Ac.shape[0]), dtype=torch.bool, device=hz.Ac.device)


def _shared_constraint_prefix(
    Ac_x: torch.Tensor,
    Ac_y: torch.Tensor,
    Ab_x: torch.Tensor,
    Ab_y: torch.Tensor,
    b_x: torch.Tensor,
    b_y: torch.Tensor,
    eq_x: Optional[torch.Tensor],
    eq_y: Optional[torch.Tensor],
) -> int:
    """Return the common prefix length of identical constraints."""
    m = min(int(Ac_x.shape[0]), int(Ac_y.shape[0]))
    if m == 0:
        return 0
    same = (Ac_x[:m] == Ac_y[:m]).all(dim=1)
    if Ab_x.shape[1]:
        same &= (Ab_x[:m] == Ab_y[:m]).all(dim=1)
    same &= (b_x[:m] == b_y[:m]).reshape(m, -1).all(dim=1)
    if eq_x is not None or eq_y is not None:
        ex = eq_x if eq_x is not None else torch.ones(
            int(Ac_x.shape[0]), dtype=torch.bool, device=Ac_x.device
        )
        ey = eq_y if eq_y is not None else torch.ones(
            int(Ac_y.shape[0]), dtype=torch.bool, device=Ac_x.device
        )
        same &= ex[:m].to(Ac_x.device) == ey[:m].to(Ac_x.device)
    return m if bool(same.all()) else int((~same).nonzero()[0, 0])


def hz_sgm_add(hz_x: HZono, hz_y: HZono) -> HZono:
    """Add HZs while preserving shared generator identities.

    Matching col_ids denote the same latent factor, so correlated terms can
    combine exactly instead of being duplicated as independent Minkowski terms.
    """
    if hz_x.col_ids is None or hz_y.col_ids is None:
        return hz_minkowski_sum(hz_x, hz_y)
    n = int(hz_x.c.shape[0])
    if int(hz_y.c.shape[0]) != n:
        raise ValueError(f"hz_sgm_add: shape mismatch {n} vs {hz_y.c.shape[0]}")
    dtype, device = hz_x.c.dtype, hz_x.c.device
    bx = hz_x.bcol_ids if hz_x.bcol_ids is not None else torch.zeros(
        0, dtype=torch.long, device=device
    )
    by = hz_y.bcol_ids if hz_y.bcol_ids is not None else torch.zeros(
        0, dtype=torch.long, device=device
    )
    Gc, cids, xc_map, yc_map = _align(hz_x.col_ids, hz_y.col_ids, hz_x.Gc, hz_y.Gc)
    Gb, bids, xb_map, yb_map = _align(bx, by, hz_x.Gb, hz_y.Gb)
    Ac_x = _scatter_cols(hz_x.Ac, xc_map, Gc.shape[1])
    Ac_y = _scatter_cols(hz_y.Ac.to(dtype=dtype, device=device), yc_map, Gc.shape[1])
    Ab_x = _scatter_cols(hz_x.Ab, xb_map, Gb.shape[1])
    Ab_y = _scatter_cols(hz_y.Ab.to(dtype=dtype, device=device), yb_map, Gb.shape[1])
    b_x = hz_x.b.to(dtype=dtype, device=device)
    b_y = hz_y.b.to(dtype=dtype, device=device)
    k = _shared_constraint_prefix(
        Ac_x, Ac_y, Ab_x, Ab_y, b_x, b_y, hz_x.eq_mask, hz_y.eq_mask
    )
    if hz_x.eq_mask is None and hz_y.eq_mask is None:
        eq_mask = None
    else:
        eq_mask = torch.cat(
            [_constraint_mask(hz_x).to(device), _constraint_mask(hz_y).to(device)[k:]],
            dim=0,
        )
    return HZono(
        c=hz_x.c + hz_y.c.to(dtype=dtype, device=device),
        Gc=Gc,
        Gb=Gb,
        Ac=torch.cat([Ac_x, Ac_y[k:]], dim=0),
        Ab=torch.cat([Ab_x, Ab_y[k:]], dim=0),
        b=torch.cat([b_x, b_y[k:]], dim=0),
        eq_mask=eq_mask,
        col_ids=cids,
        bcol_ids=bids,
    )


def hz_negate(hz: HZono) -> HZono:
    return HZono(
        c=-hz.c,
        Gc=-hz.Gc,
        Gb=-hz.Gb,
        Ac=hz.Ac.clone(),
        Ab=hz.Ab.clone(),
        b=hz.b.clone(),
        eq_mask=_clone_ids(hz.eq_mask),
        col_ids=_clone_ids(hz.col_ids),
        bcol_ids=_clone_ids(hz.bcol_ids),
    )


def hz_sub(hz_x: HZono, hz_y: HZono) -> HZono:
    return hz_sgm_add(hz_x, hz_negate(hz_y))


def hz_concat(parts) -> "HZono | None":
    parts = [p for p in parts if p is not None]
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    if any(p.col_ids is None for p in parts):
        return _hz_concat_independent(parts)
    dtype, device = parts[0].c.dtype, parts[0].c.device
    cpos: dict[int, int] = {}
    cids: list[int] = []
    for p in parts:
        for idv in p.col_ids.tolist():
            if idv not in cpos:
                cpos[idv] = len(cids)
                cids.append(idv)
    bpos: dict[int, int] = {}
    bids: list[int] = []
    for p in parts:
        pb = p.bcol_ids if p.bcol_ids is not None else torch.zeros(
            0, dtype=torch.long, device=device
        )
        for idv in pb.tolist():
            if idv not in bpos:
                bpos[idv] = len(bids)
                bids.append(idv)
    ngm, nbm = len(cids), len(bids)
    cs, Gcs, Gbs, Acs, Abs, bs, eqs = [], [], [], [], [], [], []
    for p in parts:
        cmap = torch.tensor([cpos[v] for v in p.col_ids.tolist()], dtype=torch.long, device=device)
        pb = p.bcol_ids if p.bcol_ids is not None else torch.zeros(
            0, dtype=torch.long, device=device
        )
        bmap = torch.tensor([bpos[v] for v in pb.tolist()], dtype=torch.long, device=device)
        Gc_p = p.c.new_zeros(p.c.shape[0], ngm)
        if p.Gc.shape[1]:
            Gc_p[:, cmap] = p.Gc.to(dtype=dtype, device=device)
        Gb_p = p.c.new_zeros(p.c.shape[0], nbm)
        if p.Gb.shape[1] and nbm:
            Gb_p[:, bmap] = p.Gb.to(dtype=dtype, device=device)
        Ac_p = p.Ac.new_zeros(p.Ac.shape[0], ngm)
        if p.Ac.shape[1]:
            Ac_p[:, cmap] = p.Ac.to(dtype=dtype, device=device)
        Ab_p = p.Ab.new_zeros(p.Ab.shape[0], nbm)
        if p.Ab.shape[1] and nbm:
            Ab_p[:, bmap] = p.Ab.to(dtype=dtype, device=device)
        cs.append(p.c.to(dtype=dtype, device=device))
        Gcs.append(Gc_p)
        Gbs.append(Gb_p)
        Acs.append(Ac_p)
        Abs.append(Ab_p)
        bs.append(p.b.to(dtype=dtype, device=device))
        eqs.append(_constraint_mask(p).to(device))
    return HZono(
        c=torch.cat(cs, 0),
        Gc=torch.cat(Gcs, 0),
        Gb=torch.cat(Gbs, 0),
        Ac=torch.cat(Acs, 0),
        Ab=torch.cat(Abs, 0),
        b=torch.cat(bs, 0),
        eq_mask=torch.cat(eqs, 0) if any(p.eq_mask is not None for p in parts) else None,
        col_ids=torch.tensor(cids, dtype=torch.long, device=device),
        bcol_ids=torch.tensor(bids, dtype=torch.long, device=device),
    )


def _hz_concat_independent(parts) -> HZono:
    dtype, device = parts[0].c.dtype, parts[0].c.device
    ng_tot = sum(int(p.Gc.shape[1]) for p in parts)
    nb_tot = sum(int(p.Gb.shape[1]) for p in parts)
    nc_tot = sum(int(p.Ac.shape[0]) for p in parts)
    Ac = torch.zeros(nc_tot, ng_tot, dtype=dtype, device=device)
    Ab = torch.zeros(nc_tot, nb_tot, dtype=dtype, device=device)
    cs, Gcs, Gbs, bs, eqs = [], [], [], [], []
    goff = boff = roff = 0
    for p in parts:
        n_p, ng_p = int(p.c.shape[0]), int(p.Gc.shape[1])
        nb_p, nc_p = int(p.Gb.shape[1]), int(p.Ac.shape[0])
        Gc_p = torch.zeros(n_p, ng_tot, dtype=dtype, device=device)
        Gc_p[:, goff:goff + ng_p] = p.Gc.to(dtype=dtype, device=device)
        Gb_p = torch.zeros(n_p, nb_tot, dtype=dtype, device=device)
        Gb_p[:, boff:boff + nb_p] = p.Gb.to(dtype=dtype, device=device)
        cs.append(p.c.to(dtype=dtype, device=device))
        Gcs.append(Gc_p)
        Gbs.append(Gb_p)
        if nc_p:
            Ac[roff:roff + nc_p, goff:goff + ng_p] = p.Ac.to(dtype=dtype, device=device)
            Ab[roff:roff + nc_p, boff:boff + nb_p] = p.Ab.to(dtype=dtype, device=device)
            bs.append(p.b.to(dtype=dtype, device=device))
            eqs.append(_constraint_mask(p).to(device))
        goff += ng_p
        boff += nb_p
        roff += nc_p
    return HZono(
        c=torch.cat(cs, 0),
        Gc=torch.cat(Gcs, 0),
        Gb=torch.cat(Gbs, 0),
        Ac=Ac,
        Ab=Ab,
        b=torch.cat(bs, 0) if bs else torch.zeros(0, 1, dtype=dtype, device=device),
        eq_mask=torch.cat(eqs, 0) if eqs else None,
    )


# ============================================================================
# 3. Bounds computation
# ============================================================================


def _hz_is_unconstrained(hz: HZono) -> bool:
    tol = 1e-12
    return (
        torch.all(torch.abs(hz.Ac) < tol).item()
        and torch.all(torch.abs(hz.Ab) < tol).item()
        and torch.all(torch.abs(hz.b) < tol).item()
    )


def _hz_bounds_unconstrained(hz: HZono) -> Bounds:
    n = hz.c.shape[0]
    dtype, device = hz.c.dtype, hz.c.device
    absGc = (
        hz.Gc.abs().sum(dim=1, keepdim=True)
        if hz.Gc.numel()
        else torch.zeros((n, 1), dtype=dtype, device=device)
    )
    absGb = (
        hz.Gb.abs().sum(dim=1, keepdim=True)
        if hz.Gb.numel()
        else torch.zeros((n, 1), dtype=dtype, device=device)
    )
    rad = absGc + absGb
    return Bounds(lb=(hz.c - rad).reshape(1, -1), ub=(hz.c + rad).reshape(1, -1))


def _hz_compute_bounds_gurobi(hz: HZono) -> Bounds:
    return GurobiSolver.compute_bounds(hz)


def _hz_has_inequality_constraints(hz: HZono) -> bool:
    return hz.eq_mask is not None and not bool(torch.all(hz.eq_mask).item())


def _hz_constraint_split(hz: HZono):
    Ac_np = hz.Ac.detach().cpu().numpy().astype("float64")
    Ab_np = hz.Ab.detach().cpu().numpy().astype("float64")
    b_np = hz.b.detach().cpu().numpy().astype("float64").reshape(-1)
    if Ac_np.shape[0] == 0:
        return None, None, None, None

    A = np.concatenate([Ac_np, Ab_np], axis=1)
    if hz.eq_mask is None:
        return A, b_np, None, None

    mask = hz.eq_mask.detach().cpu().numpy().astype(bool).reshape(-1)
    if mask.shape[0] != A.shape[0]:
        raise ValueError("HZ eq_mask length does not match constraint rows")

    A_eq = A[mask] if mask.any() else None
    b_eq = b_np[mask] if mask.any() else None
    ineq = ~mask
    A_ub = A[ineq] if ineq.any() else None
    b_ub = b_np[ineq] if ineq.any() else None
    return A_eq, b_eq, A_ub, b_ub


def _hz_compute_bounds_scipy(hz: HZono) -> Bounds:
    n = int(hz.c.shape[0])
    p = int(hz.Gc.shape[1])
    q = int(hz.Gb.shape[1])
    c_np = hz.c.detach().cpu().numpy().astype("float64").reshape(-1)
    Gc_np = hz.Gc.detach().cpu().numpy().astype("float64")
    Gb_np = hz.Gb.detach().cpu().numpy().astype("float64")
    A_eq, b_eq, A_ub, b_ub = _hz_constraint_split(hz)
    var_bounds = [(-1.0, 1.0)] * (p + q)

    LB = np.empty((n,), dtype=np.float64)
    UB = np.empty((n,), dtype=np.float64)
    for i in range(n):
        obj = np.concatenate([Gc_np[i], Gb_np[i]], axis=0)
        res_min = linprog(
            c=obj,
            A_eq=A_eq,
            b_eq=b_eq,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=var_bounds,
            method="highs",
        )
        if not res_min.success:
            raise RuntimeError(
                f"[linprog] MIN infeasible at dim {i}: {res_min.message}"
            )
        LB[i] = c_np[i] + res_min.fun
        res_max = linprog(
            c=-obj,
            A_eq=A_eq,
            b_eq=b_eq,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=var_bounds,
            method="highs",
        )
        if not res_max.success:
            raise RuntimeError(
                f"[linprog] MAX infeasible at dim {i}: {res_max.message}"
            )
        UB[i] = c_np[i] - res_max.fun

    dtype, device = hz.c.dtype, hz.c.device
    return Bounds(
        lb=torch.from_numpy(LB).to(device=device, dtype=dtype).reshape(1, -1),
        ub=torch.from_numpy(UB).to(device=device, dtype=dtype).reshape(1, -1),
    )


def hz_compute_bounds(hz: HZono, *, exact: bool = False) -> Bounds:
    """Compute box bounds from a hybrid zonotope.

    Args:
        hz: The hybrid zonotope.
        exact: If False (default), always use the fast unconstrained
            over-approximation (|Gc| + |Gb| radius). This is sound but
            may be wider than necessary.  If True, solve per-dimension
            LP/MILP to obtain tight bounds when equality constraints
            exist.  Use ``exact=True`` only at the final output layer
            where tight bounds matter for verification; intermediate
            layers benefit from the 1000×+ speed-up of the fast path
            with negligible precision loss (the full zonotope structure
            is still propagated via ``_hz_cache``).
    """
    if _hz_is_unconstrained(hz):
        return _hz_bounds_unconstrained(hz)
    if not exact:
        return _hz_bounds_unconstrained(hz)
    if _HAS_GUROBI and not _hz_has_inequality_constraints(hz):
        try:
            return _hz_compute_bounds_gurobi(hz)
        except Exception as e:
            # Intentional: Gurobi failures (license/timeout/numerical) fall back to scipy/unconstrained.
            logger.debug("suppressed: %s", e)
    if _HAS_SCIPY:
        try:
            return _hz_compute_bounds_scipy(hz)
        except Exception as e:
            # Intentional: scipy linprog failures fall back to the unconstrained bounds estimate.
            logger.debug("suppressed: %s", e)
    return _hz_bounds_unconstrained(hz)


# ============================================================================
# 4. HZSolver
# ============================================================================


class HZSolver(Solver):
    """Hybrid Zonotope bounds solver.

    Precision hierarchy:
      GurobiSolver (MILP, exact) > HZSolver (HZ, tight) > TorchLPSolver (box, fast)
    """

    def __init__(self):
        self._last_bounds: Optional[Bounds] = None

    def capabilities(self) -> SolverCaps:
        return SolverCaps(supports_gpu=False, supports_csp=False, supports_hz=True)

    def compute_bounds(self, hz: HZono, *, exact: bool = False) -> Bounds:
        self._last_bounds = hz_compute_bounds(hz, exact=exact)
        return self._last_bounds

    def solve_batch(
        self,
        problem: "BatchLPProblem",
        timelimit: Optional[float] = None,
    ) -> "BatchLPSolution":
        """HZSolver does not accept BatchLPProblem inputs.

        HZSolver operates on HZono (hybrid zonotope) domains via
        compute_bounds(), not on LP/CSP batch problems.  Callers that
        need batch LP solving should use TorchLPSolver or GurobiSolver.
        """
        raise NotImplementedError(
            "HZSolver does not solve CSPs; use compute_bounds() for HZ domain analysis."
        )
