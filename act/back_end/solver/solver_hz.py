from __future__ import annotations

import logging
import time

import torch
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING
from act.back_end.core import Bounds
from act.back_end.solver.solver_base import Solver, SolverCaps
from act.front_end.specs import OutKind
from act.util.stats import VerifyResult, VerifyStatus

if TYPE_CHECKING:
    from act.back_end.solver.solver_base import BatchLPProblem, BatchLPSolution
    from act.front_end.specs import OutputSpec

logger = logging.getLogger(__name__)

try:
    import numpy as np
    import scipy.sparse as sp
    from scipy.optimize import Bounds as SciPyBounds
    from scipy.optimize import LinearConstraint, milp

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


def _hz_compute_bounds_scipy(hz: HZono) -> Bounds:
    model = _lower_hz_milp(hz)
    if model.n_var == 0:
        LB = UB = model.value_center.copy()
    else:
        constraints = (
            LinearConstraint(model.A, model.row_lb, model.row_ub)
            if model.A.shape[0] else None
        )
        bounds = SciPyBounds(model.var_lb, model.var_ub)
        LB = np.empty(model.value_center.size, dtype=np.float64)
        UB = np.empty(model.value_center.size, dtype=np.float64)
        for i in range(model.value_center.size):
            obj = model.value_matrix.getrow(i).toarray().reshape(-1)
            options = {"presolve": True, "mip_rel_gap": 0.0}
            res_min = milp(
                obj,
                integrality=model.integrality,
                bounds=bounds,
                constraints=constraints,
                options=options,
            )
            res_max = milp(
                -obj,
                integrality=model.integrality,
                bounds=bounds,
                constraints=constraints,
                options=options,
            )
            if not res_min.success or not res_max.success:
                raise RuntimeError(f"HybridZ bound MILP failed at output {i}")
            LB[i] = model.value_center[i] + res_min.fun
            UB[i] = model.value_center[i] - res_max.fun

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


@dataclass(frozen=True)
class _HZMILP:
    value_center: "np.ndarray"
    value_matrix: "sp.csr_matrix"
    A: "sp.csr_matrix"
    row_lb: "np.ndarray"
    row_ub: "np.ndarray"
    var_lb: "np.ndarray"
    var_ub: "np.ndarray"
    integrality: "np.ndarray"
    n_cont: int
    n_bin: int

    @property
    def n_var(self) -> int:
        return self.n_cont + self.n_bin


@dataclass(frozen=True)
class _MILPResult:
    status: str
    x: Optional["np.ndarray"]
    nodes: int = 0


def _row_sum(mat) -> "np.ndarray":
    return np.asarray(mat.sum(axis=1), dtype=np.float64).reshape(-1)


def _lower_hz_milp(hz: "HZono | SparseHZono") -> _HZMILP:
    _require_sparse()
    if isinstance(hz, SparseHZono):
        c, Gc, Gb = hz.c, hz.Gc, hz.Gb
        n_cont, n_bin = hz.n_cont, hz.n_bin
        eq_Ac, eq_Ab, eq_b = hz.Ac, hz.Ab, hz.b
        le_Ac, le_Ab, le_b = hz.Auc, hz.Aub, hz.ub
    elif isinstance(hz, HZono):
        c = hz.c.detach().cpu().double().numpy().reshape(-1)
        Gc, Gb = _torch_to_csr(hz.Gc), _torch_to_csr(hz.Gb)
        n_cont, n_bin = Gc.shape[1], Gb.shape[1]
        Ac, Ab = _torch_to_csr(hz.Ac), _torch_to_csr(hz.Ab)
        b = hz.b.detach().cpu().double().numpy().reshape(-1)
        mask = (
            np.ones(Ac.shape[0], dtype=bool)
            if hz.eq_mask is None
            else hz.eq_mask.detach().cpu().numpy().astype(bool).reshape(-1)
        )
        if mask.size != Ac.shape[0]:
            raise ValueError("HZ eq_mask length does not match constraint rows")
        eq_Ac, eq_Ab, eq_b = Ac[mask], Ab[mask], b[mask]
        le_Ac, le_Ab, le_b = Ac[~mask], Ab[~mask], b[~mask]
    else:
        raise TypeError(f"unsupported HZ representation: {type(hz).__name__}")

    value_center = np.asarray(c, dtype=np.float64).reshape(-1) - _row_sum(Gb)
    value_matrix = sp.hstack([Gc, 2.0 * Gb], format="csr")
    blocks, lowers, uppers = [], [], []
    if eq_Ac.shape[0]:
        blocks.append(sp.hstack([eq_Ac, 2.0 * eq_Ab], format="csr"))
        rhs = np.asarray(eq_b, dtype=np.float64).reshape(-1) + _row_sum(eq_Ab)
        lowers.append(rhs)
        uppers.append(rhs)
    if le_Ac.shape[0]:
        blocks.append(sp.hstack([le_Ac, 2.0 * le_Ab], format="csr"))
        rhs = np.asarray(le_b, dtype=np.float64).reshape(-1) + _row_sum(le_Ab)
        lowers.append(np.full(rhs.size, -np.inf, dtype=np.float64))
        uppers.append(rhs)
    A = sp.vstack(blocks, format="csr") if blocks else sparse_empty(0, n_cont + n_bin)
    row_lb = np.concatenate(lowers) if lowers else np.zeros(0, dtype=np.float64)
    row_ub = np.concatenate(uppers) if uppers else np.zeros(0, dtype=np.float64)
    return _HZMILP(
        value_center=value_center,
        value_matrix=value_matrix,
        A=A,
        row_lb=row_lb,
        row_ub=row_ub,
        var_lb=np.concatenate([
            -np.ones(n_cont, dtype=np.float64),
            np.zeros(n_bin, dtype=np.float64),
        ]),
        var_ub=np.ones(n_cont + n_bin, dtype=np.float64),
        integrality=np.concatenate([
            np.zeros(n_cont, dtype=np.int32),
            np.ones(n_bin, dtype=np.int32),
        ]),
        n_cont=n_cont,
        n_bin=n_bin,
    )


def _combined_constraints(model: _HZMILP, extra_A, extra_lb, extra_ub):
    if extra_A is None or extra_A.shape[0] == 0:
        return model.A, model.row_lb, model.row_ub
    A = sp.vstack([model.A, extra_A], format="csr")
    return (
        A,
        np.concatenate([model.row_lb, np.asarray(extra_lb, dtype=np.float64)]),
        np.concatenate([model.row_ub, np.asarray(extra_ub, dtype=np.float64)]),
    )


def _valid_milp_point(
    model: _HZMILP,
    x: "np.ndarray",
    A,
    row_lb,
    row_ub,
    tol: float,
) -> bool:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size != model.n_var or not np.all(np.isfinite(x)):
        return False
    if np.any(x < model.var_lb - tol) or np.any(x > model.var_ub + tol):
        return False
    if model.n_bin:
        z = x[model.n_cont:]
        if np.any(np.abs(z - np.rint(z)) > tol):
            return False
    if A.shape[0]:
        values = np.asarray(A @ x, dtype=np.float64).reshape(-1)
        finite_lb = np.isfinite(row_lb)
        finite_ub = np.isfinite(row_ub)
        if np.any(values[finite_lb] < row_lb[finite_lb] - tol):
            return False
        if np.any(values[finite_ub] > row_ub[finite_ub] + tol):
            return False
    return True


def _solve_hz_feasibility(
    model: _HZMILP,
    deadline: float,
    *,
    extra_A=None,
    extra_lb=None,
    extra_ub=None,
    feasibility_tol: float = 1e-7,
) -> _MILPResult:
    A, row_lb, row_ub = _combined_constraints(model, extra_A, extra_lb, extra_ub)
    if model.n_var == 0:
        x = np.zeros(0, dtype=np.float64)
        status = "feasible" if _valid_milp_point(
            model, x, A, row_lb, row_ub, feasibility_tol
        ) else "infeasible"
        return _MILPResult(status, x if status == "feasible" else None)
    remaining = deadline - time.monotonic()
    if remaining <= 0.0:
        return _MILPResult("unknown", None)
    constraints = (
        LinearConstraint(A, row_lb, row_ub) if A.shape[0] else None
    )
    try:
        result = milp(
            c=np.zeros(model.n_var, dtype=np.float64),
            integrality=model.integrality,
            bounds=SciPyBounds(model.var_lb, model.var_ub),
            constraints=constraints,
            options={
                "presolve": True,
                "time_limit": max(1e-3, remaining),
                "mip_rel_gap": 0.0,
            },
        )
    except Exception as exc:
        logger.debug("HybridZ MILP failed: %s", exc)
        return _MILPResult("unknown", None)
    nodes = int(getattr(result, "mip_node_count", 0) or 0)
    x = getattr(result, "x", None)
    if x is not None and _valid_milp_point(
        model, x, A, row_lb, row_ub, feasibility_tol
    ):
        return _MILPResult("feasible", np.asarray(x, dtype=np.float64), nodes)
    if int(getattr(result, "status", -1)) == 2:
        return _MILPResult("infeasible", None, nodes)
    return _MILPResult("unknown", None, nodes)


class HZSolver(Solver):
    """Open-source Hybrid Zonotope bounds and verdict solver."""

    def __init__(self, time_limit: float = 30.0, tolerance: float = 1e-7):
        self._last_bounds: Optional[Bounds] = None
        self.time_limit = float(time_limit)
        self.tolerance = float(tolerance)
        self.last_stats: dict[str, object] = {}

    def capabilities(self) -> SolverCaps:
        return SolverCaps(supports_gpu=False, supports_csp=False, supports_hz=True)

    def compute_bounds(self, hz: HZono, *, exact: bool = False) -> Bounds:
        self._last_bounds = hz_compute_bounds(hz, exact=exact)
        return self._last_bounds

    def _unknown_results(self, batch_size: int, reason: str) -> list[VerifyResult]:
        return [
            VerifyResult(
                VerifyStatus.UNKNOWN,
                metadata={"lane": lane, "source": "hybridz", "reason": reason},
            )
            for lane in range(batch_size)
        ]

    @staticmethod
    def _recover_input(
        model: _HZMILP,
        x: "np.ndarray",
        input_hz: Optional[SparseHZono],
        input_shape: tuple[int, ...],
        lane: int,
    ) -> Optional[torch.Tensor]:
        if input_hz is None or input_hz.n_out != int(np.prod(input_shape)):
            return None
        if input_hz.n_cont > model.n_cont or input_hz.n_bin > model.n_bin:
            return None
        xi_c = x[:model.n_cont][:input_hz.n_cont]
        z = x[model.n_cont:model.n_cont + model.n_bin][:input_hz.n_bin]
        xi_b = 2.0 * z - 1.0
        value = input_hz.c.copy()
        if input_hz.n_cont:
            value += np.asarray(input_hz.Gc @ xi_c).reshape(-1)
        if input_hz.n_bin:
            value += np.asarray(input_hz.Gb @ xi_b).reshape(-1)
        full = torch.from_numpy(value.reshape(input_shape).copy())
        return full[lane].clone()

    def evaluate_spec(
        self,
        output_hz: "HZono | SparseHZono | None",
        out_spec: "OutputSpec",
        *,
        batch_size: int,
        n_out: int,
        input_hz: Optional[SparseHZono] = None,
        input_shape: Optional[tuple[int, ...]] = None,
        timelimit: Optional[float] = None,
    ) -> list[VerifyResult]:
        """Decide an output specification over a propagated Hybrid Zonotope."""
        B = int(batch_size)
        if output_hz is None:
            return self._unknown_results(B, "missing_hz_state")
        if not _HAS_SCIPY:
            return self._unknown_results(B, "scipy_unavailable")
        try:
            model = _lower_hz_milp(output_hz)
        except Exception as exc:
            return self._unknown_results(B, f"lowering_failed:{type(exc).__name__}")
        if model.value_center.size != B * int(n_out):
            return self._unknown_results(B, "output_shape_mismatch")

        encoded = out_spec.encode_linear(
            B=B,
            n_out=int(n_out),
            device=torch.device("cpu"),
            dtype=torch.float64,
        )
        C = encoded["C"].detach().cpu().double().numpy()
        thresholds = encoded["thresholds"].detach().cpu().double().numpy()
        M = int(encoded["M"])
        is_unsafe_linear = encoded["kind"] == OutKind.UNSAFE_LINEAR
        started = time.monotonic()
        deadline = started + float(
            self.time_limit if timelimit is None else timelimit
        )
        solves = 0
        nodes = 0

        def solve(extra_A=None, extra_lb=None, extra_ub=None) -> _MILPResult:
            nonlocal solves, nodes
            result = _solve_hz_feasibility(
                model,
                deadline,
                extra_A=extra_A,
                extra_lb=extra_lb,
                extra_ub=extra_ub,
                feasibility_tol=max(self.tolerance, 1e-7),
            )
            solves += 1
            nodes += result.nodes
            return result

        base = solve()
        if base.status != "feasible":
            reason = "empty_hz" if base.status == "infeasible" else "base_unknown"
            return self._unknown_results(B, reason)

        exact_witness = (
            isinstance(output_hz, SparseHZono)
            and input_hz is not None
            and output_hz.frame_id is not None
            and output_hz.frame_id == input_hz.frame_id
            and input_shape is not None
        )
        representation = "sparse" if isinstance(output_hz, SparseHZono) else "dense"
        results: list[VerifyResult] = []

        def metadata(lane: int, reason: str) -> dict[str, object]:
            return {
                "lane": lane,
                "source": "hybridz_milp",
                "representation": representation,
                "reason": reason,
            }

        for lane in range(B):
            start, stop = lane * n_out, (lane + 1) * n_out
            C_lane = C[lane * M:(lane + 1) * M]
            t_lane = thresholds[lane]
            value_matrix = model.value_matrix[start:stop]
            coeff = (sp.csr_matrix(C_lane) @ value_matrix).tocsr()
            const = C_lane @ model.value_center[start:stop]
            lane_result: Optional[VerifyResult] = None

            if is_unsafe_linear:
                expanded = solve(
                    coeff,
                    np.full(M, -np.inf, dtype=np.float64),
                    t_lane + self.tolerance - const,
                )
                if expanded.status == "infeasible":
                    lane_result = VerifyResult(
                        VerifyStatus.CERTIFIED,
                        metadata=metadata(lane, "expanded_unsafe_infeasible"),
                    )
                elif expanded.status == "feasible" and exact_witness:
                    values = const + np.asarray(coeff @ expanded.x).reshape(-1)
                    witness = expanded.x if np.all(values <= t_lane - self.tolerance) else None
                    if witness is None:
                        contracted = solve(
                            coeff,
                            np.full(M, -np.inf, dtype=np.float64),
                            t_lane - self.tolerance - const,
                        )
                        if contracted.status == "feasible":
                            values = const + np.asarray(coeff @ contracted.x).reshape(-1)
                            if np.all(values <= t_lane - self.tolerance):
                                witness = contracted.x
                    if witness is not None:
                        counterexample = self._recover_input(
                            model, witness, input_hz, input_shape, lane
                        )
                        if counterexample is not None:
                            lane_result = VerifyResult(
                                VerifyStatus.FALSIFIED,
                                counterexample=counterexample,
                                metadata=metadata(lane, "exact_unsafe_witness"),
                            )
                if lane_result is None:
                    lane_result = VerifyResult(
                        VerifyStatus.UNKNOWN,
                        metadata=metadata(lane, "unsafe_region_undecided"),
                    )
            else:
                undecided = False
                for row in range(M):
                    expanded = solve(
                        coeff[row],
                        np.array([t_lane[row] - self.tolerance - const[row]]),
                        np.array([np.inf]),
                    )
                    if expanded.status == "infeasible":
                        continue
                    if not exact_witness:
                        undecided = True
                        break
                    witness = None
                    if expanded.status == "feasible":
                        value = const[row] + float((coeff[row] @ expanded.x).item())
                        if value >= t_lane[row] + self.tolerance:
                            witness = expanded.x
                        else:
                            contracted = solve(
                                coeff[row],
                                np.array([t_lane[row] + self.tolerance - const[row]]),
                                np.array([np.inf]),
                            )
                            if contracted.status == "feasible":
                                value = const[row] + float(
                                    (coeff[row] @ contracted.x).item()
                                )
                                if value >= t_lane[row] + self.tolerance:
                                    witness = contracted.x
                    if witness is not None:
                        counterexample = self._recover_input(
                            model, witness, input_hz, input_shape, lane
                        )
                        if counterexample is not None:
                            lane_result = VerifyResult(
                                VerifyStatus.FALSIFIED,
                                counterexample=counterexample,
                                metadata=metadata(lane, "exact_violation_witness"),
                            )
                            break
                    undecided = True
                    if time.monotonic() >= deadline:
                        break
                if lane_result is None:
                    lane_result = VerifyResult(
                        VerifyStatus.UNKNOWN if undecided else VerifyStatus.CERTIFIED,
                        metadata=metadata(
                            lane,
                            "violation_region_undecided" if undecided
                            else "expanded_violations_infeasible",
                        ),
                    )
            results.append(lane_result)

        self.last_stats = {
            "elapsed": time.monotonic() - started,
            "solves": solves,
            "nodes": nodes,
            "n_cont": model.n_cont,
            "n_bin": model.n_bin,
            "n_rows": int(model.A.shape[0]),
            "representation": representation,
        }
        for result in results:
            result.metadata.update(self.last_stats)
        return results

    def solve_batch(
        self,
        problem: "BatchLPProblem",
        timelimit: Optional[float] = None,
    ) -> "BatchLPSolution":
        """HZSolver does not accept BatchLPProblem inputs.

        HZSolver operates on HZono domains via compute_bounds() and
        evaluate_spec(), not on LP/CSP batch problems. Callers that
        need batch LP solving should use TorchLPSolver or GurobiSolver.
        """
        raise NotImplementedError(
            "HZSolver does not accept BatchLPProblem; use evaluate_spec()."
        )
