#===- act/back_end/interval_tf/tf_mlp.py - MLP Interval Transfer Func ---====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   MLP Interval Transfer Functions. Provides interval-based transfer functions
#   for multi-layer perceptron operations including linear layers and
#   activation functions.
#
#===---------------------------------------------------------------------===#

import torch
from typing import List
from act.back_end.core import Bounds, Con, ConSet, Fact, Layer
from act.back_end.utils import affine_bounds, pwl_meta, bound_var_interval, scale_interval

# -------- MLP Basics --------
def tf_dense(L: Layer, Bin: Bounds) -> Fact:
    # Handle parameter compatibility: schema defines W, but optimization uses W_pos/W_neg
    W = L.params["W"]
    W_pos = L.params.get("W_pos", torch.clamp(W, min=0))
    W_neg = L.params.get("W_neg", torch.clamp(W, max=0))
    b = L.params.get("b", torch.zeros(W.shape[0], device=W.device, dtype=W.dtype))
    
    B = affine_bounds(W_pos, W_neg, b, Bin)
    C = ConSet(); C.replace(Con("EQ", tuple(L.out_vars + L.in_vars), {"tag": f"dense:{L.id}", "W": W, "b": b}))
    C.add_box(L.id, L.out_vars, B); return Fact(B,C)

def tf_bias(L: Layer, Bin: Bounds) -> Fact:
    c=L.params["c"]; B=Bounds(Bin.lb+c, Bin.ub+c)
    C=ConSet(); C.replace(Con("EQ", tuple(L.out_vars + L.in_vars), {"tag": f"bias:{L.id}", "c": c}))
    C.add_box(L.id, L.out_vars, B); return Fact(B,C)

def tf_scale(L: Layer, Bin: Bounds) -> Fact:
    a=L.params["a"]
    lb=torch.where(a>=0, a*Bin.lb, a*Bin.ub); ub=torch.where(a>=0, a*Bin.ub, a*Bin.lb)
    B=Bounds(lb,ub); C=ConSet(); C.replace(Con("EQ", tuple(L.out_vars + L.in_vars), {"tag": f"scale:{L.id}", "a": a}))
    C.add_box(L.id, L.out_vars, B); return Fact(B,C)

def tf_relu(L: Layer, Bin: Bounds) -> Fact:
    l,u=Bin.lb,Bin.ub; on=l>=0; off=u<=0; amb=~(on|off)
    lb=torch.where(off,0.0,torch.where(on,l,0.0)); ub=torch.where(off,0.0,torch.where(on,u,u))
    if torch.any(amb):
        s=u[amb]/torch.clamp(u[amb]-l[amb],min=1e-12); t=-s*l[amb]
    else: s=t=torch.empty(0, dtype=l.dtype, device=l.device)
    B=Bounds(lb,ub); C=ConSet()
    C.replace(Con("INEQ", tuple(L.out_vars+L.in_vars), {"tag":f"relu:{L.id}",
        "idx_on": torch.nonzero(on,as_tuple=True)[0],
        "idx_off": torch.nonzero(off,as_tuple=True)[0],
        "idx_amb": torch.nonzero(amb,as_tuple=True)[0],
        "slope": s, "shift": t}))
    C.add_box(L.id,L.out_vars,B); return Fact(B,C)

def tf_lrelu(L: Layer, Bin: Bounds) -> Fact:
    a=float(L.meta["alpha"]); l,u=Bin.lb,Bin.ub; on=l>=0; off=u<=0; amb=~(on|off)
    lb=torch.minimum(a*torch.minimum(l,0.0), torch.maximum(l,0.0))
    ub=torch.maximum(a*torch.maximum(u,0.0), torch.maximum(u,0.0))
    if torch.any(amb):
        s=(u[amb]-a*l[amb])/torch.clamp(u[amb]-l[amb],min=1e-12); t=a*l[amb]-s*l[amb]
    else: s=t=torch.empty(0, dtype=l.dtype, device=l.device)
    B=Bounds(lb,ub); C=ConSet()
    C.replace(Con("INEQ", tuple(L.out_vars+L.in_vars), {"tag":f"lrelu:{L.id}","alpha":a,
        "idx_on": torch.nonzero(on,as_tuple=True)[0],
        "idx_off": torch.nonzero(off,as_tuple=True)[0],
        "idx_amb": torch.nonzero(amb,as_tuple=True)[0],
        "slope": s, "shift": t}))
    C.add_box(L.id,L.out_vars,B); return Fact(B,C)

def tf_abs(L: Layer, Bin: Bounds) -> Fact:
    l,u=Bin.lb,Bin.ub; pos=l>=0; neg=u<=0; amb=~(pos|neg)
    lb=torch.minimum(torch.zeros_like(l), torch.minimum(torch.abs(l), torch.abs(u)))
    ub=torch.maximum(torch.abs(l), torch.abs(u))
    B=Bounds(lb,ub); C=ConSet()
    C.replace(Con("INEQ", tuple(L.out_vars+L.in_vars), {"tag":f"abs:{L.id}",
        "idx_pos": torch.nonzero(pos,as_tuple=True)[0],
        "idx_neg": torch.nonzero(neg,as_tuple=True)[0],
        "idx_amb": torch.nonzero(amb,as_tuple=True)[0]}))
    C.add_box(L.id,L.out_vars,B); return Fact(B,C)

def tf_clip(L: Layer, Bin: Bounds) -> Fact:
    a,b=L.params["a"],L.params["b"]; B=Bounds(torch.clamp(Bin.lb,a,b), torch.clamp(Bin.ub,a,b))
    C=ConSet(); C.replace(Con("INEQ", tuple(L.out_vars+L.in_vars), {"tag":f"clip:{L.id}","a":a,"b":b}))
    C.add_box(L.id,L.out_vars,B); return Fact(B,C)

def tf_add(L: Layer, Bx: Bounds, By: Bounds) -> Fact:
    B=Bounds(Bx.lb+By.lb, Bx.ub+By.ub); C=ConSet()
    C.replace(Con("EQ", tuple(L.out_vars + L.meta["x_vars"] + L.meta["y_vars"]), {"tag":f"add:{L.id}"}))
    C.add_box(L.id,L.out_vars,B); return Fact(B,C)

def tf_mul(L: Layer, Bx: Bounds, By: Bounds) -> Fact:
    cand=torch.stack([Bx.lb*By.lb, Bx.lb*By.ub, Bx.ub*By.lb, Bx.ub*By.ub], dim=0)
    B=Bounds(torch.min(cand,0).values, torch.max(cand,0).values); C=ConSet()
    C.replace(Con("INEQ", tuple(L.out_vars + L.meta["x_vars"] + L.meta["y_vars"]),
        {"tag":f"mcc:{L.id}","lx":Bx.lb,"ux":Bx.ub,"ly":By.lb,"uy":By.ub}))
    C.add_box(L.id,L.out_vars,B); return Fact(B,C)

def tf_concat(L: Layer, Bs: List[Bounds]) -> Fact:
    B=Bounds(torch.cat([b.lb for b in Bs],0), torch.cat([b.ub for b in Bs],0))
    C=ConSet(); C.add_box(L.id,L.out_vars,B); return Fact(B,C)

def tf_bn(L: Layer, Bin: Bounds) -> Fact:
    A,c=L.params["A"],L.params["c"]
    lb=torch.where(A>=0, A*Bin.lb+c, A*Bin.ub+c); ub=torch.where(A>=0, A*Bin.ub+c, A*Bin.lb+c)
    B=Bounds(lb,ub); C=ConSet(); C.replace(Con("EQ", tuple(L.out_vars+L.in_vars), {"tag":f"bn:{L.id}","A":A,"c":c}))
    C.add_box(L.id,L.out_vars,B); return Fact(B,C)

# -------- Less-common MLP-ish --------
def tf_sigmoid(L: Layer, Bin: Bounds) -> Fact:
    f=lambda x: 1/(1+torch.exp(-x)); B=Bounds(f(Bin.lb), f(Bin.ub))
    C=ConSet(); C.replace(Con("INEQ", tuple(L.out_vars+L.in_vars), {"tag":f"sigmoid:{L.id}","segs":pwl_meta(Bin.lb,Bin.ub,2)}))
    C.add_box(L.id,L.out_vars,B); return Fact(B,C)

def tf_tanh(L: Layer, Bin: Bounds) -> Fact:
    B=Bounds(torch.tanh(Bin.lb), torch.tanh(Bin.ub))
    C=ConSet(); C.replace(Con("INEQ", tuple(L.out_vars+L.in_vars), {"tag":f"tanh:{L.id}","segs":pwl_meta(Bin.lb,Bin.ub,2)}))
    C.add_box(L.id,L.out_vars,B); return Fact(B,C)

def tf_softplus(L: Layer, Bin: Bounds) -> Fact:
    f=lambda x: torch.log1p(torch.exp(x)); B=Bounds(f(Bin.lb), f(Bin.ub))
    C=ConSet(); C.replace(Con("INEQ", tuple(L.out_vars+L.in_vars), {"tag":f"softplus:{L.id}","segs":pwl_meta(Bin.lb,Bin.ub,2)}))
    C.add_box(L.id,L.out_vars,B); return Fact(B,C)

def tf_silu(L: Layer, Bin: Bounds) -> Fact:
    s_lb, s_ub = 1/(1+torch.exp(-Bin.lb)), 1/(1+torch.exp(-Bin.ub))
    cand=torch.stack([Bin.lb*s_lb, Bin.lb*s_ub, Bin.ub*s_lb, Bin.ub*s_ub],0)
    B=Bounds(torch.min(cand,0).values, torch.max(cand,0).values)
    C=ConSet(); C.replace(Con("INEQ", tuple(L.out_vars+L.in_vars), {"tag":f"silu:{L.id}","s_lb":s_lb,"s_ub":s_ub}))
    C.add_box(L.id,L.out_vars,B); return Fact(B,C)

def tf_max(L: Layer, By_list: List[Bounds]) -> Fact:
    lb=torch.maximum.reduce([b.lb for b in By_list]); ub=torch.maximum.reduce([b.ub for b in By_list])
    B=Bounds(lb,ub); all_y=sum((L.meta["y_vars_list"][i] for i in range(len(By_list))), [])
    C=ConSet(); C.replace(Con("INEQ", tuple(L.out_vars+all_y), {"tag":f"max:{L.id}","k":len(By_list),"mode":"convex"}))
    C.add_box(L.id,L.out_vars,B); return Fact(B,C)

def tf_min(L: Layer, By_list: List[Bounds]) -> Fact:
    lb=torch.minimum.reduce([b.lb for b in By_list]); ub=torch.minimum.reduce([b.ub for b in By_list])
    B=Bounds(lb,ub); all_y=sum((L.meta["y_vars_list"][i] for i in range(len(By_list))), [])
    C=ConSet(); C.replace(Con("INEQ", tuple(L.out_vars+all_y), {"tag":f"min:{L.id}","k":len(By_list),"mode":"convex"}))
    C.add_box(L.id,L.out_vars,B); return Fact(B,C)

def tf_square(L: Layer, Bin: Bounds) -> Fact:
    l,u=Bin.lb,Bin.ub
    lb=torch.where((l<=0)&(u>=0), 0.0, torch.minimum(l*l, u*u)); ub=torch.maximum(l*l, u*u)
    B=Bounds(lb,ub); C=ConSet(); C.replace(Con("INEQ", tuple(L.out_vars+L.in_vars), {"tag":f"square:{L.id}","segs":pwl_meta(l,u,2)}))
    C.add_box(L.id,L.out_vars,B); return Fact(B,C)

def tf_power(L: Layer, Bin: Bounds) -> Fact:
    p=float(L.meta["p"]); f=lambda x: torch.pow(torch.clamp(x,min=0.0), p)
    B=Bounds(f(Bin.lb), f(Bin.ub)); C=ConSet()
    C.replace(Con("INEQ", tuple(L.out_vars+L.in_vars), {"tag":f"power:{L.id}","p":p,"segs":pwl_meta(Bin.lb,Bin.ub,2)}))
    C.add_box(L.id,L.out_vars,B); return Fact(B,C)

# -------- Additional Activations --------
def tf_relu6(L: Layer, Bin: Bounds) -> Fact:
    """ReLU6: clamp(x, 0, 6)"""
    l, u = Bin.lb, Bin.ub
    lb = torch.clamp(l, min=0.0, max=6.0)
    ub = torch.clamp(u, min=0.0, max=6.0)
    B = Bounds(lb, ub); C = ConSet()
    C.replace(Con("INEQ", tuple(L.out_vars + L.in_vars), {"tag": f"relu6:{L.id}"}))
    C.add_box(L.id, L.out_vars, B); return Fact(B, C)

def tf_hardtanh(L: Layer, Bin: Bounds) -> Fact:
    """HardTanh: clamp(x, min_val, max_val)"""
    min_val = float(L.meta.get("min_val", -1.0))
    max_val = float(L.meta.get("max_val", 1.0))
    l, u = Bin.lb, Bin.ub
    lb = torch.clamp(l, min=min_val, max=max_val)
    ub = torch.clamp(u, min=min_val, max=max_val)
    B = Bounds(lb, ub); C = ConSet()
    C.replace(Con("INEQ", tuple(L.out_vars + L.in_vars), {"tag": f"hardtanh:{L.id}", "min_val": min_val, "max_val": max_val}))
    C.add_box(L.id, L.out_vars, B); return Fact(B, C)

def tf_hardsigmoid(L: Layer, Bin: Bounds) -> Fact:
    """HardSigmoid: clamp(alpha * x + beta, 0, 1)"""
    alpha = float(L.meta.get("alpha", 1/6))
    beta = float(L.meta.get("beta", 0.5))
    l, u = Bin.lb, Bin.ub
    # Apply linear transformation then clamp
    l_linear = alpha * l + beta
    u_linear = alpha * u + beta
    lb = torch.clamp(l_linear, min=0.0, max=1.0)
    ub = torch.clamp(u_linear, min=0.0, max=1.0)
    B = Bounds(lb, ub); C = ConSet()
    C.replace(Con("INEQ", tuple(L.out_vars + L.in_vars), {"tag": f"hardsigmoid:{L.id}", "alpha": alpha, "beta": beta}))
    C.add_box(L.id, L.out_vars, B); return Fact(B, C)

def tf_hardswish(L: Layer, Bin: Bounds) -> Fact:
    """HardSwish: x * hardsigmoid(x)"""
    l, u = Bin.lb, Bin.ub
    # HardSwish bounds are complex, use conservative approximation
    lb = torch.where(l >= 3, l, torch.where(l <= -3, torch.zeros_like(l), torch.minimum(l, torch.zeros_like(l))))
    ub = torch.where(u >= 3, u, torch.where(u <= -3, torch.zeros_like(u), torch.maximum(u, torch.zeros_like(u))))
    B = Bounds(lb, ub); C = ConSet()
    C.replace(Con("INEQ", tuple(L.out_vars + L.in_vars), {"tag": f"hardswish:{L.id}"}))
    C.add_box(L.id, L.out_vars, B); return Fact(B, C)

def tf_mish(L: Layer, Bin: Bounds) -> Fact:
    """Mish: x * tanh(softplus(x))"""
    l, u = Bin.lb, Bin.ub
    # Conservative bounds for Mish activation
    lb = torch.where(l >= 0, 0.0 * l, l)  # Negative values bounded by input
    ub = torch.where(u <= 0, 0.0 * u, u)  # Positive values bounded by input
    B = Bounds(lb, ub); C = ConSet()
    C.replace(Con("INEQ", tuple(L.out_vars + L.in_vars), {"tag": f"mish:{L.id}"}))
    C.add_box(L.id, L.out_vars, B); return Fact(B, C)

def tf_softsign(L: Layer, Bin: Bounds) -> Fact:
    """SoftSign: x / (1 + |x|)"""
    l, u = Bin.lb, Bin.ub
    # SoftSign is bounded between -1 and 1
    lb = l / (1 + torch.abs(l))
    ub = u / (1 + torch.abs(u))
    B = Bounds(lb, ub); C = ConSet()
    C.replace(Con("INEQ", tuple(L.out_vars + L.in_vars), {"tag": f"softsign:{L.id}"}))
    C.add_box(L.id, L.out_vars, B); return Fact(B, C)

# -------- Tensor Operations --------
def tf_reshape(L: Layer, Bin: Bounds) -> Fact:
    """Reshape: identity operation for bounds propagation"""
    # Reshape doesn't change the values, only the tensor shape
    B = Bounds(Bin.lb.clone(), Bin.ub.clone())
    C = ConSet()
    C.replace(Con("EQ", tuple(L.out_vars + L.in_vars), {"tag": f"reshape:{L.id}", "target_shape": L.meta.get("target_shape")}))
    C.add_box(L.id, L.out_vars, B); return Fact(B, C)

def tf_transpose(L: Layer, Bin: Bounds) -> Fact:
    """Transpose: permute dimensions (identity for bounds)"""
    # Transpose doesn't change the values, only the dimension order
    B = Bounds(Bin.lb.clone(), Bin.ub.clone())
    C = ConSet()
    C.replace(Con("EQ", tuple(L.out_vars + L.in_vars), {"tag": f"transpose:{L.id}", "perm": L.meta.get("perm")}))
    C.add_box(L.id, L.out_vars, B); return Fact(B, C)

def tf_squeeze(L: Layer, Bin: Bounds) -> Fact:
    """Squeeze: remove singleton dimensions (identity for bounds)"""
    B = Bounds(Bin.lb.clone(), Bin.ub.clone())
    C = ConSet()
    C.replace(Con("EQ", tuple(L.out_vars + L.in_vars), {"tag": f"squeeze:{L.id}", "dims": L.meta.get("dims")}))
    C.add_box(L.id, L.out_vars, B); return Fact(B, C)

def tf_unsqueeze(L: Layer, Bin: Bounds) -> Fact:
    """Unsqueeze: add singleton dimensions (identity for bounds)"""
    B = Bounds(Bin.lb.clone(), Bin.ub.clone())
    C = ConSet()
    C.replace(Con("EQ", tuple(L.out_vars + L.in_vars), {"tag": f"unsqueeze:{L.id}", "dims": L.meta.get("dims")}))
    C.add_box(L.id, L.out_vars, B); return Fact(B, C)

def tf_tile(L: Layer, Bin: Bounds) -> Fact:
    """Tile: repeat tensor along dimensions"""
    # Conservative bounds: same as input for each repetition
    repeats = L.meta.get("repeats", [1])
    B = Bounds(Bin.lb.clone(), Bin.ub.clone())
    C = ConSet()
    C.replace(Con("EQ", tuple(L.out_vars + L.in_vars), {"tag": f"tile:{L.id}", "repeats": repeats}))
    C.add_box(L.id, L.out_vars, B); return Fact(B, C)

def tf_expand(L: Layer, Bin: Bounds) -> Fact:
    """Expand: broadcast tensor to larger shape"""
    # Broadcasting doesn't change values, only shape
    B = Bounds(Bin.lb.clone(), Bin.ub.clone())
    C = ConSet()
    C.replace(Con("EQ", tuple(L.out_vars + L.in_vars), {"tag": f"expand:{L.id}", "shape": L.meta.get("shape")}))
    C.add_box(L.id, L.out_vars, B); return Fact(B, C)