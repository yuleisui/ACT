# transfer.pseudo
import torch
from typing import List
from act.back_end.core import Bounds, Con, ConSet, Fact, Layer
from act.back_end.utils import affine_bounds, pwl_meta, bound_var_interval, scale_interval

# -------- MLP Basics --------
def tf_dense(L: Layer, Bin: Bounds) -> Fact:
    B = affine_bounds(L.params["W_pos"], L.params["W_neg"], L.params["b"], Bin)
    C = ConSet(); C.replace(Con("EQ", tuple(L.out_vars + L.in_vars), {"tag": f"dense:{L.id}", "W": L.params["W"], "b": L.params["b"]}))
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
    a=float(L.params["alpha"]); l,u=Bin.lb,Bin.ub; on=l>=0; off=u<=0; amb=~(on|off)
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
    C.replace(Con("EQ", tuple(L.out_vars + L.params["x_vars"] + L.params["y_vars"]), {"tag":f"add:{L.id}"}))
    C.add_box(L.id,L.out_vars,B); return Fact(B,C)

def tf_mul(L: Layer, Bx: Bounds, By: Bounds) -> Fact:
    cand=torch.stack([Bx.lb*By.lb, Bx.lb*By.ub, Bx.ub*By.lb, Bx.ub*By.ub], dim=0)
    B=Bounds(torch.min(cand,0).values, torch.max(cand,0).values); C=ConSet()
    C.replace(Con("INEQ", tuple(L.out_vars + L.params["x_vars"] + L.params["y_vars"]),
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
    B=Bounds(lb,ub); all_y=sum((L.params["y_vars_list"][i] for i in range(len(By_list))), [])
    C=ConSet(); C.replace(Con("INEQ", tuple(L.out_vars+all_y), {"tag":f"max:{L.id}","k":len(By_list),"mode":"convex"}))
    C.add_box(L.id,L.out_vars,B); return Fact(B,C)

def tf_min(L: Layer, By_list: List[Bounds]) -> Fact:
    lb=torch.minimum.reduce([b.lb for b in By_list]); ub=torch.minimum.reduce([b.ub for b in By_list])
    B=Bounds(lb,ub); all_y=sum((L.params["y_vars_list"][i] for i in range(len(By_list))), [])
    C=ConSet(); C.replace(Con("INEQ", tuple(L.out_vars+all_y), {"tag":f"min:{L.id}","k":len(By_list),"mode":"convex"}))
    C.add_box(L.id,L.out_vars,B); return Fact(B,C)

def tf_square(L: Layer, Bin: Bounds) -> Fact:
    l,u=Bin.lb,Bin.ub
    lb=torch.where((l<=0)&(u>=0), 0.0, torch.minimum(l*l, u*u)); ub=torch.maximum(l*l, u*u)
    B=Bounds(lb,ub); C=ConSet(); C.replace(Con("INEQ", tuple(L.out_vars+L.in_vars), {"tag":f"square:{L.id}","segs":pwl_meta(l,u,2)}))
    C.add_box(L.id,L.out_vars,B); return Fact(B,C)

def tf_power(L: Layer, Bin: Bounds) -> Fact:
    p=float(L.params["p"]); f=lambda x: torch.pow(torch.clamp(x,min=0.0), p)
    B=Bounds(f(Bin.lb), f(Bin.ub)); C=ConSet()
    C.replace(Con("INEQ", tuple(L.out_vars+L.in_vars), {"tag":f"power:{L.id}","p":p,"segs":pwl_meta(Bin.lb,Bin.ub,2)}))
    C.add_box(L.id,L.out_vars,B); return Fact(B,C)