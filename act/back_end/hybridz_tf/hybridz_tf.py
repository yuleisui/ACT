# ===- act/back_end/hybridz_tf/hybridz_tf.py - HybridZ Transfer Function -====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------===#
#
# Purpose:
#   HybridZ Transfer Function Implementation.
#
#   Each hz_tf_* is a complete TF for one layer kind, combining
#   HZ zonotope propagation with interval_tf constraint generation.
#   hz_tf_* live in tf_mlp.py / tf_cnn.py alongside their layer types.
#   HZ domain ops co-locate with the hz_tf_* that use them.
#
# ===---------------------------------------------------------------------===#

""" """

import torch
from typing import Dict, Optional
from act.back_end.core import Bounds, Fact, Layer, Net, ConSet
from act.back_end.transfer_functions import TransferFunction
from act.back_end.layer_schema import LayerKind
from act.back_end.solver.solver_hz import (
    HZono,
    hz_from_bounds,
)

import act.back_end.hybridz_tf.tf_mlp as hz_mlp
import act.back_end.hybridz_tf.tf_cnn as hz_cnn
import act.back_end.hybridz_tf.tf_rnn as hz_rnn
import act.back_end.hybridz_tf.tf_transformer as hz_transformer
import act.back_end.interval_tf.tf_mlp as interval_mlp
import act.back_end.interval_tf.tf_cnn as interval_cnn


class HybridzTF(TransferFunction):
    def __init__(self):
        self._hz_cache: Dict[int, HZono] = {}
        self._cache_net_id: Optional[int] = None
        self._tanh_K: int = 1
        self._sigmoid_K: int = 1
        self._var_id_stride: int = 1

    @staticmethod
    def _net_var_id_stride(net: Net) -> int:
        max_id = -1
        for layer in net.layers:
            if layer.in_vars:
                max_id = max(max_id, max(layer.in_vars))
            if layer.out_vars:
                max_id = max(max_id, max(layer.out_vars))
        return max(max_id + 1, 1)

    _LAYER_REGISTRY = {
        # Identity / spec
        LayerKind.INPUT.value: lambda L, b, tf: Fact(bounds=b, cons=ConSet()),
        LayerKind.INPUT_SPEC.value: lambda L, b, tf: Fact(bounds=b, cons=ConSet()),
        LayerKind.ASSERT.value: lambda L, b, tf: Fact(bounds=b, cons=ConSet()),
        # MLP: HZ + interval
        LayerKind.DENSE.value: lambda L, b, tf: hz_mlp.tf_dense(L, b, tf),
        LayerKind.BIAS.value: lambda L, b, tf: hz_mlp.tf_bias(L, b, tf),
        LayerKind.SCALE.value: lambda L, b, tf: hz_mlp.tf_scale(L, b, tf),
        LayerKind.RELU.value: lambda L, b, tf: hz_mlp.tf_relu(L, b, tf),
        LayerKind.LRELU.value: lambda L, b, tf: hz_mlp.tf_lrelu(L, b, tf),
        LayerKind.TANH.value: lambda L, b, tf: hz_mlp.tf_tanh(L, b, tf),
        LayerKind.SIGMOID.value: lambda L, b, tf: hz_mlp.tf_sigmoid(L, b, tf),
        LayerKind.ERF.value: lambda L, b, tf: hz_mlp.tf_erf(L, b, tf),
        LayerKind.SQRT.value: lambda L, b, tf: hz_mlp.tf_sqrt(L, b, tf),
        LayerKind.SIN.value: lambda L, b, tf: hz_mlp.tf_sin(L, b, tf),
        LayerKind.COS.value: lambda L, b, tf: hz_mlp.tf_cos(L, b, tf),
        LayerKind.QUANTIZE.value: lambda L, b, tf: hz_mlp.tf_quantize(L, b, tf),
        LayerKind.ABS.value: lambda L, b, tf: hz_mlp.tf_abs(L, b, tf),
        LayerKind.BN.value: lambda L, b, tf: hz_mlp.tf_bn(L, b, tf),
        # Multi-input: HZ + interval
        LayerKind.ADD.value: lambda L, b, tf: hz_mlp.tf_add(L, b, tf),
        LayerKind.MUL.value: lambda L, b, tf: hz_mlp.tf_mul(L, b, tf),
        LayerKind.SUB.value: lambda L, b, tf: hz_mlp.tf_sub(L, b, tf),
        LayerKind.DIV.value: lambda L, b, tf: hz_mlp.tf_div(L, b, tf),
        LayerKind.CONCAT.value: lambda L, b, tf: hz_mlp.tf_concat(L, b, tf),
        # CNN: HZ + interval
        LayerKind.CONV2D.value: lambda L, b, tf: hz_cnn.tf_conv2d(L, b, tf),
        LayerKind.MAXPOOL2D.value: lambda L, b, tf: hz_cnn.tf_maxpool2d(L, b, tf),
        # Activations: interval-only
        LayerKind.CLIP.value: lambda L, b, tf: interval_mlp.tf_clip(L, b),
        LayerKind.SOFTPLUS.value: lambda L, b, tf: interval_mlp.tf_softplus(L, b),
        LayerKind.SILU.value: lambda L, b, tf: interval_mlp.tf_silu(L, b),
        LayerKind.RELU6.value: lambda L, b, tf: interval_mlp.tf_relu6(L, b),
        LayerKind.HARDTANH.value: lambda L, b, tf: interval_mlp.tf_hardtanh(L, b),
        LayerKind.HARDSIGMOID.value: lambda L, b, tf: interval_mlp.tf_hardsigmoid(L, b),
        LayerKind.HARDSWISH.value: lambda L, b, tf: interval_mlp.tf_hardswish(L, b),
        LayerKind.MISH.value: lambda L, b, tf: interval_mlp.tf_mish(L, b),
        LayerKind.SOFTSIGN.value: lambda L, b, tf: interval_mlp.tf_softsign(L, b),
        LayerKind.SQUARE.value: lambda L, b, tf: interval_mlp.tf_square(L, b),
        LayerKind.POWER.value: lambda L, b, tf: interval_mlp.tf_power(L, b),
        LayerKind.SIGN.value: lambda L, b, tf: hz_mlp.tf_sign(L, b, tf),
        LayerKind.REDUCE_SUM.value: lambda L, b, tf: hz_mlp.tf_reduce_sum(L, b, tf),
        LayerKind.CONSTANT.value: lambda L, b, tf: hz_mlp.tf_constant(L, b, tf),
        LayerKind.COMPARE.value: lambda L, b, tf: hz_mlp.tf_compare(L, b, tf),
        LayerKind.WHERE.value: lambda L, b, tf: hz_mlp.tf_where(L, b, tf),
        LayerKind.MATMUL.value: lambda L, b, tf: hz_mlp.tf_matmul(L, b, tf),
        LayerKind.ARG_EXTREMUM.value: lambda L, b, tf: hz_mlp.tf_arg_extremum(L, b, tf),
        LayerKind.UPSAMPLE.value: lambda L, b, tf: hz_mlp.tf_upsample(L, b, tf),
        LayerKind.SCATTER_ND.value: lambda L, b, tf: hz_mlp.tf_scatter_nd(L, b, tf),
        LayerKind.MAX.value: lambda L, b, tf: interval_mlp.tf_max(
            L, tf._net.get_all_predecessor_bounds(L.id, tf._after, tf._before)
        ),
        LayerKind.MIN.value: lambda L, b, tf: interval_mlp.tf_min(
            L, tf._net.get_all_predecessor_bounds(L.id, tf._after, tf._before)
        ),
        # CNN: interval-only
        LayerKind.AVGPOOL1D.value: lambda L, b, tf: interval_cnn.tf_avgpool1d(L, b),
        LayerKind.AVGPOOL2D.value: lambda L, b, tf: hz_cnn.tf_avgpool2d(L, b, tf),
        LayerKind.MAXPOOL3D.value: lambda L, b, tf: interval_cnn.tf_maxpool3d(L, b),
        LayerKind.PAD.value:      lambda L, b, tf: interval_cnn.tf_pad(L, b),
        LayerKind.CONV1D.value: lambda L, b, tf: interval_cnn.tf_conv1d(L, b),
        LayerKind.CONV3D.value: lambda L, b, tf: interval_cnn.tf_conv3d(L, b),
        LayerKind.CONVTRANSPOSE2D.value: lambda L, b, tf: hz_cnn.tf_convtranspose2d(L, b, tf),
        LayerKind.FLATTEN.value: lambda L, b, tf: hz_mlp.tf_flatten(L, b, tf),
        LayerKind.RESHAPE.value: lambda L, b, tf: hz_mlp.tf_reshape(L, b, tf),
        LayerKind.TRANSPOSE.value: lambda L, b, tf: hz_mlp.tf_transpose(L, b, tf),
        LayerKind.SQUEEZE.value: lambda L, b, tf: hz_mlp.tf_squeeze(L, b, tf),
        LayerKind.UNSQUEEZE.value: lambda L, b, tf: hz_mlp.tf_unsqueeze(L, b, tf),
        LayerKind.EXPAND.value: lambda L, b, tf: hz_mlp.tf_expand(L, b, tf),
        LayerKind.SLICE.value: lambda L, b, tf: hz_mlp.tf_slice(L, b, tf),
        LayerKind.GATHER.value: lambda L, b, tf: hz_mlp.tf_gather(L, b, tf),
        # RNN
        LayerKind.LSTM.value: lambda L, b, tf: hz_rnn.tf_lstm(L, b, tf),
        LayerKind.GRU.value: lambda L, b, tf: hz_rnn.tf_gru(L, b, tf),
        LayerKind.RNN.value: lambda L, b, tf: hz_rnn.tf_rnn(L, b, tf),
        LayerKind.EMBEDDING.value: lambda L, b, tf: hz_rnn.tf_embedding(L, b, tf),
        LayerKind.EMBEDDING_TF.value: lambda L, b, tf: hz_rnn.tf_embedding(L, b, tf),
        # Transformer
        LayerKind.POSENC.value: lambda L, b, tf: hz_transformer.tf_posenc(L, b, tf),
        LayerKind.LAYERNORM.value: lambda L, b, tf: hz_transformer.tf_layernorm(L, b, tf),
        LayerKind.GELU.value: lambda L, b, tf: hz_transformer.tf_gelu(L, b, tf),
        LayerKind.ATT_SCORES.value: lambda L, b, tf: hz_transformer.tf_att_scores(L, b, tf),
        LayerKind.SOFTMAX.value: lambda L, b, tf: hz_transformer.tf_softmax(L, b, tf),
        LayerKind.ATT_MIX.value: lambda L, b, tf: hz_transformer.tf_att_mix(L, b, tf),
        LayerKind.MHA_SPLIT.value: lambda L, b, tf: hz_transformer.tf_mha_split(L, b, tf),
        LayerKind.MHA_JOIN.value: lambda L, b, tf: hz_transformer.tf_mha_join(L, b, tf),
        LayerKind.MASK_ADD.value: lambda L, b, tf: hz_transformer.tf_mask_add(L, b, tf),
    }

    @property
    def name(self) -> str:
        return "HybridzTF"

    def supports_layer(self, layer_kind: str) -> bool:
        return layer_kind.upper() in self._LAYER_REGISTRY

    def get_hz(self, layer_id: int) -> Optional[HZono]:
        return self._hz_cache.get(int(layer_id))

    @staticmethod
    def _id_sig(ids: Optional[torch.Tensor]):
        if ids is None:
            return None
        vals = ids.detach().cpu().reshape(-1).tolist()
        return (tuple(ids.shape), tuple(int(v) for v in vals))

    @classmethod
    def _hz_sig(cls, hz: Optional[HZono]):
        if hz is None:
            return None
        eq_sig = None
        if hz.eq_mask is not None:
            eq_sig = (
                tuple(hz.eq_mask.shape),
                tuple(bool(v) for v in hz.eq_mask.detach().cpu().reshape(-1).tolist()),
            )
        return (
            tuple(hz.c.shape),
            tuple(hz.Gc.shape),
            tuple(hz.Gb.shape),
            tuple(hz.Ac.shape),
            tuple(hz.Ab.shape),
            tuple(hz.b.shape),
            eq_sig,
            cls._id_sig(hz.col_ids),
            cls._id_sig(hz.bcol_ids),
        )

    def side_state_signature(self, layer_id: int):
        return self._hz_sig(self._hz_cache.get(int(layer_id)))

    _HZ_MAX_INPUT_DIM = 1024

    def _col_ids_from_vars(self, bounds: Bounds, var_ids) -> Optional[torch.Tensor]:
        if not var_ids:
            return None
        n = len(var_ids)
        total = int(bounds.lb.numel())
        if n <= 0 or total % n != 0:
            return None
        base = torch.tensor(var_ids, dtype=torch.long, device=bounds.lb.device)
        B = total // n
        if B == 1:
            return base
        offsets = (
            torch.arange(B, dtype=torch.long, device=bounds.lb.device).view(-1, 1)
            * self._var_id_stride
        )
        return (offsets + base.view(1, -1)).reshape(-1)

    def _hz_from_bounds(
        self,
        bounds: Bounds,
        *,
        col_ids: Optional[torch.Tensor] = None,
    ) -> Optional[HZono]:
        lb, ub = bounds.lb.flatten(), bounds.ub.flatten()
        rad = (ub - lb) / 2.0
        ng = int((rad > 0).sum().item())
        if ng > self._HZ_MAX_INPUT_DIM:
            return None
        ids = col_ids.to(device=lb.device) if col_ids is not None else None
        return hz_from_bounds(
            bounds,
            lb.dtype,
            lb.device,
            col_ids=ids,
        )

    def apply(
        self,
        L: Layer,
        input_bounds: Bounds,
        net: Net,
        before: Dict[int, Fact],
        after: Dict[int, Fact],
    ) -> Fact:
        k = L.kind.upper()
        if k not in self._LAYER_REGISTRY:
            raise NotImplementedError(f"HybridzTF: Unsupported layer kind '{k}'")

        net_id = id(net)
        if self._cache_net_id != net_id:
            self._hz_cache.clear()
            self._cache_net_id = net_id
            self._var_id_stride = self._net_var_id_stride(net)

        self._net = net
        self._before = before
        self._after = after

        if k in ("INPUT", "INPUT_SPEC"):
            hz_init = self._hz_from_bounds(
                input_bounds,
                col_ids=self._col_ids_from_vars(input_bounds, L.out_vars),
            )
            if hz_init is not None:
                self._hz_cache[L.id] = hz_init
        elif k != "ASSERT":
            preds = net.preds.get(L.id, [])
            if preds and preds[0] in self._hz_cache:
                self._hz_cache[L.id] = self._hz_cache[preds[0]]
            elif not preds:
                hz_init = self._hz_from_bounds(
                    input_bounds,
                    col_ids=self._col_ids_from_vars(input_bounds, L.in_vars),
                )
                if hz_init is not None:
                    self._hz_cache[L.id] = hz_init

        n_out = len(L.out_vars)
        hz_carried = self._hz_cache.get(L.id)
        ngnb = (
            hz_carried.Gc.shape[1] + hz_carried.Gb.shape[1]
            if hz_carried is not None
            else 0
        )
        if max(n_out, ngnb) > self._HZ_MAX_INPUT_DIM and k not in (
            "INPUT",
            "INPUT_SPEC",
            "ASSERT",
        ):
            self._hz_cache.pop(L.id, None)

        hz_before = self._hz_cache.get(L.id)
        result = self._LAYER_REGISTRY[k](L, input_bounds, self)

        if (
            hz_before is not None
            and self._hz_cache.get(L.id) is hz_before
            and k not in ("INPUT", "INPUT_SPEC")
        ):
            self._hz_cache[L.id] = hz_from_bounds(
                result.bounds, result.bounds.lb.dtype, result.bounds.lb.device
            )

        return result
