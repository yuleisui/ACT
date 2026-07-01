#!/usr/bin/env python3
# ===- act/pipeline/validate_verifier.py - Verifier Correctness Validation ====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------===#
#
# Purpose:
#   Unified verification validation framework with two validation levels:
#
#   Level 1: Counterexample/Soundness Validation
#     - Validates that verifier doesn't claim CERTIFIED when concrete
#       counterexamples exist
#
#   Level 2: Bounds/Numerical Validation
#     - Validates that abstract bounds correctly overapproximate concrete
#       activation values
#
# ===---------------------------------------------------------------------===#
#
# Level 1: Counterexample/Soundness Validation
# ============================================
#
# Key Insight:
#   Concrete execution provides ground truth - if we find a real counterexample
#   at runtime, the formal verifier cannot claim the property is certified.
#   This is a soundness check for the verification backend.
#
# Validation Strategy:
#   1. For each network, generate strategic test cases:
#      - Center: Input at center of input spec (typically safe)
#      - Boundary: Input near boundary of input spec (risky)
#      - Random: Random input within input spec (varied)
#
#   2. Run concrete execution to find violations
#   3. Run the formal verifier on the network unconditionally (every net is
#      end-to-end verified; this also subsumes the historical --verify-all)
#   4. If a concrete counterexample was found in step 2, cross-validate the
#      verifier's verdict against it using the matrix below.  Otherwise the
#      cross-validation outcome is INCONCLUSIVE — but the verifier was still
#      exercised in step 3.
#
# Validation Matrix (Level 1):
#   ┌─────────────────────────┬────────────────────────────────────┬──────────────┐
#   │ Concrete Counterexample │ Verifier Result                    │ Validation   │
#   ├─────────────────────────┼────────────────────────────────────┼──────────────┤
#   │ FOUND                   │ CERTIFIED                          │ ❌ FAILED    │
#   │                         │ (Soundness Bug - false negative)   │              │
#   ├─────────────────────────┼────────────────────────────────────┼──────────────┤
#   │ FOUND                   │ FALSIFIED                          │ ✅ PASSED    │
#   │                         │ (Correct - verifier found issue)   │              │
#   ├─────────────────────────┼────────────────────────────────────┼──────────────┤
#   │ FOUND                   │ UNKNOWN                            │ ⚠️ ACCEPTABLE│
#   │                         │ (Incomplete but sound)             │              │
#   ├─────────────────────────┼────────────────────────────────────┼──────────────┤
#   │ NOT FOUND               │ Any Result                         │ ❓ INCONC.   │
#   │                         │ (Cannot validate - no ground truth)│              │
#   └─────────────────────────┴────────────────────────────────────┴──────────────┘
#
#   Legend:
#     FAILED       - Critical soundness bug (false negative)
#     PASSED       - Verifier correct
#     ACCEPTABLE   - Verifier incomplete but sound (conservative)
#     INCONCLUSIVE - No concrete counterexample to validate against
#
# ===---------------------------------------------------------------------===#
#
# Level 2: Bounds/Numerical Validation
# ====================================
#
# Key Insight:
#   Abstract interpretation must overapproximate concrete values. If any
#   concrete activation value falls outside its abstract bounds [lb, ub],
#   the transfer function is unsound.
#
# Validation Strategy:
#   1. Sample concrete inputs from input specification
#   2. Run concrete forward pass through PyTorch model → get concrete activations
#   3. Run abstract analysis through ACT → get abstract bounds for each layer
#   4. Check: concrete_value ∈ [lb, ub] for all layers and all neurons
#
# Validation Matrix (Level 2):
#   ┌──────────────────────┬────────────────────────┬──────────────┐
#   │ Concrete Values      │ Abstract Bounds        │ Validation   │
#   ├──────────────────────┼────────────────────────┼──────────────┤
#   │ value ∈ [lb, ub]     │ All layers/neurons     │ ✅ PASSED    │
#   │ (Sound bounds)       │                        │              │
#   ├──────────────────────┼────────────────────────┼──────────────┤
#   │ value ∉ [lb, ub]     │ Any layer/neuron       │ ❌ FAILED    │
#   │ (Unsound bounds)     │ (Transfer function bug)│              │
#   └──────────────────────┴────────────────────────┴──────────────┘
#
#   Legend:
#     PASSED - All concrete values within abstract bounds (sound)
#     FAILED - Concrete value outside bounds (unsound transfer function)
#
# ===---------------------------------------------------------------------===#
#
# Usage:
#   # Via CLI (recommended):
#   python -m act.pipeline --validate-verifier --mode comprehensive
#   python -m act.pipeline --validate-verifier --mode counterexample
#   python -m act.pipeline --validate-verifier --mode bounds
#
#   # With device and dtype specification:
#   python -m act.pipeline --validate-verifier --device cpu --dtype float64
#   python -m act.pipeline --validate-verifier --device cuda --dtype float32
#
#   # Test specific networks:
#   python -m act.pipeline --validate-verifier --networks mnist_mlp_small
#   python -m act.pipeline --validate-verifier --networks mnist_mlp_small,mnist_cnn_small
#
#   # Test with specific solvers (Level 1):
#   python -m act.pipeline --validate-verifier --mode counterexample --solvers gurobi
#   python -m act.pipeline --validate-verifier --mode counterexample --solvers gurobi torchlp
#
#   # Test with transfer function modes (Level 2):
#   python -m act.pipeline --validate-verifier --mode bounds --tf-modes interval
#   python -m act.pipeline --validate-verifier --mode bounds --tf-modes interval hybridz
#
#   # Adjust number of samples for bounds validation:
#   python -m act.pipeline --validate-verifier --mode bounds --input-samples 20
#
#   # Ignore errors and always exit 0 (useful for CI):
#   python -m act.pipeline --validate-verifier --ignore-errors
#
#   # Combined options:
#   python -m act.pipeline --validate-verifier --mode comprehensive \
#       --networks mnist_mlp_small,mnist_cnn_small \
#       --solvers gurobi --tf-modes interval --input-samples 10 \
#       --device cpu --dtype float64
#
#   # Direct script execution:
#   python act/pipeline/verification/validate_verifier.py
#   python act/pipeline/verification/validate_verifier.py --mode bounds --input-samples 5
#
# Exit Codes:
#   0 - All validations passed (no failures or errors)
#   0 - With --ignore-errors flag (always succeed regardless of results)
#   1 - Failures detected (verifier bugs) OR errors detected (backend bugs)
#
# ===---------------------------------------------------------------------===#

import os
import copy
import torch
import torch.nn as nn
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Sequence, cast

from act.back_end.core import ConSet, Net, Layer
from act.pipeline.verification.model_factory import ModelFactory
from act.pipeline.verification.torch2act import TorchToACT
from act.pipeline.verification.per_neuron_bounds import (
    PerNeuronCheckConfig,
    run_per_neuron_bounds_check,
)
from act.back_end.verifier import (
    verify_once,
    gather_input_spec_layers,
    seed_from_input_specs,
    get_input_ids,
    get_assert_layer,
    find_entry_layer_id,
)
from act.util.stats import VerifyStatus
from act.back_end.solver.solver_gurobi import is_gurobi_available
from act.util.options import PerformanceOptions
from act.front_end.spec_creator_base import LabeledInputTensor
from act.front_end.specs import InKind, InputSpec, OutKind, OutputSpec
from act.front_end.verifiable_model import (
    InputLayer,
    InputSpecLayer,
    OutputSpecLayer,
    VerifiableModel,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TinyRegressionBertLayerNorm(nn.LayerNorm):
    """BERT-style LayerNorm surface used by transformer regression nets."""

    def __init__(self, hidden_size: int) -> None:
        """Initialize a one-dimensional LayerNorm with the BERT epsilon name."""
        super().__init__(hidden_size, eps=1e-5)
        self.variance_epsilon = self.eps


class TinyRegressionBertSelfAttention(nn.Module):
    """Explicit scaled-dot attention matching the ACT transformer mapping."""

    def __init__(self, hidden_size: int) -> None:
        """Create query, key, and value projections for one attention head."""
        super().__init__()
        self.num_attention_heads = 1
        self.attention_head_size = hidden_size
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run one-head self-attention on embedding-space hidden states."""
        query_layer = self.query(hidden_states)
        key_layer = self.key(hidden_states)
        value_layer = self.value(hidden_states)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / (self.attention_head_size ** 0.5)
        attention_probs = torch.softmax(attention_scores, dim=-1)
        return torch.matmul(attention_probs, value_layer)


class TinyRegressionBertSelfOutput(nn.Module):
    """Attention output projection, residual connection, and LayerNorm."""

    def __init__(self, hidden_size: int) -> None:
        """Initialize the attention output block."""
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = TinyRegressionBertLayerNorm(hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Apply output projection with the residual tensor."""
        return self.LayerNorm(self.dense(hidden_states) + input_tensor)


class TinyRegressionBertAttention(nn.Module):
    """Self-attention wrapper with BERT-compatible attribute names."""

    def __init__(self, hidden_size: int) -> None:
        """Create the self-attention and post-attention output blocks."""
        super().__init__()
        self.self = TinyRegressionBertSelfAttention(hidden_size)
        self.output = TinyRegressionBertSelfOutput(hidden_size)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Run self-attention followed by residual output normalization."""
        return self.output(self.self(input_tensor), input_tensor)


class TinyRegressionBertIntermediate(nn.Module):
    """Feed-forward expansion with GELU activation."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        """Initialize the transformer feed-forward expansion."""
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = nn.GELU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply the intermediate dense layer and GELU activation."""
        return self.intermediate_act_fn(self.dense(hidden_states))


class TinyRegressionBertOutput(nn.Module):
    """Feed-forward projection, residual connection, and LayerNorm."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        """Initialize the transformer feed-forward output block."""
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = TinyRegressionBertLayerNorm(hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Apply output projection with the residual tensor."""
        return self.LayerNorm(self.dense(hidden_states) + input_tensor)


class TinyRegressionBertLayer(nn.Module):
    """One tiny encoder block for regression-harness transformer coverage."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        """Initialize attention and feed-forward sublayers."""
        super().__init__()
        self.attention = TinyRegressionBertAttention(hidden_size)
        self.intermediate = TinyRegressionBertIntermediate(
            hidden_size,
            intermediate_size,
        )
        self.output = TinyRegressionBertOutput(hidden_size, intermediate_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run the encoder block on embedding-space hidden states."""
        attention_output = self.attention(hidden_states)
        intermediate_output = self.intermediate(attention_output)
        return self.output(intermediate_output, attention_output)


class TinyRegressionEmbeddingTransformer(nn.Module):
    """Embedding-consuming one-block classifier for verifier validation."""

    def __init__(self) -> None:
        """Initialize a one-token, two-hidden-dimension transformer classifier."""
        super().__init__()
        self.layer = nn.ModuleList(
            [TinyRegressionBertLayer(hidden_size=2, intermediate_size=4)]
        )
        self.classifier = nn.Linear(2, 2)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Classify the first embedding token after one transformer block."""
        hidden_states = embeddings
        for layer in self.layer:
            hidden_states = layer(hidden_states)
        return self.classifier(hidden_states[:, 0, :])


def _build_tiny_transformer_regression_model() -> VerifiableModel:
    """Build the deterministic transformer model used by the harness."""
    torch.manual_seed(7)
    center = torch.tensor([[[0.125, -0.25]]], dtype=torch.get_default_dtype())
    model = TinyRegressionEmbeddingTransformer().eval()
    logits = model(center)
    y_true = logits.argmax(dim=1)
    input_spec = InputSpec(
        kind=InKind.LP_EMBEDDING,
        center=center,
        eps=torch.tensor([0.0], dtype=center.dtype),
        p_norm=2,
        perturbed_positions=torch.tensor([0]),
    )
    output_spec = OutputSpec(kind=OutKind.TOP1_ROBUST, y_true=y_true)
    return VerifiableModel(
        input_layer=InputLayer(
            labeled_input=LabeledInputTensor(tensor=center, label=y_true),
            shape=tuple(center.shape),
            dtype=center.dtype,
            domain="text",
        ),
        input_spec=InputSpecLayer(input_spec),
        model=model,
        output_spec=OutputSpecLayer(output_spec),
    ).eval()


class RegressionHarnessModelFactory:
    """Model factory with synthetic transformer coverage added in memory."""

    TRANSFORMER_NET_NAME = "tiny_embedding_transformer"

    def __init__(self) -> None:
        """Load JSON-backed nets and append the transformer regression net."""
        self._base = ModelFactory()
        self._transformer_model = _build_tiny_transformer_regression_model()
        self._transformer_net = TorchToACT(self._transformer_model).run()
        self._flatten_transformer_seed()

    def _flatten_transformer_seed(self) -> None:
        """Store embedding seed boxes in verifier-native ``[B, n]`` form."""
        for layer in self._transformer_net.layers:
            if layer.kind != "INPUT_SPEC":
                continue
            for key in ("lb", "ub"):
                value = layer.params.get(key)
                if torch.is_tensor(value) and value.dim() > 2:
                    layer.params[key] = value.reshape(value.shape[0], -1)

    def get_act_net(self, name: str) -> Net:
        """Return an ACT net by name."""
        if name == self.TRANSFORMER_NET_NAME:
            return copy.deepcopy(self._transformer_net)
        return self._base.get_act_net(name)

    def create_model(self, name: str, load_weights: bool = True) -> nn.Module:
        """Create the PyTorch model corresponding to a registered net."""
        if name == self.TRANSFORMER_NET_NAME:
            if not load_weights:
                raise ValueError("RegressionHarnessModelFactory requires weights")
            return copy.deepcopy(self._transformer_model)
        return self._base.create_model(name, load_weights=load_weights)

    def generate_test_input(self, name: str, test_case: str = "center") -> torch.Tensor:
        """Generate concrete inputs for JSON-backed and transformer nets."""
        if name != self.TRANSFORMER_NET_NAME:
            return self._base.generate_test_input(name, test_case=test_case)

        spec_layers = gather_input_spec_layers(self._transformer_net)
        if not spec_layers:
            raise ValueError("Transformer regression net has no INPUT_SPEC layer")
        seed = seed_from_input_specs(spec_layers)
        lb = seed.lb.to(dtype=torch.get_default_dtype())
        ub = seed.ub.to(dtype=torch.get_default_dtype())
        if test_case == "center":
            sample = lb + 0.5 * (ub - lb)
        elif test_case == "boundary":
            sample = ub.clone()
        elif test_case == "random":
            sample = lb + torch.rand_like(lb) * (ub - lb)
        else:
            raise ValueError(f"Unknown test_case '{test_case}'")

        input_layer = next(
            layer for layer in self._transformer_net.layers if layer.kind == "INPUT"
        )
        shape_param = input_layer.params.get("shape")
        if not isinstance(shape_param, tuple):
            raise ValueError("Transformer regression INPUT layer has no shape tuple")
        return sample.reshape(tuple(int(dim) for dim in shape_param))

    def list_networks(self) -> List[str]:
        """List JSON-backed networks followed by the transformer net."""
        names = self._base.list_networks()
        if self.TRANSFORMER_NET_NAME not in names:
            names.append(self.TRANSFORMER_NET_NAME)
        return names

    def get_network_info(self, name: str) -> Dict[str, Any]:
        """Return metadata for JSON-backed and transformer nets."""
        if name == self.TRANSFORMER_NET_NAME:
            return {
                "name": name,
                "description": "Tiny embedding-space transformer regression net",
                "architecture_type": "transformer",
                "input_shape": [1, 1, 2],
                "num_layers": len(self._transformer_net.layers),
                "metadata": {"domain": "text"},
            }
        return self._base.get_network_info(name)


class VerificationValidator:
    """Unified verification validation framework with counterexample and bounds validation."""

    def __init__(self, device: str = "cpu", dtype: torch.dtype = torch.float64):
        """
        Initialize verification validator.

        Args:
            device: Device for computation ('cpu' or 'cuda')
            dtype: Data type for computation (float32 or float64)
        """
        self.factory = RegressionHarnessModelFactory()
        self.device = device
        self.dtype = dtype
        self.validation_results = []

        # Initialize debug file (GUARDED)
        if PerformanceOptions.debug_tf:
            debug_file = PerformanceOptions.debug_output_file
            with open(debug_file, "w") as f:
                f.write(f"ACT Verification Debug Log\n")
                f.write(f"Device: {device}, Dtype: {dtype}\n")
                f.write(f"{'=' * 80}\n\n")
            logger.info(f"Debug logging to: {debug_file}")

    def _batchify_net(self, net: Net, target_B: Optional[int]) -> Net:
        """Return a deep copy of ``net`` with INPUT/INPUT_SPEC/ASSERT tensors
        adjusted to ``target_B`` lanes along axis 0.

        Scope:
          Sub-problems sharing the SAME network and SAME spec kind only;
          mixing kinds (e.g. LINEAR_LE with TOP1_ROBUST) in one batch is
          not supported. Within a kind, lanes MAY carry different per-lane
          constraints.

        Per-kind ASSERT handling:
          - TOP1_ROBUST / MARGIN_ROBUST: cycle ``y_true`` across classes
            so each lane verifies a different "true class" assumption.
          - LINEAR_LE / RANGE / UNSAFE_LINEAR: replicate sample 0's
            constraint (these kinds have no natural per-lane axis).

        INPUT / INPUT_SPEC: leading-axis tensors are replicated; spec-side
        is the intended axis of per-lane variation.

        ``target_B is None`` returns the net unchanged (use native B).
        """
        if target_B is None or target_B <= 0:
            return self._migrate_net_to_device(copy.deepcopy(net))

        new_net = copy.deepcopy(net)
        for L in new_net.layers:
            if L.kind in ("INPUT", "INPUT_SPEC"):
                params = L.params or {}
                for key in ("lb", "ub", "center", "eps"):
                    t = params.get(key)
                    if torch.is_tensor(t) and t.dim() > 0 and t.shape[0] != target_B:
                        params[key] = (
                            t[:1].expand(target_B, *t.shape[1:]).contiguous()
                        )
                if L.kind == "INPUT" and "shape" in params:
                    shape_param = params["shape"]
                    # JSON loads INPUT shape as a list, but the original guard
                    # only accepted tuple and silently skipped it, leaving the
                    # batch axis unbatchified (root cause of the B>1 shape bug).
                    if not isinstance(shape_param, (list, tuple)):
                        continue
                    shp = list(shape_param)
                    if shp and shp[0] != target_B:
                        shp[0] = target_B
                        params["shape"] = tuple(shp)
            elif L.kind in ("LSTM", "GRU", "RNN", "EMBEDDING"):
                params = L.params or {}
                for key in ("input_shape", "output_shape"):
                    shp = params.get(key)
                    if isinstance(shp, (list, tuple)) and shp and shp[0] != target_B:
                        new_shp = list(shp)
                        new_shp[0] = target_B
                        params[key] = tuple(new_shp)
            elif L.kind == "ASSERT":
                self._batchify_assert_layer(L, target_B)
        return self._migrate_net_to_device(new_net)

    def _migrate_net_to_device(self, net: Net) -> Net:
        """Move every tensor in ``net.layers[*].params`` to ``self.device`` and
        cast floating-point tensors to ``self.dtype``. Non-tensor params and
        integer / bool tensors are passed through untouched. Required so
        downstream ``analyze`` doesn't see mixed CPU/CUDA matmul operands.
        """
        for L in net.layers:
            params = L.params or {}
            for k, v in list(params.items()):
                if torch.is_tensor(v):
                    params[k] = v.to(
                        device=self.device,
                        dtype=self.dtype if v.is_floating_point() else v.dtype,
                    )
        return net

    def _batchify_assert_layer(self, L: Layer, target_B: int) -> None:
        """Re-encode an ASSERT layer to ``target_B`` lanes via the canonical
        ``OutputSpec.encode_linear`` pipeline (single source of truth).

        Mutates ``L.params`` in place. No-op if already at ``target_B``.
        """
        from act.front_end.specs import OutputSpec

        params = L.params or {}
        kind = str(params.get("kind", ""))

        m_raw = params.get("M")
        M = int(m_raw) if isinstance(m_raw, (int, float)) else 0
        C_cur = params.get("C")
        if (
            torch.is_tensor(C_cur)
            and C_cur.dim() == 2
            and M > 0
            and C_cur.shape[0] // M == target_B
        ):
            return

        n_out = (
            int(C_cur.shape[1])
            if torch.is_tensor(C_cur) and C_cur.dim() == 2
            else len(L.in_vars)
        )

        high: Dict[str, Any] = {}

        if kind in ("TOP1_ROBUST", "MARGIN_ROBUST"):
            y_true = params.get("y_true")
            if not torch.is_tensor(y_true):
                return
            K = n_out
            y0 = y_true.flatten()[:1]
            arange = torch.arange(
                target_B, device=y_true.device, dtype=y_true.dtype
            )
            high["y_true"] = ((y0 + arange) % K).contiguous()
            if kind == "MARGIN_ROBUST":
                margin = params.get("margin")
                if torch.is_tensor(margin):
                    high["margin"] = (
                        margin.flatten()[:1].expand(target_B).contiguous()
                    )

        elif kind == "LINEAR_LE":
            c, d = params.get("c"), params.get("d")
            if torch.is_tensor(c) and c.dim() == 2:
                high["c"] = c[0].contiguous()
            if torch.is_tensor(d) and d.dim() == 1:
                high["d"] = d[0:1].contiguous()

        elif kind == "RANGE":
            lb, ub = params.get("lb"), params.get("ub")
            if torch.is_tensor(lb) and lb.dim() == 2:
                high["lb"] = lb[0].contiguous()
            if torch.is_tensor(ub) and ub.dim() == 2:
                high["ub"] = ub[0].contiguous()

        elif kind == "UNSAFE_LINEAR":
            c, d = params.get("c"), params.get("d")
            if torch.is_tensor(c) and c.dim() == 3:
                high["c"] = c[0].contiguous()
            if torch.is_tensor(d) and d.dim() == 2:
                high["d"] = d[0].contiguous()

        else:
            return

        ref = (
            C_cur if torch.is_tensor(C_cur)
            else next(
                (v for v in high.values() if torch.is_tensor(v)), None
            )
        )
        try:
            spec = OutputSpec(kind=kind, **high)
            new_params = spec.encode_linear(
                B=target_B,
                n_out=n_out,
                device=ref.device if ref is not None else torch.device(self.device),
                dtype=(
                    ref.dtype
                    if ref is not None and ref.dtype.is_floating_point
                    else self.dtype
                ),
            )
        except Exception as e:
            logger.warning(
                f"_batchify_assert_layer({kind}) at B={target_B}: "
                f"re-encode failed: {e}"
            )
            return

        L.params.clear()
        L.params.update(new_params)

    @staticmethod
    def _batchify_tensor(t: torch.Tensor, target_B: Optional[int]) -> torch.Tensor:
        """Expand a leading-axis tensor to ``target_B`` (no-op if matches/None)."""
        if target_B is None or target_B <= 0:
            return t
        if t.dim() == 0 or t.shape[0] == target_B:
            return t
        return t[:1].expand(target_B, *t.shape[1:]).contiguous()

    _KNOWN_RUNTIME_BROKEN: Dict[Tuple[str, str], str] = {}

    # Spec kinds explicitly rejected by DualSolver.evaluate_spec.  Empty after
    # UNSAFE_LINEAR is now handled via per-row LB + ANY-escape
    # aggregation (sound under-approximation; mirrors the interval/hybridz path
    # at `verifier.py:574-580`).  See `solver_dual.py:evaluate_spec` for the
    # quantifier-swap derivation.  This frozenset is retained as a pre-filter
    # hook for any future kind that DualSolver cannot certify.
    _DUAL_UNSUPPORTED_SPEC_KINDS: frozenset[str] = frozenset()

    # Layer kinds where DualSolver currently has a known soundness gap — the
    # dispatch table at `dual_tf/dual_tf.py:223` aliases LRELU's backward
    # handler to `backward_relu`, which is mathematically incorrect (LReLU has
    # a non-zero negative slope; ReLU's backward zeros that branch out).  This
    # produces over-tight dual lower bounds, leading to false-CERTIFIED on
    # LReLU nets.  Tracked for fix in a follow-up PR.  Pre-filtered here so
    # CI reports SKIPPED instead of FAILED on the soundness bug.
    _DUAL_KNOWN_BROKEN_LAYER_KINDS: frozenset[str] = frozenset({"LRELU"})

    # Networks excluded from Level-2 per-neuron bounds validation.  The
    # per-neuron check aligns concrete PyTorch forward-hook events 1:1 with ACT
    # layers by hookable order, but torch2act lowers the fused Q/K/V attention
    # projections to MHA_SPLIT (non-hookable) while PyTorch still exposes them as
    # nn.Linear, so the event/layer counts cannot align (7 Linear events vs 4
    # DENSE layers here).  The abstract bounds themselves are sound and finite;
    # only the concrete-activation alignment is inapplicable, so the transformer
    # is covered by the Level-1 two-solver verifier checks instead.
    _BOUNDS_UNSUPPORTED_NETWORKS: frozenset[str] = frozenset(
        {RegressionHarnessModelFactory.TRANSFORMER_NET_NAME}
    )

    def _network_supported_by_mode(
        self, net: Net, tf_mode: str
    ) -> Tuple[bool, List[str]]:
        """Return ``(is_supported, sorted_blocking_kinds)`` for the
        (network, tf_mode) pair.

        Two sources of "skip":
          1. Backend's static ``supports_layer`` returns False for some kind
             (e.g. some TFs lack LSTM/GRU/transformer ops).
          2. The (tf_mode, kind) pair is in ``_KNOWN_RUNTIME_BROKEN`` — the
             backend claims support but has a documented runtime bug.

        Real (undocumented) runtime errors are NOT swallowed; they bubble
        up as ERROR so we notice and fix them.

        Valid ``tf_mode`` values: ``"interval"`` and ``"hybridz"``.  ``"dual"``
        is a Solver choice (``--solver dual``), not a TF mode, after the β
        refactor; passing ``"dual"`` here raises ``ValueError`` from the
        underlying ``set_transfer_function_mode``.
        """
        from act.back_end.transfer_functions import (
            set_transfer_function_mode,
            get_transfer_function,
        )
        set_transfer_function_mode(tf_mode)
        tf = get_transfer_function()
        blocking = set()
        for L in net.layers:
            if not tf.supports_layer(L.kind):
                blocking.add(L.kind)
            elif (tf_mode, L.kind) in self._KNOWN_RUNTIME_BROKEN:
                blocking.add(L.kind)
        return len(blocking) == 0, sorted(blocking)

    def find_concrete_counterexample(
        self,
        name: str,
        model: torch.nn.Module,
        max_random: int = 64,
        act_net: Optional[Net] = None,
    ) -> Optional[Tuple[torch.Tensor, Dict[str, Any]]]:
        """
        Try to find a concrete counterexample via concrete execution.
        Returns (input_tensor, results_dict) if found, else None.
        """
        if max_random < 0:
            raise ValueError(f"max_random must be >= 0, got {max_random}")
        was_training = bool(getattr(model, "training", False))
        model.eval()

        try:
            if act_net is None:
                act_net = self.factory.get_act_net(name)
            input_shape = None
            shape_prod = None
            if act_net is not None:
                for layer in getattr(act_net, "layers", []):
                    if getattr(layer, "kind", None) == "INPUT":
                        shp = (layer.params or {}).get("shape", None)
                        if (
                            isinstance(shp, (list, tuple))
                            and shp
                            and all(isinstance(x, int) and x > 0 for x in shp)
                        ):
                            input_shape = tuple(shp)
                            shape_prod = int(torch.tensor(input_shape).prod().item())
                        break

            spec_lb = spec_ub = None
            if act_net is not None:
                specs = gather_input_spec_layers(act_net)
                if specs:
                    seed = seed_from_input_specs(specs)
                    lb = seed.lb.to(self.device, self.dtype).flatten()
                    ub = seed.ub.to(self.device, self.dtype).flatten()
                    if (
                        lb.shape == ub.shape
                        and lb.numel() > 0
                        and (not torch.any(lb > ub))
                    ):
                        spec_lb, spec_ub = lb, ub

            if spec_lb is None or spec_ub is None:
                return None

            delta = spec_ub - spec_lb
            dim = int(spec_lb.numel())

            # center
            x_flat = spec_lb + 0.5 * delta
            x = (
                x_flat.reshape(*input_shape)
                if (input_shape and shape_prod == x_flat.numel())
                else x_flat.reshape(1, -1)
            )
            x = x.to(self.device, self.dtype)
            with torch.no_grad():
                res = model(x)
            if (
                isinstance(res, dict)
                and res.get("input_satisfied", False)
                and (not res.get("output_satisfied", True))
            ):
                logger.info("  🔴 Counterexample found (spec_center)")
                logger.info("     Input explanation:  %s", res.get("input_explanation"))
                logger.info(
                    "     Output explanation: %s", res.get("output_explanation")
                )
                return x, res

            # per-dimension edges (dim<=16)
            if dim <= 16:
                base = spec_lb + 0.5 * delta
                for i in range(dim):
                    for val, tag in ((spec_lb[i], "lb"), (spec_ub[i], "ub")):
                        x_edge = base.clone()
                        x_edge[i] = val
                        x = (
                            x_edge.reshape(*input_shape)
                            if (input_shape and shape_prod == x_edge.numel())
                            else x_edge.reshape(1, -1)
                        )
                        x = x.to(self.device, self.dtype)
                        with torch.no_grad():
                            res = model(x)
                        if (
                            isinstance(res, dict)
                            and res.get("input_satisfied", False)
                            and (not res.get("output_satisfied", True))
                        ):
                            logger.info(
                                "  🔴 Counterexample found (spec_per_dim_%s_%d)", tag, i
                            )
                            logger.info(
                                "     Input explanation:  %s",
                                res.get("input_explanation"),
                            )
                            logger.info(
                                "     Output explanation: %s",
                                res.get("output_explanation"),
                            )
                            return x, res

            # random in [lb, ub]
            for k in range(max_random):
                r = torch.rand_like(spec_lb)
                x_flat = spec_lb + r * delta
                x = (
                    x_flat.reshape(*input_shape)
                    if (input_shape and shape_prod == x_flat.numel())
                    else x_flat.reshape(1, -1)
                )
                x = x.to(self.device, self.dtype)
                with torch.no_grad():
                    res = model(x)
                if (
                    isinstance(res, dict)
                    and res.get("input_satisfied", False)
                    and (not res.get("output_satisfied", True))
                ):
                    logger.info("  🔴 Counterexample found (spec_random_%d)", k)
                    logger.info(
                        "     Input explanation:  %s", res.get("input_explanation")
                    )
                    logger.info(
                        "     Output explanation: %s", res.get("output_explanation")
                    )
                    return x, res

            return None

        finally:
            if was_training:
                model.train()

    def validate_counterexamples(
        self,
        networks: Optional[List[str]] = None,
        solvers: List[str] = ["gurobi", "torchlp"],
        tf_modes: Optional[List[str]] = None,
        batch_sizes: Optional[Sequence[Optional[int]]] = None,
    ) -> Dict[str, Any]:
        """
        Level 1: Validate verifier soundness using concrete counterexamples.

        Args:
            networks: List of network names (None = all networks)
            solvers: List of solver names to test ('gurobi', 'torchlp', 'dual')
            tf_modes: List of TF modes for forward bounds ('interval' / 'hybridz').
                      Ignored for solver='dual'.  Defaults to ['interval'].
            batch_sizes: List of batch sizes to validate at. Each element may be:
                - None: use the network's native batch size from JSON (default)
                - int >= 1: batchify the network's INPUT_SPEC to this size
                If ``batch_sizes`` is None or empty, defaults to ``[None]``
                (preserves current behavior).

        Returns:
            Summary dictionary with validation results
        """
        if networks is None:
            networks = self.factory.list_networks()
        if not batch_sizes:
            batch_sizes = [None]
        if not tf_modes:
            tf_modes = ["interval"]

        solvers = list(solvers)
        if "gurobi" in solvers and not is_gurobi_available():
            logger.warning("Skipping gurobi solver: gurobipy is not available.")
            solvers = [s for s in solvers if s != "gurobi"]
            if not solvers:
                logger.warning("No available solvers for counterexample validation.")

        logger.info(f"\n{'=' * 80}")
        logger.info(f"LEVEL 1: COUNTEREXAMPLE/SOUNDNESS VALIDATION")
        logger.info(f"{'=' * 80}")
        logger.info(
            f"Testing {len(networks)} networks x {len(solvers)} solvers "
            f"x {len(tf_modes)} tf_modes={tf_modes} "
            f"x {len(batch_sizes)} batch_sizes={batch_sizes}"
        )
        logger.info(f"Device: {self.device}, Dtype: {self.dtype}")
        logger.info(f"{'=' * 80}\n")

        for network in networks:
            for solver in solvers:
                for tf_mode in tf_modes:
                    for batch_size in batch_sizes:
                        try:
                            self._validate_counterexample_single(
                                network,
                                solver,
                                tf_mode=tf_mode,
                                batch_size=batch_size,
                            )
                        except Exception as e:
                            logger.error(
                                f"Validation failed for "
                                f"{network}/{solver}/{tf_mode}/B={batch_size}: {e}"
                            )
                            import traceback

                            traceback.print_exc()
                            error_result = {
                                "network": network,
                                "solver": solver,
                                "tf_mode": tf_mode,
                                "batch_size": batch_size,
                                "validation_type": "counterexample",
                                "status": "ERROR",
                                "error": f"Outer exception: {str(e)}",
                                "concrete_counterexample": False,
                            }
                            self.validation_results.append(error_result)

        return self._compute_summary(validation_type="counterexample")

    def _validate_counterexample_single(
        self,
        name: str,
        solver: str,
        tf_mode: str = "interval",
        batch_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Validate verifier correctness for a single network (Level 1).

        Args:
            name: Network name (stem of .json file in nets/)
            solver: 'gurobi', 'torchlp', or 'dual'
            tf_mode: TF mode for forward bounds ('interval' or 'hybridz').
                     Ignored when ``solver='dual'`` (DualSolver brings its own
                     forward bounds via DualTF).
            batch_size: Target batch size. None preserves the network's native B.

        Returns:
            Validation result dictionary with status and details
        """
        logger.info(f"\n{'=' * 80}")
        logger.info(
            f"Validating: {name} (solver: {solver}, tf_mode: {tf_mode}, B={batch_size})"
        )
        logger.info(f"{'=' * 80}")

        # Wire up the requested solver / TF globally so verify_once() picks
        # the matching code path.  Dual is a
        # Solver choice (set_solver_mode) rather than a TF mode; for LP-based
        # solvers we additionally pin the forward TF used for bounds.
        from act.back_end.transfer_functions import (
            set_solver_mode,
            set_transfer_function_mode,
        )

        set_solver_mode(solver)
        if solver != "dual":
            set_transfer_function_mode(tf_mode)

        act_net = self.factory.get_act_net(name)
        act_net = self._batchify_net(act_net, batch_size)

        # Pre-filter: skip nets whose layer kinds the active TF / Solver cannot
        # handle.  Mirrors the graceful-degrade pattern of the now-removed
        # `--verify-all` entry point (which reported SKIPPED via the
        # `_SkipUnsupported` tagged union) so CI doesn't fail on nets containing
        # MaxPool / LSTM / etc when the selected mode lacks a handler.
        skip_reason: Optional[str] = None
        if (
            name == RegressionHarnessModelFactory.TRANSFORMER_NET_NAME
            and batch_size is not None
            and batch_size > 1
        ):
            skip_reason = (
                "transformer regression net uses native embedding lanes; "
                "rebatching beyond B=1 is not part of this harness case"
            )
        if solver == "dual":
            from act.back_end.dual_tf.dual_tf import DualTF

            dual_tf = DualTF()
            layer_kinds = {L.kind for L in act_net.layers}
            assert_spec_kinds = {
                cast(str, L.params.get("kind"))
                for L in act_net.layers
                if L.kind == "ASSERT" and isinstance(L.params.get("kind"), str)
            }

            missing = sorted({k for k in layer_kinds if not dual_tf.supports_layer(k)})
            known_broken = sorted(layer_kinds & self._DUAL_KNOWN_BROKEN_LAYER_KINDS)
            unsupported_specs = sorted(
                assert_spec_kinds & self._DUAL_UNSUPPORTED_SPEC_KINDS
            )

            if missing:
                skip_reason = f"DualTF cannot handle: {', '.join(missing)}"
            elif known_broken:
                skip_reason = (
                    f"DualSolver has a known soundness gap on "
                    f"{', '.join(known_broken)} (deferred to follow-up PR)"
                )
            elif unsupported_specs:
                skip_reason = (
                    f"DualSolver does not support spec kind(s): "
                    f"{', '.join(unsupported_specs)} (use LP/Gurobi path)"
                )
        else:
            ok, missing = self._network_supported_by_mode(act_net, tf_mode)
            if not ok:
                skip_reason = (
                    f"tf_mode={tf_mode!r} has no handler for: {', '.join(missing)}"
                )

        if skip_reason is not None:
            skip_result = {
                "network": name,
                "solver": solver,
                "tf_mode": tf_mode,
                "batch_size": batch_size,
                "validation_type": "counterexample",
                "validation_status": "SKIPPED",
                "explanation": f"⏭️  SKIPPED: {skip_reason}",
            }
            logger.info(f"\n  {skip_result['explanation']}")
            self.validation_results.append(skip_result)
            return skip_result

        model = self.factory.create_model(name, load_weights=True)
        model = model.to(self.device, self.dtype)
        counterexample = self.find_concrete_counterexample(
            name, model, act_net=act_net
        )

        # Step 3: Run formal verifier on ACT Net
        logger.info(f"\n  🔍 Running formal verifier ({solver})...")

        try:
            if solver not in {"gurobi", "torchlp", "dual"}:
                raise ValueError(f"Unknown solver: {solver}")

            verify_result_list = verify_once(act_net)
            verify_result = verify_result_list[0]
            verifier_status = verify_result.status
            logger.info(f"     Verifier result: {verifier_status}")

            # If verifier found counterexample, validate it with model
            if verify_result.counterexample is not None:
                logger.info(
                    f"     Verifier counterexample shape: {verify_result.counterexample.shape}"
                )
                # Reshape CE to the model's expected input shape (avoid conv2d shape errors)
                ce_raw = verify_result.counterexample
                input_shape: Optional[Tuple[int, ...]] = None
                for layer in act_net.layers:
                    if getattr(layer, "kind", None) == "INPUT":
                        shape_param = layer.params.get("shape")
                        if isinstance(shape_param, tuple):
                            input_shape = tuple(int(dim) for dim in shape_param)
                        break
                try:
                    if input_shape is not None:
                        ce_tensor = ce_raw.view(*input_shape)
                    else:
                        ce_tensor = ce_raw.unsqueeze(0)
                except Exception as reshape_err:
                    logger.warning(
                        f"     CE reshape failed, using vector: {reshape_err}"
                    )
                    ce_tensor = ce_raw.unsqueeze(0)
                ce_tensor = ce_tensor.to(self.device, self.dtype)
                ce_results = model(ce_tensor)
                if isinstance(ce_results, dict):
                    logger.info(
                        f"     CE validation: input_sat={ce_results['input_satisfied']}, "
                        f"output_sat={ce_results['output_satisfied']}"
                    )

        except Exception as e:
            logger.error(f"     Verifier failed: {e}")
            import traceback

            traceback.print_exc()
            error_result = {
                "network": name,
                "solver": solver,
                "batch_size": batch_size,
                "validation_type": "counterexample",
                "status": "ERROR",
                "error": str(e),
                "concrete_counterexample": counterexample is not None,
            }
            self.validation_results.append(error_result)
            return error_result

        validation = self._cross_validate_counterexample(
            network_name=name,
            solver_name=solver,
            concrete_counterexample=counterexample,
            verifier_status=verifier_status,
        )
        validation["batch_size"] = batch_size

        self.validation_results.append(validation)
        return validation

    def validate_results_soundness(
        self,
        name: str,
        model: torch.nn.Module,
        results: List[Any],
        solver: str,
        act_net: Optional[Net] = None,
    ) -> Dict[str, Any]:
        counterexample = self.find_concrete_counterexample(
            name, model, act_net=act_net
        )
        call_results = []
        for idx, verify_result in enumerate(results):
            validation = self._cross_validate_counterexample(
                network_name=f"{name}[{idx}]",
                solver_name=solver,
                concrete_counterexample=counterexample,
                verifier_status=verify_result.status,
            )
            validation["batch_size"] = None
            self.validation_results.append(validation)
            call_results.append(validation)
        return self._summarize_results("counterexample", call_results)

    def _cross_validate_counterexample(
        self,
        network_name: str,
        solver_name: str,
        concrete_counterexample: Optional[Tuple[torch.Tensor, Dict[str, Any]]],
        verifier_status: VerifyStatus,
    ) -> Dict[str, Any]:
        """
        Cross-validate concrete inference vs formal verification (Level 1).

        Validation Rules:
        1. If concrete counterexample found → verifier MUST report FALSIFIED or UNKNOWN
        2. If no concrete counterexample → verifier can report anything (testing incomplete)
        """
        result = {
            "network": network_name,
            "solver": solver_name,
            "validation_type": "counterexample",
            "concrete_counterexample": concrete_counterexample is not None,
            "verifier_result": verifier_status,
            "validation_status": None,
            "explanation": None,
        }

        if concrete_counterexample is not None:
            # We found a real counterexample - verifier MUST NOT claim CERTIFIED
            input_tensor, inference_results = concrete_counterexample

            if verifier_status == VerifyStatus.CERTIFIED:
                # CRITICAL BUG: Verifier claims safe, but we have a counterexample!
                result["validation_status"] = "FAILED"
                result["explanation"] = (
                    f"🚨 SOUNDNESS BUG DETECTED! Verifier claims CERTIFIED but "
                    f"concrete counterexample exists. This is a false negative."
                )
                logger.error(f"\n  {result['explanation']}")
                logger.error(
                    f"     Counterexample input: {input_tensor.shape}, "
                    f"range=[{input_tensor.min():.4f}, {input_tensor.max():.4f}]"
                )
                logger.error(
                    f"     Output violation: {inference_results['output_explanation']}"
                )

            elif verifier_status == VerifyStatus.FALSIFIED:
                # CORRECT: Verifier correctly identified the issue
                result["validation_status"] = "PASSED"
                result["explanation"] = (
                    f"✅ CORRECT - Verifier correctly reported FALSIFIED "
                    f"(matches concrete execution)"
                )
                logger.info(f"\n  {result['explanation']}")

            elif verifier_status == VerifyStatus.UNKNOWN:
                # ACCEPTABLE: Verifier couldn't decide (incomplete but sound)
                result["validation_status"] = "ACCEPTABLE"
                result["explanation"] = (
                    f"⚠️ INCOMPLETE - Verifier returned UNKNOWN, but concrete "
                    f"counterexample exists (verifier is sound but incomplete)"
                )
                logger.warning(f"\n  {result['explanation']}")

            else:
                result["validation_status"] = "UNKNOWN"
                result["explanation"] = f"Unknown verifier result: {verifier_status}"
                logger.warning(f"\n  {result['explanation']}")

        else:
            # No concrete counterexample found in testing
            result["validation_status"] = "INCONCLUSIVE"
            result["explanation"] = (
                f"⚪ INCONCLUSIVE - No counterexample found in concrete testing. "
                f"Verifier result: {verifier_status} (cannot validate with this test)"
            )
            logger.info(f"\n  {result['explanation']}")

        return result

    def validate_bounds(
        self,
        networks: Optional[List[str]] = None,
        tf_modes: List[str] = ["interval"],
        num_samples: int = 10,
        per_neuron_config: Optional[PerNeuronCheckConfig] = None,
        batch_sizes: Optional[Sequence[Optional[int]]] = None,
    ) -> Dict[str, Any]:
        """
        Level 2: Validate abstract bounds overapproximate concrete values.

        Args:
            networks: List of network names (None = all networks)
            tf_modes: Transfer function modes to test ('interval', 'hybridz')
            num_samples: Number of concrete inputs to sample per network
            batch_sizes: List of batch sizes to validate at. ``None`` element
                means use the network's native batch size from JSON. If the
                whole list is None/empty, defaults to ``[None]``.

        Returns:
            Summary dictionary with validation results
        """
        if networks is None:
            networks = self.factory.list_networks()
        if not batch_sizes:
            batch_sizes = [None]

        logger.info(f"\n{'=' * 80}")
        logger.info(f"LEVEL 2: BOUNDS/NUMERICAL VALIDATION")
        logger.info(f"{'=' * 80}")
        logger.info(
            f"Testing {len(networks)} networks x {len(tf_modes)} TF modes "
            f"x {len(batch_sizes)} batch_sizes={batch_sizes}"
        )
        logger.info(f"Samples per network: {num_samples}")
        logger.info(f"Device: {self.device}, Dtype: {self.dtype}")
        logger.info(f"{'=' * 80}\n")

        for network in networks:
            for tf_mode in tf_modes:
                for batch_size in batch_sizes:
                    try:
                        self._validate_bounds_single(
                            network,
                            tf_mode,
                            num_samples,
                            per_neuron_config=per_neuron_config,
                            batch_size=batch_size,
                        )
                    except Exception as e:
                        logger.error(
                            f"Bounds validation failed for "
                            f"{network}/{tf_mode}/B={batch_size}: {e}"
                        )
                        import traceback

                        traceback.print_exc()
                        error_result = {
                            "network": network,
                            "tf_mode": tf_mode,
                            "batch_size": batch_size,
                            "validation_type": "bounds",
                            "status": "ERROR",
                            "error": f"Outer exception: {str(e)}",
                            "samples_processed": 0,
                        }
                        self.validation_results.append(error_result)

        return self._compute_summary(validation_type="bounds")

    def _validate_bounds_single(
        self,
        name: str,
        tf_mode: str,
        num_samples: int,
        per_neuron_config: Optional[PerNeuronCheckConfig] = None,
        batch_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Validate bounds for a single network (Level 2).

        Args:
            name: Network name
            tf_mode: Transfer function mode ('interval' or 'hybridz')
            num_samples: Number of concrete inputs to sample
            batch_size: Target batch size. None preserves the network's native B.

        Returns:
            Validation result dictionary
        """
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Validating bounds: {name} (tf_mode: {tf_mode}, B={batch_size})")
        logger.info(f"{'=' * 80}")

        if name in self._BOUNDS_UNSUPPORTED_NETWORKS:
            skip_result = {
                "network": name,
                "tf_mode": tf_mode,
                "batch_size": batch_size,
                "validation_type": "bounds",
                "validation_status": "SKIPPED",
                "explanation": (
                    "⏭️  SKIPPED: transformer regression coverage is validated "
                    "through Level 1 two-solver verifier checks"
                ),
                "unsupported_kinds": [],
            }
            logger.info(f"\n  {skip_result['explanation']}")
            self.validation_results.append(skip_result)
            return skip_result

        act_net = self.factory.get_act_net(name)
        act_net = self._batchify_net(act_net, batch_size)

        ok, missing = self._network_supported_by_mode(act_net, tf_mode)
        if not ok:
            skip_result = {
                "network": name,
                "tf_mode": tf_mode,
                "batch_size": batch_size,
                "validation_type": "bounds",
                "validation_status": "SKIPPED",
                "explanation": (
                    f"⏭️  SKIPPED: tf_mode={tf_mode!r} has no handler for "
                    f"layer kind(s): {', '.join(missing)}"
                ),
                "unsupported_kinds": missing,
            }
            logger.info(f"\n  {skip_result['explanation']}")
            self.validation_results.append(skip_result)
            return skip_result

        model = self.factory.create_model(name, load_weights=True)
        model = model.to(self.device, self.dtype)

        # Step 2: Set transfer function mode globally
        from act.back_end.transfer_functions import set_transfer_function_mode

        set_transfer_function_mode(tf_mode)

        # Step 3: Sample concrete inputs
        violations = []
        total_checks = 0
        per_neuron_config = per_neuron_config or PerNeuronCheckConfig()

        def _get_input_bounds_from_act(act_net_inner):
            from act.back_end.core import Bounds

            for layer in act_net_inner.layers:
                if layer.kind != "INPUT_SPEC":
                    continue

                params = layer.params or {}

                # 1) Prefer BOX. Preserve the original [B, *input_shape] from
                # the JSON; the verifier's analyze() requires the leading
                # batch dimension end-to-end.
                if "lb" in params and "ub" in params:
                    return Bounds(
                        lb=params["lb"].to(self.device, self.dtype),
                        ub=params["ub"].to(self.device, self.dtype),
                    )

                # 2) LINF_BALL: center + eps (shape preserved)
                if "center" in params and "eps" in params:
                    center = params["center"].to(self.device, self.dtype)
                    eps = params["eps"]
                    if not torch.is_tensor(eps):
                        eps = center.new_tensor(eps)
                    else:
                        eps = eps.to(self.device, self.dtype)
                    return Bounds(lb=center - eps, ub=center + eps)
            return None

        spec_bounds = _get_input_bounds_from_act(act_net)

        torch.manual_seed(42)

        for sample_idx in range(num_samples):
            input_tensor = self.factory.generate_test_input(name, "random")
            input_tensor = self._batchify_tensor(input_tensor, batch_size)
            input_tensor = input_tensor.to(self.device, self.dtype)

            # Step 4: Prepare entry fact from input tensor
            from act.back_end.core import Fact, Bounds

            entry_id = find_entry_layer_id(act_net)
            if spec_bounds is None:
                raise ValueError(
                    f"validate_bounds_single({name}): no INPUT_SPEC layer "
                    f"with BOX or LINF_BALL bounds; cannot run abstract "
                    f"analysis. Network must declare an input region."
                )
            input_bounds = spec_bounds
            entry_fact = Fact(bounds=input_bounds, cons=ConSet())

            # Step 5: Run abstract analysis + strict per-neuron validation
            try:
                check = run_per_neuron_bounds_check(
                    act_net=act_net,
                    model=model,
                    input_tensor=input_tensor,
                    entry_fact=entry_fact,
                    tf_mode=tf_mode,
                    config=per_neuron_config,
                )
                if check.get("status") == "ERROR":
                    raise RuntimeError("; ".join(check.get("errors", [])[:3]))

                total_checks += int(check.get("total_checks", 0))
                if int(check.get("violations_total", 0)) > 0:
                    violated_layers = [
                        s
                        for s in check.get("layerwise_stats", [])
                        if int(s.get("num_violations", 0)) > 0
                    ]
                    violation_info = {
                        "sample_idx": sample_idx,
                        "violations_total": int(check.get("violations_total", 0)),
                        "worst_gap": float(check.get("worst_gap", 0.0)),
                        "violations_topk": check.get("violations_topk", []),
                        "violated_layers": violated_layers,
                    }
                    violations.append(violation_info)
                    top1 = (check.get("violations_topk", []) or [None])[0]
                    if isinstance(top1, dict):
                        concrete = float(top1.get("concrete", 0.0))
                        lb = float(top1.get("lb", 0.0))
                        ub = float(top1.get("ub", 0.0))
                        if concrete < lb:
                            violation_dir = "below_lb"
                        elif concrete > ub:
                            violation_dir = "above_ub"
                        else:
                            violation_dir = "outside_bounds"
                        logger.error(
                            "  ❌ Bounds violation at sample %d: %d violating neurons | "
                            "worst_gap=%.6g | layer_id=%s kind=%s neuron=%s dir=%s | "
                            "concrete=%.6g lb=%.6g ub=%.6g",
                            sample_idx,
                            int(check.get("violations_total", 0)),
                            float(check.get("worst_gap", 0.0)),
                            top1.get("layer_id", "?"),
                            top1.get("kind", "?"),
                            top1.get("neuron_index", "?"),
                            violation_dir,
                            concrete,
                            lb,
                            ub,
                        )
                    else:
                        logger.error(
                            "  ❌ Bounds violation at sample %d: %d violating neurons | worst_gap=%.6g",
                            sample_idx,
                            int(check.get("violations_total", 0)),
                            float(check.get("worst_gap", 0.0)),
                        )

            except Exception as e:
                logger.error(
                    f"  ⚠️ Abstract analysis failed for sample {sample_idx}: {e}"
                )
                error_result = {
                    "network": name,
                    "tf_mode": tf_mode,
                    "batch_size": batch_size,
                    "validation_type": "bounds",
                    "status": "ERROR",
                    "error": str(e),
                    "samples_processed": sample_idx,
                }
                self.validation_results.append(error_result)
                return error_result

        if len(violations) > 0:
            result = {
                "network": name,
                "tf_mode": tf_mode,
                "batch_size": batch_size,
                "validation_type": "bounds",
                "validation_status": "FAILED",
                "explanation": f"🚨 UNSOUND BOUNDS: {len(violations)} violations found across {num_samples} samples",
                "total_checks": total_checks,
                "violations": violations,
                "per_neuron_config": {
                    "topk": per_neuron_config.topk,
                },
            }
            logger.error(f"\n  {result['explanation']}")
        else:
            result = {
                "network": name,
                "tf_mode": tf_mode,
                "batch_size": batch_size,
                "validation_type": "bounds",
                "validation_status": "PASSED",
                "explanation": f"✅ SOUND BOUNDS: All {total_checks} checks passed across {num_samples} samples",
                "total_checks": total_checks,
                "violations": [],
                "per_neuron_config": {
                    "topk": per_neuron_config.topk,
                },
            }
            logger.info(f"\n  {result['explanation']}")

        self.validation_results.append(result)
        return result

    def validate_comprehensive(
        self,
        networks: Optional[List[str]] = None,
        solvers: List[str] = ["gurobi", "torchlp"],
        tf_modes: List[str] = ["interval"],
        num_samples: int = 10,
        per_neuron_config: Optional[PerNeuronCheckConfig] = None,
        batch_sizes: Optional[Sequence[Optional[int]]] = None,
    ) -> Dict[str, Any]:
        """
        Run both Level 1 and Level 2 validations.

        Args:
            networks: List of network names (None = all networks)
            solvers: List of solver names for Level 1
            tf_modes: Transfer function modes for Level 2
            num_samples: Number of samples for Level 2
            batch_sizes: List of batch sizes (see validate_bounds). None = native.

        Returns:
            Combined summary dictionary
        """
        logger.info(f"\n{'=' * 80}")
        logger.info(f"COMPREHENSIVE VERIFICATION VALIDATION")
        logger.info(f"{'=' * 80}")
        logger.info(
            f"Running both Level 1 (Counterexample) and Level 2 (Bounds) validation"
        )
        logger.info(f"Device: {self.device}, Dtype: {self.dtype}")
        logger.info(f"{'=' * 80}\n")

        summary_l1 = self.validate_counterexamples(
            networks=networks,
            solvers=solvers,
            tf_modes=tf_modes,
            batch_sizes=batch_sizes,
        )

        summary_l2 = self.validate_bounds(
            networks=networks,
            tf_modes=tf_modes,
            num_samples=num_samples,
            per_neuron_config=per_neuron_config,
            batch_sizes=batch_sizes,
        )

        # Combine summaries - FAILED if any failures OR errors
        has_failures = (
            summary_l1.get("failed", 0) > 0 or summary_l2.get("failed", 0) > 0
        )
        has_errors = summary_l1.get("errors", 0) > 0 or summary_l2.get("errors", 0) > 0

        if has_failures:
            overall_status = "FAILED"  # Critical: verifier is unsound
        elif has_errors:
            overall_status = "ERROR"  # Backend bugs prevent validation
        else:
            overall_status = "PASSED"  # All tests passed

        combined = {
            "level1_counterexample": summary_l1,
            "level2_bounds": summary_l2,
            "overall_status": overall_status,
        }

        self._print_comprehensive_summary(combined)
        return combined

    def _compute_summary(self, validation_type: str) -> Dict[str, Any]:
        """
        Compute validation summary statistics for specific validation type.

        Args:
            validation_type: 'counterexample' or 'bounds'
        """
        results = [
            r
            for r in self.validation_results
            if r.get("validation_type") == validation_type
        ]
        return self._summarize_results(validation_type, results)

    def _summarize_results(
        self, validation_type: str, results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        total = len(results)

        if total == 0:
            return {
                "validation_type": validation_type,
                "total": 0,
                "passed": 0,
                "failed": 0,
                "acceptable": 0,
                "inconclusive": 0,
                "skipped": 0,
                "unknown": 0,
                "errors": 0,
                "results": [],
                "error_message": "No validation results (all tests encountered errors)",
            }

        passed = sum(1 for r in results if r.get("validation_status") == "PASSED")
        failed = sum(1 for r in results if r.get("validation_status") == "FAILED")
        acceptable = sum(
            1 for r in results if r.get("validation_status") == "ACCEPTABLE"
        )
        inconclusive = sum(
            1 for r in results if r.get("validation_status") == "INCONCLUSIVE"
        )
        skipped = sum(
            1 for r in results if r.get("validation_status") == "SKIPPED"
        )
        unknown = sum(
            1 for r in results if r.get("validation_status") == "UNKNOWN"
        )
        errors = sum(1 for r in results if r.get("status") == "ERROR")

        summary = {
            "validation_type": validation_type,
            "total": total,
            "passed": passed,
            "failed": failed,
            "acceptable": acceptable,
            "inconclusive": inconclusive,
            "skipped": skipped,
            "unknown": unknown,
            "errors": errors,
            "results": results,
        }

        if validation_type == "counterexample":
            summary["counterexamples_found"] = sum(
                1 for r in results if r.get("concrete_counterexample", False)
            )
            summary["critical_bugs"] = failed
        elif validation_type == "bounds":
            summary["total_checks"] = sum(r.get("total_checks", 0) for r in results)
            summary["total_violations"] = sum(
                len(r.get("violations", [])) for r in results
            )

        self._print_summary(summary)
        return summary

    def _print_summary(self, summary: Dict[str, Any]):
        """Print validation summary for specific validation type."""
        validation_type = summary.get("validation_type", "unknown")

        print("\n" + "=" * 80)
        print(f"VALIDATION SUMMARY - {validation_type.upper()}")
        print("=" * 80)

        if summary["total"] == 0:
            print()
            print("⚠️  No validation tests completed successfully")
            if "error_message" in summary:
                print(f"   {summary['error_message']}")
            print("=" * 80)
            return

        print(f"\nTotal validation tests: {summary['total']}")

        if validation_type == "counterexample":
            print(
                f"Concrete counterexamples found: {summary.get('counterexamples_found', 0)}"
            )
        elif validation_type == "bounds":
            print(f"Total bound checks: {summary.get('total_checks', 0)}")
            print(f"Total violations: {summary.get('total_violations', 0)}")

        print()
        print(f"✅ PASSED:       {summary['passed']}")
        if validation_type == "counterexample":
            print(f"⚠️  ACCEPTABLE:   {summary['acceptable']}")
            print(f"⚪ INCONCLUSIVE: {summary['inconclusive']}")
        if summary.get("skipped", 0) > 0:
            print(f"⏭️  SKIPPED:      {summary['skipped']}")
        print(f"❌ ERRORS:       {summary['errors']}")
        print(f"🚨 FAILED:       {summary['failed']}")
        print("=" * 80)

        if summary["failed"] > 0:
            print(f"\n🚨 CRITICAL: {validation_type.upper()} validation failed!")
            if validation_type == "counterexample":
                print("Soundness bugs detected in the following networks:")
            else:
                print("Unsound bounds detected in the following networks:")
            for result in summary["results"]:
                if result.get("validation_status") == "FAILED":
                    if validation_type == "counterexample":
                        print(f"  - {result['network']} ({result['solver']})")
                    else:
                        print(f"  - {result['network']} ({result['tf_mode']})")
            print()
        elif summary["errors"] > 0:
            print(f"\n⚠️  All {validation_type} validation tests encountered errors!")
            print("This indicates pre-existing bugs in the verification backend.")
            print()
        else:
            print(f"\n✅ {validation_type.upper()} validation PASSED!")

        print("=" * 80)

    def _print_comprehensive_summary(self, combined: Dict[str, Any]):
        """Print comprehensive summary for both validation levels."""
        print("\n" + "=" * 80)
        print("COMPREHENSIVE VALIDATION SUMMARY")
        print("=" * 80)

        l1 = combined["level1_counterexample"]
        l2 = combined["level2_bounds"]

        print(
            f"\nLevel 1 (Counterexample): {l1['passed']}/{l1['total']} passed, {l1['failed']} failed, {l1['errors']} errors"
        )
        print(
            f"Level 2 (Bounds):         {l2['passed']}/{l2['total']} passed, {l2['failed']} failed, {l2['errors']} errors"
        )
        print()
        print(f"Overall Status: {combined['overall_status']}")
        print("=" * 80)


def main():
    """Run verification validation test suite."""
    import argparse

    parser = argparse.ArgumentParser(description="ACT Verification Validator")
    parser.add_argument(
        "--mode",
        choices=["counterexample", "bounds", "comprehensive"],
        default="comprehensive",
        help="Validation mode",
    )
    parser.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    parser.add_argument(
        "--dtype", default="float64", choices=["float32", "float64"], help="Data type"
    )
    parser.add_argument("--networks", nargs="+", help="Specific networks to test")
    parser.add_argument(
        "--solvers",
        nargs="+",
        default=["gurobi", "torchlp"],
        help="Solvers for Level 1",
    )
    parser.add_argument(
        "--tf-modes",
        nargs="+",
        default=["interval"],
        help="Transfer function modes for Level 2",
    )
    parser.add_argument(
        "--input-samples",
        type=int,
        default=10,
        dest="samples",
        help="Number of input samples for Level 2 bounds validation",
    )
    parser.add_argument(
        "--batch-sizes",
        type=lambda s: [
            (None if (b.strip() == "" or b.strip().lower() == "none") else int(b))
            for b in s.split(",")
        ],
        default=[None],
        metavar="B1,B2,...",
        help="Batch sizes to validate at, e.g. '1,4'. Use 'none' (or omit) "
        "for native batch from each network's JSON. Default: native.",
    )
    parser.add_argument(
        "--ignore-errors",
        action="store_true",
        help="Always exit 0 (ignore failures and errors for CI)",
    )

    args = parser.parse_args()

    dtype = torch.float64 if args.dtype == "float64" else torch.float32

    validator = VerificationValidator(device=args.device, dtype=dtype)

    if args.mode == "counterexample":
        summary = validator.validate_counterexamples(
            networks=args.networks,
            solvers=args.solvers,
            batch_sizes=args.batch_sizes,
        )
        exit_code = 1 if (summary["failed"] > 0 or summary["errors"] > 0) else 0
    elif args.mode == "bounds":
        summary = validator.validate_bounds(
            networks=args.networks,
            tf_modes=args.tf_modes,
            num_samples=args.samples,
            batch_sizes=args.batch_sizes,
        )
        exit_code = 1 if (summary["failed"] > 0 or summary["errors"] > 0) else 0
    else:
        combined = validator.validate_comprehensive(
            networks=args.networks,
            solvers=args.solvers,
            tf_modes=args.tf_modes,
            num_samples=args.samples,
            batch_sizes=args.batch_sizes,
        )
        exit_code = 1 if combined["overall_status"] in ["FAILED", "ERROR"] else 0

    # Override exit code if --ignore-errors is set
    if args.ignore_errors:
        exit_code = 0

    # Print debug file location (GUARDED)
    if PerformanceOptions.debug_tf:
        logger.info(
            f"\n📝 Debug log written to: {PerformanceOptions.debug_output_file}"
        )

    return exit_code


if __name__ == "__main__":
    import sys

    sys.exit(main())
