"""
Torch-native DNN verification abstraction framework.

This module provides:
- Core data structures (Bounds, Con, ConSet, Fact, Layer, Net)
- Transfer functions for MLP and Transformer layers  
- Worklist-based bounds propagation analysis
- Constraint export to LP/MILP solvers
- Branch-and-bound verification with spurious CE elimination
- Input/output specification handling
- Device-aware tensor management

Example usage:
    >>> from act.back_end import *
    >>> # Create verification problem
    >>> I = InputSpec(kind=InKind.BOX, lb=torch.tensor([-1.0]), ub=torch.tensor([1.0]))
    >>> O = OutputSpec(kind=OutKind.MARGIN_ROBUST, y_true=0, margin=0.0)
    >>> # Run verification
    >>> result = verify_once(net, entry_id, input_ids, output_ids, I, O, seed_bounds, solver)
"""

# Core data structures
from .core import Bounds, Con, ConSet, Fact, Layer, Net

# Device management  
from .device_manager import initialize_device_dtype, ensure_initialized, summary, temp_device_dtype, wrap_model_fn

# Utilities
from .utils import box_join, changed_or_maskdiff, update_cache, affine_bounds
from .utils import pwl_meta, bound_var_interval, scale_interval

# Transfer functions
from .tf_mlp import (
    tf_dense, tf_bias, tf_scale, tf_relu, tf_lrelu, tf_abs, tf_clip,
    tf_add, tf_mul, tf_concat, tf_bn
)
from .tf_cnn import (
    tf_conv2d, tf_maxpool2d, tf_avgpool2d, tf_flatten
)
from .tf_rnn import (
    tf_lstm, tf_gru, tf_rnn, tf_embedding
)
from .tf_transformer import (
    tf_posenc, tf_layernorm, tf_gelu, tf_att_scores,
    tf_softmax, tf_att_mix, tf_mha_split, tf_mha_join, tf_mask_add
)

# Analysis algorithms
from .analyze import dispatch_tf, analyze

# Constraint export
from .cons_exportor import export_to_solver, to_numpy

# Solver interfaces
from .solver_base import Solver, SolverCaps, SolveStatus
from .solver_gurobi import GurobiSolver
from .solver_torch import TorchLPSolver

# Verification specifications
from .verify_status import (
    VerifStatus, VerifResult,
    seed_from_input_spec, add_input_spec, materialise_input_poly,
    add_negated_output_spec_to_solver, verify_once
)
from act.front_end.specs import InputSpec, OutputSpec, InKind, OutKind

# Branch-and-bound verification
from .bab import BabNode, verify_bab

__all__ = [
    # Core
    'Bounds', 'Con', 'ConSet', 'Fact', 'Layer', 'Net',
    # Device management
    'initialize_device_dtype', 'ensure_initialized', 'summary', 'temp_device_dtype', 'wrap_model_fn',
    # Utilities  
    'box_join', 'changed_or_maskdiff', 'update_cache', 'affine_bounds',
    'pwl_meta', 'bound_var_interval', 'scale_interval',
    # Transfer functions
    'tf_dense', 'tf_bias', 'tf_scale', 'tf_relu', 'tf_lrelu', 'tf_abs', 'tf_clip',
    'tf_add', 'tf_mul', 'tf_concat', 'tf_bn',
    'tf_conv2d', 'tf_maxpool2d', 'tf_avgpool2d', 'tf_flatten',  # CNN transfer functions
    'tf_lstm', 'tf_gru', 'tf_rnn', 'tf_embedding',  # RNN transfer functions
    'tf_posenc', 'tf_layernorm', 'tf_gelu', 'tf_att_scores',
    'tf_softmax', 'tf_att_mix', 'tf_mha_split', 'tf_mha_join', 'tf_mask_add',
    # Analysis
    'dispatch_tf', 'analyze',
    # Export
    'export_to_solver', 'to_numpy',
    # Solvers
    'Solver', 'SolverCaps', 'SolveStatus', 'GurobiSolver', 'TorchLPSolver',
    # Specs
    'InputSpec', 'OutputSpec', 'InKind', 'OutKind', 'VerifStatus', 'VerifResult',
    'seed_from_input_spec', 'add_input_spec', 'verify_once',
    # BaB
    'BabNode', 'verify_bab'
]