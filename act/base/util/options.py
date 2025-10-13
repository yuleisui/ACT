#===- util.options.py ----ACT Native Parameters & Option Definitions---------#
#
#                 ACT: Abstract Constraints Transformer
#
# Copyright (C) <2025->  ACT Team
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Authors: ACT Team   
#                                                                         
# External tool compatibility parameters are adapted from:               
#   - α,β-CROWN: https://github.com/Verified-Intelligence/alpha-beta-CROWN 
#     Copyright (C) 2021-2025 The α,β-CROWN Team                           
#     Licensed under BSD 3-Clause License                                  
#   - ERAN: https://github.com/eth-sri/eran                                
#     Copyright ETH Zurich, Licensed under Apache 2.0 License     
#   - This integration enables unified command-line interface across        
#     different verification backends while maintaining ACT's native  
#     capabilities and novel contributions.   
#
# Purpose:
#   Centralised definition of native parameters and command-line options for
#   the Abstract Constraints Transformer (ACT), including defaults, validation,
#   and help text used across the toolchain.
#
#===----------------------------------------------------------------------===#
    
import argparse

def get_parser():

    parser = argparse.ArgumentParser(description='Abstract Constraint Transformer (ACT) - Unified Neural Network Verification Framework')
    
    # ACT Core Verifier Selection
    parser.add_argument('--verifier', type=str, default=None, 
                        choices=['base', 'eran', 'abcrown', 'hybridz', 'act'],
                        help='Backend verification engine. "base": ACT native base bounds analysis, "eran": ERAN external verifier, "abcrown": αβ-CROWN external verifier, "hybridz": ACT novel hybrid zonotope verifier, "act": ACT torch-native abstraction framework')
    parser.add_argument('--method', type=str, default=None, 
                        help='Verification method. ERAN: [deepzono, refinezono, deeppoly, refinepoly], αβ-CROWN: [alpha, beta, alpha_beta], ACT-HybridZ: [hybridz, hybridz_relaxed, hybridz_relaxed_with_bab], ACT-Base: [base], ACT-Native: [base, torch-native]')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                        help='Computation device (cpu or cuda)')
    parser.add_argument('--dtype', type=str, default='float64', 
                        choices=['float16', 'float32', 'float64'],
                        help='Floating point precision for numerical computations (float16, float32, float64). Default: float64 for maximum precision')
    parser.add_argument('--solver', type=str, default='auto', 
                        choices=['auto', 'gurobi', 'torch', 'both'],
                        help='Solver backend for constraint solving. "auto": try Gurobi first, fallback to PyTorch, "gurobi": Gurobi MILP solver only, "torch": PyTorch LP solver only, "both": test both solvers')

    # ACT Verification Features
    parser.add_argument('--bounds_prop_metadata', action='store_true', default=True,
                        help='ACT: Enable bounds propagation metadata tracking for detailed statistics and debugging')
    parser.add_argument('--no_bounds_prop_metadata', dest='bounds_prop_metadata', action='store_false',
                        help='ACT: Disable bounds propagation metadata tracking to improve performance')

    # ACT Hybrid Zonotope Novel Features
    parser.add_argument('--relaxation_ratio', type=float, default=1.0,
                        help='ACT Hybrid Zonotope relaxation ratio: 0.0=full-precision MILP, 1.0=fully-relaxed LP, 0.0~1.0=partially-relaxed MILP+LP. Only applies to hybridz_relaxed method (hybridz_relaxed_with_bab forces 1.0)')
    parser.add_argument('--enable_generator_merging', action='store_true', default=False,
                        help='ACT innovation: Enable parallel generator merging optimization in the final linear layer (hybrid zonotope enhancement)')
    parser.add_argument('--cosine_threshold', type=float, default=0.95,
                        help='ACT innovation: Cosine similarity threshold for parallel generator detection (0.0-1.0, higher=stricter)')

    # ACT CI/CD Environment Configuration
    parser.add_argument('--ci', action='store_true', default=False,
                        help='ACT CI mode: Use scipy.linprog instead of Gurobi for LP solving (no commercial license required). Automatically enables fallback to open-source solvers when Gurobi license is unavailable')

    # ACT Specification Refinement (Branch-and-Bound) Framework
    parser.add_argument('--enable_spec_refinement', action='store_true', default=False,
                        help='ACT innovation: Enable specification refinement BaB verification. Automatically triggers when initial abstract verification returns UNKNOWN/UNSAT')
    parser.add_argument('--bab_max_depth', type=int, default=8,
                        help='ACT BaB: Maximum search depth')
    parser.add_argument('--bab_max_subproblems', type=int, default=500,
                        help='ACT BaB: Maximum number of subproblems')
    parser.add_argument('--bab_time_limit', type=float, default=300.0,
                        help='ACT BaB: Time limit in seconds')
    parser.add_argument('--bab_split_tolerance', type=float, default=1e-6,
                        help='ACT BaB: Split tolerance')
    parser.add_argument('--bab_verbose', action='store_true', default=True,
                        help='ACT BaB: Enable verbose output')

    # ================================================================================
    # EXTERNAL TOOL COMPATIBILITY PARAMETERS
    # ================================================================================
    # The following parameters are designed for compatibility with external verification
    # tools (ERAN and αβ-CROWN) to enable unified interface calls. Parameter names and
    # descriptions are adapted from:
    # 
    # - αβ-CROWN (α,β-CROWN): https://github.com/Verified-Intelligence/alpha-beta-CROWN
    #   Copyright (C) 2021-2025 The α,β-CROWN Team
    #   Licensed under BSD 3-Clause License
    #   Primary authors: Huan Zhang, Zhouxing Shi, Xiangru Zhong
    #
    # - ERAN: https://github.com/eth-sri/eran
    #   Copyright ETH Zurich
    #   Licensed under Apache 2.0 License
    # ================================================================================

    # Input Specification (adapted from αβ-CROWN specification hierarchy)
    parser.add_argument('--input_lb', type=float, nargs='+', default=None,
                        help='Lower bounds for set-based input specification (box constraints)')
    parser.add_argument('--input_ub', type=float, nargs='+', default=None,
                        help='Upper bounds for set-based input specification (box constraints)')

    # File I/O (adapted from αβ-CROWN general hierarchy)
    parser.add_argument('--root_path', type=str, default="",
                        help='Root path prefix for relative file paths')

    # Model Configuration (adapted from αβ-CROWN model hierarchy)
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to neural network model file (ONNX format supported)')
    parser.add_argument('--load_model', type=str, default=None, 
                        help='Load pretrained model from specified path (αβ-CROWN compatibility)')

    # Data Configuration (adapted from αβ-CROWN data hierarchy)
    parser.add_argument("--start", type=int, default=0, 
                        help='Start from the i-th property in specified dataset')
    parser.add_argument("--end", type=int, default=10000, 
                        help='End with the (i-1)-th property in the dataset')
    parser.add_argument("--select_instance", type=int, nargs='+', default=None,
                        help='Select specific instances to verify (list of indices)')
    parser.add_argument('--num_outputs', type=int, default=10,
                        help="Number of output classes for classification problems")
    parser.add_argument("--mean", nargs='+', type=float, default=None,
                        help='Mean values for data preprocessing normalisation (single value or per-channel list)')
    parser.add_argument("--std", nargs='+', type=float, default=None,
                        help='Standard deviation values for data preprocessing normalisation (single value or per-channel list)')
    parser.add_argument('--pkl_path', type=str, default=None,
                        help="Load verification properties from pickle file (oval20 dataset support)")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset name (mnist, cifar10, etc.) or path to CSV file")
    parser.add_argument("--anchor", type=str, default=None,
                        help="Anchor dataset path for data point anchoring in specifications")
    parser.add_argument("--filter_path", type=str, default=None,
                        help='Pickle file containing examples to skip during verification')
    parser.add_argument("--data_idx_file", type=str, default=None,
                        help='Text file with list of example IDs to run')

    # Specification Configuration (adapted from αβ-CROWN specification hierarchy)
    parser.add_argument("--spec_type", type=str, default=None, 
                        choices=['local_lp', 'local_vnnlib', 'set_vnnlib', "set_box"],
                        help='Verification specification type: "local_lp"=Lp norm around data points, "local_vnnlib"=VNNLIB with anchor points, "set_vnnlib"=set-based VNNLIB (e.g., AcasXu), "set_box"=box constraints')
    parser.add_argument("--robustness_type", type=str, default="verified-acc",
                        choices=["verified-acc", "runnerup", "clean-acc", "specify-target", "all-positive"],
                        help='Robustness verification target: "verified-acc"=verify against all labels, "runnerup"=verify against runner-up labels only')
    parser.add_argument("--norm", type=str, default=None, choices=['1', '2', 'inf'],
                        help='Lp-norm for epsilon perturbation in robustness verification')
    parser.add_argument("--epsilon", type=float, default=None,
                        help='Perturbation bound (Lp norm). If unset, dataset-specific defaults may apply')
    parser.add_argument("--epsilon_min", type=float, default=0.,
                        help='Minimum perturbation bound (typically 0.0)')
    parser.add_argument("--vnnlib_path", type=str, default=None,
                        help='Path to VNNLIB specification file (overrides Lp/robustness verification arguments)')
    parser.add_argument("--vnnlib_path_prefix", type=str, default='',
                        help='Prefix to add to VNNLIB specification paths (for correcting malformed CSV files)')
    parser.add_argument("--rhs_offset", type=float, default=None,
                        help='Offset to add to right-hand side of constraints (advanced usage)')
    # ================================================================================

    
    # ACT Torch-native Abstraction Framework Demo Parameters
    parser.add_argument('--demo_input_dim', type=int, default=3,
                        help='Input dimension for demo network (ACT torch-native abstraction framework)')
    parser.add_argument('--demo_output_dim', type=int, default=4,
                        help='Output dimension for demo network (ACT torch-native abstraction framework)')
    parser.add_argument('--demo_target_class', type=int, default=None,
                        help='Target class for robustness verification (default: auto-select, ACT torch-native abstraction framework)')
    
    # ================================================================================
    return parser