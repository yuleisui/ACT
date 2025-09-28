#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#########################################################################
##   Abstract Constraint Transformer (ACT) - Main Entry Point          ##
##                                                                     ##
##   doctormeeee (https://github.com/doctormeeee) and contributors     ##
##   Copyright (C) 2024-2025                                           ##
##                                                                     ##
#########################################################################

import argparse
import sys
import time
from model import Model
from dataset import Dataset
from spec import Spec, InputSpec, OutputSpec
from type import SpecType, VerificationStatus
from eran_verifier import ERANVerifier
from abcrown_verifier import ABCROWNVerifier
from interval_verifier import IntervalVerifier
from hybridz_verifier import HybridZonotopeVerifier

def main():
    parser = argparse.ArgumentParser(description='Abstract Constraint Transformer (ACT) - Unified Neural Network Verification Framework')
    
    # ================================================================================
    # ACT NATIVE PARAMETERS
    # ================================================================================
    
    # ACT Core Verifier Selection
    parser.add_argument('--verifier', type=str, default=None, 
                        choices=['interval', 'eran', 'abcrown', 'hybridz'],
                        help='Backend verification engine. "interval": ACT native interval analysis, "eran": ERAN external verifier, "abcrown": αβ-CROWN external verifier, "hybridz": ACT novel hybrid zonotope verifier')
    parser.add_argument('--method', type=str, default=None, 
                        help='Verification method. ERAN: [deepzono, refinezono, deeppoly, refinepoly], αβ-CROWN: [alpha, beta, alpha_beta], ACT-HybridZ: [hybridz, hybridz_relaxed, hybridz_relaxed_with_bab], ACT-Interval: [interval]')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                        help='Computation device (cpu or cuda)')

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
    #
    # These parameters enable seamless integration and unified command-line interface
    # across different verification backends while maintaining ACT's native capabilities.
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
                        help="Load verification properties from pickle file (legacy oval20 dataset support)")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset name (mnist, cifar10, etc.) or path to CSV file")
    parser.add_argument("--anchor", type=str, default=None,
                        help="Anchor dataset path for data point anchoring in specifications")
    parser.add_argument("--filter_path", type=str, default=None,
                        help='Pickle file containing examples to skip during verification')
    parser.add_argument("--data_idx_file", type=str, default=None,
                        help='Text file with list of example IDs to run')

    # Specification Configuration (adapted from αβ-CROWN specification hierarchy)
    parser.add_argument("--spec_type", type=str, default='local_lp', 
                        choices=['local_lp', 'local_vnnlib', 'set_vnnlib', "set_box"],
                        help='Verification specification type: "local_lp"=Lp norm around data points, "local_vnnlib"=VNNLIB with anchor points, "set_vnnlib"=set-based VNNLIB (e.g., AcasXu), "set_box"=box constraints')
    parser.add_argument("--robustness_type", type=str, default="verified-acc",
                        choices=["verified-acc", "runnerup", "clean-acc", "specify-target", "all-positive"],
                        help='Robustness verification target: "verified-acc"=verify against all labels, "runnerup"=verify against runner-up labels only')
    parser.add_argument("--norm", type=str, default='inf', choices=['1', '2', 'inf'],
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

    parsed_args = parser.parse_args(sys.argv[1:])
    args_dict = vars(parsed_args)

    if args_dict["mean"] is None:

        if args_dict["dataset"] is None or (args_dict["dataset"] is not None and args_dict["dataset"].endswith('.csv')):
            args_dict["mean"] = [0.0]
        elif args_dict["dataset"] == 'mnist':
            args_dict["mean"] = [0.1307]
        elif args_dict["dataset"] in ['cifar', 'cifar10']:
            args_dict["mean"] = [0.4914, 0.4822, 0.4465]
        else:
            args_dict["mean"] = [0.0]

    if args_dict["std"] is None:

        if args_dict["dataset"] is None or (args_dict["dataset"] is not None and args_dict["dataset"].endswith('.csv')):
            args_dict["std"] = [1.0]
        elif args_dict["dataset"] == 'mnist':
            args_dict["std"] = [0.3081]
        elif args_dict["dataset"] in ['cifar', 'cifar10']:
            args_dict["std"] = [0.2023, 0.1994, 0.2010]
        else:
            args_dict["std"] = [1.0]

    print(f"Using mean: {args_dict['mean']}, std: {args_dict['std']}")

    model = Model(model_path=args_dict["model_path"],
                  device = args_dict["device"])

    # For VNNLIB spec types, use vnnlib_path instead of dataset path
    dataset_path_for_init = args_dict["dataset"]
    if args_dict["spec_type"] in ["local_vnnlib", "set_vnnlib"] and args_dict["vnnlib_path"] is not None:
        dataset_path_for_init = args_dict["vnnlib_path"]
        print(f"🔍 Using VNNLIB file as dataset path: {dataset_path_for_init}")

    dataset = Dataset(dataset_path=dataset_path_for_init,
                      anchor_csv_path=args_dict["anchor"],
                      device=args_dict["device"],
                      spec_type=args_dict["spec_type"],
                      start=args_dict["start"],
                      end=args_dict["end"],
                      num_outputs=args_dict["num_outputs"],
                      mean=args_dict["mean"],
                      std=args_dict["std"],
                      preprocess=True)

    input_spec = InputSpec(dataset = dataset,
                           norm = args_dict["norm"],
                           epsilon = args_dict["epsilon"],
                           vnnlib_path = args_dict["vnnlib_path"])

    output_spec = OutputSpec(dataset = dataset)

    spec = Spec(input_spec=input_spec,
                output_spec=output_spec, model=model)

    verifier_type = args_dict["verifier"]

    method = args_dict["method"]

    if verifier_type == 'eran' and method in ['deepzono', 'refinezono', 'deeppoly', 'refinepoly']:
        if dataset.dataset_path not in ["mnist", "cifar10", "acasxu"]:
            raise ValueError(f"ERAN verifier with method {method} is not supported for dataset {dataset.dataset_path}. \
                             Please use \'mnist\', \'cifar10\' or \'acasxu\'.")
        if args_dict["enable_spec_refinement"]:
            print("⚠️  ERAN verifier is an external verifier, does not support specification refinement BaB, automatically disabled")
        verifier = ERANVerifier(dataset, method, spec)
        verifier.verify(proof=None, public_inputs=None)

    elif verifier_type == 'abcrown' and method in ['alpha', 'beta', 'alpha_beta']:

        if dataset.dataset_path not in ["mnist", "cifar", "eran"]:
            raise ValueError(f"abCrown verifier with method {method} is not supported for dataset {dataset.dataset_path}. \
                             Please use \'mnist\', \'cifar\', 'eran'.")
        if args_dict["enable_spec_refinement"]:
            print("⚠️  ABCROWN verifier is an external verifier, does not support native specification refinement BaB, automatically disabled")
        verifier = ABCROWNVerifier(dataset, method, spec)
        verifier.verify(proof=None, public_inputs=None)

    elif verifier_type == 'interval':
        if method != 'interval':
            raise ValueError(f"Interval verifier only supports 'interval' method, got {method}.")
        verifier = IntervalVerifier(dataset, method, spec)

        if args_dict["enable_spec_refinement"]:
            print("Enabling specification refinement BaB verification")
            verifier.bab_config.update({
                'enabled': True,
                'max_depth': args_dict["bab_max_depth"],
                'max_subproblems': args_dict["bab_max_subproblems"],
                'time_limit': args_dict["bab_time_limit"],
                'split_tolerance': args_dict["bab_split_tolerance"],
                'verbose': args_dict["bab_verbose"]
            })
            print(f"Maximum depth: {args_dict['bab_max_depth']}")
            print(f"Maximum subproblems: {args_dict['bab_max_subproblems']}")
            print(f"Time limit: {args_dict['bab_time_limit']} seconds")

        else:
            print(" Specification refinement BaB verification disabled")
            verifier.bab_config['enabled'] = False

        result = verifier.verify()
        if result == VerificationStatus.SAT:
            print("✅ The property is satisfied.")
        elif result == VerificationStatus.UNSAT:
            print("❌ The property is not satisfied.")
        else:
            print("❓ The property status is unknown.")

    elif verifier_type == 'hybridz':
        if method not in ['interval', 'hybridz', 'hybridz_relaxed', 'hybridz_relaxed_with_bab']:
            raise ValueError(f"Hybrid Zonotope verifier only supports methods 'interval', 'hybridz', 'hybridz_relaxed', 'hybridz_relaxed_with_bab', got {method}.")

        if method == 'hybridz_relaxed_with_bab':

            relaxation_ratio = 1.0
            if args_dict["relaxation_ratio"] != 1.0:
                print(f"⚠️  hybridz_relaxed_with_bab method forces relaxation_ratio=1.0 (full relaxed LP), ignoring user setting of {args_dict['relaxation_ratio']}")
            else:
                print(f"hybridz_relaxed_with_bab method using relaxation_ratio=1.0 (full relaxed LP)")
        else:

            relaxation_ratio = args_dict["relaxation_ratio"]
            if method == 'hybridz_relaxed':
                print(f"hybridz_relaxed method using relaxation_ratio={relaxation_ratio}")

        verifier = HybridZonotopeVerifier(dataset, method, spec, args_dict["device"],
                                          relaxation_ratio,
                                          args_dict["enable_generator_merging"],
                                          args_dict["cosine_threshold"],
                                          ci_mode=args_dict["ci"])

        if args_dict["enable_spec_refinement"]:
            if method == 'hybridz_relaxed_with_bab':
                print("Enabling specification refinement BaB verification (hybridz_relaxed_with_bab)")
                verifier.bab_config.update({
                    'enabled': True,
                    'max_depth': args_dict["bab_max_depth"],
                    'max_subproblems': args_dict["bab_max_subproblems"],
                    'time_limit': args_dict["bab_time_limit"],
                    'split_tolerance': args_dict["bab_split_tolerance"],
                    'verbose': args_dict["bab_verbose"]
                })
                print(f"Maximum depth: {args_dict['bab_max_depth']}")
                print(f"Maximum subproblems: {args_dict['bab_max_subproblems']}")
                print(f"Time limit: {args_dict['bab_time_limit']} seconds")

            else:
                print(f"⚠️  Specification refinement BaB only supports hybridz_relaxed_with_bab method, current method is {method}, automatically disabled")
                verifier.bab_config['enabled'] = False
        else:
            print(" Specification refinement BaB verification disabled")
            verifier.bab_config['enabled'] = False

        start_time = time.time()
        result = verifier.verify()
        end_time = time.time()
        print(f"⏱️  Total verification time: {end_time - start_time:.2f} seconds")
        if result == VerificationStatus.SAT:
            print("✅ The property is satisfied.")
        elif result == VerificationStatus.UNSAT:
            print("❌ The property is not satisfied.")
        else:
            print("❓ The property status is unknown.")
    else:
        raise ValueError(f"Unsupported verifier: {verifier_type}. Supported verifiers: 'eran', 'abcrown', 'interval', 'hybridz'.")

if __name__ == "__main__":
    main()