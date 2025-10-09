#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##############################################################################
##   Abstract Constraint Transformer (ACT) - Main Entry Point               ##
##                                                                          ##
##   doctormeeee (https://github.com/doctormeeee) and contributors          ##
##   Copyright (C) 2024-2025                                                ##
##                                                                          ##
##   This program integrates multiple neural network verification           ##
##   tools and provides novel hybrid zonotope verification methods.         ##
##                                                                          ##
##   External tool compatibility parameters are adapted from:               ##
##   - α,β-CROWN: https://github.com/Verified-Intelligence/alpha-beta-CROWN ##
##     Copyright (C) 2021-2025 The α,β-CROWN Team                           ##
##     Licensed under BSD 3-Clause License                                  ##
##   - ERAN: https://github.com/eth-sri/eran                                ##
##     Copyright ETH Zurich, Licensed under Apache 2.0 License              ##
##                                                                          ##
##   This integration enables unified command-line interface across         ##
##   different verification backends while maintaining ACT's native         ##
##   capabilities and novel contributions.                                  ##
##                                                                          ##
##############################################################################
import sys
import time
import os
import configparser

# Import commendline paser
from util.options import get_parser

# Import ACT modules
from input_parser.model import Model
from input_parser.dataset import Dataset
from input_parser.spec import Spec, InputSpec, OutputSpec
from input_parser.type import VerifyResult
from abstract_constraint_solver.eran.eran_verifier import ERANVerifier
from abstract_constraint_solver.abcrown.abcrown_verifier import abCrownVerifier
from abstract_constraint_solver.interval.base_verifier import BaseVerifier
from abstract_constraint_solver.hybridz.hybridz_verifier import HybridZonotopeVerifier

def load_verifier_default_configs(verifier, method, dataset):
    if not verifier or not dataset:
        return {}
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    config_root = os.path.join(project_root, 'configs')
    config_root = os.path.abspath(config_root)
    
    config_file = os.path.join(config_root, f"{verifier}_defaults.ini")
    if not os.path.exists(config_file):
        return {}
    
    config = configparser.ConfigParser()
    config.read(config_file)
    
    defaults = {}
    
    # Determine dataset section name based on dataset type
    if dataset in ['mnist', 'cifar', 'cifar10']:
        dataset_section = dataset.upper()
    elif dataset.endswith('.csv'):
        dataset_section = 'CSV'
    elif dataset.endswith('.vnnlib'):
        dataset_section = 'VNNLIB'
    else:
        dataset_section = dataset.upper()  # Fallback for other named datasets
    
    # Load non-method-specific default configs
    if dataset_section in config:
        print(f"Loading {verifier} defaults for dataset type: {dataset_section}")
        for key, value in config[dataset_section].items():
            defaults[key] = _parse_config_value(key, value)
    
    # method-specific default configs (only for HybridZ currently)
    if verifier == 'hybridz':
        method_section = f"{dataset_section}.{method.upper()}"  # Use dataset_section instead of dataset
        if method_section in config:
            print(f"Loading {verifier} method-specific defaults: {method_section}")
            for key, value in config[method_section].items():
                defaults[key] = _parse_config_value(key, value)  # Override dataset defaults
    
    return defaults

def _parse_config_value(key, value):
    """Parse a single config value based on its key"""
    if key in ['mean', 'std']:
        value_clean = value.strip('[]"\'')
        if ',' in value_clean:
            return [float(v.strip()) for v in value_clean.split(',')]
        else:
            return [float(value_clean)]
    elif key in ['enable_spec_refinement']: 
        return value.lower() == 'true'
    elif key in ['relaxation_ratio', 'bab_max_depth', 'bab_max_subproblems', 'bab_time_limit']:
        try:
            return float(value) if '.' in value else int(value)
        except ValueError:
            return value
    else:
        return value.strip('"\'')

def main():
    
    parser = get_parser()
    parsed_args = parser.parse_args(sys.argv[1:])
    args_dict = vars(parsed_args)

    # Load and apply default configurations from ini files
    defaults = load_verifier_default_configs(args_dict.get('verifier'), args_dict.get('method'), args_dict.get('dataset'))
    for key, value in defaults.items():
        if args_dict.get(key) is None:  # Only set if not provided by user
            args_dict[key] = value
            print(f"Using default {key}: {value}")

    # Legacy fallback for missing mean/std (safety net for datasets not in config)
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
        print(f"Using VNNLIB file as dataset path: {dataset_path_for_init}")

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

    spec = Spec(model=model,
                input_spec=input_spec,
                output_spec=output_spec)

    verifier_type = args_dict["verifier"]

    method = args_dict["method"]

    if verifier_type == 'eran' and method in ['deepzono', 'refinezono', 'deeppoly', 'refinepoly']:
        if dataset.dataset_path not in ["mnist", "cifar10", "acasxu"]:
            raise ValueError(f"ERAN verifier with method {method} is not supported for dataset {dataset.dataset_path}. \
                             Please use \'mnist\', \'cifar10\' or \'acasxu\'.")
        if args_dict["enable_spec_refinement"]:
            print("⚠️  ERAN verifier is an external verifier, does not support specification refinement BaB, automatically disabled")
        verifier = ERANVerifier(method, spec)
        verifier.verify(proof=None, public_inputs=None)

    elif verifier_type == 'abcrown' and method in ['alpha', 'beta', 'alpha_beta']: # TODO

        if dataset.dataset_path not in ["mnist", "cifar", "cifar10", "eran"]:
            raise ValueError(f"abCrown verifier with method {method} is not supported for dataset {dataset.dataset_path}. \
                             Please use \'mnist\', \'cifar\', \'cifar10\', or \'eran\'.")
        
        if dataset.dataset_path == "cifar10":
            print("⚠️  Dataset name 'cifar10' is deprecated for αβ-CROWN verifier, using 'cifar' instead")
            dataset.dataset_path = "cifar"

        if args_dict["enable_spec_refinement"]:
            print("⚠️  abCrown verifier is an external verifier, does not support native specification refinement BaB, automatically disabled")
        verifier = abCrownVerifier(method, spec)
        verifier.verify(proof=None, public_inputs=None)

    elif verifier_type == 'interval':
        if method != 'interval':
            raise ValueError(f"Interval verifier only supports 'interval' method, got {method}.")
        verifier = BaseVerifier(spec)

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
        if result == VerifyResult.SAT:
            print("✅ The property is satisfied.")
        elif result == VerifyResult.UNSAT:
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

        verifier = HybridZonotopeVerifier(method, spec, args_dict["device"],
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
        if result == VerifyResult.SAT:
            print("✅ The property is satisfied.")
        elif result == VerifyResult.UNSAT:
            print("❌ The property is not satisfied.")
        else:
            print("❓ The property status is unknown.")
    else:
        raise ValueError(f"Unsupported verifier: {verifier_type}. Supported verifiers: 'eran', 'abcrown', 'interval', 'hybridz'.")

if __name__ == "__main__":
    main()