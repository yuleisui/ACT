#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Runner for external wrapper verifiers (moved from act/main.py).

This file is a near-copy of the original `act/main.py` entrypoint but placed
under `act/wrapper_exts/` and renamed to `ext_runner.py`. Other modules in the
repository that invoked the old entrypoint should be updated to import or
invoke this module instead.
"""
from __future__ import annotations

import sys
import time
import os
import configparser

# Import commendline paser
from act.util.options import get_parser
from act.util.path_config import get_config_root

# Import ACT modules
from act.wrapper_exts.ext_config import Model, Dataset, Spec, InputSpec, OutputSpec, VerifyResult
from act.wrapper_exts.eran.eran_verifier import ERANVerifier
from act.wrapper_exts.abcrown.abcrown_verifier import abCrownVerifier


def load_verifier_default_configs(verifier, method, dataset):
    if not verifier or not dataset:
        return {}
    
    config_root = get_config_root()
    
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
                             Please use \'mnist\', \'cifar10\', \'cifar\', or \'eran\'.")
        
        if dataset.dataset_path == "cifar10":
            print("⚠️  Dataset name 'cifar10' is deprecated for αβ-CROWN verifier, using 'cifar' instead")
            dataset.dataset_path = "cifar"

        if args_dict["enable_spec_refinement"]:
            print("⚠️  abCrown verifier is an external verifier, does not support native specification refinement BaB, automatically disabled")
        verifier = abCrownVerifier(method, spec)
        verifier.verify(proof=None, public_inputs=None)

    else:
        raise ValueError(f"Unsupported verifier: {verifier_type}. Supported verifiers: 'eran', 'abcrown'.")


if __name__ == "__main__":
    main()
