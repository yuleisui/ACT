#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#########################################################################
##   Abstract Constraint Transformer (ACT) - ERAN Verifier           ##
##                                                                     ##
##   Copyright (C) 2024-2025 ACT Development Team                      ##
##                                                                     ##
#########################################################################

import os
import subprocess
import torch
from base_verifier import BaseVerifier
from dataset import Dataset
from spec import Spec

class ERANVerifier(BaseVerifier):
    def __init__(self, dataset : Dataset, method, spec : Spec, device: str = 'cpu'):
        super().__init__(dataset, spec, device)

        self.method = method

        print(f"🔍 [ERAN DEBUG] Input bounds info:")
        print(f"input_lb shape: {self.input_lb.shape}")
        print(f"input_ub shape: {self.input_ub.shape}")
        print(f"input_lb unique values: {len(torch.unique(self.input_lb.view(-1)))}")
        print(f"input_ub unique values: {len(torch.unique(self.input_ub.view(-1)))}")
        print(f"input_lb range: [{self.input_lb.min():.6f}, {self.input_lb.max():.6f}]")
        print(f"input_ub range: [{self.input_ub.min():.6f}, {self.input_ub.max():.6f}]")
        print(f"input_lb first 10 values: {self.input_lb.view(-1)[:10].tolist()}")
        print(f"input_ub first 10 values: {self.input_ub.view(-1)[:10].tolist()}")
        print(f"Dataset input center shape: {self.dataset.input_center.shape if hasattr(self.dataset, 'input_center') and self.dataset.input_center is not None else 'None'}")
        if hasattr(self.dataset, 'input_center') and self.dataset.input_center is not None:
            print(f"Dataset input center unique values: {len(torch.unique(self.dataset.input_center.view(-1)))}")
            print(f"Dataset input center first 10 values: {self.dataset.input_center.view(-1)[:10].tolist()}")
        print(f"🔍 [ERAN DEBUG] End of input bounds info")
        print("="*80)

    def verify(self, proof, public_inputs):

        netname = self.spec.model.model_path
        if netname is not None and not os.path.isabs(netname):
            netname = os.path.abspath(netname)

        vnnlib_path = self.spec.input_spec.vnnlib_path
        if vnnlib_path is not None and not os.path.isabs(vnnlib_path):
            vnnlib_path = os.path.abspath(vnnlib_path)
        args_dict = {

            "netname" : netname,
            "epsilon" : self.spec.input_spec.epsilon,
            "domain" : self.method,
            "vnn_lib_spec" : vnnlib_path,
            "dataset": self.dataset.dataset_path,
            "from_test" : self.dataset.start,
            "num_tests" : self.dataset.end - self.dataset.start,
            "mean" : self.dataset.mean,
            "std"  : self.dataset.std,
            "t-norm" : self.spec.input_spec.norm.value,
        }

        args_list = []
        for k, v in args_dict.items():
            if v is not None:
                args_list.append(f"--{k}")

                if isinstance(v, list) and k in ['mean', 'std']:

                    for val in v:
                        args_list.append(str(val))
                else:
                    args_list.append(str(v))

        conda_env_name = "act-eran"
        
        # Dynamic path detection for ERAN runner
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.join(current_dir, '..')
        eran_tf_verify_path = os.path.join(project_root, 'modules', 'eran', 'tf_verify')
        eran_tf_verify_path = os.path.abspath(eran_tf_verify_path)

        cmd = ["conda", "run", "--no-capture-output", "-n", conda_env_name, "python3", "-u", "__main__.py"] + args_list

        try:
            print("[ERANVerifier] ERAN verifier running now, please wait for result generation.")
            print(f"[ERANVerifier] Command: {' '.join(cmd)}")
            print(f"[ERANVerifier] Working directory: {eran_tf_verify_path}")
            
            # Use real-time output instead of waiting for completion
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                cwd=eran_tf_verify_path
            )
            
            # Print output in real-time
            print("[ERANVerifier] Real-time output:")
            print("-" * 60)
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(output.strip())
            
            return_code = process.poll()
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, cmd)
            print("[ERANVerifier] ERAN verification completed successfully")
            return return_code
        except subprocess.CalledProcessError as e:
            print("[ERANVerifier] ERAN execution failed:")
            print(f"Return code: {e.returncode}")
            raise RuntimeError("ERAN verification failed.") from e
        except Exception as e:
            print(f"[ERANVerifier] Unexpected error: {e}")
            if process.poll() is None:
                process.terminate()
            raise RuntimeError("ERAN verification failed.") from e