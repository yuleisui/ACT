#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#########################################################################
##   Abstract Constraint Transformer (ACT) - αβ-CROWN Verifier         ##
##                                                                     ##
##   doctormeeee (https://github.com/doctormeeee) and contributors     ##
##   Copyright (C) 2024-2025                                           ##
##                                                                     ##
#########################################################################

import os
import subprocess
from base_verifier import BaseVerifier
from dataset import Dataset
from spec import Spec
from type import SpecType

class ABCROWNVerifier(BaseVerifier):
    def __init__(self, dataset : Dataset, method, spec : Spec, device: str = 'cpu'):
        super().__init__(dataset, spec, device)
        self.method = method

    def verify(self, proof, public_inputs):
        print(self.spec.input_spec.norm)
        if self.spec.spec_type == SpecType.LOCAL_LP:
            spec_type = 'lp'
        elif self.spec.spec_type == SpecType.SET_BOX:
            spec_type = 'box'

        elif self.spec.spec_type == SpecType.SET_VNNLIB or self.spec.spec_type == SpecType.LOCAL_VNNLIB:
            spec_type = 'bound'
        else:
            raise ValueError(f"Unsupported specification type for ABCROWN: {self.spec.spec_type}. Supported types are 'local_lp', 'set_box', 'set_vnnlib'.")

        netname = self.spec.model.model_path
        if netname is not None and not os.path.isabs(netname):
            netname = os.path.abspath(netname)

        norm = float(self.spec.input_spec.norm.value)

        vnnlib_path = self.spec.input_spec.vnnlib_path
        if vnnlib_path is not None and not os.path.isabs(vnnlib_path):
            vnnlib_path = os.path.abspath(vnnlib_path)

        args_dict = {
            "config": "empty_config.yaml",

            "device": self.device,
            "dataset": self.dataset.dataset_path.upper(),
            "start" : self.dataset.start,
            "end" : self.dataset.end,
            "num_outputs"  : self.dataset.num_outputs,
            "mean" : self.dataset.mean,
            "std"  : self.dataset.std,
            "spec_type" : spec_type,
            "norm" : norm,
            "epsilon" : self.spec.input_spec.epsilon,
            "vnnlib_path" : vnnlib_path
        }

        if netname.endswith(".onnx"):
            args_dict["onnx_path"] = netname
        elif netname.endswith(".pth"):
            args_dict["model"] = netname
        else:
            raise ValueError(f"Unsupported model file type: {netname}")


        args_list = []
        for k, v in args_dict.items():
            if v is not None:
                args_list.append(f"--{k}")
                if isinstance(v, list) and k in ['mean', 'std']:
                    for val in v:
                        args_list.append(str(val))
                else:
                    args_list.append(str(v))

        print("aruguments checking for abcrown")
        print(args_dict)

        conda_env_name = "act-abcrown"
        
        # Dynamic path detection for ABCROWN runner
        current_dir = os.path.dirname(os.path.abspath(__file__))
        verifier_path = os.path.abspath(current_dir) 
        
        cmd = ["conda", "run", "--no-capture-output", "-n", conda_env_name, "python3", "-u", "abcrown_runner.py", self.method] + args_list

        try:
            print("[ABCROWNVerifier] ABCROWN verifier running now, please wait for result generation.")
            print(f"[ABCROWNVerifier] Command: {' '.join(cmd)}")
            print(f"[ABCROWNVerifier] Working directory: {verifier_path}")
            
            # Use real-time output instead of waiting for completion
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                cwd=verifier_path
            )
            
            # Print output in real-time
            print("[ABCROWNVerifier] Real-time output:")
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
            print("[ABCROWNVerifier] ABCROWN verification completed successfully")
            return return_code
        except subprocess.CalledProcessError as e:
            print("[ABCROWNVerifier] ABCROWN execution failed:")
            print(f"Return code: {e.returncode}")
            raise RuntimeError("ABCROWN verification failed.") from e
        except Exception as e:
            print(f"[ABCROWNVerifier] Unexpected error: {e}")
            if process.poll() is None:
                process.terminate()
            raise RuntimeError("ABCROWN verification failed.") from e