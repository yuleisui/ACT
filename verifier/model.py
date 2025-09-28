#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#########################################################################
##   Abstract Constraint Transformer (ACT) - Network Model Parser      ##
##                                                                     ##
##   doctormeeee (https://github.com/doctormeeee) and contributors     ##
##   Copyright (C) 2024-2025                                           ##
##                                                                     ##
#########################################################################

import os
import torch
import onnx
import tensorflow as tf
from onnx2pytorch import ConvertModel
import subprocess

class Model:
    def __init__(self, model_path: str = None, device: str = 'cpu'):
        self.model_path = model_path
        self.device = device
        self.pytorch_model = None
        self._onnx_input_shape = None
        self._tf_input_shape = None

        if model_path is not None:
            self._auto_load_and_convert()

    def _auto_load_and_convert(self):
        if os.path.isfile(self.model_path):
            if self.model_path.endswith('.pt') or self.model_path.endswith('.pth'):
                self.load_pt_model()
            elif self.model_path.endswith('.onnx'):
                self._onnx_input_shape = self._get_onnx_input_shape(self.model_path)
                self.convert_onnx_to_pytorch()
            elif self.model_path.endswith('.h5') or self.model_path.endswith('.keras'):
                self._tf_input_shape = self._get_tf_input_shape(self.model_path)
                self.convert_tf_to_pytorch()
            else:
                raise ValueError(f"Unknown model file type: {self.model_path}")
        elif os.path.isdir(self.model_path):
            if os.path.exists(os.path.join(self.model_path, 'saved_model.pb')):
                self.convert_tf_to_pytorch()
            else:
                raise ValueError(f"Unknown model directory: {self.model_path}")
        else:
            raise FileNotFoundError(f"Model path not found: {self.model_path}")

    def _get_onnx_input_shape(self, onnx_path):
        onnx_model = onnx.load(onnx_path)
        input_tensor = onnx_model.graph.input[0]
        shape = []
        for dim in input_tensor.type.tensor_type.shape.dim:

            shape.append(dim.dim_value if dim.dim_value > 0 else 1)
        return tuple(shape)

    def _get_tf_input_shape(self, tf_path):
        try:
            if tf_path.endswith('.h5') or tf_path.endswith('.keras'):
                model = tf.keras.models.load_model(tf_path)
                return tuple(model.input_shape)
            elif os.path.isdir(tf_path) and os.path.exists(os.path.join(tf_path, 'saved_model.pb')):
                model = tf.keras.models.load_model(tf_path)
                return tuple(model.input_shape)
        except Exception as e:
            print(f"[WARN] Failed to get TF input shape: {e}")
        return None

    def get_expected_input_shape(self):

        if self._onnx_input_shape is not None:
            return self._onnx_input_shape
        if self._tf_input_shape is not None:
            return self._tf_input_shape
        if hasattr(self.pytorch_model, 'input_shape'):
            return self.pytorch_model.input_shape
        for shape in [(1, 1, 28, 28), (1, 3, 32, 32), (1, 5)]:
            try:
                dummy = torch.zeros(*shape)
                self.pytorch_model(dummy)
                return shape
            except Exception:
                continue
        raise RuntimeError("Cannot infer model input shape automatically.")

    def load_pt_model(self):
        try:
            self.pytorch_model = torch.load(self.model_path, map_location=self.device)
            self.pytorch_model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load PyTorch model: {e}")

    def convert_onnx_to_pytorch(self):
        try:
            onnx_model = onnx.load(self.model_path)
            self.pytorch_model = ConvertModel(onnx_model, experimental=True)
            self.pytorch_model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to convert ONNX model to PyTorch: {e}")

    def convert_tf_to_pytorch(self, onnx_path="tmp_model.onnx"):
        model_type = None
        if os.path.isfile(self.model_path):
            if self.model_path.endswith('.h5') or self.model_path.endswith('.keras'):
                model_type = 'keras'
        elif os.path.isdir(self.model_path):
            if os.path.exists(os.path.join(self.model_path, 'saved_model.pb')):
                model_type = 'saved_model'
        else:
            raise ValueError("Unknown model path type.")

        if model_type == 'keras':
            cmd = [
                "python", "-m", "tf2onnx.convert",
                "--keras", self.model_path,
                "--output", onnx_path
            ]
        elif model_type == 'saved_model':
            cmd = [
                "python", "-m", "tf2onnx.convert",
                "--saved-model", self.model_path,
                "--output", onnx_path
            ]
        else:
            raise ValueError("Model type not supported for conversion.")

        print(f"[INFO] Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

        try:
            onnx_model = onnx.load(onnx_path)
            self.pytorch_model = ConvertModel(onnx_model, experimental=True)
            self.pytorch_model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to convert TF/Keras model to PyTorch: {e}")