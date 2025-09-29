#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#########################################################################
##   Abstract Constraint Transformer (ACT) - Specification Classes     ##
##                                                                     ##
##   doctormeeee (https://github.com/doctormeeee) and contributors     ##
##   Copyright (C) 2024-2025                                           ##
##                                                                     ##
#########################################################################

from typing import Optional
import torch
from dataset import Dataset
from model import Model
from type import LPNormType, SpecType, VerificationStatus

class BaseSpec:
    def __init__(self, dataset : Dataset = None, model : Model = None, status: VerificationStatus = VerificationStatus.UNKNOWN):
        self.dataset = dataset
        self.model = model

class InputSpec(BaseSpec):
    def __init__(self,
                 dataset : Dataset,
                 norm: Optional[LPNormType] = None,
                 epsilon: Optional[float] = None,
                 vnnlib_path: str = None,
                 ):

        super().__init__(dataset)
        if isinstance(self.dataset.spec_type, str):
            self.spec_type = SpecType(self.dataset.spec_type.lower())
        else:
            self.spec_type = self.dataset.spec_type

        if isinstance(norm, str):
            norm = LPNormType(norm.lower())

        self.norm = norm
        self.input_center = None
        self.epsilon = epsilon
        self.vnnlib_path = vnnlib_path
        self.input_lb = None
        self.input_ub = None

        if self.input_lb is not None and self.input_ub is not None:
            self._input_validation()

        if self.spec_type not in [SpecType.LOCAL_LP, SpecType.LOCAL_VNNLIB, SpecType.SET_VNNLIB, SpecType.SET_BOX]:
            raise ValueError(f"Unsupported specification type: {self.spec_type}")
        if self.spec_type == SpecType.LOCAL_LP and (norm is None or epsilon is None):
            raise ValueError("Norm type and epsilon value must be specified for local LP specifications")

        if self.spec_type == SpecType.LOCAL_LP:

            print(f"📊 [InputSpec] Epsilon = {epsilon}")

            if self.norm == LPNormType.LINF:
                if dataset.preprocess and hasattr(self.dataset, 'mean') and hasattr(self.dataset, 'std') and self.dataset.mean is not None and self.dataset.std is not None:

                    self.input_center = self.dataset.input_center

                    if isinstance(self.dataset.mean, list) and isinstance(self.dataset.std, list):

                        mean_tensor = torch.tensor(self.dataset.mean, dtype=torch.float32, device=self.dataset.input_center.device)
                        std_tensor = torch.tensor(self.dataset.std, dtype=torch.float32, device=self.dataset.input_center.device)

                        if len(self.dataset.input_center.shape) == 4:
                            mean_tensor = mean_tensor.view(1, -1, 1, 1)
                            std_tensor = std_tensor.view(1, -1, 1, 1)
                        elif len(self.dataset.input_center.shape) == 3:
                            mean_tensor = mean_tensor.view(-1, 1, 1)
                            std_tensor = std_tensor.view(-1, 1, 1)
                        elif len(self.dataset.input_center.shape) == 1:

                            C = len(self.dataset.mean)
                            pixels_per_channel = self.dataset.input_center.shape[0] // C
                            mean_tensor = mean_tensor.repeat_interleave(pixels_per_channel)
                            std_tensor = std_tensor.repeat_interleave(pixels_per_channel)
                        elif len(self.dataset.input_center.shape) == 2:
                            C = len(self.dataset.mean)
                            pixels_per_channel = self.dataset.input_center.shape[1] // C
                            mean_tensor = mean_tensor.repeat_interleave(pixels_per_channel).unsqueeze(0)
                            std_tensor = std_tensor.repeat_interleave(pixels_per_channel).unsqueeze(0)

                        original_pixels = self.dataset.input_center * std_tensor + mean_tensor
                    else:

                        mean = self.dataset.mean[0] if isinstance(self.dataset.mean, list) else self.dataset.mean
                        std = self.dataset.std[0] if isinstance(self.dataset.std, list) else self.dataset.std
                        original_pixels = self.dataset.input_center * std + mean

                    print(f"📊 Inverse normalized pixel value range: [{original_pixels.min():.6f}, {original_pixels.max():.6f}]")

                    lb_raw = torch.clamp(original_pixels - epsilon, 0.0, 1.0)
                    ub_raw = torch.clamp(original_pixels + epsilon, 0.0, 1.0)

                    print(f"📊 [0,1] space perturbation+clip range: LB=[{lb_raw.min():.6f}, {lb_raw.max():.6f}], UB=[{ub_raw.min():.6f}, {ub_raw.max():.6f}]")

                    if isinstance(self.dataset.mean, list) and isinstance(self.dataset.std, list):

                        self.input_lb = (lb_raw - mean_tensor) / std_tensor
                        self.input_ub = (ub_raw - mean_tensor) / std_tensor
                    else:

                        self.input_lb = (lb_raw - mean) / std
                        self.input_ub = (ub_raw - mean) / std

                    print(f"📊 Final normalized bounds: LB=[{self.input_lb.min():.6f}, {self.input_lb.max():.6f}], UB=[{self.input_ub.min():.6f}, {self.input_ub.max():.6f}]")

                    print(f"📊 Original center range: [{self.input_center.min():.6f}, {self.input_center.max():.6f}]")
                    print(f"📊 Perturbation interval width range: [{(self.input_ub - self.input_lb).min():.6f}, {(self.input_ub - self.input_lb).max():.6f}]")

                    original_lb = original_pixels - epsilon
                    original_ub = original_pixels + epsilon
                    clipped_lb = (original_lb < 0.0).sum().item()
                    clipped_ub = (original_ub > 1.0).sum().item()
                    total_pixels = original_pixels.numel()

                    print(f"📊 Physical clip statistics: LB clipped={clipped_lb}/{total_pixels} ({clipped_lb/total_pixels*100:.2f}%), UB clipped={clipped_ub}/{total_pixels} ({clipped_ub/total_pixels*100:.2f}%)")

                else:

                    self.input_center = self.dataset.input_center
                    original_pixels = self.dataset.input_center
                    self.input_lb = torch.clamp(original_pixels - epsilon, 0.0, 1.0)
                    self.input_ub = torch.clamp(original_pixels + epsilon, 0.0, 1.0)
                    self.input_center = (self.input_lb + self.input_ub) / 2.0

        elif self.spec_type == SpecType.LOCAL_VNNLIB:
            self.input_center = self.dataset.input_center
            self.input_lb = self.dataset.input_lb
            self.input_ub = self.dataset.input_ub

        elif self.spec_type in [SpecType.SET_VNNLIB, SpecType.SET_BOX]:
            self.input_lb = self.dataset.input_lb
            self.input_ub = self.dataset.input_ub
            # Calculate input_center as the midpoint of bounds
            self.input_center = (self.input_lb + self.input_ub) / 2.0

        else:
            raise ValueError(f"Unsupported spec type: {self.spec_type}")

    def _input_validation(self):
        if not torch.is_tensor(self.input_lb) or not torch.is_tensor(self.input_ub):
            raise ValueError("Input bounds must be torch tensors")
        if self.input_lb.shape != self.input_ub.shape:
            raise ValueError("Input bounds must have the same shape")
        if not torch.all(self.input_lb <= self.input_ub):
            raise ValueError("Lower bounds must be less than or equal to upper bounds")

    def _apply_preprocessing(self, tensor: torch.Tensor) -> torch.Tensor:
        mean, std = self.preprocessing.get("mean"), self.preprocessing.get("std")
        if mean is not None and std is not None:
            return (tensor - mean) / std
        return tensor

    def get_input_size(self) -> int:
        return self.input_lb.shape[0]

class OutputSpec(BaseSpec):
    def __init__(self, dataset: Dataset):
        super().__init__(dataset)
        if isinstance(self.dataset.spec_type, str):
            self.spec_type = SpecType(self.dataset.spec_type.lower())
        else:
            self.spec_type = self.dataset.spec_type
        self.labels = None
        self.output_constraints = None

        print(self.spec_type)
        if self.spec_type == SpecType.LOCAL_LP:
            self.labels = dataset.labels

        elif self.spec_type == SpecType.LOCAL_VNNLIB:
            self.labels = dataset.labels
            self.output_constraints = dataset.output_constraints

        elif self.spec_type == SpecType.SET_VNNLIB:
            self.output_constraints = dataset.output_constraints

        elif self.spec_type == SpecType.SET_BOX:
            self.output_constraints = dataset.output_constraints

        else:
            raise ValueError(f"Unsupported spec type: {self.spec_type}")

class Spec(BaseSpec):
    def __init__(self,
                 model : Model,
                 input_spec: InputSpec,
                 output_spec: OutputSpec):

        if input_spec.dataset != output_spec.dataset:
            raise ValueError("Input and output specifications must belong to the same dataset")

        if input_spec.spec_type != output_spec.spec_type:
            raise ValueError("Input and output specifications must have the same specification type")
        super().__init__(dataset=input_spec.dataset, model=model)
        self.input_spec = input_spec
        self.output_spec = output_spec
        self.spec_type = self.input_spec.spec_type

