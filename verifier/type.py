#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#########################################################################
##   Abstract Constraint Transformer (ACT) - Type Definitions         ##
##                                                                     ##
##   doctormeeee (https://github.com/doctormeeee) and contributors     ##
##   Copyright (C) 2024-2025                                           ##
##                                                                     ##
#########################################################################

from enum import Enum

class SplitType(Enum):
    INPUT = "input"
    INPUT_GRAD = "input_grad"
    INPUT_SB = "input_sb"
    RELU_GRAD = "relu_grad"
    RELU_SR = "relu_babsr"
    RELU_CE = "relu_ce"

class VerificationStatus(Enum):
    SAT = "satisfiable"
    UNSAT = "unsatisfiable"
    CLEAN_FAILURE = "clean_failure"
    UNKNOWN = "unknown"
    TIMEOUT = "timeout"
    ERROR = "error"

class SpecType(Enum):
    LOCAL_LP = "local_lp"
    LOCAL_VNNLIB = "local_vnnlib"

    SET_VNNLIB = "set_vnnlib"
    SET_BOX = "set_box"

class LPNormType(Enum):
    LINF = "inf"
    L2 = "2"
    L1 = "1"