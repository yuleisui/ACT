#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#########################################################################
##   Abstract Constraint Transformer (ACT) - Path Configuration        ##
##                                                                     ##
##   doctormeeee (https://github.com/doctormeeee) and contributors     ##
##   Copyright (C) 2024-2025                                           ##
##                                                                     ##
##   This module provides unified path configuration for all ACT       ##
##   modules, ensuring consistent imports regardless of file location. ##
##                                                                     ##
#########################################################################

import os
import sys

def setup_act_paths():
    current_file = os.path.abspath(__file__)
    verifier_root = os.path.dirname(current_file)
    if verifier_root not in sys.path:
        sys.path.insert(0, verifier_root)
    return verifier_root

verifier_root = setup_act_paths()