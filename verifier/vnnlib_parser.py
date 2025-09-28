#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#########################################################################
##   Abstract Constraint Transformer (ACT) - VNNLIB Parser             ##
##                                                                     ##
##   doctormeeee (https://github.com/doctormeeee) and contributors     ##
##   Copyright (C) 2024-2025                                           ##
##                                                                     ##
#########################################################################

import re

class VNNLIBParser:

    def _var_extraction(var_string):

        match_indexed = re.match(r"^([A-Za-z_]+)_([0-9]+)$", var_string)
        if match_indexed:
            var_group = match_indexed.group(1)
            var_index = int(match_indexed.group(2))
            return var_group, var_index

        match_plain = re.match(r"^([A-Za-z_]+)$", var_string)
        if match_plain:
            var_group = match_plain.group(1)
            return var_group, -1

    def _num_extraction(var_string):
        match = re.match(r"^([+-]?)(\d+(\.\d*)?|\.\d+)$", var_string.strip())
        if match is None:
            return None
        sign = 1 if match.group(1) == "+" else -1
        return sign * float(match.group(2))

    @staticmethod

    def parse_term(input_clause):
        terms = input_clause.split()
        parsed_results = []
        sign_flag = None
        for term in terms:
            if term in ('+', '-'):
                sign_flag = 1 if term == '+' else -1
            else:
                num = VNNLIBParser._num_extraction(term)
                if num is None:
                    var_group, var_index = VNNLIBParser._var_extraction(term)

                    value = float(sign_flag) if sign_flag is not None else 1.0
                else:
                    var_group = "const"
                    var_index = -1

                    value = sign_flag * float(num) if sign_flag is not None else float(num)
                parsed_results.append((var_group, var_index, value))
                sign_flag = None

        return parsed_results

    @staticmethod

    def identify_declare_const(lines):
        input_vars, output_vars, anchors, utility = [], [], [], []
        for line in lines:
            if line.startswith("(declare-const"):
                match_indexed = re.match(r"\(declare-const ([A-Za-z_]+)_([0-9]+) [A-Za-z]+\)", line)
                if match_indexed:
                    var_group = match_indexed.group(1)
                    var_index = int(match_indexed.group(2))
                    if var_group == "X":
                        input_vars.append(("X", var_index))
                    elif var_group == "Y":
                        output_vars.append(("Y", var_index))
                    elif var_group == "X_hat":
                        anchors.append(("X_hat", var_index))
                    else:
                        utility.append((var_group, var_index))

                else:

                    match_plain = re.match(r"\(declare-const ([A-Za-z_]+) [A-Za-z]+\)", line)
                    if match_plain:
                        var_group = match_plain.group(1)
                        utility.append((var_group, -1))
        return input_vars, output_vars, anchors, utility

    @staticmethod

    def is_local(lines):
        return any("X_hat" in l or "eps" in l for l in lines)