#===- util.path_config.py ----ACT Path Configuration ---------------------#
#
#                 ACT: Abstract Constraints Transformer
#
# Copyright (C) <2025->  ACT Team
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Purpose:
#   Python path configuration utilities for the Abstract Constraints Transformer
#   (ACT), ensuring proper module imports and path resolution across the
#   verification framework components.
#
#===----------------------------------------------------------------------===#

import os
import sys

def setup_act_paths():
    """Set up ACT project paths for proper module imports."""
    current_file = os.path.abspath(__file__)
    act_root = os.path.dirname(os.path.dirname(current_file))  # go up from util/ to act/
    if act_root not in sys.path:
        sys.path.insert(0, act_root)
    return act_root

def get_project_root():
    """Get the project root directory (parent of act/)."""
    current_file = os.path.abspath(__file__)
    # Current file is in act/util/path_config.py
    # Project root is two levels up
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
    return project_root

# Set up paths
act_root = setup_act_paths()
project_root = get_project_root()