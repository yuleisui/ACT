#!/usr/bin/env python3
"""
Entry point for ACT Back-End CLI.

Allows running the back-end CLI via: python -m act.back_end

Copyright (C) 2025 SVF-tools/ACT
License: AGPLv3+
"""

import sys

from act.config.backend_cli import main

if __name__ == "__main__":
    sys.exit(main())
