#!/usr/bin/env python3
"""
Entry point for ACT Front-End Unified CLI.

Allows running the unified CLI via: python -m act.front_end

This provides auto-detection between TorchVision and VNNLIB creators,
along with common commands that work across both data sources.

Copyright (C) 2025 SVF-tools/ACT
License: AGPLv3+
"""

from act.config.frontend_cli import main

if __name__ == "__main__":
    main()
