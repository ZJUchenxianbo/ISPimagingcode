#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Point-scatterer imaging with direct sampling and U-Net enhancement."""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from experiments.imaging import main as run_imaging


def main():
    print("\n== Point-scatterer imaging ==")
    run_imaging()


if __name__ == "__main__":
    main()
