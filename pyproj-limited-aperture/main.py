#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Limited-aperture coherence and Gram matrix analysis."""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from experiments.coherence import main as run_coherence


def main():
    print("\n== Limited-aperture coherence ==")
    run_coherence()


if __name__ == "__main__":
    main()
