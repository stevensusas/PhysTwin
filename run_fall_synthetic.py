#!/usr/bin/env python3
"""Wrapper to run stage2/script_generate_fall_synthetic.py with proper config."""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "stage2"))

from qqtt.utils import cfg

# Load config before running the script (sloth uses real.yaml)
cfg.load_from_yaml("configs/real.yaml")

# Import and run
from stage2.script_generate_fall_synthetic import main
main()
