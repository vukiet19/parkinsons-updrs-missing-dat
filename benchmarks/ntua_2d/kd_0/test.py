"""Evaluate the basic baseline on held-out subjects."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.cli import test_main

if __name__ == '__main__':
    test_main('kd_0')
