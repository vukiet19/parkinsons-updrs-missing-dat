"""Train the baseline with CAFR, PF-CMAD/PFAC, and SDU."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.cli import train_main

if __name__ == '__main__':
    train_main('kd_1')
