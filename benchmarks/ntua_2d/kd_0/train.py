"""Train the basic teacher-student baseline."""

import sys
from pathlib import Path

# Resolve shared components when this file is launched from either directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.cli import train_main

if __name__ == '__main__':
    train_main('kd_0')
