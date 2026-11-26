"""Run from any working directory; paths are relative to this kit."""

import os
from pathlib import Path

from core.export import main

if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
