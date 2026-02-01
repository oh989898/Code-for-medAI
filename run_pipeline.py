#!/usr/bin/env python

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from medai_toolbox.codegen import main  # noqa: E402


if __name__ == "__main__":
    main()
