#!/usr/bin/env python
"""Run only the notebook cells tagged with 'smoke'.

This lets beginners validate that the core logic runs without downloading
large models or weights.
"""
import json
import sys
from pathlib import Path


def run_smoke_cells(notebook_path: Path) -> None:
    data = json.loads(notebook_path.read_text(encoding="utf-8"))
    namespace = {}
    for cell in data.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        tags = cell.get("metadata", {}).get("tags", [])
        if "smoke" not in tags:
            continue
        source = "".join(cell.get("source", []))
        exec(compile(source, str(notebook_path), "exec"), namespace)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_notebook_smoke.py <notebook.ipynb>")
        sys.exit(1)

    for nb in sys.argv[1:]:
        run_smoke_cells(Path(nb))
        print(f"OK: smoke cells ran for {nb}")
