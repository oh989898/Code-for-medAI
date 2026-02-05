import subprocess
import sys
from pathlib import Path


NOTEBOOKS = [
    "notebooks/01_sam_medical_segmentation.ipynb",
    "notebooks/02_titan_pathology.ipynb",
    "notebooks/03_clip_multimodal.ipynb",
]


def test_notebook_smoke_cells():
    script = Path("scripts/run_notebook_smoke.py")
    for notebook in NOTEBOOKS:
        result = subprocess.run(
            [sys.executable, str(script), notebook],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
