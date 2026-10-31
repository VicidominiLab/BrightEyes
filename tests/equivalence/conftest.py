import matplotlib

matplotlib.use("Agg")
import sys
from pathlib import Path

REFERENCE_DIR = Path(__file__).parent / "reference"

if REFERENCE_DIR.exists():
    sys.path.insert(0, str(REFERENCE_DIR))
