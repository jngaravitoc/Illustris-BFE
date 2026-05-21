from pathlib import Path
import os

# Base directory
_DEFAULT_PROJECT_DIR = Path(__file__).resolve().parents[2]
PROJECT_DIR = Path(os.environ.get("ILLUSTRIS_BFE", str(_DEFAULT_PROJECT_DIR)))


# Data directories
SIM_NAME = os.environ.get("ILLUSTRIS_SIM", "").strip()
DATA_ROOT = PROJECT_DIR / "data"
DATA_PATH = DATA_ROOT / SIM_NAME if SIM_NAME else DATA_ROOT
TEMP_DATA_PATH = PROJECT_DIR / "temp_data"
FIGURES_PATH = PROJECT_DIR / "temp_figures"


# Create directories if needed
