from pathlib import Path
import os

# Base directory
PROJECT_DIR =  Path(os.environ.get("ILLUSTRIS_BFE"))


# Data directories
DATA_PATH = PROJECT_DIR / "data"
TEMP_DATA_PATH = PROJECT_DIR / "temp_data"
FIGURES_PATH = PROJECT_DIR / "temp_figures"


# Create directories if needed
