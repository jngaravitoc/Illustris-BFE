from pathlib import Path
import matplotlib.pyplot as plt

STYLE_PATH = Path(__file__).resolve().parent / "illustris_bfe.mplstyle"
plt.style.use(STYLE_PATH)
