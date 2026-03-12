import subprocess
from pathlib import Path

def create_gif(png_pattern: str, output_gif: str, output_path: str = ".", delay: int = 5, loop: int = 0):
    """
    Create a GIF from a sequence of PNG images using ImageMagick's convert.

    Parameters
    ----------
    png_pattern : str
        Glob pattern for PNG files, e.g., 'm12b_OP_disk_*.png'
    output_gif : str
        Name of the output GIF file.
    output_path : str, optional
        Folder to save the GIF (default: current directory '.').
    delay : int, optional
        Delay between frames in 1/100s units (default: 5 → 0.05s per frame).
    loop : int, optional
        Number of times to loop the GIF (0 → infinite, default: 0).
    """
    # Check that at least one file matches the pattern
    files = list(Path().glob(png_pattern))
    if not files:
        raise FileNotFoundError(f"No PNG files match the pattern: {png_pattern}")

    # Make sure output directory exists
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Full path to the output GIF
    gif_path = output_dir / output_gif

    # Build the command
    cmd = [
        "convert",
        "-delay", str(delay),
        "-loop", str(loop),
        *[str(f) for f in sorted(files)],
        str(gif_path)
    ]

    # Run the command
    subprocess.run(cmd, check=True)
    print(f"Created GIF: {gif_path}")
