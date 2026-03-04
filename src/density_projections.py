import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Circle

def density_projections(p, R, lim=3000, vmax=1e4):
    """
    Plot two projections (x-y and y-z) of particle positions
    with black background, white axes, and a circle of radius R.

    Parameters
    ----------
    p : dict-like
        Particle data, must contain "x" as (N,3) array of positions.
    R : float
        Radius of the reference circle (same units as positions).
    lim : float, optional
        Axis limits for plots (default: 3000).
    vmax : float, optional
        Upper limit for color normalization (default: 1e4).
    """
    # Normalization for hexbin
    norm = mpl.colors.LogNorm(vmin=1, vmax=vmax)

    kwargs = {
        "extent": [-lim, lim, -lim, lim],
        "norm": norm,
        "cmap": "twilight",
        "gridsize": 200
    }

    fig, ax = plt.subplots(1, 2, figsize=(12, 6),
                           facecolor="black",
                           constrained_layout=True)

    # Common style for axes
    for a in ax:
        a.set_facecolor("black")
        a.tick_params(colors="white")
        for spine in a.spines.values():
            spine.set_color("white")

    # x-y projection
    hb1 = ax[0].hexbin(p["x"][:, 0], p["x"][:, 1], **kwargs)
    circ1 = Circle((0, 0), R, color="white", fill=False, lw=1.5)
    ax[0].add_patch(circ1)
    ax[0].set_xlabel("x", color="white")
    ax[0].set_ylabel("y", color="white")
    ax[0].set_xlim(-lim, lim)
    ax[0].set_ylim(-lim, lim)

    # y-z projection
    hb2 = ax[1].hexbin(p["x"][:, 1], p["x"][:, 2], **kwargs)
    circ2 = Circle((0, 0), R, color="white", fill=False, lw=1.5)
    ax[1].add_patch(circ2)
    ax[1].set_xlabel("y", color="white")
    ax[1].set_ylabel("z", color="white")
    ax[1].set_xlim(-lim, lim)
    ax[1].set_ylim(-lim, lim)

    # Shared colorbar
    cbar = fig.colorbar(hb1, ax=ax, orientation="vertical", shrink=0.8)
    cbar.set_label("Counts", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    return fig, ax

