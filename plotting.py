# This script stores the plotting functions.
import os
import numpy as np
import matplotlib.pyplot as plt

def plot_overlay(temp_c: np.ndarray,
                 y_exp: np.ndarray,
                 y_fit: np.ndarray,
                 residual: np.ndarray,
                 title: str,
                 ylabel: str,
                 method: str,
                 data_type: str,
                 out_path: str,
                 ) -> None:
    """
    Make a 2-panel PNG:
      - top: experimental (open circles) vs fitted (line)
      - bottom: residuals (fit - exp)

    bottom_height_ratio: fraction of total height used for the residual panel
                         e.g., 0.3 = bottom is 30%, top is 70%
    """

    # Create a figure
    fig = plt.figure(figsize=(6, 6))

    # Ratio: top gets (1 - bottom_ratio), bottom gets bottom_ratio
    bottom_height_ratio = 0.15
    gs = plt.GridSpec(2, 1,
                      height_ratios=[1 - bottom_height_ratio, bottom_height_ratio]
                      )

    ax = fig.add_subplot(gs[0])
    axr = fig.add_subplot(gs[1], sharex=ax)

    # Setup colours of the plot. UV-Vis is red, qPCR is green
    if "uv" in method.casefold():
        colour = '#EA5958' # Red
    elif "qpcr" in method.casefold():
        colour = '#B1D095' # Green
    else:
        colour = '#6699FF' # Blue

    # Top panel --------------------------------------------------------
    ax.plot(temp_c, y_exp,
            linestyle="none",
            marker="o",
            markersize=6,
            markerfacecolor="none",
            markeredgecolor=colour,
            label="Experimental Data"
            )

    ax.plot(temp_c, y_fit,
            linestyle="-", linewidth=2,
            color="black",
            label="Fitted Curve",
            )

    ax.set_ylabel(ylabel, fontsize=16, fontweight="bold")
    ax.set_title(title, fontsize=16, fontweight="bold")

    # Legend
    ax.legend(loc="best", frameon=False, fontsize=12)

    # Axes styling
    ax.grid(False)
    ax.tick_params(axis="both", labelsize=10, width=2)

    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(2)

    # make tick numbers bold
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")

    # Hide tick labels and axis label
    ax.tick_params(labelbottom=False)  # hide tick labels on top plot
    ax.xaxis.label.set_visible(False)  # hide x-axis label on top plot

    # Bottom panel -----------------------------------------------------
    axr.plot(temp_c, residual,
             linestyle="none",
             marker="o", markersize=6,
             markerfacecolor="none",
             markeredgecolor=colour
             )

    axr.axhline(0, color="black", lw=1, ls="--")
    axr.set_xlabel("Temperature (°C)", fontsize=16, fontweight="bold")
    axr.set_ylabel("Residual", fontsize=16, fontweight="bold")

    # Axes styling
    axr.grid(False)
    axr.tick_params(axis="both", labelsize=10, width=2)
    for spine in axr.spines.values():
        spine.set_color("black")
        spine.set_linewidth(2)

    # make tick numbers bold
    for lbl in axr.get_xticklabels() + axr.get_yticklabels():
        lbl.set_fontweight("bold")

    # Set the Y axis of the residual plots
    if data_type == "direct":
        if "uv" in method.casefold():
            axr.set_yticks([-0.004, 0, 0.004])
        elif "qpcr" in method.casefold():
            axr.set_yticks([-1000, 0, 1000])
    elif data_type == "derivative":
        if "uv" in method.casefold():
            axr.set_yticks([-0.001, 0, 0.001])
        elif "qpcr" in method.casefold():
            axr.set_yticks([-500, 0, 500])

    axr.set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)