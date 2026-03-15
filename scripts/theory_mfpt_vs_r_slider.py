"""
Interactive plot: theoretical MFPT vs reset rate r, with a slider for m0.

This script uses the analytical all-to-all theory in voter_model.solution_fpt.
No simulations are run here.

Usage:
    python scripts/theory_mfpt_vs_r_slider.py
"""

from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


# Ensure imports work when running from scripts/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from voter_model.solution_fpt import mean_fpt


def theoretical_mfpt_curve(N: int, m0: float, r_values: np.ndarray, M: int) -> np.ndarray:
    """Compute MFPT(r) from the analytical formula for fixed N, m0."""
    vals = [mean_fpt(N=N, m0=m0, r=float(r), M=M) for r in r_values]
    return np.asarray(vals, dtype=float)


def main() -> None:
    # Fixed parameters; edit if needed.
    N = 1000
    M = 800

    # r-grid shown on the x-axis. Keep r > 0 for log-scale x-axis.
    r_min = 1e-3
    r_max = 20.0
    nr = 140
    r_values = np.linspace(r_min, r_max, nr)

    # Initial slider value.
    m0_init = 0.0

    # Initial curve.
    y_init = theoretical_mfpt_curve(N=N, m0=m0_init, r_values=r_values, M=M)

    fig, (ax_ylog, ax_xlog) = plt.subplots(1, 2, figsize=(13.0, 5.0))
    plt.subplots_adjust(left=0.07, right=0.98, top=0.88, bottom=0.20, wspace=0.24)

    (line_ylog,) = ax_ylog.plot(r_values, y_init, lw=2.4, color="#1f77b4")
    ax_ylog.set_yscale("log")
    ax_ylog.set_xlabel("reset rate r")
    ax_ylog.set_ylabel("MFPT (theory)")
    ax_ylog.set_title(f"MFPT vs r (y-log), N={N}, M={M}")
    ax_ylog.grid(True, which="both", alpha=0.25)

    (line_xlog,) = ax_xlog.plot(r_values, y_init, lw=2.4, color="#d62728")
    ax_xlog.set_xscale("log")
    ax_xlog.set_xlabel("reset rate r")
    ax_xlog.set_ylabel("MFPT (theory)")
    ax_xlog.set_title(f"MFPT vs r (x-log), N={N}, M={M}")
    ax_xlog.grid(True, which="both", alpha=0.25)

    # Slider for m0 in [-1, 1].
    slider_ax = fig.add_axes([0.12, 0.08, 0.76, 0.05])
    m0_slider = Slider(
        ax=slider_ax,
        label="m0",
        valmin=-0.999,
        valmax=0.999,
        valinit=m0_init,
        valstep=0.001,
    )

    def update(_val: float) -> None:
        m0 = float(m0_slider.val)
        y = theoretical_mfpt_curve(N=N, m0=m0, r_values=r_values, M=M)
        line_ylog.set_ydata(y)
        line_xlog.set_ydata(y)

        # Keep y-axis responsive while avoiding wild jumps from tiny values.
        positive = y[np.isfinite(y) & (y > 0)]
        if positive.size > 0:
            y_lo = 0.9 * positive.min()
            y_hi = 1.1 * positive.max()
            if y_hi > y_lo:
                ax_ylog.set_ylim(y_lo, y_hi)
                ax_xlog.set_ylim(y_lo, y_hi)

        fig.canvas.draw_idle()

    m0_slider.on_changed(update)
    plt.show()


if __name__ == "__main__":
    main()
