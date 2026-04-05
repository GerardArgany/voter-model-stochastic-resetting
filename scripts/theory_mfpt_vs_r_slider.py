"""
Interactive plot: theoretical MFPT vs reset rate r, with a slider for m0.

This script uses the analytical all-to-all theory in voter_model.solution_fpt.
No simulations are run here.

Usage:
    python scripts/theory_mfpt_vs_r_slider.py
"""

import argparse
import threading
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import SymLogNorm
from matplotlib.widgets import Slider


# Ensure imports work when running from scripts/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from voter_model.solution_fpt import B, la


def theoretical_mfpt_curve(N: int, m0: float, r_values: np.ndarray, M: int) -> np.ndarray:
    """Compute MFPT(r) from the ratio-of-sums analytical expression.

    We mask points too close to denominator zeros to avoid plotting artificial
    line segments across divergences.

    MFPT(r) = [sum_l B_{2l}/(r/N + lambda'_{2l+1})]
              / [sum_l B_{2l} lambda'_{2l+1}/(r/N + lambda'_{2l+1})].
    """
    sum_term, sum_lambda_term, _, _, _ = theoretical_components_curves(
        N=N,
        m0=m0,
        r_values=r_values,
        M=M,
    )

    y = np.full_like(sum_term, np.nan, dtype=float)
    finite = np.isfinite(sum_term) & np.isfinite(sum_lambda_term)
    # Relative threshold around poles of the MFPT denominator sum.
    scale = max(float(np.nanmax(np.abs(sum_lambda_term[finite]))) if np.any(finite) else 1.0, 1.0)
    pole_tol = 1e-8 * scale
    stable = finite & (np.abs(sum_lambda_term) > pole_tol)
    y[stable] = sum_term[stable] / sum_lambda_term[stable]

    # Suppress extreme spikes so curves remain readable near divergences.
    y[np.abs(y) > 1e12] = np.nan
    return y


def theoretical_components_curves(
    N: int,
    m0: float,
    r_values: np.ndarray,
    M: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute components of the parenthesized MFPT expression.

    Returns:
        (sum_term, sum_lambda_term, first_parenthesis, second_parenthesis, closure_gap) where
        sum_term(r) = sum_n B_n / (r/N + lambda_n)
        sum_lambda_term(r) = sum_n B_n lambda_n / (r/N + lambda_n)
        first_parenthesis(r) = (1 - m0^2) * sum_term(r)
        second_parenthesis(r) = 1 - (r/N) * first_parenthesis(r)
        closure_gap(r) = second_parenthesis(r) - (1 - m0^2) * sum_lambda_term(r)
    """
    i = np.arange(M)
    n_even = 2 * i
    n_odd = n_even + 1

    b_vals = B(n_even, m0)
    la_vals = la(n_odd, N)

    r_ct = r_values / N
    denom = r_ct[:, None] + la_vals[None, :]

    sum_term = np.sum(b_vals[None, :] / denom, axis=1)
    sum_lambda_term = np.sum((b_vals * la_vals)[None, :] / denom, axis=1)
    first_parenthesis = (1.0 - m0 ** 2) * sum_term
    second_parenthesis = 1.0 - r_ct * first_parenthesis
    closure_gap = second_parenthesis - (1.0 - m0 ** 2) * sum_lambda_term

    return sum_term, sum_lambda_term, first_parenthesis, second_parenthesis, closure_gap


def zero_crossings(r_values: np.ndarray, y_values: np.ndarray) -> np.ndarray:
    """Estimate r-values where y(r)=0 using linear interpolation at sign changes."""
    r = np.asarray(r_values, dtype=float)
    y = np.asarray(y_values, dtype=float)

    roots: list[float] = []
    finite = np.isfinite(y)
    for i in range(len(r) - 1):
        if not (finite[i] and finite[i + 1]):
            continue
        y0 = y[i]
        y1 = y[i + 1]
        if y0 == 0.0:
            roots.append(float(r[i]))
            continue
        if y0 * y1 < 0.0:
            # Linear interpolation between the two adjacent samples.
            rr = r[i] - y0 * (r[i + 1] - r[i]) / (y1 - y0)
            roots.append(float(rr))
    return np.asarray(roots, dtype=float)


def compute_second_parenthesis_map(
    N: int,
    r_values: np.ndarray,
    m0_values: np.ndarray,
    M_map: int,
) -> np.ndarray:
    """Compute second-parenthesis field S2(m0, r)=1-(r/N)(1-m0^2)sum B/(r/N+lambda)."""
    out = np.empty((len(r_values), len(m0_values)), dtype=float)
    for j, m0_s in enumerate(m0_values):
        _, _, _, second_scan, _ = theoretical_components_curves(
            N=N,
            m0=float(m0_s),
            r_values=r_values,
            M=M_map,
        )
        out[:, j] = second_scan
    return out


def compute_true_denominator_map(
    N: int,
    r_values: np.ndarray,
    m0_values: np.ndarray,
    M_map: int,
) -> np.ndarray:
    """Compute true MFPT denominator field D(m0,r)=sum B_n*lambda_n/(r/N+lambda_n)."""
    out = np.empty((len(r_values), len(m0_values)), dtype=float)
    for j, m0_s in enumerate(m0_values):
        _, sum_lambda_term, _, _, _ = theoretical_components_curves(
            N=N,
            m0=float(m0_s),
            r_values=r_values,
            M=M_map,
        )
        out[:, j] = sum_lambda_term
    return out


def save_static_second_parenthesis_subspace_plot(
    N: int = 1000,
    M_map: int = 10_000,
    nr_map: int = 320,
    nm0_map: int = 241,
) -> Path:
    """Generate and save static (m0,r) map of the second-parenthesis inside expression."""
    r_values = np.linspace(0.0, float(N), nr_map)
    m0_scan = np.linspace(-0.99, 0.99, nm0_map)
    second_map = compute_second_parenthesis_map(N=N, r_values=r_values, m0_values=m0_scan, M_map=M_map)

    abs_nonzero = np.abs(second_map[np.isfinite(second_map) & (second_map != 0.0)])
    linthresh = max(float(np.percentile(abs_nonzero, 5)) if abs_nonzero.size > 0 else 1e-6, 1e-12)

    fig = Figure(figsize=(9.2, 6.3))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    im = ax.imshow(
        second_map,
        origin="lower",
        aspect="auto",
        extent=[m0_scan[0], m0_scan[-1], r_values[0], r_values[-1]],
        interpolation="nearest",
        resample=False,
        cmap="coolwarm",
        norm=SymLogNorm(linthresh=linthresh, vmin=np.nanmin(second_map), vmax=np.nanmax(second_map)),
    )
    cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.05)
    cbar.set_label("Numerator sum: sum B_{2l}/(r/N+lambda'_{2l+1})")

    if np.nanmin(second_map) <= 0.0 <= np.nanmax(second_map):
        ax.contour(m0_scan, r_values, second_map, levels=[0.0], colors="black", linewidths=1.8)

    ax.set_xlabel("m0")
    ax.set_ylabel("r")
    ax.set_title(f"Static second-parenthesis subspace (N={N}, M_map={M_map}, r in [0, N])")
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(0.0, float(N))
    ax.grid(True, which="both", alpha=0.2)

    out_dir = PROJECT_ROOT / "figures" / "all_to_all"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"second_parenthesis_subspace_static_N{N}_M{M_map}_rmaxN.pdf"
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    return out_path


def main() -> None:
    # Fixed parameters; edit if needed.
    N = 1000
    M_init = 800
    M_map_init = 800

    # Always produce the static m0-r subspace figure on normal runs,
    # but do it in background so the slider UI appears immediately.
    def _build_static_map_background() -> None:
        try:
            static_path = save_static_second_parenthesis_subspace_plot(
                N=N,
                M_map=10_000,
                nr_map=320,
                nm0_map=241,
            )
            print(f"Auto-saved static second-parenthesis map: {static_path}")
        except Exception as exc:
            print(f"Static-map auto-save failed: {exc}")

    threading.Thread(target=_build_static_map_background, daemon=True).start()

    # r-grid shown on the x-axis. Keep r > 0 for log-scale x-axis.
    r_min = 1e-3
    r_max = N
    nr = 140
    r_values = np.linspace(r_min, r_max, nr)
    nr_map = 480
    r_values_map = np.linspace(r_min, r_max, nr_map)

    # Initial slider value.
    m0_init = 0.0

    # Initial curves.
    y_init = theoretical_mfpt_curve(N=N, m0=m0_init, r_values=r_values, M=M_init)
    _, sum_lambda_init, _, second_paren_init, closure_init = theoretical_components_curves(
        N=N, m0=m0_init, r_values=r_values, M=M_init
    )
    _, sum_lambda_map_init, _, _, _ = theoretical_components_curves(
        N=N, m0=m0_init, r_values=r_values_map, M=M_map_init
    )

    fig = plt.figure(figsize=(13.5, 8.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0])
    ax_ylog = fig.add_subplot(gs[0, 0])
    ax_xlog = fig.add_subplot(gs[0, 1])
    ax_den = fig.add_subplot(gs[1, 0])
    ax_zero = fig.add_subplot(gs[1, 1])
    plt.subplots_adjust(left=0.07, right=0.98, top=0.93, bottom=0.16, wspace=0.24, hspace=0.34)

    (line_ylog,) = ax_ylog.plot(r_values, y_init, lw=2.4, color="#1f77b4")
    ax_ylog.set_yscale("log")
    ax_ylog.set_xlabel("reset rate r")
    ax_ylog.set_ylabel("MFPT (theory)")
    ax_ylog.set_title(f"MFPT vs r (y-log), N={N}, M={M_init}")
    ax_ylog.grid(True, which="both", alpha=0.25)

    (line_xlog,) = ax_xlog.plot(r_values, y_init, lw=2.4, color="#d62728")
    ax_xlog.set_xscale("log")
    ax_xlog.set_xlabel("reset rate r")
    ax_xlog.set_ylabel("MFPT (theory)")
    ax_xlog.set_title(f"MFPT vs r (x-log), N={N}, M={M_init}")
    ax_xlog.grid(True, which="both", alpha=0.25)

    (line_den,) = ax_den.plot(r_values, second_paren_init, lw=2.2, color="#9467bd")
    (line_den_true,) = ax_den.plot(
        r_values,
        (1.0 - m0_init ** 2) * sum_lambda_init,
        lw=1.8,
        ls="--",
        color="#ff7f0e",
    )
    ax_den.axhline(0.0, lw=1.0, color="black", alpha=0.5)
    ax_den.set_xlabel("reset rate r")
    ax_den.set_ylabel("MFPT ratio sums")
    ax_den.set_title("Numerator vs denominator sum")
    ax_den.legend(["Numerator: sum B_{2l}/(r/N+lambda')", "Denominator: sum B_{2l}*lambda'/(r/N+lambda')"], loc="best")
    ax_den.set_yscale("symlog", linthresh=1e-6)
    ax_den.grid(True, which="both", alpha=0.25)

    # Map in (m0, r) for true MFPT denominator D(m0,r).
    m0_scan = np.linspace(-0.99, 0.99, 181)

    second_map = compute_second_parenthesis_map(N=N, r_values=r_values_map, m0_values=m0_scan, M_map=M_map_init)
    den_map = compute_true_denominator_map(N=N, r_values=r_values_map, m0_values=m0_scan, M_map=M_map_init)
    abs_nonzero = np.abs(den_map[np.isfinite(den_map) & (den_map != 0.0)])
    linthresh = max(float(np.percentile(abs_nonzero, 5)) if abs_nonzero.size > 0 else 1e-6, 1e-12)
    im = ax_zero.imshow(
        den_map,
        origin="lower",
        aspect="auto",
        extent=[m0_scan[0], m0_scan[-1], r_values_map[0], r_values_map[-1]],
        interpolation="nearest",
        resample=False,
        cmap="coolwarm",
        norm=SymLogNorm(linthresh=linthresh, vmin=np.nanmin(den_map), vmax=np.nanmax(den_map)),
    )
    cbar = fig.colorbar(im, ax=ax_zero, pad=0.02, fraction=0.048)
    cbar.set_label("True denominator: sum B_{2l}*lambda'_{2l+1}/(r/N+lambda'_{2l+1})")

    contour_zero = None
    den_map_min = np.nanmin(den_map)
    den_map_max = np.nanmax(den_map)
    import sys
    print(f"DEBUG: den_map range: [{den_map_min:.6e}, {den_map_max:.6e}]", flush=True)
    
    # Check if any negatives are present
    has_neg = np.any(den_map < 0)
    has_pos = np.any(den_map > 0)
    print(f"DEBUG: Has negative values: {has_neg}, Has positive values: {has_pos}", flush=True)
    
    # Extract slice from 2D map at m0=0 and compare with 1D computation
    m0_zero_idx = np.argmin(np.abs(m0_scan - 0.0))
    den_map_slice_at_m0_0 = den_map[:, m0_zero_idx]
    print(f"DEBUG: Slice at m0={m0_scan[m0_zero_idx]:.4f}: min={np.nanmin(den_map_slice_at_m0_0):.6e}, max={np.nanmax(den_map_slice_at_m0_0):.6e}", flush=True)
    print(f"DEBUG: Direct 1D sum_lambda_init: min={np.nanmin(sum_lambda_init):.6e}, max={np.nanmax(sum_lambda_init):.6e}", flush=True)
    
    # Find zeros in both
    roots_from_1d = zero_crossings(r_values, sum_lambda_init)
    roots_from_map_slice = zero_crossings(r_values_map, den_map_slice_at_m0_0)
    print(f"DEBUG: Zeros from 1D: {roots_from_1d}", flush=True)
    print(f"DEBUG: Zeros from 2D slice: {roots_from_map_slice}", flush=True)
    
    if den_map_min <= 0.0 <= den_map_max:
        print("DEBUG: Condition passed, drawing solid black contour for denominator=0")
        contour_zero = ax_zero.contour(m0_scan, r_values_map, den_map, levels=[0.0], colors="black", linewidths=2.0, zorder=10)
    else:
        print("DEBUG: Condition FAILED - no denominator zero contour in current domain")

    roots_init = zero_crossings(r_values_map, sum_lambda_map_init)
    (line_zero_m0,) = ax_zero.plot([m0_init, m0_init], [r_min, r_max], lw=1.5, color="#ffd400", alpha=0.95, label="current m0")
    (line_zero_curr,) = ax_zero.plot(
        np.full(roots_init.shape, m0_init),
        roots_init,
        marker="o",
        ls="None",
        ms=6,
        color="#00ff7f",
        label="denominator=0 at current m0",
    )
    ax_zero.set_xlabel("m0")
    ax_zero.set_ylabel("r")
    ax_zero.set_title(f"True-denominator subspace + denominator=0 contour (M_map={M_map_init})")
    ax_zero.set_xlim(-1.0, 1.0)
    ax_zero.set_ylim(r_min, r_max)
    ax_zero.grid(True, which="both", alpha=0.25)
    legend_zero = ax_zero.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.95, edgecolor="black")
    legend_zero.set_zorder(20)

    print("Initial diagnostics (m0=0):")
    i_den_min = int(np.nanargmin(np.abs(sum_lambda_init)))
    i_sec_min = int(np.nanargmin(np.abs(second_paren_init)))
    i_gap_max = int(np.nanargmax(np.abs(closure_init)))
    print(
        f"  min |true denominator sum| at r={r_values[i_den_min]:.6g}, "
        f"value={(sum_lambda_init[i_den_min]):.6e}"
    )
    print(
        f"  min |second parenthesis| at r={r_values[i_sec_min]:.6g}, "
        f"value={(second_paren_init[i_sec_min]):.6e}"
    )
    print(
        f"  max |closure gap| at r={r_values[i_gap_max]:.6g}, "
        f"gap={(closure_init[i_gap_max]):.6e}"
    )

    # Main slider for m0 in [-1, 1] (drives upper/lower-left/lower-right MFPT panels).
    slider_ax = fig.add_axes([0.12, 0.08, 0.76, 0.03])
    m0_slider = Slider(
        ax=slider_ax,
        label="m0 (main plots)",
        valmin=-0.999,
        valmax=0.999,
        valinit=m0_init,
        valstep=0.001,
    )

    trunc_ax = fig.add_axes([0.12, 0.03, 0.76, 0.03])
    trunc_slider = Slider(
        ax=trunc_ax,
        label="M (truncation, all plots)",
        valmin=50,
        valmax=2000,
        valinit=M_map_init,
        valstep=10,
    )

    def update(_val: float) -> None:
        m0 = float(m0_slider.val)
        M_active = int(round(trunc_slider.val))
        y = theoretical_mfpt_curve(N=N, m0=m0, r_values=r_values, M=M_active)
        _, sum_lambda, _, second_paren, closure_gap = theoretical_components_curves(
            N=N, m0=m0, r_values=r_values, M=M_active
        )
        line_ylog.set_ydata(y)
        line_xlog.set_ydata(y)
        line_den.set_ydata(second_paren)
        line_den_true.set_ydata((1.0 - m0 ** 2) * sum_lambda)
        ax_ylog.set_title(f"MFPT vs r (y-log), N={N}, M={M_active}")
        ax_xlog.set_title(f"MFPT vs r (x-log), N={N}, M={M_active}")

        # Keep y-axis responsive while avoiding wild jumps from tiny values.
        positive = y[np.isfinite(y) & (y > 0)]
        if positive.size > 0:
            y_lo = 0.9 * positive.min()
            y_hi = 1.1 * positive.max()
            if y_hi > y_lo:
                ax_ylog.set_ylim(y_lo, y_hi)
                ax_xlog.set_ylim(y_lo, y_hi)

        # Autoscale component panels while remaining robust to NaN/Inf.
        den_finite = second_paren[np.isfinite(second_paren)]
        if den_finite.size > 0:
            den_lo, den_hi = den_finite.min(), den_finite.max()
            if den_hi > den_lo:
                pad = 0.08 * (den_hi - den_lo)
                ax_den.set_ylim(den_lo - pad, den_hi + pad)

        # Update current-m0 markers in denominator subspace panel.
        _, sum_lambda_map_curr, _, _, _ = theoretical_components_curves(
            N=N, m0=m0, r_values=r_values_map, M=M_active
        )
        roots_curr = zero_crossings(r_values_map, sum_lambda_map_curr)
        line_zero_m0.set_xdata([m0, m0])
        line_zero_m0.set_ydata([r_min, r_max])
        line_zero_curr.set_xdata(np.full(roots_curr.shape, m0))
        line_zero_curr.set_ydata(roots_curr)

        # Print concise divergence diagnostics for current m0.
        i_den_min = int(np.nanargmin(np.abs(sum_lambda)))
        i_sec_min = int(np.nanargmin(np.abs(second_paren)))
        i_gap_max = int(np.nanargmax(np.abs(closure_gap)))
        print(
            f"m0={m0:+.3f} | r_den_min={r_values[i_den_min]:.6g}, "
            f"den={sum_lambda[i_den_min]:.3e} | "
            f"r_sec_min={r_values[i_sec_min]:.6g}, sec={second_paren[i_sec_min]:.3e} | "
            f"max_gap={closure_gap[i_gap_max]:.3e}@r={r_values[i_gap_max]:.6g}"
        )

        fig.canvas.draw_idle()

    def update_map_truncation(_val: float) -> None:
        nonlocal second_map, den_map, contour_zero
        M_map = int(round(trunc_slider.val))
        second_map = compute_second_parenthesis_map(N=N, r_values=r_values_map, m0_values=m0_scan, M_map=M_map)
        den_map = compute_true_denominator_map(N=N, r_values=r_values_map, m0_values=m0_scan, M_map=M_map)

        abs_nonzero_local = np.abs(den_map[np.isfinite(den_map) & (den_map != 0.0)])
        linthresh_local = max(float(np.percentile(abs_nonzero_local, 5)) if abs_nonzero_local.size > 0 else 1e-6, 1e-12)

        im.set_data(den_map)
        im.set_norm(SymLogNorm(linthresh=linthresh_local, vmin=np.nanmin(den_map), vmax=np.nanmax(den_map)))

        if contour_zero is not None:
            try:
                contour_zero.remove()
            except Exception:
                for coll in contour_zero.collections:
                    coll.remove()
            contour_zero = None
        if np.nanmin(den_map) <= 0.0 <= np.nanmax(den_map):
            contour_zero = ax_zero.contour(m0_scan, r_values_map, den_map, levels=[0.0], colors="black", linewidths=2.0, zorder=10)

        cbar.update_normal(im)
        ax_zero.set_title(f"True-denominator subspace + denominator=0 contour (M_map={M_map})")

        # Keep current-m0 root markers consistent with the displayed map truncation.
        m0_curr = float(m0_slider.val)
        _, sum_lambda_curr, _, _, _ = theoretical_components_curves(
            N=N,
            m0=m0_curr,
            r_values=r_values_map,
            M=M_map,
        )
        roots_curr = zero_crossings(r_values_map, sum_lambda_curr)
        line_zero_m0.set_xdata([m0_curr, m0_curr])
        line_zero_curr.set_xdata(np.full(roots_curr.shape, m0_curr))
        line_zero_curr.set_ydata(roots_curr)

        print(f"map truncation updated: M_map={M_map}")

        # Keep upper and lower-left curves synchronized with truncation.
        update(_val)

        fig.canvas.draw_idle()

    m0_slider.on_changed(update)
    trunc_slider.on_changed(update_map_truncation)

    # Ensure all panels are synchronized with slider initial values.
    update(0.0)
    update_map_truncation(0.0)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MFPT theory diagnostics and static denominator maps.")
    parser.add_argument("--static-map", action="store_true", help="Generate static (m0,r) denominator map and exit.")
    parser.add_argument("--N", type=int, default=1000, help="System size N.")
    parser.add_argument("--M-static", type=int, default=10000, help="Truncation M for static map.")
    parser.add_argument("--nr-static", type=int, default=320, help="Number of r points for static map.")
    parser.add_argument("--nm0-static", type=int, default=241, help="Number of m0 points for static map.")
    args = parser.parse_args()

    if args.static_map:
        out = save_static_second_parenthesis_subspace_plot(
            N=int(args.N),
            M_map=int(args.M_static),
            nr_map=int(args.nr_static),
            nm0_map=int(args.nm0_static),
        )
        print(f"Saved static second-parenthesis map to: {out}")
    else:
        main()
