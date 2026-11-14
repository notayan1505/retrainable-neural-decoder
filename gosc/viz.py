"""Matplotlib figures for the tile game (slide-friendly, 16:9, dpi 200)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from .board import DIRS

DPI = 200
PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2", "#EECA3B", "#9D755D"]
EDGE_COLOR = "#222222"
_SEG = {"N": ((0, 0), (1, 0)), "S": ((0, 1), (1, 1)), "W": ((0, 0), (0, 1)), "E": ((1, 0), (1, 1))}


def _soft(hex_color, mix=0.55):
    c = np.array(matplotlib.colors.to_rgb(hex_color))
    return tuple(c * (1 - mix) + mix)


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, which="both", alpha=0.25)


def draw_board(board, ax=None, title=None, highlight_tiles=None, edge_labels=False):
    """Draw tiles, patch colours/ids, solid Z edges, dashed X edges."""
    if ax is None:
        _, ax = plt.subplots(figsize=(max(4, board.width * 1.2), max(3, board.height * 1.2)))
    for x in range(board.width):
        for y in range(board.height):
            ax.add_patch(Rectangle((x, y), 1, 1, fc="white", ec="#CCCCCC", lw=0.8))
    for i, (pid, p) in enumerate(board.patches.items()):
        col = _soft(PALETTE[i % len(PALETTE)])
        for (x, y) in p.tiles:
            ax.add_patch(Rectangle((x, y), 1, 1, fc=col, ec="none"))
        cx = np.mean([t[0] for t in p.tiles]) + 0.5
        cy = np.mean([t[1] for t in p.tiles]) + 0.5
        ax.text(cx, cy, str(pid), ha="center", va="center", fontsize=13, fontweight="bold", color="#111111")
        for ((x, y), side), ty in p.edges.items():
            (x0, y0), (x1, y1) = _SEG[side]
            ax.plot([x + x0, x + x1], [y + y0, y + y1], color=EDGE_COLOR, lw=3,
                    ls="-" if ty == "Z" else (0, (2, 1.5)), solid_capstyle="butt", zorder=3)
            if edge_labels:
                dx, dy = DIRS[side]
                ax.text(x + 0.5 + 0.36 * dx, y + 0.5 + 0.36 * dy, ty, ha="center", va="center",
                        fontsize=7, color="#555555", zorder=4)
    for (x, y) in (highlight_tiles or ()):
        ax.add_patch(Rectangle((x, y), 1, 1, fc="none", ec="#888888", hatch="///", lw=0, zorder=2))
    ax.set_xlim(0, board.width)
    ax.set_ylim(board.height, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=12)
    return ax


def draw_trace(result, path, ncols=4, edge_labels=False):
    boards = result.boards
    n = len(boards)
    ncols = max(1, min(ncols, n))
    nrows = -(-n // ncols)
    bw, bh = boards[0].width, boards[0].height
    fig, axes = plt.subplots(nrows, ncols, figsize=(max(3.2, bw * 0.9) * ncols, max(2.2, bh * 0.9) * nrows + 0.5),
                             squeeze=False)
    t = 0
    for k, ax in enumerate(axes.flat):
        if k >= n:
            ax.axis("off")
            continue
        if k > 0:
            t += result.step_times[k - 1]
        draw_board(boards[k], ax, title="initial (t = 0)" if k == 0 else f"step {k} (t = {t})",
                   edge_labels=edge_labels)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, format="png")
    plt.close(fig)
    return path


def draw_timeline(result, path):
    times = list(result.step_times)
    fig, ax = plt.subplots(figsize=(10, 0.5 * len(times) + 1.5))
    ys = np.arange(len(times))
    left = np.concatenate([[0], np.cumsum(times)[:-1]]) if times else []
    ax.barh(ys, times, left=left, color=PALETTE[0], edgecolor="white")
    ax.set_yticks(ys)
    ax.set_yticklabels([f"step {i + 1}" for i in ys])
    ax.invert_yaxis()
    ax.set_xlabel("time (units of d code cycles)")
    ax.set_title(f"Per-step duration (total = {result.total_time})")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, format="png")
    plt.close(fig)
    return path


def plot_space_time(rows, path):
    fig, ax = plt.subplots(figsize=(10, 5.6))
    names = list(dict.fromkeys(r["name"] for r in rows))
    for i, nm in enumerate(names):
        rs = sorted((r for r in rows if r["name"] == nm), key=lambda r: r["physical_qubits"])
        x = [r["physical_qubits"] for r in rs]
        y = [r["hours"] for r in rs]
        c = PALETTE[i % len(PALETTE)]
        ax.plot(x, y, "-o", color=c, label=nm, lw=2, ms=7)
        for r in rs:
            ax.annotate(f"k={r['k']}", (r["physical_qubits"], r["hours"]), textcoords="offset points",
                        xytext=(6, 6), fontsize=9, color=c)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("physical qubits")
    ax.set_ylabel("runtime (hours)")
    ax.set_title("Space-time trade-off (k = distillation blocks)")
    ax.legend(frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, format="png")
    plt.close(fig)
    return path


def plot_logical_error(ps, rates, path, slope_fit=True):
    ps, rates = np.asarray(ps, float), np.asarray(rates, float)
    fig, ax = plt.subplots(figsize=(8, 5.6))
    ax.loglog(ps, rates, "o", color=PALETTE[0], ms=8, label="simulation")
    lo, hi = min(ps.min(), rates[rates > 0].min() if (rates > 0).any() else ps.min()), max(ps.max(), rates.max())
    ax.loglog([lo, hi], [lo, hi], "--", color="#888888", label="y = x (no encoding)")
    m = (ps > 0) & (rates > 0)
    if slope_fit and m.sum() >= 2:
        s, b = np.polyfit(np.log(ps[m]), np.log(rates[m]), 1)
        xs = np.geomspace(ps[m].min(), ps[m].max(), 50)
        ax.loglog(xs, np.exp(b) * xs ** s, "-", color=PALETTE[1], lw=2, label=f"fit: slope = {s:.2f}")
    ax.set_xlabel("physical error rate p")
    ax.set_ylabel("logical error rate")
    ax.set_title("Logical vs physical error")
    ax.legend(frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, format="png")
    plt.close(fig)
    return path
