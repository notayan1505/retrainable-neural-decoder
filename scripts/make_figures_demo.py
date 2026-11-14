"""Save example figures to presentation/figures/."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np

from gosc.board import Board
from gosc.engine import Engine
from gosc.ops import PPM, Init
from gosc.viz import draw_timeline, draw_trace, plot_logical_error, plot_space_time

out = pathlib.Path(__file__).resolve().parents[1] / "presentation" / "figures"
out.mkdir(parents=True, exist_ok=True)

r = Engine(Board(4, 2)).run([[Init("a", "+", (0, 0), "XNS"), Init("b", "+", (1, 0), "XNS")],
                             [PPM((("a", "Z"), ("b", "Z")))]])
draw_trace(r, out / "bell_trace.png")
draw_timeline(r, out / "bell_timeline.png")
region = frozenset({(1, 0), (1, 1), (2, 1), (3, 1)})
r = Engine(Board(6, 4)).run([
    [Init("y", "+", (2, 0)), Init("z", "+", (3, 0)), Init("x1", "+", (1, 2), "XNS"), Init("x2", "+", (4, 1))],
    [PPM((("y", "Y"), ("x1", "X"), ("z", "Z"), ("x2", "X")), region)]])
draw_trace(r, out / "ppm4_trace.png", ncols=3)
draw_timeline(r, out / "ppm4_timeline.png")
rows = [dict(name=n, k=k, tiles=10 * k, time_per_T=5 / k, hours=100 / k * f, physical_qubits=1e5 * k * f)
        for n, f in (("15-to-1", 1), ("20-to-4", 2)) for k in (1, 2, 4, 8)]
plot_space_time(rows, out / "space_time.png")
ps = np.array([1e-3, 2e-3, 4e-3, 8e-3])
plot_logical_error(ps, 30 * ps ** 2, out / "logical_error.png")
print("saved to", out)
