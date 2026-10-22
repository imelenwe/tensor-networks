"""Draw one Trotter step of the 12-qubit hadron circuit with the bricks outlined and numbered.

Run from this folder:  python draw_brickwork.py
Writes brickwork_annotated.png (mass sign per site: - - + + ...).
The circuit comes straight from construct_circuit in hadron_aqc_utils.py; Qiskit draws qubit q at
height -q and gate column c at x = c, so the boxes below are placed in those units.
"""
import sys
import matplotlib
matplotlib.use("Agg")
from matplotlib.patches import Polygon, Rectangle, Patch

sys.path.insert(0, "..")
from hadron_aqc_utils import construct_circuit, KINETIC_STRENGTH, ELECTRIC_FIELD_STRENGTH, MASS

COLOURS = {"pair": "#E8590C", "swap": "#2B8A3E", "electric": "#7048E8", "one": "#868E96"}
HALF_WIDTH, HALF_HEIGHT = 0.45, 0.42      # half a gate column, and the margin above/below a wire


def draw(file_name, n_sites=6, kinetic=KINETIC_STRENGTH, electric=ELECTRIC_FIELD_STRENGTH, mass=MASS, highlight_mass=False):
    circuit = construct_circuit(n_sites, 1, kinetic, electric, mass)
    figure = circuit.draw("mpl", fold=-1)
    axes = figure.axes[0]

    def label(x, y, text, colour):
        axes.text(x, y, text, fontsize=11, fontweight="bold", color="white", ha="left", va="bottom", zorder=10,
                  bbox=dict(boxstyle="round,pad=0.15", fc=colour, ec="none"))

    def brick(number, kind, top_qubit, start_column_top, start_column_bottom, end_column):
        """Outline a two-qubit brick; the top and bottom wires may start in different columns (an L shape)."""
        top, bottom = -top_qubit + HALF_HEIGHT, -(top_qubit + 1) - HALF_HEIGHT
        middle = -top_qubit - 0.5
        corners = [(start_column_top - HALF_WIDTH, top), (end_column + HALF_WIDTH, top), (end_column + HALF_WIDTH, bottom),
                   (start_column_bottom - HALF_WIDTH, bottom), (start_column_bottom - HALF_WIDTH, middle),
                   (start_column_top - HALF_WIDTH, middle)]
        axes.add_patch(Polygon(corners, closed=True, fill=False, ec=COLOURS[kind], lw=3.2, zorder=9))
        label(start_column_top - HALF_WIDTH + 0.03, top - 0.03 - 0.25, str(number), COLOURS[kind])

    def swap_brick(number, top_qubit, column):
        top, bottom = -top_qubit + 0.31, -(top_qubit + 1) - 0.31
        axes.add_patch(Rectangle((column - HALF_WIDTH + 0.09, bottom), 2 * HALF_WIDTH - 0.18, top - bottom,
                                 fill=False, ec=COLOURS["swap"], lw=3.2, zorder=9))
        label(column - HALF_WIDTH + 0.12, top - 0.03 - 0.25, str(number), COLOURS["swap"])

    def one_qubit(qubit, column):
        axes.add_patch(Rectangle((column - HALF_WIDTH + 0.06, -qubit - 0.39), 2 * HALF_WIDTH - 0.12, 0.78,
                                 fill=False, ec=COLOURS["one"], lw=2, ls="--", zorder=9))

    for qubit in [1, 3, 5, 7, 9, 11]:                 # starting X flips (vacuum)
        one_qubit(qubit, 0)
    number = 1
    for qubit in range(0, 12, 2):                     # pair layer 1: columns 1-8
        brick(number, "pair", qubit, 1, 1, 8); number += 1
    for qubit in [1, 3, 5, 7, 9]:                     # swap layer: column 9
        swap_brick(number, qubit, 9); number += 1
    for qubit in [2, 4, 6, 8]:                        # pair layer 2: columns 10-17
        brick(number, "pair", qubit, 10, 10, 17); number += 1
    for qubit in [3, 7]:                              # swap layer: column 18
        swap_brick(number, qubit, 18); number += 1
    electric_columns = {0: (9, 11, 14, 15), 2: (18, 20, 23, 24), 4: (19, 21, 24, 25),   # (top start, bottom start, end, mass)
                        6: (18, 20, 23, 24), 8: (19, 21, 24, 25), 10: (10, 12, 15, 16)}
    for qubit in range(0, 12, 2):                     # electric gates, then the mass Rz on both qubits
        top_start, bottom_start, end, mass_column = electric_columns[qubit]
        brick(number, "electric", qubit, top_start, bottom_start, end); number += 1
        one_qubit(qubit, mass_column); one_qubit(qubit + 1, mass_column)
        if highlight_mass:                            # red outline around the two mass Rz gates of this site
            axes.add_patch(Rectangle((mass_column - HALF_WIDTH - 0.04, -(qubit + 1) - 0.5), 2 * HALF_WIDTH + 0.08, 1.98,
                                     fill=False, ec="#D62828", lw=3.5, zorder=11))

    figure.legend(handles=[
        Patch(fc="none", ec=COLOURS["pair"], lw=3, label="pair gate (bricks 1–6, 12–15)"),
        Patch(fc="none", ec=COLOURS["swap"], lw=3, label="swap (bricks 7–11, 16–17)"),
        Patch(fc="none", ec=COLOURS["electric"], lw=3, label="electric gate (bricks 18–23)"),
        Patch(fc="none", ec=COLOURS["one"], lw=2, ls="--", label="one-qubit: starting X flips, mass Rz (not bricks)"),
    ], loc="lower center", ncol=4, fontsize=13, frameon=False)
    figure.savefig(file_name, dpi=80, bbox_inches="tight", facecolor="white")
    print("saved", file_name)


if __name__ == "__main__":
    draw("brickwork_annotated.png")
