"""Compress the hadron circuit with AQC-Tensor, for the vacuum, the meson, or both (one after the other).

Run from the hadron-aqc folder, for example:
    python scripts/run_hadron_aqc.py --sites 6 --steps 20 --coarse-steps 4 --target-bond 64 --training-bond 64 --states vacuum meson --gradient jax

Everything the run makes goes into one folder, named after its settings, for example
results/12q_20steps_4cheap_target64_training64_jax/ (a single state adds its name, e.g. ..._training64_jax_meson/):
    summary.md   the results as a table (readable)
    run.log      everything printed on screen, including warnings and errors
    results.json the same numbers as summary.md, for programs
    angles.npz   the trained angles (one entry per state)
    progress_<state>.npy  the latest angles during training, updated after every training step (deleted when that state finishes)
    target_<state>.pkl    the target, saved once built (reused by --resume instead of rebuilding it)

If a run is stopped (e.g. a cluster time limit), run the same command again with --resume:
states already finished are skipped, the saved target is loaded, and training continues from progress_<state>.npy.
--max-hours stops each state's training after that many hours; the state is then finished normally (counts, table).
"""
import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so the utils file one folder up can be imported
from hadron_aqc_utils import (build_target_using_lib, check_target, build_cheap_trainable_circuit,
                              make_simulator_settings, train_to_target, mps_fidelity,
                              tensornetwork_from_circuit, construct_circuit, count_cnots,
                              count_and_verify, KINETIC_STRENGTH, ELECTRIC_FIELD_STRENGTH, MASS)

parser = argparse.ArgumentParser(description="Compress the hadron circuit with AQC-Tensor")
parser.add_argument("--sites", type=int, required=True)
parser.add_argument("--steps", type=int, required=True)
parser.add_argument("--coarse-steps", type=int, required=True)
parser.add_argument("--target-bond", type=int, required=True)
parser.add_argument("--training-bond", type=int, required=True)
parser.add_argument("--states", nargs="+", choices=["vacuum", "meson"], required=True)
parser.add_argument("--gradient", choices=["jax", "explicit"], required=True,
                    help="jax: trains on the exact circuit, ignores --training-bond, high memory; "
                         "explicit: applies --training-bond during training, low memory")
parser.add_argument("--resume", action="store_true", help="continue a stopped run from its saved progress")
parser.add_argument("--max-hours", type=float, default=None, help="stop each state's training after this many hours (default: no limit)")
args = parser.parse_args()

# one folder per run, named after its settings
run_name = (f"{2 * args.sites}q_{args.steps}steps_{args.coarse_steps}cheap"
            f"_target{args.target_bond}_training{args.training_bond}_{args.gradient}")
if len(args.states) == 1:
    run_name += f"_{args.states[0]}"   # a single-state run gets its own folder, never mixed with a both-states run
hadron_aqc_folder = Path(__file__).resolve().parent.parent
run_folder = hadron_aqc_folder / "results" / run_name
run_folder.mkdir(parents=True, exist_ok=True)
run_folder_to_show = run_folder.relative_to(hadron_aqc_folder)   # "results/...", never the full path with the user's home folder


class ScreenAndFile:
    """Sends everything printed (including warnings and errors) both to the screen and to run.log."""
    def __init__(self, screen, file):
        self.screen = screen
        self.file = file
    def write(self, text):
        self.screen.write(text)
        self.file.write(text)
        self.file.flush()
    def flush(self):
        self.screen.flush()
        self.file.flush()
    def __getattr__(self, name):   # anything else a library asks for (e.g. isatty, encoding) is answered by the screen
        return getattr(self.screen, name)

log_file = open(run_folder / "run.log", "a" if args.resume else "w")   # resume adds to the old log instead of wiping it
sys.stdout = ScreenAndFile(sys.stdout, log_file)
sys.stderr = ScreenAndFile(sys.stderr, log_file)

start_time = time.time()
def log(message):
    print(f"[{(time.time() - start_time) / 60:6.1f} min] {message}", flush=True)


def how_long(seconds):
    """Short times in seconds, long ones in minutes."""
    return f"{seconds:.0f} s" if seconds < 90 else f"{seconds / 60:.1f} min"


def save_angles(file_path, angles):
    """Write to a temporary file first, then rename: if the run is killed mid-write, the old file stays intact."""
    temporary_path = file_path.with_suffix(".tmp")
    with open(temporary_path, "wb") as angles_file:
        np.save(angles_file, angles)
    os.replace(temporary_path, file_path)


def save_target(file_path, target_mps):
    """Same write-then-rename as save_angles, for the target."""
    temporary_path = file_path.with_suffix(".tmp")
    with open(temporary_path, "wb") as target_file:
        pickle.dump(target_mps, target_file)
    os.replace(temporary_path, file_path)


def write_summary(results):
    """summary.md: the settings, then one table with a column per state."""
    labels = [label for label in ["vacuum", "meson"] if label in results]
    rows = [
        ("target health: norm violation / count of filled seats",
         lambda r: f"{r['norm_violation']:.4f} / {r['count_of_filled_seats']:.4f} (must be {args.sites})"),
        ("angles", lambda r: f"{r['angles']}"),
        ("fidelity before → after training", lambda r: f"{r['fidelity_before']:.4f} → **{r['fidelity_after']:.6f}**"),
        ("training steps", lambda r: f"{r['training_steps']} ({r['stopped']})"),
        ("CNOTs (original → trained)", lambda r: f"{r['original_cnots']} → **{r['cnots']}**"),
        ("two-qubit depth", lambda r: f"{r['original_two_qubit_depth']} → **{r['two_qubit_depth']}**"),
        ("total depth", lambda r: f"{r['original_total_depth']} → {r['total_depth']}"),
        ("fidelity of the transpiled circuit", lambda r: f"{r['transpiled_fidelity']:.6f}"),
        (f"recheck at max_bond={2 * args.training_bond}", lambda r: f"{r['fidelity_recheck']:.6f}"),
        ("time", lambda r: f"{r['minutes']:.1f} min"),
    ]
    lines = [f"# {run_name}", "",
             f"{2 * args.sites} qubits ({args.sites} sites), {args.steps} Trotter steps, {args.coarse_steps}-step cheap circuit, "
             f"target max_bond={args.target_bond}, training max_bond={args.training_bond}, gradient={args.gradient}. "
             f"Started {results['started']}.", "",
             "| | " + " | ".join(labels) + " |",
             "|---|" + "---|" * len(labels)]
    for row_name, show in rows:
        lines.append(f"| {row_name} | " + " | ".join(show(results[label]) for label in labels) + " |")
    (run_folder / "summary.md").write_text("\n".join(lines) + "\n")


log(f"started {time.strftime('%Y-%m-%d %H:%M')}: {2 * args.sites} qubits ({args.sites} sites), {args.steps} Trotter steps, "
    f"{args.coarse_steps}-step cheap circuit, target max_bond={args.target_bond}, training max_bond={args.training_bond}, gradient={args.gradient}")
if args.gradient == "jax":
    log(f"note: with gradient=jax the training uses the exact circuit; max_bond={args.training_bond} only applies to the printed fidelities")
log(f"saving to {run_folder_to_show}")

results = {"settings": vars(args), "started": time.strftime('%Y-%m-%d %H:%M')}
trained_angles = {}
if args.resume and (run_folder / "results.json").exists():
    results = json.loads((run_folder / "results.json").read_text())       # states already finished
    trained_angles = dict(np.load(run_folder / "angles.npz"))
    log(f"resuming: already finished: {[label for label in args.states if label in results] or 'none'}")
training_settings = make_simulator_settings(max_bond=args.training_bond, gradient=args.gradient)
check_settings = make_simulator_settings(max_bond=2 * args.training_bond, gradient=args.gradient)

for label in args.states:
    if label in results:
        log(f"{label}: already finished, skipping")
        continue
    use_meson_not_vacuum = (label == "meson")
    state_start_time = time.time()

    # target
    target_file = run_folder / f"target_{label}.pkl"
    if args.resume and target_file.exists():
        log(f"{label}: loading saved target from {target_file.name}...")
        with open(target_file, "rb") as saved:
            target_mps = pickle.load(saved)
    else:
        log(f"{label}: building target...")
        target_mps = build_target_using_lib(args.sites, args.steps, args.target_bond, use_meson_not_vacuum)
        save_target(target_file, target_mps)
    norm_violation, _, count_of_filled_seats = check_target(target_mps)
    norm_violation = round(float(norm_violation), 6) + 0.0   # a rounding leftover like -1e-13 becomes 0.0 (adding 0.0 also turns -0.0 into 0.0)
    log(f"{label}: target ready (max_bond={args.target_bond}), norm violation={norm_violation:.4f}, "
        f"count of filled seats={count_of_filled_seats:.4f} (must be {args.sites})")

    # cheap trainable circuit
    ansatz, starting_angles = build_cheap_trainable_circuit(args.sites, args.steps, args.coarse_steps, use_meson_not_vacuum)
    untrained_mps = tensornetwork_from_circuit(ansatz.assign_parameters(starting_angles), training_settings)
    fidelity_before = float(mps_fidelity(untrained_mps, target_mps))
    log(f"{label}: cheap {args.coarse_steps}-step circuit, {len(starting_angles)} angles, fidelity before training={fidelity_before:.6f}")
    progress_file = run_folder / f"progress_{label}.npy"
    if args.resume and progress_file.exists():
        starting_angles = np.load(progress_file)   # "fidelity before training" above stays the cheap circuit's, for the table
        log(f"{label}: resuming training from {progress_file.name}")

    # train
    time_limit = "" if args.max_hours is None else f", stops after {args.max_hours} h"
    log(f"{label}: training... (a progress line after every training step{time_limit})")
    steps_done = 0
    training_start_time = last_step_time = time.time()
    hit_time_limit = False
    def show_progress(fidelity, angles):
        global steps_done, last_step_time, hit_time_limit
        steps_done += 1
        save_angles(progress_file, angles)
        log(f"{label}:   training step {steps_done}, fidelity={fidelity:.6f}, this step took {how_long(time.time() - last_step_time)}")
        last_step_time = time.time()
        if args.max_hours is not None and time.time() - training_start_time > args.max_hours * 3600:
            hit_time_limit = True
            raise StopIteration   # scipy's way to end training early; it returns the latest angles

    result = train_to_target(ansatz, starting_angles, target_mps, training_settings, progress=show_progress)
    trained_angles[label] = result.x
    np.savez(run_folder / "angles.npz", **trained_angles)
    trained_mps = tensornetwork_from_circuit(ansatz.assign_parameters(result.x), training_settings)
    fidelity_after = float(mps_fidelity(trained_mps, target_mps))
    if hit_time_limit:
        stopped = f"stopped by --max-hours {args.max_hours}"
    else:
        stopped = "converged" if result.success else f"did NOT converge: {result.message}"
    log(f"{label}: trained, fidelity={fidelity_after:.6f} ({result.nit} training steps, training max_bond={args.training_bond}), "
        f"{stopped}; angles saved to angles.npz")

    # count CNOTs: original circuit vs trained circuit
    log(f"{label}: counting CNOTs (transpiling both circuits) and checking the transpiled circuit...")
    original_circuit = construct_circuit(args.sites, args.steps, KINETIC_STRENGTH, ELECTRIC_FIELD_STRENGTH, MASS,
                                         use_meson_not_vacuum=use_meson_not_vacuum)
    original_cnots, original_two_qubit_depth, original_total_depth, _ = count_cnots(original_circuit)
    counts = count_and_verify(ansatz, result.x, target_mps, training_settings)
    log(f"{label}: CNOTs {original_cnots} -> {counts['cnots']}, two-qubit depth {original_two_qubit_depth} -> {counts['two_qubit_depth']}, "
        f"total depth {original_total_depth} -> {counts['total_depth']}, transpiled fidelity={float(counts['transpiled_fidelity']):.6f}")

    # recheck: same trained circuit, simulated with double the bond
    log(f"{label}: recheck at max_bond={2 * args.training_bond}...")
    recheck_mps = tensornetwork_from_circuit(ansatz.assign_parameters(result.x), check_settings)
    fidelity_recheck = float(mps_fidelity(recheck_mps, target_mps))
    log(f"{label}: recheck at max_bond={2 * args.training_bond}: fidelity={fidelity_recheck:.6f}")

    results[label] = {"norm_violation": norm_violation, "count_of_filled_seats": float(count_of_filled_seats),
                      "angles": len(starting_angles), "fidelity_before": fidelity_before,
                      "fidelity_after": fidelity_after, "fidelity_recheck": fidelity_recheck,
                      "training_steps": int(result.nit), "stopped": stopped, "original_cnots": original_cnots,
                      "original_two_qubit_depth": original_two_qubit_depth, "original_total_depth": original_total_depth,
                      "cnots": counts["cnots"], "two_qubit_depth": counts["two_qubit_depth"], "total_depth": counts["total_depth"],
                      "transpiled_fidelity": float(counts["transpiled_fidelity"]),
                      "qubit_order_unchanged": counts["qubit_order_unchanged"],
                      "minutes": (time.time() - state_start_time) / 60}

    # save after each state, so a crash later doesn't lose this state's results
    with open(run_folder / "results.json", "w") as results_file:
        json.dump(results, results_file, indent=2)
    write_summary(results)
    progress_file.unlink(missing_ok=True)   # only now: results.json holds this state, so --resume will skip it
    log(f"{label}: saved results.json and summary.md")

log(f"finished {time.strftime('%Y-%m-%d %H:%M')}; see {run_folder_to_show / 'summary.md'}")
