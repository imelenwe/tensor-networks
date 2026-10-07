# run_hadron_aqc.py

Compresses the SU(2) hadron (loop-string-hadron) Trotter circuit with AQC-Tensor. For each chosen state (vacuum, meson) it:

1. builds the **target**: the full circuit as an MPS, cut to `--target-bond`, plus a health check (saved, so `--resume` doesn't rebuild it);
2. builds a **cheap circuit** (same circuit, fewer and bigger Trotter steps) that IBM's library turns into a trainable circuit;
3. **trains** its angles to match the target (L-BFGS-B, IBM's explicit gradient, 64-bit), simulating it at `--training-bond`;
4. **checks** the trained circuit after transpiling, and again at 2× the training bond;
5. **counts** CNOTs and depth, original vs trained (level 3, linear chain).

The main result is the **trained angles**.

## Setup

Python 3.12. Clone only the part of the repository the script needs (about 1 MB):

```bash
git clone --filter=blob:none --sparse <this repository's URL>
cd tensor-networks
git sparse-checkout set hadron-aqc/scripts
cd hadron-aqc
pip install -r scripts/requirements.txt
```

This brings `hadron-aqc/scripts/` and the files directly in `hadron-aqc/` (including `hadron_aqc_utils.py`, which the script imports from one folder up), and nothing else. Later, `git pull` picks up fixes.

Check: `python scripts/run_hadron_aqc.py --help` lists the options.

## Commands to run (copy-paste, from the `hadron-aqc` folder)

Run each state as its own job; they are independent. **Resources per job: 1 core is enough** (training uses about one core), memory below.

**1. Quick test (~2 min, any machine).** Checks the setup. Expected: fidelity above 0.9999, CNOTs 1231 → about 273, "stopped by --max-hours 0.03".

```bash
python scripts/run_hadron_aqc.py --sites 6 --steps 20 --coarse-steps 4 --target-bond 64 --training-bond 64 --states meson --max-hours 0.03
```

**2. 32 qubits, 20 steps.** Target ~20 min on a laptop (~80 min on 4 cluster cores); training ~3 min per step (the first steps up to ~9 min), 100–300 steps: about 5–15 h per state. Memory: 8 GB is plenty (measured peak 0.6 GB in training). Fits one 4-day job.

```bash
python scripts/run_hadron_aqc.py --sites 16 --steps 20 --coarse-steps 4 --target-bond 1024 --training-bond 256 --states meson --max-hours 72 2>> stderr_32q_meson.log
python scripts/run_hadron_aqc.py --sites 16 --steps 20 --coarse-steps 4 --target-bond 1024 --training-bond 256 --states vacuum --max-hours 72 2>> stderr_32q_vacuum.log
```

**3. 120 qubits, 20 steps (the benchmark).** Not yet run by us. Expect: target several hours (2.7 h on a laptop); training steps ~4× longer than at 32 qubits, so days; memory 16 GB (the bond-1024 target alone is ~3.4 GB). At bond 1024 the target loses ~1.5% (norm violation ~0.012–0.015, slightly above the 0.01 guide; known and accepted for this run).

```bash
python scripts/run_hadron_aqc.py --sites 60 --steps 20 --coarse-steps 4 --target-bond 1024 --training-bond 256 --states meson --max-hours 72 2>> stderr_120q_meson.log
python scripts/run_hadron_aqc.py --sites 60 --steps 20 --coarse-steps 4 --target-bond 1024 --training-bond 256 --states vacuum --max-hours 72 2>> stderr_120q_vacuum.log
```

`--max-hours 72` leaves room in a 4-day job for building the target and the final counting. If a job is killed anyway (time limit, crash, node failure), rerun the exact same line with `--resume` added before `2>>`, e.g.

```bash
python scripts/run_hadron_aqc.py --sites 60 --steps 20 --coarse-steps 4 --target-bond 1024 --training-bond 256 --states meson --max-hours 72 --resume 2>> stderr_120q_meson.log
```

**What to send back:** the `results/<run name>/` folder (without the large `target_*.pkl`) and the `stderr_*.log` file.

## Options

| option | meaning |
|---|---|
| `--sites` | number of sites (qubits = 2 × sites); use an even number |
| `--steps` | Trotter steps of the target (the benchmark uses 20) |
| `--coarse-steps` | Trotter steps of the cheap starting circuit (4 worked for 20 steps) |
| `--target-bond` | max bond when building the target |
| `--training-bond` | max bond when simulating the circuit during training (the main cost) |
| `--states` | `vacuum`, `meson`, or `vacuum meson` (run one per job to work in parallel) |
| `--max-hours` | stop each state's training after this many hours, then finish it normally (counts, table). Needed: in 64-bit the training rarely stops by itself |
| `--resume` | continue a killed run (see below) |

Strengths are fixed in `hadron_aqc_utils.py`: kinetic 0.15, electric 0.01, mass 0.03 per Trotter step.

## Output

Everything goes to `results/<run name>/`, named after the settings, e.g. `results/32q_20steps_4cheap_target1024_training256_meson/`:

| file | what it is |
|---|---|
| `summary.md` | **start here**: the results table |
| `angles.npz` | **the main result**: trained angles, one entry per state |
| `run.log` | every printed line with elapsed time; read this first when something looks wrong |
| `results.json` | the numbers in `summary.md`, for programs |
| `target_<state>.pkl` | the saved target (0.6 GB at 32 qubits, 3.4 GB at 120), loaded by `--resume` |
| `progress_<state>.npy` | only during training: the latest angles, used by `--resume` |

The `2>> stderr_*.log` in the commands keeps any messages printed outside Python (`>>` adds to the file, so a resumed run doesn't wipe it).

## Is it working?

- **Target line:** `norm violation` below 0.01 and `count of filled seats` equal to `--sites`.
- **Training:** one `training step N` line per step, fidelity rising.
- **End:** `stopped by --max-hours` or `converged`, and the recheck close to the trained fidelity.

## When something goes wrong

| what you see | what to do |
|---|---|
| norm violation above 0.01 | raise `--target-bond` (except 120 qubits at 1024: ~0.012–0.015 is expected, see above) |
| the first training steps are slow | normal: they take up to ~3× longer than later ones |
| job killed (time limit, crash) | run the same command plus `--resume` |
| `did NOT converge` | the 2000-step limit was reached before `--max-hours`; the result is still saved and usable |
| recheck much lower than the trained fidelity | raise `--training-bond` |

**`--resume`:** finished states are skipped; the saved target is loaded; training continues from `progress_<state>.npy` (step numbers restart at 1); `run.log` is continued, not overwritten. A state stopped by `--max-hours` counts as finished. Without `--resume`, the same command starts the folder fresh.

## Using the trained angles

The trainable circuit is rebuilt from the same settings (its shape is always the same), then the angles are filled in:

```python
import numpy as np
from hadron_aqc_utils import build_cheap_trainable_circuit, snap_tiny_angles

angles = np.load("results/<run name>/angles.npz")["meson"]
ansatz, _ = build_cheap_trainable_circuit(n_sites, num_trotter_steps, num_coarse_steps, use_meson_not_vacuum=True)
circuit = ansatz.assign_parameters(snap_tiny_angles(angles))
```

`circuit` is an ordinary Qiskit circuit, starting from all |0⟩: transpile it for the device and add measurements as usual. `snap_tiny_angles` sets angles below 1e-6 to exactly 0 (avoids a Qiskit level-3 transpile bug).
