# run_hadron_aqc.py

Compresses the SU(2) hadron (loop-string-hadron) Trotter circuit with AQC-Tensor. For each chosen state (vacuum, meson) it:

1. builds the **target**: the full circuit as an MPS, cut to `--target-bond`, plus a health check;
2. builds a **cheap circuit** (same circuit, fewer and bigger Trotter steps) that IBM's library turns into a trainable circuit;
3. **trains** its angles to match the target (L-BFGS-B), simulating it at `--training-bond`;
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

Run each state as its own job; they are independent. Ask for a whole node: training uses every core it gets.

**1. Quick test (30 s, any machine).** Checks the setup. Expected: fidelity 0.999879, CNOTs 1231 → 278.

```bash
python scripts/run_hadron_aqc.py --sites 6 --steps 20 --coarse-steps 4 --target-bond 64 --training-bond 64 --states meson
```

**2. 32 qubits, 20 steps.** Target ~20 min, then ~15 min per training step, 100–200 steps: about 25–50 h per state. Memory 2–6 GB.

```bash
python scripts/run_hadron_aqc.py --sites 16 --steps 20 --coarse-steps 4 --target-bond 1024 --training-bond 256 --states meson 2>> stderr_32q_meson.log
python scripts/run_hadron_aqc.py --sites 16 --steps 20 --coarse-steps 4 --target-bond 1024 --training-bond 256 --states vacuum 2>> stderr_32q_vacuum.log
```

**3. 120 qubits, 20 steps (the benchmark).** Not yet run by us. Expect: target ~3 h; each training step several times slower than at 32 qubits (days in total, so `--resume` across job limits will be needed); memory at least 32 GB (the bond-1024 target alone is ~3.4 GB). At bond 1024 the target loses ~1.5% (norm violation ~0.012–0.015, slightly above the 0.01 guide; known and accepted for this run).

```bash
python scripts/run_hadron_aqc.py --sites 60 --steps 20 --coarse-steps 4 --target-bond 1024 --training-bond 256 --states meson 2>> stderr_120q_meson.log
python scripts/run_hadron_aqc.py --sites 60 --steps 20 --coarse-steps 4 --target-bond 1024 --training-bond 256 --states vacuum 2>> stderr_120q_vacuum.log
```

**If a job stops** (time limit, crash): rerun the exact same line with `--resume` added before `2>>`, e.g.

```bash
python scripts/run_hadron_aqc.py --sites 16 --steps 20 --coarse-steps 4 --target-bond 1024 --training-bond 256 --states meson --resume 2>> stderr_32q_meson.log
```

**What to send back:** the whole `results/<run name>/` folder and the `stderr_*.log` file.

## Options

| option | meaning |
|---|---|
| `--sites` | number of sites (qubits = 2 × sites); use an even number |
| `--steps` | Trotter steps of the target (the benchmark uses 20) |
| `--coarse-steps` | Trotter steps of the cheap starting circuit (4 worked for 20 steps) |
| `--target-bond` | max bond when building the target |
| `--training-bond` | max bond when simulating the circuit during training (the main cost) |
| `--states` | `vacuum`, `meson`, or `vacuum meson` (run one per job to work in parallel) |
| `--resume` | continue a stopped run (see below) |

Strengths are fixed in `hadron_aqc_utils.py`: kinetic 0.15, electric 0.01, mass 0.03 per Trotter step.

## Output

Everything goes to `results/<run name>/`, named after the settings, e.g. `results/32q_20steps_4cheap_target1024_training256_meson/`:

| file | what it is |
|---|---|
| `summary.md` | **start here**: the results table |
| `angles.npz` | **the main result**: trained angles, one entry per state |
| `run.log` | every printed line with elapsed time; read this first when something looks wrong |
| `results.json` | the numbers in `summary.md`, for programs |
| `progress_<state>.npy` | only during training: the latest angles, used by `--resume` |

`run.log` does not capture messages printed by jax's compiler; the `2>> stderr_*.log` in the commands above keeps those (`>>` adds to the file, so a resumed run doesn't wipe it).

## Is it working?

- **Target line:** `norm violation` below 0.01 and `count of filled seats` equal to `--sites`.
- **Training:** one `training step N` line per step, fidelity rising.
- **End:** `converged`, and the recheck close to the trained fidelity.

Measured on a laptop (M4 Pro, 14 cores, 24 GB): 12 qubits ~30 s in total. 32 qubits, 20 steps: target ~20 min at bond 1024 (bond 512 loses 3% of the state: too low); training at bond 256 ~15 min per step (the first step ~25 min), 100–200 steps expected, memory 2–6 GB.

## When something goes wrong

| what you see | what to do |
|---|---|
| norm violation above 0.01 | raise `--target-bond` (except 120 qubits at 1024: ~0.012–0.015 is expected, see above) |
| no new line for a long time | normal at the first training step (one-time setup); compare with the times above |
| `Constant folding ... taking > 1s` | harmless, appears once when training starts |
| job stopped (time limit, crash) | run the same command plus `--resume` |
| `did NOT converge` | the 2000-step limit was reached; `--resume` trains further |
| recheck much lower than the trained fidelity | raise `--training-bond` |

**`--resume`:** finished states are skipped; training restarts from `progress_<state>.npy` (step numbers restart at 1); `run.log` is continued, not overwritten. Without `--resume`, the same command starts the folder fresh.

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
