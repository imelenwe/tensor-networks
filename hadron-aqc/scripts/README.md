# run_hadron_aqc.py

Compresses the SU(2) hadron (loop-string-hadron) Trotter circuit with AQC-Tensor. For each chosen state (vacuum, meson) it:

1. builds the **target**: the full circuit as an MPS, cut to `--target-bond`, plus a health check (saved, so `--resume` doesn't rebuild it);
2. builds a **cheap circuit** (same circuit, fewer and bigger Trotter steps) that IBM's library turns into a trainable circuit;
3. **trains** its angles to match the target (L-BFGS-B; slopes from `--gradient jax` or `explicit`, see below);
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

Two circuits, chosen with `--x` (the group's x): **`100`** (tracker issue #254) and **`200`**. Same gates, different
strengths; x = 200 is much harder to compress. Run **one state per job** (vacuum and meson are independent), so a
full set is 2 circuits × 2 states = 4 jobs per size. Always add `--zero-out-zz` (2 CNOTs instead of 3 in ~60% of blocks).

**1. Quick test (~30 s, any machine).** Checks the setup. Expected (measured 2026-10-26, 12 qubits, 5-step start):
CNOTs 673 → 301, fidelity ≈ 0.9995 (x = 100) / 0.9948 (x = 200), "converged".

```bash
python scripts/run_hadron_aqc.py --x 100 --sites 6 --steps 11 --coarse-steps 5 --target-bond 64 --training-bond 64 --states meson --gradient jax --zero-out-zz
python scripts/run_hadron_aqc.py --x 200 --sites 6 --steps 11 --coarse-steps 5 --target-bond 64 --training-bond 64 --states meson --gradient jax --zero-out-zz
```

**2. Larger runs (HPC).** Set the two variables on the first line, then run the line for your size. Each job writes to
its own folder `results/<qubits>q_x<x>_11steps_<k>cheap_..._<state>/` and its own error log.

```bash
X=100; STATE=meson      # X: 100 or 200;  STATE: meson or vacuum  (one job each)

# 32 qubits
python scripts/run_hadron_aqc.py --x $X --sites 16 --steps 11 --coarse-steps 5 --target-bond 1024 --training-bond 512 --states $STATE --gradient jax --planner-minutes 5 --max-hours 12 --zero-out-zz 2>> stderr_32q_x${X}_${STATE}.log
# 40 qubits
python scripts/run_hadron_aqc.py --x $X --sites 20 --steps 11 --coarse-steps 5 --target-bond 1024 --training-bond 512 --states $STATE --gradient jax --planner-minutes 10 --max-hours 24 --zero-out-zz 2>> stderr_40q_x${X}_${STATE}.log
# 120 qubits (the benchmark size)
python scripts/run_hadron_aqc.py --x $X --sites 60 --steps 11 --coarse-steps 5 --target-bond 1024 --training-bond 512 --states $STATE --gradient jax --planner-minutes 30 --max-hours 72 --zero-out-zz 2>> stderr_120q_x${X}_${STATE}.log
```

| what to set | how |
|---|---|
| `--sites` | qubits ÷ 2; use an even number (16 = 32 qubits, 20 = 40, 60 = 120) |
| `--coarse-steps` | 5 for both x. For x = 200, also try **7** (better start, more CNOTs, ~1.4× RAM and time per step; its own `_7cheap_` folder) |
| `--planner-minutes` | planning time for the training contraction, done once per job: 5 at 32 qubits, 10 at 40, 30 at 120 |
| `--max-hours` | training stops cleanly after this; set it ~6 h below the job's wall-time limit (target build + final counting) |
| memory to request | see the table below |

| size | RAM to request (5-step start) | time | basis |
|---|---|---|---|
| 12 qubits | any machine | ~30 s | measured |
| 32 qubits | 16 GB (7-step: 64 GB) | target ~6 min, ~2.5 min per training step, levels off in ~1.5–3 h | measured on a laptop (x = 200); a 7-step start did not fit 24 GB |
| 40 qubits | 32 GB | ~1.5–3 h | measured on the earlier 20-step circuit (13 GB, 1.5 h) |
| 120 qubits | 128 GB (7-step: 256 GB) | target ~1 h, planning ~30 min, ~5 min per training step, 10–30 h | **estimate**, extrapolated from 24–40 qubits on the earlier circuit |

What to expect at x = 200 (measured at 32 qubits, 5-step start, before the mass correction): fidelity levels off at
0.90 (vacuum) / 0.94 (meson). The group expects lower fidelity at x = 200; x = 100 should stay much higher (0.9995 at 12 qubits).

If a job is killed (time limit, crash, node failure), rerun the exact same line with `--resume` added before `2>>`:
finished states are skipped, the saved target and contraction plan are reused, and training continues from the last step.

**What to send back:** the `results/<run name>/` folder (without the large `target_*.pkl`) and the `stderr_*.log` file.

## Options

| option | meaning |
|---|---|
| `--x` | the group's x (hopping ratio): `100` (tracker issue #254) or `200`. Sets the kinetic strength (x × 0.0015) and the mass (2√x × 0.0015); the electric field strength (−0.226125) is the same for both. Folder name gets `_x100` / `_x200`. |
| `--sites` | number of sites (qubits = 2 × sites); use an even number |
| `--steps` | Trotter steps of the target (the group's current circuit: 11; the earlier tracker benchmark used 20) |
| `--coarse-steps` | Trotter steps of the cheap starting circuit (5, about half of 11; 4 was used for the earlier 20-step circuit) |
| `--target-bond` | max bond when building the target |
| `--training-bond` | max bond when simulating the circuit during training (the main cost) |
| `--states` | `vacuum`, `meson`, or `vacuum meson` (run one per job to work in parallel) |
| `--gradient` | `jax` or `explicit`: how training gets its slopes (see "Which gradient?") |
| `--max-hours` | stop each state's training after this many hours, then finish it normally (counts, table). Needed with 64-bit numbers (explicit, or jax with `JAX_ENABLE_X64=True`): they rarely stop by themselves |
| `--planner-minutes` | jax only: let cotengra plan the order of the big training contraction for this many minutes (5 up to ~40 qubits, ~30 at 120). Same results, far less memory and time; the plan is saved in `contraction_plans/` and reused by `--resume`. Default: off |
| `--zero-out-zz` | keep the ZZ angle at 0 in every two-qubit block where it starts at 0 (pair and electric blocks): those compile to 2 CNOTs instead of 3. 12 qubits: CNOTs ~350 → 301, two-qubit depth 82 → 72, fidelity −0.001. The run folder gets `_zerozz`. |
| `--resume` | continue a killed run (see below) |

Strengths come from `--x` through `strengths_for_x` in `hadron_aqc_utils.py`, per Trotter step (dt = 0.0015): kinetic = x·dt, mass = 2·√x·dt, electric −0.226125 for both: x = 100 → 0.15 / −0.226125 / 0.03, x = 200 → 0.3 / −0.226125 / 0.0424264. Mass sign per site (− − + + …); hadron start = both qubits of the two centre sites. Checked gate for gate: x = 100 equals tracker issue #254's files (`SU2_T20_{SCV,MID}.qasm`); x = 200 equals the group's 2026-10-19 files (`SU2_T{T}_*_200.qasm`) once their mass 0.03 is corrected to 0.0424264. The gradient comparison below comes from the earlier circuit (kinetic 0.15, electric −0.01, per-qubit mass, 20 steps).

## Which gradient?

| | `jax` | `explicit` |
|---|---|---|
| what training simulates | the **exact** circuit: `--training-bond` is **ignored** during training | the circuit cut to `--training-bond` |
| result | trains toward the true answer | correct only if `--training-bond` is large enough; too small and it trains a blurred circuit |
| memory | grows with circuit size: ~1 GB at 12 qubits, ~50 GB at 32 (32-bit), ~91 GB (64-bit) | small: ~0.3 GB at 12 qubits, ~2 GB at 32 |
| numbers | 32-bit by default (stops by itself); `JAX_ENABLE_X64=True` for 64-bit | always 64-bit |

Measured at 32 qubits, 20 steps, meson, training bond 256, fidelity checked at bond 1024: jax **0.991**, explicit **0.949**. At 20 steps the state needs ~bond 1024, so explicit at 256 is too blurred. **Use jax unless memory doesn't allow it.** In both cases `--training-bond` still sets the bond of the printed fidelities, and the recheck uses 2× it. **With jax, trust the row "fidelity reached in training (exact, jax)" in `summary.md`:** the other fidelities are simulated at the training bond (and 2× it), which at 20 steps is too small to see the state (32 qubits, vacuum: 0.99 in training vs 0.19 at bond 256).

## Output

Everything goes to `results/<run name>/`, named after the settings, e.g. `results/32q_x100_11steps_5cheap_target1024_training512_jax_zerozz_meson/`:

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
| recheck much higher or lower than the trained fidelity | the printed fidelity is blurred: raise `--training-bond` (and with explicit, the training itself was blurred) |

**`--resume`:** finished states are skipped; the saved target is loaded; training continues from `progress_<state>.npy` (step numbers restart at 1); `run.log` is continued, not overwritten. A state stopped by `--max-hours` counts as finished. Without `--resume`, the same command starts the folder fresh.

## Using the trained angles

The trainable circuit is rebuilt from the same settings (its shape is always the same), then the angles are filled in:

```python
import numpy as np
from hadron_aqc_utils import build_cheap_trainable_circuit, snap_tiny_angles

angles = np.load("results/<run name>/angles.npz")["meson"]
ansatz, _ = build_cheap_trainable_circuit(n_sites, num_trotter_steps, num_coarse_steps, use_meson_not_vacuum=True,
                                         zero_out_zz=True)   # True for a run folder ending in _zerozz
circuit = ansatz.assign_parameters(snap_tiny_angles(angles))
```

`circuit` is an ordinary Qiskit circuit, starting from all |0⟩: transpile it for the device and add measurements as usual. `snap_tiny_angles` sets angles below 1e-6 to exactly 0 (avoids a Qiskit level-3 transpile bug).
