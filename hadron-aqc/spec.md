# Hadron AQC — spec

Apply AQCtensor's Algorithm 1 (same method as `AQC/aqctensor-algorithm1.ipynb`) to the SU(2) Loop-String-Hadron (LSH) circuit from IBM's tutorial: run the deep Trotter circuit through TEBD to get a target state, then train a much shallower AQC circuit to reproduce it — same physics, far fewer two-qubit gates.

## Why this matters

The tutorial compresses nothing: it sends the full deep circuit (>3,400 two-qubit gates at 60 qubits) to hardware and fights the noise with mitigation. More sites or more time means more gates, and eventually noise wins. AQC's payoff is the same physics with a fraction of the gates. It also tests whether the compression (proven on the toy XYZ chain) generalises to a real, actively-studied problem.

## Context

- The question: implement AQC-Tensor on this circuit and see what happens with TEBD; later cross-check against a parallel implementation on the same circuit (reportedly better fidelity than "the original AQC" at small qubit counts).
- **Now the benchmark (2026-10-02):** quantum-advantage-tracker instance `su2_hadron_dynamics_lsh_x_100_meson` (+ `_SCV` = our vacuum) — see Phase 4.
- **Parallel implementation's protocol (2026-09-29):** target = hadron circuit as quimb MPS (SVD cutoff 1e-12); start = shallower Trotter (4–5 steps for a 10-step target) + X layer; each 2q gate → KAK block, 1q gates → rotations, init reproduces shallow circuit exactly; IBM `qiskit-addon-aqc-tensor` `MaximizeStateFidelity` + L-BFGS-B; ansatz MPS capped at χ=128, final fidelity rechecked at 256/512; cost = transpile both circuits to CX on a linear chain (same settings), compare CX counts. Same method as ours; the library automates mapping + gives autodiff gradients. For 3.4: match their CX counting, also run `num_coarse_steps` = 4/5.

## Files

| file | what it is |
|---|---|
| `tutorials/IBM-loop-string-hadron-dynamics.ipynb` | the tutorial, raw reference copy |
| `hadron-aqc/notebooks/hadron-aqc-algorithm1.ipynb` | **done** — circuit + measurement pipeline rebuilt in our own names, plain-English physics, heatmaps. Verified identical to the tutorial (118 circuit cases + helpers, diff = 0). |
| `hadron-aqc/hadron_aqc_utils.py` | copies of the hadron + Algorithm 1 functions, **plus the library route** (the AQC recipe written once, shared by every size section and the cluster script): `build_target_using_lib`, `check_target`, `build_cheap_trainable_circuit`, `train_to_target`, `count_cnots`, `count_and_verify`, `make_simulator_settings`, `mps_fidelity`, `tensors_to_library_target`. Never modify `AQC/aqctensor-algorithm1.ipynb`. |
| `hadron-aqc/notebooks/hadron-aqc-mapping.ipynb` | the trainable circuit built by hand (old OPTIONAL section: brickwork + `build_aqc_hadron_circuit`, 4 tables, annotated picture). Runs on its own; = library (147 CX at 2 steps). Place to try block changes. |
| `hadron-aqc/notebooks/hadron-aqc-tebd.ipynb` | **current work** — the AQC-Tensor notebook. Imports from the utils file. **Kernel: `qgss26`** (Py 3.12.13, Qiskit 2.5.0, numpy 2.5.1) — test there, not `qcml-ibmqc`. |
| `hadron-aqc/scripts/run_hadron_aqc.py` | one command per run: `python scripts/run_hadron_aqc.py --sites --steps --coarse-steps --target-bond --training-bond --states [vacuum] [meson] [--max-hours H] [--resume]`. Writes `results/<qubits>q_<steps>steps_<k>cheap_target<b>_training<b>[_<state>]/` with `summary.md` (table), `run.log`, `results.json`, `angles.npz`, `target_<state>.pkl` (not in git). Log line + saved angles after every training step; `--max-hours` ends training cleanly; `--resume` loads the saved target and continues a killed run; rechecks at 2× training bond. Handover docs: `scripts/README.md`, `scripts/requirements.txt` (partial clone: `git sparse-checkout set hadron-aqc/scripts`, ~1 MB). |
| `hadron-aqc/results/` | trained angles, run folders; `targets_120qubits_merged/` (8.4 GB, not in git). |

## Key facts

- **Circuit:** `construct_circuit(n_sites, num_trotter_steps, kinetic_strength=0.15, electric_field_strength=0.01, mass=0.03, use_meson_not_vacuum=...)`. 2 qubits per site. X gates for vacuum/meson are inside the circuit, so it starts from all-|0⟩. Step 0 skips its first SWAP layer. All two-qubit gates act on neighbouring qubits.
- **Measurement:** Z per qubit (`SparsePauliOp`, list reversed `[::-1]` for Qiskit's right-to-left order) → `get_physical_particle_count` → `calc_meson_signal(meson_counts, vacuum_counts, n_sites)`.
- **Ground truth at 12 qubits:** `Statevector(construct_circuit(...))`, not an exact Hamiltonian — measures compression error only. (Tutorial's own θ formula disagrees in sign with its code; treat the code as the definition.)
- **Gate counts (6 sites):** raw CNOTs 73 / 155 / 237 / 811 for 1 / 2 / 3 / 10 steps. **Bar to beat: 611** at 12 qubits/10 steps, 3455 at 60 qubits — Qiskit O3 transpiled on the tutorial's linear-chain coupling map (a coupling-map-free transpile cheats by relabelling qubits).
- **Two-qubit gate positions per step:** 23 at step 0 (10 pair, 7 SWAP, 6 electric), 26 afterwards. Our hand-built AQC: 3 CX per position = 147 at k=2. Library (merges, O3 on a line): 129 at k=2, ~200 at k=3.
- **Entanglement grows fast:** bond dimension 48 at 2 steps, 176 at 5, still growing after 10 (quimb needed ≈800 at scale). At 12 qubits max bond is 64, so `max_bond=64` = exact.
- **Triplet block expressibility:** the triplet *alone* fits Pair (0.75) and Electric (0.05) to ~1e-11; SWAP = all-zero triplet. It can NOT absorb a preceding Rz⊗Rz before a Pair gate (miss ~1e-2) → the Rz before each brick is the home for carried mass turns (+ free knobs, no CNOTs).
- **MPS truncation bug** (`apply_2q_gate_mps_tensor`): cuts without canonical form and never renormalises — norm leaks (12-qubit random circuit at χ=16: norm² 0.88, reported fidelity 0.78 vs true 0.89). Not used from Phase 2 on — the library (quimb) does all cutting.
- **IBM library route (scratch prototype 2026-09-30, isolated venv on top of qgss26 — adds packages only, numpy/qiskit unchanged):** `generate_ansatz_from_circuit(coarse, qubits_initially_zero=True)` does 1.3a+1.3b+1.4 in one call (same 147 CX at k=2, same before-training 0.5999/0.7152, untrained = coarse exactly); needs the *flat* circuit for quimb simulation (it can't simulate our named blocks). `MaximizeStateFidelity` + L-BFGS-B (`jac=True`, autodiff) trains in seconds. Our TEBD target = exact (fidelity 1.0). Results vs 10-step target (611 CX, O3 on a line):

  | k | Trotter fid vac/mes (CX) | AQC trained fid vac/mes (CX) | time |
  |---|---|---|---|
  | 2 | 0.600 / 0.715 (115) | 0.99965 / **0.9665** (129) | 6 s |
  | 3 | 0.851 / 0.899 (177) | **0.99997 / 0.99999** (~200) | 10 s |
  | 4 | 0.938 / 0.958 (239) | 0.999993 / 0.999996 (~279) | 13 s |
  | 5 | 0.972 / 0.982 (301) | 0.999996 / 0.999997 (~348) | 15 s |

  Meson at k=2 gets stuck (local minimum) → **k=3 is the sweet spot** (~3× fewer CX than 611). Our own finite-difference training (vacuum k=2: 0.9994 in 24 min, stopped by `maxfun`) agrees with the library's 0.99965 → independent cross-check.
- **Why a cheap Trotter circuit first (t=10 target, library, tested 2026-09-30):** k=1 (62 CX) trains only to 0.20 / 0.43 (vac/mes) — too few gate layers to build the entanglement; k=2 (129) 0.9996 / 0.967; k=3 (~200) 0.99997 / 0.99999. Same k=3 shape from **random** angles: 0.968 / 0.981. → the cheap circuit sets the gate budget AND gives the good starting point.
- **Mass term** barely changes dynamics (m = 0 / 0.03 / 0.3 give near-identical output) — open question.

## Plan

**Route:** IBM's library (`qiskit-addon-aqc-tensor` + quimb) for training, and for the target from step 7 on. Our own TEBD only builds and checks the 12-qubit target. The hand-built mapping (old OPTIONAL section) now lives in `hadron-aqc-mapping.ipynb`. Work top-down.

**Notebook (`hadron-aqc-tebd.ipynb`): one section per size, steps restart at 1 in each; long runs and file writes behind switches at the top (`save_12_qubit_trained_angles`, `build_120_qubit_targets`, planned 32/80-qubit ones); "Run All" writes nothing and takes ~2 min.**

**12 qubits (6 sites, 10 Trotter steps):**
- [x] 1. Check our MPS code — = statevector (diff ~4e-15).
- [x] 2. Build the 12-qubit target — our TEBD, vacuum + meson, norm 1.
- [x] 3. Hand the target to IBM's library — `tensors_to_library_target`; = library TEBD (1.000000000000).
- [x] 4. Build the cheap trainable circuit — 3-step circuit (strengths × 10/3) → `generate_ansatz_from_circuit`; before training 0.851 / 0.899.
- [x] 5. Train it — `MaximizeStateFidelity` + L-BFGS-B → **0.999967 / 0.999986**; `results/trained_aqc_hadron_12q_k3.npz`.
- [x] 6. Count CNOTs and depth — O3, linear chain, seed 0, angles < 1e-6 snapped to 0 first: CNOTs 611 → 201 / 202; two-qubit depth 147 → 47 / 48; total depth 347 → 165 / 163. Transpiled circuit fidelity = 0.999967 / 0.999986 (unchanged; qubit order unchanged).
- [x] 7. Target with a bond cap + health check — `build_target_using_lib` (gates on each qubit pair merged into one block, then quimb `CircuitMPS(max_bond, cutoff=0)`; ~2× fewer cuts, far less norm loss) and `check_target` (norm violation, chance each qubit reads 1, count of filled seats = `n_sites`).
- [x] 8. Calibrate the convergence check — the change when `max_bond` doubles = error of the smaller one; count of filled seats alone is weak. Rule: no qubit changes by > 0.001 and norm violation < 0.01.

**32 qubits (16 sites, 10 Trotter steps; 30 not used: an odd number of sites gives wrong vacuum/meson starting states):**
- [x] 1. Targets — converged by `max_bond` 128 (change 128→256 ≤ 0.0001); trained against 256.
- [x] 2. Cheap trainable circuit — 3-step start, 2044 angles; before training 0.643 / 0.671.
- [x] 3. Train — training at `max_bond` 128: 0.99901 / 0.99950, ~4 min each; `results/trained_aqc_hadron_32q_k3.npz`.
- [x] 4. Count + verify — same circuit checked at 256 and 512: **0.99922 / 0.99951** (the 128 cap slightly under-reported vacuum). CNOTs 1796 → 611, two-qubit depth 147 → 48, total depth 347 → 175 / 174. Transpiled circuit keeps its fidelity. No failure at ~30 qubits.

**20 Trotter steps (the benchmark's step count), via the script:**
- [x] 12 qubits, 4-step start: **0.999906 / 0.999879**, CNOTs 1231 → 278, two-qubit depth 297 → 65, ~30 s. Compression holds at 20 steps; the saving grows with steps (3× at 10 steps, 4.4× at 20).
- [x] 32 qubits, 4-step start — too slow for the laptop: target bond 512 loses 3% of the state (norm violation 0.0295, too low); bond 1024 loses 0.15% (20 min). Training at bond 256: ~15 min per step (step 1: 23 min, 0.13 → 0.47; step 2: 16.5 min → 0.55), 100–200 steps expected → 25–50 h per state. Stopped; cluster job.
- [x] Cluster feedback (2026-10-07): jax gradient needs 50 GB (32-bit) / 91 GB (64-bit) at 32 qubits → out of memory. Switched `make_simulator_settings` to `autodiff_backend="explicit"` (IBM's own gradient): 32 qubits at training bond 256 → peak 0.64 GB, ~2.7 min per step (first steps up to 9 min), 1 core, 64-bit. 12 qubits: 0.29 GB vs 1.09 GB (jax), 1231 → 273 CNOTs, 0.99995 in 2 min. 64-bit training rarely stops by itself → `--max-hours`. Cluster's different numbers (norm violation 0.0000, before-training 0.231) = meson; our laptop run was vacuum.
- [ ] 32 and 120 qubits, 20 steps, on the cluster (`--max-hours 72`).
- [ ] Notebook: 12-qubit Step 5 now trains with the explicit gradient (64-bit, up to 2000 steps, ~1 h). Options: cap `maxiter`, or keep jax for the notebook. Not urgent.

**120 qubits (60 sites, 20 Trotter steps):**
- [x] 1. Build the targets — `results/targets_120qubits_merged/` (not in git), `max_bond` 256 / 512 / 1024, 6.5 h total (1024: 2.7 h each). At 1024: norm 0.985 / 0.988, change 512→1024 0.0054 / 0.0056 (≈ 0.0014 error est.), count = 60.

**Next:**
- [x] 10. Fidelity of the **transpiled** circuit at 12 qubits — done in step 6, holds — snap angles < 1e-6 to 0 before transpiling (Qiskit 2.5.x level-3 bug with near-zero angles, reported by the parallel implementation: fidelity fell to ~0.92 at 40/80 qubits).
- [ ] 11. About 30 qubits (15 sites), 10 steps: target + training, 3- and 4-step starts (compression failed at ~30 qubits before). Train at bond 128, check at higher bond (bond 128 can mislead on hard states).
- [ ] 12. 120-qubit training — needs a cluster; or share the 120-qubit targets + merging trick with the parallel implementation.

**Parallel implementation (2026-10-04 report):** meson, 10 steps, up to 80 qubits at F > 0.999 (4–5-step starts; 80 qubits: 2180 CX / 2q depth 68 at F 0.999425; 21–31 h per run on a cluster). A tweak (one angle held at zero in ~60% of blocks) saves a further 12–14% CNOTs. Not yet: 20 steps at δt = 0.0015, or 120 qubits. Their 12-qubit counts are at optimisation level 1, so not directly comparable with ours (level 3).

**Choices (revisit if needed):** start from all-|0⟩ (X gates inside the circuit); vacuum and meson trained separately.

**Open questions:** Is the negligible mass-term effect expected? How does training time grow with size on a laptop?
