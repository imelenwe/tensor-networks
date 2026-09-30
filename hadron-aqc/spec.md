# Hadron AQC — spec

Apply AQCtensor's Algorithm 1 (same method as `AQC/aqctensor-algorithm1.ipynb`) to the SU(2) Loop-String-Hadron (LSH) circuit from IBM's tutorial: run the deep Trotter circuit through TEBD to get a target state, then train a much shallower AQC circuit to reproduce it — same physics, far fewer two-qubit gates.

## Why this matters

The tutorial compresses nothing: it sends the full deep circuit (>3,400 two-qubit gates at 60 qubits) to hardware and fights the noise with mitigation. More sites or more time means more gates, and eventually noise wins. AQC's payoff is the same physics with a fraction of the gates. It also tests whether the compression (proven on the toy XYZ chain) generalises to a real, actively-studied problem.

## Context

- The question: implement AQC-Tensor on this circuit and see what happens with TEBD; later cross-check against a parallel implementation on the same circuit (reportedly better fidelity than "the original AQC" at small qubit counts).
- Deprioritised: matching GitHub issue #227's quimb-reported `Q=59.9985`.
- **Parallel implementation's protocol (2026-09-29):** target = hadron circuit as quimb MPS (SVD cutoff 1e-12); start = shallower Trotter (4–5 steps for a 10-step target) + X layer; each 2q gate → KAK block, 1q gates → rotations, init reproduces shallow circuit exactly; IBM `qiskit-addon-aqc-tensor` `MaximizeStateFidelity` + L-BFGS-B; ansatz MPS capped at χ=128, final fidelity rechecked at 256/512; cost = transpile both circuits to CX on a linear chain (same settings), compare CX counts. Same method as ours; the library automates mapping + gives autodiff gradients. For 3.4: match their CX counting, also run `num_coarse_steps` = 4/5.

## Files

| file | what it is |
|---|---|
| `tutorials/IBM-loop-string-hadron-dynamics.ipynb` | the tutorial, raw reference copy |
| `hadron-aqc/hadron-aqc-algorithm1.ipynb` | **done** — circuit + measurement pipeline rebuilt in our own names, plain-English physics, heatmaps. Verified identical to the tutorial (118 circuit cases + helpers, diff = 0). |
| `hadron-aqc/hadron_aqc_utils.py` | copies of the hadron functions + Algorithm 1's AQC/MPS machinery, for the new notebook. Copies only — **never modify `AQC/aqctensor-algorithm1.ipynb`** (one user-approved exception, 2026-09-29: a single cell drawing the inside of one triplet block, inserted after the building-blocks cell; all other cells unchanged). `unpack_thetas` deliberately left out (tied to XYZ brick-wall layout). |
| `hadron-aqc/hadron-aqc-tebd.ipynb` | **current work** — the AQC-Tensor notebook. Imports from the utils file. **Kernel: `qgss26`** (Py 3.12.13, Qiskit 2.5.0, numpy 2.5.1) — test there, not `qcml-ibmqc`. |

## Key facts

- **Circuit:** `construct_circuit(n_sites, num_trotter_steps, kinetic_strength=0.15, electric_field_strength=0.01, mass=0.03, use_meson_not_vacuum=...)`. 2 qubits per site. X gates for vacuum/meson are inside the circuit, so it starts from all-|0⟩. Step 0 skips its first SWAP layer. All two-qubit gates act on neighbouring qubits.
- **Measurement:** Z per qubit (`SparsePauliOp`, list reversed `[::-1]` for Qiskit's right-to-left order) → `get_physical_particle_count` → `calc_meson_signal(meson_counts, vacuum_counts, n_sites)`.
- **Ground truth at 12 qubits:** `Statevector(construct_circuit(...))`, not an exact Hamiltonian — measures compression error only. (Tutorial's own θ formula disagrees in sign with its code; treat the code as the definition.)
- **Gate counts (6 sites):** raw CNOTs 73 / 155 / 237 / 811 for 1 / 2 / 3 / 10 steps. **Bar to beat: 611** at 12 qubits/10 steps, 3455 at 60 qubits — Qiskit O3 transpiled on the tutorial's linear-chain coupling map (a coupling-map-free transpile cheats by relabelling qubits).
- **Two-qubit gate positions per step:** 23 at step 0 (10 pair, 7 SWAP, 6 electric), 26 afterwards. Our hand-built AQC: 3 CX per position = 147 at k=2. Library (merges, O3 on a line): 129 at k=2, ~200 at k=3.
- **Entanglement grows fast:** bond dimension 48 at 2 steps, 176 at 5, still growing after 10 (quimb needed ≈800 at scale). At 12 qubits max bond is 64, so `max_bond=64` = exact.
- **Triplet block expressibility:** the triplet *alone* fits Pair (0.75) and Electric (0.05) to ~1e-11; SWAP = all-zero triplet. It can NOT absorb a preceding Rz⊗Rz before a Pair gate (miss ~1e-2) → the Rz before each brick is the home for carried mass turns (+ free knobs, no CNOTs).
- **MPS truncation bug** (`apply_2q_gate_mps_tensor`): cuts without canonical form and never renormalises — norm leaks (12-qubit random circuit at χ=16: norm² 0.88, reported fidelity 0.78 vs true 0.89). Irrelevant while χ=64 at 12 qubits; must be fixed before any truncation study.
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

**Route (decided 2026-09-30):** our own TEBD builds the target; IBM's `qiskit-addon-aqc-tensor` 0.3.1 (installed in `qgss26`, adds packages only; pre-install snapshot `.qgss26-before-aqc-library.txt`) builds the trainable circuit + starting angles and trains it against **our** target. Our hand-built mapping (1.3a/1.3b) stays in the notebook as an OPTIONAL "how it works" section; our own starting angles (old 1.4) and finite-difference training are dropped.

**Phase 0 — set up**
- [x] 0.1 `hadron_aqc_utils.py` (copies + `keep_gates_as_blocks` option). 0.2 notebook created.
- [x] 0.3 our MPS simulator = `Statevector` on the hadron circuit (max diff ~4e-15).

**Phase 1 — 12 qubits, exact target**
- [x] 1.1 Target via our TEBD: `vacuum_target_tensors`, `meson_target_tensors` (norm 1, = exact).
- [x] 1.3a/1.3b OPTIONAL — brickwork (49 bricks) + our trainable circuit (147 CX); mapping tables + `brickwork_annotated.png`.
- [x] 1.5a Install the library into `qgss26`; re-tested there.
- [x] 1.5b Library imports + `tensors_to_library_target` → `vacuum_target_mps`, `meson_target_mps`. MPS-only check vs library's own TEBD (`cutoff=0.0`): 1.000000000000 both → our TEBD = IBM's TEBD.
- [x] 1.5c Cheap circuit (`num_coarse_steps = 3`, strengths × 10/3) → `generate_ansatz_from_circuit(..., qubits_initially_zero=True)` → trainable circuit + starting angles. Before-training score 0.851366 / 0.899267 (699 angles). Copy check dropped (library guarantees ansatz(init) = circuit).
- [x] 1.5d Train (`MaximizeStateFidelity` + L-BFGS-B, `jac=True`), vacuum + meson; save trained angles to disk. Result: 0.851→0.999967 (14 steps) / 0.899→0.999986 (25 steps), ~20 s; saved `trained_aqc_hadron_12q_k3.npz`.
- [x] 1.5e CX count (O3, linear chain, seed 0): 10-step 611 → trained 201 / 202. Also vs plain Trotter (table in Key facts).
- [ ] 1.6 *(OPTIONAL — presentation picture)* Heatmap `meson_signal`, t = 1..10 × {vacuum, meson}, trained vs target. Scratch-tested: k(t) = ceil(3t/10); fidelity ≥ 0.991 all t (weakest t=3), max signal diff 0.032; Z read from MPS via `local_expectation` (matches old statevector order).
- [x] 1.7 Conclusions markdown cell at the end of `hadron-aqc-tebd.ipynb`.

**Phase 2 — what TEBD cutting does (the main open question)**
- [ ] 2.1 Fix truncation in utils (canonical form before each cut, renormalise); verify norm = 1. Cross-check against quimb's truncated MPS.
- [ ] 2.2 Targets at `max_bond` 8 / 16 / 32 vs exact: how much TEBD loses.
- [ ] 2.3 Train against each cut target, score against the **exact** state.

**Phase 3 — scale (only if needed)**
- [ ] 3.1 Particle count read straight from the MPS (mind `tensor_idx = L-1-q`, `[::-1]`).
- [ ] 3.2 20–24 qubits, then 60 (`n_sites` even); t = 1..10 × {vacuum, meson}.
- [ ] 3.3 Compare with the parallel implementation under one protocol (same library, CX counting, k = 4/5 too).

**Choices made (revisit if needed):** start from all-|0⟩ (avoid `prep_init_state=False`, which adds SWAPs); structure-following ansatz, not plain brick wall; 10-step target only at first; vacuum and meson trained separately. Worth testing later: a particle-number-conserving ansatz variant (the Pair gate conserves number, the generic triplet doesn't) — may explain the parallel implementation's edge.

**Open questions:** is 12 qubits enough, or 20–24 / 60? Is the negligible mass-term effect expected? ("Original AQC" = IBM's `qiskit-addon-aqc-tensor` — answered by the parallel implementation's protocol.)

**Parked:** why the SWAP layers cross between the string-in and string-out halves.
