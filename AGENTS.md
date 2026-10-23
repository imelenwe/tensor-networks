# AGENTS.md

How to work in this repository. For coding agents (Claude Code, Codex, Cursor, …) and people.
Current state, results and open questions live in each folder's `spec.md`, not here.

## What this repo is

Learning and research on tensor networks, ending in one research task: compressing the SU(2)
loop-string-hadron Trotter circuit (up to 120 qubits) with IBM's AQC-Tensor.

| folder | what it is |
|---|---|
| `tutorials/` | tensors.net exercises; the IBM loop-string-hadron tutorial notebook |
| `AQC/` | AQC Algorithm 1 from scratch (own MPS code), verified at 50 qubits. **Finished: do not modify the notebook.** |
| `hadron-aqc/` | the research task (below) |
| `learn/` | hand-written explainer and status pages, served on GitHub Pages |

### hadron-aqc

| path | role |
|---|---|
| `hadron_aqc_utils.py` | **single source of truth**: circuit, strengths, target, trainable circuit, training, CNOT counts. Change the circuit here only. |
| `scripts/run_hadron_aqc.py` | the long runs (laptop or cluster); `scripts/README.md` has the commands |
| `tests/` | the checks in "Verification" below |
| `notebooks/*-edu.ipynb` | learning notebooks; import from the utils file, never copy its code |
| `notebooks/hadron-aqc-tebd.ipynb` | older results notebook; due for a rewrite, don't extend it |
| `results/<run name>/` | one folder per run: `summary.md`, `angles.npz`, `run.log`, `results.json` |
| `spec.md` | current state, decisions, results |

## Setup

- Environment: conda env `qgss26` (Python 3.12, Qiskit 2.5); versions pinned in `hadron-aqc/scripts/requirements.txt`, plus `pytest` for the tests.
- Run scripts with live output: `conda run -n qgss26 --no-capture-output python -u <script>`
  (without these flags `conda run` holds all output until the end).
- Execute a notebook: `jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=qgss26 <file>`
  (nbconvert is not installed inside `qgss26`; use the base env's `jupyter`).

## Verification

Every change is checked before it is called done. Report the check output, not just "done".

### Which check, when (run only what the change touches)

| what changed | run |
|---|---|
| docs, `spec.md`, `learn/` pages, notebooks' text, results folders | nothing; for pages, the no-names scan |
| plotting, logging, the script's output format | fast tests (`python -m pytest tests`, ~2 s) |
| `hadron_aqc_utils.py` training / counting / trainable-circuit code | fast tests + `--run-slow` (~30 s) |
| the circuit or its strengths (`construct_circuit`, constants) | fast tests + `--run-slow` + reference files |
| reading files, answering questions, checking a running job | nothing |

Run tests once, after the change is finished, not after every edit. If a test fails, fix and rerun only that test
(`python -m pytest tests -k <name>`).

### 1. Before writing code

- Read `spec.md`.
- Write down what the change should do **as a number** (e.g. "CNOTs about 14% lower, fidelity within 0.001").
  If you can't say which number should move, you are not ready to code.

### 2. The tests (from `hadron-aqc/`)

```bash
python -m pytest tests                     # fast checks, ~2 s
python -m pytest tests --run-slow          # + the 30-second training check
HADRON_REFERENCE_QASM=<folder> python -m pytest tests    # + compare with the group's circuit files
```

| test | checks |
|---|---|
| `test_circuit_matches_reference_files` | our circuit = the group's QASM files gate for gate, T1…T11, vacuum and hadron (skipped unless `HADRON_REFERENCE_QASM` points at a local copy; the files are not committed) |
| `test_hadron_start_bits` | 12 qubits start as `0101 0101 0101` (vacuum) and `0101 1010 0101` (hadron) |
| `test_mass_signs_per_site` | mass Rz pattern − − + + … (same sign inside a site) |
| `test_target_is_physical` | the 12-qubit target keeps its norm and holds one filled seat per site |
| `test_trainable_circuit_starts_as_cheap_circuit` | at its starting angles, the trainable circuit is exactly the cheap circuit |
| `test_trainable_circuit_shape` | 1,167 angles; two-qubit blocks: 50 pair, 30 electric, 47 SWAP |
| `test_zero_out_zz` | zeroing out ZZ fixes exactly the 80 pair + electric ZZ angles (1,087 angles left), no SWAP's |
| `test_quick_training` (slow) | 12-qubit training still reaches fidelity ≥ 0.995 with ≤ 360 CNOTs (≥ 0.994, ≤ 310 with ZZ zeroed out) |

- **Expected numbers in the tests change only on purpose**, and the commit message says why.

### 3. Before a long run

- Run the same command small first (12 → 20 → 32 qubits); note step time and RAM at each size.
- Check RAM fits: real RAM ≈ 7–9× the planner's "all tables at once" figure. A 24 GB laptop is too small
  for 32 qubits with a 7-step start; use the cluster.

### 4. During and after a long run

- Target line: `norm violation` below 0.01 (≈ 0.012–0.016 is known at 120 qubits, bond 1024) and
  `count of filled seats` equal to the number of sites.
- Training: fidelity rising each step; the run ends with `converged` or `stopped by --max-hours`.
  A log that just stops with no Python error means the process was killed (usually out of RAM).
- With `--gradient jax`, trust "fidelity reached in training (exact)"; the bond-limited rechecks read low
  when the state needs a bigger bond.
- Compare with the previous results, then record the result in `spec.md`.

### Pages (`learn/`)

No personal names; links work; readable at phone width; dark mode works.

## Rules

- **No personal names anywhere**: code, notebooks, specs, pages, commit messages. The repo is public.
  Refer to people as "the group" or "a collaborator".
- **Before every commit**, scan the staged diff for names, home-folder paths (`/Users/…`) and large files.
- **Never commit:** `target_*.pkl` (0.5–3 GB), `contraction_plans/`, `*.log`, cluster job output (holds usernames),
  the group's QASM files.
- **Commit and push only when the user asks.**
- Keep `spec.md` short: current state, not history.

## Gotchas (each one cost us hours)

- **Compare circuits as DAGs** (`circuit_to_dag(a) == circuit_to_dag(b)`), never as QASM text: the same circuit can
  list gates in a different order. Load QASM with `QuantumCircuit.from_qasm_file` (the strict `qasm2.load` rejects `swap`).
- **The cotengra planner only during training.** `train_to_target(..., planner=...)` turns it on inside a `with` block.
  Turned on globally, it makes target building crawl (50 min lost once).
- **jax is 32-bit by default.** `JAX_ENABLE_X64=True` doubles RAM and rarely stops by itself (use `--max-hours`).
- **Snap tiny angles to 0** (`snap_tiny_angles`) before a level-3 transpile; a Qiskit bug otherwise.
- **Mass sign is per site** (− − + + …), not per qubit; the per-qubit version made the mass term do nothing.
- **Hadron start:** site s sits on qubits (2s, 2s+2) if s is even, (2s−1, 2s+1) if odd.
- **Two-qubit blocks cost 3 CNOTs** in IBM's trainable circuit (exp(i(a XX + b YY + c ZZ))); a block with c = 0
  needs only 2 (`--zero-out-zz`). Look at what each building block costs before tuning anything else.

## Style

- Plain, simple English in pages, notebooks and comments; define a technical term the first time.
- Explain a circuit through its diagram before any physics.
- Variable names say what they are (`kinetic_strength`, not `x`); translate symbols from papers into ours.
- Learn pages: one topic per page, a left navigation pane for a series, dark mode, works on a phone.
