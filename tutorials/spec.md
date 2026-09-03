# Tensor Networks — spec / design notes

Read this file first in any new session on this repo. Location: `tutorials/spec.md` — tutorial work lives inside `tutorials/`. Working agreement (brainstorm first, print code, commit only when told) is in the user's global CLAUDE.md — not repeated here.

## Purpose

Exercises and notes following the [tensors.net](https://www.tensors.net) tutor series. Same 4 tutorials exist in three languages on the site (`p-tutorial-N` Python, `tutorial-N` MATLAB, `j-tutorial-N` Julia) — plan is Python for all 4, plus Julia for 1–2 of them as a learning goal.

## Tutorials & languages

| # | Title | URL | Language | Status |
|---|---|---|---|---|
| 1 | Tensor Contractions | https://www.tensors.net/p-tutorial-1 | Python | not started |
| 2 | Tensor Decompositions | https://www.tensors.net/p-tutorial-2 | Python | not started |
| 3 | Gauge Freedom | https://www.tensors.net/p-tutorial-3 | Python | not started |
| 4 | Canonical Forms | https://www.tensors.net/p-tutorial-4 | Python | not started |

Julia pass: which 1–2 tutorials to redo in Julia — not decided yet.

Update this table + README's "Tutorials followed" checklist together as tutorials are completed.

## Environment

- Python: conda env `qgss26` (Python 3.12.13). Jupyter itself runs from base conda env (has JupyterLab); kernelspecs live in the shared `~/Library/Jupyter/kernels` dir, so all kernels show up regardless of which env launches Jupyter (also true for VS Code's Jupyter extension).
- Julia: installed via `juliaup` (official installer, kept fully separate from conda). Julia 1.12.7. `IJulia` kernel (`julia-1.12`) registered and visible alongside `qgss26`.
- One notebook = one kernel = one language; Python and Julia versions of a tutorial are separate `.ipynb` files, not mixed in one notebook.
- Qiskit not used in this repo — tutorials are plain tensor/array exercises (numpy in Python; base arrays / `ITensors.jl` in Julia if needed later).

## Notebooks

- `tutorials/tutorial-1-contractions.ipynb` — Python, kernel `qgss26`
- `tutorials/tutorial-1-contractions-julia.ipynb` — Julia, kernel `julia-1.12`

Both currently just a title cell + link to the tensors.net page — content not yet written.

## Next steps

- Work through Tutorial 1 (Tensor Contractions) in Python first.
- Decide which 1–2 tutorials get a Julia pass.
- Fill in README's "Contents" section as exercises get added.
