# Spec / session continuity notes — tensor-networks

Read this file first in any new session on this repo, to pick up where the last session left off.

## Purpose

Exercises and notes following the [tensors.net](https://www.tensors.net) **Python** tutorial series (the `p-tutorial-N` pages — there are also MATLAB `tutorial-N` and Julia `j-tutorial-N` variants of the same 4 tutorials, not being followed here).

## Repo setup (done)

- Local folder: `~/tensor-networks`
- GitHub: public repo at https://github.com/imelenwe/tensor-networks, remote `origin`, branch `main`
- `.gitignore`: standard Python template (from github/gitignore)
- No license
- Initial commit made and pushed: `README.md` + `.gitignore`

## Tutorial series being followed

| # | Title | URL | Status |
|---|---|---|---|
| 1 | Tensor Contractions | https://www.tensors.net/p-tutorial-1 | not started |
| 2 | Tensor Decompositions | https://www.tensors.net/p-tutorial-2 | not started |
| 3 | Gauge Freedom | https://www.tensors.net/p-tutorial-3 | not started |
| 4 | Canonical Forms | https://www.tensors.net/p-tutorial-4 | not started |

Each is listed with a checkbox in `README.md` under "Tutorials followed" — update both files together as tutorials are completed.

## Outstanding as of this session

- `README.md` was edited locally (expanded the tutorial checklist from just #1 to the full 1–4 series) but **not yet committed/pushed** — waiting on explicit go-ahead per the working agreement below.

## Working agreement for this repo (from user's global CLAUDE.md)

- Brainstorm/design first, before writing code.
- Spec-driven: keep this file (or similar plan docs) current before implementation.
- Print code in the response — do not edit files directly — until the user explicitly gives permission. (This applies to actual tutorial/exercise code the user is learning by retyping; repo scaffolding like this spec file is not code and can be written directly when asked.)
- Explain every new code block line by line, every time, no exceptions.
- Commit only when explicitly told to. Never push without being told to.
- Plain language, minimal jargon.

## Next steps

- User will work through `p-tutorial-1` (Tensor Contractions) first.
- Commit the pending README update when told.
- As exercises get added, fill in the README's "Contents" section and this file's tutorial status table.
