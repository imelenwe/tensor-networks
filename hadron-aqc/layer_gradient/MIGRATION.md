# Migration: layer gradient

**Goal:** a third way to get the training slopes, next to jax and IBM's explicit gradient: the **layer gradient**.
A forward pass keeps the state before every few layers; a **backward layer pass** undoes the layers on the target,
last layer first, and at each layer combines "state before" with "target side after" into a 16-number table per
two-qubit block, which gives the slopes of that block's angles. Both states are chains trimmed to a link size
(`max_bond`), so memory is set by the link size, not by the circuit (workbook: `learn/circuit-backprop-workbook.html`,
sections 6 and 7).

## Steps

| # | step | done when |
|---|---|---|
| 1 | `source/`: a working version, copied unchanged | files byte-identical to the copy we were given |
| 2 | one interface (below), wired into `hadron_aqc_utils.train_to_target` and `scripts/run_hadron_aqc.py --gradient layer` | a 12-qubit run finishes |
| 3 | `tests/test_layer_gradient.py`: generic tests that use only the interface, never `source/` itself | all pass on `source/` |
| 4 | `simplified/`: our rewrite (plain names, one job per file, dead code removed) | the same tests pass on `simplified/` |
| 5 | delete `source/`, move `simplified/` up into `layer_gradient/`, remove the switch | the same tests pass |

While both versions exist, `LAYER_GRADIENT_VERSION=source|simplified` (environment variable) picks one; default `source`.

## The interface (what the tests and the utils file use)

```python
from layer_gradient import LayerGradientSettings, LayerGradientObjective

settings = LayerGradientSettings(max_bond=64, stride=6, use_gpu=False, single_precision=False)
objective = LayerGradientObjective(ansatz, target_mps, settings)   # target_mps: our quimb target
loss, slopes = objective(angles)              # L = 1 - fidelity and dL/d(angle) for every angle
overlaps = objective.layer_overlaps(angles)   # <B_k|S_k> at every layer, from the backward layer pass
```

## Tests (expected numbers)

| test | size | expected |
|---|---|---|
| the backward layer pass keeps the overlap: every `<B_k|S_k>` = the exact overlap | 6 qubits | within 1e-10 |
| loss = exact 1 - fidelity (statevector) | 6 qubits | within 1e-10 |
| slopes = nudging the exact statevector loss | 6 qubits | within 1e-6 |
| saving a state every layer or almost never gives the same answer | 6 qubits | within 1e-10 |
| a small link size really trims (loss differs from exact) | 6 qubits | differs by > 1e-6 |
| loss and all slopes = jax's, hadron circuit | 12 qubits, slow | see the test |
| training with the layer gradient reaches the usual fidelity and CNOTs | 12 qubits, slow | >= 0.994, <= 310 CNOTs |

## Files of the simplified version (learning guide: `simplified/CODE_MAP.md`)

| file | what it does |
|---|---|
| `two_qubit_blocks_to_matrices.py` | reads every box of the trainable circuit (KAK, ZXZ) and merges them into one 4×4 matrix per two-qubit block, plus how that matrix changes per angle |
| `tensor_chain.py` | a state as a chain (one small array per qubit): apply a block, trim to `max_bond`, overlap |
| `forward_backward_pass.py` | the forward pass (keeps every `stride`-th state) and the backward layer pass (undo each layer on T, last first) → one 16-number table per block |
| `objective.py` | what scipy calls each training step: angles in → (L, all slopes) out (the chain rule lives here) |

Function by function, source → simplified: `zero_mps, move_center, apply_two, overlap` → `tensor_chain.py` (`all_zeros_chain,
move_centre, apply_block, overlap`); `layers, apply_layer, _left, _right, _gate_env, layer_envs, mps_forward,
mps_value_and_envs` → `forward_backward_pass.py` (`split_into_layers, apply_layer, _grow_from_left, _grow_from_right,
_table, layer_tables, forward_and_backward`); `qiskit_instrs, build_units, _embed, unit_matrices` →
`two_qubit_blocks_to_matrices.py` (`read_boxes, merge_into_blocks, _on_block, block_matrices`); `loss_and_grad, target_from_quimb,
MPSGradientObjective` → `objective.py` (`loss_and_slopes, target_as_chain, LayerGradientObjective`). Removed (unused, or
replaced by the tests): `apply_one`, `to_dense`, `dense_apply`, `dense_value_and_envs`, `target_from_statevector`.
**A fix in `source/`:** diff it against our copy, find the function above, add a test that shows the bug, port it.

## Status

- [x] 1 · [x] 2 · [x] 3 · [x] 4 (2026-11-02): all 8 tests pass on both versions. Measured: 6-qubit overlaps and loss
      within 6e-15 of exact; 12 qubits vs 64-bit jax: loss 3e-13, slopes 2e-9 (relative); 12-qubit training passes the
      0.994 / 310-CNOT bar (about 6–8 min per state: 64-bit training runs long, 0.4 s per step, mostly building blocks).
- [ ] 5 · on hold: `source/` stays the running version until the simplified code has been read through and the
      source has settled; then delete `source/` and `_source_version.py`, move `simplified/` up, remove the switch.
