# Code map: the layer gradient

Drawn for the **3-qubit example** of the workbook (`learn/circuit-backprop-workbook.html`, section 6): a pair block,
a SWAP block and an electric block, 3 layers, 31 angles. For 12 or 120 qubits the code is the same; only the counts
grow (more blocks, more layers, more angles).

## Where the chains come from

`tensor_chain.py` doesn't take a given state and make a chain out of it. It is a **toolbox**: four operations on states
that are *already* stored as chains. Two chains get created, at two places:

| chain | created where | how |
|---|---|---|
| S0 = \|0…0⟩, the start | `tensor_chain.py` → `all_zeros_chain` | built from nothing: every qubit "is 0" |
| T, the target | `objective.py` → `target_as_chain` | converted from our quimb target (built earlier by `build_target_using_lib`) |

Every other state (S1, S2, …, and B2, B1, …) comes from applying or undoing blocks on these two, using the toolbox.

## The code map, as one training step

```text
scipy: "here are the 31 angles, give me L and the slopes"
  │
  ▼
objective.py ─ LayerGradientObjective(angles)
  │
  ├─1─► two_qubit_blocks_to_matrices.py
  │       angles → one 4×4 matrix per block (3 blocks)
  │              + each matrix's change per angle
  │
  ├─2─► forward_backward_pass.py
  │       forward:  S0 → S1 → S2 → S3, keep some;  o = ⟨T|S3⟩
  │       backward: layer 3, 2, 1: tables from (S before, B after),
  │                 then undo the layer on B
  │       ──uses──► tensor_chain.py
  │                   all_zeros_chain  (make S0)
  │                   apply_block      (apply / undo one block + trim)
  │                   move_centre      (put the weights where we work)
  │                   overlap          (⟨T|S3⟩)
  │
  └─3─► objective.py ─ loss_and_slopes
          tables × changes → L and all 31 slopes  ──► back to scipy
```

## Each file: what goes in, what comes out

| file | in | out |
|---|---|---|
| `two_qubit_blocks_to_matrices.py` | the trainable circuit + the current angles | 3 matrices (4 × 4) + their changes per angle |
| `forward_backward_pass.py` | the 3 matrices + the target chain T | o + 3 tables (16 numbers each) |
| `tensor_chain.py` | chains + one block | the changed chain, or an overlap number |
| `objective.py` | angles (from scipy) | (L, 31 slopes) (to scipy) |

That is all the method does in one training step. scipy then turns the angles and calls it again for the next step.

## Learning, one file at a time

- [ ] `tensor_chain.py`: the toolbox (start chain, move the weights, apply a block + trim, overlap)
- [ ] `forward_backward_pass.py`: the two passes and the 16-number tables
- [ ] `two_qubit_blocks_to_matrices.py`: circuit boxes → 4×4 matrices + their changes per angle
- [ ] `objective.py`: putting it together, and the chain rule (`loss_and_slopes`)
