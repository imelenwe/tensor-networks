# MPS adjoint gradient for AQC

This computes the AQC loss `1 - |<target|U(x)|0>|^2` and its gradient over **all** ansatz
parameters from one forward and one backward pass through the circuit, with both states held as
MPSs at a fixed bond `chi`. It is backpropagation (the adjoint method) done with tensor networks.

- **Memory** is set by the bond, not by the circuit: a few MPSs at bond `chi`. A 120-qubit run
  at bond 1024 peaked at about 8 GB of host RAM and runs on one GPU. (Your `--gradient jax` run
  at 120 qubits asked for 1.5 TB.)
- **Time**: one gradient costs about two to three MPS runs of the circuit, instead of one overlap
  per parameter (AQC-Tensor's explicit gradient).
- **Accuracy**: approximate, like any bond-limited gradient, because the states are truncated at
  `chi`. Exact when `chi` holds the states.

## Files

| file | what |
|---|---|
| `fastgrad_core.py` | the method: MPS forward pass, backward pass, layer environments, adapter. numpy or cupy, no Qiskit. The exact file used in my runs (sha256 `b8fb18cd…`). |
| `mps_objective.py` | `MPSGradientObjective` for a parametrised Qiskit ansatz, `target_from_quimb`, `target_from_statevector` |
| `check_gradient.py` | self-check against the exact gradient (8 qubits) and a short L-BFGS-B run. Needs numpy, scipy, qiskit |

## Quick start: in place of AQC-Tensor's objective

`obj(x)` returns `(loss, gradient)` like AQC-Tensor's objective, so the scipy call stays as it is:

```python
import numpy as np
from scipy.optimize import minimize
from mps_objective import MPSGradientObjective, target_from_quimb

target = target_from_quimb(target_mps)     # the quimb MPS (or CircuitMPS) you already build as the AQC target
obj = MPSGradientObjective(ansatz, target, chi=512)               # ansatz: parametrised QuantumCircuit
res = minimize(obj, x0, jac=True, method="L-BFGS-B", options={"maxiter": 500})
```

On a GPU, with half the memory:

```python
import cupy as cp
obj = MPSGradientObjective(ansatz, target, chi=1024, xp=cp, dtype=np.complex64, host=True)
```

`host=True` keeps the saved forward states in host RAM instead of GPU memory. `stride` (default 6)
is how often a forward state is saved (see below).

Check the trained circuit at a larger bond than you trained at. The training value is computed on
the truncated state, so it comes out somewhat **below** the true fidelity:

```python
print("F at 2*chi:", 1 - MPSGradientObjective(ansatz, target, chi=2 * 512)(res.x)[0])
```

## How it works

Write `f(x) = <T|U(x)|0>` and the loss `L = 1 - |f|^2`.

**1. Gates become 4x4 units.** Every one-qubit instruction is merged into the two-qubit
instruction next to it on its qubit (`build_units`). The circuit becomes a list of 4x4 blocks
`G_1 ... G_M` on neighbouring sites, and each parameter sits in one or more blocks.

**2. Layers.** The blocks are grouped into layers of disjoint blocks, as early as possible
(`layers`): `U = L_K ... L_2 L_1`.

**3. Forward pass.** `psi_k = L_k ... L_1 |0>` as an MPS, applying each block with an SVD and
truncating at `chi`. Only the state before every `stride`-th layer is kept (`mps_forward`); this
is gradient checkpointing, as in deep learning.

**4. Backward pass.** Start from the target and undo the layers from the top:
`phi_k = L_(k+1)^dagger ... L_K^dagger |T>`, also truncated at `chi`. For every layer,
`f = <phi_k| L_k |psi_(k-1)>`. The forward state `psi_(k-1)` comes from the nearest saved state,
recomputed once per segment of `stride` layers (`mps_value_and_envs`).

**5. Environments.** In `<phi_k| L_k |psi_(k-1)>`, contract the two MPSs site by site from the
left and from the right, keeping every partial result (`layer_envs`). Leaving out one block `G`
of the layer gives its environment, the 4x4 array `E_G = df/dG` (`_gate_env`). One left sweep
and one right sweep give `E_G` for every block in the layer.

**6. Chain rule.** For each block, `dG/dx_p` comes from central differences of the instruction
matrices (`unit_matrices`; 4x4 matrices only, `h = 1e-6`). Then

```text
df/dx_p = sum over blocks G containing p of  sum_ab  E_G[a, b] * (dG/dx_p)[a, b]
dL/dx_p = -2 Re( conj(f) * df/dx_p )                                  (loss_and_grad)
```

`dense_value_and_envs` is the same algorithm on an exact state vector, for checking at small sizes.

## Cost and memory, measured

On the LSH tracker model (SU(2), meson initial state, 10 first-order Trotter steps), at the same
bond and on the same GPU (one call = loss and full gradient):

| per call | AQC-Tensor explicit gradient | this, complex128 | this, complex64 |
|---|---|---|---|
| 24 qubits | 189 s | 60 s | 28 s |
| 120 qubits (H100, bond 1024, 5-layer ansatz, about 13,560 parameters) | 5,120 s | 1,016 s | 655 s |
| gradient error vs the exact gradient, 24 qubits (relative) | 3.6e-2 | 8.5e-3 | 8.5e-3 |

Memory: each saved state is at most `n * chi^2 * 2` numbers (about 2 GB for 120 sites at
`chi = 1024` in complex64, less where the bonds are smaller), so `stride` trades memory for
recomputation. A 120-qubit run peaked at about 8 GB of host memory.

## What to watch

- **The bond must hold the state.** For the 120-qubit tracker circuits, x = 100 trained well at
  bond 128 (with AQC-Tensor's explicit gradient), but x = 200 needed bond 1024 (with this one). `check_gradient.py` shows a random (highly entangled)
  8-qubit target at bond 8 and 4, where the gradient is poor.
- **Ansatz:** two-qubit gates on neighbouring qubits `(q, q + 1)` only (either qubit order), and
  every qubit in at least one two-qubit gate; `build_units` raises an error otherwise. Parameters
  are ordered as `ansatz.parameters`. Parameter expressions are fine.
- **Target:** a list of `(left, physical, right)` arrays, end bonds of size 1, site `q` = qubit `q`
  (what `target_from_quimb` gives for a Qiskit circuit converted to quimb).
- **Two-qubit order inside the core:** a 4x4 block is indexed with the lower site as the more
  significant bit (quimb's order). The adapter converts from Qiskit's order, so this only matters
  if you call the core directly.
- Gate matrices are rebuilt with Qiskit's `Operator` at every call. That costs seconds at 120
  qubits, small next to the MPS work.

## Running the check

```text
$ python check_gradient.py
8 qubits, 111 parameters, 132 instructions
exact (statevector, central differences): loss 0.9960667117
MPS chi=16: loss 0.9960667117  |dloss| 0.0e+00  gradient rel. error 8.8e-09
MPS chi= 8: loss 0.9977941884  |dloss| 1.7e-03  gradient rel. error 4.5e-01
MPS chi= 4: loss 0.9988743071  |dloss| 2.8e-03  gradient rel. error 8.7e-01
L-BFGS-B, 30 iterations: loss 0.9961 -> 0.5272 (exact at the end 0.5272), 36 objective calls
CHECK_OK
```
