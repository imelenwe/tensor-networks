# AQC — spec

Reproducing arXiv:2301.08609 (`AQCtensor`). Idea: build a small trainable circuit, initialize it to match a Trotter circuit, then optimize its parameters against a fidelity cost so it reproduces a target time-evolved state in far fewer layers than plain Trotterization.

## Stack

Python + Qiskit, env `qgss26`. 100% classical sim, no QPU involved.

## v1 scope

1D XYZ chain (Eq. 1). Small-`L` pieces validated against exact diagonalization (`L=6`); training now runs fully MPS-native at `L=50`.

Implementation lives in `AQC/aqctensor-algorithm1.ipynb`.

## Steps (paper's Algorithm 1 = steps 10, 11-13, 14 below)

1. [x] Hamiltonian + exact time-evolved state — ground truth (`get_hamiltonian`, `get_Neel_State`, `timeevolution`)
2. [x] Bond gate (Fig. 3) — verified against exact `expm` (`get_bond_gate_for_hxyz`)
3. [x] Second-order Trotter circuit (Fig. 4) — generates the target state (`second_order_trotter_circuit`)
4. [x] Brickwork Ansatz circuit (Fig. 7/8) — trainable circuit, mirrors the Trotter circuit's pair schedule (`aqc_parametrized_circuit` + helpers)
5. [x] Trotter-init — seed the Ansatz so it starts out identical to the Trotter circuit (`trotter_aqc_init_params`)
6. [x] Cost function (`1 - fidelity`) — exact-statevector version (`aqc_cost`)
7. [x] Optimizer (`L-BFGS-B`) + first real comparison, small `L` (`train_aqc`, `compare_aqc_vs_trotter`)
8. [x] MPS-native overlap, verified exact match against statevector fidelity (`statevector_to_mps`, `mps_overlap`, `aqc_cost_using_mps`)
9. [x] MPS-native gate application (`apply_1q_gate_to_mps_tensor`, `apply_2q_gate_mps_tensor`, `circuit_to_mps_tensors`) — verified against `Statevector.evolve`
10. [x] TEBD: build the target state at scale as an MPS, no `2^L` statevector (`get_Neel_State_as_mps`, `circuit_to_mps_tensors` on the Trotter circuit) — **Algorithm 1, step 1**
11. [x] Fully MPS-native cost function — AQC ansatz output built directly as MPS, no `Statevector.evolve` anywhere (`aqc_cost_tebd_mps`)
12. [x] Training driver with no exact diagonalization — takes a pre-built TEBD target instead of computing it internally (`train_aqc_mps_at_scale`) — **Algorithm 1, step 2**
13. [x] Ran training end-to-end at `L=50`: shallow ansatz (`steps_aqc=2`, 1826 params) trained against a deep TEBD target (`steps_tebd=10`) — **fidelity = 0.999495**, cost = 0.000505. Trained circuit: depth 22, 369 two-qubit gates vs. Trotter's depth 41, 1545 two-qubit gates (~2.3x fewer gates, ~1.9x shallower). Trained parameters persisted to a dated `.npz` file (`trained_aqc_L{L}_stepsaqc{steps_aqc}_stepstebd{steps_tebd}_{timestamp}.npz`) so the circuit can be reloaded/inspected without rerunning the ~24 min optimization.
14. [x] **(Algorithm 1, step 3)** Append `k` additional Trotter steps onto the trained circuit (`append_trotter_steps`). Verified two ways: exact-diagonalization correctness check at small `L` (fidelity 0.999989 extending to a later time), and a real `L=50` scale demo appending 5 steps (fidelity 0.999467; extended circuit depth 43 / 1489 gates vs. an equivalent from-scratch Trotter circuit's depth 61 / 2260 gates at the same total time).

**Algorithm 1 is now complete end to end.** See the notebook's final "Conclusion" markdown cell for the full summary.
