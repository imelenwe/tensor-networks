# AQC — spec

Reproducing arXiv:2301.08609 (`AQCtensor`). Idea: build a small trainable circuit, initialize it to match a Trotter circuit, then optimize its parameters against a fidelity cost so it reproduces a target time-evolved state in far fewer layers than plain Trotterization.

## Stack

Python + Qiskit, env `qgss26`. 100% classical sim, no QPU involved.

## v1 scope

1D XYZ chain (Eq. 1) only, validated against exact diagonalization (currently `L=6`).

Implementation lives in `AQC/aqc-phase1-sandbox.ipynb`.

## Steps

1. [x] Hamiltonian + exact time-evolved state — ground truth (`get_hamiltonian`, `get_Neel_State`, `timeevolution`)
2. [x] Bond gate (Fig. 3) — verified against exact `expm` (`get_bond_gate_for_hxyz`)
3. [x] Second-order Trotter circuit (Fig. 4) — generates the target state; fused even/odd schedule + per-site field term, verified 2nd-order convergence (`second_order_trotter_circuit`)
4. [x] Brickwork Ansatz circuit (Fig. 7/8) — trainable circuit, mirrors the Trotter circuit's pair schedule (`aqc_cnot_block`, `aqc_triplet_block`, `aqc_init_layer`, `aqc_field_layer`, `aqc_parametrized_circuit`)
5. [ ] Trotter-init — seed the Ansatz so it starts out identical to the Trotter circuit
6. [ ] Cost function — fidelity between Ansatz output and target state
7. [ ] Optimizer (ADAM → L-BFGS-B) — this step *is* AQC
8. [ ] Compare optimized Ansatz vs. Trotter at equal depth (mini Fig. 9)
9. [ ] Later: real MPS overlap (not exact statevector) + scale toward the paper's qubit counts. Appendix A models (NNN chain, 2D hex lattice) deferred indefinitely.
