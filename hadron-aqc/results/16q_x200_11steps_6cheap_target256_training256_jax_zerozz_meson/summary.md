# 16q_x200_11steps_6cheap_target256_training256_jax_zerozz_meson

16 qubits (8 sites), x = 200 (kinetic 0.3, electric -0.226125, mass 0.0424264), 11 Trotter steps, 6-step cheap circuit, target max_bond=256, training max_bond=256, gradient=jax. Started 2026-11-02 12:37.

| | meson |
|---|---|
| target health: norm violation / count of filled seats | 0.0000 / 8.0000 (must be 8) |
| angles | 1808 |
| fidelity reached in training (exact, jax) | **0.996206** |
| fidelity before → after training, both simulated at max_bond=256 | 0.8621 → 0.996230 |
| training steps | 82 (converged) |
| CNOTs (original → trained) | 934 → **504** |
| two-qubit depth | 162 → **87** |
| total depth | 383 → 348 |
| fidelity of the transpiled circuit (at max_bond=256) | 0.996230 |
| recheck at max_bond=512 | 0.996230 |
| time | 2.6 min |
