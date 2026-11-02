# 16q_x200_11steps_5cheap_target256_training256_jax_zerozz_meson

16 qubits (8 sites), x = 200 (kinetic 0.3, electric -0.226125, mass 0.0424264), 11 Trotter steps, 5-step cheap circuit, target max_bond=256, training max_bond=256, gradient=jax. Started 2026-11-02 12:19.

| | meson |
|---|---|
| target health: norm violation / count of filled seats | 0.0000 / 8.0000 (must be 8) |
| angles | 1506 |
| fidelity reached in training (exact, jax) | **0.984622** |
| fidelity before → after training, both simulated at max_bond=256 | 0.7082 → 0.984644 |
| training steps | 83 (converged) |
| CNOTs (original → trained) | 934 → **418** |
| two-qubit depth | 162 → **72** |
| total depth | 383 → 285 |
| fidelity of the transpiled circuit (at max_bond=256) | 0.984644 |
| recheck at max_bond=512 | 0.984644 |
| time | 2.5 min |
