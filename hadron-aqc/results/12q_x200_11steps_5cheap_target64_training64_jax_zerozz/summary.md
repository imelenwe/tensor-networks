# 12q_x200_11steps_5cheap_target64_training64_jax_zerozz

12 qubits (6 sites), x = 200 (kinetic 0.3, electric -0.226125, mass 0.0424264), 11 Trotter steps, 5-step cheap circuit, target max_bond=64, training max_bond=64, gradient=jax. Started 2026-10-26 17:47.

| | vacuum | meson |
|---|---|---|
| target health: norm violation / count of filled seats | 0.0000 / 6.0000 (must be 6) | 0.0000 / 6.0000 (must be 6) |
| angles | 1087 | 1087 |
| fidelity reached in training (exact, jax) | **0.994533** | **0.994759** |
| fidelity before → after training, both simulated at max_bond=64 | 0.7971 → 0.994554 | 0.8808 → 0.994778 |
| training steps | 84 (converged) | 134 (converged) |
| CNOTs (original → trained) | 673 → **301** | 673 → **301** |
| two-qubit depth | 162 → **72** | 162 → **72** |
| total depth | 383 → 291 | 383 → 299 |
| fidelity of the transpiled circuit (at max_bond=64) | 0.994554 | 0.994778 |
| recheck at max_bond=128 | 0.994554 | 0.994778 |
| time | 0.2 min | 0.2 min |
