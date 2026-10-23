# 12q_11steps_5cheap_target64_training64_jax_zerozz

12 qubits (6 sites), 11 Trotter steps, 5-step cheap circuit, target max_bond=64, training max_bond=64, gradient=jax. Started 2026-10-23 19:39.

| | vacuum | meson |
|---|---|---|
| target health: norm violation / count of filled seats | 0.0000 / 6.0000 (must be 6) | 0.0000 / 6.0000 (must be 6) |
| angles | 1087 | 1087 |
| fidelity reached in training (exact, jax) | **0.995037** | **0.994435** |
| fidelity before → after training, both simulated at max_bond=64 | 0.8133 → 0.995057 | 0.8829 → 0.994456 |
| training steps | 73 (converged) | 81 (converged) |
| CNOTs (original → trained) | 673 → **301** | 673 → **301** |
| two-qubit depth | 162 → **72** | 162 → **72** |
| total depth | 383 → 298 | 383 → 291 |
| fidelity of the transpiled circuit (at max_bond=64) | 0.995057 | 0.994456 |
| recheck at max_bond=128 | 0.995057 | 0.994456 |
| time | 0.2 min | 0.2 min |
