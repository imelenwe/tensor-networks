# 12q_x100_11steps_5cheap_target64_training64_jax_zerozz

12 qubits (6 sites), x = 100 (kinetic 0.15, electric -0.226125, mass 0.03), 11 Trotter steps, 5-step cheap circuit, target max_bond=64, training max_bond=64, gradient=jax. Started 2026-10-26 17:46.

| | vacuum | meson |
|---|---|---|
| target health: norm violation / count of filled seats | 0.0000 / 6.0000 (must be 6) | 0.0000 / 6.0000 (must be 6) |
| angles | 1087 | 1087 |
| fidelity reached in training (exact, jax) | **0.999369** | **0.999498** |
| fidelity before → after training, both simulated at max_bond=64 | 0.9445 → 0.999386 | 0.9657 → 0.999517 |
| training steps | 88 (converged) | 154 (converged) |
| CNOTs (original → trained) | 673 → **301** | 673 → **301** |
| two-qubit depth | 162 → **72** | 162 → **72** |
| total depth | 362 → 296 | 362 → 295 |
| fidelity of the transpiled circuit (at max_bond=64) | 0.999386 | 0.999517 |
| recheck at max_bond=128 | 0.999386 | 0.999517 |
| time | 0.2 min | 0.2 min |
