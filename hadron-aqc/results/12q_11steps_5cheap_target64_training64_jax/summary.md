# 12q_11steps_5cheap_target64_training64_jax

12 qubits (6 sites), 11 Trotter steps, 5-step cheap circuit, target max_bond=64, training max_bond=64, gradient=jax. Started 2026-10-22 09:58.

| | vacuum | meson |
|---|---|---|
| target health: norm violation / count of filled seats | 0.0000 / 6.0000 (must be 6) | 0.0000 / 6.0000 (must be 6) |
| angles | 1167 | 1167 |
| fidelity reached in training (exact, jax) | **0.995666** | **0.996404** |
| fidelity before → after training, both simulated at max_bond=64 | 0.8133 → 0.995677 | 0.8829 → 0.996415 |
| training steps | 47 (converged) | 191 (converged) |
| CNOTs (original → trained) | 673 → **347** | 673 → **351** |
| two-qubit depth | 162 → **82** | 162 → **82** |
| total depth | 383 → 276 | 383 → 309 |
| fidelity of the transpiled circuit (at max_bond=64) | 0.995677 | 0.996416 |
| recheck at max_bond=128 | 0.995677 | 0.996415 |
| time | 0.4 min | 0.5 min |
