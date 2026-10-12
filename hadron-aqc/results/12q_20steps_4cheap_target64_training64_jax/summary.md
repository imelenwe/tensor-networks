# 12q_20steps_4cheap_target64_training64_jax

12 qubits (6 sites), 20 Trotter steps, 4-step cheap circuit, target max_bond=64, training max_bond=64, gradient=jax. Started 2026-10-12 11:40.

| | vacuum | meson |
|---|---|---|
| target health: norm violation / count of filled seats | 0.0000 / 6.0000 (must be 6) | 0.0000 / 6.0000 (must be 6) |
| angles | 933 | 933 |
| fidelity reached in training (exact, jax) | **0.999915** | **0.999867** |
| fidelity before → after training, both simulated at max_bond=64 | 0.7619 → 0.999923 | 0.8158 → 0.999880 |
| training steps | 50 (converged) | 27 (converged) |
| CNOTs (original → trained) | 1231 → **277** | 1231 → **278** |
| two-qubit depth | 297 → **65** | 297 → **65** |
| total depth | 697 → 245 | 697 → 232 |
| fidelity of the transpiled circuit (at max_bond=64) | 0.999923 | 0.999880 |
| recheck at max_bond=128 | 0.999923 | 0.999880 |
| time | 0.3 min | 0.3 min |
