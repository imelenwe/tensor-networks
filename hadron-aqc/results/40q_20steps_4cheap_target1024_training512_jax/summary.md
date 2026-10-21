# 40q_20steps_4cheap_target1024_training512_jax

40 qubits (20 sites), 20 Trotter steps, 4-step cheap circuit, target max_bond=1024, training max_bond=512, gradient=jax. Started 2026-10-16 14:24.

| | vacuum | meson |
|---|---|---|
| target health: norm violation / count of filled seats | 0.0027 / 20.0000 (must be 20) | 0.0002 / 20.0000 (must be 20) |
| angles | 3446 | 3446 |
| fidelity reached in training (exact, jax) | **0.977486** | **0.985172** |
| fidelity before → after training, both simulated at max_bond=512 | 0.1255 → 0.601005 | 0.1734 → 0.917455 |
| training steps | 97 (converged) | 89 (stopped by --max-hours 1.5) |
| CNOTs (original → trained) | 4570 → **1039** | 4570 → **1030** |
| two-qubit depth | 297 → **65** | 297 → **65** |
| total depth | 697 → 236 | 697 → 235 |
| fidelity of the transpiled circuit (at max_bond=512) | 0.581143 | 0.922724 |
| recheck at max_bond=1024 | 0.960030 | 0.983186 |
| time | 106.1 min | 110.0 min |
