# 32q_11steps_5cheap_target1024_training512_jax

32 qubits (16 sites), 11 Trotter steps, 5-step cheap circuit, target max_bond=1024, training max_bond=512, gradient=jax. Started 2026-10-22 10:07.

| | vacuum | meson |
|---|---|---|
| target health: norm violation / count of filled seats | 0.0162 / 16.0000 (must be 16) | 0.0028 / 16.0000 (must be 16) |
| angles | 3412 | 3412 |
| fidelity reached in training (exact, jax) | **0.901400** | **0.943441** |
| fidelity before → after training, both simulated at max_bond=512 | 0.2923 → 0.491491 | 0.4256 → 0.829060 |
| training steps | 37 (stopped by --max-hours 2.0) | 43 (stopped by --max-hours 2.0) |
| CNOTs (original → trained) | 1978 → **1025** | 1978 → **1020** |
| two-qubit depth | 162 → **82** | 162 → **82** |
| total depth | 383 → 278 | 383 → 282 |
| fidelity of the transpiled circuit (at max_bond=512) | 0.502178 | 0.831575 |
| recheck at max_bond=1024 | 0.855459 | 0.936838 |
| time | 139.9 min | 134.5 min |
