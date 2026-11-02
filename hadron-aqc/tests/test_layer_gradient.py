"""The layer gradient (layer_gradient/): its value, its slopes and its backward layer pass, against exact answers.

Uses only the interface in layer_gradient/__init__.py, so the same tests check every version of the code
(layer_gradient/MIGRATION.md). Exact answers come from Qiskit statevectors (6 qubits) and from jax (12 qubits).
"""
import numpy as np
import pytest
from qiskit.quantum_info import Statevector
from qiskit_addon_aqc_tensor.objective import MaximizeStateFidelity

from hadron_aqc_utils import construct_circuit, build_cheap_trainable_circuit, build_target_using_lib, \
    make_simulator_settings, train_to_target, count_and_verify, KINETIC_STRENGTH, ELECTRIC_FIELD_STRENGTH, MASS
from layer_gradient import LayerGradientObjective, LayerGradientSettings

SMALL_SITES, SMALL_STEPS, SMALL_COARSE_STEPS = 3, 4, 2   # 6 qubits: exact answers from the statevector are instant
N_SITES, N_STEPS, N_COARSE_STEPS = 6, 11, 5               # 12 qubits, as in test_trainable_circuit.py
EXACT_BOND = 64            # holds every 6- and 12-qubit state exactly (the largest link at 12 qubits is 2^6 = 64)
NUDGE = 1e-5               # for slopes by nudging the exact loss: (L(angle + NUDGE) - L(angle - NUDGE)) / (2 NUDGE)
# slow check, as (lowest fidelity, most CNOTs): the same bar as jax with ZZ zeroed out (test_trainable_circuit.py)
EXPECTED_TRAINING = (0.994, 310)


@pytest.fixture(scope="module")
def small_problem():
    """6 qubits, meson, ZZ zeroed out: the trainable circuit at angles moved off its start (so no slope is 0 by
    symmetry), and the 4-step circuit as the target, both as a chain (for the layer gradient) and exactly."""
    ansatz, starting_angles = build_cheap_trainable_circuit(SMALL_SITES, SMALL_STEPS, SMALL_COARSE_STEPS,
                                                            use_meson_not_vacuum=True, zero_out_zz=True)
    angles = np.asarray(starting_angles) + np.random.default_rng(0).normal(0, 0.1, len(starting_angles))
    target_mps = build_target_using_lib(SMALL_SITES, SMALL_STEPS, EXACT_BOND, use_meson_not_vacuum=True)
    target_state = Statevector(construct_circuit(SMALL_SITES, SMALL_STEPS, KINETIC_STRENGTH, ELECTRIC_FIELD_STRENGTH,
                                                 MASS, use_meson_not_vacuum=True))
    return ansatz, angles, target_mps, target_state


def exact_overlap(ansatz, angles, target_state):
    """<T|S>: the target against the trainable circuit's final state, both as full statevectors."""
    return target_state.inner(Statevector(ansatz.assign_parameters(angles)))


def exact_loss(ansatz, angles, target_state):
    return 1 - abs(exact_overlap(ansatz, angles, target_state)) ** 2


def test_backward_layer_pass_keeps_the_overlap(small_problem):
    """Undoing the same layer on both sides keeps the overlap: <B_k|S_k> is the same number at every layer."""
    ansatz, angles, target_mps, target_state = small_problem
    objective = LayerGradientObjective(ansatz, target_mps, LayerGradientSettings(max_bond=EXACT_BOND))
    overlaps = objective.layer_overlaps(angles)
    expected = exact_overlap(ansatz, angles, target_state)
    assert len(overlaps) > 10                       # really many layers, not one
    for overlap in overlaps:
        assert overlap == pytest.approx(expected, abs=1e-10)


def test_loss_matches_exact(small_problem):
    ansatz, angles, target_mps, target_state = small_problem
    objective = LayerGradientObjective(ansatz, target_mps, LayerGradientSettings(max_bond=EXACT_BOND))
    loss, _ = objective(angles)
    assert loss == pytest.approx(exact_loss(ansatz, angles, target_state), abs=1e-10)


def test_slopes_match_nudging(small_problem):
    """Slopes of 30 angles (chosen at random, seeded) against nudging the exact loss."""
    ansatz, angles, target_mps, target_state = small_problem
    objective = LayerGradientObjective(ansatz, target_mps, LayerGradientSettings(max_bond=EXACT_BOND))
    _, slopes = objective(angles)
    for i in np.random.default_rng(1).choice(len(angles), size=30, replace=False):
        up, down = angles.copy(), angles.copy()
        up[i] += NUDGE
        down[i] -= NUDGE
        nudged_slope = (exact_loss(ansatz, up, target_state) - exact_loss(ansatz, down, target_state)) / (2 * NUDGE)
        assert slopes[i] == pytest.approx(nudged_slope, abs=1e-6), f"angle {i}"


def test_saved_states_do_not_change_the_answer(small_problem):
    """Keeping the state before every layer, or only before the first, gives the same loss and slopes
    (the states in between are rebuilt in the backward layer pass)."""
    ansatz, angles, target_mps, _ = small_problem
    every_layer = LayerGradientObjective(ansatz, target_mps, LayerGradientSettings(max_bond=EXACT_BOND, stride=1))
    first_only = LayerGradientObjective(ansatz, target_mps, LayerGradientSettings(max_bond=EXACT_BOND, stride=1000))
    loss_every, slopes_every = every_layer(angles)
    loss_first, slopes_first = first_only(angles)
    assert loss_first == pytest.approx(loss_every, abs=1e-10)
    assert np.allclose(slopes_first, slopes_every, rtol=0, atol=1e-10)


def test_small_bond_trims(small_problem):
    """max_bond is applied: at link size 2 the loss is no longer exact."""
    ansatz, angles, target_mps, target_state = small_problem
    objective = LayerGradientObjective(ansatz, target_mps, LayerGradientSettings(max_bond=2))
    loss, _ = objective(angles)
    assert abs(loss - exact_loss(ansatz, angles, target_state)) > 1e-6


@pytest.fixture
def jax_in_64_bit():
    """jax runs in 32-bit by default: at 12 qubits its loss is off by about 5e-5. Compare in 64-bit, then switch back."""
    import jax
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", False)


@pytest.mark.slow
def test_slopes_match_jax_12_qubits(jax_in_64_bit):
    """Our hadron circuit, 12 qubits, meson, ZZ zeroed out: loss and all 1,087 slopes against jax (the library's
    exact gradient, in 64-bit), at the starting angles and at angles moved off them.
    Measured with the first version: loss within 3e-13, slopes within 2e-9 (relative)."""
    target_mps = build_target_using_lib(N_SITES, N_STEPS, EXACT_BOND, use_meson_not_vacuum=True)
    ansatz, starting_angles = build_cheap_trainable_circuit(N_SITES, N_STEPS, N_COARSE_STEPS, use_meson_not_vacuum=True,
                                                            zero_out_zz=True)
    jax_objective = MaximizeStateFidelity(target_mps, ansatz, make_simulator_settings(max_bond=EXACT_BOND, gradient="jax"))
    layer_objective = LayerGradientObjective(ansatz, target_mps, LayerGradientSettings(max_bond=EXACT_BOND))
    moved_angles = np.asarray(starting_angles) + np.random.default_rng(0).normal(0, 0.1, len(starting_angles))
    for angles in (np.asarray(starting_angles), moved_angles):
        jax_loss, jax_slopes = jax_objective.loss_function(angles)
        loss, slopes = layer_objective(angles)
        assert len(slopes) == 1087
        assert loss == pytest.approx(float(jax_loss), abs=1e-10)
        assert np.linalg.norm(slopes - jax_slopes) / np.linalg.norm(jax_slopes) < 1e-7


@pytest.mark.slow
@pytest.mark.parametrize("use_meson_not_vacuum", [False, True])
def test_quick_training_layer_gradient(use_meson_not_vacuum):
    """12-qubit training with the layer gradient (ZZ zeroed out) reaches the same fidelity and compression as jax."""
    target = build_target_using_lib(N_SITES, N_STEPS, max_bond=EXACT_BOND, use_meson_not_vacuum=use_meson_not_vacuum)
    ansatz, starting_angles = build_cheap_trainable_circuit(N_SITES, N_STEPS, N_COARSE_STEPS, use_meson_not_vacuum,
                                                            zero_out_zz=True)
    settings = make_simulator_settings(max_bond=EXACT_BOND, gradient="layer")
    result = train_to_target(ansatz, starting_angles, target, settings,
                             layer_gradient=LayerGradientSettings(max_bond=EXACT_BOND))
    fidelity = 1 - result.fun
    counts = count_and_verify(ansatz, result.x, target, settings)
    lowest_fidelity, most_cnots = EXPECTED_TRAINING
    assert fidelity >= lowest_fidelity
    assert counts["cnots"] <= most_cnots
    assert counts["transpiled_fidelity"] == pytest.approx(fidelity, abs=1e-3)
