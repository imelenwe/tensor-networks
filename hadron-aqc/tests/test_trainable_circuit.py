"""Checks on the trainable circuit IBM's library builds from our cheap circuit, and (slow) on training itself.

The expected numbers below are for the current circuit (12 qubits, 11 steps, 5-step start).
Change them only on purpose, and say why in the commit message.
"""
from collections import Counter

import numpy as np
import pytest
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector
from qiskit_addon_aqc_tensor.ansatz_generation import KAK

from hadron_aqc_utils import construct_circuit, build_cheap_trainable_circuit, build_target_using_lib, \
    make_simulator_settings, train_to_target, count_and_verify, KINETIC_STRENGTH, ELECTRIC_FIELD_STRENGTH, MASS

N_SITES, N_STEPS, N_COARSE_STEPS = 6, 11, 5

EXPECTED_ANGLES = 1167
EXPECTED_BLOCKS = {"pair": 50, "electric": 30, "swap": 47}   # two-qubit blocks, sorted by their starting (a, b, c)
EXPECTED_ANGLES_ZERO_OUT_ZZ = 1087                         # 1167 minus the 80 ZZ angles fixed at 0 (50 pair + 30 electric)
# slow check, as {zero_out_zz: (lowest fidelity, most CNOTs)}; measured: off 0.9957 / 0.9960 with 347 / 350 CNOTs,
# on 0.9950 / 0.9945 with 301 / 301 CNOTs (vacuum / meson)
EXPECTED_TRAINING = {False: (0.995, 360), True: (0.994, 310)}


def block_kind(a, b, c):
    """Name a two-qubit block by its starting XX, YY, ZZ strengths (a, b, c)."""
    if np.allclose([a, b, c], np.pi / 4):
        return "swap"                       # a SWAP needs all three at pi/4
    if abs(c) < 1e-9 and abs(b) < 1e-9:
        return "electric"                   # only an XX part
    if abs(c) < 1e-9 and np.isclose(a, b):
        return "pair"                       # equal XX and YY: the hop
    return "other"


@pytest.mark.parametrize("zero_out_zz", [False, True])
def test_trainable_circuit_starts_as_cheap_circuit(zero_out_zz):
    """At its starting angles, the trainable circuit is exactly the cheap 5-step circuit (zeroing out ZZ changes nothing here)."""
    scale = N_STEPS / N_COARSE_STEPS
    for use_meson_not_vacuum in [False, True]:
        ansatz, starting_angles = build_cheap_trainable_circuit(N_SITES, N_STEPS, N_COARSE_STEPS, use_meson_not_vacuum,
                                                                zero_out_zz=zero_out_zz)
        cheap_circuit = construct_circuit(N_SITES, N_COARSE_STEPS, KINETIC_STRENGTH * scale,
                                          ELECTRIC_FIELD_STRENGTH * scale, MASS * scale,
                                          use_meson_not_vacuum=use_meson_not_vacuum)
        trainable_state = Statevector(ansatz.assign_parameters(starting_angles))
        cheap_state = Statevector(cheap_circuit)
        fidelity = abs(trainable_state.inner(cheap_state)) ** 2
        assert fidelity == pytest.approx(1, abs=1e-8)


def test_trainable_circuit_shape():
    """Fingerprint: number of angles and two-qubit blocks of each kind."""
    ansatz, starting_angles = build_cheap_trainable_circuit(N_SITES, N_STEPS, N_COARSE_STEPS, use_meson_not_vacuum=True)
    starting_angles = np.asarray(starting_angles)
    position = {parameter: i for i, parameter in enumerate(ansatz.parameters)}
    kinds = Counter()
    for instruction in ansatz.data:
        if isinstance(instruction.operation, KAK):
            a, b, c = (starting_angles[position[p]] for p in instruction.operation.params)
            kinds[block_kind(a, b, c)] += 1
    assert len(starting_angles) == EXPECTED_ANGLES
    assert dict(kinds) == EXPECTED_BLOCKS


def test_zero_out_zz():
    """Zeroing out ZZ fixes exactly the ZZ angle of the pair and electric blocks, never a SWAP's."""
    ansatz, starting_angles = build_cheap_trainable_circuit(N_SITES, N_STEPS, N_COARSE_STEPS, use_meson_not_vacuum=True,
                                                            zero_out_zz=True)
    fixed_zz = [instruction for instruction in ansatz.data if isinstance(instruction.operation, KAK)
                and not isinstance(instruction.operation.params[2], Parameter)]    # a plain number now, not a knob
    assert len(starting_angles) == EXPECTED_ANGLES_ZERO_OUT_ZZ
    assert len(fixed_zz) == EXPECTED_BLOCKS["pair"] + EXPECTED_BLOCKS["electric"]


@pytest.mark.slow
@pytest.mark.parametrize("zero_out_zz", [False, True])
@pytest.mark.parametrize("use_meson_not_vacuum", [False, True])
def test_quick_training(use_meson_not_vacuum, zero_out_zz):
    """The 30-second training at 12 qubits still reaches its usual fidelity and compression."""
    target = build_target_using_lib(N_SITES, N_STEPS, max_bond=64, use_meson_not_vacuum=use_meson_not_vacuum)
    ansatz, starting_angles = build_cheap_trainable_circuit(N_SITES, N_STEPS, N_COARSE_STEPS, use_meson_not_vacuum,
                                                            zero_out_zz=zero_out_zz)
    settings = make_simulator_settings(max_bond=64, gradient="jax")
    result = train_to_target(ansatz, starting_angles, target, settings)
    fidelity = 1 - result.fun
    counts = count_and_verify(ansatz, result.x, target, settings)
    lowest_fidelity, most_cnots = EXPECTED_TRAINING[zero_out_zz]
    assert fidelity >= lowest_fidelity
    assert counts["cnots"] <= most_cnots
    assert counts["transpiled_fidelity"] == pytest.approx(fidelity, abs=1e-3)
