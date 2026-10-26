"""Checks on the hadron circuit itself: is it the group's circuit, and are its start and mass term right?"""
import json
import re

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag

from hadron_aqc_utils import construct_circuit, build_target_using_lib, check_target, strengths_for_x, \
    KINETIC_STRENGTH, ELECTRIC_FIELD_STRENGTH, MASS, TROTTER_TIME_STEP

N_SITES = 6           # 12 qubits: small enough to check in seconds
N_STEPS = 11


def build(num_trotter_steps, use_meson_not_vacuum, n_sites=N_SITES):
    return construct_circuit(n_sites, num_trotter_steps, KINETIC_STRENGTH, ELECTRIC_FIELD_STRENGTH, MASS,
                             use_meson_not_vacuum=use_meson_not_vacuum)


def starting_bits(circuit):
    """Which qubits the circuit flips before the first Trotter step, as a string like '0101 0101 0101'."""
    bits = np.zeros(circuit.num_qubits, dtype=int)
    for instruction in circuit.data:
        if instruction.operation.name != "x":
            break                                            # the X flips all come first
        bits[circuit.find_bit(instruction.qubits[0]).index] ^= 1
    text = "".join(map(str, bits))
    return " ".join(text[i:i + 4] for i in range(0, len(text), 4))


@pytest.mark.parametrize("x, expected", [
    (100, (0.15, -0.226125, 0.03)),         # tracker issue #254
    (200, (0.3, -0.226125, 0.0424264)),     # mass grows with the square root of x, not with x
])
def test_strengths_for_x(x, expected):
    assert strengths_for_x(x) == expected


def test_circuit_matches_reference_files(reference_qasm_folder):
    """Our circuit equals the group's QASM files gate for gate (compared as DAGs: gate order within a layer may differ).
    Each file's x comes from the manifest.json next to it (its kinetic strength c = x * dt)."""
    files = sorted(reference_qasm_folder.glob("**/SU2_T*_*.qasm"))
    assert files, f"no SU2_T*_*.qasm files under {reference_qasm_folder}"
    for file in files:
        reference = QuantumCircuit.from_qasm_file(str(file))   # knows qelib1 gates such as swap
        num_trotter_steps = int(re.search(r"SU2_T(\d+)_", file.name).group(1))   # SU2_T11_... -> 11
        use_meson_not_vacuum = "_MID" in file.name               # MID = hadron, SCV = vacuum
        x = round(json.loads((file.parent / "manifest.json").read_text())["c"] / TROTTER_TIME_STEP)
        kinetic_strength, electric_field_strength, mass = strengths_for_x(x)
        ours = construct_circuit(reference.num_qubits // 2, num_trotter_steps, kinetic_strength, electric_field_strength,
                                 mass, use_meson_not_vacuum=use_meson_not_vacuum)
        assert circuit_to_dag(ours) == circuit_to_dag(reference), f"differs from {file.name} (x = {x})"


@pytest.mark.parametrize("use_meson_not_vacuum, expected", [
    (False, "0101 0101 0101"),    # vacuum: every second qubit flipped
    (True, "0101 1010 0101"),     # hadron: both qubits of the two centre sites flipped on top
])
def test_hadron_start_bits(use_meson_not_vacuum, expected):
    assert starting_bits(build(N_STEPS, use_meson_not_vacuum)) == expected


def test_mass_signs_per_site():
    """The mass Rz gates close every step, one per qubit: - - + + - - ... (same sign inside a site)."""
    one_step = build(1, use_meson_not_vacuum=False)
    mass_gates = one_step.data[-one_step.num_qubits:]
    qubits = [one_step.find_bit(g.qubits[0]).index for g in mass_gates]
    angles = [float(g.operation.params[0]) for g in mass_gates]
    assert all(g.operation.name == "rz" for g in mass_gates)
    assert qubits == list(range(one_step.num_qubits))
    expected = [-MASS if (q // 2) % 2 == 0 else MASS for q in qubits]
    assert np.allclose(angles, expected), f"mass angles {angles}"


@pytest.mark.parametrize("use_meson_not_vacuum", [False, True])
def test_target_is_physical(use_meson_not_vacuum):
    """The full 12-qubit target (exact at bond 64) keeps its norm and holds one filled seat per site."""
    target = build_target_using_lib(N_SITES, N_STEPS, max_bond=64, use_meson_not_vacuum=use_meson_not_vacuum)
    norm_violation, _, count_of_filled_seats = check_target(target)
    assert abs(norm_violation) < 1e-6
    assert count_of_filled_seats == pytest.approx(N_SITES, abs=1e-6)
