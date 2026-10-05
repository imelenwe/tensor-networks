"""Shared functions for the hadron AQC-Tensor notebook (hadron-aqc-tebd.ipynb).

Copied verbatim from:
  - hadron-aqc/hadron-aqc-algorithm1.ipynb  (hadron circuit + measurement helpers)
  - AQC/aqctensor-algorithm1.ipynb          (AQC building blocks + MPS machinery)
Those notebooks are not modified; edits to the copies here stay local to this file.

Local edits: pair/electric circuits are named ("pair", "electric"), and construct_circuit has
keep_gates_as_blocks (default False = identical output to the original) so bricks can be read back.

The last section ("Library route") is new: the AQC recipe written once, using IBM's
qiskit-addon-aqc-tensor + quimb, shared by every qubit-size section and the cluster script.
"""
from typing import Optional

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator


# ---------------- hadron circuit (from hadron-aqc-algorithm1.ipynb) ----------------

def pair_hamiltonian_circuit(rzparam: float) -> QuantumCircuit:
    qc = QuantumCircuit(2, name="pair") # name lets build_hadron_brickwork find it when kept as one block
    qc.cx(1, 0)
    qc.h(1)
    qc.rz(-rzparam, 1)
    qc.cx(0,1)
    qc.rz(rzparam, 1)
    qc.cx(0, 1)
    qc.h(1)
    qc.cx(1,0)
    return qc


def electric_hamiltonian_circuit(theta: float) -> QuantumCircuit:
    qc = QuantumCircuit(2, name="electric") # name lets build_hadron_brickwork find it when kept as one block
    qc.x(0)
    qc.rz(theta/2, 0)
    qc.cx(0,1)
    qc.rz(-theta/2, 1)
    qc.cx(0,1)
    qc.rz(theta/2, 1)
    qc.x(0)
    return qc


def construct_circuit(
n_sites,
num_trotter_steps, 
kinetic_strength,
electric_field_strength,
mass, 
qubits_per_site=2,
add_visual_barriers = False,
add_measurements = False,
prep_init_state=True,
use_meson_not_vacuum=False,
keep_gates_as_blocks=False, # True: pair/electric gates go in as one named block each (wrap=True), so their positions can be read back
):
    num_qubits = qubits_per_site * n_sites
    qc = QuantumCircuit(num_qubits)
    if num_trotter_steps <=0:
        return qc
    # vaccum state Step 1: At even sites we expect quarks. At odd sites we expect antiquarks. The vacuum is the starting state where neither is present yet. To say "quark not present" on an even site, we set it to (0,0). To say "antiquark not present" on an odd site, we set it to (1,1). That's the alternating pattern.
    if prep_init_state:
        i = 1 # represents site 1
        while i < n_sites:
            for j in range(qubits_per_site): #runs 2 times as qubits per site=2
                qc.x(i + j*n_sites) # So it flips pairs (1,7), (3,9), (5,11) 
            i+=2 # jump the pairs.
        if use_meson_not_vacuum:
            center = [num_qubits //2 -1, num_qubits // 2]
            qc.x(center) ## flip center two qubits to inject one meson excitation -> Physical: we want to watch a single quark–antiquark pair born at one spot and see how it spreads over time. Placing it at the center gives it equal room to travel left or right — the evolution comes out symmetric, which makes the heatmap readable. Placing it near an edge would make it bump into the boundary almost immediately.
    else:
        # caller handles state just do swapping
        i=1
        while i < num_qubits-1:
            qc.swap(i, i+1)
            i = i + 4 # not this path is not used at all

    # Trotterization
    for step in range(num_trotter_steps):
        if add_visual_barriers:
            qc.barrier() # draws a visual divider in circuit diagrams, no physics

        # 1. SWAP Layer 1- adjacent qubits at pos 1, 5, 9...
        if step > 0:
            i=1
            while i < num_qubits-1:
                qc.swap(i, i+1)
                i = i + 2*qubits_per_site # step by 4    

        # 2. pair layer 1 - runs EVERY step including step 0
        j = 0
        while j < num_qubits - 2:
            qc.compose(pair_hamiltonian_circuit(kinetic_strength),[j, j+1], inplace=True, wrap=keep_gates_as_blocks)
            j = j+2
        if n_sites % 2 == 0:
            qc.compose(pair_hamiltonian_circuit(kinetic_strength),[j, j+1], inplace=True, wrap=keep_gates_as_blocks)          
        
        # 3. SWAP layer 2 — crosses n_i and n_o halves together
        i = 1
        while i < num_qubits - 1:
            qc.swap(i, i + 1)
            i = i + qubits_per_site     # step by 2

        # 4. Pair layer 2 — offset by 2, picks up pairs that layer 1 missed
        j = 2
        while j < num_qubits - 3:
            qc.compose(pair_hamiltonian_circuit(kinetic_strength), [j, j + 1], inplace=True, wrap=keep_gates_as_blocks)
            j = j + 2
        if n_sites % 2 != 0:
            qc.compose(pair_hamiltonian_circuit(kinetic_strength), [j, j + 1], inplace=True, wrap=keep_gates_as_blocks)

        # 5. SWAP layer 3 — partial undo, restores layout for electric layer
        i = 3
        while i < num_qubits - 1:
            qc.swap(i, i + 1)
            i = i + 2 * qubits_per_site     # step by 4

        # 6 Electric layer - rotates each site's 2 qubit about Z axis
        if electric_field_strength!=0:
            electric_circ = electric_hamiltonian_circuit(electric_field_strength)
            for j in range(n_sites):
                qc.compose(electric_circ, [2*j, 2*j+1], inplace=True, wrap=keep_gates_as_blocks) # place the gate on qubits (0,1), (2,3), (4,5), (6,7), (8,9), (10,11)

        # 7. Mass term — alternating Z-rotations on every qubit (staggered fermion sign)
        for q in range(num_qubits):
            if q % 2 == 0:
                qc.rz(-1 * mass, q)     # even qubits rotate by -m
            else:
                qc.rz(mass, q)          # odd qubits rotate by +m

        # measurements
    if add_measurements:
        qc.measure_all()

    return qc


def get_probabilities(expectationval: float):
    return round((1-expectationval)/2, 3)


def get_physical_particle_count(expectation_val_data, n_sites):
    all_counts_persite_perstep = []
    for expectation_val in expectation_val_data:
        probs = [get_probabilities(exp) for exp in expectation_val]
        counts_per_site = []
        for site in range(n_sites):
            occupied_count = probs[2*site] + probs[2*site + 1]
            if site % 2 == 0:
                num_filled_persite = occupied_count          # even site: filled slot = quark present
            else:
                num_filled_persite = 2.0 - occupied_count   # odd site: filled slot = antiquark absent, flip
            counts_per_site.append(num_filled_persite)
        all_counts_persite_perstep.append(counts_per_site)
    return all_counts_persite_perstep


def calc_meson_signal(meson_counts, vacuum_counts, n_sites):
    meson_signal_perstep = []
    for step in range(len(meson_counts)):
        signal_per_site = [abs(meson_counts[step][site]- vacuum_counts[step][site]) for site in range(n_sites)]
        meson_signal_perstep.append(signal_per_site)
    return meson_signal_perstep


# ---------------- AQC building blocks + MPS machinery (from aqctensor-algorithm1.ipynb) ----------------

def aqc_cnot_block(theta1, theta2, theta3, theta4, reverse=False):
    qc = QuantumCircuit(2)
    if reverse:
        qc.cx(1, 0)
    else:
        qc.cx(0, 1)
    qc.ry(theta1, 0)
    qc.rz(theta2, 0)
    qc.ry(theta3, 1)
    qc.rx(theta4, 1)
    return qc


def aqc_triplet_block(thetas): # triplet block
    qc = QuantumCircuit(2)
    for block_index, reverse in enumerate([True, False, True]):
        theta1, theta2, theta3, theta4 = thetas[4*block_index: 4*block_index + 4] # we need 4 thetas for each block
        block = aqc_cnot_block(theta1, theta2, theta3, theta4, reverse=reverse).to_gate(label=f"block{block_index}")
        qc.append(block, [0, 1])
    return qc


def aqc_init_layer(L, thetas): # green init layer
    qc = QuantumCircuit(L)
    for q in range(L):
        theta1, theta2, theta3  = thetas[3*q: 3*q + 3] # we need 3 thetas for each qubit
        qc.rz(theta1, q)
        qc.ry(theta2, q)
        qc.rz(theta3, q)
    return qc


def aqc_field_layer(L, thetas): # orange Rz
    qc = QuantumCircuit(L)
    for q in range(L):
        qc.rz(thetas[q], q)
    return qc


def pack_thetas(thetas_init, thetas_field, thetas_triplet):
    return np.concatenate([thetas_init, thetas_field, thetas_triplet])


def statevector_to_mps(state, L):
    tensors=[]
    M = np.array(state, dtype=complex).reshape(2, -1)
    left_bond = 1

    for q in range(L-1):
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        new_bond = len(S)
        tensors.append(U.reshape(left_bond, 2, new_bond))
        M = (np.diag(S)@Vt).reshape(new_bond*2, -1)
        left_bond = new_bond

    tensors.append(M.reshape(left_bond, 2, 1))
    return tensors


def mps_to_statevector(tensors):
    result = tensors[0]
    for tensor in tensors[1:]:
        result = np.tensordot(result, tensor, axes=([-1],[0]))
    return result.reshape(-1)


def mps_overlap(tensors_psi, tensors_phi):
    E = np.ones((1,1), dtype=complex)
    for A, B in zip(tensors_psi, tensors_phi):
        Bc = np.conj(B)
        E = np.tensordot(E, A, axes=([0], [0]))
        E = np.tensordot(E, Bc, axes=([0,1], [0,1]))
    return E[0,0]


def apply_1q_gate_to_mps_tensor(inputtensor, gate):
    return np.tensordot(gate, inputtensor, axes=([1],[1])).transpose(1, 0, 2)


def apply_2q_gate_mps_tensor(inputtensor1, inputtensor2, gate, max_bond=None):
    mergedtoonetensor = np.tensordot(inputtensor1, inputtensor2, axes=([2], [0]))
    gate_tensor = gate.reshape(2,2,2,2) # 4 by 4 2 qubit matrix flattened. each axis has 2 states so it can represent the value of 00, 01, 10, 11 CNOT below is an e.g.
    '''         col=0  col=1  col=2  col=3
        row=0 [   1      0      0      0  ]
        row=1 [   0      1      0      0  ]
        row=2 [   0      0      0      1  ]
        row=3 [   0      0      1      0  ]
    '''
    gateappliedtensor = np.tensordot(gate_tensor, mergedtoonetensor, axes =([2,3],[1,2]))
    result = gateappliedtensor.transpose(2, 0, 1, 3)
    l, _, _, r = result.shape
    matrix = result.reshape(l*2, 2*r)
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)

    if max_bond is not None:
        # KNOWN LIMITATION: this cut is only the best possible truncation if the rest of the chain
        # is in canonical form (not maintained here), and S is not renormalized afterwards, so the
        # state's norm leaks with every cut. Harmless for the L=50 XYZ run (chi=32 never actually
        # truncated: norm^2 = 1.000000, fidelity unchanged at chi=64/128) -- must be fixed before any
        # run where truncation really bites (see hadron-aqc/spec.md, Step 0).
        k = min(max_bond, len(S)) # note S is a 1-D list of numbers at the point
        U, S, Vt = U[:, :k], S[:k], Vt[:k, :]

    new_bond = len(S)
    new_tensor1 = U.reshape(l,2,new_bond)
    new_tensor2 = (np.diag(S) @Vt).reshape(new_bond, 2, r)
    return new_tensor1, new_tensor2


def circuit_to_mps_tensors(start_tensors, circuit, max_bond=None):
    start_tensors = list(start_tensors)
    L = circuit.num_qubits
    for instruction in circuit.data:
        gate_matrix = Operator(instruction.operation).data
        qubits = [circuit.find_bit(q).index for q in instruction.qubits] # instruction.qubits gives you Qubit objects, not plain numbers instruction.qubits = (Qubit(index=3), Qubit(index=2))
        if len(qubits) == 1:
            tensor_idx = L -1 -qubits[0]
            start_tensors[tensor_idx] = apply_1q_gate_to_mps_tensor(start_tensors[tensor_idx], gate_matrix)
        else:
            qubitA, qubitB = qubits
            tensor_idx_A = L -1 - qubitA
            tensor_idx_B = L -1 - qubitB
            if tensor_idx_B < tensor_idx_A:
                lo, hi, gate_to_use = tensor_idx_B, tensor_idx_A, gate_matrix
            else:
                lo, hi = tensor_idx_A, tensor_idx_B
                gate_to_use = gate_matrix.reshape(2,2,2,2).transpose(1,0,3,2).reshape(4,4)
            start_tensors[lo], start_tensors[hi] = apply_2q_gate_mps_tensor(start_tensors[lo], start_tensors[hi], gate_to_use, max_bond=max_bond)
    return start_tensors


# ---------------- Library route: the AQC recipe, written once (IBM qiskit-addon-aqc-tensor + quimb) ----------------
from functools import partial

import quimb as qu
import quimb.tensor as qtn
from scipy.optimize import minimize
from qiskit import transpile
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.passes import Collect2qBlocks, ConsolidateBlocks
from qiskit_addon_aqc_tensor.ansatz_generation import generate_ansatz_from_circuit
from qiskit_addon_aqc_tensor.simulation import tensornetwork_from_circuit, compute_overlap
from qiskit_addon_aqc_tensor.simulation.quimb import QuimbSimulator
from qiskit_addon_aqc_tensor.objective import MaximizeStateFidelity

# ---- Settings: every number below that affects results, in one place ----
# physics: the tutorial's strengths, applied once per Trotter step
KINETIC_STRENGTH = 0.15          # pair gates: how easily particles hop (and quark-antiquark pairs are created)
ELECTRIC_FIELD_STRENGTH = 0.01   # electric gates: energy cost of the string between quark and antiquark
MASS = 0.03                      # Rz turns at the end of each step: quark mass
# simulation
TARGET_SVD_CUTOFF = 0.0          # building targets: nothing is dropped except by max_bond
TRAINING_SVD_CUTOFF = 1e-10      # simulating the trainable circuit: quimb's default, written out
# training (scipy L-BFGS-B)
TRAINING_MAX_STEPS = 2000
TRAINING_FTOL = 1e-14            # stop when the fidelity stops improving by more than this
TRAINING_GTOL = 1e-10            # stop when every slope is smaller than this
# counting CNOTs and depth
TRANSPILE_BASIS_GATES = ["cx", "rz", "sx", "x"]
TRANSPILE_LEVEL = 3
TRANSPILE_SEED = 0
TINY_ANGLE_THRESHOLD = 1e-6      # angles below this are set to 0 before transpiling (Qiskit level-3 bug)


def make_simulator_settings(max_bond=None):
    """quimb MPS simulator for the library. max_bond=None: no cap."""
    return QuimbSimulator(partial(qtn.CircuitMPS, max_bond=max_bond, cutoff=TRAINING_SVD_CUTOFF), autodiff_backend="jax")


def mps_fidelity(mps_a, mps_b):
    return abs(compute_overlap(mps_a, mps_b)) ** 2


def tensors_to_library_target(tensors):
    """Repackage our own TEBD tensors (circuit_to_mps_tensors) as a quimb MPS the library can use."""
    lib_arrays = [tensor.transpose(2, 0, 1) for tensor in reversed(tensors)]
    lib_arrays[0] = lib_arrays[0][0]          # drop the dummy size-1 link at the start
    lib_arrays[-1] = lib_arrays[-1][:, 0, :]  # and at the end
    return qtn.CircuitMPS(psi0=qtn.MatrixProductState(lib_arrays, shape="lrp"))


def build_target_using_lib(n_sites, num_trotter_steps, max_bond, use_meson_not_vacuum,
                           kinetic_strength=KINETIC_STRENGTH, electric_field_strength=ELECTRIC_FIELD_STRENGTH, mass=MASS):
    """Target MPS. Gates on the same qubit pair are merged into one block first (fewer cuts),
    then quimb applies them, cutting each link to max_bond (None = no cap)."""
    qc = construct_circuit(n_sites, num_trotter_steps, kinetic_strength, electric_field_strength, mass,
                           use_meson_not_vacuum=use_meson_not_vacuum)
    merged_qc = PassManager([Collect2qBlocks(), ConsolidateBlocks(force_consolidate=True)]).run(qc)
    target_mps = qtn.CircuitMPS(qc.num_qubits, max_bond=max_bond, cutoff=TARGET_SVD_CUTOFF)
    for instruction in merged_qc.data:
        qubits = [merged_qc.find_bit(q).index for q in instruction.qubits]
        target_mps.apply_gate_raw(Operator(instruction.operation).data, tuple(reversed(qubits)))  # quimb orders the two qubits the other way
    return target_mps


def check_target(target_mps):
    """Health check: norm violation, chance each seat (qubit) is filled, count of filled seats (must equal n_sites)."""
    norm = abs(compute_overlap(target_mps, target_mps))
    z_val_per_seat = [target_mps.local_expectation(qu.pauli("Z"), (q,)).real / norm for q in range(target_mps.N)]
    chance_of_filled_per_seat = [(1 - z) / 2 for z in z_val_per_seat]
    count_of_filled_seats = sum(chance_of_filled_per_seat)
    return 1 - norm, chance_of_filled_per_seat, count_of_filled_seats


def build_cheap_trainable_circuit(n_sites, num_trotter_steps, num_coarse_steps, use_meson_not_vacuum,
                                  kinetic_strength=KINETIC_STRENGTH, electric_field_strength=ELECTRIC_FIELD_STRENGTH, mass=MASS):
    """Same circuit with fewer, bigger steps (strengths x num_trotter_steps/num_coarse_steps);
    the library turns it into a trainable circuit whose starting angles copy it exactly."""
    strength_scale = num_trotter_steps / num_coarse_steps
    coarse_circuit = construct_circuit(n_sites, num_coarse_steps,
                                       kinetic_strength * strength_scale,
                                       electric_field_strength * strength_scale,
                                       mass * strength_scale,
                                       use_meson_not_vacuum=use_meson_not_vacuum)
    return generate_ansatz_from_circuit(coarse_circuit, qubits_initially_zero=True)


def train_to_target(ansatz, starting_angles, target_mps, simulator_settings, maxiter=TRAINING_MAX_STEPS, progress=None):
    """Turn the angles until the trainable circuit matches the target (L-BFGS-B, exact slopes via jax).
    progress: optional function, called after every training step with the current fidelity."""
    training_objective = MaximizeStateFidelity(target_mps, ansatz, simulator_settings)

    def after_each_step(intermediate_result):   # scipy calls this after every step; .fun = 1 - fidelity
        if progress is not None:
            progress(1 - intermediate_result.fun)

    return minimize(training_objective, starting_angles, method="L-BFGS-B", jac=True, callback=after_each_step,
                    options={"maxiter": maxiter, "ftol": TRAINING_FTOL, "gtol": TRAINING_GTOL})


def count_cnots(circuit):
    """CNOTs, two-qubit depth, total depth, after transpiling to a straight line of qubits (settings above)."""
    linear_chain = CouplingMap.from_line(circuit.num_qubits)
    transpiled = transpile(circuit, basis_gates=TRANSPILE_BASIS_GATES, coupling_map=linear_chain,
                           optimization_level=TRANSPILE_LEVEL, seed_transpiler=TRANSPILE_SEED)
    cnots = transpiled.count_ops().get("cx", 0)
    two_qubit_depth = transpiled.depth(lambda instruction: instruction.operation.num_qubits == 2)
    total_depth = transpiled.depth()
    return cnots, two_qubit_depth, total_depth, transpiled


def snap_tiny_angles(angles, threshold=TINY_ANGLE_THRESHOLD):
    """Angles below threshold -> exactly 0 (avoids a Qiskit level-3 transpile bug with near-zero angles)."""
    return np.where(np.abs(angles) < threshold, 0.0, angles)


def count_and_verify(ansatz, trained_angles, target_mps, simulator_settings):
    """Count the trained circuit after transpiling, and check the TRANSPILED circuit still matches the target."""
    trained_circuit = ansatz.assign_parameters(snap_tiny_angles(trained_angles))
    cnots, two_qubit_depth, total_depth, transpiled = count_cnots(trained_circuit)
    qubit_order_unchanged = list(transpiled.layout.final_index_layout()) == list(range(transpiled.num_qubits))
    transpiled_mps = tensornetwork_from_circuit(transpiled, simulator_settings)
    return {"cnots": cnots, "two_qubit_depth": two_qubit_depth, "total_depth": total_depth,
            "qubit_order_unchanged": qubit_order_unchanged,
            "transpiled_fidelity": mps_fidelity(transpiled_mps, target_mps)}

