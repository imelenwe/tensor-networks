"""The trainable circuit as two-qubit blocks: every box (ZXZ, KAK, ...) is merged into one 4x4 matrix per two-qubit
block, and for each of the block's angles we get how that matrix changes per unit of the angle.

A block = one two-qubit box (KAK) + the one-qubit boxes (ZXZ) after it on its two qubits; the boxes at the very start
of a qubit join the first block after them. The 16 entries are computed from all of the block's angles (they are not
the angles). The block's matrix is the product of its boxes' matrices, in the chain's qubit order.
"""
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

QISKIT_TO_CHAIN_ORDER = [0, 2, 1, 3]   # Qiskit: a box's first qubit is the last bit; chain: the first site is the first bit
NUDGE = 1e-6                           # a box's change per angle: (box(angle + NUDGE) - box(angle - NUDGE)) / (2 NUDGE)


@dataclass
class Box:
    qubits: tuple                         # one or two qubits
    angle_indices: tuple                  # which of the trainable circuit's angles it uses
    matrix_at: Callable                   # angles -> the box's matrix (Qiskit's order over its own qubits)


@dataclass
class Block:
    site: int                                        # acts on sites (site, site + 1)
    parts: list = field(default_factory=list)        # (box index, "first" | "second" | "both"), in the order applied
    angle_indices: list = field(default_factory=list)


def read_boxes(ansatz):
    """Every instruction of the trainable circuit as a Box; its matrix comes from Qiskit (Operator)."""
    position = {angle: i for i, angle in enumerate(ansatz.parameters)}
    boxes = []
    for instruction in ansatz.data:
        if instruction.operation.name == "barrier":
            continue
        qubits = tuple(ansatz.find_bit(qubit).index for qubit in instruction.qubits)
        alone = QuantumCircuit(len(qubits))
        alone.append(instruction.operation, list(range(len(qubits))))
        angles = list(alone.parameters)
        indices = tuple(position[angle] for angle in angles)
        if not angles:                                  # e.g. a box whose angles are all locked: one fixed matrix
            fixed = np.asarray(Operator(alone).data)
            boxes.append(Box(qubits, indices, lambda x, fixed=fixed: fixed))
            continue

        def matrix_at(x, alone=alone, angles=angles, indices=indices):
            values = {angle: float(x[i]) for angle, i in zip(angles, indices)}
            return np.asarray(Operator(alone.assign_parameters(values)).data)
        boxes.append(Box(qubits, indices, matrix_at))
    return boxes


def merge_into_blocks(boxes):
    """Group the boxes into blocks, one per two-qubit box (see the module docstring)."""
    blocks, last_block_on, waiting = [], {}, {}
    for index, box in enumerate(boxes):
        if len(box.qubits) == 2:
            first, second = sorted(box.qubits)
            if second != first + 1:
                raise ValueError(f"a two-qubit box on qubits {box.qubits}: only neighbours (q, q + 1) are supported")
            block = Block(first)
            block.parts = ([(j, "first") for j in waiting.pop(first, [])] + [(j, "second") for j in waiting.pop(second, [])]
                           + [(index, "both")])
            blocks.append(block)
            last_block_on[first] = last_block_on[second] = block
        elif len(box.qubits) == 1:
            qubit = box.qubits[0]
            if qubit in last_block_on:
                block = last_block_on[qubit]
                block.parts.append((index, "first" if qubit == block.site else "second"))
            else:
                waiting.setdefault(qubit, []).append(index)
        else:
            raise ValueError(f"a box on {len(box.qubits)} qubits")
    if waiting:
        raise ValueError(f"qubits {sorted(waiting)} have no two-qubit box")
    for block in blocks:
        block.angle_indices = sorted({i for j, _ in block.parts for i in boxes[j].angle_indices})
    return blocks


def _on_block(matrix, qubits, where):
    """A box's matrix as a 4x4 on its block's two sites, in the chain's order."""
    if where == "both":
        return matrix[np.ix_(QISKIT_TO_CHAIN_ORDER, QISKIT_TO_CHAIN_ORDER)] if qubits[0] < qubits[1] else matrix
    return np.kron(matrix, np.eye(2)) if where == "first" else np.kron(np.eye(2), matrix)


def block_matrices(blocks, boxes, angles):
    """Each block's 4x4 matrix at these angles, and for each of its angles the block's change per unit of that angle.
    Product rule: only the boxes holding the angle change; the rest of the block stays as it is."""
    matrices, changes = [], []
    for block in blocks:
        parts = [_on_block(boxes[j].matrix_at(angles), boxes[j].qubits, where) for j, where in block.parts]
        before = [np.eye(4, dtype=complex)]            # before[k] = parts[k-1] ... parts[0]
        for part in parts:
            before.append(part @ before[-1])
        after = [np.eye(4, dtype=complex)]             # after[k] = the last k parts
        for part in reversed(parts):
            after.append(after[-1] @ part)
        matrices.append(before[-1])
        block_changes = []
        for angle in block.angle_indices:
            change = np.zeros((4, 4), dtype=complex)
            for k, (j, where) in enumerate(block.parts):
                box = boxes[j]
                if angle in box.angle_indices:
                    up, down = angles.copy(), angles.copy()
                    up[angle] += NUDGE
                    down[angle] -= NUDGE
                    box_change = (box.matrix_at(up) - box.matrix_at(down)) / (2 * NUDGE)
                    change += after[len(parts) - 1 - k] @ _on_block(box_change, box.qubits, where) @ before[k]
            block_changes.append((angle, change))
        changes.append(block_changes)
    return matrices, changes
