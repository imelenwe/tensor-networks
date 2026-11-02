"""LayerGradientObjective: what scipy calls each training step, angles in -> (L, all slopes) out.

objective(angles) returns (L, slopes) like the library's MaximizeStateFidelity, so it plugs into
scipy's minimize(..., jac=True). L is computed on the trimmed forward state.
Steps: two_qubit_blocks_to_matrices (angles -> 4x4 blocks) -> forward_backward_pass (o and one table per block) -> loss_and_slopes.
"""
import numpy as np

from .two_qubit_blocks_to_matrices import read_boxes, merge_into_blocks, block_matrices
from .forward_backward_pass import forward_and_backward

TRIM_CUTOFF = 1e-12   # trimming also drops weights below this fraction of the largest


def loss_and_slopes(value, tables, changes, n_angles):
    """The chain rule. L = 1 - |o|^2 and dL/d(angle) = -2 Re(conj(o) do/d(angle)), where do/d(angle) is the sum, over
    the blocks holding the angle, of (the block's table x the block's change per angle): 16 products each."""
    overlap_slopes = np.zeros(n_angles, dtype=complex)
    for table, block_changes in zip(tables, changes):
        for angle, change in block_changes:
            overlap_slopes[angle] += np.sum(change * table)
    return 1.0 - abs(value) ** 2, -2.0 * np.real(np.conj(value) * overlap_slopes)


def target_as_chain(target_mps):
    """Our quimb target (CircuitMPS or MatrixProductState) as a list of (left, 0/1, right) arrays, site q = qubit q;
    the two end sites get a link of size 1."""
    if hasattr(target_mps, "psi"):
        target_mps = target_mps.psi
    last = target_mps.L - 1
    chain = []
    for q in range(target_mps.L):
        left = target_mps.bond(q - 1, q) if q > 0 else None
        right = target_mps.bond(q, q + 1) if q < last else None
        array = target_mps[q].transpose(*[i for i in (left, target_mps.site_ind(q), right) if i is not None]).data
        if left is None:
            array = array[None]
        if right is None:
            array = array[..., None]
        chain.append(array)
    return chain


class LayerGradientObjective:
    """settings: a LayerGradientSettings (max_bond, stride, use_gpu, single_precision)."""

    def __init__(self, ansatz, target_mps, settings):
        self.boxes = read_boxes(ansatz)
        self.blocks = merge_into_blocks(self.boxes)
        self.n_qubits, self.n_angles = ansatz.num_qubits, ansatz.num_parameters
        self.settings = settings
        if settings.use_gpu:
            import cupy as xp
            self.keep = lambda chain: [xp.asnumpy(array) for array in chain]     # kept states wait in host memory
            self.restore = lambda chain: [xp.asarray(array) for array in chain]
        else:
            xp = np
            self.keep = self.restore = list
        self.xp = xp
        self.dtype = np.complex64 if settings.single_precision else np.complex128
        target = target_as_chain(target_mps)
        if len(target) != self.n_qubits:
            raise ValueError(f"the target has {len(target)} sites, the trainable circuit {self.n_qubits} qubits")
        self.target = [xp.asarray(array).astype(self.dtype) for array in target]

    def _passes(self, angles):
        angles = np.array(angles, dtype=float)
        matrices, changes = block_matrices(self.blocks, self.boxes, angles)
        gates = [(block.site, self.xp.asarray(matrix).astype(self.dtype)) for block, matrix in zip(self.blocks, matrices)]
        value, tables, layer_overlaps = forward_and_backward(gates, self.n_qubits, self.target, self.settings.max_bond,
                                                             TRIM_CUTOFF, self.xp, self.settings.stride,
                                                             self.keep, self.restore)
        return value, tables, changes, layer_overlaps

    def __call__(self, angles):
        value, tables, changes, _ = self._passes(angles)
        if self.settings.use_gpu:
            tables = self.xp.asnumpy(self.xp.stack(tables))
        return loss_and_slopes(value, tables, changes, self.n_angles)

    def layer_overlaps(self, angles):
        """<B_k|S_k> at every layer k, last layer first (all equal the overlap; a check on the backward layer pass)."""
        return self._passes(angles)[3]
