"""AQC objective with an MPS adjoint gradient, for a parametrised Qiskit ansatz (uses fastgrad_core).

    from mps_objective import MPSGradientObjective, target_from_quimb
    obj = MPSGradientObjective(ansatz, target_from_quimb(target_mps), chi=1024)
    res = scipy.optimize.minimize(obj, x0, jac=True, method="L-BFGS-B")

obj(x) returns (1 - |<target|U(x)|0>|^2, gradient), like AQC-Tensor's objective, so it can take
its place in a scipy optimiser. The value is computed on the forward state truncated at bond chi.

ansatz: a parametrised QuantumCircuit whose two-qubit gates act on neighbouring qubits (q, q + 1)
        and in which every qubit has at least one two-qubit gate. Parameters are ordered as in
        ansatz.parameters (as AQC-Tensor orders them).
target: a list of (left, physical, right) arrays with site q = qubit q; target_from_quimb and
        target_from_statevector make one.
xp:     numpy (CPU) or cupy (GPU). dtype: complex128, or complex64 for half the memory.
stride: forward states are kept before every stride-th layer of gates and the ones in between
        are recomputed in the backward pass (more stride = less memory, a little more time).
host:   with cupy, keep those saved states in host RAM instead of GPU memory.
"""
from __future__ import annotations

import numpy as np

import fastgrad_core as core


def qiskit_instrs(ansatz):
    """(qubits, parameter indices, matfn) per instruction; matfn(x) is the instruction's matrix
    (Qiskit's order over its own qubits) with the ansatz parameters set to x."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    index = {p: i for i, p in enumerate(ansatz.parameters)}
    out = []
    for inst in ansatz.data:
        if inst.operation.name == "barrier":
            continue
        qs = tuple(ansatz.find_bit(q).index for q in inst.qubits)
        tiny = QuantumCircuit(len(qs))
        tiny.append(inst.operation, list(range(len(qs))))
        params = list(tiny.parameters)
        ids = tuple(index[p] for p in params)
        if not params:
            fixed = np.asarray(Operator(tiny).data)
            out.append((qs, ids, lambda x, m=fixed: m))
            continue

        def fn(x, tiny=tiny, params=params, ids=ids):
            values = dict(zip(params, (float(x[i]) for i in ids)))
            return np.asarray(Operator(tiny.assign_parameters(values)).data)
        out.append((qs, ids, fn))
    return out


def target_from_quimb(mps):
    """A quimb MatrixProductState (or a CircuitMPS, through .psi) as (left, physical, right) arrays,
    with size-1 end bonds; site q = qubit q, as in Qiskit-to-quimb circuit conversion."""
    if hasattr(mps, "psi"):
        mps = mps.psi
    out = []
    for i in range(mps.L):
        left = mps.bond(i - 1, i) if i > 0 else None
        right = mps.bond(i, i + 1) if i < mps.L - 1 else None
        a = mps[i].transpose(*[x for x in (left, mps.site_ind(i), right) if x is not None]).data
        if left is None:
            a = a[None]
        if right is None:
            a = a[..., None]
        out.append(a)
    return out


def target_from_statevector(psi):
    """A Qiskit statevector (qubit q = bit q of the index) as an exact chain, for small tests."""
    psi = np.asarray(psi, dtype=complex)
    n = int(round(np.log2(psi.size)))
    rest = psi.reshape([2] * n).transpose(list(range(n))[::-1]).reshape(1, -1)   # axes: qubit 0 .. n-1
    chain, left = [], 1
    for _ in range(n - 1):
        u, s, vh = np.linalg.svd(rest.reshape(left * 2, -1), full_matrices=False)
        k = max(1, int((s > 1e-14 * s[0]).sum()))
        chain.append(u[:, :k].reshape(left, 2, k))
        rest, left = s[:k, None] * vh[:k], k
    chain.append(rest.reshape(left, 2, 1))
    return chain


class MPSGradientObjective:
    """1 - |<target|U(x)|0>|^2 and its gradient over the ansatz parameters (fastgrad_core)."""

    def __init__(self, ansatz, target, chi, xp=np, dtype=np.complex128, cutoff=1e-12, stride=6, host=False):
        self.instrs = qiskit_instrs(ansatz)
        self.units = core.build_units(self.instrs)
        self.n, self.n_params = ansatz.num_qubits, ansatz.num_parameters
        if len(target) != self.n:
            raise ValueError(f"target has {len(target)} sites, ansatz {self.n} qubits")
        self.xp, self.dtype, self.chi, self.cutoff, self.stride = xp, dtype, chi, cutoff, stride
        self.target = [xp.asarray(t).astype(dtype) for t in target]
        on_gpu = hasattr(xp, "asnumpy")
        self.keep = (lambda st: [xp.asnumpy(a) for a in st]) if host and on_gpu else list
        self.back = (lambda st: [xp.asarray(a) for a in st]) if host and on_gpu else list
        self.calls = 0

    def gates(self, x):
        """The merged 4x4 units at x (quimb order) and their derivatives."""
        mats, derivs = core.unit_matrices(self.units, self.instrs, np.array(x, dtype=float))
        return [(u["sites"], self.xp.asarray(m).astype(self.dtype)) for u, m in zip(self.units, mats)], derivs

    def loss_function(self, x):
        gates, derivs = self.gates(x)
        f, envs = core.mps_value_and_envs(self.n, gates, self.target, self.chi, self.cutoff, self.xp,
                                          self.stride, self.keep, self.back)
        envs = self.xp.asnumpy(self.xp.stack(envs)) if hasattr(self.xp, "asnumpy") else envs
        self.calls += 1
        return core.loss_and_grad(f, envs, derivs, self.n_params)

    __call__ = loss_function
