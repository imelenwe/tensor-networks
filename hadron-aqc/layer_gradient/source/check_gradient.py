"""Self-check: the MPS gradient against the exact one on a small random problem, then a short
scipy run. Needs numpy, scipy and qiskit only (CPU).

    python check_gradient.py

1. 8 qubits, an ansatz of one-qubit rotations and RXX/RYY/RZZ/CX blocks on neighbours (both
   qubit orders), a random target. At a bond that holds the state exactly (chi = 16), loss and
   gradient must equal the exact ones (Qiskit statevector, central differences) to about 1e-7.
2. At smaller bonds (chi = 8, 4) the value and gradient are approximate: their errors are printed.
   A random target is highly entangled, so here a small bond is a poor approximation; physical
   states (LSH after a few Trotter steps) need far less bond than their size allows.
3. A few L-BFGS-B iterations with jac=True, as the objective would be used in AQC.
"""
import time

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize

from mps_objective import MPSGradientObjective, target_from_statevector

N, LAYERS, SEED = 8, 3, 7


def ansatz(n, layers):
    count = layers * (n * 2 + (n - 1) * 3)
    p, k = ParameterVector("x", count), 0
    qc = QuantumCircuit(n)
    for layer in range(layers):
        for q in range(n):
            qc.rz(p[k], q)
            qc.ry(p[k + 1], q)
            k += 2
        for start in (0, 1):
            for q in range(start, n - 1, 2):
                qc.rxx(p[k], q, q + 1)
                qc.ryy(p[k + 1], q + 1, q)          # reversed qubit order on purpose
                qc.rzz(p[k + 2], q, q + 1)
                qc.cx(q + 1, q) if layer % 2 else qc.cx(q, q + 1)
                k += 3
    return qc


def exact_loss(qc, x, target_sv):
    return 1.0 - abs(Statevector(qc.assign_parameters(x)).inner(target_sv)) ** 2


def main():
    rng = np.random.default_rng(SEED)
    qc = ansatz(N, LAYERS)
    target_sv = Statevector(ansatz(N, LAYERS + 2).assign_parameters(rng.uniform(-np.pi, np.pi, qc.num_parameters
                                                                                + 2 * (N * 2 + (N - 1) * 3))))
    target = target_from_statevector(target_sv.data)
    x = rng.uniform(-np.pi, np.pi, qc.num_parameters)
    print(f"{N} qubits, {qc.num_parameters} parameters, {qc.size()} instructions")

    h = 1e-6
    t0 = time.monotonic()
    g_exact = np.array([(exact_loss(qc, x + h * e, target_sv) - exact_loss(qc, x - h * e, target_sv)) / (2 * h)
                        for e in np.eye(qc.num_parameters)])
    l_exact = exact_loss(qc, x, target_sv)
    print(f"exact (statevector, central differences): loss {l_exact:.10f}  ({time.monotonic() - t0:.1f} s)")

    for chi in (2 ** (N // 2), 8, 4):
        obj = MPSGradientObjective(qc, target, chi=chi, stride=2)
        t0 = time.monotonic()
        loss, grad = obj(x)
        err = np.linalg.norm(grad - g_exact) / np.linalg.norm(g_exact)
        print(f"MPS chi={chi:2d}: loss {loss:.10f}  |dloss| {abs(loss - l_exact):.1e}  "
              f"gradient rel. error {err:.1e}  ({time.monotonic() - t0:.2f} s)")
        if chi == 2 ** (N // 2):
            assert abs(loss - l_exact) < 1e-7 and err < 1e-6, (loss, l_exact, err)

    obj = MPSGradientObjective(qc, target, chi=2 ** (N // 2))
    res = minimize(obj, x, jac=True, method="L-BFGS-B", options={"maxiter": 30})
    print(f"L-BFGS-B, 30 iterations: loss {obj(x)[0]:.4f} -> {res.fun:.4f} "
          f"(exact at the end {exact_loss(qc, res.x, target_sv):.4f}), {obj.calls} objective calls")
    print("CHECK_OK")


if __name__ == "__main__":
    main()
