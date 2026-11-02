#!/usr/bin/env python3
"""Stage 29 core: the fidelity of a circuit on an MPS and its gradient for every gate, from one
forward pass, one backward pass and per-layer environments (numpy or cupy; no Qiskit).

AQC-Tensor 0.3.1 does one full overlap per angle, so a 120q R5 try costs 13,560 overlaps of
120-site MPSs (85 min at bond 1024). Here the environment of every gate,
E_g = d<T|U|0>/dG_g, comes from contracting two MPSs once per layer of disjoint gates, so a
try costs about two to three forward passes. An exact state-vector version (adjoint method)
checks it at small n and is an exact simulator up to about 30 qubits.

Conventions. A gate is (sites, matrix), sites (q,) or (q, q + 1); a two-site matrix is
indexed [(a1 a2), (b1 b2)] with site q the more significant bit (quimb's order), a the output.
MPS tensors are (left, physical, right). Truncation: SVD at the orthogonality centre, keep at
most chi values above cutoff * largest, no renormalisation. Arrays are never changed in
place, so states can be kept as shallow copies (lists).

The adapter (build_units, unit_matrices, loss_and_grad) takes a circuit as (qubits, param
indices, matfn) per instruction, matfn(x) being the matrix in Qiskit's order over the
instruction's own qubits, merges one-qubit instructions into two-qubit units and gives
1 - |f|^2 and its gradient over the parameters."""
from __future__ import annotations

import math

import numpy as np

LOW_FIRST = [0, 2, 1, 3]   # Qiskit 4x4 (qargs[0] least significant) <-> first site most significant


# ---------------------------------------------------------------- MPS
def zero_mps(n, xp, dtype):
    a = xp.zeros((1, 2, 1), dtype=dtype)
    a[0, 0, 0] = 1
    return [a] * n


def move_center(mps, c, to, xp):
    """Move the orthogonality centre from site c to site `to` (QR steps right, LQ steps left)."""
    while c < to:
        dl, d, dr = mps[c].shape
        q, r = xp.linalg.qr(mps[c].reshape(dl * d, dr))
        mps[c] = q.reshape(dl, d, q.shape[1])
        mps[c + 1] = xp.tensordot(r, mps[c + 1], axes=(1, 0))
        c += 1
    while c > to:
        dl, d, dr = mps[c].shape
        q, r = xp.linalg.qr(mps[c].reshape(dl, d * dr).T)
        mps[c] = q.T.reshape(q.shape[1], d, dr)
        mps[c - 1] = xp.tensordot(mps[c - 1], r.T, axes=(2, 0))
        c -= 1
    return c


def apply_one(mps, q, m, xp):
    mps[q] = xp.tensordot(m, mps[q], axes=(1, 1)).transpose(1, 0, 2)


def apply_two(mps, c, q, m, chi, cutoff, xp, absorb_right=True):
    """Apply a two-site gate on (q, q + 1) at the orthogonality centre and truncate; returns the centre."""
    c = move_center(mps, c, q if c <= q else q + 1, xp)
    dl, dr = mps[q].shape[0], mps[q + 1].shape[2]
    t = xp.tensordot(mps[q], mps[q + 1], axes=(2, 0))
    t = xp.tensordot(m.reshape(2, 2, 2, 2), t, axes=([2, 3], [1, 2]))
    u, s, vh = xp.linalg.svd(t.transpose(2, 0, 1, 3).reshape(dl * 2, 2 * dr), full_matrices=False)
    k = max(1, min(chi, int((s > cutoff * s[0]).sum())))
    u, s, vh = u[:, :k], s[:k], vh[:k]
    if absorb_right:
        mps[q], mps[q + 1] = u.reshape(dl, 2, k), (s[:, None] * vh).reshape(k, 2, dr)
        return q + 1
    mps[q], mps[q + 1] = (u * s).reshape(dl, 2, k), vh.reshape(k, 2, dr)
    return q


def overlap(bra, ket, xp):
    """<bra|ket> by transfer matrices."""
    env = xp.ones((1, 1), dtype=ket[0].dtype)
    for a, b in zip(bra, ket):
        env = xp.tensordot(a.conj(), xp.tensordot(env, b, axes=(1, 0)), axes=([0, 1], [0, 1]))
    return complex(env[0, 0])


def to_dense(mps, xp):
    """The MPS as a flat state vector, qubit 0 the most significant."""
    v = mps[0].reshape(2, -1)
    for a in mps[1:]:
        v = xp.tensordot(v, a, axes=(v.ndim - 1, 0))
    return v.reshape(-1)


# ---------------------------------------------------------------- layers and sweeps
def layers(gates, n):
    """ASAP layers of disjoint gates: a gate goes one layer after the last gate on its sites."""
    nxt, level = [0] * n, []
    for sites, _ in gates:
        lv = max(nxt[s] for s in sites)
        level.append(lv)
        for s in sites:
            nxt[s] = lv + 1
    groups = [[] for _ in range(max(level) + 1)]
    for g, lv in enumerate(level):
        groups[lv].append(g)
    return groups


def apply_layer(mps, c, gates, idx, chi, cutoff, xp, dagger=False):
    """Apply the disjoint gates idx, sweeping away from the centre; returns the new centre."""
    order = sorted(idx, key=lambda g: gates[g][0][0])
    right = c <= (gates[order[0]][0][0] + gates[order[-1]][0][-1]) / 2
    for g in (order if right else order[::-1]):
        sites, m = gates[g]
        if dagger:
            m = m.conj().T
        if len(sites) == 1:
            apply_one(mps, sites[0], m, xp)
        else:
            c = apply_two(mps, c, sites[0], m, chi, cutoff, xp, absorb_right=right)
    return c


def mps_forward(n, gates, groups, chi, cutoff, xp, dtype, stride=0, keep=list):
    """U|0> truncated at chi; with stride, the states before every stride-th layer are kept."""
    ket, c, saved = zero_mps(n, xp, dtype), 0, {}
    for lv, idx in enumerate(groups):
        if stride and lv % stride == 0:
            saved[lv] = (keep(ket), c)
        c = apply_layer(ket, c, gates, idx, chi, cutoff, xp)
    return ket, saved


# ---------------------------------------------------------------- environments
def _left(env, bra, ket, i, w, m, xp):
    """Absorb block (i, width w, gate m or None) into a left environment (bra bond, ket bond)."""
    t = xp.tensordot(env, ket[i], axes=(1, 0))                              # (B, b1, K)
    if w == 1:
        if m is not None:
            t = xp.tensordot(m, t, axes=(1, 1)).transpose(1, 0, 2)         # (B, a, K)
        return xp.tensordot(bra[i].conj(), t, axes=([0, 1], [0, 1]))
    t = xp.tensordot(t, ket[i + 1], axes=(2, 0))                            # (B, b1, b2, K)
    t = xp.tensordot(m.reshape(2, 2, 2, 2), t, axes=([2, 3], [1, 2]))       # (a1, a2, B, K)
    u = xp.tensordot(bra[i].conj(), t, axes=([0, 1], [2, 0]))               # (G, a2, K)
    return xp.tensordot(bra[i + 1].conj(), u, axes=([0, 1], [0, 1]))


def _right(env, bra, ket, i, w, m, xp):
    """Absorb block (i, w, m) into a right environment (bra bond, ket bond)."""
    t = xp.tensordot(ket[i + w - 1], env, axes=(2, 1))                      # (K, b, B)
    if w == 1:
        if m is not None:
            t = xp.tensordot(m, t, axes=(1, 1)).transpose(1, 0, 2)         # (K, a, B)
        return xp.tensordot(bra[i].conj(), t, axes=([1, 2], [1, 2]))
    t = xp.tensordot(ket[i], t, axes=(2, 0))                                # (K, b1, b2, B)
    t = xp.tensordot(m.reshape(2, 2, 2, 2), t, axes=([2, 3], [1, 2]))       # (a1, a2, K, B)
    u = xp.tensordot(bra[i + 1].conj(), t, axes=([1, 2], [1, 3]))           # (G, a1, K)
    return xp.tensordot(bra[i].conj(), u, axes=([1, 2], [1, 0]))


def _gate_env(lenv, renv, bra, ket, i, w, xp):
    x = xp.tensordot(lenv, ket[i], axes=(1, 0))                             # (B, b1, K)
    if w == 1:
        x = xp.tensordot(x, renv, axes=(2, 1))                              # (B, b, B')
        return xp.tensordot(bra[i].conj(), x, axes=([0, 2], [0, 2]))       # (a, b)
    x = xp.tensordot(x, ket[i + 1], axes=(2, 0))                            # (B, b1, b2, K')
    x = xp.tensordot(x, renv, axes=(3, 1))                                  # (B, b1, b2, B')
    y = xp.tensordot(bra[i].conj(), bra[i + 1].conj(), axes=(2, 0))         # (B, a1, a2, B')
    return xp.tensordot(y, x, axes=([0, 3], [0, 3])).reshape(4, 4)


def layer_envs(bra, ket, gates, idx, xp):
    """E_g = d<bra|layer|ket>/dG_g for the gates idx of one layer, and <bra|layer|ket>."""
    n = len(ket)
    start = {gates[g][0][0]: g for g in idx}
    blocks, i = [], 0
    while i < n:
        g = start.get(i)
        w = 1 if g is None else len(gates[g][0])
        blocks.append((i, w, g))
        i += w
    one = xp.ones((1, 1), dtype=ket[0].dtype)
    left, env = [one], one
    for i, w, g in blocks:
        env = _left(env, bra, ket, i, w, None if g is None else gates[g][1], xp)
        left.append(env)
    right, env = [one], one
    for i, w, g in reversed(blocks):
        env = _right(env, bra, ket, i, w, None if g is None else gates[g][1], xp)
        right.append(env)
    right.reverse()                      # right[b]: blocks b.. ; right[len(blocks)]: nothing
    out = {}
    for b, (i, w, g) in enumerate(blocks):
        if g is not None:
            out[g] = _gate_env(left[b], right[b + 1], bra, ket, i, w, xp)
    return out, complex(left[-1][0, 0])


def mps_value_and_envs(n, gates, target, chi, cutoff, xp, stride=1, keep=list, back=list):
    """f = <target|U|0> on the truncated forward state, and E_g = df/dG_g for every gate, from
    the forward state before each layer and the target taken back through the later layers
    (both truncated at chi). States before every stride-th layer are kept with keep() (e.g.
    copied to the host); the others are recomputed once per segment; back() restores them."""
    groups = layers(gates, n)
    ket, saved = mps_forward(n, gates, groups, chi, cutoff, xp, target[0].dtype, stride, keep)
    f = overlap(target, ket, xp)
    del ket
    bra = list(target)
    cb = move_center(bra, 0, n - 1, xp)
    envs, seg = [None] * len(gates), {}
    for lv in reversed(range(len(groups))):
        if lv not in seg:
            base = lv - lv % stride
            st, cs = saved.pop(base)
            st = back(st)
            seg = {base: st}
            for j in range(base, lv):
                st = list(st)
                cs = apply_layer(st, cs, gates, groups[j], chi, cutoff, xp)
                seg[j + 1] = st
        e, _ = layer_envs(bra, seg.pop(lv), gates, groups[lv], xp)
        for g, v in e.items():
            envs[g] = v
        cb = apply_layer(bra, cb, gates, groups[lv], chi, cutoff, xp, dagger=True)
    return f, envs


# ---------------------------------------------------------------- exact state vector
def dense_apply(psi, sites, m, xp):
    """Apply a gate on contiguous ascending sites to a flat state vector (qubit 0 most significant)."""
    n = int(round(math.log2(psi.size)))
    q, k = sites[0], len(sites)
    v = psi.reshape(2 ** q, 2 ** k, 2 ** (n - q - k))
    return xp.tensordot(m, v, axes=(1, 1)).transpose(1, 0, 2).reshape(-1)


def dense_value_and_envs(n, gates, target, xp):
    """Exact f = <target|U|0> and df/dG_g by the adjoint method (target: flat state vector)."""
    psi = xp.zeros(2 ** n, dtype=target.dtype)
    psi[0] = 1
    for sites, m in gates:
        psi = dense_apply(psi, sites, m, xp)
    f = complex(xp.vdot(target, psi))
    lam, envs = target, [None] * len(gates)
    for g in reversed(range(len(gates))):
        sites, m = gates[g]
        md = m.conj().T
        psi = dense_apply(psi, sites, md, xp)
        q, k = sites[0], len(sites)
        shape = (2 ** q, 2 ** k, 2 ** (n - q - k))
        envs[g] = xp.tensordot(lam.conj().reshape(shape), psi.reshape(shape), axes=([0, 2], [0, 2]))
        lam = dense_apply(lam, sites, md, xp)
    return f, envs


# ---------------------------------------------------------------- adapter (parameters)
def build_units(instrs):
    """Merge each one-qubit instruction into the two-qubit unit before it on its qubit (leading
    ones into the first unit after). Returns units: dict(sites=(lo, lo + 1), parts=[(instr,
    slot)] in application order with slot 'lo', 'hi' or 'both', params=sorted indices)."""
    units, last, lead = [], {}, {}
    for k, (qs, _, _) in enumerate(instrs):
        if len(qs) == 2:
            lo, hi = min(qs), max(qs)
            if hi != lo + 1:
                raise ValueError(f"non-neighbour gate on {qs}")
            parts = [(j, "lo") for j in lead.pop(lo, [])] + [(j, "hi") for j in lead.pop(hi, [])]
            units.append({"sites": (lo, hi), "parts": parts + [(k, "both")]})
            last[lo] = last[hi] = len(units) - 1
        elif len(qs) == 1:
            if qs[0] in last:
                u = units[last[qs[0]]]
                u["parts"].append((k, "lo" if qs[0] == u["sites"][0] else "hi"))
            else:
                lead.setdefault(qs[0], []).append(k)
        else:
            raise ValueError(f"gate on {len(qs)} qubits")
    if lead:
        raise ValueError(f"qubits {sorted(lead)} have no two-qubit gate")
    for u in units:
        u["params"] = sorted({p for j, _ in u["parts"] for p in instrs[j][1]})
    return units


def _embed(m, qs, slot):
    if slot == "both":
        return m[np.ix_(LOW_FIRST, LOW_FIRST)] if qs[0] < qs[1] else m
    return np.kron(m, np.eye(2)) if slot == "lo" else np.kron(np.eye(2), m)


def unit_matrices(units, instrs, x, h=1e-6):
    """Each unit's 4x4 matrix (quimb order) at x, and for each of its parameters the derivative
    (central differences of the instruction matrices, product rule). x is restored."""
    mats, derivs = [], []
    for u in units:
        parts = [_embed(instrs[j][2](x), instrs[j][0], slot) for j, slot in u["parts"]]
        pre, post = [np.eye(4, dtype=complex)], [np.eye(4, dtype=complex)]
        for g in parts:
            pre.append(g @ pre[-1])                # pre[k] = parts[k-1] ... parts[0]
        for g in reversed(parts):
            post.append(post[-1] @ g)              # post[k] = the last k parts
        mats.append(pre[-1])
        ds = []
        for p in u["params"]:
            d = np.zeros((4, 4), dtype=complex)
            for k, (j, slot) in enumerate(u["parts"]):
                qs, ps, fn = instrs[j]
                if p in ps:
                    x0 = x[p]
                    x[p] = x0 + h
                    gp = fn(x)
                    x[p] = x0 - h
                    gm = fn(x)
                    x[p] = x0
                    d += post[len(parts) - 1 - k] @ _embed((gp - gm) / (2 * h), qs, slot) @ pre[k]
            ds.append((p, d))
        derivs.append(ds)
    return mats, derivs


def loss_and_grad(f, envs, derivs, n_params):
    """1 - |f|^2 and its gradient, from df/dG (envs, one per unit) and dG/dθ (derivs)."""
    gf = np.zeros(n_params, dtype=complex)
    for e, ds in zip(envs, derivs):
        for p, d in ds:
            gf[p] += np.sum(d * e)
    return 1.0 - abs(f) ** 2, -2.0 * np.real(np.conj(f) * gf)
