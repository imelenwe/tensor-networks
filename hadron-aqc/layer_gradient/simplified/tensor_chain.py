"""Chains (MPS): a list with one array per qubit, indexed (left link, the qubit's 0/1, right link).

The weights sit at one site, the centre. A two-qubit block is applied at the centre, then the pair of sites is split
again with an SVD and trimmed to at most max_bond weights (no rescaling afterwards). Arrays are never changed in place,
so a copy of the list is a cheap snapshot of the chain.

xp is numpy (CPU) or cupy (GPU).
"""


def all_zeros_chain(n_qubits, xp, dtype):
    """|0...0>: every site says "this qubit is 0", every link has size 1."""
    site = xp.zeros((1, 2, 1), dtype=dtype)
    site[0, 0, 0] = 1
    return [site] * n_qubits


def move_centre(chain, centre, to, xp):
    """Move the centre from site `centre` to site `to`, one QR split per site; returns `to`."""
    while centre < to:
        left, _, right = chain[centre].shape
        q, r = xp.linalg.qr(chain[centre].reshape(left * 2, right))
        chain[centre] = q.reshape(left, 2, q.shape[1])
        chain[centre + 1] = xp.tensordot(r, chain[centre + 1], axes=(1, 0))
        centre += 1
    while centre > to:
        left, _, right = chain[centre].shape
        q, r = xp.linalg.qr(chain[centre].reshape(left, 2 * right).T)
        chain[centre] = q.T.reshape(q.shape[1], 2, right)
        chain[centre - 1] = xp.tensordot(chain[centre - 1], r.T, axes=(2, 0))
        centre -= 1
    return centre


def apply_block(chain, centre, site, block, max_bond, cutoff, xp, weights_go_right=True):
    """Apply a 4x4 block to sites (site, site + 1), split them again and trim; returns the new centre.
    The block is indexed [(out1 out2), (in1 in2)], site the more significant bit."""
    centre = move_centre(chain, centre, site if centre <= site else site + 1, xp)
    left, right = chain[site].shape[0], chain[site + 1].shape[2]
    pair = xp.tensordot(chain[site], chain[site + 1], axes=(2, 0))                     # (left, in1, in2, right)
    pair = xp.tensordot(block.reshape(2, 2, 2, 2), pair, axes=([2, 3], [1, 2]))        # (out1, out2, left, right)
    u, weights, vh = xp.linalg.svd(pair.transpose(2, 0, 1, 3).reshape(left * 2, 2 * right), full_matrices=False)
    kept = max(1, min(max_bond, int((weights > cutoff * weights[0]).sum())))           # trimming
    u, weights, vh = u[:, :kept], weights[:kept], vh[:kept]
    if weights_go_right:
        chain[site], chain[site + 1] = u.reshape(left, 2, kept), (weights[:, None] * vh).reshape(kept, 2, right)
        return site + 1
    chain[site], chain[site + 1] = (u * weights).reshape(left, 2, kept), vh.reshape(kept, 2, right)
    return site


def overlap(bra, ket, xp):
    """<bra|ket>: walk left to right, carrying a small (bra link, ket link) table; at the end it is 1 x 1."""
    partial = xp.ones((1, 1), dtype=ket[0].dtype)
    for bra_site, ket_site in zip(bra, ket):
        partial = xp.tensordot(bra_site.conj(), xp.tensordot(partial, ket_site, axes=(1, 0)), axes=([0, 1], [0, 1]))
    return complex(partial[0, 0])
