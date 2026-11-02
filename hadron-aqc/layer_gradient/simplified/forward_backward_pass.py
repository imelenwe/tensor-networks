"""The layer gradient's two passes over the circuit (workbook: learn/circuit-backprop-workbook.html, section 6).

A gate is (site, block): a 4x4 block on sites (site, site + 1), indexed [(out1 out2), (in1 in2)].
Forward pass: S_0 = |0...0>, S_k = layer k applied to S_(k-1); keep S before every stride-th layer.
Backward layer pass, last layer first: B = T; at layer k combine S_(k-1) (before) and B_k (after) into one 4x4 table
per gate, d<B_k| layer k |S_(k-1)> / d(block); then undo layer k on B (conjugate transpose), giving B_(k-1).
"""
from .tensor_chain import all_zeros_chain, move_centre, apply_block, overlap


def split_into_layers(gates, n_qubits):
    """Layers of gates that share no qubit: each gate goes one layer after the last gate on its qubits."""
    first_free_layer = [0] * n_qubits
    layers = []
    for index, (site, _) in enumerate(gates):
        layer = max(first_free_layer[site], first_free_layer[site + 1])
        if layer == len(layers):
            layers.append([])
        layers[layer].append(index)
        first_free_layer[site] = first_free_layer[site + 1] = layer + 1
    return layers


def apply_layer(chain, centre, gates, layer, max_bond, cutoff, xp, undo=False):
    """Apply one layer's gates, or undo them (conjugate transpose, for the target side). Sweeps away from the centre,
    so the centre moves as little as possible; returns the new centre."""
    order = sorted(layer, key=lambda index: gates[index][0])
    first_site, last_site = gates[order[0]][0], gates[order[-1]][0] + 1
    rightwards = centre <= (first_site + last_site) / 2
    for index in (order if rightwards else order[::-1]):
        site, block = gates[index]
        centre = apply_block(chain, centre, site, block.conj().T if undo else block, max_bond, cutoff, xp,
                             weights_go_right=rightwards)
    return centre


# ---- the 16-number tables: d<B| layer |S> / d(block) for every gate of one layer ----
# Indices in the comments: B, K = bra and ket links; in, out = the qubit's 0/1 before and after the gate.

def _grow_from_left(partial, bra, ket, site, block, xp):
    """Add one empty site (block None) or one gate's two sites to a partial overlap from the left (B, K)."""
    t = xp.tensordot(partial, ket[site], axes=(1, 0))                                   # (B, in, K)
    if block is None:
        return xp.tensordot(bra[site].conj(), t, axes=([0, 1], [0, 1]))
    t = xp.tensordot(t, ket[site + 1], axes=(2, 0))                                     # (B, in1, in2, K)
    t = xp.tensordot(block.reshape(2, 2, 2, 2), t, axes=([2, 3], [1, 2]))               # (out1, out2, B, K)
    t = xp.tensordot(bra[site].conj(), t, axes=([0, 1], [2, 0]))                        # (B, out2, K)
    return xp.tensordot(bra[site + 1].conj(), t, axes=([0, 1], [0, 1]))


def _grow_from_right(partial, bra, ket, site, block, xp):
    """The same from the right."""
    last = site if block is None else site + 1
    t = xp.tensordot(ket[last], partial, axes=(2, 1))                                   # (K, in, B)
    if block is None:
        return xp.tensordot(bra[site].conj(), t, axes=([1, 2], [1, 2]))
    t = xp.tensordot(ket[site], t, axes=(2, 0))                                         # (K, in1, in2, B)
    t = xp.tensordot(block.reshape(2, 2, 2, 2), t, axes=([2, 3], [1, 2]))               # (out1, out2, K, B)
    t = xp.tensordot(bra[site + 1].conj(), t, axes=([1, 2], [1, 3]))                    # (B, out1, K)
    return xp.tensordot(bra[site].conj(), t, axes=([1, 2], [1, 0]))


def _table(from_left, from_right, bra, ket, site, xp):
    """Everything around one gate, the gate left out: a 4x4 table [(out1 out2), (in1 in2)]."""
    x = xp.tensordot(from_left, ket[site], axes=(1, 0))                                 # (B, in1, K)
    x = xp.tensordot(x, ket[site + 1], axes=(2, 0))                                     # (B, in1, in2, K)
    x = xp.tensordot(x, from_right, axes=(3, 1))                                        # (B, in1, in2, B')
    y = xp.tensordot(bra[site].conj(), bra[site + 1].conj(), axes=(2, 0))               # (B, out1, out2, B')
    return xp.tensordot(y, x, axes=([0, 3], [0, 3])).reshape(4, 4)


def layer_tables(bra, ket, gates, layer, xp):
    """One table per gate of the layer, from one sweep from the left and one from the right (every partial result
    kept), and the overlap <bra| layer |ket>."""
    gate_at = {gates[index][0]: index for index in layer}
    pieces, site = [], 0                       # (site, gate index or None); a gate covers two sites
    while site < len(ket):
        pieces.append((site, gate_at.get(site)))
        site += 1 if gate_at.get(site) is None else 2
    block_of = lambda index: None if index is None else gates[index][1]     # noqa: E731
    one = xp.ones((1, 1), dtype=ket[0].dtype)
    from_left = [one]
    for site, index in pieces:
        from_left.append(_grow_from_left(from_left[-1], bra, ket, site, block_of(index), xp))
    from_right = [one]
    for site, index in reversed(pieces):
        from_right.append(_grow_from_right(from_right[-1], bra, ket, site, block_of(index), xp))
    from_right.reverse()                       # from_right[p]: pieces p, p + 1, ...; from_right[-1]: nothing
    tables = {index: _table(from_left[p], from_right[p + 1], bra, ket, site, xp)
              for p, (site, index) in enumerate(pieces) if index is not None}
    return tables, complex(from_left[-1][0, 0])


def forward_and_backward(gates, n_qubits, target, max_bond, cutoff, xp, stride, keep=list, restore=list):
    """The forward pass, then the backward layer pass.
    Returns o = <T|S_final> (on the trimmed forward state), one 4x4 table per gate (do/d block), and <B_k|S_k> for
    every layer, last layer first. keep/restore copy the kept states (e.g. to host memory and back on a GPU)."""
    layers = split_into_layers(gates, n_qubits)

    state, centre, kept = all_zeros_chain(n_qubits, xp, target[0].dtype), 0, {}
    for number, layer in enumerate(layers):
        if number % stride == 0:
            kept[number] = (keep(state), centre)
        centre = apply_layer(state, centre, gates, layer, max_bond, cutoff, xp)
    value = overlap(target, state, xp)
    del state

    target_side = list(target)
    target_centre = move_centre(target_side, 0, n_qubits - 1, xp)
    tables, layer_overlaps, states_before = [None] * len(gates), [], {}
    for number in reversed(range(len(layers))):
        if number not in states_before:        # rebuild this stretch of states, once, from the nearest kept one
            first = number - number % stride
            state, centre = kept.pop(first)
            state = restore(state)
            states_before = {first: state}
            for earlier in range(first, number):
                state = list(state)
                centre = apply_layer(state, centre, gates, layers[earlier], max_bond, cutoff, xp)
                states_before[earlier + 1] = state
        these_tables, layer_overlap = layer_tables(target_side, states_before.pop(number), gates, layers[number], xp)
        for index, table in these_tables.items():
            tables[index] = table
        layer_overlaps.append(layer_overlap)
        target_centre = apply_layer(target_side, target_centre, gates, layers[number], max_bond, cutoff, xp, undo=True)
    return value, tables, layer_overlaps
