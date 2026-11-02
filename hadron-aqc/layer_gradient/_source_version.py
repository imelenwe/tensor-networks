"""Migration steps 1-3: the interface (see __init__.py) on top of the unchanged files in source/.
Deleted together with source/ in step 5."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "source"))   # source/ imports its own files by bare name
import fastgrad_core as core                                          # noqa: E402
from mps_objective import MPSGradientObjective, target_from_quimb     # noqa: E402


class LayerGradientObjective:
    def __init__(self, ansatz, target_mps, settings):
        xp, keep_saved_states_in_host_memory = np, False
        if settings.use_gpu:
            import cupy as xp
            keep_saved_states_in_host_memory = True
        dtype = np.complex64 if settings.single_precision else np.complex128
        self.inner = MPSGradientObjective(ansatz, target_from_quimb(target_mps), chi=settings.max_bond, xp=xp,
                                          dtype=dtype, stride=settings.stride, host=keep_saved_states_in_host_memory)

    def __call__(self, angles):
        return self.inner(angles)

    def layer_overlaps(self, angles):
        """<B_k|S_k> at every layer k, last layer first: the backward layer pass, keeping each layer's overlap."""
        inner = self.inner
        gates, _ = inner.gates(angles)
        groups = core.layers(gates, inner.n)
        _, saved = core.mps_forward(inner.n, gates, groups, inner.chi, inner.cutoff, inner.xp, inner.target[0].dtype,
                                    stride=1)
        target_side = list(inner.target)
        centre = core.move_center(target_side, 0, inner.n - 1, inner.xp)
        overlaps = []
        for layer in reversed(range(len(groups))):
            state_before, _ = saved[layer]
            _, overlap = core.layer_envs(target_side, state_before, gates, groups[layer], inner.xp)
            overlaps.append(overlap)
            centre = core.apply_layer(target_side, centre, gates, groups[layer], inner.chi, inner.cutoff, inner.xp,
                                      dagger=True)
        return overlaps
