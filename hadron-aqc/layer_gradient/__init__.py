"""The layer gradient: training slopes from one forward pass and one backward layer pass on trimmed chains.

    from layer_gradient import LayerGradientSettings, LayerGradientObjective
    objective = LayerGradientObjective(ansatz, target_mps, LayerGradientSettings(max_bond=64))
    loss, slopes = objective(angles)              # L = 1 - fidelity, and dL/d(angle) for every angle
    overlaps = objective.layer_overlaps(angles)   # <B_k|S_k> at every layer (all equal the overlap)

See MIGRATION.md for the files, the tests and the plan.
"""
import os
from dataclasses import dataclass


@dataclass
class LayerGradientSettings:
    max_bond: int                    # link size cap for the forward state and the target side
    stride: int = 6                  # keep the forward state before every stride-th layer; the rest are rebuilt
    use_gpu: bool = False            # cupy on a GPU instead of numpy
    single_precision: bool = False   # complex64 instead of complex128: half the memory


VERSION = os.environ.get("LAYER_GRADIENT_VERSION", "source")   # migration switch (MIGRATION.md); removed in step 5
if VERSION == "source":
    from ._source_version import LayerGradientObjective
elif VERSION == "simplified":
    from .simplified import LayerGradientObjective
else:
    raise ValueError(f"LAYER_GRADIENT_VERSION must be source or simplified, not {VERSION!r}")

__all__ = ["LayerGradientSettings", "LayerGradientObjective"]
