"""Wright (item-person) map: abilities and difficulties on the shared latent scale.

The single most communicative figure in the report — it shows at a glance whether the item pool
brackets the models or sits entirely below them.
"""

from __future__ import annotations

from scfm_irt.irt.models import IRTFit


def plot_wright_map(fit: IRTFit, path: str, bins: int = 60) -> None:
    """Difficulty histogram on one side of a shared theta axis, model abilities on the other.

    Optionally group difficulties by cell type so the reader can see *which* types occupy the
    hard end of the scale.
    """
    raise NotImplementedError("TODO(phase-8)")
