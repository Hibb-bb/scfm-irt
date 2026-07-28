"""Dataset registry, loading, splits, and label harmonization."""

from scfm_irt.data.registry import DatasetSpec, get_spec, list_datasets
from scfm_irt.data.splits import Split, make_split

__all__ = ["DatasetSpec", "Split", "get_spec", "list_datasets", "make_split"]
