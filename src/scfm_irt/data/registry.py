"""Dataset registry.

Two families matter for this evaluation (plan section 11):

- **Annotation benchmarks** (Zheng68k, hPancreas, MS, PBMC variants, ...) — the standard eval
  sets, mostly modest class counts.
- **Atlas / integration sets** (Immune atlas, Lung atlas, Heart atlas, ...) — high class counts
  and many donors, which is where the *difficulty spread* IRT needs comes from (MATH.md
  section 8.3). Prioritize a few of these; without them most items sit at the ceiling and
  abilities barely separate.

Several pancreas/immune/PBMC sets recur across benchmarks under different names and
preprocessing — dedupe before pooling or the same cells get counted twice (plan section 10.7).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    path: str  # local .h5ad, after download_data.py
    label_key: str  # .obs column holding the ground-truth cell type
    batch_key: str | None = None
    donor_key: str | None = None
    url: str | None = None  # source, for download_data.py
    family: str = "annotation"  # "annotation" | "atlas"
    notes: str = ""
    # Filled in by `describe`; kept here only as the expected value to check against, so a
    # silently-transposed or re-preprocessed download is caught early (plan section 11).
    expected_n_cells: int | None = None
    expected_n_types: int | None = None


# TODO(phase-0): fill in paths/urls and the expected dims from the project inventory.
_REGISTRY: dict[str, DatasetSpec] = {
    "zheng68k": DatasetSpec(
        name="zheng68k",
        path="data/raw/zheng68k.h5ad",
        label_key="celltype",
        donor_key=None,
        family="annotation",
        notes="Heavily imbalanced; prefer macro-averaged reporting (plan section 10.9).",
    ),
    "hpancreas": DatasetSpec(
        name="hpancreas",
        path="data/raw/hpancreas.h5ad",
        label_key="Celltype",
        batch_key="tech",
        family="annotation",
        notes="Multi-protocol; cross_batch split is the natural hard setting.",
    ),
    "immune_human": DatasetSpec(
        name="immune_human",
        path="data/raw/immune_human.h5ad",
        label_key="final_annotation",
        batch_key="batch",
        donor_key="donor",
        family="atlas",
        notes="scIB integration set; high class count, good difficulty spread.",
    ),
    "lung_atlas": DatasetSpec(
        name="lung_atlas",
        path="data/raw/lung_atlas.h5ad",
        label_key="cell_type",
        batch_key="batch",
        donor_key="donor",
        family="atlas",
        notes="~50 cell types; a priority source of hard items.",
    ),
}

# Benchmarks refer to the same underlying data under different names; resolve here so the
# dedupe rule of plan section 10.7 has a single place to live.
_ALIASES: dict[str, str] = {
    "pbmc68k": "zheng68k",
    "pancreas": "hpancreas",
}


def list_datasets(family: str | None = None) -> list[str]:
    """Registered dataset names, optionally filtered to one family."""
    return sorted(n for n, s in _REGISTRY.items() if family is None or s.family == family)


def get_spec(name: str) -> DatasetSpec:
    key = _ALIASES.get(name, name)
    if key not in _REGISTRY:
        raise KeyError(f"unknown dataset {name!r}; registered: {list_datasets()}")
    return _REGISTRY[key]


def register(spec: DatasetSpec) -> None:
    """Add a dataset at runtime (used by tests and by ad-hoc experiments)."""
    if spec.name in _REGISTRY:
        raise ValueError(f"dataset {spec.name!r} is already registered")
    _REGISTRY[spec.name] = spec
