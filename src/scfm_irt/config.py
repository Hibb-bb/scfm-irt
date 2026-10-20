"""Run configuration.

A run is fully declared by one YAML file: which datasets, which split scheme, which embedders,
the probe settings, and the IRT model. Nothing downstream is hard-coded — see
docs/IMPLEMENTATION_PLAN.md section 4.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

SplitScheme = Literal["stratified", "cross_donor", "cross_batch"]
IRTModel = Literal["1pl", "2pl", "3pl", "4pl"]
IRTMethod = Literal["mle_autodiff", "bayes_variational", "numpyro"]
ItemGranularity = Literal["cell", "type_dataset"]


@dataclass(frozen=True)
class SplitConfig:
    scheme: SplitScheme = "cross_donor"
    test_frac: float = 0.2


@dataclass(frozen=True)
class ProbeConfig:
    penalty: Literal["l2", "l1"] = "l2"
    C: float | None = None  # None => select by stratified CV on the train split
    max_iter: int = 1000


@dataclass(frozen=True)
class IRTConfig:
    model: IRTModel = "1pl"
    method: IRTMethod = "bayes_variational"
    item_granularity: ItemGranularity = "cell"
    # None => standard (non-explanatory) model. A list enables the LLTM-style explanatory
    # fit of MATH.md section 11.1, which predicts difficulty from cell covariates.
    item_covariates: list[str] | None = None
    n_steps: int = 2000
    # Leave-one-model-out: calibrate item difficulties without the scored model (≥3 models).
    loo_calibration: bool = True
    loo_n_steps: int = 400


@dataclass(frozen=True)
class RunConfig:
    run_name: str
    datasets: list[str]
    embedders: list[str]
    seed: int = 0
    split: SplitConfig = field(default_factory=SplitConfig)
    probe: ProbeConfig = field(default_factory=ProbeConfig)
    irt: IRTConfig = field(default_factory=IRTConfig)
    outputs_dir: str = "runs"
    cache_dir: str = "cache/embeddings"

    @property
    def run_dir(self) -> Path:
        return Path(self.outputs_dir) / self.run_name

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def config_hash(self) -> str:
        """Stable hash of the full config, logged into every run directory.

        Same config + same seed must reproduce an identical response table (plan section 5.2).
        """
        import hashlib

        payload = yaml.safe_dump(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:12]


def load_config(path: str | Path) -> RunConfig:
    """Read a run config from YAML.

    Unknown top-level keys are rejected rather than silently ignored — a typo in a config is a
    silently wrong run otherwise.
    """
    raw = yaml.safe_load(Path(path).read_text()) or {}

    known = set(RunConfig.__dataclass_fields__)
    unknown = set(raw) - known
    if unknown:
        raise ValueError(
            f"unknown config keys: {sorted(unknown)}; expected a subset of {sorted(known)}"
        )

    for key, cls in (("split", SplitConfig), ("probe", ProbeConfig), ("irt", IRTConfig)):
        if key in raw and raw[key] is not None:
            raw[key] = cls(**raw[key])

    return RunConfig(**raw)
