"""Config loading and reproducibility.

A typo in a config is otherwise a silently wrong run, and an unstable config hash breaks the
"same config + same seed => identical results" guarantee (plan section 5.2).
"""

from __future__ import annotations

import pytest
import yaml

from scfm_irt.config import RunConfig, load_config

CONFIGS = ["annotation_irt_v1", "smoke_dev", "explanatory_v1"]


@pytest.mark.parametrize("name", CONFIGS)
def test_shipped_configs_load(name):
    cfg = load_config(f"configs/{name}.yaml")
    assert cfg.run_name == name
    assert cfg.datasets and cfg.embedders


def test_config_hash_is_stable_and_sensitive(tmp_path):
    base = RunConfig(run_name="r", datasets=["d"], embedders=["e"], seed=0)
    same = RunConfig(run_name="r", datasets=["d"], embedders=["e"], seed=0)
    other = RunConfig(run_name="r", datasets=["d"], embedders=["e"], seed=1)

    assert base.config_hash() == same.config_hash()
    assert base.config_hash() != other.config_hash()


def test_unknown_key_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        yaml.safe_dump({"run_name": "r", "datasets": ["d"], "embedders": ["e"], "sed": 0})
    )
    with pytest.raises(ValueError, match="unknown config keys"):
        load_config(path)


def test_run_dir_derives_from_name():
    cfg = RunConfig(run_name="my_run", datasets=["d"], embedders=["e"], outputs_dir="runs")
    assert str(cfg.run_dir) == "runs/my_run"


def test_explanatory_covariates_require_numpyro():
    """The one cross-field constraint worth enforcing at fit time."""
    from scfm_irt.irt.fit import fit_irt

    with pytest.raises(ValueError, match="numpyro"):
        fit_irt(None, method="bayes_variational", item_covariates=["log_depth"])
