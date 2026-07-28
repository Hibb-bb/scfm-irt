"""The analysis half must never import an embedder.

This is the architectural boundary as an executable check rather than a README promise: if
someone adds `from scfm_irt.embedders import ...` to the IRT or analysis code, the light
environment stops being installable and CI stops being able to run at all.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "scfm_irt"
LIGHT_MODULES = ["probe", "response", "irt", "analysis"]


def _imported_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


@pytest.mark.parametrize("package", LIGHT_MODULES)
def test_light_side_does_not_import_embedders(package):
    offenders = [
        (path.relative_to(SRC), name)
        for path in (SRC / package).rglob("*.py")
        for name in _imported_names(path)
        if "embedders" in name
    ]
    assert not offenders, f"light-side modules import embedders: {offenders}"


@pytest.mark.parametrize("package", LIGHT_MODULES)
def test_light_side_does_not_import_heavy_frameworks_at_module_scope(package):
    """torch/pyro/jax belong inside estimator functions, so the package imports without them."""
    heavy = {"torch", "pyro", "jax", "numpyro", "py_irt", "transformers"}
    offenders = [
        (path.relative_to(SRC), name)
        for path in (SRC / package).rglob("*.py")
        for name in _imported_names(path)
        if name.split(".")[0] in heavy
    ]
    assert not offenders, f"module-scope heavy imports: {offenders}"


def test_package_imports_without_optional_dependencies():
    for mod in ["scfm_irt", "scfm_irt.irt", "scfm_irt.response", "scfm_irt.analysis"]:
        __import__(mod)
    assert "torch" not in sys.modules
    assert "py_irt" not in sys.modules


def test_embedder_registry_does_not_import_models_eagerly():
    """Factories are lazy — listing embedders must not drag in any model's dependency stack."""
    from scfm_irt.embedders.registry import list_embedders

    assert "dev_pca" in list_embedders()
    assert "torch" not in sys.modules
