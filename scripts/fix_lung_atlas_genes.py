#!/usr/bin/env python
"""Fix gene symbols on an existing lung_atlas.h5ad (census numeric var index)."""

from __future__ import annotations

import argparse

import anndata as ad

from scfm_irt.data.h5ad_compat import harmonize_var_names, sanitize_adata, write_h5ad_compat


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", default="data/raw/lung_atlas.h5ad")
    args = parser.parse_args()

    print(f"[fix-lung] reading {args.path}")
    adata = ad.read_h5ad(args.path)
    print(f"[fix-lung] before: var_names[:5]={adata.var_names[:5].tolist()}")
    fixed = harmonize_var_names(adata)
    print(f"[fix-lung] after:  var_names[:5]={fixed.var_names[:5].tolist()}")
    write_h5ad_compat(sanitize_adata(fixed), args.path)
    print(f"[fix-lung] rewrote {args.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
