"""The response table — the bridge between the embedding half and the IRT half."""

from scfm_irt.response.build import build_response_table, to_matrix
from scfm_irt.response.schema import RESPONSE_SCHEMA, validate_response_table

__all__ = [
    "RESPONSE_SCHEMA",
    "build_response_table",
    "to_matrix",
    "validate_response_table",
]
