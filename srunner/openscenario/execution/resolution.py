"""Backend resolution boundary for semantic positions."""

from __future__ import absolute_import

from srunner.openscenario.execution.registry import UnsupportedFeatureError
from srunner.openscenario.validation.diagnostics import Diagnostic


def resolve_position(position, context):
    """Resolve a semantic position only at the backend boundary."""
    resolver = getattr(context.backend, "resolve_position", None) if context.backend is not None else None
    if resolver is None:
        raise UnsupportedFeatureError(position, Diagnostic(
            "unsupported_feature", "Backend does not implement position resolution",
            source=getattr(position, "source", None)))
    return resolver(position, context)
