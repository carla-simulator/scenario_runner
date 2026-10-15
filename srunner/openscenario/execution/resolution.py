"""Backend resolution boundary for semantic positions."""

from __future__ import absolute_import

from srunner.openscenario.execution.registry import backend_capability_error


def resolve_position(position, context):
    """Resolve a semantic position only at the backend boundary."""
    resolver = getattr(context.backend, "resolve_position", None) if context.backend is not None else None
    if resolver is None:
        raise backend_capability_error(position, "Backend does not implement position resolution")
    return resolver(position, context)
