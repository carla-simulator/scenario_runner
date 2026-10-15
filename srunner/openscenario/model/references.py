"""Semantic references retained before backend resolution."""

from __future__ import absolute_import


class CatalogReference(object):
    def __init__(self, catalog_name, entry_name, parameter_assignments=None, source=None):
        self.catalog_name = catalog_name
        self.entry_name = entry_name
        self.parameter_assignments = dict(parameter_assignments or {})
        self.source = source


class EntityReference(object):
    def __init__(self, entity_ref, source=None):
        self.entity_ref = entity_ref
        self.source = source


class Position(object):
    """Coordinate-system-neutral position boundary."""

    def __init__(self, kind, attributes=None, source=None):
        self.kind = kind
        self.attributes = dict(attributes or {})
        self.source = source


class Route(object):
    def __init__(self, name=None, strategy=None, positions=None, source=None):
        self.name = name
        self.strategy = strategy
        self.positions = list(positions or [])
        self.source = source


class Trajectory(object):
    def __init__(self, name=None, shape=None, time_reference=None, following_mode=None,
                 control_points=None, source=None):
        self.name = name
        self.shape = shape
        self.time_reference = time_reference
        self.following_mode = following_mode
        self.control_points = list(control_points or [])
        self.source = source


class Condition(object):
    def __init__(self, name=None, edge=None, source=None):
        self.name = name
        self.edge = edge
        self.source = source
