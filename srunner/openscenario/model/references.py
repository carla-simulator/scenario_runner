"""Semantic references retained before backend resolution."""

from __future__ import absolute_import


class CatalogReference(object):
    def __init__(self, catalog_name, entry_name, parameter_assignments=None, source=None):
        self.catalog_name = catalog_name
        self.entry_name = entry_name
        self.parameter_assignments = dict(parameter_assignments or {})
        self.source = source


class CatalogEntry(object):
    """Resolved semantic catalog entry; backend resolution is deliberately later."""

    def __init__(self, catalog_type, name, element=None, parameters=None, source=None):
        self.catalog_type = catalog_type
        self.name = name
        self.element = element
        self.parameters = dict(parameters or {})
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

    @classmethod
    def from_xml(cls, element):
        if element is None:
            raise ValueError("Position element is required")
        children = list(element)
        if len(children) != 1:
            raise ValueError("Position must contain exactly one position type")
        child = children[0]
        kind = child.tag
        attributes = dict(child.attrib)
        if kind == "RoutePosition":
            route = child.find("RouteRef")
            in_route = child.find("InRoutePosition")
            if route is None or (route.find("Route") is None and route.find("CatalogReference") is None) or in_route is None:
                raise ValueError("RoutePosition requires a RouteRef reference")
            route_ref = route.find("CatalogReference")
            if route_ref is None:
                route_ref = route.find("Route")
            attributes["route_ref"] = dict(route_ref.attrib)
            attributes["route_ref_type"] = route_ref.tag
            attributes["in_route"] = dict(in_route.attrib) if in_route is not None else {}
            return RoutePosition(attributes["route_ref"], attributes["in_route"], element)
        return cls(kind, attributes, element)


class RoutePosition(Position):
    """Route position retained semantically; execution belongs to PR 3."""

    def __init__(self, route_ref, in_route=None, source=None):
        attributes = {"route_ref": dict(route_ref), "in_route": dict(in_route or {})}
        super(RoutePosition, self).__init__("RoutePosition", attributes, source)


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
