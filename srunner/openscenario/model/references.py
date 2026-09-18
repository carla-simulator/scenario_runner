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
    """Semantic OSC route; positions remain OSC objects until backend resolution."""

    def __init__(self, name=None, strategy=None, positions=None, closed=False,
                 parameter_declarations=None, source=None):
        self.name = name
        self.strategy = strategy
        self.positions = list(positions or [])
        self.closed = closed
        self.parameter_declarations = list(parameter_declarations or [])
        self.source = source

    @classmethod
    def from_xml(cls, element):
        if element is None or element.tag != "Route":
            raise ValueError("Route element is required")
        positions = []
        for waypoint in element.findall("Waypoint"):
            position = waypoint.find("Position")
            if position is None:
                raise ValueError("Route waypoint requires a Position")
            positions.append({
                "route_strategy": waypoint.attrib.get("routeStrategy"),
                "position": Position.from_xml(position),
                "source": waypoint,
            })
        if len(positions) < 2:
            raise ValueError("Route requires at least two waypoints")
        return cls(name=element.attrib.get("name"),
                   positions=positions,
                   closed=element.attrib.get("closed", "false").lower() == "true",
                   source=element)


class Trajectory(object):
    """Semantic trajectory preserving shape and timing semantics."""

    def __init__(self, name=None, shape=None, time_reference=None, following_mode=None,
                 control_points=None, knots=None, order=None, closed=False, source=None):
        self.name = name
        self.shape = shape
        self.time_reference = time_reference
        self.following_mode = following_mode
        self.control_points = list(control_points or [])
        self.knots = list(knots or [])
        self.order = order
        self.closed = closed
        self.source = source

    @classmethod
    def from_xml(cls, element, time_reference=None, following_mode=None):
        if element is None or element.tag != "Trajectory":
            raise ValueError("Trajectory element is required")
        shape = element.find("Shape")
        if shape is None or len(shape) != 1:
            raise ValueError("Trajectory requires exactly one shape")
        shape_element = list(shape)[0]
        shape_name = shape_element.tag
        points = []
        knots = []
        order = None
        if shape_name == "Polyline":
            for vertex in shape_element.findall("Vertex"):
                position = vertex.find("Position")
                if position is None or "time" not in vertex.attrib:
                    raise ValueError("Polyline vertex requires time and Position")
                points.append({"time": float(vertex.attrib["time"]),
                               "position": Position.from_xml(position),
                               "source": vertex})
            if len(points) < 2:
                raise ValueError("Polyline requires at least two vertices")
        elif shape_name == "Clothoid":
            position = shape_element.find("Position")
            points = [{"position": Position.from_xml(position),
                       "attributes": dict(shape_element.attrib),
                       "source": shape_element}]
        elif shape_name == "Nurbs":
            points = [{"attributes": dict(point.attrib),
                       "position": Position.from_xml(point.find("Position")),
                       "source": point} for point in shape_element.findall("ControlPoint")]
            if len(points) < 2:
                raise ValueError("Nurbs requires at least two control points")
            knots = [float(k.attrib["value"]) for k in shape_element.findall("Knot")]
            order = int(shape_element.attrib.get("order", "0"))
        else:
            raise ValueError("Unsupported trajectory shape {!r}".format(shape_name))
        return cls(name=element.attrib.get("name"), shape=shape_name,
                   time_reference=time_reference, following_mode=following_mode,
                   control_points=points,
                   knots=knots, order=order,
                   closed=element.attrib.get("closed", "false").lower() == "true",
                   source=element)


class Condition(object):
    def __init__(self, name=None, edge=None, source=None):
        self.name = name
        self.edge = edge
        self.source = source


class RelativeAngleCondition(Condition):
    kind = "RelativeAngleCondition"
    action_type = "RelativeAngleCondition"

    def __init__(self, entity_ref, angle, rule, coordinate_system="entity", name=None, edge=None, source=None):
        super(RelativeAngleCondition, self).__init__(name=name, edge=edge, source=source)
        self.entity_ref = entity_ref
        self.angle = angle
        self.rule = rule
        self.coordinate_system = coordinate_system


class RelativeClearanceCondition(Condition):
    kind = "RelativeClearanceCondition"
    action_type = "RelativeClearanceCondition"

    def __init__(self, entity_ref, distance, freespace, rule, distance_type="cartesianDistance",
                 name=None, edge=None, source=None):
        super(RelativeClearanceCondition, self).__init__(name=name, edge=edge, source=source)
        self.entity_ref = entity_ref
        self.distance = distance
        self.freespace = freespace
        self.rule = rule
        self.distance_type = distance_type
