"""Semantic references retained before backend resolution."""

from __future__ import absolute_import

from srunner.openscenario.validation.diagnostics import Diagnostic, ValidationError


def invalid(message, source=None):
    """Return the ValidationError raised for a structurally invalid document element."""
    return ValidationError([Diagnostic("semantic_error", message, source=source)])


class CatalogReference(object):
    def __init__(self, catalog_name, entry_name, parameter_assignments=None, source=None):
        self.catalog_name = catalog_name
        self.entry_name = entry_name
        self.parameter_assignments = dict(parameter_assignments or {})
        self.source = source

    @classmethod
    def from_xml(cls, element):
        if element is None:
            return None
        assignments = {}
        container = element.find("ParameterAssignments")
        if container is not None:
            for assignment in container.findall("ParameterAssignment"):
                assignments[assignment.attrib.get("parameterRef")] = assignment.attrib.get("value")
        return cls(element.attrib.get("catalogName"), element.attrib.get("entryName"), assignments, element)


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
            raise invalid("Position element is required")
        children = list(element)
        if len(children) != 1:
            raise invalid("Position must contain exactly one position type", element)
        child = children[0]
        if child.tag == "RoutePosition":
            return RoutePosition.from_xml(child, element)
        return cls(child.tag, dict(child.attrib), element)


class RoutePosition(Position):
    """Position along a route; the route stays a typed reference until a backend resolves it."""

    def __init__(self, route_ref, in_route=None, source=None, route=None, catalog_reference=None):
        attributes = {"route_ref": dict(route_ref), "in_route": dict(in_route or {})}
        super(RoutePosition, self).__init__("RoutePosition", attributes, source)
        self.route = route
        self.catalog_reference = catalog_reference

    @classmethod
    def from_xml(cls, element, source=None):
        route_ref = element.find("RouteRef")
        in_route = element.find("InRoutePosition")
        if route_ref is None or in_route is None:
            raise invalid("RoutePosition requires RouteRef and InRoutePosition", source)
        route_element = route_ref.find("Route")
        reference_element = route_ref.find("CatalogReference")
        if route_element is None and reference_element is None:
            raise invalid("RoutePosition requires a RouteRef reference", source)
        in_route_children = list(in_route)
        in_route_attributes = dict(in_route_children[0].attrib) if len(in_route_children) == 1 else {}
        if in_route_children:
            in_route_attributes["kind"] = in_route_children[0].tag
        referenced = reference_element if reference_element is not None else route_element
        return cls(dict(referenced.attrib), in_route_attributes, source,
                   route=Route.from_xml(route_element) if route_element is not None else None,
                   catalog_reference=CatalogReference.from_xml(reference_element))


class Route(object):
    """Semantic OSC route; positions remain OSC objects until backend resolution.

    ``closed`` and the waypoint route strategies stay lexical (they may be parameter references).
    """

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
            raise invalid("Route element is required", element)
        positions = []
        for waypoint in element.findall("Waypoint"):
            position = waypoint.find("Position")
            if position is None:
                raise invalid("Route waypoint requires a Position", waypoint)
            positions.append({
                "route_strategy": waypoint.attrib.get("routeStrategy"),
                "position": Position.from_xml(position),
                "source": waypoint,
            })
        if len(positions) < 2:
            raise invalid("Route {!r} requires at least two waypoints".format(element.attrib.get("name")), element)
        return cls(name=element.attrib.get("name"),
                   positions=positions,
                   closed=element.attrib.get("closed", "false"),
                   source=element)


class TimeReference(object):
    """``TimeReference`` of a FollowTrajectoryAction: either ``None`` or ``Timing``."""

    NONE = "None"
    TIMING = "Timing"
    DOMAINS = ("absolute", "relative")

    def __init__(self, kind, domain=None, offset=None, scale=None, source=None):
        self.kind = kind
        self.domain = domain
        self.offset = offset
        self.scale = scale
        self.source = source

    @property
    def uses_timing(self):
        return self.kind == self.TIMING

    @classmethod
    def from_xml(cls, element):
        if element is None:
            raise invalid("FollowTrajectoryAction requires a TimeReference")
        children = list(element)
        if len(children) != 1 or children[0].tag not in (cls.NONE, cls.TIMING):
            raise invalid("TimeReference must contain exactly one of None or Timing", element)
        child = children[0]
        if child.tag == cls.NONE:
            return cls(cls.NONE, source=element)
        missing = [name for name in ("domainAbsoluteRelative", "offset", "scale") if name not in child.attrib]
        if missing:
            raise invalid("Timing requires the attributes {}".format(", ".join(missing)), child)
        domain = child.attrib["domainAbsoluteRelative"]
        if domain not in cls.DOMAINS and not domain.startswith("$"):
            raise invalid("Timing domainAbsoluteRelative must be 'absolute' or 'relative', got {!r}".format(
                domain), child)
        return cls(cls.TIMING, domain, child.attrib["offset"], child.attrib["scale"], element)


FOLLOWING_MODES = ("follow", "position")


def following_mode_from_xml(element):
    """Read ``TrajectoryFollowingMode/@followingMode``."""
    if element is None:
        raise invalid("FollowTrajectoryAction requires a TrajectoryFollowingMode")
    mode = element.attrib.get("followingMode")
    if mode not in FOLLOWING_MODES and not (mode or "").startswith("$"):
        raise invalid("followingMode must be 'follow' or 'position', got {!r}".format(mode), element)
    return mode


class Trajectory(object):
    """Semantic trajectory; vertex times and ``closed`` stay lexical until execution resolves them."""

    SHAPES = ("Polyline", "Clothoid", "ClothoidSpline", "Nurbs")

    def __init__(self, name=None, shape=None, control_points=None, knots=None, order=None, closed=False,
                 source=None):
        self.name = name
        self.shape = shape
        self.control_points = list(control_points or [])
        self.knots = list(knots or [])
        self.order = order
        self.closed = closed
        self.source = source

    @classmethod
    def from_xml(cls, element):
        if element is None or element.tag != "Trajectory":
            raise invalid("Trajectory element is required", element)
        shape = element.find("Shape")
        if shape is None or len(shape) != 1:
            raise invalid("Trajectory {!r} requires exactly one shape".format(element.attrib.get("name")), element)
        shape_element = list(shape)[0]
        shape_name = shape_element.tag
        points = []
        knots = []
        order = None
        if shape_name == "Polyline":
            for vertex in shape_element.findall("Vertex"):
                position = vertex.find("Position")
                if position is None:
                    raise invalid("Polyline vertex requires a Position", vertex)
                points.append({"time": vertex.attrib.get("time"),
                               "position": Position.from_xml(position),
                               "source": vertex})
            if len(points) < 2:
                raise invalid("Polyline of trajectory {!r} requires at least two vertices".format(
                    element.attrib.get("name")), shape_element)
        elif shape_name == "Clothoid":
            position = shape_element.find("Position")
            points = [{"position": Position.from_xml(position) if position is not None else None,
                       "attributes": dict(shape_element.attrib),
                       "source": shape_element}]
        elif shape_name == "ClothoidSpline":
            points = [{"attributes": dict(segment.attrib), "source": segment}
                      for segment in shape_element.findall("ClothoidSplineSegment")]
        elif shape_name == "Nurbs":
            for point in shape_element.findall("ControlPoint"):
                position = point.find("Position")
                if position is None:
                    raise invalid("Nurbs control point requires a Position", point)
                points.append({"attributes": dict(point.attrib), "position": Position.from_xml(position),
                               "source": point})
            if len(points) < 2:
                raise invalid("Nurbs requires at least two control points", shape_element)
            knots = [k.attrib.get("value") for k in shape_element.findall("Knot")]
            order = shape_element.attrib.get("order")
        else:
            raise invalid("Unknown trajectory shape {!r}".format(shape_name), shape_element)
        return cls(name=element.attrib.get("name"), shape=shape_name, control_points=points,
                   knots=knots, order=order,
                   closed=element.attrib.get("closed", "false"),
                   source=element)
