"""Typed entity conditions that are executed through the semantic condition registry."""

from __future__ import absolute_import

from srunner.openscenario.model.references import invalid

ANGLE_TYPES = ("heading", "pitch", "roll")
COORDINATE_SYSTEMS = ("entity", "lane", "road", "trajectory", "world")


def _required(element, names):
    missing = [name for name in names if name not in element.attrib]
    if missing:
        raise invalid("{} requires the attributes {}".format(element.tag, ", ".join(missing)), element)


def _boolean(element, name):
    value = element.attrib.get(name)
    if value in ("true", "1"):
        return True
    if value in ("false", "0"):
        return False
    if (value or "").startswith("$"):
        return value
    raise invalid("{}@{} must be a boolean, got {!r}".format(element.tag, name, value), element)


class Condition(object):
    def __init__(self, name=None, edge=None, source=None):
        self.name = name
        self.edge = edge
        self.source = source


class RelativeAngleCondition(Condition):
    """Compare the angle of the triggering entity relative to ``entity_ref`` within a tolerance."""

    kind = "RelativeAngleCondition"
    action_type = "RelativeAngleCondition"

    def __init__(self, entity_ref, angle_type, angle, angle_tolerance, coordinate_system="entity",
                 name=None, edge=None, source=None):
        super(RelativeAngleCondition, self).__init__(name=name, edge=edge, source=source)
        self.entity_ref = entity_ref
        self.angle_type = angle_type
        self.angle = angle
        self.angle_tolerance = angle_tolerance
        self.coordinate_system = coordinate_system

    @classmethod
    def from_xml(cls, element, name=None, edge=None):
        _required(element, ("entityRef", "angleType", "angle", "angleTolerance"))
        angle_type = element.attrib["angleType"]
        if angle_type not in ANGLE_TYPES and not angle_type.startswith("$"):
            raise invalid("RelativeAngleCondition angleType must be one of {}, got {!r}".format(
                ", ".join(ANGLE_TYPES), angle_type), element)
        coordinate_system = element.attrib.get("coordinateSystem", "entity")
        if coordinate_system not in COORDINATE_SYSTEMS and not coordinate_system.startswith("$"):
            raise invalid("RelativeAngleCondition coordinateSystem must be one of {}, got {!r}".format(
                ", ".join(COORDINATE_SYSTEMS), coordinate_system), element)
        return cls(element.attrib["entityRef"], angle_type, element.attrib["angle"],
                   element.attrib["angleTolerance"], coordinate_system, name, edge, element)


class RelativeClearanceCondition(Condition):
    """Free-space check around the triggering entity over relative lanes and a longitudinal window."""

    kind = "RelativeClearanceCondition"
    action_type = "RelativeClearanceCondition"

    def __init__(self, opposite_lanes, freespace, distance_forward=None, distance_backward=None,
                 lane_ranges=None, entity_refs=None, name=None, edge=None, source=None):
        super(RelativeClearanceCondition, self).__init__(name=name, edge=edge, source=source)
        self.opposite_lanes = opposite_lanes
        self.freespace = freespace
        self.distance_forward = distance_forward
        self.distance_backward = distance_backward
        self.lane_ranges = list(lane_ranges or [])
        self.entity_refs = list(entity_refs or [])

    @classmethod
    def from_xml(cls, element, name=None, edge=None):
        _required(element, ("oppositeLanes", "freeSpace"))
        lane_ranges = [(lane_range.attrib.get("from"), lane_range.attrib.get("to"))
                       for lane_range in element.findall("RelativeLaneRange")]
        entity_refs = []
        for entity in element.findall("EntityRef"):
            if "entityRef" not in entity.attrib:
                raise invalid("RelativeClearanceCondition EntityRef requires entityRef", entity)
            entity_refs.append(entity.attrib["entityRef"])
        return cls(_boolean(element, "oppositeLanes"), _boolean(element, "freeSpace"),
                   element.attrib.get("distanceForward"), element.attrib.get("distanceBackward"),
                   lane_ranges, entity_refs, name, edge, element)


ENTITY_CONDITIONS = {
    "RelativeAngleCondition": RelativeAngleCondition,
    "RelativeClearanceCondition": RelativeClearanceCondition,
}
