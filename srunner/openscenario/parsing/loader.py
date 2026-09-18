"""Version-aware OpenSCENARIO XML loading, independent of CARLA execution."""

from __future__ import absolute_import

import os
import xml.etree.ElementTree as ET

from srunner.openscenario.model.document import OpenScenarioDocument


class OpenScenarioVersion(object):
    def __init__(self, major, minor, patch=None):
        self.major = int(major)
        self.minor = int(minor)
        self.patch = None if patch in (None, "") else int(patch)

    def __str__(self):
        if self.patch is None:
            return "{}.{}".format(self.major, self.minor)
        return "{}.{}.{}".format(self.major, self.minor, self.patch)

    def __repr__(self):
        return "OpenScenarioVersion({})".format(str(self))

    def __eq__(self, other):
        return isinstance(other, OpenScenarioVersion) and (
            self.major, self.minor, self.patch) == (other.major, other.minor, other.patch)


class OpenScenarioLoader(object):
    """Load XML and retain the declared standard version."""

    def load(self, source):
        root, source_name = self._read_root(source)
        if root.tag != "OpenSCENARIO":
            raise ValueError("Expected OpenSCENARIO root element, got {!r}".format(root.tag))
        header = root.find("FileHeader")
        if header is None:
            raise ValueError("OpenSCENARIO document is missing FileHeader")
        if "revMajor" not in header.attrib or "revMinor" not in header.attrib:
            raise ValueError("FileHeader must declare revMajor and revMinor")
        version = OpenScenarioVersion(
            header.attrib["revMajor"], header.attrib["revMinor"], header.attrib.get("revPatch"))
        return OpenScenarioDocument(root, version, source_name)

    @staticmethod
    def _read_root(source):
        if isinstance(source, ET.ElementTree):
            return source.getroot(), None
        if isinstance(source, ET.Element):
            return source, None
        if hasattr(source, "read"):
            return ET.parse(source).getroot(), getattr(source, "name", None)
        if not isinstance(source, str) and not isinstance(source, bytes):
            raise TypeError("source must be an XML path, file object, ElementTree, or Element")
        return ET.parse(source).getroot(), os.path.abspath(source)


def load_document(source):
    return OpenScenarioLoader().load(source)
