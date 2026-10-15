"""Version-aware OpenSCENARIO XML loading, independent of CARLA execution."""

from __future__ import absolute_import

import os
import xml.etree.ElementTree as ET

from srunner.openscenario.model.document import OpenScenarioDocument
from srunner.openscenario.validation.diagnostics import Diagnostic, ValidationError


def _malformed(message, source=None):
    return ValidationError([Diagnostic("schema_error", message, source=source,
                                       code=Diagnostic.MALFORMED_DOCUMENT)])


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
    """Load XML and retain the declared standard version.

    Malformed documents raise ValidationError (a ValueError) with a MALFORMED_DOCUMENT diagnostic.
    """

    def load(self, source):
        root, source_name = self._read_root(source)
        if root.tag != "OpenSCENARIO":
            raise _malformed("Expected OpenSCENARIO root element, got {!r}".format(root.tag), root)
        header = root.find("FileHeader")
        if header is None:
            raise _malformed("OpenSCENARIO document is missing FileHeader", root)
        if "revMajor" not in header.attrib or "revMinor" not in header.attrib:
            raise _malformed("FileHeader must declare revMajor and revMinor", header)
        try:
            version = OpenScenarioVersion(
                header.attrib["revMajor"], header.attrib["revMinor"], header.attrib.get("revPatch"))
        except ValueError as error:
            raise _malformed("Invalid FileHeader revision: {}".format(error), header) from error
        return OpenScenarioDocument(root, version, source_name)

    @staticmethod
    def _read_root(source):
        # Duck-typed: xmlschema 1.0.x re-imports xml.etree.ElementTree, so isinstance
        # checks against this module's classes fail for trees parsed before that import.
        if hasattr(source, "getroot"):
            return source.getroot(), None
        if hasattr(source, "tag") and hasattr(source, "attrib"):
            return source, None
        if hasattr(source, "read"):
            return OpenScenarioLoader._parse(source), getattr(source, "name", None)
        if not isinstance(source, str) and not isinstance(source, bytes):
            raise TypeError("source must be an XML path, file object, ElementTree, or Element")
        return OpenScenarioLoader._parse(source), os.path.abspath(source)

    @staticmethod
    def _parse(source):
        try:
            return ET.parse(source).getroot()
        except ET.ParseError as error:
            raise _malformed("XML parse error: {}".format(error)) from error


def load_document(source):
    return OpenScenarioLoader().load(source)
