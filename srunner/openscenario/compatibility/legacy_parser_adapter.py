"""Narrow adapter for unmigrated OpenSCENARIO behavior."""

from __future__ import absolute_import

from srunner.tools.openscenario_parser import OpenScenarioParser


class LegacyParserAdapter(object):
    """Keep legacy helpers available without exposing them to new model code."""

    def __init__(self, parser=None):
        self.parser = parser or OpenScenarioParser

    def convert_position(self, position):
        return self.parser.convert_position_to_transform(position)

    def catalog_entry(self, catalogs, reference):
        return self.parser.get_catalog_entry(catalogs, reference)
