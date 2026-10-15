"""OpenSCENARIO semantic and execution boundaries.

The legacy parser remains available in :mod:`srunner.tools.openscenario_parser`.
This package provides the incremental architecture used by new feature paths.
"""

from srunner.openscenario.parsing.loader import OpenScenarioLoader, load_document

__all__ = ["OpenScenarioLoader", "load_document"]
