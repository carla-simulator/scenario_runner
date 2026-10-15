"""Semantic catalog resolution, kept independent from simulator blueprints."""

from __future__ import absolute_import

import copy

from srunner.openscenario.model.references import CatalogEntry


class CatalogResolver(object):
    """Resolve a catalog reference to an OSC entry without backend conversion."""

    def __init__(self, catalogs=None):
        self.catalogs = catalogs or {}

    def resolve(self, reference, parameter_store=None):
        catalog = self.catalogs.get(reference.catalog_name, {})
        if reference.entry_name not in catalog:
            raise KeyError("Catalog entry '{}:{}' is not defined".format(
                reference.catalog_name, reference.entry_name))
        entry = catalog[reference.entry_name]
        if isinstance(entry, CatalogEntry):
            element = copy.deepcopy(entry.element)
            parameters = dict(entry.parameters)
            catalog_type = entry.catalog_type
        else:
            element = copy.deepcopy(entry)
            parameters = {}
            catalog_type = reference.catalog_name
        resolved_parameters = dict(parameters)
        for name, value in reference.parameter_assignments.items():
            resolved_parameters[name] = parameter_store.resolve(value) if parameter_store is not None else value
        return CatalogEntry(catalog_type, reference.entry_name, element, resolved_parameters, reference.source)
