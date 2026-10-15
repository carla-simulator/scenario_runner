"""Explicit schema-validation boundary for OpenSCENARIO documents."""

from __future__ import absolute_import

import os

from srunner.openscenario.parsing.loader import load_document
from srunner.openscenario.validation.diagnostics import Diagnostic, ValidationError


class SchemaValidator(object):
    def __init__(self, schema_path=None):
        self.schema_path = schema_path or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "OpenSCENARIO.xsd")

    def validate(self, source, raise_on_error=True):
        try:
            import xmlschema
            xmlschema.XMLSchema(self.schema_path).validate(source.root if hasattr(source, "root") else source)
        except Exception as error:
            diagnostic = Diagnostic("schema_error", str(error), source=getattr(source, "source", None))
            if raise_on_error:
                raise ValidationError([diagnostic])
            return [diagnostic]
        return []

    def validate_document(self, source, raise_on_error=True):
        document = source if hasattr(source, "root") else load_document(source)
        return self.validate(document, raise_on_error=raise_on_error)
