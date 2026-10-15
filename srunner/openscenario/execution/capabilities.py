"""Capability queries kept separate from the semantic model."""

from __future__ import absolute_import


class BackendCapabilities(object):
    def __init__(self, capabilities=None):
        self._capabilities = set(capabilities or [])

    def supports(self, capability):
        return capability in self._capabilities

    def add(self, capability):
        self._capabilities.add(capability)

    def as_set(self):
        return set(self._capabilities)
