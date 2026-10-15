"""Mapping view over the runtime's global OpenSCENARIO parameter dictionary."""

from __future__ import absolute_import

from collections.abc import MutableMapping


class GlobalParameterMapping(MutableMapping):
    """Live view of ``CarlaDataProvider``'s global OSC parameters.

    Binding a ParameterStore to this mapping keeps the semantic store and the
    values read by legacy consumers (``ParameterRef``, ``ChangeParameter``,
    ``CheckParameter``) in sync.  CarlaDataProvider is imported on first use so
    the semantic package stays importable without CARLA.
    """

    @staticmethod
    def _parameters():
        # pylint: disable=import-outside-toplevel
        from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
        return CarlaDataProvider._global_osc_parameters  # pylint: disable=protected-access

    def __getitem__(self, name):
        return self._parameters()[name]

    def __setitem__(self, name, value):
        self._parameters()[name] = value

    def __delitem__(self, name):
        del self._parameters()[name]

    def __iter__(self):
        return iter(list(self._parameters()))

    def __len__(self):
        return len(self._parameters())
