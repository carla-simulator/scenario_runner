#!/usr/bin/env python

# Copyright (c) 2026 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Blueprint alias precedence and LightManager probing, against a fake CARLA world."""

import fnmatch
import os
import random
import sys
import unittest
from unittest import mock

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

class _RealCarlaCase(unittest.TestCase):
    """Imports the real carla stack lazily and restores sys.modules afterwards, so
    other suites that install a fake `carla` are unaffected."""

    @classmethod
    def setUpClass(cls):
        cls._modules_before = set(sys.modules)
        cls._fake_carla = None
        if not hasattr(sys.modules.get("carla"), "Waypoint"):
            cls._fake_carla = sys.modules.pop("carla", None)
        try:
            import carla  # noqa: F401
            from srunner.scenariomanager import carla_data_provider, lights_sim
        except ImportError as exc:
            cls._cleanup()
            raise unittest.SkipTest("carla stack not importable: {}".format(exc))
        cls.provider_module = carla_data_provider
        cls.CarlaDataProvider = carla_data_provider.CarlaDataProvider
        cls.lights_sim = lights_sim

    @classmethod
    def _cleanup(cls):
        for name in set(sys.modules) - cls._modules_before:
            sys.modules.pop(name, None)
        if cls._fake_carla is not None:
            sys.modules["carla"] = cls._fake_carla

    @classmethod
    def tearDownClass(cls):
        cls._cleanup()


class FakeBlueprint:
    def __init__(self, bp_id):
        self.id = bp_id

    def has_attribute(self, name):
        return False


class FakeLibrary:
    def __init__(self, ids):
        self._blueprints = [FakeBlueprint(i) for i in ids]

    def filter(self, pattern):
        return [b for b in self._blueprints if fnmatch.fnmatch(b.id, pattern)]


class BlueprintAliasPrecedenceTests(_RealCarlaCase):

    def setUp(self):
        provider = self.CarlaDataProvider
        self._saved = (provider._blueprint_library, provider._rng)
        provider._rng = random.Random(0)

    def tearDown(self):
        self.CarlaDataProvider._blueprint_library, self.CarlaDataProvider._rng = self._saved

    def _create(self, model, library_ids):
        self.CarlaDataProvider._blueprint_library = FakeLibrary(library_ids)
        with mock.patch.object(self.provider_module, "resolve_blueprint_id",
                               side_effect=lambda m: {"vehicle.tesla.model3": "vehicle.lincoln.mkz"}.get(m, m)):
            return self.CarlaDataProvider.create_blueprint(model, "scenario", None, "car")

    def test_existing_blueprint_beats_alias(self):
        bp = self._create("vehicle.tesla.model3", ["vehicle.tesla.model3", "vehicle.lincoln.mkz"])
        self.assertEqual(bp.id, "vehicle.tesla.model3")

    def test_alias_used_when_blueprint_missing(self):
        bp = self._create("vehicle.tesla.model3", ["vehicle.lincoln.mkz"])
        self.assertEqual(bp.id, "vehicle.lincoln.mkz")


class RouteLightsLightManagerTests(_RealCarlaCase):

    def _build(self, world):
        with mock.patch.object(self.lights_sim.CarlaDataProvider, "get_world", return_value=world):
            return self.lights_sim.RouteLightsBehavior(mock.Mock())

    def test_uses_light_manager_when_exposed(self):
        manager = mock.Mock()
        world = mock.Mock(spec=["get_lightmanager"])
        world.get_lightmanager.return_value = manager
        behavior = self._build(world)
        self.assertIs(behavior._light_manager, manager)
        manager.set_day_night_cycle.assert_called_once_with(False)

    def test_skips_street_lights_when_api_absent(self):
        behavior = self._build(mock.Mock(spec=[]))
        self.assertIsNone(behavior._light_manager)


if __name__ == "__main__":
    unittest.main()
