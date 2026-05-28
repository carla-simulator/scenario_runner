#!/usr/bin/env python

# Copyright (c) 2026 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Unit tests for `srunner.tools.carla_compat`.

Carla-free: forces version detection via the `SR_CARLA_VERSION` env override
and reloads the module per test so both the UE4 (0.9.x / PhysX) and UE5
(0.10.x / Chaos) code paths are exercised from the same process.
"""

from __future__ import print_function

import importlib
import os
import sys
import unittest

# Make `import srunner.tools.carla_compat` resolvable when this file is run
# directly from `tests/` (the existing OSC2 unit suites do the same).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _reload_carla_compat(version):
    """Reload `srunner.tools.carla_compat` under a forced version."""
    os.environ["SR_CARLA_VERSION"] = version
    sys.modules.pop("srunner.tools.carla_compat", None)
    return importlib.import_module("srunner.tools.carla_compat")


class CarlaCompatVersionDetectionTests(unittest.TestCase):
    """Version probe + IS_UE5 / IS_UE4 flags."""

    def setUp(self):
        self._saved = os.environ.get("SR_CARLA_VERSION")

    def tearDown(self):
        if self._saved is None:
            os.environ.pop("SR_CARLA_VERSION", None)
        else:
            os.environ["SR_CARLA_VERSION"] = self._saved
        sys.modules.pop("srunner.tools.carla_compat", None)

    def test_env_override_resolves_to_version(self):
        from packaging.version import Version
        compat = _reload_carla_compat("0.10.0")
        self.assertEqual(compat.CARLA_VERSION, Version("0.10.0"))

    def test_is_ue5_true_at_threshold(self):
        compat = _reload_carla_compat("0.10.0")
        self.assertTrue(compat.IS_UE5)
        self.assertFalse(compat.IS_UE4)
        self.assertTrue(compat.is_ue5())

    def test_is_ue5_true_above_threshold(self):
        compat = _reload_carla_compat("0.10.1")
        self.assertTrue(compat.IS_UE5)

    def test_is_ue5_false_for_0_9_x(self):
        compat = _reload_carla_compat("0.9.16")
        self.assertFalse(compat.IS_UE5)
        self.assertTrue(compat.IS_UE4)
        self.assertFalse(compat.is_ue5())

    def test_is_ue5_false_for_0_9_14_minimum(self):
        compat = _reload_carla_compat("0.9.14")
        self.assertFalse(compat.IS_UE5)


class CarlaCompatBlueprintAliasTests(unittest.TestCase):
    """`resolve_blueprint_id` translation tables — UE5 alias, UE4 passthrough."""

    def setUp(self):
        self._saved = os.environ.get("SR_CARLA_VERSION")

    def tearDown(self):
        if self._saved is None:
            os.environ.pop("SR_CARLA_VERSION", None)
        else:
            os.environ["SR_CARLA_VERSION"] = self._saved
        sys.modules.pop("srunner.tools.carla_compat", None)

    def test_ue5_renames_lincoln_mkz(self):
        compat = _reload_carla_compat("0.10.0")
        self.assertEqual(
            compat.resolve_blueprint_id("vehicle.lincoln.mkz_2017"),
            "vehicle.lincoln.mkz",
        )

    def test_ue5_renames_audi_tt_with_ue4_prefix(self):
        compat = _reload_carla_compat("0.10.0")
        self.assertEqual(
            compat.resolve_blueprint_id("vehicle.audi.tt"),
            "vehicle.ue4.audi.tt",
        )

    def test_ue5_substitutes_removed_tesla_model3(self):
        compat = _reload_carla_compat("0.10.0")
        # tesla.model3 is gone in 0.10.0 — alias substitutes a documented car.
        self.assertEqual(
            compat.resolve_blueprint_id("vehicle.tesla.model3"),
            "vehicle.lincoln.mkz",
        )

    def test_ue5_substitutes_removed_two_wheelers(self):
        compat = _reload_carla_compat("0.10.0")
        for legacy in (
            "vehicle.diamondback.century",
            "vehicle.bh.crossbike",
            "vehicle.gazelle.omafiets",
            "vehicle.kawasaki.ninja",
        ):
            self.assertEqual(
                compat.resolve_blueprint_id(legacy),
                "vehicle.mini.cooper",
                msg="expected mini.cooper substitute for {!r}".format(legacy),
            )

    def test_ue5_remaps_first_pedestrian(self):
        compat = _reload_carla_compat("0.10.0")
        # 0.10.0 pedestrians start at .0015 — .0001 is gone.
        self.assertEqual(
            compat.resolve_blueprint_id("walker.pedestrian.0001"),
            "walker.pedestrian.0015",
        )

    def test_ue5_passes_unknown_ids_through(self):
        compat = _reload_carla_compat("0.10.0")
        # Wildcard or unknown ids should fall through so the existing
        # filter() path still gets a chance.
        self.assertEqual(
            compat.resolve_blueprint_id("vehicle.*"),
            "vehicle.*",
        )
        self.assertEqual(
            compat.resolve_blueprint_id("vehicle.future_model.x"),
            "vehicle.future_model.x",
        )

    def test_ue5_non_string_inputs_pass_through(self):
        compat = _reload_carla_compat("0.10.0")
        self.assertIsNone(compat.resolve_blueprint_id(None))
        self.assertEqual(compat.resolve_blueprint_id(42), 42)

    def test_ue4_is_a_passthrough(self):
        compat = _reload_carla_compat("0.9.16")
        for legacy in (
            "vehicle.lincoln.mkz_2017",
            "vehicle.tesla.model3",
            "vehicle.audi.tt",
            "walker.pedestrian.0001",
            "vehicle.diamondback.century",
        ):
            self.assertEqual(compat.resolve_blueprint_id(legacy), legacy)


class CarlaCompatCategoryFallbackTests(unittest.TestCase):
    """`actor_blueprint_categories()` — per-category fallback ids per engine."""

    def setUp(self):
        self._saved = os.environ.get("SR_CARLA_VERSION")

    def tearDown(self):
        if self._saved is None:
            os.environ.pop("SR_CARLA_VERSION", None)
        else:
            os.environ["SR_CARLA_VERSION"] = self._saved
        sys.modules.pop("srunner.tools.carla_compat", None)

    def test_ue5_car_fallback_resolves_on_0_10_0(self):
        compat = _reload_carla_compat("0.10.0")
        cats = compat.actor_blueprint_categories()
        self.assertEqual(cats["car"], "vehicle.lincoln.mkz")
        self.assertEqual(cats["pedestrian"], "walker.pedestrian.0015")
        # Two-wheelers absent from 0.10.0 — fall back to a four-wheel car.
        self.assertEqual(cats["bicycle"], "vehicle.mini.cooper")
        self.assertEqual(cats["motorbike"], "vehicle.mini.cooper")

    def test_ue4_car_fallback_keeps_legacy_ids(self):
        compat = _reload_carla_compat("0.9.16")
        cats = compat.actor_blueprint_categories()
        self.assertEqual(cats["car"], "vehicle.tesla.model3")
        self.assertEqual(cats["pedestrian"], "walker.pedestrian.0001")
        self.assertEqual(cats["bicycle"], "vehicle.diamondback.century")
        self.assertEqual(cats["motorbike"], "vehicle.kawasaki.ninja")

    def test_both_engines_expose_the_same_categories(self):
        ue4 = _reload_carla_compat("0.9.16").actor_blueprint_categories()
        ue5 = _reload_carla_compat("0.10.0").actor_blueprint_categories()
        self.assertEqual(set(ue4.keys()), set(ue5.keys()))


if __name__ == "__main__":
    unittest.main()
