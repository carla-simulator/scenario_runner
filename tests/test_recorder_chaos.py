#!/usr/bin/env python

# Copyright (c) 2026 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Unit tests for `srunner.metrics.tools.recorder_chaos` (the UE5 / Chaos
recorder Physics Control walker) and the new UE5 section-skip handlers
added to `MetricsParser` / `Osc2TraceParser`.

Carla-free: installs a minimal stub `carla` module into `sys.modules`
before importing the targets, so the suite runs in any Python env with
no CARLA wheel installed.
"""

from __future__ import print_function

import os
import sys
import types
import unittest


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_FIXTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


# ---------------------------------------------------------------------------
# Fake `carla` module — covers everything the metrics parsers reach for.
# ---------------------------------------------------------------------------

def _install_fake_carla():
    """Install a stub `carla` module if (and only if) one isn't already loaded.

    Real `carla` wheels expose strict C++ bindings; the fake stays permissive
    so unit tests aren't held hostage to upstream attribute-set semantics.
    """
    if "carla" in sys.modules and hasattr(sys.modules["carla"], "VehiclePhysicsControl"):
        return

    fake = types.ModuleType("carla")

    class _Bag(object):
        """Permissive object that accepts kwargs and arbitrary setattrs."""

        def __init__(self, *args, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class Vector3D(object):
        def __init__(self, x=0.0, y=0.0, z=0.0):
            self.x = float(x)
            self.y = float(y)
            self.z = float(z)

        def __sub__(self, other):
            return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)

        def __truediv__(self, scalar):
            return Vector3D(self.x / scalar, self.y / scalar, self.z / scalar)

    class Vector2D(object):
        def __init__(self, x=0.0, y=0.0):
            self.x = float(x)
            self.y = float(y)

    class Location(Vector3D):
        pass

    class Rotation(object):
        def __init__(self, pitch=0.0, yaw=0.0, roll=0.0):
            self.pitch = float(pitch)
            self.yaw = float(yaw)
            self.roll = float(roll)

    class Transform(object):
        def __init__(self, location=None, rotation=None):
            self.location = location or Location()
            self.rotation = rotation or Rotation()

    class BoundingBox(object):
        def __init__(self, location=None, extent=None):
            self.location = location or Location()
            self.extent = extent or Vector3D()

    class VehicleControl(_Bag):
        pass

    class Color(object):
        def __init__(self, r=0, g=0, b=0):
            self.r, self.g, self.b = r, g, b

    class _Enum(object):
        def __init__(self, name):
            self.name = name

        def __repr__(self):
            return "<carla enum {}>".format(self.name)

    class VehicleLightState(object):
        NONE = _Enum("None")
        Position = _Enum("Position")
        LowBeam = _Enum("LowBeam")
        HighBeam = _Enum("HighBeam")
        Brake = _Enum("Brake")
        RightBlinker = _Enum("RightBlinker")
        LeftBlinker = _Enum("LeftBlinker")
        Reverse = _Enum("Reverse")
        Fog = _Enum("Fog")
        Interior = _Enum("Interior")
        Special1 = _Enum("Special1")
        Special2 = _Enum("Special2")

    class TrafficLightState(object):
        Red = _Enum("Red")
        Yellow = _Enum("Yellow")
        Green = _Enum("Green")
        Off = _Enum("Off")
        Unknown = _Enum("Unknown")

    class LightGroup(object):
        NONE = _Enum("LightGroup.NONE")

    class LightState(_Bag):
        pass

    class VehiclePhysicsControl(_Bag):
        pass

    class WheelPhysicsControl(_Bag):
        pass

    class GearPhysicsControl(_Bag):
        pass

    fake.Vector3D = Vector3D
    fake.Vector2D = Vector2D
    fake.Location = Location
    fake.Rotation = Rotation
    fake.Transform = Transform
    fake.BoundingBox = BoundingBox
    fake.VehicleControl = VehicleControl
    fake.Color = Color
    fake.VehicleLightState = VehicleLightState
    fake.TrafficLightState = TrafficLightState
    fake.LightGroup = LightGroup
    fake.LightState = LightState
    fake.VehiclePhysicsControl = VehiclePhysicsControl
    fake.WheelPhysicsControl = WheelPhysicsControl
    fake.GearPhysicsControl = GearPhysicsControl

    sys.modules["carla"] = fake


_install_fake_carla()

# Force IS_UE5 before importing modules that snapshot it at import time.
os.environ["SR_CARLA_VERSION"] = "0.10.0"
sys.modules.pop("srunner.tools.carla_compat", None)
sys.modules.pop("srunner.metrics.tools.recorder_chaos", None)
sys.modules.pop("srunner.metrics.tools.metrics_parser", None)
sys.modules.pop("srunner.metrics.tools.osc2_trace_parser", None)

from srunner.metrics.tools import recorder_chaos
from srunner.metrics.tools.metrics_parser import MetricsParser
from srunner.metrics.tools.osc2_trace_parser import Osc2TraceParser


# ---------------------------------------------------------------------------
# A tiny parser-like object so `parse_chaos_physics_block` can be exercised
# without dragging in the full MetricsParser / recorder framing.
# ---------------------------------------------------------------------------

class _MockParser(object):
    def __init__(self, rows):
        self._rows = list(rows)
        self._i = 0
        self.frame_row = self._rows[0] if self._rows else ""

    def next_row(self):
        self._i += 1
        if self._i < len(self._rows):
            self.frame_row = self._rows[self._i]
        else:
            self.frame_row = ""

    def get_row_elements(self, indent_num, split_string):
        return self.frame_row[indent_num:].split(split_string)


# ---------------------------------------------------------------------------
# Wheel-line tokenizer
# ---------------------------------------------------------------------------


class ChaosWheelTokenizerTests(unittest.TestCase):
    """`_parse_wheel_line` — Chaos wheel data row tokenizer."""

    SAMPLE = (
        " axle_type:  offset: (0.000000, 0.000000, 0.000000) wheel_radius: 35.5 "
        "wheel_width: 23 wheel_mass: 20 cornering_stiffness: 1000 "
        "friction_force_multiplier: 3.5 side_slip_modifier: 1 slip_threshold: 20 "
        "skid_threshold: 20 max_steer_angle: 70 affected_by_steering: 1 "
        "affected_by_brake: 1 affected_by_handbrake: 1 affected_by_engine: 1 "
        "abs_enabled: 1 traction_control_enabled: 1 max_wheelspin_rotation: 30 "
        "external_torque_combine_method:   lateral_slip_graph: [] "
        "suspension_axis: (0.000000, 0.000000, -1.000000) "
        "suspension_force_offset: (0.000000, 0.000000, 0.000000) "
        "suspension_max_raise: 8 suspension_max_drop: 10 suspension_damping_ratio: 1 "
        "wheel_load_ratio: 0.5 spring_rate: 250 spring_preload: 50 "
        "suspension_smoothing: 4 rollbar_scaling: 0.3 sweep_shape:  sweep_type:  "
        "max_brake_torque: 1000 max_hand_brake_torque: 2000 wheel_index: 0 "
        "location: (0.000000, 0.000000, 0.000000) "
        "old_location: (0.000000, 0.000000, 0.000000) "
        "velocity: (0.000000, 0.000000, 0.000000)"
    )

    def test_parses_friction_force_multiplier(self):
        w = recorder_chaos._parse_wheel_line(self.SAMPLE)
        self.assertEqual(w.friction_force_multiplier, 3.5)

    def test_parses_wheel_geometry_scalars(self):
        w = recorder_chaos._parse_wheel_line(self.SAMPLE)
        self.assertEqual(w.wheel_radius, 35.5)
        self.assertEqual(w.wheel_width, 23.0)
        self.assertEqual(w.wheel_mass, 20.0)
        self.assertEqual(w.max_steer_angle, 70.0)
        self.assertEqual(w.cornering_stiffness, 1000.0)

    def test_parses_brake_torques(self):
        w = recorder_chaos._parse_wheel_line(self.SAMPLE)
        self.assertEqual(w.max_brake_torque, 1000.0)

    def test_remaps_recorder_max_hand_brake_torque_to_chaos_attr(self):
        w = recorder_chaos._parse_wheel_line(self.SAMPLE)
        # Recorder dumps `max_hand_brake_torque` (extra underscore); the
        # documented Chaos attribute is `max_handbrake_torque`.
        self.assertEqual(w.max_handbrake_torque, 2000.0)

    def test_parses_boolean_flags(self):
        w = recorder_chaos._parse_wheel_line(self.SAMPLE)
        self.assertTrue(w.affected_by_steering)
        self.assertTrue(w.affected_by_brake)
        self.assertTrue(w.affected_by_handbrake)
        self.assertTrue(w.affected_by_engine)
        self.assertTrue(w.abs_enabled)
        self.assertTrue(w.traction_control_enabled)

    def test_parses_vector_field(self):
        w = recorder_chaos._parse_wheel_line(self.SAMPLE)
        self.assertEqual(w.suspension_axis.x, 0.0)
        self.assertEqual(w.suspension_axis.y, 0.0)
        self.assertEqual(w.suspension_axis.z, -1.0)

    def test_handles_empty_value_between_keys(self):
        # `axle_type:` with no value, immediately followed by `offset: ...`
        # — empty values must be skipped, not crash.
        line = " axle_type:  offset: (0.0, 0.0, 0.0) wheel_radius: 35.5"
        w = recorder_chaos._parse_wheel_line(line)
        self.assertEqual(w.wheel_radius, 35.5)
        self.assertFalse(hasattr(w, "axle_type"))

    def test_skips_runtime_only_fields(self):
        # `wheel_index`, `location`, `old_location`, `velocity` aren't on
        # WheelPhysicsControl — they should be silently dropped.
        w = recorder_chaos._parse_wheel_line(self.SAMPLE)
        for attr in ("wheel_index", "location", "old_location", "velocity",
                     "external_torque_combine_method", "lateral_slip_graph",
                     "sweep_shape", "sweep_type"):
            self.assertFalse(hasattr(w, attr),
                             "runtime-only field {!r} should not be set".format(attr))

    def test_skips_unknown_fields(self):
        # An unknown field shouldn't crash and shouldn't be applied.
        line = " future_field: 7.5 wheel_radius: 35.5"
        w = recorder_chaos._parse_wheel_line(line)
        self.assertEqual(w.wheel_radius, 35.5)
        self.assertFalse(hasattr(w, "future_field"))


# ---------------------------------------------------------------------------
# Full Physics Control block walker (driven by _MockParser)
# ---------------------------------------------------------------------------


class ChaosPhysicsBlockTests(unittest.TestCase):
    """`parse_chaos_physics_block` — full Physics Control events block walker."""

    BLOCK_HEAD = "  Id: 49"
    SCALAR_ROWS = [
        "   max_torque = 550",
        "   max_rpm = 6500",
        "   MOI = 5",
        "   rev_down_rate = 600",
        "   differential_type = ",
        "   front_rear_split = 0.5",
        "   use_gear_auto_box = true",
        "   gear_change_time = 0.1",
        "   final_ratio = 3.21",
        "   change_up_rpm = 4000",
        "   change_down_rpm = 1500",
        "   transmission_efficiency = 0.9",
        "   mass = 1696",
        "   drag_coefficient = 0.3",
        "   center_of_mass = (0.15, 0, 0.35)",
        "   torque_curve = (0, 500) (5000, 500) (0, 500) (1000, 347)",
        "   steering_curve = (0, 1) (10, 0.5) (20, 0.9)",
        "   forward_gear_ratios:",
        "    gear 0: ratio 4",
        "    gear 1: ratio 2.5",
        "    gear 2: ratio 1.912",
        "   reverse_gear_ratios:",
        "    gear 0: ratio 2.943",
        "   wheels:",
        "wheel #0:",
        " axle_type:  wheel_radius: 35.5 friction_force_multiplier: 3.5 "
        "max_brake_torque: 1000 max_hand_brake_torque: 2000 wheel_index: 0",
        "wheel #1:",
        " axle_type:  wheel_radius: 35.5 friction_force_multiplier: 3.5 "
        "max_brake_torque: 1000 max_hand_brake_torque: 2000 wheel_index: 1",
        # Terminator — a new top-level section.
        " Traffic Light time events: 0",
    ]

    def _parse(self):
        rows = [self.BLOCK_HEAD] + self.SCALAR_ROWS
        parser = _MockParser(rows)
        dest = {}
        recorder_chaos.parse_chaos_physics_block(parser, dest)
        return dest, parser

    def test_actor_key_present(self):
        dest, _ = self._parse()
        self.assertIn(49, dest)

    def test_scalar_fields_populated(self):
        dest, _ = self._parse()
        pc = dest[49]
        self.assertEqual(pc.max_torque, 550.0)
        self.assertEqual(pc.max_rpm, 6500.0)
        self.assertEqual(pc.mass, 1696.0)
        self.assertEqual(pc.drag_coefficient, 0.3)
        self.assertEqual(pc.gear_change_time, 0.1)
        self.assertEqual(pc.final_ratio, 3.21)
        self.assertEqual(pc.change_up_rpm, 4000.0)
        self.assertEqual(pc.change_down_rpm, 1500.0)
        self.assertEqual(pc.front_rear_split, 0.5)

    def test_use_gear_auto_box_remapped_to_chaos_attr(self):
        dest, _ = self._parse()
        pc = dest[49]
        # Recorder field is `use_gear_auto_box`; Chaos attr is
        # `use_automatic_gears`.
        self.assertTrue(pc.use_automatic_gears)

    def test_differential_type_empty_value_does_not_crash(self):
        # Empty `differential_type = ` line must be skipped, not raise.
        dest, _ = self._parse()
        pc = dest[49]
        self.assertFalse(hasattr(pc, "differential_type"))

    def test_moi_recorder_field_is_not_aliased_to_rev_up_moi(self):
        # `MOI` is a server-internal artifact; Chaos `rev_up_moi` is a
        # different quantity. No silent remap.
        dest, _ = self._parse()
        pc = dest[49]
        self.assertFalse(hasattr(pc, "rev_up_moi"))
        self.assertFalse(hasattr(pc, "moi"))

    def test_center_of_mass_parsed_as_vector(self):
        dest, _ = self._parse()
        pc = dest[49]
        self.assertEqual(pc.center_of_mass.x, 0.15)
        self.assertEqual(pc.center_of_mass.z, 0.35)

    def test_torque_and_steering_curves_parsed_as_vector2d_lists(self):
        dest, _ = self._parse()
        pc = dest[49]
        self.assertEqual(len(pc.torque_curve), 4)
        self.assertEqual(pc.torque_curve[0].x, 0.0)
        self.assertEqual(pc.torque_curve[0].y, 500.0)
        self.assertEqual(len(pc.steering_curve), 3)

    def test_forward_and_reverse_gear_ratios_parsed_as_lists(self):
        dest, _ = self._parse()
        pc = dest[49]
        self.assertEqual(pc.forward_gear_ratios, [4.0, 2.5, 1.912])
        self.assertEqual(pc.reverse_gear_ratios, [2.943])

    def test_wheels_list_populated(self):
        dest, _ = self._parse()
        pc = dest[49]
        self.assertEqual(len(pc.wheels), 2)
        self.assertEqual(pc.wheels[0].wheel_radius, 35.5)
        self.assertEqual(pc.wheels[0].friction_force_multiplier, 3.5)
        # max_hand_brake_torque (recorder) → max_handbrake_torque (Chaos attr)
        self.assertEqual(pc.wheels[0].max_handbrake_torque, 2000.0)

    def test_walker_exits_at_block_terminator(self):
        # After the loop, frame_row should sit at the next top-level section.
        _, parser = self._parse()
        self.assertTrue(parser.frame_row.startswith(" Traffic Light time events"))

    def test_empty_block_does_not_crash(self):
        # `Physics Control events: 0` — caller advanced past the header,
        # so the walker sees the next section straight away and exits.
        parser = _MockParser([" Traffic Light time events: 0"])
        dest = {}
        recorder_chaos.parse_chaos_physics_block(parser, dest)
        self.assertEqual(dest, {})


# ---------------------------------------------------------------------------
# End-to-end MetricsParser / Osc2TraceParser on the captured 0.10.0 dump.
# Confirms the section-skip handlers (Vehicle door animations / Weathers /
# Walkers Bones) plus the IS_UE5 branch wire up correctly.
# ---------------------------------------------------------------------------


class MetricsParserUE5FixtureTests(unittest.TestCase):
    """Run the full parser on a real 0.10.0 recorder dump (Town10HD_Opt)."""

    @classmethod
    def setUpClass(cls):
        with open(os.path.join(_FIXTURE_DIR, "recorder_ue5_town10.txt")) as f:
            cls.text = f.read()

    def test_metrics_parser_reaches_physics_control(self):
        sim, actors, frames = MetricsParser(self.text).parse_recorder_info()
        self.assertEqual(sim["map"], "Town10HD_Opt")
        self.assertGreater(len(frames), 0)
        # The Physics Control block lives in Frame 1 (index 0).
        self.assertIn(49, frames[0]["events"]["physics_control"])

    def test_metrics_parser_populates_chaos_scalars(self):
        _, _, frames = MetricsParser(self.text).parse_recorder_info()
        pc = frames[0]["events"]["physics_control"][49]
        self.assertEqual(pc.max_torque, 550.0)
        self.assertEqual(pc.max_rpm, 6500.0)
        self.assertEqual(pc.mass, 1696.0)
        self.assertTrue(pc.use_automatic_gears)
        self.assertEqual(len(pc.wheels), 4)
        # Each wheel carries the Chaos friction multiplier.
        for w in pc.wheels:
            self.assertEqual(w.friction_force_multiplier, 3.5)

    def test_metrics_parser_walks_past_vehicle_door_animations(self):
        # Without the UE5 section-skip handler the walker would stall on
        # `Vehicle door animations` and never reach Positions; positions
        # being non-empty proves the handler fired.
        _, _, frames = MetricsParser(self.text).parse_recorder_info()
        self.assertGreater(len(frames[0]["actors"]), 0,
                           "section-skip handler missed `Vehicle door animations`")

    def test_metrics_parser_still_reports_vehicle_controls(self):
        # Vehicle animations is downstream of the new sections — its data
        # ending up in frame_state['actors'][id]['control'] confirms the
        # walker traversed past every skip-handler in order.
        _, _, frames = MetricsParser(self.text).parse_recorder_info()
        any_control = any(
            "control" in d for fr in frames for d in fr["actors"].values()
        )
        self.assertTrue(any_control)

    def test_osc2_trace_parser_matches_metrics_parser_shape(self):
        sim_a, _, frames_a = MetricsParser(self.text).parse_recorder_info()
        sim_b, _, frames_b = Osc2TraceParser(self.text).parse_recorder_info()
        self.assertEqual(sim_a["map"], sim_b["map"])
        self.assertEqual(len(frames_a), len(frames_b))
        # Same actor seen in both physics_control dicts.
        self.assertEqual(
            set(frames_a[0]["events"]["physics_control"].keys()),
            set(frames_b[0]["events"]["physics_control"].keys()),
        )


if __name__ == "__main__":
    unittest.main()
