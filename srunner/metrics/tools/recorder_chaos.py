#!/usr/bin/env python

# Copyright (c) 2026 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Chaos-format helpers for the CARLA recorder text dump on UE5 / CARLA 0.10.0+.

The recorder serialization of `Physics Control events` diverges structurally
between the PhysX (0.9.x) and Chaos (0.10.0+) backends:

PhysX (0.9.x)::

     Physics Control events: 1
      Id: 42
       max_rpm = 5000
       ...
       wheel 1 = (tire_friction 3.5 damping_rate 0.25 ...)
       ...
       gear 1 = (ratio 4 down_ratio 0.5 up_ratio 0.65)

Chaos (0.10.0+)::

     Physics Control events: 1
      Id: 42
       max_torque = 550
       use_gear_auto_box = true
       ...
       torque_curve = (0, 500) (5000, 500) ...
       forward_gear_ratios:
        gear 0: ratio 4
        gear 1: ratio 2.5
       reverse_gear_ratios:
        gear 0: ratio 2.943
       wheels:
    wheel #0:
     axle_type:  offset: (0, 0, 0) wheel_radius: 35.5 ... friction_force_multiplier: 3.5 ...

`wheel #N:` headers and their data lines start at column 0/1 (no shared indent
with the rest of the physics block), so the per-version paths cannot share a
walker. This module owns the Chaos walker; PhysX stays inline in the parsers.

Field-name and type reference (citations in `carla-ue5/Docs/python_api.md`):
- `carla.VehiclePhysicsControl` — §carla.VehiclePhysicsControl, lines ~2793-2895.
- `carla.WheelPhysicsControl`   — §carla.WheelPhysicsControl,   lines ~3209-3320.
"""

from __future__ import print_function

import carla


# ---------------------------------------------------------------------------
# carla.VehiclePhysicsControl — known Chaos fields the recorder dumps
# ---------------------------------------------------------------------------

_VEHICLE_SCALAR_FIELDS = frozenset({
    "max_torque", "max_rpm", "idle_rpm", "brake_effect",
    "rev_up_moi", "rev_down_rate",
    "front_rear_split", "gear_change_time", "final_ratio",
    "change_up_rpm", "change_down_rpm", "transmission_efficiency",
    "mass", "drag_coefficient",
    "chassis_width", "chassis_height", "downforce_coefficient",
    "sleep_threshold", "sleep_slope_limit",
})

_VEHICLE_INT_FIELDS = frozenset({
    "differential_type",
})

_VEHICLE_BOOL_FIELDS = frozenset({
    "use_automatic_gears", "use_sweep_wheel_collision",
})

# Recorder dumps the legacy PhysX-flavoured name even on Chaos servers;
# remap to the documented Chaos attribute.
_VEHICLE_BOOL_ALIASES = {
    "use_gear_auto_box": "use_automatic_gears",
}

# Field names the recorder dumps that aren't on Chaos VehiclePhysicsControl.
# `MOI` is a server-internal artefact (Chaos exposes `rev_up_moi`, which is
# a different quantity — do not silently remap).
_VEHICLE_IGNORED = frozenset({
    "MOI",
})


# ---------------------------------------------------------------------------
# carla.WheelPhysicsControl — known Chaos fields the recorder dumps per wheel
# ---------------------------------------------------------------------------

_WHEEL_SCALAR_FIELDS = frozenset({
    "axel_type",
    "max_steer_angle", "wheel_radius", "wheel_width", "wheel_mass",
    "cornering_stiffness", "friction_force_multiplier",
    "side_slip_modifier", "slip_threshold", "skid_threshold",
    "max_wheelspin_rotation",
    "max_brake_torque", "max_handbrake_torque",
    "suspension_force_offset",
    "suspension_max_raise", "suspension_max_drop", "suspension_damping_ratio",
    "wheel_load_ratio", "spring_rate", "spring_preload",
    "suspension_smoothing", "rollbar_scaling",
})

_WHEEL_BOOL_FIELDS = frozenset({
    "affected_by_steering", "affected_by_brake",
    "affected_by_handbrake", "affected_by_engine",
    "abs_enabled", "traction_control_enabled",
})

_WHEEL_VECTOR_FIELDS = frozenset({
    "offset", "suspension_axis",
})

# Recorder ↔ Chaos Python attribute name remap.
# - `max_hand_brake_torque` (recorder) vs `max_handbrake_torque` (Chaos attr).
# - `axle_type` (recorder, correct spelling) vs `axel_type` (Chaos binding,
#   carla-ue5/Docs/python_api.md §carla.WheelPhysicsControl.axel_type — the
#   upstream attribute really is mis-spelled). Drop this alias once upstream
#   renames the binding.
_WHEEL_FIELD_ALIASES = {
    "max_hand_brake_torque": "max_handbrake_torque",
    "axle_type": "axel_type",
}

# Runtime telemetry the recorder appends after the wheel struct; not on
# WheelPhysicsControl itself and not useful for replay.
_WHEEL_IGNORED = frozenset({
    "wheel_index", "location", "old_location", "velocity",
    "external_torque_combine_method", "lateral_slip_graph",
    "sweep_shape", "sweep_type",
})


# ---------------------------------------------------------------------------
# Parsing primitives
# ---------------------------------------------------------------------------


def _parse_vector3d(token_str):
    """Parse a '(x, y, z)' substring into a carla.Vector3D."""
    s = token_str.strip()
    if not (s.startswith("(") and s.endswith(")")):
        raise ValueError("not a vector literal: {!r}".format(token_str))
    inner = s[1:-1]
    parts = [p.strip() for p in inner.split(",")]
    if len(parts) != 3:
        raise ValueError("expected 3 components: {!r}".format(token_str))
    return carla.Vector3D(float(parts[0]), float(parts[1]), float(parts[2]))


def _parse_vector2d_list(s):
    """Parse '(0, 500) (5000, 500) ...' into a list of carla.Vector2D.

    Best-effort: a missing closing paren stops parsing at that point rather
    than raising, so a truncated recorder line just yields the pairs it
    could decode.
    """
    out = []
    i = 0
    while i < len(s):
        if s[i] == "(":
            j = s.find(")", i)
            if j == -1:
                break
            inner = s[i + 1:j]
            parts = [p.strip() for p in inner.split(",")]
            if len(parts) == 2:
                try:
                    out.append(carla.Vector2D(float(parts[0]), float(parts[1])))
                except ValueError:
                    pass
            i = j + 1
        else:
            i += 1
    return out


def _set_if_settable(obj, name, value):
    """Best-effort setattr; tolerate read-only / unknown attributes."""
    try:
        setattr(obj, name, value)
    except (AttributeError, TypeError, ValueError):
        pass


# ---------------------------------------------------------------------------
# Wheel line tokenizer
# ---------------------------------------------------------------------------


def _parse_wheel_line(line):
    """Build a Chaos `WheelPhysicsControl` from a single recorder wheel line.

    The recorder emits each wheel on one line as space-separated `key: value`
    pairs, where values may be scalars, vectors `(x, y, z)`, or empty (when
    two keys are adjacent: `axle_type:  offset: ...`). Unknown keys are
    skipped — the format is not stable across upstream releases.
    """
    wheel = carla.WheelPhysicsControl()
    tokens = line.split()

    cur_key = None
    cur_vals = []

    def flush():
        if cur_key is None:
            return
        _apply_wheel_field(wheel, cur_key, cur_vals)

    for tok in tokens:
        if tok.endswith(":") and not tok.startswith("("):
            flush()
            cur_key = tok[:-1]
            cur_vals = []
        else:
            cur_vals.append(tok)
    flush()
    return wheel


def _apply_wheel_field(wheel, key, val_tokens):
    if key in _WHEEL_FIELD_ALIASES:
        key = _WHEEL_FIELD_ALIASES[key]
    if key in _WHEEL_IGNORED:
        return
    if not val_tokens:
        return

    if key in _WHEEL_VECTOR_FIELDS:
        try:
            _set_if_settable(wheel, key, _parse_vector3d(" ".join(val_tokens)))
        except ValueError:
            pass
        return

    if key in _WHEEL_BOOL_FIELDS:
        try:
            _set_if_settable(wheel, key, bool(int(val_tokens[0])))
        except ValueError:
            pass
        return

    if key in _WHEEL_SCALAR_FIELDS:
        try:
            _set_if_settable(wheel, key, float(val_tokens[0]))
        except ValueError:
            pass
        return

    # Unknown field — skip silently. The recorder may emit fields the local
    # Python binding doesn't expose (or vice versa); robust parsing tolerates
    # the gap rather than crashing the metrics tool.


# ---------------------------------------------------------------------------
# Vehicle scalar line ('   key = value')
# ---------------------------------------------------------------------------


def _apply_vehicle_scalar(physics_control, name, value_str):
    if name in _VEHICLE_IGNORED:
        return

    if name == "center_of_mass" and value_str.startswith("("):
        try:
            physics_control.center_of_mass = _parse_vector3d(value_str)
        except (ValueError, AttributeError):
            pass
        return

    if name in ("torque_curve", "steering_curve"):
        try:
            _set_if_settable(physics_control, name, _parse_vector2d_list(value_str))
        except ValueError:
            pass
        return

    if not value_str:
        # Recorder may emit fields with empty values (`differential_type = `).
        # Guard before the bool branches so an empty bool isn't silently
        # coerced to False.
        return

    if name in _VEHICLE_BOOL_ALIASES:
        _set_if_settable(physics_control,
                         _VEHICLE_BOOL_ALIASES[name],
                         value_str == "true")
        return

    if name in _VEHICLE_BOOL_FIELDS:
        _set_if_settable(physics_control, name, value_str == "true")
        return

    if name in _VEHICLE_INT_FIELDS:
        try:
            _set_if_settable(physics_control, name, int(float(value_str)))
        except ValueError:
            pass
        return

    if name in _VEHICLE_SCALAR_FIELDS:
        try:
            _set_if_settable(physics_control, name, float(value_str))
        except ValueError:
            pass
        return


# ---------------------------------------------------------------------------
# Shared section-skip helper for the new 0.10.0 recorder sections that the
# 0.9.x text-walkers have no schema for. The pattern is identical in both
# `MetricsParser` and `Osc2TraceParser`: header at one-space indent, body
# rows at two-space indent.
# ---------------------------------------------------------------------------


def skip_indented_section(parser, header_prefix):
    """If `parser.frame_row` starts with `header_prefix`, consume it and any
    rows indented with two leading spaces. No-op otherwise."""
    if not parser.frame_row.startswith(header_prefix):
        return
    parser.next_row()
    while parser.frame_row.startswith("  "):
        parser.next_row()


# ---------------------------------------------------------------------------
# Block walker — entry point used by metrics_parser / osc2_trace_parser
# ---------------------------------------------------------------------------


def _is_block_terminator(row):
    """Return True if `row` ends the current Physics Control events block."""
    if not row:
        return True
    if row.startswith("Frame "):
        return True
    # New top-level recorder section: ` <Capital>...`.
    if row.startswith(" ") and len(row) > 1 and row[1].isupper():
        return True
    return False


def parse_chaos_physics_block(parser, physics_dest):
    """Parse a UE5 / Chaos Physics Control events block.

    `parser` must expose the same attributes the in-file parsers do
    (`frame_row`, `get_row_elements(indent, sep)`, `next_row()`). On entry,
    `parser.frame_row` points at the first line AFTER ` Physics Control
    events: N` (i.e. the first `  Id: N` row). On exit, `parser.frame_row`
    points at the row that terminated the block (next section or `Frame`).

    Populates `physics_dest` (the events.physics_control dict) in place.
    """
    while not _is_block_terminator(parser.frame_row):
        row = parser.frame_row
        if row.startswith("  Id:"):
            _parse_chaos_actor(parser, physics_dest)
        else:
            # Stray / unexpected row — advance to avoid an infinite loop.
            parser.next_row()


def _parse_chaos_actor(parser, physics_dest):
    elements = parser.get_row_elements(2, " ")  # 'Id:' 'N'
    actor_id = int(elements[1])
    physics_control = carla.VehiclePhysicsControl()
    forward_gear_ratios = []
    reverse_gear_ratios = []
    wheels = []
    parser.next_row()

    while True:
        row = parser.frame_row
        if _is_block_terminator(row) or row.startswith("  Id:"):
            break

        if row.startswith("   forward_gear_ratios:"):
            parser.next_row()
            while parser.frame_row.startswith("    gear "):
                toks = parser.frame_row.split()
                if len(toks) >= 4 and toks[0] == "gear" and toks[2] == "ratio":
                    forward_gear_ratios.append(float(toks[3]))
                parser.next_row()
            continue

        if row.startswith("   reverse_gear_ratios:"):
            parser.next_row()
            while parser.frame_row.startswith("    gear "):
                toks = parser.frame_row.split()
                if len(toks) >= 4 and toks[0] == "gear" and toks[2] == "ratio":
                    reverse_gear_ratios.append(float(toks[3]))
                parser.next_row()
            continue

        if row.startswith("   wheels:"):
            parser.next_row()
            while parser.frame_row.startswith("wheel #"):
                parser.next_row()
                if parser.frame_row.startswith(" ") and not parser.frame_row.startswith("  "):
                    wheels.append(_parse_wheel_line(parser.frame_row))
                    parser.next_row()
            continue

        if row.startswith("   ") and " = " in row:
            name, _, value = row.lstrip().partition(" = ")
            _apply_vehicle_scalar(physics_control, name.strip(), value.strip())
            parser.next_row()
            continue

        # Unknown indented row — skip to avoid getting stuck.
        parser.next_row()

    _set_if_settable(physics_control, "forward_gear_ratios", forward_gear_ratios)
    _set_if_settable(physics_control, "reverse_gear_ratios", reverse_gear_ratios)
    _set_if_settable(physics_control, "wheels", wheels)
    physics_dest[actor_id] = physics_control
