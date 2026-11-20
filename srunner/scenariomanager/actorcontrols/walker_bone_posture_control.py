#!/usr/bin/env python

# Copyright (c) 2026 Mark Colley
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Skeletal posture actor controller for CARLA pedestrians (`carla.WalkerBoneControlIn`).

Supports smartphone head-down gaze (`ON_PHONE`: `crl_neck__C` +32 deg, `crl_foreArm__R` -65 deg),
blind/low-vision 1.2 Hz white-cane sweep, older-adult kyphotic thoracic spine (+8 deg) with
rate-limited neck yaw, and seated wheelchair pelvis postures inside CARLA ScenarioRunner.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

try:
    import carla
except ImportError:  # pragma: no cover
    carla = None  # type: ignore[assignment]

from srunner.scenariomanager.actorcontrols.basic_control import BasicControl

GAZE_ON_PHONE: int = 0
GAZE_ON_ROAD: int = 1


@dataclass(frozen=True)
class BoneTransformEuler:
    """Euler bone override (degrees) & local translation (meters) for `carla.WalkerBoneControlIn`."""

    bone_name: str
    pitch_deg: float = 0.0
    yaw_deg: float = 0.0
    roll_deg: float = 0.0
    tx_m: float = 0.0
    ty_m: float = 0.0
    tz_m: float = 0.0

    def to_dict(self) -> dict[str, float | str]:
        return {
            "bone_name": self.bone_name,
            "pitch_deg": self.pitch_deg,
            "yaw_deg": self.yaw_deg,
            "roll_deg": self.roll_deg,
            "tx_m": self.tx_m,
            "ty_m": self.ty_m,
            "tz_m": self.tz_m,
        }


def build_walker_bone_overrides(
    archetype: str = "adult_baseline",
    gaze_state: int | str = GAZE_ON_ROAD,
    sim_time_s: float = 0.0,
    active_gaze_yaw_deg: float = 0.0,
    prev_neck_yaw_deg: float | None = None,
    dt_s: float = 0.05,
    surface_pitch_deg: float = 0.0,
    surface_roll_deg: float = 0.0,
) -> list[BoneTransformEuler]:
    """
    Construct skeletal bone transforms across vulnerable road user (VRU) cohorts:
      - `ON_PHONE`: `crl_neck__C` pitch `+32 deg`, `crl_Head__C` pitch `+10 deg`,
        `crl_arm__R` pitch `-35 deg` / yaw `-10 deg`, `crl_foreArm__R` pitch `-65 deg` / yaw `-20 deg`.
      - `blind_low_vision`: `1.2 Hz` sinusoidal white-cane wrist sweep (`+/- 22 deg` on `crl_hand__R`).
      - `older_adult` / `older_adult_rollator`: kyphotic spine `crl_spine01__C` pitch `+8 deg`
        + bounded neck yaw (`<= 90 deg/s`, `+/- 55 deg`).
      - `wheelchair_manual` / `powered_scooter`: seated pelvis (`crl_hips__C` `tz = -0.42 m`,
        bilateral hip flexion `+88 deg`, knee flexion `-88 deg`).
    """
    is_on_phone = (gaze_state == GAZE_ON_PHONE) or (
        isinstance(gaze_state, str) and gaze_state.upper() == "ON_PHONE"
    )
    bones: list[BoneTransformEuler] = []

    if archetype in ("wheelchair_manual", "powered_scooter"):
        bones.extend(
            [
                BoneTransformEuler(
                    "crl_hips__C",
                    pitch_deg=float(surface_pitch_deg),
                    roll_deg=float(surface_roll_deg),
                    tz_m=-0.42,
                ),
                BoneTransformEuler("crl_thigh__L", pitch_deg=88.0),
                BoneTransformEuler("crl_thigh__R", pitch_deg=88.0),
                BoneTransformEuler("crl_leg__L", pitch_deg=-88.0),
                BoneTransformEuler("crl_leg__R", pitch_deg=-88.0),
            ]
        )

    if is_on_phone:
        bones.extend(
            [
                BoneTransformEuler("crl_neck__C", pitch_deg=32.0, yaw_deg=0.0, roll_deg=0.0),
                BoneTransformEuler("crl_Head__C", pitch_deg=10.0, yaw_deg=0.0, roll_deg=0.0),
                BoneTransformEuler("crl_arm__R", pitch_deg=-35.0, yaw_deg=-10.0, roll_deg=0.0),
                BoneTransformEuler("crl_foreArm__R", pitch_deg=-65.0, yaw_deg=-20.0, roll_deg=0.0),
            ]
        )
        return bones

    if archetype in ("older_adult", "older_adult_rollator"):
        target_yaw = float(np.clip(active_gaze_yaw_deg, -55.0, 55.0))
        if prev_neck_yaw_deg is not None and dt_s > 0.0:
            max_delta = 90.0 * float(dt_s)
            delta = float(np.clip(target_yaw - float(prev_neck_yaw_deg), -max_delta, max_delta))
            target_yaw = float(np.clip(float(prev_neck_yaw_deg) + delta, -55.0, 55.0))
        bones.extend(
            [
                BoneTransformEuler("crl_spine01__C", pitch_deg=8.0),
                BoneTransformEuler("crl_neck__C", yaw_deg=target_yaw),
            ]
        )
    elif archetype == "blind_low_vision":
        cane_sweep_deg = 22.0 * math.sin(2.0 * math.pi * 1.2 * float(sim_time_s))
        bones.extend(
            [
                BoneTransformEuler("crl_neck__C", yaw_deg=float(np.clip(active_gaze_yaw_deg, -45.0, 45.0))),
                BoneTransformEuler("crl_foreArm__R", pitch_deg=-25.0),
                BoneTransformEuler("crl_hand__R", yaw_deg=cane_sweep_deg),
            ]
        )
    else:
        bones.append(
            BoneTransformEuler("crl_neck__C", yaw_deg=float(np.clip(active_gaze_yaw_deg, -75.0, 75.0)))
        )

    return bones


def to_carla_walker_bone_control(
    bones: list[BoneTransformEuler],
    carla_module: Any = None,
    force_dict: bool = False,
) -> Any:
    """Convert `BoneTransformEuler` list into `carla.WalkerBoneControlIn` or RPC dict."""
    if force_dict:
        return {"bone_transforms": [b.to_dict() for b in bones]}

    mod = carla_module if carla_module is not None else carla
    if mod is not None and hasattr(mod, "WalkerBoneControlIn"):
        bone_tuples = [
            (
                b.bone_name,
                mod.Transform(
                    mod.Location(x=b.tx_m, y=b.ty_m, z=b.tz_m),
                    mod.Rotation(pitch=b.pitch_deg, yaw=b.yaw_deg, roll=b.roll_deg),
                ),
            )
            for b in bones
        ]
        return mod.WalkerBoneControlIn(bone_tuples)

    return {"bone_transforms": [b.to_dict() for b in bones]}


class WalkerBonePostureControl(BasicControl):
    """
    ScenarioRunner `BasicControl` plugin driving both pedestrian root locomotion and
    `carla.WalkerBoneControlIn` skeletal posture overrides (`ON_PHONE`, cane, elderly, wheelchair).

    Compatible with `ActorControl` dynamic loading via `module="walker_bone_posture_control"`.
    """

    def __init__(self, actor: Any, args: dict[str, Any] | None = None) -> None:
        super().__init__(actor)
        cfg = args or {}
        self.archetype: str = str(cfg.get("archetype", "adult_baseline"))
        raw_gaze = cfg.get("gaze_state", "ON_ROAD")
        self.gaze_state: int = (
            GAZE_ON_PHONE
            if str(raw_gaze).upper() in ("0", "ON_PHONE")
            else GAZE_ON_ROAD
        )
        self.active_gaze_yaw_deg: float = float(cfg.get("active_gaze_yaw_deg", 0.0))
        self.dt_s: float = float(cfg.get("dt_s", 0.05))
        self._sim_time_s: float = 0.0
        self._prev_neck_yaw_deg: float = 0.0
        self.last_bones: list[BoneTransformEuler] = []

    def reset(self) -> None:
        """Reset controller reference."""
        if self._actor and getattr(self._actor, "is_alive", False):
            self._actor = None
        self.last_bones = []

    def run_step(self) -> None:
        """Advance pedestrian root velocity and apply skeletal `WalkerBoneControlIn` overrides."""
        self._sim_time_s += self.dt_s
        self.last_bones = build_walker_bone_overrides(
            archetype=self.archetype,
            gaze_state=self.gaze_state,
            sim_time_s=self._sim_time_s,
            active_gaze_yaw_deg=self.active_gaze_yaw_deg,
            prev_neck_yaw_deg=self._prev_neck_yaw_deg,
            dt_s=self.dt_s,
        )
        for b in self.last_bones:
            if b.bone_name == "crl_neck__C":
                self._prev_neck_yaw_deg = b.yaw_deg

        if not self._actor or not getattr(self._actor, "is_alive", True):
            return

        if hasattr(self._actor, "get_control") and hasattr(self._actor, "apply_control"):
            control = self._actor.get_control()
            control.speed = max(0.0, float(self._target_speed))
            if self._waypoints:
                self._reached_goal = False
                location = self._waypoints[0].location
                direction = location - self._actor.get_location()
                norm = math.sqrt(direction.x**2 + direction.y**2)
                if norm > 1e-5:
                    control.direction = direction / norm
                if norm < 1.0:
                    self._waypoints = self._waypoints[1:]
                    if not self._waypoints:
                        self._reached_goal = True
            elif hasattr(self._actor, "get_transform"):
                control.direction = self._actor.get_transform().rotation.get_forward_vector()
            self._actor.apply_control(control)

        if hasattr(self._actor, "set_bones"):
            bone_ctrl = to_carla_walker_bone_control(self.last_bones)
            self._actor.set_bones(bone_ctrl)
            if hasattr(self._actor, "show_pose"):
                self._actor.show_pose()
