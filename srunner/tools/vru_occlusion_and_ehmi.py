#!/usr/bin/env python

# Copyright (c) 2026 Horizon Europe MSCA STEP-UP Consortium
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
3D VRU Sightline Occlusion Raycaster, POMDP Phantom Occupancy Grid & CIE 1.5 Hz eHMI Controller.

Implements MSCA STEP-UP DC8 & DC9 CARLA evaluation tools:
  - Exact 3D Kay-Kajiya slab raycaster over K = 16 anthropometric VRU keypoints
  - Approach detection lead-time sweep & curb-extension daylighting evaluation
  - POMDP `PhantomVruBeliefGrid` for occluded VRU emergence risk
  - CIE high-contrast `1.5 Hz` pulsing cyan CAV eHMI shader controller
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np

EHMI_MODE_MAP: dict[str, int] = {
    "OFF": 0,
    "CRUISING_NON_YIELDING": 1,
    "VRU_DETECTED_YIELDING": 2,
}
EHMI_ID_TO_NAME: dict[int, str] = {v: k for k, v in EHMI_MODE_MAP.items()}


@dataclass(frozen=True)
class VruAnthropometry:
    """Anthropometric eye height & shoulder semi-width for 3D skeletal keypoint synthesis."""

    eye_height_m: float
    shoulder_radius_m: float


VRU_ANTHROPOMETRY: dict[str, VruAnthropometry] = {
    "child_6_8yr": VruAnthropometry(eye_height_m=1.18, shoulder_radius_m=0.19),
    "child_7_9yr": VruAnthropometry(eye_height_m=1.22, shoulder_radius_m=0.20),
    "child_9_11yr": VruAnthropometry(eye_height_m=1.32, shoulder_radius_m=0.21),
    "wheelchair_manual": VruAnthropometry(eye_height_m=1.18, shoulder_radius_m=0.38),
    "powered_scooter": VruAnthropometry(eye_height_m=1.24, shoulder_radius_m=0.42),
    "older_adult": VruAnthropometry(eye_height_m=1.55, shoulder_radius_m=0.27),
    "older_adult_rollator": VruAnthropometry(eye_height_m=1.52, shoulder_radius_m=0.32),
    "blind_low_vision": VruAnthropometry(eye_height_m=1.65, shoulder_radius_m=0.30),
    "adult_phone": VruAnthropometry(eye_height_m=1.60, shoulder_radius_m=0.25),
    "adult_baseline": VruAnthropometry(eye_height_m=1.65, shoulder_radius_m=0.24),
}


@dataclass(frozen=True)
class OccluderBox3D:
    """Axis-aligned 3D bounding box of a curbside sightline occluder."""

    box_id: str
    center_xy: tuple[float, float]
    length_m: float = 4.90
    width_m: float = 1.95
    height_m: float = 1.78
    z_base_m: float = 0.0
    category: str = "parked_suv"

    def to_aabb(self) -> tuple[np.ndarray, np.ndarray]:
        cx, cy = float(self.center_xy[0]), float(self.center_xy[1])
        hl, hw = 0.5 * float(self.length_m), 0.5 * float(self.width_m)
        bmin = np.array([cx - hl, cy - hw, float(self.z_base_m)], dtype=float)
        bmax = np.array(
            [cx + hl, cy + hw, float(self.z_base_m) + float(self.height_m)], dtype=float
        )
        return bmin, bmax

    @classmethod
    def parked_suv(cls, box_id: str, center_xy: tuple[float, float]) -> OccluderBox3D:
        return cls(
            box_id=box_id,
            center_xy=center_xy,
            length_m=4.90,
            width_m=1.95,
            height_m=1.78,
            category="parked_suv",
        )

    @classmethod
    def transit_bus(cls, box_id: str, center_xy: tuple[float, float]) -> OccluderBox3D:
        return cls(
            box_id=box_id,
            center_xy=center_xy,
            length_m=12.0,
            width_m=2.55,
            height_m=3.15,
            category="transit_bus",
        )


def sample_16_skeletal_keypoints_3d(
    xy: tuple[float, float],
    archetype: str = "adult_baseline",
    gaze_state: int = 1,
) -> np.ndarray:
    """Generate `K = 16` 3D anatomical keypoints `(16, 3)` scaled to cohort stature."""
    anthro = VRU_ANTHROPOMETRY.get(archetype, VRU_ANTHROPOMETRY["adult_baseline"])
    z_eye = anthro.eye_height_m
    z_top = z_eye + 0.10
    x0, y0 = float(xy[0]), float(xy[1])
    w_sh = anthro.shoulder_radius_m * 0.85
    head_drop = 0.05 if gaze_state == 0 else 0.0

    pts = [
        [x0, y0, z_top - head_drop],
        [x0, y0, z_eye - head_drop],
        [x0, y0, 0.86 * z_top],
        [x0 - w_sh, y0, 0.82 * z_top],
        [x0 + w_sh, y0, 0.82 * z_top],
        [x0 - w_sh, y0, 0.64 * z_top],
        [x0 + w_sh, y0, 0.64 * z_top],
        [x0 - w_sh, y0, 0.48 * z_top],
        [x0 + 0.5 * w_sh, y0, 0.72 * z_top if gaze_state == 0 else 0.48 * z_top],
        [x0, y0, 0.68 * z_top],
        [x0 - 0.6 * w_sh, y0, 0.53 * z_top],
        [x0 + 0.6 * w_sh, y0, 0.53 * z_top],
        [x0 - 0.6 * w_sh, y0, 0.29 * z_top],
        [x0 + 0.6 * w_sh, y0, 0.29 * z_top],
        [x0 - 0.6 * w_sh, y0, 0.06 * z_top],
        [x0 + 0.6 * w_sh, y0, 0.06 * z_top],
    ]
    return np.asarray(pts, dtype=float)


def segment_intersects_aabb_3d(
    p0: np.ndarray,
    p1: np.ndarray,
    box_min: np.ndarray,
    box_max: np.ndarray,
) -> bool:
    """Exact 3D Kay-Kajiya slab ray-segment intersection test against an AABB."""
    d = np.asarray(p1, dtype=float) - np.asarray(p0, dtype=float)
    t_min = 0.0
    t_max = 1.0
    for axis in range(3):
        if abs(d[axis]) < 1e-9:
            if p0[axis] < box_min[axis] or p0[axis] > box_max[axis]:
                return False
        else:
            inv_d = 1.0 / d[axis]
            t1 = (box_min[axis] - p0[axis]) * inv_d
            t2 = (box_max[axis] - p0[axis]) * inv_d
            if t1 > t2:
                t1, t2 = t2, t1
            t_min = max(t_min, t1)
            t_max = min(t_max, t2)
            if t_min > t_max:
                return False
    return t_max >= t_min and (t_min < 1.0 - 1e-9 and t_max > 1e-9)


def compute_cav_vru_visibility(
    cav_sensor_xyz: tuple[float, float, float],
    vru_xy: tuple[float, float],
    archetype: str = "adult_baseline",
    occluders: Sequence[OccluderBox3D] = (),
    gaze_state: int = 1,
    eh_lux: float = 1000.0,
) -> dict[str, float | bool]:
    """Cast 16 skeletal keypoints from CAV sensor rig against 3D occluder boxes."""
    sensor_pt = np.asarray(cav_sensor_xyz, dtype=float)
    keypoints = sample_16_skeletal_keypoints_3d(vru_xy, archetype, gaze_state=gaze_state)
    aabbs = [occ.to_aabb() for occ in occluders]

    visible_k = 0
    for kp in keypoints:
        blocked = False
        for bmin, bmax in aabbs:
            if segment_intersects_aabb_3d(sensor_pt, kp, bmin, bmax):
                blocked = True
                break
        if not blocked:
            visible_k += 1

    nu_vis = float(visible_k) / float(len(keypoints))
    dist_m = float(np.linalg.norm(keypoints[1] - sensor_pt))
    lux_factor = min(1.0, 0.55 + 0.45 * math.log10(max(1.0, float(eh_lux))) / 2.0)
    range_factor = math.exp(-max(0.0, dist_m - 15.0) / 45.0)
    if nu_vis < 0.25:
        p_det = 0.05 * nu_vis
    else:
        p_det = float(np.clip(nu_vis * lux_factor * range_factor, 0.0, 0.995))

    return {
        "visibility_ratio": nu_vis,
        "visible_keypoints": float(visible_k),
        "p_det": p_det,
        "detected_reliable": bool(nu_vis >= 0.30 and p_det >= 0.45),
    }


def apply_curb_extension_daylighting(
    vru_curb_xy: tuple[float, float],
    occluders: Sequence[OccluderBox3D],
    extension_depth_m: float = 2.0,
    clear_upstream_m: float = 10.5,
) -> tuple[tuple[float, float], list[OccluderBox3D]]:
    """
    Apply a DC4 curb-extension (bulb-out) geometric operator to curbside waiting position
    and upstream parked-vehicle sightline occluders:
      - Advances pedestrian curb waiting position toward the travel lane by `extension_depth_m`
      - Clears parked vehicles within `clear_upstream_m` of the crosswalk centerline (`x = 0`).
    """
    x0, y0 = float(vru_curb_xy[0]), float(vru_curb_xy[1])
    y_advanced = y0 + float(extension_depth_m) if y0 < 0.0 else y0 - float(extension_depth_m)
    remaining = [occ for occ in occluders if abs(float(occ.center_xy[0])) > float(clear_upstream_m)]
    return (x0, y_advanced), remaining


def evaluate_approach_detection_lead_time(
    archetype: str,
    vru_curb_xy: tuple[float, float],
    occluders: Sequence[OccluderBox3D],
    approach_speed_kmh: float = 30.0,
    cav_lane_y: float = -1.65,
    cav_sensor_z: float = 1.50,
    eh_lux: float = 5000.0,
) -> dict[str, float]:
    """
    Sweep an approaching CAV along `y = cav_lane_y` from `x = +42 m` down to `x = 0.5 m`
    at `approach_speed_kmh` and compute continuous unobstructed sightline horizon & lead time.
    """
    v_ms = max(0.5, float(approach_speed_kmh) / 3.6)
    xs_outward = np.linspace(0.5, 42.0, 167)
    continuous_detect_x = 0.5
    vis_trace: list[float] = []
    still_continuous = True

    for x_cav in xs_outward:
        res = compute_cav_vru_visibility(
            cav_sensor_xyz=(float(x_cav), cav_lane_y, cav_sensor_z),
            vru_xy=vru_curb_xy,
            archetype=archetype,
            occluders=occluders,
            eh_lux=eh_lux,
        )
        vis_trace.append(float(res["visibility_ratio"]))
        if still_continuous and bool(res["detected_reliable"]):
            continuous_detect_x = float(x_cav)
        elif x_cav > 2.5:
            still_continuous = False

    return {
        "first_detection_distance_m": continuous_detect_x,
        "detection_lead_time_s": continuous_detect_x / v_ms,
        "mean_visibility_ratio": float(np.mean(vis_trace)),
    }


class PhantomVruBeliefGrid:
    """DC8 POMDP occluded phantom VRU occupancy belief grid & CAV speed-cap planner."""

    def __init__(
        self,
        bounds_xy: tuple[float, float, float, float] = (-6.0, -8.0, 18.0, 6.0),
        res_m: float = 1.0,
        prior_prob: float = 0.08,
        probe_height_m: float = 1.15,
    ) -> None:
        self.bounds_xy = bounds_xy
        self.res_m = max(0.5, float(res_m))
        self.prior_prob = float(prior_prob)
        self.probe_height_m = float(probe_height_m)

        xmin, ymin, xmax, ymax = bounds_xy
        self.xs = np.arange(xmin, xmax + 1e-6, self.res_m)
        self.ys = np.arange(ymin, ymax + 1e-6, self.res_m)
        self.belief = np.full((len(self.ys), len(self.xs)), self.prior_prob, dtype=float)
        self.shadow_mask = np.zeros_like(self.belief, dtype=bool)

    def compute_shadow_mask(
        self,
        cav_sensor_xyz: tuple[float, float, float],
        occluders: Sequence[OccluderBox3D],
    ) -> np.ndarray:
        sensor_pt = np.asarray(cav_sensor_xyz, dtype=float)
        aabbs = [occ.to_aabb() for occ in occluders]
        mask = np.zeros_like(self.belief, dtype=bool)
        if not aabbs:
            self.shadow_mask = mask
            return mask

        for iy, y_val in enumerate(self.ys):
            for ix, x_val in enumerate(self.xs):
                target = np.array([float(x_val), float(y_val), self.probe_height_m], dtype=float)
                for bmin, bmax in aabbs:
                    if segment_intersects_aabb_3d(sensor_pt, target, bmin, bmax):
                        mask[iy, ix] = True
                        break
        self.shadow_mask = mask
        return mask

    def step_belief(
        self,
        dt_s: float,
        cav_sensor_xyz: tuple[float, float, float],
        occluders: Sequence[OccluderBox3D],
        detected_vru_xys: Sequence[tuple[float, float]] = (),
        p_detect_clear: float = 0.94,
        p_false_alarm: float = 0.02,
    ) -> np.ndarray:
        b_prev = self.belief
        padded = np.pad(b_prev, 1, mode="edge")
        neighbor_avg = 0.25 * (
            padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:]
        )
        mix = min(0.35, 0.85 * float(dt_s))
        b_pred = (1.0 - mix) * b_prev + mix * neighbor_avg

        shadow = self.compute_shadow_mask(cav_sensor_xyz, occluders)
        b_pred = np.where(shadow, np.clip(b_pred + 0.55 * float(dt_s), 0.0, 0.85), b_pred)

        obs_occupied = np.zeros_like(self.belief, dtype=bool)
        for vx, vy in detected_vru_xys:
            ix = int(np.clip(round((float(vx) - self.xs[0]) / self.res_m), 0, len(self.xs) - 1))
            iy = int(np.clip(round((float(vy) - self.ys[0]) / self.res_m), 0, len(self.ys) - 1))
            obs_occupied[iy, ix] = True

        num_pos = p_detect_clear * b_pred
        den_pos = num_pos + p_false_alarm * (1.0 - b_pred) + 1e-9
        b_pos = num_pos / den_pos

        num_neg = (1.0 - p_detect_clear) * b_pred
        den_neg = num_neg + (1.0 - p_false_alarm) * (1.0 - b_pred) + 1e-9
        b_neg = num_neg / den_neg

        b_upd = np.where(shadow, b_pred, np.where(obs_occupied, b_pos, b_neg))
        self.belief = np.clip(b_upd, 0.002, 0.995)
        return self.belief

    def query_prob(self, x_m: float, y_m: float) -> float:
        ix = int(np.clip(round((float(x_m) - self.xs[0]) / self.res_m), 0, len(self.xs) - 1))
        iy = int(np.clip(round((float(y_m) - self.ys[0]) / self.res_m), 0, len(self.ys) - 1))
        return float(self.belief[iy, ix])

    def max_conflict_zone_risk(
        self,
        x_range: tuple[float, float] = (-3.5, 8.0),
        y_range: tuple[float, float] = (-6.5, 2.5),
    ) -> float:
        x_mask = (self.xs >= x_range[0]) & (self.xs <= x_range[1])
        y_mask = (self.ys >= y_range[0]) & (self.ys <= y_range[1])
        sub = self.belief[np.ix_(y_mask, x_mask)]
        return float(np.max(sub)) if sub.size > 0 else 0.0

    def recommended_cav_speed_cap_ms(
        self,
        cav_x_m: float,
        nominal_speed_ms: float = 11.11,
        comfortable_decel_ms2: float = 3.2,
        reaction_buf_s: float = 0.35,
    ) -> float:
        risk = self.max_conflict_zone_risk()
        if risk < 0.20 or cav_x_m <= 0.5:
            return float(nominal_speed_ms)
        dist_to_crosswalk = max(0.5, float(cav_x_m) - 2.0)
        a = max(0.5, float(comfortable_decel_ms2))
        tr = float(reaction_buf_s)
        v_stop_bound = -a * tr + math.sqrt((a * tr) ** 2 + 2.0 * a * dist_to_crosswalk)
        blend = min(1.0, (risk - 0.20) / 0.45)
        v_cap = (1.0 - blend) * float(nominal_speed_ms) + blend * v_stop_bound
        return float(max(1.5, min(float(nominal_speed_ms), v_cap)))


@dataclass(frozen=True)
class EhmiShaderState:
    """CIE 1931 high-contrast CAV external HMI (`MI_StepUp_eHMI`) shader uniform state."""

    mode_id: int
    mode_name: str
    cie_1931_xy: tuple[float, float]
    linear_rgb: tuple[float, float, float]
    peak_luminance_cd_m2: float
    instantaneous_luminance_cd_m2: float
    pulse_freq_hz: float
    weber_contrast: float
    cie_high_contrast_compliant: bool
    projected_zebra_distance_m: float


class CarlaEhmiShaderController:
    """Evaluates and synchronizes CIE `1.5 Hz` pulsing cyan eHMI state onto CARLA vehicles."""

    def __init__(self, initial_mode: int | str = "OFF") -> None:
        self.mode: int | str = initial_mode
        self.last_state: EhmiShaderState | None = None

    def update(
        self,
        sim_time_s: float,
        mode: int | str | None = None,
        ambient_eh_lux: float = 500.0,
        vehicle_actor: Any = None,
    ) -> EhmiShaderState:
        if mode is not None:
            self.mode = mode
        state = evaluate_ehmi_shader_state(
            self.mode, sim_time_s=sim_time_s, ambient_eh_lux=ambient_eh_lux
        )
        self.last_state = state
        if vehicle_actor is not None and hasattr(vehicle_actor, "set_light_state"):
            try:
                import carla  # type: ignore[import-not-found]

                if state.mode_id == 2:
                    vehicle_actor.set_light_state(carla.VehicleLightState.Special1)
                elif state.mode_id == 1:
                    vehicle_actor.set_light_state(carla.VehicleLightState.Position)
                else:
                    vehicle_actor.set_light_state(carla.VehicleLightState.NONE)
            except Exception:
                pass
        return state


def evaluate_ehmi_shader_state(
    mode: int | str,
    sim_time_s: float = 0.0,
    ambient_eh_lux: float = 500.0,
    bumper_reflectance_rho: float = 0.18,
) -> EhmiShaderState:
    """Evaluate CIE high-contrast CAV eHMI shader state (`OFF`, `CRUISING_NON_YIELDING`, `VRU_DETECTED_YIELDING`)."""
    if isinstance(mode, str):
        key = mode.strip().upper()
        if key not in EHMI_MODE_MAP:
            raise ValueError(f"Unknown eHMI mode '{mode}'")
        mode_id = EHMI_MODE_MAP[key]
    else:
        mode_id = int(mode)
        if mode_id not in EHMI_ID_TO_NAME:
            raise ValueError(f"Unknown eHMI mode id '{mode_id}'")

    mode_name = EHMI_ID_TO_NAME[mode_id]
    l_bg = max(0.5, float(bumper_reflectance_rho) * max(0.0, float(ambient_eh_lux)) / math.pi)

    if mode_id == 0:
        return EhmiShaderState(
            mode_id=0,
            mode_name=mode_name,
            cie_1931_xy=(0.3127, 0.3290),
            linear_rgb=(0.0, 0.0, 0.0),
            peak_luminance_cd_m2=0.0,
            instantaneous_luminance_cd_m2=0.0,
            pulse_freq_hz=0.0,
            weber_contrast=0.0,
            cie_high_contrast_compliant=False,
            projected_zebra_distance_m=0.0,
        )

    if mode_id == 1:
        peak_lum = 900.0 if ambient_eh_lux > 2000.0 else 550.0
        contrast = peak_lum / l_bg
        return EhmiShaderState(
            mode_id=1,
            mode_name=mode_name,
            cie_1931_xy=(0.560, 0.420),
            linear_rgb=(1.0, 0.68, 0.12),
            peak_luminance_cd_m2=peak_lum,
            instantaneous_luminance_cd_m2=peak_lum,
            pulse_freq_hz=0.0,
            weber_contrast=contrast,
            cie_high_contrast_compliant=contrast >= 3.0,
            projected_zebra_distance_m=0.0,
        )

    pulse_hz = 1.5
    peak_lum = 2600.0 if ambient_eh_lux > 2000.0 else 1100.0
    mod = 0.65 + 0.35 * (0.5 * (1.0 + math.sin(2.0 * math.pi * pulse_hz * float(sim_time_s))))
    inst_lum = peak_lum * mod
    peak_contrast = peak_lum / l_bg
    return EhmiShaderState(
        mode_id=2,
        mode_name=mode_name,
        cie_1931_xy=(0.190, 0.340),
        linear_rgb=(0.0, 0.92, 1.0),
        peak_luminance_cd_m2=peak_lum,
        instantaneous_luminance_cd_m2=inst_lum,
        pulse_freq_hz=pulse_hz,
        weber_contrast=peak_contrast,
        cie_high_contrast_compliant=peak_contrast >= 3.0,
        projected_zebra_distance_m=2.5,
    )
