#!/usr/bin/env python

# Copyright (c) 2026 Horizon Europe MSCA STEP-UP Consortium
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Pytest suite verifying STEP-UP VRU skeletal bone posture overrides, 3D Kay-Kajiya
sightline occlusion & curb-extension lead-time gains, POMDP phantom occupancy grid,
CIE 1.5 Hz pulsing cyan eHMI shader contract, and ASAM OpenSCENARIO 1.2 XML scenarios.
"""

from __future__ import annotations

from pathlib import Path
import types
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from srunner.scenariomanager.actorcontrols.walker_bone_posture_control import (
    GAZE_ON_PHONE,
    GAZE_ON_ROAD,
    WalkerBonePostureControl,
    build_walker_bone_overrides,
    to_carla_walker_bone_control,
)
from srunner.tools.vru_occlusion_and_ehmi import (
    CarlaEhmiShaderController,
    OccluderBox3D,
    PhantomVruBeliefGrid,
    apply_curb_extension_daylighting,
    compute_cav_vru_visibility,
    evaluate_approach_detection_lead_time,
    evaluate_ehmi_shader_state,
    sample_16_skeletal_keypoints_3d,
)

SRUNNER_ROOT = Path(__file__).resolve().parents[1]


def test_walker_bone_posture_overrides_and_controller() -> None:
    """Verify ON_PHONE, 1.2 Hz cane sweep, kyphotic older-adult spine, and wheelchair bones."""
    phone_bones = {
        b.bone_name: b
        for b in build_walker_bone_overrides("adult_phone", GAZE_ON_PHONE)
    }
    assert phone_bones["crl_neck__C"].pitch_deg == pytest.approx(32.0)
    assert phone_bones["crl_Head__C"].pitch_deg == pytest.approx(10.0)
    assert phone_bones["crl_arm__R"].pitch_deg == pytest.approx(-35.0)
    assert phone_bones["crl_foreArm__R"].pitch_deg == pytest.approx(-65.0)
    assert phone_bones["crl_foreArm__R"].yaw_deg == pytest.approx(-20.0)

    # 1.2 Hz white-cane sweep quarter-cycle (+22 deg) and three-quarter-cycle (-22 deg)
    b_pos = {
        b.bone_name: b
        for b in build_walker_bone_overrides("blind_low_vision", GAZE_ON_ROAD, sim_time_s=0.25 / 1.2)
    }
    b_neg = {
        b.bone_name: b
        for b in build_walker_bone_overrides("blind_low_vision", GAZE_ON_ROAD, sim_time_s=0.75 / 1.2)
    }
    assert b_pos["crl_hand__R"].yaw_deg == pytest.approx(22.0, abs=1e-4)
    assert b_neg["crl_hand__R"].yaw_deg == pytest.approx(-22.0, abs=1e-4)

    # Older adult kyphotic spine (+8 deg) and bounded neck yaw (<= 90 deg/s)
    b_elder = {
        b.bone_name: b
        for b in build_walker_bone_overrides(
            "older_adult", GAZE_ON_ROAD, active_gaze_yaw_deg=50.0, prev_neck_yaw_deg=0.0, dt_s=0.10
        )
    }
    assert b_elder["crl_spine01__C"].pitch_deg == pytest.approx(8.0)
    assert b_elder["crl_neck__C"].yaw_deg == pytest.approx(9.0)

    # Wheelchair seated pelvis
    b_wc = {
        b.bone_name: b
        for b in build_walker_bone_overrides("wheelchair_manual", GAZE_ON_ROAD, surface_pitch_deg=3.5)
    }
    assert b_wc["crl_hips__C"].tz_m == pytest.approx(-0.42)
    assert b_wc["crl_hips__C"].pitch_deg == pytest.approx(3.5)
    assert b_wc["crl_thigh__L"].pitch_deg == pytest.approx(88.0)

    # ScenarioRunner BasicControl plugin integration with mock Walker actor
    applied_bones = []
    mock_walker = types.SimpleNamespace(
        is_alive=True,
        set_bones=lambda ctrl: applied_bones.append(ctrl),
        show_pose=lambda: None,
    )
    ctrl = WalkerBonePostureControl(
        mock_walker,
        args={"archetype": "adult_phone", "gaze_state": "ON_PHONE", "dt_s": 0.05},
    )
    ctrl.update_target_speed(1.12)
    ctrl.run_step()
    assert len(ctrl.last_bones) == 4
    assert len(applied_bones) == 1
    assert len(to_carla_walker_bone_control(ctrl.last_bones, force_dict=True)["bone_transforms"]) == 4


def test_3d_child_sightline_occlusion_and_curb_extension_lead_time_gain() -> None:
    """Verify curb_extension yields > +0.8s detection lead-time gain at 30 km/h for child_7_9yr."""
    kps = sample_16_skeletal_keypoints_3d((0.0, -5.2), "child_7_9yr")
    assert kps.shape == (16, 3)

    # Curbside parked SUV row upstream of crosswalk
    parked_suvs = [
        OccluderBox3D.parked_suv(f"suv_{i}", center_xy=(x_pos, -4.15))
        for i, x_pos in enumerate((5.5, 11.5, 17.5, 23.5))
    ]
    child_curb_xy = (0.0, -5.2)

    vis_blocked = compute_cav_vru_visibility(
        cav_sensor_xyz=(14.0, -1.65, 1.50),
        vru_xy=child_curb_xy,
        archetype="child_7_9yr",
        occluders=parked_suvs,
    )
    assert vis_blocked["detected_reliable"] is False

    lead_before = evaluate_approach_detection_lead_time(
        archetype="child_7_9yr",
        vru_curb_xy=child_curb_xy,
        occluders=parked_suvs,
        approach_speed_kmh=30.0,
    )

    advanced_curb_xy, daylit_occluders = apply_curb_extension_daylighting(
        vru_curb_xy=child_curb_xy,
        occluders=parked_suvs,
        extension_depth_m=2.0,
        clear_upstream_m=10.5,
    )
    lead_after = evaluate_approach_detection_lead_time(
        archetype="child_7_9yr",
        vru_curb_xy=advanced_curb_xy,
        occluders=daylit_occluders,
        approach_speed_kmh=30.0,
    )

    delta_lead_s = lead_after["detection_lead_time_s"] - lead_before["detection_lead_time_s"]
    assert delta_lead_s > 0.80


def test_pomdp_phantom_vru_belief_grid_and_ehmi_shader() -> None:
    """Verify POMDP occluded shadow belief retention and 1.5 Hz pulsing cyan eHMI shader."""
    suv = OccluderBox3D.parked_suv("suv_0", center_xy=(5.5, -4.15))
    grid = PhantomVruBeliefGrid(bounds_xy=(-4.0, -7.0, 14.0, 4.0), res_m=1.0)
    cav_sensor = (12.0, -1.65, 1.50)

    for _ in range(6):
        grid.step_belief(dt_s=0.10, cav_sensor_xyz=cav_sensor, occluders=[suv])

    assert grid.query_prob(6.0, -1.0) < 0.03       # Unobstructed travel lane collapses
    assert grid.query_prob(1.0, -6.0) > 0.30       # Occluded shadow behind SUV retains phantom risk
    assert grid.recommended_cav_speed_cap_ms(12.0, nominal_speed_ms=11.11) < 11.11

    ehmi_ctrl = CarlaEhmiShaderController(initial_mode="VRU_DETECTED_YIELDING")
    s_peak = ehmi_ctrl.update(sim_time_s=0.1667, ambient_eh_lux=5000.0)
    s_trough = ehmi_ctrl.update(sim_time_s=0.5000, ambient_eh_lux=5000.0)
    s_off = evaluate_ehmi_shader_state("OFF")

    assert s_off.peak_luminance_cd_m2 == 0.0
    assert s_peak.pulse_freq_hz == pytest.approx(1.5)
    assert s_peak.projected_zebra_distance_m == pytest.approx(2.5)
    assert s_peak.cie_high_contrast_compliant is True
    assert s_peak.instantaneous_luminance_cd_m2 > s_trough.instantaneous_luminance_cd_m2


@pytest.mark.parametrize(
    "xosc_filename",
    [
        "dc02_phone_distracted_crossing.xosc",
        "dc08_parked_suv_child_occlusion.xosc",
        "dc08_multilane_ehmi_overtrust.xosc",
    ],
)
def test_stepup_openscenario_1_2_examples(xosc_filename: str) -> None:
    """Validate ASAM OpenSCENARIO 1.2 XML structure for all 3 STEP-UP scenarios."""
    xosc_path = SRUNNER_ROOT / "examples" / "stepup" / xosc_filename
    assert xosc_path.exists()

    root = ET.parse(xosc_path).getroot()
    assert root.tag == "OpenSCENARIO"
    header = root.find("FileHeader")
    assert header is not None
    assert header.attrib.get("revMajor") == "1"
    assert header.attrib.get("revMinor") == "2"

    entities = root.find("Entities")
    assert entities is not None
    names = {obj.attrib["name"] for obj in entities.findall("ScenarioObject")}
    assert "ego_vehicle" in names

    storyboard = root.find("Storyboard")
    assert storyboard is not None
    assert storyboard.find("Init") is not None
    assert storyboard.find("Story") is not None
    assert storyboard.find("StopTrigger") is not None
