"""Pre-training gates for upstream identity, overlay, reset, and collision."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from .config import ExperimentConfig
from .model import compile_model


@dataclass(frozen=True)
class AuditReport:
    passed: bool
    upstream_baseline_passed: bool
    action_size: int
    hand_friction: tuple[float, float]
    foot_friction: tuple[float, float]
    joint_limits_valid: bool
    minimum_hand_clearance_m: float
    maximum_hand_penetration_m: float
    hand_contacts: int
    support_contacts: tuple[str, ...]
    initial_support_contacts: tuple[str, ...]
    support_occupancy_maintained: bool
    minimum_root_height_m: float
    final_root_height_m: float
    prohibited_contacts: tuple[str, ...]
    settling_passed: bool
    nonfinite: bool
    notes: tuple[str, ...]


def _nominal_data(model: mujoco.MjModel, config: ExperimentConfig) -> mujoco.MjData:
    data = mujoco.MjData(model)
    key = model.keyframe(config.reset.keyframe)
    data.qpos[:] = key.qpos
    data.ctrl[:] = key.ctrl
    slope = math.radians(config.slope_degrees)
    ramp_quat = np.asarray([math.cos(slope / 2.0), 0.0, -math.sin(slope / 2.0), 0.0])
    result = np.empty(4)
    mujoco.mju_mulQuat(result, ramp_quat, data.qpos[3:7].copy())
    data.qpos[3:7] = result
    data.qpos[:3] = key.qpos[2] * np.asarray(
        [-math.sin(slope), 0.0, math.cos(slope)]
    )
    mujoco.mj_forward(model, data)
    return data


def audit_model(config: ExperimentConfig, *, settle_seconds: float = 2.0) -> AuditReport:
    from mujoco_playground._src import mjx_env
    from mujoco_playground._src.locomotion.g1 import base, g1_constants

    # A clean environment has no Menagerie checkout until explicitly
    # materialized.  Do this before the untouched-upstream parity model is
    # compiled, not only later when the Himalaya overlay is constructed.
    mjx_env.ensure_menagerie_exists()
    upstream_assets = base.get_assets()
    upstream_model = mujoco.MjModel.from_xml_string(
        g1_constants.FEET_ONLY_ROUGH_TERRAIN_XML.read_text(),
        assets=upstream_assets,
    )
    upstream_data = mujoco.MjData(upstream_model)
    upstream_key = upstream_model.keyframe("knees_bent")
    upstream_data.qpos[:] = upstream_key.qpos
    upstream_data.ctrl[:] = upstream_key.ctrl
    mujoco.mj_forward(upstream_model, upstream_data)
    mujoco.mj_step(upstream_model, upstream_data)
    upstream_baseline_passed = bool(
        upstream_model.nu == 29
        and np.all(np.isfinite(upstream_data.qpos))
        and np.all(np.isfinite(upstream_data.qvel))
    )

    model = compile_model(config)
    data = _nominal_data(model, config)
    key = model.keyframe(config.reset.keyframe)
    slope = math.radians(config.slope_degrees)
    normal = np.asarray([-math.sin(slope), 0.0, math.cos(slope)])
    hand_ids = {
        model.geom("left_hand_collision").id,
        model.geom("right_hand_collision").id,
    }
    floor_id = model.geom("floor").id
    support_by_id = {
        model.geom("left_hand_collision").id: "left_hand",
        model.geom("right_hand_collision").id: "right_hand",
        model.geom("left_foot").id: "left_foot",
        model.geom("right_foot").id: "right_foot",
    }

    def contact_state() -> tuple[set[int], set[str], set[str]]:
        hand_ids_active: set[int] = set()
        support_names: set[str] = set()
        for contact in data.contact:
            pair = {int(contact.geom1), int(contact.geom2)}
            if floor_id in pair and bool(pair & hand_ids):
                hand_ids_active.update(pair & hand_ids)
            if floor_id in pair:
                support_names.update(
                    support_by_id[geom_id]
                    for geom_id in pair
                    if geom_id in support_by_id
                )
        prohibited = {
            name
            for name in ("pelvis", "torso", "head")
            if data.sensordata[
                model.sensor_adr[model.sensor(f"himalaya_{name}_floor_found").id]
            ]
            > 0
        }
        return hand_ids_active, support_names, prohibited

    def hand_clearance(geom_id: int) -> float:
        rotation = data.geom_xmat[geom_id].reshape(3, 3)
        size = model.geom_size[geom_id]
        extent = float(size[0] + size[1] * abs(rotation[:, 2] @ normal))
        return float(data.geom_xpos[geom_id] @ normal) - extent

    minimum_clearance = min(hand_clearance(geom) for geom in hand_ids)
    penetration = max(0.0, -minimum_clearance)
    hand_contact_ids, initial_support_names, prohibited_names = contact_state()
    expected_supports = set(support_by_id.values())
    support_maintained = initial_support_names == expected_supports
    root_height = float(data.qpos[:3] @ normal)
    minimum_root_height = root_height
    steps = max(0, round(settle_seconds / model.opt.timestep))
    for _ in range(steps):
        mujoco.mj_step(model, data)
        _, current_supports, current_prohibited = contact_state()
        support_maintained &= current_supports == expected_supports
        prohibited_names.update(current_prohibited)
        root_height = float(data.qpos[:3] @ normal)
        minimum_root_height = min(minimum_root_height, root_height)
        penetration = max(
            penetration,
            max(0.0, -min(hand_clearance(geom) for geom in hand_ids)),
        )
        minimum_clearance = min(
            minimum_clearance,
            min(hand_clearance(geom) for geom in hand_ids),
        )
    lowers, uppers = model.jnt_range[1:].T
    joint_limits_valid = bool(
        np.all(key.qpos[7:] >= lowers - 1.0e-8)
        and np.all(key.qpos[7:] <= uppers + 1.0e-8)
        and np.all(key.ctrl >= model.actuator_ctrlrange[:, 0] - 1.0e-8)
        and np.all(key.ctrl <= model.actuator_ctrlrange[:, 1] + 1.0e-8)
    )
    hand_friction = tuple(
        float(model.pair(f"{side}_hand_floor").friction[0])
        for side in ("left", "right")
    )
    foot_friction = tuple(
        float(model.pair(f"{side}_foot_floor").friction[0])
        for side in ("left", "right")
    )
    nonfinite = not bool(np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel)))
    _, settled_support_names, current_prohibited = contact_state()
    prohibited_names.update(current_prohibited)
    settling_passed = bool(
        minimum_root_height >= config.reset.minimum_root_height_m
        and not prohibited_names
        and not nonfinite
    )
    notes = (
        "microspike friction values are uncalibrated simulation assumptions",
        "settling verifies the reset/controller target; it is not learned balance evidence",
        "exact four-contact occupancy is diagnostic and is not a survival condition",
    )
    passed = bool(
        upstream_baseline_passed
        and model.nu == 29
        and hand_friction == (config.contact.hand_sliding_friction,) * 2
        and foot_friction == (config.contact.foot_sliding_friction,) * 2
        and joint_limits_valid
        and penetration <= config.contact.max_hand_penetration_m
        and len(hand_contact_ids) == 2
        and initial_support_names == expected_supports
        and settling_passed
    )
    return AuditReport(
        passed=passed,
        upstream_baseline_passed=upstream_baseline_passed,
        action_size=model.nu,
        hand_friction=hand_friction,
        foot_friction=foot_friction,
        joint_limits_valid=joint_limits_valid,
        minimum_hand_clearance_m=minimum_clearance,
        maximum_hand_penetration_m=penetration,
        hand_contacts=len(hand_contact_ids),
        support_contacts=tuple(sorted(settled_support_names)),
        initial_support_contacts=tuple(sorted(initial_support_names)),
        support_occupancy_maintained=support_maintained,
        minimum_root_height_m=minimum_root_height,
        final_root_height_m=root_height,
        prohibited_contacts=tuple(sorted(prohibited_names)),
        settling_passed=settling_passed,
        nonfinite=nonfinite,
        notes=notes,
    )


def write_audit(report: AuditReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def render_collision_audit(
    config: ExperimentConfig, path: Path, *, settle_seconds: float = 0.0
) -> Path:
    """Render the audited reset, optionally after zero-action settling."""

    model = compile_model(config)
    data = _nominal_data(model, config)
    for _ in range(round(settle_seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)
    option = mujoco.MjvOption()
    option.geomgroup[3] = True
    option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        renderer.update_scene(data, camera="track", scene_option=option)
        pixels = renderer.render()
    from PIL import Image

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(path)
    return path
