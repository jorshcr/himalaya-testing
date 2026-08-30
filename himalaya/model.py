"""Deterministic, in-memory overlay on the pinned upstream G1 model."""

from __future__ import annotations

from dataclasses import dataclass
import math
from xml.etree import ElementTree as ET

import numpy as np

from .config import ExperimentConfig, MENAGERIE_REVISION


ALL_FOURS_QPOS = (
    # Audited forward-hand support pose.  The previous pose folded both arms
    # behind the shoulders and made the head an unsupported cantilever.
    0.0, 0.0, 0.3180000000,
    0.7047546641, 0.0, 0.7094511001, 0.0,
    -1.073643179, 0.0, 0.0, 1.327866394, 0.1304456157, 0.0,
    -1.073643179, 0.0, 0.0, 1.327866394, 0.1304456157, 0.0,
    0.0, 0.0, 0.073,
    -0.3133458950, 1.3059744290, 0.3345391130, -0.0773406090,
    0.2684730470, 0.0625170370, -1.6100000000,
    -0.3133458950, -1.3059744290, -0.3345391130, -0.0773406090,
    -0.2684730470, 0.0625170370, 1.6100000000,
)

# Position-actuator targets are deliberately distinct from the observed reset
# pose.  They provide the static preload required to carry the robot's weight;
# treating qpos as ctrl was the main reason the old reset collapsed.
ALL_FOURS_CTRL = (
    -1.2306813674, 0.0002833644, 0.0002214790, 0.7234758024,
    0.1495123556, 0.0007496896,
    -1.2308270018, 0.0002716737, 0.0000911773, 0.7243249109,
    0.1492630893, -0.0008690798,
    -0.0003791447, -0.0000225161, 0.2507063063,
    -0.4091417065, 1.2692527413, 0.3133063789, -0.0653610824,
    0.3437745735, 0.0654024026, -1.5590607426,
    -0.4088444085, -1.2694228453, -0.3134203066, -0.0653856314,
    -0.3435858548, 0.0654027031, 1.5591411785,
)


@dataclass(frozen=True)
class OverlayBundle:
    scene_xml: str
    assets: dict[str, bytes]


def _numbers(values: tuple[float, ...]) -> str:
    return " ".join(f"{value:.10g}" for value in values)


def _named_body(root: ET.Element, name: str) -> ET.Element:
    body = root.find(f".//body[@name='{name}']")
    if body is None:
        raise ValueError(f"upstream G1 is missing body {name!r}")
    return body


def _patch_robot_xml(source: bytes, config: ExperimentConfig) -> bytes:
    root = ET.fromstring(source)
    contact = root.find("contact")
    if contact is None:
        raise ValueError("upstream G1 XML has no contact section")
    for side in ("left", "right"):
        pair = contact.find(f"pair[@name='{side}_foot_floor']")
        if pair is None:
            raise ValueError(f"upstream G1 is missing {side} foot pair")
        pair.set("friction", f"{config.contact.foot_sliding_friction:g} {config.contact.foot_sliding_friction:g}")

    # Minimal central proxies exist only to make a fallen state observable.
    ET.SubElement(
        _named_body(root, "pelvis"), "geom",
        name="himalaya_pelvis_proxy", type="sphere", size="0.07",
        pos="0 0 -0.08", contype="0", conaffinity="0", group="3",
    )
    torso = _named_body(root, "torso_link")
    ET.SubElement(
        torso, "geom", name="himalaya_torso_proxy", type="capsule",
        size="0.09", fromto="0.01 0 0.08 0.01 0 0.2",
        contype="0", conaffinity="0", group="3",
    )
    ET.SubElement(
        torso, "geom", name="himalaya_head_proxy", type="sphere",
        size="0.06", pos="0 0 0.43", contype="0", conaffinity="0", group="3",
    )

    keyframe = root.find("keyframe")
    if keyframe is None:
        keyframe = ET.SubElement(root, "keyframe")
    ET.SubElement(
        keyframe,
        "key",
        name=config.reset.keyframe,
        qpos=_numbers(ALL_FOURS_QPOS),
        ctrl=_numbers(ALL_FOURS_CTRL),
    )
    return ET.tostring(root, encoding="utf-8")


def build_overlay_bundle(config: ExperimentConfig) -> OverlayBundle:
    """Load upstream assets and apply only the reviewed Himalaya delta."""

    from mujoco_playground._src import mjx_env
    from mujoco_playground._src.locomotion.g1 import base, g1_constants

    if mjx_env.MENAGERIE_COMMIT_SHA != MENAGERIE_REVISION:
        raise RuntimeError(
            "Playground expects an unexpected Menagerie revision: "
            f"{mjx_env.MENAGERIE_COMMIT_SHA}"
        )
    mjx_env.ensure_menagerie_exists()
    assets = dict(base.get_assets())
    robot_name = "g1_mjx_feetonly.xml"
    assets[robot_name] = _patch_robot_xml(assets[robot_name], config)

    scene_path = g1_constants.FEET_ONLY_ROUGH_TERRAIN_XML
    scene = ET.fromstring(scene_path.read_text())
    contact = ET.SubElement(scene, "contact")
    for side in ("left", "right"):
        ET.SubElement(
            contact, "pair", name=f"{side}_hand_floor",
            geom1=f"{side}_hand_collision", geom2="floor", condim="3",
            solref=f"{config.contact.hand_contact_time_constant_s:g} 1",
            friction=(
                f"{config.contact.hand_sliding_friction:g} "
                f"{config.contact.hand_sliding_friction:g}"
            ),
        )
    for name in ("pelvis", "torso", "head"):
        ET.SubElement(
            contact, "pair", name=f"himalaya_{name}_floor",
            geom1=f"himalaya_{name}_proxy", geom2="floor", condim="3",
        )

    sensors = ET.SubElement(scene, "sensor")
    for side in ("left", "right"):
        ET.SubElement(
            sensors, "contact", name=f"{side}_hand_floor_found",
            geom1=f"{side}_hand_collision", geom2="floor",
            reduce="mindist", num="1", data="found",
        )
        ET.SubElement(
            sensors, "framelinvel", name=f"{side}_palm_global_linvel",
            objtype="site", objname=f"{side}_palm",
        )
        ET.SubElement(sensors, "force", name=f"{side}_hand_force", site=f"{side}_palm")
        ET.SubElement(sensors, "torque", name=f"{side}_hand_torque", site=f"{side}_palm")
    for name in ("pelvis", "torso", "head"):
        ET.SubElement(
            sensors, "contact", name=f"himalaya_{name}_floor_found",
            geom1=f"himalaya_{name}_proxy", geom2="floor",
            reduce="mindist", num="1", data="found",
        )
    return OverlayBundle(ET.tostring(scene, encoding="unicode"), assets)


def compile_model(config: ExperimentConfig):
    import mujoco

    bundle = build_overlay_bundle(config)
    model = mujoco.MjModel.from_xml_string(bundle.scene_xml, assets=bundle.assets)
    model.opt.timestep = 0.002
    configure_slope_heightfield(model, config.slope_degrees)
    return model


def configure_slope_heightfield(model, slope_degrees: float) -> None:
    """Encode true grade in z-up heightfield data with a smooth reset patch."""

    if model.nhfield != 1:
        raise ValueError("expected exactly one rough-terrain heightfield")
    rows, cols = int(model.hfield_nrow[0]), int(model.hfield_ncol[0])
    address = int(model.hfield_adr[0])
    base_thickness = float(model.hfield_size[0, 3])
    count = rows * cols
    source = model.hfield_data[address : address + count].reshape(rows, cols).copy()
    half_x, half_y = model.hfield_size[0, :2]
    # MuJoCo maps increasing heightfield columns to increasing world X.
    # Keeping this sign explicit makes +X the audited uphill direction.
    x = np.linspace(-half_x, half_x, cols)
    y = np.linspace(-half_y, half_y, rows)
    xx, yy = np.meshgrid(x, y)
    outside_x = np.clip((np.abs(xx) - 0.75) / 0.25, 0.0, 1.0)
    outside_y = np.clip((np.abs(yy) - 0.45) / 0.25, 0.0, 1.0)
    blend = np.maximum(outside_x, outside_y)
    blend = blend * blend * (3.0 - 2.0 * blend)
    span = float(np.ptp(source))
    if span <= 0:
        raise ValueError("rough terrain heightfield is constant")
    roughness = ((source - float(np.min(source))) / span - 0.5) * 0.05
    surface = math.tan(math.radians(slope_degrees)) * xx + roughness * blend
    low, height = float(np.min(surface)), float(np.ptp(surface))
    model.hfield_data[address : address + count] = ((surface - low) / height).ravel()
    model.hfield_size[0, 2] = height
    # hfield_size[3] is the solid base thickness, not a vertical datum.  Put
    # negative elevation in the geom transform so the normalized samples map
    # back to the requested signed surface.
    model.hfield_size[0, 3] = base_thickness
    model.geom_pos[model.geom("floor").id, 2] = low
    # World-body geom transforms are compile-time constants in MuJoCo.  Refresh
    # them before native stepping or conversion to MJX.
    import mujoco

    mujoco.mj_setConst(model, mujoco.MjData(model))
