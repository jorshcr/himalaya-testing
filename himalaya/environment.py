"""Stationary all-fours balance environment built on DeepMind's G1 task."""

from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.g1 import g1_constants
from mujoco_playground._src.locomotion.g1 import joystick as upstream_joystick

from .config import ExperimentConfig, default_config, to_playground_config
from .model import build_overlay_bundle, configure_slope_heightfield
from .physics import (
    force_weighted_support_anchor,
    slope_conditioned_com_height,
    terrain_aligned_zmp,
    terrain_descriptor,
    zmp_reward,
)


class FourContactBalanceEnv(upstream_joystick.Joystick):
    """Thin upstream subclass with four-contact stationary semantics.

    Actor observations remain exactly the upstream proprioceptive ``state``.
    Contact forces and the terrain descriptor are appended only to
    ``privileged_state``.
    """

    def __init__(self, experiment: ExperimentConfig | None = None) -> None:
        self.experiment = experiment or default_config()
        cfg = to_playground_config(self.experiment)
        mjx_env.ensure_menagerie_exists()
        # Reproduce G1Env construction while supplying our in-memory overlay.
        mjx_env.MjxEnv.__init__(self, cfg, None)
        bundle = build_overlay_bundle(self.experiment)
        self._model_assets = bundle.assets
        self._mj_model = mujoco.MjModel.from_xml_string(
            bundle.scene_xml, assets=bundle.assets
        )
        self._mj_model.opt.timestep = self.sim_dt
        self._mj_model.jnt_range[1:] = g1_constants.RESTRICTED_JOINT_RANGE
        self._mj_model.actuator_ctrlrange[:] = g1_constants.RESTRICTED_JOINT_RANGE
        configure_slope_heightfield(self._mj_model, self.experiment.slope_degrees)
        self._mj_model.vis.global_.offwidth = 1920
        self._mj_model.vis.global_.offheight = 1080
        self._mjx_model = mjx.put_model(self._mj_model, impl=cfg.impl)
        self._xml_path = "generated://himalaya/g1-four-contact-rough.xml"
        self._post_init()

        slope = math.radians(self.experiment.slope_degrees)
        cosine, sine = math.cos(slope), math.sin(slope)
        self._slope_radians = slope
        self._ramp_tangent = jp.array([cosine, 0.0, sine])
        self._ramp_cross = jp.array([0.0, 1.0, 0.0])
        self._ramp_normal = jp.array([-sine, 0.0, cosine])
        self._ramp_quat = jp.array(
            [math.cos(slope / 2.0), 0.0, -math.sin(slope / 2.0), 0.0]
        )

        key = self._mj_model.keyframe(self.experiment.reset.keyframe)
        self._init_q = jp.asarray(key.qpos)
        self._reference_pose = jp.asarray(key.qpos[7:])
        # Zero policy action means the audited load-bearing preload, not the
        # observed (gravity-deflected) joint pose.
        self._default_pose = jp.asarray(key.ctrl)
        self._nominal_root_quat = jp.asarray(key.qpos[3:7])
        self._hand_sensor_ids = np.asarray(
            [self._mj_model.sensor(f"{side}_hand_floor_found").id for side in ("left", "right")]
        )
        self._forbidden_sensor_adr = jp.asarray(
            [
                self._mj_model.sensor_adr[
                    self._mj_model.sensor(f"himalaya_{name}_floor_found").id
                ]
                for name in ("pelvis", "torso", "head")
            ]
        )
        self._hand_velocity_adr = jp.asarray(
            [
                list(
                    range(
                        self._mj_model.sensor_adr[
                            self._mj_model.sensor(f"{side}_palm_global_linvel").id
                        ],
                        self._mj_model.sensor_adr[
                            self._mj_model.sensor(f"{side}_palm_global_linvel").id
                        ]
                        + 3,
                    )
                )
                for side in ("left", "right")
            ]
        )
        self._body_masses = jp.asarray(self._mj_model.body_mass)
        self._total_mass = jp.sum(self._body_masses)
        self._torso_id = self._mj_model.body("torso_link").id
        actuator_joint_ids = self._mj_model.actuator_trnid[:, 0]
        self._actuator_dof_ids = jp.asarray(
            self._mj_model.jnt_dofadr[actuator_joint_ids]
        )
        self._arm_actuator_ids = jp.asarray(
            [
                self._mj_model.actuator(f"{side}_{joint}_joint").id
                for side in ("left", "right")
                for joint in (
                    "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                    "wrist_roll", "wrist_pitch", "wrist_yaw",
                )
            ]
        )
        self._wrist_actuator_ids = jp.asarray(
            [
                self._mj_model.actuator(f"{side}_wrist_{axis}_joint").id
                for side in ("left", "right")
                for axis in ("roll", "pitch", "yaw")
            ]
        )
        self._joint_ranges = jp.asarray(self._uppers - self._lowers)

    def sample_command(self, rng: jax.Array) -> jax.Array:
        del rng
        return jp.zeros(3)

    def reset(self, rng: jax.Array) -> mjx_env.State:
        # Upstream initializes the exact actor/critic structures and metrics.
        state = super().reset(rng)
        rng, pos_rng, yaw_rng, joint_rng, vel_rng, drop_rng = jax.random.split(
            rng, 6
        )
        qpos = self._init_q
        uv = jax.random.uniform(
            pos_rng,
            (2,),
            minval=-self.experiment.reset.position_jitter_m,
            maxval=self.experiment.reset.position_jitter_m,
        )
        plane_point = uv[0] * self._ramp_tangent + uv[1] * self._ramp_cross
        drop_height = jax.random.uniform(
            drop_rng,
            (),
            minval=self.experiment.reset.drop_height_min_m,
            maxval=self.experiment.reset.drop_height_max_m,
        )
        qpos = qpos.at[:3].set(
            plane_point
            + (self._init_q[2] + drop_height) * self._ramp_normal
        )
        joint_noise = jax.random.uniform(
            joint_rng,
            (29,),
            minval=-self.experiment.reset.joint_jitter_rad,
            maxval=self.experiment.reset.joint_jitter_rad,
        )
        qpos = qpos.at[7:].set(
            jp.clip(self._reference_pose + joint_noise, self._lowers, self._uppers)
        )
        limit = math.radians(self.experiment.reset.yaw_jitter_degrees)
        yaw = jax.random.uniform(yaw_rng, (), minval=-limit, maxval=limit)
        from mujoco.mjx._src import math as mjx_math

        yaw_quat = mjx_math.axis_angle_to_quat(jp.array([0.0, 0.0, 1.0]), yaw)
        local_quat = mjx_math.quat_mul(yaw_quat, self._nominal_root_quat)
        qpos = qpos.at[3:7].set(mjx_math.quat_mul(self._ramp_quat, local_quat))
        qvel = jax.random.uniform(
            vel_rng,
            (self.mjx_model.nv,),
            minval=-self.experiment.reset.velocity_jitter,
            maxval=self.experiment.reset.velocity_jitter,
        )
        data = mjx_env.make_data(
            self.mj_model,
            qpos=qpos,
            qvel=qvel,
            ctrl=self._default_pose,
            impl=self.mjx_model.impl.value,
            naconmax=self._config.naconmax,
            njmax=self._config.njmax,
        )
        data = mjx.forward(self.mjx_model, data)
        foot_contact = self._foot_contacts(data)
        com = self._com_position(data)
        state.info.update(
            rng=rng,
            command=jp.zeros(3),
            phase_dt=jp.zeros(1),
            last_com_position=com,
            last_com_velocity=jp.zeros(3),
            start_com_position=com,
            reset_drop_height_m=drop_height,
        )
        obs = self._get_obs(data, state.info, foot_contact)
        state = state.replace(
            data=data, obs=obs, reward=jp.zeros(()), done=jp.zeros(())
        )
        return self._set_diagnostics(state)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        next_state = super().step(state, action)
        com = self._com_position(next_state.data)
        velocity = (com - next_state.info["last_com_position"]) / self.dt
        next_state.info["last_com_position"] = com
        next_state.info["last_com_velocity"] = velocity
        next_state.info["command"] = jp.zeros(3)
        next_state.info["phase_dt"] = jp.zeros(1)
        return self._set_diagnostics(next_state)

    def _foot_contacts(self, data: mjx.Data) -> jax.Array:
        return jp.asarray(
            [
                data.sensordata[self._mj_model.sensor_adr[sensor_id]] > 0
                for sensor_id in self._feet_floor_found_sensor
            ]
        )

    def _hand_contacts(self, data: mjx.Data) -> jax.Array:
        return jp.asarray(
            [
                data.sensordata[self._mj_model.sensor_adr[sensor_id]] > 0
                for sensor_id in self._hand_sensor_ids
            ]
        )

    def _sensor_vectors(self, data: mjx.Data, suffix: str) -> jax.Array:
        return jp.stack(
            [
                mjx_env.get_sensor_data(self.mj_model, data, f"{side}_{suffix}")
                for side in ("left", "right")
            ]
        )

    def _contact_forces(self, data: mjx.Data) -> tuple[jax.Array, jax.Array]:
        def terrain_normal_force(suffix: str, site_ids: jax.Array) -> jax.Array:
            local_force = self._sensor_vectors(data, suffix)
            rotation = data.site_xmat[site_ids].reshape((-1, 3, 3))
            world_force = jp.einsum("nij,nj->ni", rotation, local_force)
            return jp.abs(world_force @ self._ramp_normal)

        feet = terrain_normal_force("foot_force", self._feet_site_id)
        hands = terrain_normal_force("hand_force", self._hands_site_id)
        return feet, hands

    def _com_position(self, data: mjx.Data) -> jax.Array:
        return jp.sum(data.xipos * self._body_masses[:, None], axis=0) / self._total_mass

    def _get_obs(
        self, data: mjx.Data, info: dict[str, Any], contact: jax.Array
    ) -> mjx_env.Observation:
        obs = super()._get_obs(data, info, contact)
        # Keep a stable critic ABI across stages so the actor normalizer can be
        # restored safely.  Stage I reserves these slots as zeros; Stage II
        # activates the training-only terrain descriptor and resets the critic.
        descriptor = jp.where(
            self.experiment.stage == "posture-adapter",
            terrain_descriptor(self._slope_radians, 0.0, facing_uphill=True),
            jp.zeros(5),
        )
        feet, hands = self._contact_forces(data)
        privileged = jp.hstack(
            [
                obs["privileged_state"],
                descriptor,
                self._hand_contacts(data).astype(jp.float32),
                feet / 500.0,
                hands / 500.0,
                self._sensor_vectors(data, "hand_torque").ravel() / 25.0,
            ]
        )
        return {"state": obs["state"], "privileged_state": privileged}

    def _get_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
        metrics: dict[str, Any],
        done: jax.Array,
        first_contact: jax.Array,
        contact: jax.Array,
    ) -> dict[str, jax.Array]:
        del first_contact, contact
        feet_force, hands_force = self._contact_forces(data)
        foot_contact = self._foot_contacts(data)
        hand_contact = self._hand_contacts(data)
        positions = jp.vstack(
            [data.site_xpos[self._feet_site_id], data.site_xpos[self._hands_site_id]]
        )
        forces = jp.hstack([feet_force, hands_force])
        active_contacts = jp.hstack([foot_contact, hand_contact])
        support = force_weighted_support_anchor(
            positions,
            forces,
            active_contacts,
            epsilon=self.experiment.contact.support_epsilon_n,
        )
        com = self._com_position(data)
        com_velocity = (com - info["last_com_position"]) / self.dt
        com_acceleration = (com_velocity - info["last_com_velocity"]) / self.dt
        zmp = terrain_aligned_zmp(
            com,
            com_acceleration,
            support,
            self._ramp_normal,
            denominator_epsilon=self.experiment.contact.zmp_denominator_epsilon,
        )
        metrics["validation/zmp_deviation_m"] = jp.linalg.norm(zmp - support)
        torso_z = data.xmat[self._torso_id].reshape((3, 3))[:, 2]
        orientation_error = jp.sum(jp.square(torso_z - self._ramp_tangent))
        displacement = com - info["start_com_position"]
        drift = jp.square(jp.dot(displacement, self._ramp_tangent)) + jp.square(
            jp.dot(displacement, self._ramp_cross)
        )
        foot_velocity = data.sensordata[self._foot_linvel_sensor_adr]
        hand_velocity = data.sensordata[self._hand_velocity_adr]
        foot_tangent = foot_velocity - (
            foot_velocity @ self._ramp_normal
        )[:, None] * self._ramp_normal
        hand_tangent = hand_velocity - (
            hand_velocity @ self._ramp_normal
        )[:, None] * self._ramp_normal
        energy = jp.sum(
            jp.abs(data.actuator_force * data.qvel[self._actuator_dof_ids])
        )
        target_height = slope_conditioned_com_height(
            self.experiment.reset.nominal_com_height_m,
            self._slope_radians,
            maximum_slope_radians=math.radians(30.0),
            facing_uphill=True,
        )
        height = jp.dot(com, self._ramp_normal)
        masked_forces = jp.where(active_contacts, forces, 0.0)
        load_share = masked_forces / (jp.sum(masked_forces) + 1.0e-6)
        root_height = jp.dot(data.qpos[:3], self._ramp_normal)
        pose_error = (data.qpos[7:] - self._reference_pose) / self._joint_ranges
        wrist_moment = jp.linalg.norm(
            self._sensor_vectors(data, "hand_torque"), axis=-1
        )
        arm_force = jp.sum(jp.abs(data.actuator_force[self._arm_actuator_ids]))
        return {
            "alive": jp.ones(()),
            "terrain_zmp": zmp_reward(
                zmp, support, sigma=self.experiment.contact.zmp_sigma_m
            ),
            "orientation": jp.exp(-orientation_error / 0.25),
            "root_height": jp.exp(
                -jp.square(
                    (root_height - self._init_q[2])
                    / self.experiment.reset.root_height_sigma_m
                )
            ),
            "drift": drift,
            "hand_slip": jp.sum(
                jp.linalg.norm(hand_tangent, axis=-1) * hand_contact
            ),
            "foot_slip": jp.sum(
                jp.linalg.norm(foot_tangent, axis=-1) * foot_contact
            ),
            "action_magnitude": jp.sum(jp.square(action)),
            "action_rate": jp.sum(jp.square(action - info["last_act"])),
            "pose_deviation": jp.mean(jp.square(pose_error)),
            "energy": energy,
            "termination": done,
            "com_height": jp.exp(
                -jp.square(
                    (height - target_height)
                    / self.experiment.reset.com_height_sigma_m
                )
            ),
            "load_balance": jp.exp(-jp.var(load_share) / 0.02),
            "wrist_moment": jp.sum(jp.square(wrist_moment / 25.0)),
            "arm_loading": arm_force / 1000.0,
        }

    def _get_termination(self, data: mjx.Data) -> jax.Array:
        torso_z = data.xmat[self._torso_id].reshape((3, 3))[:, 2]
        alignment = jp.dot(torso_z, self._ramp_tangent)
        root_height = jp.dot(data.qpos[:3], self._ramp_normal)
        forbidden = jp.any(data.sensordata[self._forbidden_sensor_adr] > 0)
        nonfinite = ~jp.all(jp.isfinite(data.qpos)) | ~jp.all(jp.isfinite(data.qvel))
        return (
            (alignment < 0.25)
            | (root_height < self.experiment.reset.minimum_root_height_m)
            | forbidden
            | nonfinite
        )

    def _set_diagnostics(self, state: mjx_env.State) -> mjx_env.State:
        data = state.data
        feet, hands = self._contact_forces(data)
        hand_contact = self._hand_contacts(data)
        foot_contact = self._foot_contacts(data)
        com = self._com_position(data)
        displacement = com - state.info["start_com_position"]
        drift = jp.linalg.norm(
            jp.asarray(
                [
                    jp.dot(displacement, self._ramp_tangent),
                    jp.dot(displacement, self._ramp_cross),
                ]
            )
        )
        state.metrics["validation/drift_m"] = drift
        state.metrics["validation/com_height_m"] = jp.dot(com, self._ramp_normal)
        state.metrics["validation/speed_mps"] = jp.linalg.norm(data.qvel[:2])
        state.metrics["validation/hand_contact_ratio"] = jp.mean(
            hand_contact.astype(jp.float32)
        )
        state.metrics["validation/foot_contact_ratio"] = jp.mean(
            foot_contact.astype(jp.float32)
        )
        state.metrics["validation/peak_hand_force_n"] = jp.max(hands)
        state.metrics["validation/peak_foot_force_n"] = jp.max(feet)
        state.metrics["validation/peak_wrist_moment_nm"] = jp.max(
            jp.linalg.norm(self._sensor_vectors(data, "hand_torque"), axis=-1)
        )
        state.metrics["validation/prohibited_body_contact"] = jp.any(
            data.sensordata[self._forbidden_sensor_adr] > 0
        ).astype(jp.float32)
        state.metrics["validation/nonfinite"] = (
            ~jp.all(jp.isfinite(data.qpos)) | ~jp.all(jp.isfinite(data.qvel))
        ).astype(jp.float32)
        state.metrics["validation/actuator_saturation_ratio"] = jp.max(
            jp.abs(data.actuator_force)
            / jp.maximum(jp.abs(self.mj_model.actuator_forcerange[:, 1]), 1.0)
        )
        return state


def make_env(
    *,
    stage: str = "balance-prior",
    slope_degrees: float = 0.0,
    implementation: str = "jax",
) -> FourContactBalanceEnv:
    config = ExperimentConfig(
        stage=stage, slope_degrees=float(slope_degrees), implementation=implementation
    )
    return FourContactBalanceEnv(config)
