"""The single authoritative experiment configuration.

Values here are simulation assumptions.  In particular, the microspike
friction coefficients are not measurements and imply no physical capability.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal


PLAYGROUND_REVISION = "8a4b4642d8eba8a80ac99ed125cb62c16e1457ad"
MENAGERIE_REVISION = "1b86ece576591213e2b666ebf59508454200ca97"


@dataclass(frozen=True)
class UpstreamConfig:
    playground_revision: str = PLAYGROUND_REVISION
    menagerie_revision: str = MENAGERIE_REVISION
    environment: str = "G1JoystickRoughTerrain"
    robot: str = "unitree_g1/g1_29dof_rev_1_0"


@dataclass(frozen=True)
class ContactConfig:
    hand_sliding_friction: float = 0.9
    foot_sliding_friction: float = 1.0
    sensitivity_scale: float = 0.8
    max_hand_penetration_m: float = 0.001
    support_epsilon_n: float = 1.0e-3
    zmp_denominator_epsilon: float = 1.0e-5
    zmp_sigma_m: float = 0.12
    hand_contact_time_constant_s: float = 0.005


@dataclass(frozen=True)
class ResetConfig:
    keyframe: str = "four_contact_home"
    position_jitter_m: float = 0.04
    yaw_jitter_degrees: float = 3.0
    joint_jitter_rad: float = 0.01
    velocity_jitter: float = 0.02
    minimum_root_height_m: float = 0.24
    nominal_com_height_m: float = 0.225
    com_height_sigma_m: float = 0.04
    root_height_sigma_m: float = 0.05


@dataclass(frozen=True)
class RewardConfig:
    alive: float = 0.5
    terrain_zmp: float = 2.0
    orientation: float = 1.0
    root_height: float = 2.0
    com_height: float = 1.0
    drift: float = -4.0
    hand_slip: float = -0.5
    foot_slip: float = -0.5
    action_magnitude: float = -0.02
    action_rate: float = -0.02
    pose_deviation: float = -0.2
    energy: float = -1.0e-4
    # Upstream multiplies every component by ctrl_dt (0.02).  This scale
    # therefore produces a fixed -100 terminal cost instead of the old -2.
    termination: float = -5000.0
    # Stage-II-only soft posture terms.
    load_balance: float = 0.25
    wrist_moment: float = -0.05
    arm_loading: float = -0.1


@dataclass(frozen=True)
class PPOConfig:
    timesteps_per_stage: int = 40_000_000
    num_envs: int = 8192
    seed: int = 2026
    checkpoint_interval_steps: int = 25_000_000
    checkpoint_repo: str = "iteratehack/himalaya-stage1-checkpoints"
    discounting: float = 0.997
    unroll_length: int = 64
    batch_size: int = 256
    num_minibatches: int = 32
    num_updates_per_batch: int = 4
    learning_rate: float = 3.0e-4
    entropy_cost: float = 0.002


@dataclass(frozen=True)
class EvaluationConfig:
    trials: int = 64
    duration_seconds: float = 20.0
    minimum_success_rate: float = 0.90
    maximum_drift_m: float = 0.25
    push_time_seconds: float = 5.0
    push_delta_velocity_mps: float = 0.35
    recovery_seconds: float = 2.0


@dataclass(frozen=True)
class ExperimentConfig:
    """Complete contract for one Himalaya experiment."""

    stage: Literal["balance-prior", "posture-adapter"] = "balance-prior"
    slope_degrees: float = 0.0
    curriculum_degrees: tuple[float, ...] = (0.0, 5.0, 10.0, 15.0, 30.0)
    implementation: Literal["jax", "warp"] = "jax"
    action_scale: float = 0.25
    episode_length: int = 1000
    upstream: UpstreamConfig = field(default_factory=UpstreamConfig)
    contact: ContactConfig = field(default_factory=ContactConfig)
    reset: ResetConfig = field(default_factory=ResetConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def __post_init__(self) -> None:
        if self.slope_degrees not in self.curriculum_degrees:
            raise ValueError(
                f"slope must be one of {self.curriculum_degrees}, got {self.slope_degrees}"
            )
        if self.contact.hand_sliding_friction <= 0:
            raise ValueError("hand friction must be positive")
        if self.contact.foot_sliding_friction <= 0:
            raise ValueError("foot friction must be positive")
        if not 0 < self.evaluation.minimum_success_rate <= 1:
            raise ValueError("minimum success rate must be in (0, 1]")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def default_config(
    *,
    stage: Literal["balance-prior", "posture-adapter"] = "balance-prior",
    slope_degrees: float = 0.0,
) -> ExperimentConfig:
    return ExperimentConfig(stage=stage, slope_degrees=float(slope_degrees))


def to_playground_config(config: ExperimentConfig):
    """Derive the upstream ConfigDict; no experiment defaults live there."""

    from mujoco_playground._src.locomotion.g1 import joystick

    cfg = joystick.default_config()
    with cfg.unlocked():
        cfg.impl = config.implementation
        cfg.episode_length = config.episode_length
        cfg.action_scale = config.action_scale
        cfg.restricted_joint_range = True
        cfg.noise_config.level = 1.0
        cfg.push_config.enable = False  # Evaluation injects deterministic pushes.
        cfg.lin_vel_x = [0.0, 0.0]
        cfg.lin_vel_y = [0.0, 0.0]
        cfg.ang_vel_yaw = [0.0, 0.0]
        cfg.command_config.a = [0.0, 0.0, 0.0]
        cfg.command_config.b = [0.0, 0.0, 0.0]
        cfg.njmax = 192
        cfg.naconmax = 64
        scales = cfg.reward_config.scales
        for name in list(scales.keys()):
            scales[name] = 0.0
        for name, value in asdict(config.reward).items():
            scales[name] = value
        if config.stage == "balance-prior":
            scales.load_balance = 0.0
            scales.wrist_moment = 0.0
            scales.arm_loading = 0.0
    return cfg
