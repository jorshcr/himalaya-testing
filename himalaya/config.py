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
    drop_height_min_m: float = 0.25
    drop_height_max_m: float = 0.35
    minimum_root_height_m: float = 0.20
    nominal_com_height_m: float = 0.225
    com_height_sigma_m: float = 0.04
    root_height_sigma_m: float = 0.05
    uphill_com_offset_m: float = 0.015
    downhill_com_offset_m: float = -0.010
    collapsed_com_margin_m: float = 0.025


@dataclass(frozen=True)
class DomainRandomizationConfig:
    """Stage-I robustness envelope around the nominal simulation model."""

    enabled: bool = True
    slope_degrees: tuple[float, ...] = (0.0, 5.0, 10.0, 15.0, 30.0)
    gravity_scale_range: tuple[float, float] = (0.95, 1.05)
    hand_friction_scale_range: tuple[float, float] = (0.75, 1.20)
    foot_friction_scale_range: tuple[float, float] = (0.75, 1.20)
    link_mass_scale_range: tuple[float, float] = (0.80, 1.20)
    joint_friction_scale_range: tuple[float, float] = (0.30, 2.50)
    joint_damping_scale_range: tuple[float, float] = (0.70, 1.30)
    armature_scale_range: tuple[float, float] = (0.80, 1.20)
    sensor_noise_level: float = 1.5
    push_interval_seconds: tuple[float, float] = (2.5, 6.0)
    push_magnitude_mps: tuple[float, float] = (0.15, 0.75)


@dataclass(frozen=True)
class LocomotionConfig:
    """Command and biomechanical priors used only by locomotion Stage II."""

    tracking_sigma_mps: float = 0.20
    progress_normalizer_mps: float = 0.80
    course_length_m: float = 20.0
    course_width_m: float = 3.0
    course_margin_m: float = 1.0
    episode_length: int = 2000
    progress_deadline_slack_fraction: float = 0.10
    swing_hip_beta0_rad: float = -1.05
    swing_hip_beta1_rad_per_rad: float = 0.50
    swing_hip_clip_degrees: float = 30.0
    swing_hip_sigma_rad: float = 0.25
    hip_power_scale_w: float = 100.0


@dataclass(frozen=True)
class WaveGaitConfig:
    """Soft four-limb crawl clock and promotion curriculum.

    The clock activates the four phase values already present in the upstream
    actor observation.  It never adds actor or critic inputs.
    """

    enabled: bool = True
    sequence: tuple[str, ...] = (
        "left_hand",
        "right_foot",
        "right_hand",
        "left_foot",
    )
    curriculum_degrees: tuple[float, ...] = (0.0, 5.0, 10.0, 15.0, 20.0, 30.0)
    bootstrap_degrees: tuple[float, ...] = (0.0, 5.0)
    cycle_frequency_hz: float = 0.50
    swing_window_fraction: float = 0.24
    landing_start_fraction: float = 0.65
    swing_clearance_m: float = 0.055
    swing_clearance_sigma_m: float = 0.025
    forward_placement_target_m: float = 0.10
    forward_placement_sigma_m: float = 0.06
    minimum_recontact_advance_m: float = 0.025
    stance_velocity_sigma_mps: float = 0.08
    shoulder_pitch_beta0_rad: float = -0.45
    shoulder_pitch_beta1_rad_per_rad: float = 0.35
    shoulder_pitch_sigma_rad: float = 0.25
    bootstrap_randomization_fraction: float = 0.02
    bootstrap_sensor_noise_level: float = 0.05
    bootstrap_reset_jitter_scale: float = 0.10
    bootstrap_drop_height_m: float = 0.0
    no_progress_seconds: float = 2.0
    no_progress_delta_m: float = 0.04
    minimum_progress_speed_fraction: float = 0.50
    minimum_progress_speed_floor_mps: float = 0.04
    command_speed_ranges_mps: tuple[tuple[float, float, float], ...] = (
        (0.0, 0.08, 0.25),
        (5.0, 0.08, 0.25),
        (10.0, 0.10, 0.30),
        (15.0, 0.12, 0.35),
        (20.0, 0.14, 0.40),
        (30.0, 0.16, 0.45),
    )
    # Phase guidance is deliberately strongest during bootstrap and anneals as
    # the terrain curriculum becomes harder, leaving the gait soft at 30 deg.
    reward_scales_by_slope: tuple[tuple[float, float], ...] = (
        (0.0, 1.00),
        (5.0, 0.90),
        (10.0, 0.75),
        (15.0, 0.60),
        (20.0, 0.45),
        (30.0, 0.30),
    )
    minimum_promotion_progress_m: tuple[tuple[float, float], ...] = (
        (0.0, 0.50),
        (5.0, 0.75),
        (10.0, 1.00),
        (15.0, 1.50),
        (20.0, 2.00),
        (30.0, 3.00),
    )

    def is_bootstrap(self, slope_degrees: float) -> bool:
        return float(slope_degrees) in self.bootstrap_degrees

    def command_range(self, slope_degrees: float) -> tuple[float, float]:
        slope = float(slope_degrees)
        for level, minimum, maximum in self.command_speed_ranges_mps:
            if level == slope:
                return minimum, maximum
        raise ValueError(f"missing wave-gait command range for {slope:g} degrees")

    def reward_scale(self, slope_degrees: float) -> float:
        slope = float(slope_degrees)
        for level, scale in self.reward_scales_by_slope:
            if level == slope:
                return scale
        raise ValueError(f"missing wave-gait reward scale for {slope:g} degrees")

    def promotion_progress(self, slope_degrees: float) -> float:
        slope = float(slope_degrees)
        for level, progress in self.minimum_promotion_progress_m:
            if level == slope:
                return progress
        raise ValueError(f"missing wave-gait promotion gate for {slope:g} degrees")


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


@dataclass(frozen=True)
class StageIIRewardConfig:
    """Locomotion adaptation of HumoSlope's descriptor-gated BSGA objective."""

    alive: float = 0.5
    terrain_zmp: float = 0.5
    tracking_forward_velocity: float = 8.0
    uphill_progress: float = 6.0
    course_progress: float = 3.0
    course_completion: float = 15.0
    progress_deficit: float = -8.0
    stagnation: float = -5.0
    lateral_velocity: float = -1.0
    yaw_rate: float = -0.25
    terrain_posture: float = 0.75
    root_height: float = 0.5
    slope_com_height: float = 1.0
    collapsed_com: float = -1.0
    load_balance: float = 0.1
    drift: float = -1.5
    hip_propulsion: float = 0.5
    swing_hip_guidance: float = 0.5
    swing_shoulder_guidance: float = 0.5
    wave_swing_clearance: float = 1.5
    wave_forward_placement: float = 1.25
    wave_recontact_ahead: float = 2.0
    wave_stance_stability: float = 1.0
    wave_backward_placement: float = -1.0
    wave_missed_swing_window: float = -1.5
    wave_stance_slip: float = -1.0
    hand_slip: float = -0.25
    foot_slip: float = -0.25
    action_magnitude: float = -0.02
    action_rate: float = -0.03
    pose_deviation: float = -0.1
    wrist_moment: float = -0.1
    arm_loading: float = -0.15
    energy: float = -1.0e-4
    termination: float = -5000.0


@dataclass(frozen=True)
class PPOConfig:
    timesteps_per_stage: int = 200_000_000
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
    curriculum_degrees: tuple[float, ...] = (0.0, 5.0, 10.0, 15.0, 20.0, 30.0)
    implementation: Literal["jax", "warp"] = "jax"
    action_scale: float = 0.25
    episode_length: int = 1000
    upstream: UpstreamConfig = field(default_factory=UpstreamConfig)
    contact: ContactConfig = field(default_factory=ContactConfig)
    reset: ResetConfig = field(default_factory=ResetConfig)
    domain_randomization: DomainRandomizationConfig = field(
        default_factory=DomainRandomizationConfig
    )
    locomotion: LocomotionConfig = field(default_factory=LocomotionConfig)
    wave_gait: WaveGaitConfig = field(default_factory=WaveGaitConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    stage2_reward: StageIIRewardConfig = field(default_factory=StageIIRewardConfig)
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
        if self.reset.drop_height_min_m < 0:
            raise ValueError("minimum drop height must be non-negative")
        if self.reset.drop_height_max_m < self.reset.drop_height_min_m:
            raise ValueError("maximum drop height must be at least the minimum")
        if not self.domain_randomization.slope_degrees:
            raise ValueError("domain randomization requires at least one slope")
        if any(
            slope not in self.curriculum_degrees
            for slope in self.domain_randomization.slope_degrees
        ):
            raise ValueError("domain-randomized slopes must use the reviewed curriculum")
        if not 0 < self.evaluation.minimum_success_rate <= 1:
            raise ValueError("minimum success rate must be in (0, 1]")
        if self.wave_gait.sequence != (
            "left_hand",
            "right_foot",
            "right_hand",
            "left_foot",
        ):
            raise ValueError("wave-gait sequence must be LH, RF, RH, LF")
        if self.wave_gait.curriculum_degrees != self.curriculum_degrees:
            raise ValueError("wave-gait and experiment curricula must match")
        if not 0 < self.wave_gait.swing_window_fraction <= 0.25:
            raise ValueError("wave-gait swing window must be in (0, 0.25]")
        if not 0 < self.wave_gait.landing_start_fraction < 1:
            raise ValueError("wave-gait landing fraction must be in (0, 1)")
        if self.wave_gait.cycle_frequency_hz <= 0:
            raise ValueError("wave-gait cycle frequency must be positive")
        if self.wave_gait.no_progress_seconds <= 0:
            raise ValueError("no-progress truncation window must be positive")
        if self.wave_gait.no_progress_delta_m <= 0:
            raise ValueError("no-progress delta must be positive")
        if not 0 <= self.wave_gait.bootstrap_randomization_fraction < 1:
            raise ValueError("bootstrap randomization fraction must be in [0, 1)")
        if self.wave_gait.minimum_progress_speed_floor_mps <= 0:
            raise ValueError("minimum progress speed floor must be positive")
        if not 0 < self.wave_gait.minimum_progress_speed_fraction <= 1:
            raise ValueError("minimum progress speed fraction must be in (0, 1]")
        for slope in self.curriculum_degrees:
            minimum, maximum = self.wave_gait.command_range(slope)
            if not 0 <= minimum <= maximum:
                raise ValueError("wave-gait command ranges must be ordered")
            if not 0 <= self.wave_gait.reward_scale(slope) <= 1:
                raise ValueError("wave-gait reward scale must be in [0, 1]")
            if self.wave_gait.promotion_progress(slope) <= 0:
                raise ValueError("wave-gait promotion progress must be positive")

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
        cfg.episode_length = (
            config.locomotion.episode_length
            if config.stage == "posture-adapter"
            else config.episode_length
        )
        cfg.action_scale = config.action_scale
        cfg.restricted_joint_range = True
        bootstrap = (
            config.stage == "posture-adapter"
            and config.wave_gait.is_bootstrap(config.slope_degrees)
        )
        cfg.noise_config.level = (
            config.wave_gait.bootstrap_sensor_noise_level
            if bootstrap
            else config.domain_randomization.sensor_noise_level
        )
        cfg.push_config.enable = config.domain_randomization.enabled and not bootstrap
        cfg.push_config.interval_range = list(
            config.domain_randomization.push_interval_seconds
        )
        cfg.push_config.magnitude_range = list(
            config.domain_randomization.push_magnitude_mps
        )
        cfg.lin_vel_x = [0.0, 0.0]
        cfg.lin_vel_y = [0.0, 0.0]
        cfg.ang_vel_yaw = [0.0, 0.0]
        cfg.command_config.a = [0.0, 0.0, 0.0]
        cfg.command_config.b = [0.0, 0.0, 0.0]
        if config.stage == "posture-adapter":
            cfg.lin_vel_x = list(config.wave_gait.command_range(config.slope_degrees))
        cfg.njmax = 192
        cfg.naconmax = 64
        scales = cfg.reward_config.scales
        for name in list(scales.keys()):
            scales[name] = 0.0
        reward_names = set(asdict(config.reward)) | set(asdict(config.stage2_reward))
        for name in reward_names:
            scales[name] = 0.0
        selected_reward = (
            config.stage2_reward
            if config.stage == "posture-adapter"
            else config.reward
        )
        for name, value in asdict(selected_reward).items():
            scales[name] = value
    return cfg
