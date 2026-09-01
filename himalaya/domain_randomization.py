"""Vectorized dynamics randomization for the canonical Stage-I prior."""

from __future__ import annotations

import jax
import jax.numpy as jp

from .config import ExperimentConfig


def make_domain_randomizer(
    env,
    config: ExperimentConfig,
    num_envs: int,
    *,
    slope_degrees: tuple[float, ...] | None = None,
):
    """Build a deterministic, stratified multi-slope MJX randomizer.

    Tilted gravity over a level local support plane is the rigid-frame
    equivalent of world-vertical gravity over an inclined plane. A single
    vectorized batch can cover every reviewed slope without privileged labels.
    """

    settings = config.domain_randomization
    bootstrap = (
        config.stage == "posture-adapter"
        and config.wave_gait.is_bootstrap(config.slope_degrees)
    )
    bootstrap_fraction = config.wave_gait.bootstrap_randomization_fraction
    bootstrap_bounds = (1.0 - bootstrap_fraction, 1.0 + bootstrap_fraction)

    def effective_bounds(bounds):
        return bootstrap_bounds if bootstrap else bounds

    rng = jax.random.PRNGKey(config.ppo.seed + 17_029)
    keys = jax.random.split(rng, num_envs)
    choices = jp.asarray(slope_degrees or settings.slope_degrees)
    slope_indices = jax.random.permutation(rng, jp.arange(num_envs)) % choices.size
    slopes = choices[slope_indices]
    hand_pairs = jp.asarray(
        [env.mj_model.pair(f"{side}_hand_floor").id for side in ("left", "right")]
    )
    foot_pairs = jp.asarray(
        [env.mj_model.pair(f"{side}_foot_floor").id for side in ("left", "right")]
    )

    def randomization_fn(model):
        def uniform(key, bounds, shape=()):
            return jax.random.uniform(
                key, shape=shape, minval=float(bounds[0]), maxval=float(bounds[1])
            )

        @jax.vmap
        def randomize_one(key, slope_degrees):
            split = jax.random.split(key, 7)
            gravity_magnitude = 9.81 * uniform(
                split[0], effective_bounds(settings.gravity_scale_range)
            )
            slope = jp.deg2rad(slope_degrees)
            gravity = jp.asarray(
                [
                    -gravity_magnitude * jp.sin(slope),
                    0.0,
                    -gravity_magnitude * jp.cos(slope),
                ]
            )
            pair_friction = model.pair_friction
            pair_friction = pair_friction.at[hand_pairs, 0].multiply(
                uniform(
                    split[1], effective_bounds(settings.hand_friction_scale_range)
                )
            )
            pair_friction = pair_friction.at[foot_pairs, 0].multiply(
                uniform(
                    split[2], effective_bounds(settings.foot_friction_scale_range)
                )
            )
            mass_scale = uniform(
                split[3],
                effective_bounds(settings.link_mass_scale_range),
                shape=(model.nbody,),
            )
            body_mass = model.body_mass * mass_scale
            body_inertia = model.body_inertia * mass_scale[:, None]
            dof_frictionloss = model.dof_frictionloss.at[6:].set(
                model.dof_frictionloss[6:]
                * uniform(
                    split[4],
                    effective_bounds(settings.joint_friction_scale_range),
                    shape=(29,),
                )
            )
            dof_damping = model.dof_damping.at[6:].set(
                model.dof_damping[6:]
                * uniform(
                    split[5],
                    effective_bounds(settings.joint_damping_scale_range),
                    shape=(29,),
                )
            )
            dof_armature = model.dof_armature.at[6:].set(
                model.dof_armature[6:]
                * uniform(
                    split[6],
                    effective_bounds(settings.armature_scale_range),
                    shape=(29,),
                )
            )
            return (
                gravity,
                pair_friction,
                body_mass,
                body_inertia,
                dof_frictionloss,
                dof_damping,
                dof_armature,
            )

        values = randomize_one(keys, slopes)
        fields = (
            "opt.gravity",
            "pair_friction",
            "body_mass",
            "body_inertia",
            "dof_frictionloss",
            "dof_damping",
            "dof_armature",
        )
        randomized = model.tree_replace(dict(zip(fields, values, strict=True)))
        in_axes = jax.tree_util.tree_map(lambda _: None, model)
        in_axes = in_axes.tree_replace({field: 0 for field in fields})
        return randomized, in_axes

    return randomization_fn
