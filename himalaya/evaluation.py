"""Fixed-seed robust-hold evaluation and promotion contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jp
import numpy as np

from .config import ExperimentConfig
from .environment import FourContactBalanceEnv


@dataclass(frozen=True)
class EvaluationReport:
    trials: int
    successes: int
    success_rate: float
    falls_or_prohibited_contacts: int
    nonfinite_trials: int
    recovered_from_push: int
    mean_final_drift_m: float
    peak_hand_force_n: float
    peak_wrist_moment_nm: float
    promotion_passed: bool
    traction_scale: float


def evaluate(
    env: FourContactBalanceEnv,
    policy: Callable,
    config: ExperimentConfig,
    *,
    traction_scale: float = 1.0,
) -> EvaluationReport:
    # A sensitivity run receives a separately constructed, scaled config/env;
    # the canonical environment is never mutated in place.
    trials = config.evaluation.trials
    reset = jax.jit(jax.vmap(env.reset))
    step = jax.jit(jax.vmap(env.step))
    act = jax.jit(jax.vmap(policy))
    rng = jax.random.PRNGKey(config.ppo.seed)
    rng, reset_rng = jax.random.split(rng)
    state = reset(jax.random.split(reset_rng, trials))
    # Keep the rollout state and reductions on device.  The previous version
    # copied every metric to the host at every step, serializing 1000 GPU
    # dispatches and making the required 64-trial gate impractically slow.
    active = jp.ones(trials, dtype=bool)
    failed = jp.zeros(trials, dtype=bool)
    nonfinite = jp.zeros(trials, dtype=bool)
    recovered = jp.zeros(trials, dtype=bool)
    peak_hand = jp.zeros(trials)
    peak_wrist = jp.zeros(trials)
    final_drift = jp.zeros(trials)
    steps = round(config.evaluation.duration_seconds / env.dt)
    push_step = round(config.evaluation.push_time_seconds / env.dt)
    recovery_deadline = push_step + round(config.evaluation.recovery_seconds / env.dt)
    for index in range(steps):
        if index == push_step:
            qvel = state.data.qvel.at[:, 1].add(
                config.evaluation.push_delta_velocity_mps
            )
            state = state.replace(data=state.data.replace(qvel=qvel))
        rng, action_rng = jax.random.split(rng)
        action = act(state.obs, jax.random.split(action_rng, trials))[0]
        action = jp.where(active[:, None], action, jp.zeros_like(action))
        state = step(state, action)
        metrics = state.metrics
        done = state.done > 0.5
        prohibited = metrics["validation/prohibited_body_contact"] > 0.5
        bad_number = metrics["validation/nonfinite"] > 0.5
        failed |= active & (done | prohibited)
        nonfinite |= active & bad_number
        active &= ~(done | prohibited | bad_number)
        final_drift = metrics["validation/drift_m"]
        peak_hand = jp.maximum(peak_hand, metrics["validation/peak_hand_force_n"])
        peak_wrist = jp.maximum(peak_wrist, metrics["validation/peak_wrist_moment_nm"])
        if push_step < index <= recovery_deadline:
            recovered |= active & (
                (final_drift <= config.evaluation.maximum_drift_m)
                & (metrics["validation/speed_mps"] <= 0.10)
            )
    success = (
        active
        & recovered
        & (final_drift <= config.evaluation.maximum_drift_m)
        & ~nonfinite
    )
    success, failed, nonfinite, recovered, final_drift, peak_hand, peak_wrist = (
        np.asarray(value)
        for value in (
            success,
            failed,
            nonfinite,
            recovered,
            final_drift,
            peak_hand,
            peak_wrist,
        )
    )
    rate = float(np.mean(success))
    return EvaluationReport(
        trials=trials,
        successes=int(np.sum(success)),
        success_rate=rate,
        falls_or_prohibited_contacts=int(np.sum(failed)),
        nonfinite_trials=int(np.sum(nonfinite)),
        recovered_from_push=int(np.sum(recovered)),
        mean_final_drift_m=float(np.mean(final_drift)),
        peak_hand_force_n=float(np.max(peak_hand)),
        peak_wrist_moment_nm=float(np.max(peak_wrist)),
        promotion_passed=rate >= config.evaluation.minimum_success_rate,
        traction_scale=traction_scale,
    )


def write_evaluation(report: EvaluationReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
