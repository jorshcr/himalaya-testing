"""Single command line interface for the canonical Himalaya workflow."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

from .audit import audit_model, render_collision_audit, write_audit
from .config import ContactConfig, ExperimentConfig
from .environment import FourContactBalanceEnv
from .evaluation import evaluate, write_evaluation
from .provenance import checkpoint_digest, write_run_manifest
from .rendering import render_policy
from .training import load_policy, train


STAGES = ("balance-prior", "posture-adapter")
SLOPES = (0.0, 5.0, 10.0, 15.0, 30.0)


def _config(args, *, sensitivity: bool = False) -> ExperimentConfig:
    config = ExperimentConfig(
        stage=args.stage,
        slope_degrees=float(args.slope),
        implementation=getattr(args, "implementation", "jax"),
    )
    if sensitivity:
        scale = config.contact.sensitivity_scale
        config = replace(
            config,
            contact=replace(
                config.contact,
                hand_sliding_friction=config.contact.hand_sliding_friction * scale,
                foot_sliding_friction=config.contact.foot_sliding_friction * scale,
            ),
        )
    return config


def _audit(args) -> int:
    config = _config(args)
    report = audit_model(config, settle_seconds=args.settle_seconds)
    output = Path(args.output)
    write_audit(report, output)
    render_collision_audit(config, output.with_suffix(".png"))
    print(json.dumps(report.__dict__, indent=2, default=list))
    return 0 if report.passed else 2


def _train(args) -> int:
    config = _config(args)
    output = Path(args.output).resolve()
    write_run_manifest(
        output,
        config,
        command=sys.argv,
        extra={"status": "training", "restore": args.restore},
    )

    def progress(step, metrics):
        print(f"step={step} reward={float(metrics.get('eval/episode_reward', 0.0)):.4f}", flush=True)

    train(
        config,
        output,
        restore=Path(args.restore).resolve() if args.restore else None,
        timesteps=args.timesteps,
        num_envs=args.num_envs,
        progress_fn=progress,
    )
    return 0


def _evaluate(args) -> int:
    sensitivity = bool(args.sensitivity)
    config = _config(args, sensitivity=sensitivity)
    env = FourContactBalanceEnv(config)
    policy = load_policy(config, Path(args.checkpoint).resolve())
    scale = config.contact.sensitivity_scale if sensitivity else 1.0
    report = evaluate(env, policy, config, traction_scale=scale)
    output = Path(args.output).resolve()
    write_evaluation(report, output)
    write_run_manifest(
        output.parent,
        config,
        command=sys.argv,
        checkpoint=Path(args.checkpoint).resolve(),
        extra={"evaluation": report.__dict__},
    )
    if report.promotion_passed and not sensitivity:
        stage_result = {
            "schema_version": 1,
            "stage": config.stage,
            "slope_degrees": config.slope_degrees,
            "promotion_passed": True,
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "checkpoint_sha256": checkpoint_digest(Path(args.checkpoint).resolve()),
            "evaluation": str(output),
        }
        (output.parent / "stage_result.json").write_text(
            json.dumps(stage_result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report.__dict__, indent=2))
    return 0 if report.promotion_passed or sensitivity else 2


def _render(args) -> int:
    config = _config(args)
    policy = load_policy(config, Path(args.checkpoint).resolve())
    path = render_policy(config, policy, Path(args.output), seconds=args.seconds)
    print(path)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="himalaya", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(item):
        item.add_argument("--stage", choices=STAGES, default="balance-prior")
        item.add_argument("--slope", type=float, choices=SLOPES, default=0.0)
        item.add_argument("--implementation", choices=("jax", "warp"), default="jax")

    audit = sub.add_parser("audit", help="run pre-training model/reset gates")
    common(audit)
    audit.add_argument("--settle-seconds", type=float, default=0.25)
    audit.add_argument("--output", default="runs/audit/audit.json")
    audit.set_defaults(func=_audit)

    training = sub.add_parser("train", help="train one explicit stage")
    common(training)
    training.add_argument("--output", required=True)
    training.add_argument("--restore")
    training.add_argument("--timesteps", type=int)
    training.add_argument("--num-envs", type=int)
    training.set_defaults(func=_train)

    evaluation = sub.add_parser("evaluate", help="run fixed-seed robust-hold evaluation")
    common(evaluation)
    evaluation.add_argument("--checkpoint", required=True)
    evaluation.add_argument("--output", required=True)
    evaluation.add_argument("--sensitivity", action="store_true")
    evaluation.set_defaults(func=_evaluate)

    rendering = sub.add_parser("render", help="render a checkpoint-bound rollout")
    common(rendering)
    rendering.add_argument("--checkpoint", required=True)
    rendering.add_argument("--output", required=True)
    rendering.add_argument("--seconds", type=float, default=20.0)
    rendering.set_defaults(func=_render)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
