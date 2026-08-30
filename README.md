# Himalaya G1 stationary four-contact balance

This is the canonical, simulation-only Himalaya repository. It trains the
29-DOF Unitree G1 to hold a nominal all-fours posture on rough slopes while
recovering from bounded pushes.

The project uses the two-stage insight from
[HumoSlope](https://arxiv.org/html/2607.07830v1), but it is **not a HumoSlope
reproduction**. HumoSlope studies bipedal locomotion in Isaac Lab. This project
adapts its terrain-aligned balance prior and training-only posture conditioning
to stationary multi-contact balance in MuJoCo Playground.

## Verified foundation

- MuJoCo Playground commit: `8a4b4642d8eba8a80ac99ed125cb62c16e1457ad`
- Menagerie commit expected by Playground: `1b86ece576591213e2b666ebf59508454200ca97`
- Upstream task: `G1JoystickRoughTerrain`
- Actions: 29 joint-position residual targets
- Actor: upstream proprioception only
- Critic/rewards/evaluation: may use terrain and simulated contact data

Upstream XML is never copied. `himalaya.model` reads the pinned model and
creates an in-memory overlay containing only the reviewed contact pairs,
fall-detection proxies, friction values, sensors, and all-fours keyframe.

The reset keyframe separates observed joint position from its load-bearing
position-actuator target. Its arms place the elbows and palms ahead/outboard
rather than folding the forearms toward the robot center. On level terrain the
zero-action reset passes a 20-second native MuJoCo settling gate without
central-body contact and with less than 1 mm audited hand penetration.
Training resets then lift that pose by a uniformly sampled `0.25–0.35 m` along
the terrain normal (centered at `0.30 m`), so the policy must absorb randomized
landing impacts before holding balance.

Canonical Stage I is a 200-million-step robust balance prior. Every PPO batch
is stratified across equivalent 0°, 5°, 10°, 15°, and 30° inclines and varies
sliding friction, link mass/inertia, joint friction, damping, armature, gravity
magnitude, sensor noise, timed pushes, and the reset state.

## Microspike abstraction

Microspikes are represented only by nominal Coulomb sliding friction:

- hands: `0.9`
- feet: `1.0`

Stage-I dynamics randomization scales each nominal sliding coefficient by
`0.75–1.20`; torsional and rolling friction remain unchanged. These are
uncalibrated simulation assumptions. They do not establish physical
traction, attachment strength, wrist safety, or Sim-to-Real feasibility. A
separate non-promoting sensitivity evaluation uses 80% of both values.

## Two stages

1. `balance-prior`: zero-command balance with a four-contact,
   terrain-aligned apparent-force ZMP regularizer.
2. `posture-adapter`: warm-start the Stage-I actor, reset the critic, and
   activate training-only terrain descriptors plus soft CoM, torso, load, and
   wrist/arm posture priors.

Hip-propulsion, downhill knee-braking, and swing-leg priors are deliberately
absent because the present objective is stationary balance, not locomotion.

Curriculum: `0° → 5° → 10° → 15° → 30°`. Each stage must pass audit,
compiled smoke, quantitative evaluation, and reviewed video gates before the
next stage or larger compute budget.

## Commands

```powershell
uv sync --frozen --extra test

himalaya audit --slope 0 --output runs/audit-0/audit.json
himalaya train --stage balance-prior --slope 0 --output runs/stage1-0
himalaya evaluate --stage balance-prior --slope 0 `
  --checkpoint runs/stage1-0/checkpoints/40000000 `
  --output runs/stage1-0/evaluation.json
himalaya render --stage balance-prior --slope 0 `
  --checkpoint runs/stage1-0/checkpoints/40000000 `
  --output runs/stage1-0/rollout.mp4
```

`audit` writes a signed-clearance JSON report plus time-zero and settled PNGs
with collision proxies and contact forces visible. Positive terrain X is
explicitly audited as uphill; the heightfield geom carries the signed vertical
datum rather than misusing MuJoCo's solid-base thickness.

For a cheap wiring test, override both workload controls:

```powershell
himalaya train --stage balance-prior --slope 0 --output runs/smoke `
  --timesteps 2048 --num-envs 8
```

Stage II requires a promoted Stage-I checkpoint:

```powershell
himalaya train --stage posture-adapter --slope 30 `
  --restore runs/stage1-30/checkpoints/40000000 `
  --output runs/stage2-30
```

## Evidence boundaries

- Model/reset audit is not learned balance.
- Reward improvement is not robust behavior.
- A video is qualitative evidence, not a promotion result.
- Simulation evidence is not physical feasibility or safety evidence.

Final promotion requires at least 90% of 64 fixed trials to survive 20 seconds,
remain within the configured terrain-frame drift bound, avoid prohibited body
contact and non-finite state, and recover from the deterministic push.
