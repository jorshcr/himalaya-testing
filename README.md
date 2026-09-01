# Himalaya G1 balance and soft wave-gait crawling

This is the canonical, simulation-only Himalaya repository. It trains the
29-DOF Unitree G1 first to hold a nominal all-fours posture on rough slopes,
then to crawl uphill with a soft four-limb wave rhythm.

The project uses the two-stage insight from
[HumoSlope](https://arxiv.org/html/2607.07830v1), but it is **not a HumoSlope
reproduction**. HumoSlope studies bipedal locomotion in Isaac Lab. This project
adapts its terrain-aligned balance prior and training-only posture conditioning
to multi-contact balance and crawling in MuJoCo Playground.

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
Stage-I training resets lift that pose by a uniformly sampled `0.25–0.35 m` along
the terrain normal (centered at `0.30 m`), so the policy must absorb randomized
landing impacts before holding balance.
The minimum-root-height termination gate is `0.20 m`; drift, prohibited-body
contact, orientation, non-finite state, and push recovery remain independent
acceptance criteria, so lowering this gate does not redefine collapse as success.

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
2. `posture-adapter`: warm-start the Stage-I actor, reset the critic, and train
   command-conditioned uphill locomotion with training-only terrain descriptors,
   slope-conditioned posture, hip/shoulder guidance, and a soft wave-gait clock.

The Stage-II objective follows HumoSlope's descriptor-gated structure. It adds
forward-command tracking and uphill progress to a slope-conditioned CoM target,
terrain-relative posture, stance-hip propulsion, phase-gated hip/shoulder and
clearance guidance, and upper-body feasibility regularizers. Stage I remains a
zero-command stationary balance prior.

Stage II activates the four phase values already reserved at the end of the
upstream 103-value actor observation; it adds no observation fields and keeps
the 233-value critic ABI. The clock follows `left hand → right foot → right
hand → left foot`, with a single soft swing window at a time. It rewards swing
clearance, forward placement, recontact ahead of the prior support, and quiet
loaded stance limbs. Backward placement, missed windows, and stance slip are
penalized. These are annealed preferences, not exact contact-count rewards,
binary gait constraints, or survival conditions.

The 0° and 5° bootstrap levels command `0.08–0.25 m/s`, use zero drop height,
disable pushes, reduce reset jitter to 10% of nominal, narrow dynamics
randomization to ±2%, and truncate an episode after two seconds without at
least 4 cm of new progress. Phase guidance then anneals from `1.0` at 0° to
`0.3` at 30°.

The heavy crawl curriculum uses a physical 20 m incline, resets near its lower
margin, and provides 40-second episodes. Forward tracking, instantaneous uphill
speed, normalized course progress, and course completion dominate the Stage-II
weights; balance terms remain safeguards rather than the primary objective.
The crawl stage also imposes a time-indexed progress schedule. Policies are
penalized both for falling behind the required course fraction and for moving
slower than half of the sampled command (with a `0.04 m/s` floor), preventing
a stable stationary solution from scoring well merely by surviving.

GPU launch hygiene is part of the reproducible run specification. Container
commands must follow the jobs CLI `--` argument separator, and remote Python/HF
commands run with an explicit UTF-8 locale and disabled progress bars. These
requirements prevent launcher option capture and locale-dependent downloader
failures before training begins.
Checkpoint inputs are fetched by the repository-owned
`scripts/download_checkpoint.py` helper using locked `httpx`, authenticated
retries, atomic replacement, and mandatory SHA-256 verification. This avoids
job-log terminal encoding failures and assumptions about external download
executables in the pinned image.
The launch manifest records the extracted checkpoint-content digest separately
from the compressed artifact digest; downloads are verified against the latter,
as published in `checkpoint.json` and Hugging Face LFS metadata.
On Windows, read remote job logs with local `PYTHONUTF8=1` and classify runs
from the authoritative job status. A log-reader encoding error is an observer
failure and must not trigger a training relaunch by itself.

Curriculum: `0° → 5° → 10° → 15° → 20° → 30°`. The initial 0° adapter
restores the Stage-I actor and resets its critic. Every later level restores the
promoted actor and critic from the immediately preceding slope. Promotion
requires survival plus measurable forward crawling (`0.5, 0.75, 1.0, 1.5,
2.0, 3.0 m` respectively), audit, quantitative evaluation, and reviewed video.

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

Stage II begins on level terrain from the reviewed Stage-I actor:

```powershell
himalaya train --stage posture-adapter --slope 0 `
  --restore runs/stage1-30/checkpoints/40000000 `
  --output runs/stage2-wave-0

himalaya evaluate --stage posture-adapter --slope 0 `
  --checkpoint runs/stage2-wave-0/checkpoints/40000000 `
  --output runs/stage2-wave-0/evaluation.json
```

## Evidence boundaries

- Model/reset audit is not learned balance.
- Reward improvement is not robust behavior.
- A video is qualitative evidence, not a promotion result.
- Simulation evidence is not physical feasibility or safety evidence.

Balance promotion requires at least 90% of 64 fixed trials to survive 20
seconds, remain within the terrain-frame drift bound, avoid prohibited body
contact and non-finite state, and recover from the deterministic push. Crawl
promotion replaces stationary push recovery with the configured forward
distance gate while retaining survival, lateral drift, prohibited-contact, and
finite-state requirements.
