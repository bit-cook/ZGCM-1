# ZGCM-1 Reinforcement Learning

ZGCM-1 explores mixed reinforcement learning on mathematics, code, and
general-capability tasks using **Group Relative Policy Optimization (GRPO)**.
The recipe follows Section 4.2 of the technical report, and this directory
contains the exact configuration, reward implementation, and training entry
used for the final math GRPO stage of the released model, together with the
code and instruction-following reward implementations used by the mixed-RL
experiments. The training framework is
[AReaL](https://github.com/areal-project/AReaL)
v2.0.0; see [Environment](ENVIRONMENT.md).

## Data and rewards

Prompt difficulty is estimated using rollouts from the initial policy. Problems
that are already solved frequently are removed to concentrate training on
useful learning signals.

- Mathematics uses binary answer-correctness rewards.
- Code uses the fraction of executable tests passed.
- Instruction-following tasks use the fraction of explicit constraints
  satisfied.
- Invalid or truncated responses receive no positive reward.

The released `rewards/` package implements the mathematics, code, and
instruction-following rewards, plus the domain dispatcher used by the mixed-RL
experiments (`rewards/rewards.py`):

- **Mathematics** (`math_reward.py`): the visible answer is extracted after a
  closed `</think>` section (`completion.py`) and compared against the gold
  labels with a math-verify worker (`precision=6`, `timeout=8s`, bidirectional
  equivalence), returning a strictly binary `r_raw ∈ {0, 1}`.
- **Code** (`sandbox_client.py`): the visible answer's last fenced Python block
  is verified against content-hashed test assets inside an isolated execution
  sandbox, always reached over HTTP; the reward is the fraction of selected
  tests passed (`task_type` is `code` for function-call tests or `code_stdio`
  for stdin/stdout comparison). `sandbox_server.py` / `sandbox_runner.py` /
  `sandbox_rootfs.py` implement the reference sandbox service: a loopback HTTP
  server that executes each candidate in a user/network/PID namespace, a
  read-only chroot on tmpfs, with per-test CPU/memory/process/file rlimits, no
  network egress, and nonce-matched results.
- **Instruction following** (`ifeval_reward.py`): each prompt carries an
  `ifeval_spec` of instruction ids and kwargs evaluated with Open-Instruct's
  IFEvalG checker registry; the reward is the fraction of satisfied
  constraints, and a checker exception is treated as infrastructure failure
  rather than a wrong answer.

The general-domain judge of the internal pipeline relied on an internal
service and is therefore not redistributed; the router raises for unknown
domains so an unsupported mixture fails loudly instead of silently scoring
zero.

### Correct-only length penalty

Long correct answers are shaped with a linear penalty on the number of
generated tokens (implemented in `train/length_shaped_math_grpo.py`):

```
penalty = 0.05 * clip((n_out - 16384) / (65536 - 16384), 0, 1)
r       = r_raw * (1 - penalty)
```

- The penalty ramps linearly from 16,384 generated tokens to the 65,536 cap
  and only applies when `r_raw = 1`: a fully correct answer at the cap
  receives 0.95 instead of 1.0, a wrong answer always receives 0.
- A response truncated by the length cap (`stop_reason = length`) receives
  reward 0 outright.
- Evaluation runs disable the penalty (`apply_length_penalty = False`) and
  score raw correctness only.

### Dynamic sampling

Response groups are filtered by **raw binary correctness**, not by the shaped
reward (`accept_math_group`): a group of 8 responses is discarded when it is
all-correct or all-wrong (zero correctness variance), when any reward falls
outside [0, 1] or is non-finite, or when the longest trajectory exceeds the
69,632-token learner-side cap.

## Training settings

Final math GRPO stage (`configs/math_grpo_length_shaped.yaml`):

| Setting | Value |
| --- | --- |
| Optimizer objective | GRPO (no critic, no reference policy, `kl_ctl = 0`) |
| Actor optimizer | Adam, lr `1e-6` constant (3% warmup), weight decay 0.01, grad clip 1.0 |
| Clipping | `eps_clip = 0.2`, token-level importance sampling |
| Reward normalization | group mean/std over 8 responses (unbiased) |
| Advantage normalization | batch level |
| Minibatches per update | 12, sequence packing up to 69,632 tokens (FFD) |
| Rejection mask | token-level ratio upper bound 5.0 |
| Sampling per step | 384 prompts × 8 responses = 3,072 trajectories |
| Sampling parameters | temperature 1.0, top-p 1.0 |
| Maximum generated response | 65,536 tokens |
| Total prompt–response context | 98,304 tokens |
| Maximum prompt length | 4,096 tokens |
| Steps / epochs | 177 / 3 |
| Evaluation | every 10 steps, 8 samples per prompt, raw correctness |

Earlier mixed-RL experiments described in the technical report also explored a
larger response-group scale at actor lr `2e-6` and a reference-policy KL
variant; the final released stage uses the values in the table above.

## Repository layout

```
rl/
├── configs/math_grpo_length_shaped.yaml   # run config (env-parameterized)
├── rewards/
│   ├── completion.py                      # visible-answer + Python-block extraction after </think>
│   ├── math_reward.py                     # math-verify equivalence reward
│   ├── ifeval_reward.py                   # Open-Instruct IFEvalG constraint reward
│   ├── sandbox_client.py                  # Code reward: HTTP sandbox client + asset hashing
│   ├── sandbox_server.py                  # reference sandbox service (namespace isolation)
│   ├── sandbox_runner.py                  # trusted inner runner inside the chroot
│   ├── sandbox_rootfs.py                  # minimal read-only rootfs builder (tmpfs)
│   └── rewards.py                         # domain router (math / code / if) + async entry
└── train/
    └── length_shaped_math_grpo.py         # training entry: workflow + group filter
                                             # + version-anchored evaluation
```

See [Environment](ENVIRONMENT.md) for the experiment inputs and dependency
pins, and [Experiment Protocol](RUNNING.md) for how to launch a run.
