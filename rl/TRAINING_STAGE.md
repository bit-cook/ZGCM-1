# Reinforcement Learning

ZGCM-1 explores mixed reinforcement learning on mathematics, code, and
general-capability tasks using GRPO. Training first estimates prompt
difficulty from initial-policy rollouts, then provides learning signals
through domain-specific rewards and dynamic sampling. This directory contains
the complete configuration, reward implementations, and training entry for the
final math GRPO stage of the released model; the training framework is
[AReaL](https://github.com/areal-project/AReaL) v2.0.0.

Mathematics uses a binary answer-correctness reward (the visible answer is
extracted after `</think>` and judged with math-verify bidirectional
equivalence, `precision=6`, 8-second per-call timeout). Code uses the fraction
of executable tests passed (execution happens inside an isolated sandbox,
reached only over HTTP). General tasks use correctness or instruction-following
criteria (instruction following is the fraction of Open-Instruct IFEvalG
checkers satisfied). The reward implementations for all three domains are
released in `rewards/` in this directory.

**Correct-only length penalty**: applied linearly over generated tokens, only
to samples that are answered correctly:

```
penalty = 0.05 × clip((n_out − 16384) / (65536 − 16384), 0, 1)
r       = r_raw × (1 − penalty)
```

The penalty ramps linearly from 16,384 tokens up to the 65,536 cap, where it
deducts at most 0.05 (a fully correct answer at the cap scores 0.95).
Responses truncated by the length cap receive reward 0, and evaluation runs
apply no penalty. **Dynamic sampling** filters on raw binary correctness: an
entire group is discarded when its 8 responses are all-correct or all-wrong,
when any reward is out of range or non-finite, or when the longest trajectory
in the group exceeds 69,632 tokens.

The final stage samples 384 prompts × 8 responses per step (3,072
trajectories) at temperature 1.0, top-p 1.0. The Adam optimizer uses
lr `1e-6` constant (3% warmup), weight decay 0.01, and grad clip 1.0;
`eps_clip=0.2` (token-level importance sampling) with group reward
normalization plus batch-level advantage normalization and 12 minibatches per
update; there is no critic or reference policy (`kl_ctl=0`). The maximum
generation length is 65,536 tokens within a 98,304-token total context, with
prompts capped at 4,096 tokens; training runs 177 steps / 3 epochs, with a raw
correctness evaluation on 8 samples every 10 steps. The earlier experiments
described in Section 4.2 of the technical report also explored a larger
response-group scale (lr `2e-6`) and a reference-policy KL variant.

See Section 4.2 of the technical report for the method and parameters, and
[README](README.md), [configs/](configs/), and [train/](train/) in this
directory for the configuration, rewards, and training scripts.
