# Experiment Protocol

Section 4.2 of the ZGCM-1 technical report describes the following mixed-RL
procedure; `train/length_shaped_math_grpo.py` implements the final math GRPO
stage end to end, and `train/code_grpo.py` / `train/if_grpo.py` implement the
code and instruction-following stages of the mixed-RL experiments with the
same procedure minus the length penalty:

1. Estimate prompt difficulty using rollouts from the initial policy and remove
   problems that the model already solves frequently.
2. Sample response groups of 8 responses for 384 prompts per training step
   (temperature 1.0, top-p 1.0).
3. Score responses with the binary mathematics verifier in `rewards/`, then
   apply the correct-only linear length penalty (16,384 → 65,536 tokens, max
   0.05); truncated responses receive reward 0.
4. Filter groups whose raw binary correctness has zero variance (dynamic
   sampling), plus groups with out-of-range/non-finite rewards or trajectories
   longer than the 69,632-token learner cap.
5. Construct group-relative advantages (group reward normalization, batch
   advantage normalization) and optimize the policy with GRPO at an actor
   learning rate of `1e-6` (`eps_clip = 0.2`, no reference policy, `kl_ctl = 0`).

The response and total-context budgets are 65,536 and 98,304 tokens,
respectively, with prompts capped at 4,096 tokens.

## Launching the released configurations

Training data is not redistributed with this repository; prepare the prompt
JSONL files first (schema below), then:

```bash
export ZGCM_MODEL_PATH=/path/to/zgcm1_checkpoint   # HF checkpoint with tokenizer
export ZGCM_TRAIN_JSONL=/path/to/train.jsonl
export ZGCM_VALID_JSONL=/path/to/valid.jsonl
export AREAL_ADMIN_API_KEY=local-key               # any non-empty string locally
export ZGCM_RL_STATUS_ROOT=./status                # eval markers land here

cd rl
PYTHONPATH=. python train/length_shaped_math_grpo.py configs/math_grpo_length_shaped.yaml
```

The code and instruction-following stages additionally need their domain
reward services (see the two sections below):

```bash
# code stage: sandbox service + asset root from "Code sandbox"
PYTHONPATH=. python train/code_grpo.py configs/code_grpo.yaml

# instruction-following stage: Open-Instruct registry from
# "Instruction-following checkers"
PYTHONPATH=. python train/if_grpo.py configs/if_grpo.yaml
```

Optional gates: `ZGCM_EXPECTED_TRAIN_ROWS` / `ZGCM_EXPECTED_VALID_ROWS` enforce
exact row counts; `ZGCM_N_NODES` / `ZGCM_N_GPUS_PER_NODE` size the cluster.

### Prompt JSONL schema

One JSON object per line:

```json
{
  "sample_id": "math-000155d909841838fb9e67c7",
  "domain": "math",
  "task_type": "math",
  "messages": [{"role": "user", "content": "..."}],
  "answers": ["28"],
  "benchmark": "aime24"
}
```

`sample_id`, `domain`, `task_type`, `messages`, `answers` are required
(`domain` must be `math` for the math training entry, `answers` non-empty);
`benchmark` is optional and only splits evaluation statistics. The code and
instruction-following entries validate their own domain and the extra fields
shown below. Prompts must render to at most 4,096 tokens.

The reward router (`rewards/rewards.py`) used by the mixed-RL experiments
additionally accepts two row shapes, required by the corresponding training
entries:

```json
{"sample_id": "code-...", "domain": "code", "task_type": "code",
 "messages": [...], "answers": [""], "code_asset_hash": "<64 hex>",
 "prompt_hash": "<64 hex>"}
```

```json
{"sample_id": "if-...", "domain": "if", "task_type": "if",
 "messages": [...], "answers": [""],
 "ifeval_spec": [{"instruction_id": "language:response_language",
                  "kwargs": {"language": "en"}}]}
```

- `domain: "code"` rows select the sandbox reward: `task_type` is `code`
  (function-call tests) or `code_stdio` (stdin/stdout comparison), and
  `code_asset_hash` names a content-addressed test asset (see below).
- `domain: "if"` rows select the IFEvalG reward: `ifeval_spec` lists the
  instruction ids and kwargs checked against the visible answer.

The released training entries each train on a single domain; since the
router dispatches per row, a custom run can mix these row shapes in one
JSONL by using a standard AReaL workflow instead.

### Code sandbox

Code rewards are always computed by an isolated service reached over HTTP;
training never executes model output in-process. Test assets live under a
root directory as `<root>/<hash[:2]>/<hash>.json.gz`, where each file is a
JSON test list whose SHA-256 equals its own file name, plus a top-level
`manifest.json`.

The reference server (`rewards/sandbox_server.py`, requires root) executes
each candidate inside a user/network/PID namespace and a read-only tmpfs
chroot with per-test rlimits, attests itself at startup, and fails closed if
any isolation property is missing:

```bash
cd rl
export ZGCM_CODE_SANDBOX_TOKEN=<random string, at least 32 chars>
python -m rewards.sandbox_server --port 8090 \
    --asset-root /path/to/code_assets \
    --rootfs /dev/shm/zgcm-code-rootfs --build-rootfs
```

Point the training-side client at it:

```bash
export ZGCM_CODE_SANDBOX_MODE=attested-http   # or namespace-http
export ZGCM_CODE_SANDBOX_URL=http://127.0.0.1:8090
export ZGCM_CODE_SANDBOX_TOKEN=$ZGCM_CODE_SANDBOX_TOKEN
export ZGCM_CODE_ASSET_ROOT=/path/to/code_assets
```

`attested-http` additionally pins the service identity (fetch `/health` once
after startup and write the returned identity fields to a JSON file referenced
by `ZGCM_CODE_EXPECTED_IDENTITY_FILE`). `ZGCM_CODE_MAX_TESTS` (default 8) and
`ZGCM_CODE_TEST_TIMEOUT_SECONDS` (default 2) bound each verification.
`ZGCM_CODE_SANDBOX_MODE=open-instruct-http` instead sends the selected tests
verbatim to an Open-Instruct-compatible `/test_program` (or
`/test_program_stdio`) endpoint, so any existing sandbox service can be
reused.

### Instruction-following checkers

The `if` reward needs Open-Instruct's IFEvalG registry and NLTK data:

```bash
export ZGCM_OPEN_INSTRUCT_ROOT=/path/to/open-instruct
export ZGCM_OPEN_INSTRUCT_SITE_PACKAGES=/path/to/open-instruct/venv/lib/python3.12/site-packages
export NLTK_DATA=/path/to/nltk_data
```

`rewards.ifeval_reward.validate_ifeval_registry` and `ifeval_preflight` can
pre-check that every instruction id used by the prompt files exists in the
registry before launch.

### Evaluation and recovery

The math and instruction-following entries evaluate every 10 steps (and at
step 0 and the final step) with 8 samples per prompt and raw reward scoring
(correctness, resp. fraction of constraints satisfied); results are written
as version-anchored markers under
`$ZGCM_RL_STATUS_ROOT/evaluations/version_XXX.json`, which also makes
periodic evaluation idempotent across restarts. The code entry evaluates at
the baseline and final steps plus every 51 steps with 1 sample per prompt
scoring the binary all-tests-pass@1 metric; after training it audits the
dumped evaluation rollouts (completeness, version pinning, binary rewards)
and writes `code_eval_baseline.json`, `code_eval_stepXXX.json`, and
`code_eval_final.json` markers under `$ZGCM_RL_STATUS_ROOT`. Checkpoints are
saved periodically (every 10 steps for math and instruction following, every
51 steps for code) and recovery state every step (`recover.mode: auto`).
