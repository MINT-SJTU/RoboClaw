# Robot Training Skill

Use this skill when a user wants RoboClaw to train, reproduce, debug, or evaluate a robot policy.

## Goal

Turn natural-language training intent into a confirmed RoboClaw training plan, then submit it to the EVO_Train bridge only after the plan has enough information and the user understands the first-hour cost.

## Workflow

1. Identify the training type:
   - local LeRobot dataset policy training;
   - EVO_Train benchmark workflow such as MetaWorld or LIBERO;
   - RLinf-backed VLA+RL post-training;
   - custom GitHub project;
   - debugging or result review for an existing run.
2. Collect only missing fields:
   - dataset or code source;
   - workflow and params;
   - target metric;
   - smoke-test or full-run intent;
   - budget or max runtime.
3. Query backend catalogs:
   - `GET /api/vla-rl/playground`;
   - `GET /api/train/gpu-skus`;
   - `GET /api/train/images`.
4. Generate a plan:
   - `POST /api/train/plan`;
   - inspect `missingFields`, `warnings`, `estimatedMinimumCostCents`, and `readyToStart`.
5. Match runtime requirements:
   - `POST /api/train/runtime-match`;
   - compare backend, model family, benchmark/env, algorithm, training mode, required capabilities, GPU memory, image, and blocking reasons.
6. Ask for confirmation before paid compute.
7. Start the run:
   - `POST /api/train/cloud/start` with `workflow`, `params`, `sku_id`, and `image_id` for EVO_Train workflows;
   - or local dataset fields for local RoboClaw training.
8. Monitor and diagnose:
   - `GET /api/train/current`;
   - `GET /api/train/status/{job_id}`;
   - use logs, metrics, videos, checkpoints, and billing records as evidence.

## Planning Pattern

Borrow the Dexbotic / RLinf-style breakdown for embodied VLA training:

- Environment: benchmark, simulator, robot, observation shape, action space.
- Algorithm: SFT, PPO, GRPO, BC, ACT, diffusion policy, or custom.
- Model: base checkpoint, LoRA/full fine-tune, tokenizer/action head constraints.
- Runtime: GPU SKU, CUDA/image choice, dataset path, output path.
- Metrics: success rate, reward, loss, rollout videos, checkpoint availability.
- Failure guards: smoke test first, short eval, artifact existence check.

## VLA Infra Layers

Keep these layers separate when guiding users or generating backend requests:

- Data Layer: dataset source, schema, camera views, language instructions, state/action format, normalization stats.
- Model Layer: policy family, pretrained checkpoint, action dimension, image inputs, tokenizer or processor requirements.
- Experiment Layer: train/eval recipe, hyperparameters, smoke-test profile, expected artifacts.
- Runtime Layer: local or cloud mode, GPU SKU, image, budget, provider task status.
- Inference Layer: deployable checkpoint, action denormalization, robot compatibility, safety checks.

Do not let provider details leak upward into Data or Model choices. Normal users choose a model family and GPU SKU, not AutoDL UUIDs.

Treat benchmark/env choices such as LIBERO, MetaWorld, ManiSkill, RoboSuite, and IsaacLab tasks as task requirements. Treat RLinf, LeRobot, Dexbotic, and custom projects as the built-in backend choices. A single plan may need both.

For anything outside the built-in interfaces, preserve the user's backend name and put concrete runtime needs into `requiredCapabilities`, for example `["cuda121", "mujoco", "sapien", "libero_assets", "rollout_video"]`. The backend list is extensible only through declared launcher, preflight, and artifact contracts.

## Guardrails

- Do not ask normal users for provider UUIDs, AutoDL SSH details, or platform tokens.
- Do not start paid compute from a vague request.
- Do not select a GPU/image pair until `/api/train/runtime-match` returns at least one compatible candidate or clearly explains the blocking reason.
- Prefer a smoke test when the code, dataset, image, or metric is uncertain.
- Treat raw command mode as trusted-admin debugging only.
- Do not recommend a more expensive GPU unless the workload or logs justify it.

## EVO_Train Request Shape

```json
{
  "username": "user",
  "task_name": "libero-smoke-001",
  "workflow": "evf_libero",
  "params": {
    "suite": "libero_object_task",
    "epochs": 1,
    "evalEpisodes": 2
  },
  "sku_id": "autodl-4090d",
  "image_id": "robotics-cu121"
}
```

For VLA+RL post-training, use `workflow: "rlinf_vla"` with Dexbotic/RoboClaw-style `project_backend` by default. The project should own the model adapter and launch module, while RLinf remains the backend.

Do not treat `rlinf` as just a catalog label. Check `/api/vla-rl/profiles.backendInterfaces.rlinf` and preserve the interface fields:

- `rlinfExtModule` is the registry injection module;
- `RLINF_EXT_MODULE` is exported for the remote launcher;
- preflight imports `rlinf`, `launcherModule`, and the registry/env/reward modules;
- GRPO/PPO launcher selection should come from the backend interface or explicit profile, not from free-form text;
- EVO_Train writes `run_contract.json` before launch.

LeRobot, Dexbotic, and Custom also expose backend interface contracts. For OpenPI, OpenVLA, GR00T, Octo, RoboMimic, IsaacLab, RL-library launchers, or any other backend, require an administrator-provided `backendInterface` contract before presenting it as integrated.

The built-in RoboClaw GRPO profile is an experimental contract for RLinf post-training. It records `groupSize`, `placementStrategy`, and a Hydra config path, and it requires true import preflight. Treat the current `roboclaw_vla.rl.launcher` as structurally implemented but not live-validated until it runs inside a full RLinf image.

If an administrator configured `ROBOCLAW_VLA_BACKEND_INTERFACES_JSON` or `ROBOCLAW_VLA_BACKEND_INTERFACES_FILE`, prefer the merged `/api/vla-rl/profiles.backendInterfaces` result over hard-coded assumptions. Pass the selected interface through `params.backendInterface` when submitting an EVO_Train workflow.

```json
{
  "workflow": "rlinf_vla",
  "params": {
    "launchMode": "project_backend",
    "repoUrl": "https://github.com/dexmal/dexbotic.git",
    "workdir": "/root/autodl-tmp/dexbotic",
    "configName": "libero_goal_ppo_dexbotic_pi0",
    "launcherModule": "dexbotic.rl.model_rl_libero_pi0",
    "rlinfExtModule": "dexbotic.rl.rlinf_registry",
    "suite": "libero_goal",
    "datasetPath": "/root/autodl-tmp/datasets/libero",
    "artifactPath": "/root/autodl-tmp/evo_train/jobs/vla-rl/artifacts"
  }
}
```

Use `rlinf_frontend` only for low-level debugging where the remote repo or image is RLinf itself.

For project-owned RL scripts that do not use RLinf as the public entrypoint, keep the same `project_backend` workflow and choose a structured `launcherKind`:

- `python_module`: project module entrypoint plus optional `rlinfExtModule`.
- `deepspeed_script`: SimpleVLA-RL-style distributed training script with `scriptPath`, `sftModelPath`, and `datasetName`.
- `python_script`: project script that accepts `--config-name`.

When preparing a paid VLA-RL run, include a result contract where possible:

- `modelRegistryName`, `policyFamily`, `envModule`, `rewardModule`;
- `metricPaths`, `resultFiles`, `successMetric`;
- `robotEmbodiment`, `observationSchema`, `actionSchema`.

EVO_Train writes these fields to `run_contract.json` before training starts so RoboClaw can review artifacts and decide whether a checkpoint is deployable without guessing from logs.

Normalize new VLA capabilities into stable fields instead of leaving them as prose:

- Uni-NaVid, NaVILA, GR00TN1, GR00T, Pi0.5, DM0, CogACT, OFT -> `modelFamily`;
- GRPO / PPO -> `algorithm`;
- co-training / joint optimization -> `trainingMode: "co_training"`;
- action expert + LLM -> `coTrainingTargets`;
- SO-101 / XLeRobot -> `robotAdapter`;
- Blackwell GPU image -> `imageProfile: "blackwell"`.

Prefer built-in training profiles when the user names only a model family:

- Pi0 -> `builtinTrainingProfile: "dexbotic_pi0_rlinf"`;
- DM0 -> `builtinTrainingProfile: "dexbotic_dm0_rlinf"`;
- SimpleVLA-RL -> `builtinTrainingProfile: "dexbotic_simplevla_rl"`;
- RoboClaw-owned or future VLA models -> `builtinTrainingProfile: "roboclaw_rlinf_backend"`.

If the user provides an explicit launcher, script, or repository, preserve it instead of replacing it with a built-in profile.

## Done Criteria

- The plan is ready or its missing fields are clear.
- GPU and image choices came from platform catalogs.
- First-hour cost is visible.
- The user confirmed before `start`.
- Results can be checked through RoboClaw status and policy endpoints.
