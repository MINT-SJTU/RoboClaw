# RoboClaw VLA Infra Contract

RoboClaw should treat VLA training as an infrastructure workflow, not as one large training command. The shape is inspired by Dexbotic's layered VLA framework style and RLinf's embodied training recipes, but stays native to RoboClaw and EVO_Train.

## Layer Boundaries

```text
User / AI conversation
  -> Training skill
  -> VLA plan contract
  -> RoboClaw HTTP training API
  -> EVO_Train TCP workflow
  -> Cloud/local runtime
  -> Checkpoint / metrics / deployment
```

## Core Layers

### Data

Owns:

- dataset source;
- schema and robot embodiment;
- camera views;
- language instructions;
- state/action dimensions;
- normalization stats.

Must not own:

- provider token;
- GPU UUID;
- billing policy;
- raw remote shell details.

### Model

Owns:

- policy family, such as ACT, Diffusion Policy, SmolVLA, GR00T, pi0, or custom VLA;
- pretrained checkpoint;
- tokenizer or processor;
- action head and output format;
- deployment compatibility.

### Experiment

Owns:

- workflow name;
- hyperparameters;
- smoke-test or full-run profile;
- train/eval stages;
- artifact expectations;
- success metric.

Example:

```json
{
  "workflow": "custom_project",
  "params": {
    "repoUrl": "https://github.com/example/vla-project.git",
    "setupCommand": "python -m pip install -r requirements.txt",
    "trainCommand": "python train.py --config configs/smoke.yaml",
    "evalCommand": "python eval.py --ckpt outputs/latest.pt",
    "artifactPath": "/root/autodl-tmp/custom_project/outputs"
  }
}
```

### Runtime

Owns:

- local or cloud execution;
- GPU SKU;
- image;
- runtime compatibility matching across backend, model, benchmark, CUDA/image, memory, disk, and cost;
- budget and first-hour charge;
- task status;
- logs and artifact download.

Normal users should see `sku_id` and `image_id`; provider internals stay server-side.

### Inference

Owns:

- checkpoint selection;
- robot compatibility checks;
- action normalization and safety limits;
- deployment status in RoboClaw policy registry.

## Skill Flow

1. Ask what the user wants to train or reproduce.
2. Classify the request into Data, Model, Experiment, Runtime, and Inference fields.
3. Generate a plan with `/api/train/plan`.
4. Query environment choices with `/api/train/gpu-skus` and `/api/train/images`.
5. Match task requirements with `/api/train/runtime-match` before choosing paid compute.
6. Require user confirmation before `/api/train/cloud/start`.
7. Monitor task status and billing.
8. Convert successful artifacts into policy entries that RoboClaw can deploy.

## RoboClaw Control-Plane APIs

RoboClaw owns the VLA-RL control plane above EVO_Train.

### Backend Contract

`vla_rl_backend` is the generic workflow contract. `rlinf_vla` remains as a compatibility alias for existing RLinf recipes.

### Playground Spec

`GET /api/vla-rl/playground`

Returns a developer-playground contract that frontends and AI agents can render directly:

- ordered stages from user intent to plan, runtime match, cost confirmation, execution, artifact review, and deployability;
- API entrypoints for each stage;
- normal-user, advanced-user, and admin-only input boundaries;
- guardrails that prevent provider tokens, SSH details, and AutoDL UUIDs from leaking into normal user flows;
- built-in backend interfaces, profiles, and smoke-test defaults.

This endpoint is the product-facing layer inspired by embodied developer playgrounds: the UI can guide a user through the whole training loop without hard-coding RoboClaw internals.

Concrete built-in backend interfaces are declared through `/api/vla-rl/profiles`:

- `lerobot`: first-class RoboClaw policy fine-tuning backend;
- `rlinf`: distributed embodied RL backend;
- `dexbotic`: project-owned launcher style, often using RLinf underneath;
- `custom`: any project that can provide a launcher and artifact contract.

The list is not closed. `/api/vla-rl/profiles` exposes `backendKindExtensible: true`, and a new backend can be accepted when it supplies the same launcher, preflight, and artifact contract fields.

Every backend must materialize the same run contract: model family, robot embodiment, dataset, checkpoint, artifact path, metrics, observation schema, and action schema.

`backendInterfaces` is the real integration surface. For example, the RLinf interface declares:

- `workflow: "rlinf_vla"`;
- registry injection through `rlinfExtModule` and `RLINF_EXT_MODULE`;
- preflight imports for `rlinf`, launcher module, registry module, and optional env/reward modules;
- launch contracts for `python_module`, `python_script`, and `deepspeed_script`;
- algorithm-to-launcher hints, such as GRPO/PPO to the project-owned Python launcher;
- `run_contract.json` as the artifact contract.

The current concrete interface contracts cover RLinf, LeRobot, Dexbotic, and Custom. Other projects such as OpenPI, OpenVLA, RoboMimic, IsaacLab, or RL-library launchers should be added through the configurable backend-interface contract only when their preflight and artifact expectations are declared.

RoboClaw also carries a minimal `roboclaw_vla.rl` adapter package so EVO_Train preflight can import the declared registry, launcher, and evaluator modules. The included launcher now mirrors RLinf actor/rollout/env worker orchestration as an experimental structural reference. It should be validated inside a live RLinf image before being treated as a production RoboClaw-owned RLinf runner.

Administrators can add or override interface contracts without code changes through `ROBOCLAW_VLA_BACKEND_INTERFACES_JSON` or `ROBOCLAW_VLA_BACKEND_INTERFACES_FILE`. RoboClaw exposes the merged result from `/api/vla-rl/profiles`, and EVO_Train preserves a submitted `params.backendInterface` in the generated plan and remote `run_contract.json`. Direct EVO_Train clients can also configure interfaces through `EVO_TRAIN_VLA_BACKEND_INTERFACES_JSON` or `EVO_TRAIN_VLA_BACKEND_INTERFACES_FILE`.

Benchmarks such as LIBERO, MetaWorld, ManiSkill, and IsaacLab envs are not backend identities. They should be expressed as `benchmark` or `envType`, then matched against SKU and image capabilities through `/api/train/runtime-match`.

### Plan

`POST /api/vla-rl/plan`

Normalizes user language into a platform plan before sending it to EVO_Train:

- model family: Pi0.5, GR00TN1, DM0, CogACT, OFT, NaVILA, Uni-NaVid;
- algorithm: PPO / GRPO;
- training mode: RL post-training or co-training;
- robot adapter: SO-101, XLeRobot, or project-specific;
- image profile: for example Blackwell;
- deployability hints for missing robot/action/observation contracts.

### Runtime Match

`POST /api/train/runtime-match`

Matches the AI/user plan against the administrator-maintained runtime catalog:

- backend kind, such as RLinf, LeRobot, Dexbotic, Custom, or an administrator-configured project backend;
- model family, such as Pi0, DM0, GR00TN1, Uni-NaVid, or ACT;
- benchmark/env type, such as LIBERO, ManiSkill, MetaWorld, or IsaacLab;
- algorithm and training mode, such as PPO, GRPO, RL post-training, or co-training;
- required capabilities, such as CUDA/Torch family, MuJoCo, SAPIEN, Isaac Sim, LIBERO assets, rollout video, dataset format, or custom project tags;
- minimum GPU memory and disk expectations;
- compatible GPU SKU and image pairs with blocking reasons and risks.

This is the product boundary that keeps AutoDL `gpu_spec_uuid`, `image_uuid`, CUDA versions, and service-fee pricing out of the normal user's workflow.

### Artifact Review

`POST /api/vla-rl/artifact-review`

Reads `run_contract.json` plus metrics and returns:

- success metric and value;
- checkpoint/artifact presence;
- deployability summary.

### Deployability Gate

`POST /api/vla-rl/deployability`

Checks whether a trained checkpoint is compatible with the target robot before it enters policy deployment:

- robot embodiment;
- observation schema;
- action schema;
- required checkpoint/artifact fields.

## Why This Belongs In RoboClaw

RoboClaw owns the product and AI control plane: user intent, hardware context, datasets, policies, deployment, and safety. EVO_Train owns execution: queueing, provider lifecycle, billing, command materialization, and artifact collection.

This keeps the PR clean:

- RoboClaw receives VLA infra concepts and user-facing APIs.
- EVO_Train remains the execution backend.
- Providers remain replaceable.
- AI guidance becomes repeatable and testable.
