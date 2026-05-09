# Workflow Planning API

RoboClaw now exposes a unified workflow spec for the common embodied pipeline:

- `record`: collect data from the robot
- `train`: fit a policy from a dataset
- `infer`: run a policy back on hardware and record the evaluation dataset

The goal is not to add another dashboard. The goal is to make one workflow description reusable across planning, validation, and execution.

## What The Planner Resolves

Given a single `WorkflowSpec`, RoboClaw can:

- validate whether each stage is runnable
- infer missing dataset names between stages
- infer the checkpoint path for inference from the planned training output
- compile the concrete command that each stage would run

This makes `plan` the main review surface before anyone starts a real robot job.

## API Surface

- `POST /api/workflows/validate`
  Returns the compiled workflow plus issues, without starting any stage.
- `POST /api/workflows/plan`
  Returns the compiled stage plan, including derived dataset names, checkpoint paths, and commands.
- `POST /api/workflows/run/{phase}`
  Starts a single validated phase, where `phase` is `record`, `train`, or `infer`.

The run endpoints consume the same derived dataset and checkpoint values that appear in `plan`, so review and execution stay aligned.

## Example Spec

```json
{
  "name": "pick-cube-pipeline",
  "hardware": {
    "useCameras": true
  },
  "record": {
    "enabled": true,
    "task": "pick cube",
    "datasetName": "pick_cube_v1",
    "numEpisodes": 5
  },
  "train": {
    "enabled": true,
    "policyType": "act",
    "steps": 2000
  },
  "infer": {
    "enabled": true,
    "datasetName": "eval_pick_cube_v1",
    "numEpisodes": 2
  }
}
```

## Review Flow

1. Open the FastAPI docs page at `/docs`.
2. Use `POST /api/workflows/plan` with a workflow spec.
3. Check whether the returned stages are `ready`, whether datasets flow between stages as expected, and whether the checkpoint path matches the intended policy output.
4. Only then call `POST /api/workflows/run/{phase}` for the stage you want to execute.
