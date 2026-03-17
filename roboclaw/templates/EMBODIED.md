# Embodied Workspace Policy

`roboclaw/embodied/` is framework code. It holds generic protocols, reusable robot manifests, shared sensor types, runtime abstractions, transport contracts, and common procedure definitions.

Concrete user setups must live in this workspace under `embodied/`, not in the package source tree.

## Rule

- Edit `roboclaw/embodied/` only when changing generic framework behavior.
- Put user-, lab-, or demo-specific embodied assets under `embodied/` in this workspace.
- Reuse built-in robot and sensor ids when available instead of copying framework definitions.

## Recommended Workspace Layout

```text
embodied/
  README.md
  intake/
  robots/
  sensors/
  assemblies/
  deployments/
  adapters/
    ros2/
  simulators/
    worlds/
    scenarios/
  notes/
  _templates/
```

## Generation Flow

1. As soon as the user identifies the robot class or model, write or update an intake note under `embodied/intake/` with the facts already known.
2. Reuse built-in component manifests where possible.
3. If the robot already exists in framework code, inspect its built-in manifest, notes, and setup hints before asking follow-up questions.
4. Use the current RoboClaw integration path: `catalog -> runtime -> procedures -> adapters -> ROS2 -> embodiment`.
5. Infer obvious facts from the framework, repo, local environment, or existing workspace assets before asking the user.
6. Ask only for setup-specific facts that are still missing and are necessary for the next concrete step.
7. Generate setup-specific files under `embodied/assemblies`, `embodied/deployments`, `embodied/adapters`, and `embodied/simulators`.
8. If a robot or sensor is local-only and not reusable enough for framework, define it under `embodied/robots` or `embodied/sensors`.
9. Keep ids stable across later iterations so the same setup can be refined incrementally.

## First-Run Objective

For a first-time user, the immediate success criteria are:

1. create or refine workspace setup assets
2. load them back into catalog successfully
3. complete `connect`
4. complete `calibrate` if supported, or explain why not
5. complete a small safe `move`
6. complete `debug`
7. complete `reset`

## First-Run Interaction Policy

- The user should not need to understand internal terms such as framework code, workspace assets, adapters, ROS2 namespaces, topics, actions, or setup file layout.
- When the user says something like "I want to install SO101" or "I want to connect a real robot arm", start intake immediately with the facts already known.
- Do not block intake on all deployment details being present up front.
- Prefer asking one small next-step question at a time, not a large questionnaire.
- For known robots already represented in framework code, inspect their built-in manifest and setup hints first.
- RoboClaw currently assumes ROS2 as the execution and integration path for embodiment setup.
- Do not ask the user to choose between ROS2, SDK, or other stacks during first-run setup.
- Defer low-level connection details such as serial device names, IPs, package paths, namespaces, and driver variants until they are actually needed for the next step.
- Sensor questions are acceptable only when they affect the generated setup or the next procedure step.
- The right first-run behavior is:
  1. recognize the robot
  2. inspect built-in robot hints when available
  3. create intake
  4. reuse built-in definitions
  5. ask only the smallest missing setup-specific question
  6. continue refining the same setup

## Ownership Boundary

- Framework examples should not hardcode one user's SO101, Piper, xArm, or humanoid setup.
- A workspace assembly may reference a built-in robot id such as `so101`, but the specific ROS2 namespace, topics, camera mounts, and deployment connection values belong in workspace files.
- Workspace loader convention: generated Python files should export one of `ROBOT`, `SENSOR`, `ASSEMBLY`, `DEPLOYMENT`, `ADAPTER`, `WORLD`, `SCENARIO`, or their plural forms.
- Workspace contract metadata: generated Python files should also define `WORKSPACE_ASSET = WorkspaceAssetContract(...)` with `kind`, `schema_version`, `export_convention`, and `migration_policy`.
- Files under `embodied/_templates/` are intentionally minimal scaffolds, not ready-made arm demos. Replace the placeholders from intake facts instead of copying them verbatim.
- The current framework path is `catalog -> runtime -> procedures -> adapters -> ROS2 -> embodiment`.
