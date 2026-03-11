# RoboClaw

RoboClaw is an embodied AI assistant framework inspired by OpenClaw, but not limited to OpenClaw's original assistant only in Cyberspace.

The goal is not to ship a single hardcoded robot demo. The goal is to build an open, extensible framework that can connect arbitrary embodiments, environments, and tasks under one coherent architecture.

RoboClaw is intended to converge toward:

- an OpenClaw-style control plane
- a true embodiment abstraction layer
- an execution/runtime layer for real robots and simulators
- a learning-ready backbone for demonstrations, replay, evaluation, and future policy improvement

In short:

**RoboClaw is not just OpenClaw for robots.**

It is an attempt to turn the OpenClaw-style agent control plane into the front door of a broader embodied intelligence system.

## Why RoboClaw

OpenClaw is strong at control-plane problems:

- multi-user access
- sessions and routing
- agent orchestration
- tools and extensions
- remote access and multi-surface interaction
- memory and assistant workflows

But embodied AI needs more than a control plane.

A useful embodied framework must also represent:

- what a body is
- what sensors exist
- what actions are semantically available
- how to ground language into embodiment-specific motion and perception
- how to execute safely in real hardware and simulation
- how to collect episodes and improve over time

RoboClaw exists to add those missing layers without collapsing everything back into a pile of robot-specific scripts, raw ROS primitives, or prompt-only behavior.

## Product Direction

RoboClaw is intended to be:

- a personal embodied AI assistant
- a reusable architecture for embodied AI systems
- framework-first rather than demo-first
- able to support arbitrary embodiments, environments, and tasks over time
- compatible with both simulation and real hardware backends
- designed for future memory, few-shot learning, lifelong learning, self-deployment, and multi-user collaboration

Near-term, RoboClaw is not trying to prove the final architecture in one step.

The near-term objective is a credible, extensible, testable skeleton with one small validated path, then repeated convergence through critique, implementation, and audit.

## Design Principles

RoboClaw follows a few non-negotiable principles.

### 1. OpenClaw is the control-plane inspiration, not the full robot stack

OpenClaw should inform the control plane, session model, routing, tools, and remote interaction model.

It should not be treated as the final abstraction for robot execution.

### 2. Gateway must not directly own low-level robot execution

The Gateway should coordinate users, sessions, tools, and tasks.

It should not become the place where joint commands, controller logic, or robot-specific runtime details accumulate.

### 3. Raw ROS primitives are not the long-term abstraction

ROS2 is important as a runtime and middleware layer, but topics, services, and actions alone do not solve embodiment abstraction.

RoboClaw should sit above raw middleware and expose typed semantic contracts.

### 4. Prompting alone cannot absorb embodiment differences

Different robots, sensors, controllers, and environments must be grounded through explicit interfaces and contracts.

The system should not assume the model can infer everything from ad hoc prompts.

### 5. Semantic actions beat raw motor control

The preferred path is:

`task -> skill contract -> supervisor -> control API -> execution runtime -> backend adapter`

rather than model-generated low-level motor commands.

### 6. Simulation and hardware should be parallel backends

Real hardware and simulators should both sit behind execution adapters.

The framework should not hardcode itself around one robot, one simulator, or one demo environment.

## Architecture

RoboClaw is organized around four explicit planes.

### 1. Control Plane

The Control Plane is responsible for:

- users
- sessions
- agent orchestration
- tool routing
- remote access
- task-level coordination

This is the part most directly inspired by OpenClaw.

It provides the system-facing interface for users and higher-level agents, but it should remain separated from robot-specific execution details.

### 2. Embodiment Plane

The Embodiment Plane is the core RoboClaw addition.

It is responsible for:

- body schema
- sensor schema
- capability graph
- familiarization runtime
- kinematics and frame semantics
- embodiment-specific grounding and calibration
- semantic action mapping

A robot entering RoboClaw should not just expose raw topics and devices. It should first become a standardized embodied object with known capabilities, constraints, and action semantics.

This plane is what allows RoboClaw to aim at arbitrary embodiments over time.

### 3. Execution Plane

The Execution Plane is responsible for:

- ROS2 runtime
- ros2_control and controller management
- action, service, and topic adapters
- simulator adapters
- hardware adapters
- supervisor and safety policies
- runtime execution loops

This is where abstract skill requests become concrete execution.

The key idea is that the same higher-level contract should be able to target:

- a real hardware backend
- a simulation backend
- different execution runtimes behind stable adapters

### 4. Learning Plane

The Learning Plane is responsible for:

- episodes
- demonstrations
- replay
- evaluation
- policy improvement
- sim-to-real support
- future few-shot and lifelong learning integration

This plane is intentionally future-facing, but it is part of the architecture from the start so that data, traces, and execution history can later become learning assets instead of dead logs.

## Embodiment Model

RoboClaw should represent a body through explicit structures rather than hidden assumptions.

Important concepts include:

- **Body schema**: joints, effectors, mobility type, control modes, limits
- **Sensor schema**: cameras, depth sensors, proprioception, force, tactile, state streams
- **Semantic actions**: move forward, move to pose, grasp, release, look_at, reset, relocalize
- **Capability graph**: what the embodiment can currently do in its present state
- **Familiarization runtime**: the onboarding phase where a body is inspected, lightly exercised, grounded, and validated

A useful familiarization flow may include:

- enumerating joints, sensors, topics, actions, and services
- probing small safe motions
- mapping action inputs to observed motion
- checking home pose, limits, directionality, and reset behavior
- generating initial workspace, safety zones, and recovery assumptions
- marking unreliable modules or degraded capabilities

This is a critical difference between RoboClaw and a pure assistant runtime.

## System Flow

A typical RoboClaw path should look like this:

`user goal -> planner -> skill contract -> supervisor -> control API -> execution adapter -> simulator or hardware backend`

That means:

1. A user or agent expresses a task.
2. The Control Plane turns that task into a bounded plan.
3. The plan is expressed through typed skill contracts.
4. A supervisor checks capability, safety, and execution conditions.
5. The Execution Plane selects the correct backend adapter.
6. The embodiment performs the action in simulation or on real hardware.
7. Observations, traces, and outcomes are recorded for later evaluation and learning.

## Example Backends

RoboClaw should support parallel backends such as:

- real hardware drivers, for example SO-101 and future embodiments
- camera and sensor nodes
- Gazebo / gz
- Isaac Sim
- MuJoCo
- Webots
- custom simulators

The framework should not overfit to any one of them.

## User Levels

RoboClaw should serve multiple kinds of users.

### 1. Casual users

People who want to connect an embodiment, issue tasks, and interact with a personal embodied assistant without rebuilding the stack.

### 2. Makers and developers

People who want to tweak capabilities, add custom skills, modify adapters, or introduce new environments.

### 3. Deep integrators and researchers

People who want to define new body models, new execution adapters, new familiarization procedures, or future learning pipelines.

This layered product framing is important: RoboClaw should be easy to use at the surface without blocking deep customization underneath.

## What RoboClaw Is Not

RoboClaw should not become any of the following:

- a thin OpenClaw fork with some ROS2 tools added
- a single hardcoded robot demo
- a prompt wrapper over raw robot topics and services
- a system where the model directly emits low-level joint commands as the default path
- a scaffold that looks modular but has no exercised path

## Current Development Strategy

The current strategy is iterative.

Each round should:

- restate a narrow goal
- propose an architecture slice
- critique that proposal
- converge a smaller approved scope
- implement only that scope
- validate at least one real path
- document what is real, stubbed, deferred, and not yet validated

This is deliberate. RoboClaw should converge through repeated small validated steps, not through a giant top-down scaffold.

## Roadmap and Future TODO

### Near term

- stabilize the core typed contracts between planning, supervision, embodiment, and execution
- validate one small end-to-end path through the system
- add a real simulator-facing adapter behind the same execution interface
- introduce bounded task-to-skill mapping instead of fixed planners
- improve embodiment familiarization beyond static capability descriptions

### Mid term

- add a real SO-101 embodiment path
- support one ROS2-facing execution backend behind the same contracts
- connect simulator and real-hardware backends symmetrically
- formalize capability discovery and safety supervision
- make user-facing task entry routes work through the control plane

### Longer term

- add demonstration and episode capture as first-class learning artifacts
- support replay, evaluation, and policy improvement loops
- support few-shot and lifelong learning workflows
- support self-deployment and collaborative multi-user operation
- broaden the framework to arbitrary embodiments and arbitrary environments

## Guiding Thesis

The core thesis of RoboClaw is simple:

**The missing piece in embodied AI is not only a stronger model. It is a reusable architecture that can turn arbitrary embodiment, arbitrary environment, and arbitrary task into a structured, grounded, and extensible system.**

RoboClaw is meant to be that architecture.
