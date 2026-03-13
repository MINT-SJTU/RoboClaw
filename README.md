<h1>
  <img src="assets/roboclaw_icon.png" alt="RoboClaw icon" width="84" />
  RoboClaw
</h1>

<p>
  <img src="https://img.shields.io/badge/status-early_stage-orange" alt="status">
  <img src="https://img.shields.io/badge/open-source-blue" alt="open source">
  <img src="https://img.shields.io/badge/community-co--create-green" alt="community">
  <img src="https://img.shields.io/badge/focus-embodied_ai-black" alt="focus">
</p>

**RoboClaw** is an open-source co-created project for embodied intelligence assistants.

## ✨ What We Want To Build

Imagine a setup where you can take a robot arm, add a camera, spend a few minutes calibrating it, and then instruct it in natural language to complete a concrete task.

In the short term, we want to establish a practical paradigm: different embodiments, environments, and tasks should all be able to plug into the system through a unified semantic interface, with deployable implementations added step by step.

In the long term, we want RoboClaw to do more than execute tasks. It should also participate in training and analysis: judging whether a task is complete, evaluating execution quality, identifying when and why failures happen, helping recover the scene after failure, and analyzing what can be improved in both reasoning and execution. The goal is to support the continuous development of embodied intelligence systems, not just complete a single task once.

This paradigm is intended to progressively connect:

`Goal Understanding -> Planning -> Semantic Skills -> Bridge Layer -> Execution Layer -> Real / Simulated Carriers`

At the moment, we organize the project into four layers:

- `Assistant Layer`: users, sessions, orchestration, tools, and remote collaboration
- `Embodiment Layer`: embodiment modeling, capability abstraction, calibration, recovery, and training assistance
- `Execution Layer`: execution interfaces, messaging, services, actions, supervision, and state feedback
- `Carrier Layer`: simulators, real robots, deployment, and feedback loops

## 🌱 Current Status

RoboClaw is still at a very early stage.

Right now, we are mainly working on:

- building the first end-to-end embodied execution pipeline
- validating the critical path from semantic interfaces to the execution layer

The direction is clear, and we will continue making the process public as we move forward.

## 📦 Installation

### Conda

```bash
conda create -n roboclaw python=3.11 -y
conda activate roboclaw
pip install -e ".[dev]"
```

## 📢 Community Co-Creation

RoboClaw aims to move forward in a genuinely open and collaborative way.

That means some important decisions that should not be made by maintainers alone will be discussed openly, with the community invited to participate where possible, such as:

- which real robots to support first
- which simulation platforms to support first
- the priority of the first batch of core features
- what should be prioritized on the roadmap

If you care about embodied AI architecture, capability abstraction, execution pipelines, or robot integration, you can contribute through:

- `Issues`: bug reports, feature requests, and implementation suggestions
- `Pull Requests`: direct code or documentation improvements
- `GitHub Discussions`: conversations around direction, design, and usage

Areas where contributions are especially useful right now include:

- embodied AI architecture design
- capability abstraction and semantic skill interfaces
- ROS2 and execution-layer integration
- simulator support
- real robot adaptation
- evaluation and validation
- documentation and developer experience

## ❓ FAQ

### What is RoboClaw?

RoboClaw is an open-source project for embodied intelligence assistants. It focuses not only on model capability, but also on embodiment abstraction, skill interfaces, execution supervision, simulation integration, and real robot deployment.

### What is the relationship between RoboClaw, OpenClaw, and nanobot?

OpenClaw and nanobot have been important inspirations for us, and RoboClaw continues to evolve from nanobot. But RoboClaw is not focused on being a general assistant. Its emphasis is on embodiment, skills, execution supervision, simulation, and real-world robot deployment.

### How is this different from the usual "goal understanding -> planning -> sub-skills -> execution" approach?

The pipeline itself is not new. Many projects are already exploring it. What is different is that RoboClaw does not want to hard-code "sub-skills" into a fixed set of capabilities, nor bind the execution side to a single robot. We care more about building an extensible connection between semantics and action, so different embodiments can access, implement, and extend their own skills under one shared paradigm.

### Will RoboClaw support multi-robot scenarios?

RoboClaw will support multi-robot scenarios in the future, but that is not the top priority in the first stage. Right now, we are more focused on getting single-robot capability abstraction, semantic skill interfaces, execution supervision, and the simulation-to-real pipeline into solid shape before moving into more complex problems such as multi-robot coordination, task allocation, and state synchronization.

## 🗺️ To-do List

- [x] Set up the open-source repository, publish the initial README, and add GitHub-native proposal entry points
- [ ] Launch GitHub Discussions and start the first logo / icon community vote
- [ ] Add the first architecture document
- [ ] Define unified embodied capabilities and semantic interfaces
- [ ] Connect the bridge layer, execution layer, and the first simulation platform
- [ ] Support the first real robot platform
- [ ] Design safe-stop and recovery mechanisms
- [ ] ...

Coming soon.

## 🤝 Collaboration Notes

RoboClaw is still in an early stage, and the immediate focus is on making the core functionality and execution pipeline solid.

If you care about embodied intelligence, robot execution systems, simulation platforms, capability abstraction, or evaluation, you are welcome to join through issues, discussions, or PRs.

If you want to be an active contributor, please contact us by emailing bozhaonanjing [[@]] gmail [[DOT]] com

## 🙏 Acknowledgments

RoboClaw references and inherits part of its initial thinking from [nanobot](https://github.com/HKUDS/nanobot). We appreciate its lightweight practice along the OpenClaw line, which helped us build the first prototype faster and continue evolving toward embodied intelligence.
