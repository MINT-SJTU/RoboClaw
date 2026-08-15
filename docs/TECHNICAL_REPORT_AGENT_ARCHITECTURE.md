# RoboClaw Agent 架构整理草稿

RoboClaw 可以被理解为一个以 Embodied Main Agent 为核心的主-子 Agent 协同架构。系统不是简单堆叠通信、机器人控制、数据管理和训练模块，而是围绕具身任务执行过程，将不同能力组织为主 Agent、专门子 Agent，以及可扩展的 Skill/Tool 层。

## 1. Embodied Main Agent

Embodied Main Agent 是系统的核心决策与调度中心。它负责接收来自用户的自然语言请求，结合当前会话上下文、系统状态和可用工具，判断任务目标、拆解执行步骤，并决定是否需要调用子 Agent 或具体工具。

这一部分可以涵盖以下能力：

- 子 Agent 调度与协同。
- 任务目标解析与阶段划分。
- 执行状态汇总与下一步决策。
- 全局上下文构造。

Embodied Main Agent 并不直接承担具体工具调用和底层模型调用，而是作为高层编排者，维护任务目标、执行阶段、全局上下文和子 Agent 之间的协作关系。

## 2. Long-Horizon Task Memory Sub-Agent

Long-Horizon Task Memory Sub-Agent 负责维护任务级记忆。它关注的不是简单保存聊天记录，而是围绕长程任务、阶段性结果、当前子任务进展和用户交互偏好组织记忆，从而支持 Embodied Main Agent 在多轮、多阶段任务中持续决策。

这一部分可以涵盖以下能力：

- 历史长程任务与阶段性结果索引。
- 当前任务拆解与子任务进展维护，包括执行反馈、关键中间结果、失败原因，以及各 sub-agent 产生的重要结论。
- 用户意图与交互风格维护。

它的重点是维护任务连续性。对于“整理桌面”这类抽象任务，系统不仅需要理解最终目标，还需要记住当前已经完成了哪些步骤、哪些物体已经处理、下一步依赖什么条件，以及过去类似任务中有哪些经验可以复用。

## 3. Data Lifecycle Sub-Agent

Data Lifecycle Sub-Agent 负责围绕机器人学习数据的完整生命周期工作。它关注的不是单次机器人动作，而是数据如何被采集、组织、检查、清理、标注、训练使用，并最终回流到策略迭代中。

这一部分可以涵盖以下能力：

- 数据采集与 dataset 版本管理。
- 数据操作：质量检查、清洗、筛选、标注。
- 训练周期数据管理：训练记录、推理结果回流。

因此，技术报告中有关数据管理、训练与策略闭环的内容，可以围绕这个子 Agent 展开。

## 4. Embodiment Adaptation Sub-Agent

Embodiment Adaptation Sub-Agent 负责把抽象任务适配到具体物理实体上。它解决的问题是：同样一句用户指令，在不同机械臂、不同相机、不同末端执行器、不同硬件连接状态下，应该如何落到可执行的机器人链路中。

这一部分可以涵盖以下能力：

- 硬件资源管理：包括机器人本体、传感器、执行器和末端设备等物理实体。
- 硬件自检、适配与健康监视：包括校准、自动校准、连接问题定位、SDK 自适应和状态检查。
- 任务执行链路适配：包括遥操作、回放、策略推理等任务在具体硬件上的可执行性判断与链路打通。

它的重点不是简单控制硬件，而是完成具身任务和具体硬件之间的能力对齐。

## 5. Scalable Skill and Tool Layer

Scalable Skill and Tool Layer 负责维护系统可调用的能力集合。它把底层服务、外部 API、本地命令、机器人操作、数据操作等封装成统一的工具或 skill，使主 Agent 和子 Agent 可以通过结构化方式调用。

这一部分可以涵盖以下能力：

- file / shell / web / message 等基础工具。
- setup / calibration / teleop / record / replay / train / infer 等机器人工具。
- hub 数据和策略同步工具。
- 用户自定义 skill。
- OpenClaw 风格 skill 扩展。
- 工具 schema 管理。
- 工具执行结果回填。
- 工具调用权限与运行时约束。

这一层是 Agent 决策转化为真实执行的接口层。

## 6. 架构叙事

按照上述结构，原有功能并没有丢失，而是被放入了更清楚的主-子 Agent 协同关系中。RoboClaw 不只是机器人控制台、数据管理工具和聊天入口的组合，而是一个面向具身任务的 Agent harness：主 Agent 负责推理和调度，Long-Horizon Task Memory Sub-Agent 负责维护任务连续性，Data Lifecycle Sub-Agent 和 Embodiment Adaptation Sub-Agent 分别负责数据生命周期与物理实体适配，Skill/Tool 层负责把决策转化为可执行动作，最终连接机器人硬件、数据资产、训练框架和外部服务。
