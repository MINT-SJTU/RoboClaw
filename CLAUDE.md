# RoboClaw Agent 开发指南

---

## Part 1: 终极目标

用自然语言控制任何机器人。用户不碰代码、不看文档、不配环境。

- **L0** — 没有机器人？打开就能在仿真里对话操控。
- **L1** — 刚买了机械臂？几句话完成连接、校准、让它动起来。
- **L2** — 想做复杂任务？对话组合出抓取、搬运等技能。
- **L3** — 做科研？对话驱动数据采集、训练、部署、监督，完整闭环。
- 接入新机器人不写代码，对话描述硬件，Agent 生成 adapter。
- 借鉴 LeRobot 的 dataset 格式和训练范式，自己实现，不引入外部依赖。
- 内置 ACT、Diffusion Policy 等主流算法，对话选择和调参。
- 仿真到真机无缝切换。每个本体自带安全约束，框架强制执行。

---

## Part 2: TODO

> Claude B (loop) 按此列表工作。完成后标记 ✅，遇到阻塞标记 🚫。

### 当前批次

- [x] 1. 摄像头组合：本体 + 摄像头在 assembly 中绑定，实时取图，数据可录入 episode ✅
- [x] 2. 数据采集格式：借鉴 LeRobot dataset 范式，实现带图像的 episode 录制格式（依赖 #1） ✅
- [x] 3. 数据采集 GUI：采集时自动启动 HTTP 可视化界面（摄像头画面 + episode 进度 + 关节状态），借鉴 LeRobot，映射端口供远程浏览器访问 ✅
- [x] 4. ACT 训练 recipe：借鉴 LeRobot ACT 实现，作为第一个内置可训练 policy（依赖 #2） ✅
- [x] 5. Layer 2 能力查询接口：从 primitive 的 CapabilityFamily 自动聚合本体能力，Agent 可查询 ✅
- [x] 6. 接入 PiperX 作为第二个 builtin 本体，参考 Evo-RL（GitHub），验证框架泛化（依赖 #5） ✅
- [ ] 7. 验收测试（见 `docs/acceptance-test.md`）：A1 夹爪开合 / A4 仿真体验 / A6 采集 10 episode / A7 训练 ACT policy 🚫 Docker 镜像已构建，远程测试通过 43 个用例，等待用户确认硬件连接后执行对话验收


---

## Part 3: 工作规范

### 写代码

- 代码交给 Codex 写（`/codex-dispatch`）。Claude Code 负责读代码、设计方案、写 worker prompt、review 结果、跑验证、与用户沟通。
- 每次结构性改动后，跑 `bash scripts/embodied_lines.sh` 并与上次对比。如果行数增长，必须给出理由或重构到不涨。向用户汇报 before/after。

### 代码规范

- 框架代码放 `roboclaw/embodied/`，用户资产放 `~/.roboclaw/workspace/embodied/`。
- 不在通用层硬编码具体本体。本体特定逻辑只能在 `builtins/<id>.py`、manifest、profile、bridge 里。
- 生产框架代码保持英文。
- 最小化代码量。不引入新抽象，除非它消除的代码比引入的更多。
- RoboClaw 对用户的对话必须保持高层次和通用化。不暴露串口路径、底层协议细节、内部技术实现。用户应无感地完成操作。

### 验证

- 本地：`python -m pytest tests/ -x -q`
- 远程 Docker：按 memory 中存储的流程执行（`reference_remote_validation.md`）
- **每次代码改动后，必须按 `docs/acceptance-test.md` 执行验收测试。** Claude Code 扮演小白用户，自由对话 RoboClaw，通过摄像头验证夹爪开合。测试通过不算验收，对话走通才算。
