# RoboClaw Web UI

RoboClaw 的 Web 用户界面，提供与 RoboClaw agent 的实时对话能力。

## 功能

- **对话界面**: 与 RoboClaw agent 实时对话
- **设置**: 配置 AI provider (API base / API key)

## 开发

### 安装依赖

```bash
npm install
```

### 启动开发服务器

```bash
npm run dev
```

访问 http://localhost:5173

### 构建生产版本

```bash
npm run build
```

## 技术栈

- React 18
- TypeScript
- Vite
- Tailwind CSS
- Zustand (状态管理)
- React Router
- WebSocket (实时通信)

## 架构

```
src/
├── features/          # 功能模块
│   ├── chat/         # 对话界面
│   └── settings/     # 设置页面
├── shared/           # 共享代码
│   ├── components/   # 共享组件
│   └── api/          # API 客户端
└── assets/           # 静态资源
```

## 与后端通信

Web UI 通过 WebSocket 与 RoboClaw 后端通信：

- WebSocket 端点: `ws://localhost:8765/ws`
- REST API: `http://localhost:8765/api/*`

确保后端 Web Channel 已启动：

```bash
roboclaw web start
```
