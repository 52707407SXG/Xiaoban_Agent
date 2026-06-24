# Xiaoban-Agent

Xiaoban-Agent（站小伴 / 小伴）是 My Stand 原生服务端 Agent。

它基于成熟 Agent runtime 改造，保留主循环、工具调用、记忆、技能、网关、
定时任务、多模型 Provider、通道适配和安全审批等核心能力；同时把身份、上下文、
模块插件接口、权限边界和回复习惯改造成 My Stand 专用。

## 定位

小伴不是一个前端聊天框，也不是通用聊天机器人。

它是 My Stand 的服务端大脑：

- Web、微信、飞书、企业微信、QQ、CLI 都只是 Connector。
- 对话状态、消息去重、TaskState、Memory、ToolRegistry、ContextBuilder
  都在服务端。
- My Stand 模块能力统一走 `Mystand Module Capability Contract v0.1`
  （MMCC v0.1）。
- 模块说明书让小伴知道怎么用，模块 API token 才让小伴有权限执行。

## 第一版目标

首个 GitHub 版本必须已经是 Xiaoban-Agent，不上传纯上游 baseline。

当前首发改造范围：

- Xiaoban 原生身份和系统提示词底座。
- `xiaoban` 命令入口。
- MMCC v0.1 manifest 解析、校验、权限策略、ToolRegistry 适配骨架。
- My Stand 源码披露护栏。
- 本地 smoke 验证脚本。
- 后续接 My Stand 主站、微信、飞书、企业微信、QQ、Event Center 的接口预留。

## 本地验证

```bash
python3 scripts/xiaoban_validate.py
python3 scripts/xiaoban_smoke.py
python3 scripts/xiaoban_server_smoke.py
python3 -m xiaoban.cli --version
python3 bin/xiaoban --version
./bin/xiaoban --version
```

预期输出：

```txt
xiaoban-smoke ok
xiaoban-server-smoke ok
```

完整测试环境准备好后，再运行：

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
pip install -r requirements-dev.txt
pytest tests/xiaoban
```

## 关键边界

- 不直接读写 My Stand SQLite、WAL、file-storage。
- 不把页面路径当作长期插件协议。
- 不把源码整段发给普通用户。
- 不把 API key、token、secret 写进提示词、日志或 Git。
- 未授权用户即使知道 `moduleId.toolName` 也不能调用模块工具。
- 写操作必须有稳定 `message_id` / `idempotencyKey`，避免重试重复执行。
- 当前 MMCC seed fixtures 只保留接口、权限、dataScope 和 contextProvider；
  `agent.tools` 先保持空数组，未获批准前不自动开放模块工具。
- `InMemoryDurableReceiveStore` 和 `InMemoryIdentityDirectory` 只用于 smoke/dev，
  生产必须换 SQLite 或 My Stand 后端持久层。

## 服务器安装文档

- `docs/server-install.md`
- `docs/production-readiness.md`
- `docs/release-checklist.md`
- `docs/store-contract.md`
- `docs/connectors/web-desktop-pet-contract.md`
- `docs/connectors/cross-channel-sync.md`

## 运行底座

小伴保留成熟 Agent runtime 的核心能力，但用户可见身份是 Xiaoban/站小伴。
上游 runtime 来源只用于工程追溯和后续升级对齐，不作为用户侧身份。

本仓库首发追踪文档：

- `XIAOBAN_TRANSFORMATION_TRACKER.md`
- `docs/xiaoban-xiaoban-fork-strategy.md`
- `docs/mmcc-v0.1-agent-plugin-alignment.md`
