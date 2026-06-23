# Xiaoban-Agent 中文说明

Xiaoban-Agent 是 My Stand 的原生服务端 Agent，中文名叫站小伴，也可以叫小伴。

它的目标不是做一个简版聊天机器人，而是在成熟 Agent runtime 的基础上，
改造成真正懂 My Stand、懂房产经纪人业务、懂模块权限、懂工具调用边界的
服务端大脑。

## 小伴应该做什么

- 解释 My Stand 是什么、怎么用、适合谁。
- 根据当前用户 ID、角色、公司关系和历史记忆识别人。
- 查 My Stand 帮助中心。
- 在授权范围内读取 My Stand 模块说明书和上下文摘要。
- 通过 MMCC v0.1 发现模块工具。
- 用户给出模块 API token 后，按权限调用模块工具。
- 接 Web、微信、飞书、企业微信、QQ 等入口。
- 对高危动作做确认、去重、审计和回滚要求。

## 当前首发状态

首发版本正在把成熟 Agent runtime 改造成小伴：

- 已新增 `xiaoban/` 原生层。
- 已新增 MMCC v0.1 manifest、校验器、权限策略和上下文 provider 选择器。
- 已新增源码披露护栏。
- 已新增 `scripts/xiaoban_smoke.py` 本地 smoke。

## 验证

```bash
python3 scripts/xiaoban_smoke.py
```

看到：

```txt
xiaoban-smoke ok
```

说明第一层小伴化骨架可用。

