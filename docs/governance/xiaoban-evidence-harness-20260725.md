# My Stand 治理主线任务卡

## 主线

- RUN_ID：xiaoban-evidence-harness-20260725
- PROGRAM_MAP_ID：xiaoban-agent / My Stand API channel
- MAIN_GOAL：真实站内资料回答必须先取得本轮工具证据；写入守卫不得误伤读取。
- CURRENT_WAVE：证据执行闸门
- WAVE_STATUS：进行中
- CURRENT_STAGE：MODULE_GATE
- CURRENT_ITEM（唯一进行中）：真实 K3 验收
- NEXT_ACTION（只写一条）：用正式授权 ID 走网站网关真实读取。
- PASS_TO_ADVANCE：工具被强制调用，回复与授权结果一致；失败时出口拒绝编造。
- DO_NOT_TOUCH：My Stand API、数据库、权限模型、提示词主体、其他 Agent 通道。
- BACKUP_ID：发布前生成
- START_HEAD：890e592
- REPLAN_COUNT：0
- 当前模块波次：1 / 1
- 当前模块是否已关闭上一波：是

## 波次范围

- 允许修改的模块/调用链：API toolset、首次模型调用、My Stand 出口守卫。
- 允许路径：`gateway/platforms/api_server.py`、`gateway/mystand_integrity_guard.py`、`agent/chat_completion_helpers.py` 及对应测试。
- 明确非目标：不新增索引系统，不改业务数据，不改权限，不增加长提示词。
- 当前业务不变量 ID：tenant-isolation；authorization-wall；verified-write-receipt。
- 相邻回归范围：API Server、AUTH 读写桥、资源查询、流式/非流式出口。

## Diff 预算

| 类别 | 预计 | 实际 | 结果 |
| --- | --- | --- | --- |
| 生产源码 | 3 文件，净增长不超过 300 行 | 3 文件，预算内 | 通过 |
| 测试 | 对应单测与 API 回归 | 391 项通过 | 通过 |
| 文档 | 本任务卡与发布日志 | 1 张任务卡 | 通过 |
| 生成物/lockfile | 不允许 | 0 | 通过 |

## 阶段门禁

- [x] 失败项或行为基线明确。
- [x] 最窄测试通过。
- [x] 模块测试通过一次。
- [x] 当前范围无已知 P0/P1。
- [x] diff 未超预算，无新抽象或目录扩张。
- [ ] 生产备份、真实 K3 与正式站路径通过。
- [ ] 当前项关闭并写回唯一 NEXT_ACTION。
