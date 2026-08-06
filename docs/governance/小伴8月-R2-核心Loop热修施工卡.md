# 小伴 8 月 R2 核心 Loop 热修施工卡

日期：2026-08-06

状态：`BLOCKED_LIVE_RETEST_REQUIRED / ISOLATED_ONLY / NOT_RELEASED`

## 1. 唯一结果

普通 My Stand 请求由同一个模型持续执行：模型提出工具调用，真实 ToolResult 回到同一回合，模型继续采样，直到不再提出工具并自然给出最终答卷。代码不再用 128KB 字节数猜上下文是否可用，也不再把普通请求的固定第八次调用改造成强制收尾槽。

true MoA 的固定 final slot、身份/租户/权限、写确认、幂等、耐久投递和实际 usage 结算不在本轮改写。

## 2. Codex 对照结论

- `run_turn` 以 `model_needs_follow_up || pending_input` 决定是否继续；没有待执行工具或新输入时自然结束。
- 工具提议先进入历史，结果按原 call 顺序回灌；普通工具失败也是绑定原 call id 的 ToolResult，取消则终止 turn。
- compact 由上下文 token 状态在采样前或工具结果后触发；compact 成功后继续同一 turn，compact fatal 则结束本轮。
- transport retry 只处理可判定的传输失败并有次数上限；工具副作用结果不因网络错误盲目重派。
- Codex 没有“第八次必须 final”的业务规则；My Stand 的逐调用账本只记录和结算真实物理调用，不能替模型选择何时结束。

固定源码基线：`openai/codex@feee0b07c7564455e253312e62e6dba69dc861d3`。

## 3. 最小施工边界

1. 普通 signed loop 保留模型工具能力直至自然 `stop`；true MoA 才使用固定 final slot。
2. 工具成功、空、拒绝、失败和 unknown 均通过 canonical ToolResult 投影回模型；非法工具名和非法参数也形成有界、不可泄密的纠错结果。
3. 空响应、相同非法调用和运行时异常必须有 no-progress 停止线，不能形成无限付费循环。
4. compact 使用同一 provider/model，输入只来自模型可见投影；原始用户目标、trusted steer、verified write receipt 与 dispatched+unknown 边界不得由模型摘要替代。
5. 普通与 true MoA 共用停止 controller 解析，围住 provider dispatch、tool handler、ToolResult commit、response consume 和 final persistence。
6. Provider/transport fatal 产生独立 typed failure；它不是小伴答卷，也不得触发已执行工具重放。

## 4. 风险与回滚

- 正式服务、正式数据库和正式静态目录不动；只在隔离 worktree、临时 home 和隔离端口运行。
- 真实 DeepSeek 只执行下列六个固定场景各一次，不重复烧费；全部工具为临时只读/确定性工具，不访问正式业务数据。
- 断网前 9 文件草稿已冻结为 Git ref：`refs/backup/xiaoban-r2-loop-hotfix-pre-resume-20260806`，对象 `20165f17a97a6c549cf8528463b0c3062ca0d43f`。
- 任一权限、账本、取消或 usage 回归不通过即停止，不发布；回滚只需放弃隔离候选或从上述 ref 恢复目标文件。

## 5. 六个真实场景

1. 普通对话：无工具，自然 final。
2. 单工具成功：先说具体执行摘要，结果回灌后自然 final。
3. 六工具长链：六个确定性只读步骤各执行一次，最后自然汇总。
4. 大结果与 compact：大 ToolResult 后触发同模型 compact，工具不重放，目标不丢失。
5. 工具失败：真实 failed ToolResult 回灌，模型说明失败原因、已完成项和下一步。
6. 混合结果：success/empty/denied/unknown 的有界结果进入同一模型，唯一 final 不冒充未确认成功。

## 6. 六项交付物

1. Codex source map 与差异结论。
2. 断网现场备份 ref 与目标文件清单。
3. 最小核心 Loop diff。
4. 定向无付费测试与必要回归结果。
5. 隔离 gateway 的六场景真实 DeepSeek 脱敏结果及逐调用 usage。
6. 隔离进程/临时目录清理、未发布证明和完成条件复核。

## 7. 2026-08-06 真实门禁结果

- 离线预检通过：候选源码绑定、固定路由 `deepseek/deepseek-v4-pro`、8 个临时确定性工具、隔离端口、耐久账本与 compact 校准均正确。compact 粗估从 `83,088` tokens 跨到 `106,191`，阈值为 `97,952`。
- 实测候选为 `eb08eaf651f3346b12c019b88a2e92479349122c + dirty snapshot 8d5edf1968002db9d6ebf4454a84eeb387b6104f4cd361c2ae2fe84cfc95dda0`；隔离配置 hash 为 `55e809f0d745c4b51d4ff7bdec1094c77d78417d7c49d255e26b7a59344839f2`。
- 普通对话通过：1 次物理调用，`7,869 input / 34 output / 7,903 total`，自然 final，未调用工具。
- 单工具成功通过：首次 harness 因错误假设 tool handler 必须收到 `tool_call_id` 而误报；随后只读联查临时 state 与耐久账本确认真实路径为 1 次工具、2 次物理调用，canonical outcome=`success`，模型先给非模板执行说明，最终答复引用了只存在于 ToolResult 的 evidence。usage 为 `16,236 input / 223 output / 16,459 total`，两笔均 `completed/reported`；未重放该场景。
- 六工具长链完成了六个严格串行工具调用、六个 `dispatched/success` ToolResult、7 次物理调用和唯一自然 final，六枚隐藏 evidence 全部回到 final；usage 为 `63,982 input / 1,069 output / 65,051 total`，七笔均 `completed/reported`。
- 长链公开执行摘要验收失败：模型每轮实际生成了具体人话，但内容包含当前 query 或上一步 evidence；DLP 在完整值脱敏前先命中派生片段，丢弃模型原句，gateway 随后退回固定模板。该行为直接违反本卡第 5.2/5.3 条，门禁立即停止。
- 大结果与 compact、工具失败、混合结果三个场景从未启动，没有对应 Provider 调用。整次验收共 10 次物理调用，`88,087 input / 1,326 output / 89,413 total`；账本估算费用合计约 `$0.05076682`。

## 8. 阻断修复与停止点

- `gateway/platforms/api_server.py` 已做最小修复：在扫描派生片段前，从扫描副本移除完整受保护值；确认没有残余派生泄露后，才在正式输出中替换完整值。这样保留模型原句结构，同时继续对简称、尾号和派生标识 fail closed。
- gateway 已删除工具名驱动的固定摘要 fallback。模型没有安全可公开的人话时只保留结构化工具生命周期，不再由代码冒充小伴说固定模板。
- 定向与完整离线验证：`tests/gateway/test_api_server.py` 为 `311 passed`，可信 usage/cancel/chat-control 为 `65 passed`，`tests/gateway/trusted_action_runtime/test_web_stream_lifecycle.py` 为 `11 passed`；`scripts/xiaoban_validate.py`、Ruff、`py_compile`、`git diff --check` 通过。独立只读复审对本次 DLP/fallback 修复结论为 `release-clear`，唯一残余是尚未真实复验。
- 正式 `mystand-api.service` 与 `xiaoban-agent.service` 全程保持 `active/running`、`NRestarts=0`；正式 SQLite 的 size/mtime 前后不变。未发布、未重启正式服务、未写正式数据。
- 临时 harness、隔离 home、隔离 SQLite、日志和端口均已删除/释放；脱敏结果只保留在本施工卡。
- 当前停止点：修复后的候选尚无真实 DeepSeek 复验，六场景也未全部完成，因此不得发布或宣布完成。任何重复长链或启动剩余三个付费场景，都需要刚哥重新明确预算授权。
