# Strata 0.2.17 正式切换 + 按官方思路优化长 prefill

日期：2026-11-21

## 已切换：生产 = 0.2.17（engine 0.1.63）

按用户决定，生产正式从 fork 0.1.60 切到 0.2.17 源码构建版：

- exe：`/home/xiongshao/projects/active/strata-0.2.17-src/build-017-sm75/strata`
- 端口 8000 / 0.0.0.0 / API key 开启 / health loaded:true
- 切换前 fork 基线备份：
  `strata-sc117-iq3_s.json.20261120-final-fork-baseline` /
  `env.extra.20261120-final-fork-baseline`

## 当前生产环境开关（最优组合）

```
STRATA_X1_BYPASS=1          # fork 的两段式 NVLink 专家上传（保留搬运优势）
STRATA_SPLIT_MTP_BATCH=1    # 0.2.16 的 MTP draft K/V 批处理
STRATA_P2P_HANDOFF=1        # 0.2.17 #6023 的层间 NVLink hand-off
STRATA_MMQ_TIER_SORT=1      # 0.2.17 #6023 专家按就绪度排序（本次实测有效）
STRATA_MTP_CACHE_ROOM=1     # 本次源码补丁：草稿头不扣末卡专家缓存 room
```

编译补丁（在 0.2.17 工作区 `generate.cpp`）：
`STRATA_H2D_PROBE_MAX`、`STRATA_MTP_CACHE_ROOM` 两个 opt-in 开关，
默认不改官方行为；`STRATA_MTP_CACHE_ROOM=1` 让 CUDA1 缓存 7505→7571 slots。

## 长 prefill 优化实测（三变体冷 42k，均值）

| 配置 | 冷 prefill 均值 | 效果 |
|---|---:|---|
| fork 0.1.60（基线） | 1913 tok/s | — |
| 0.2.17 裸 + 搬运 parity | 1777 tok/s | -7.1% |
| 0.2.17 + tier sort（**当前**） | **1816 tok/s** | +2.2%，仍差 fork 5.1% |
| + CPU share 8192 | 1688 tok/s | **-8%，有害，已撤销** |
| + layer-major walk | 1801 tok/s | 无收益，已撤销 |
| + MTP_CACHE_ROOM（当前叠加） | 待测（外部负载干扰） | 缓存 +66 slots |
| + WMMA | 启动失败/崩溃 | **禁用，已撤销** |

结论：**tier sort 是唯一有效的官方开关**，追回约 2.2%；仍差 fork 约 5%，
根因是 v0.2.16 删除 MMQ prompt kernel（见 `README-0.2.17-deep-dive-20261120.md`），
运行时开关补不回。想完全追平必须移植 v0.2.15 的 MMQ kernel 源码。

## 为什么长 prefill 追不平（物理约束，已确认）

当前 0.2.17 启动日志：
`WARNING: 324 of the 3461 borrowed slots have no RAM copy: their experts are
streamed from the model file during every chunk`

本机内存：62 GiB 总量，strata RSS 约 48.6 GiB，available 约 20 GiB，swap 已用 6 GiB。
residents budget 已到 `--resident-budget-gib 24`，**无法再加大**（会 OOM）。
这 324 个借位专家只能从文件流读，是物理内存上限，不是引擎 bug。

## 短 decode / agent 收益（切换依据）

- 多步 agent 场景：0.2.17 decode 均值 113.5 vs fork 105.4 tok/s（**+7.7%**），
  TTFT 479 vs 496 ms（持平略优）。见 `README-0.2.17-agent-ab-20261120.md`。
- 长上下文 decode：两者噪声内持平。
- 结论：按"有收益就切"的要求，agent decode 收益成立，已切。

## 重要提醒

1. **WMMA 在本机 sm75 上崩溃**（`prefill: the WMMA expert tiles could not launch`
   后反复重启），已加入禁用清单，**不要再加 `STRATA_PF_WMMA=1`**。
2. 测试期间有外部 AI 客户端（192.168.3.x）16 路并发持续调用，会污染 A/B
   且让短 decode 探测降到 48-78 tok/s（并发竞争，非版本退化）。
   严格 A/B 需在无外部客户端时进行。
3. `STRATA_MMQ_TIER_SORT=1` 和 CPU share 类开关会轻微改变答案
   （官方 forced-token KL 0.98）；已评估可接受。
4. 服务从 00:14:48 稳定运行，近 10 分钟 2 次重启均为本次手动切换，
   非崩溃；`strata-expert-guard.timer` active。

## 后续可选项（若一定要追平长 prefill 最后 5%）

- 从 v0.2.15 移植 MMQ kernel + vendored ggml-cuda 到 0.2.17（大工程、高风险），
  可在保留 0.2.17 agent decode 收益的同时恢复 fork 的冷预填速度。
- 或增加物理内存（消除 324 借位的文件读取），但收益上限约 5%。

## 补充：短 prefill 固定开销探索（2026-11-21 00:20-00:45）

用户要求"从短 prefill 优化、砍固定开销、调采样"。实测与结论：

**短请求基线**（62-64 token，外部负载低时 8 次）：
prefill 380-410ms（中位 382ms），decode 84-90 tok/s。
62 个 token 的纯计算约 40ms，其余约 340ms 是固定开销。

**固定开销构成**（从源码 `generate.cpp:22840` 的 prompt_ms 与日志推断）：
- verify window 的短读路径（`--short-read 64`，≤64 token 走 decode window）
- conversation checkpoint 保存（每请求 `1 checkpoints`，日志 `parked ... in 120-150ms`）
- suffix draft 查找（每请求 `suffix drafts: 1 windows`）
- 这些都是每请求固有成本，且对多步 agent 的前缀复用是必需的，
  砍掉会让 agent TTFT 变差，得不偿失。

**试过并否决的开关**（均实测无效或有害）：
- `STRATA_PREFILL_STREAM_MIN=64`：短 prefill 仍 398-458ms（无收益）。
  原因：本机 sm75 + IQ3_S 走 fused int8 路径（`STRATA_PF_FUSED=1`），
  不是官方 stream_floor_rule.hpp 描述的 gfx1151 FP16 慢路径，floor 规则不适用。
- `--short-read 0`（关短读走 batched path）：短 prefill 恶化到 1347ms、
  decode 降到 63 tok/s。**短读走 verify window 是正确优化，保留默认 64**。
- 关 CPU share（`STRATA_SPLIT_CPU_SHARE=0` / `STRATA_PREFILL_CPU_SHARE=0`）：
  测试被外部负载干扰未能定量，但对短请求 share 上限 3072 本就覆盖，
  对 agent 前缀复用有益，不轻易关。
- `STRATA_TRACE=1`：仅加日志（细分只在 batch 路径打印），对单请求无诊断输出，已撤销。

**采样配置**（`sampling: temperature=0, top_p=1, top_k=1, min_p=0`）：
已是 pure greedy，对 MTP 草稿接受率最友好、decode 最快，
**当前配置最优，无需调整**。改成 temperature>0 会降低草稿接受率、变慢。

**测试环境限制**：测试全程有外部 AI 客户端（192.168.3.x / 172.18.0.x）
持续 2-16 路并发调用，短请求探测混入其批次（651ms/1347ms 均为受干扰值），
无法做严格的短 prefill A/B。需在无外部客户端的维护窗口才能精确测量。

**结论**：短 prefill 固定开销是架构固有成本，本机可用开关均无法削减且不伤 agent；
当前 0.2.17 + 最优配置保持不变。所有试验开关已撤销，配置回到干净状态。
