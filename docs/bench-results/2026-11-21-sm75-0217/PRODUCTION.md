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

## 补充：重启后首个请求的 15 秒（2026-11-21 16:30-16:50 实测）

用户反馈"最新日志明显退步"。复核结论：**不是退步，是重启后首个请求的冷读**。

### 现象

最近 5 次重启的首个请求全部命中同一模式：

| 启动 | 首个 prompt | 磁盘读 | major faults | decode 等待 |
|---|---|---|---|---|
| 16:13:27 | 19987t 1859 tok/s | 56402 MB | 1229 | 14912 ms |
| 16:15:12 | 21019t 2018 tok/s | 56179 MB | 1279 | 15561 ms |
| 16:17:09 | 19508t 1992 tok/s | 56177 MB | 1200 | 14867 ms |
| 16:18:58 | 19508t 1919 tok/s | 56054 MB | 1185 | 14989 ms |
| 16:21:56 | 19987t 1932 tok/s | 56931 MB | 1157 | 15159 ms |

### 成因（源码）

`engine/src/core/expert_source.cpp:4144` 的 `drop_file_cache()` 在页锁定补集前
执行 `madvise(MADV_DONTNEED)` + `posix_fadvise(POSIX_FADV_DONTNEED)`，
把模型文件的页缓存整份丢掉（启动日志："the model files' cached pages were released"）。
首个大 prompt 因此只能从 NVMe 重读约 38 GB 专家。

### 对照实验（同一进程，页缓存已热）

服务已跑满一轮后，发**内容全新**（0 reuse）的冷 prompt：

| prompt | tok/s |
|---|---|
| 21465 | 2113.8 |
| 22277 | 2136.7 |
| 22006 | 2134.1 |
| 30075 | 2278.8 / 2292.3 |
| 45139 | 2337.5 / 2314.5 |

**同样 20K 档：重启后首个 1932 → 页缓存热后 2114-2137（+9~11%）。
30K 档 2278-2292、45K 档 2315-2338，已回到 0.1.43 的 2300+ 水平。**

即：2300+ 只在 ≥30K 成立；20K 档的天花板约 2130（与历史 15-26K 档 p50=1930 相比，
热缓存下还有约 10% 余量，冷缓存则吃掉这 10%）。

### 对策：重启后预热

`~/.local/share/strata/warmup.sh`：等 `/health` loaded 后自动发一次约 20K 的
预热请求，把 15 秒冷读挪到用户感知之外。用法：

```bash
systemctl --user restart strata.service && ~/.local/share/strata/warmup.sh
```

不改引擎，无兼容性风险。

### 专家命中率与 decode（无退化）

本次 20 请求窗口：专家命中率 91.3-96.8%，drafts accepted 255/280、259/287、239/262；
decode 400@20K = 103.8 / 119.8 / 108.9 tok/s，与交接基线一致。
全历史 decode 分档 p50：0-12K 89.0、12-24K 99.2、24-40K 90.2、40-60K 88.4、
60-90K 86.9、90-300K 84.1 tok/s（n=4827），长上下文衰减平缓，无断崖。

## 补充：`--vram-reserve-later-mib 640` 与短增量固定开销（2026-11-21 17:18-18:10 实测）

### 卡1 保留空间能不能换成专家槽：能，但换不来空闲显存

`--vram-reserve-later-mib` 只控制**启动后留给其它程序**的保留量，缓存会立刻把让出的
空间吃掉。A/B（其余参数不动）：

| 值 | CUDA1 专家槽 | CUDA1 空闲显存 |
|---:|---:|---:|
| 1100（原） | 8134 | 912 MiB |
| 640（本次） | **8360（+226）** | **584 MiB** |

结论：**卡1 空闲显存不会因为调低保留而变大**——省下的 460 MiB 被 `expert cache`
当场转成 226 个槽。想“把卡1 空闲填满”在架构上做不到，除非把槽数上限再抬高，
而那会让卡1 与卡0 的缓存不对称、并把风险压到 512 MiB 驱动阈值以下。
卡1 桌面进程实测只占 162 MiB（Xorg 105 / gnome-shell 7 / awesun 11 / firefox 39），
不是桌面吃掉 1042 MiB。

保留 `640` 的理由是收益为正、风险可接受：短增量 486 → 479 ms，decode 与
prefill 均在噪声内（见下表）。回滚备份：
`install/strata-sc117-iq3_s.json.bak-20261121-1800-later1100`。

### 卡0 的 431 MiB 告警不是“没利用满”

日志 “432 MiB of the card's memory are free … Below 512 MiB the driver can move GPU …”
说的是**驱动在 512 MiB 以下会搬页**，属于性能提示而非可分配余量。卡0 的
431 MiB 已被 `expert cache` 9178 槽（16.99 GiB）+ MTP 837 MiB + pipeline 窗口占满，
再抬只会把驱动逼进搬页路径。日志自身建议“decode 正常时什么都不用做”，维持现状。

### 短增量真正的成本：不是退步，是 batched 路径的固定开销

同一会话连续追加（`/tmp/incr.py`，20K 上下文，`max_tokens=1`，greedy）：

| 新增 token | read | wall |
|---:|---:|---:|
| 20 | 48 | 406 ms |
| 63 | 96 | 505 ms |
| 128 | 168 | 580 ms |
| 256 | 310 | 703 ms |
| 512 | 601 | 911 ms |

拟合：**约 370 ms 固定 + 1.06 ms/token**。固定部分与增量大小无关，
在 65-128 档就是 `PRODUCTION.md` 早先记录的 ~400 ms（290 ms 读 + 84 ms checkpoint
+ 30 ms windows）。低于 `--short-read 64` 的尾巴走 verify window，15 token 只需 246 ms。

对照历史按“读取 token 分档”的 p50（同一日志 7918 请求）：

| read token | 11-20 | 11-21 | 变化 |
|---:|---:|---:|---|
| 0-32 | 281 ms | 319 ms | +14% |
| 32-64 | 413 ms | 560 ms | +36% |
| 64-128 | **1636 ms** | **533 ms** | **-67%** |
| 128-256 | 2171 ms | 1052 ms | -52% |
| 256-600 | 2511 ms | 1449 ms | -42% |
| 600-2k | 3191 ms | 3077 ms | -4% |
| 2k+ | 23732 ms | 6754 ms | -72% |

`2k+` 档的 23732 → 6754 主要是 11-20 那天混入了大量冷缓存请求，不能直接当性能提升；
但 64-600 三档是**实打实的改善**（`STRATA_SPLIT_SMALL_OWN=512` 免掉 lend/stream/refill），
所以“最新日志明显退步”在数据上不成立。

### 仍然存在的三个痛点（按性价比排序）

1. **321 个借用槽没有 RAM 副本**：每个 chunk 多读约 0.83 GB。与 `SMALL_OWN`、
   `--resident-headroom-gib` 都无关，补集被 `choose_resident_keep_from` 卡在 19.07 GiB。
2. **双卡下 SSD 溢出层被静默禁用**：`generate.cpp:13500` 的条件是
   `conversations.enabled() && stages.empty()`，分层双卡 `stages` 非空 → 永不启用，
   日志无任何提示；多会话切换的 evictions 是硬丢。
3. **重启后首个请求冷读约 15 秒**：`expert_source.cpp:4144 drop_file_cache()`
   在页锁定补集前丢整份页缓存。已用 `warmup.sh` 规避。

### 本轮配置结论

- 保留 `--vram-reserve-later-mib 640`（+226 槽，短增量 -7 ms）。
- 保留 `STRATA_SPLIT_SMALL_OWN=512`（去掉后短增量 611/609/606 ms，退化 25%）。
- 不加 `STRATA_RESIDENT_HEADROOM_GIB`（补集仍是 19.52 GiB，无效）。
- 不动 `--short-read`（默认 64 是当前最优：90 token 走 batched，15 token 走 windows）。
- 卡0 不加 `--vram-reserve-mib`，接受 431 MiB 告警。
