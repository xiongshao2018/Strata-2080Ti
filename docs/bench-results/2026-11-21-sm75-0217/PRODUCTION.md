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

### A/B：关掉双卡 CPU share —— 明确变差，已回滚

上一版 `PRODUCTION.md` 把 `STRATA_SPLIT_CPU_SHARE=0` 记为“测试被外部负载干扰未能定量”。
本轮在无外部客户端时补测（同一 `/tmp/incr.py`，20K 上下文）：

| 新增 token | share 开（生产） | share 关 |
|---:|---:|---:|
| 20 | 406 ms | 448 ms |
| 63 | 505 ms | 607 ms |
| 128 | 580 ms | 700 ms |
| 256 | 703 ms | 821 ms |
| 512 | 911 ms | 1032 ms |

share 关掉后短增量**全线慢 10-20%**，且越长越明显。结论：双卡默认开启的 CPU share
对 agent 前缀复用是净收益，**必须保留**。`env.extra` 已回滚到不含
`STRATA_SPLIT_CPU_SHARE` / `STRATA_PREFILL_CPU_SHARE` 的状态
（备份 `env.extra.bak-20261121-1730-nocpushare`）。

### 冷预填吞吐随长度单调上升（20K 是天花板，不是退步）

同一进程、页缓存已热、内容全新（0 reuse）的冷 prompt，本轮实测：

| prompt | tok/s |
|---:|---:|
| 20311 | 2008 |
| 27060 | 2241 |
| 37186 | 2313 |
| 54061 | **2413** |

**2300+ 需要 ≥30K**；20K 档约 2000，25K 档约 2200。这与 0.1.43 的“2300+”不矛盾——
0.1.43 的 2300+ 也是长 prompt 数字。所以“20-30K 冷预填只有 2100”不是退步，
而是长度本身摊不薄每 chunk 的专家遍历成本（b/a ≈ 5000）。

要再往上抬只能加长 chunk（`--prefill auto:16384`）或提高 pipe 比，
但 16K chunk 会让借用槽从 3408 涨到约 6800，`--resident-budget-gib 24` 下的
RAM 副本补集跟不上，反而触发“无 RAM 副本”流式读。**当前 8192 chunk 是最优折中。**

## 补充：按引擎版本核对“是否退步”（2026-11-21 18:00-18:45 日志全量统计）

“最新日志明显退步”这个判断，用全量日志按引擎版本分段统计后**不成立**。
切分依据是引擎日志里的 `session is up (engine X)`，共 181 段 / 8328 条请求。

decode 只统计 `gen >= 64`（`gen=1` 是固定开销，混进去会把数字压低）：

| 版本 | decode p50 (gen≥64) | 全量预填 p50 | 8-16K | 16-32K | 32-64K | 64K+ |
|---|---:|---:|---:|---:|---:|---:|
| 0.1.56 | 88.6 (n=6662) | 2039.6 (n=177) | 1217 (n=9) | 1402 (n=10) | 1955 (n=24) | 2058 (n=134) |
| 0.1.62 | 92.9 (n=28) | 1771.8 (n=25) | — | — | 1772 (n=25) | — |
| 0.1.63 | **90.2** (n=402) | 1714.1 (n=537) | **1423** (n=307) | **1982** (n=97) | **2179** (n=107) | **2452** (n=26) |

**逐档对比 0.1.63 全部更快。** 全量预填的“总 p50 变低”是采样偏差：
0.1.63 的样本 307/537 落在本来就最慢的 8-16K 档，而 0.1.56 的样本 134/177 落在 64K+ 档。
把同一长度档放在一起比，结论正好相反。

复现脚本：`/tmp/ver.py`（按 `session is up` 切 boot，过滤 `gen>=64`）。

## 补充：prefill 曲线与 `--prefill-pipe` 复核（本轮实测，生产配置 pipe=5000）

重启回生产配置后的冷预填（`reused` 之外的纯 read）：

| prompt | tok/s | | prompt | tok/s |
|---:|---:|---|---:|---:|
| 2308 | 1169 | | 18058 | 1986 |
| 4558 | 1360 | | 22557 | 2169 |
| 6808 | 1515 | | 27058 | 2213 |
| 10183 | 1688 | | 36058 | 2246 |
| 13558 | 1881 | | 45057 | 2289 |

重启后复测（本次）：9K 1507、13.5K 1960、18K 2071、27K 2272、36K **2349** tok/s。

9-18K 弱档的成因是三者叠加，不是内核退化：`--prefill-pipe 5000` 只切 2 块（流水线气泡）、
末尾 refill（`refilled 2707 slots in 527 ms`）、尾部 4 token 的 batched 读
（`read 4 tokens (batched) in 651 ms`）。9K 档 trace 里 4 个 chunk 合计约 4.2 s，
`read` 总 5.39 s，**约 1.19 s 发生在最后一个 chunk 之后**。

`--prefill-pipe` 更细分块是负收益，已 A/B 否定：

| pipe | 9K tok/s | 27K tok/s |
|---|---:|---:|
| 5000（生产） | **1655** | **2192** |
| 200 | 1341 | 1834 |

更细的块（1280/1536/2048/2304）让专家重复流式读变多，**慢 15-20%**。`pipe=0` / `pipe=1`
与 5000 几乎相同。**保持 5000。**

## 补充：`--short-read` 保持 64（修正上一版判断）

上一版记的“64 token 拐点、超过 64 走慢路径要改大”是**解析错误**造成的误判：
当时用 `m[-1]` 只取到了末尾 4 token 的尾部窗口，不是整段读。

76K 上下文、95 token 增量，各重复 6 次：

| short_read | 总耗时 (ms) | p50 |
|---|---|---:|
| 64（默认） | 1044, 1037, 1019, 1027, 1023, 1037 | 1032 |
| 128 | 985, 997, 1011, 1000, 1041, 1035 | 1006 |

差 26 ms，噪声级。`short_read=0`（纯 batched）明显更慢（78 token：892 ms vs 704 ms）。
**保持默认 64，不要改。**

## 补充：真正的痛点——5.9% 的请求吃掉 53% 的预填时间

增量请求（`reused>0`）共 6832 条：

| read 阈值 | 条数 | 占比 | 总耗时 | 占全部 prompt 时间 |
|---|---:|---:|---:|---:|
| ≥1024 | 1060 | 15.5% | 14108 s | 64.5% |
| ≥4096 | 403 | **5.9%** | 11653 s | **53.3%** |
| ≥8192 | 310 | 4.5% | 11072 s | 50.6% |

按 prompt 大小看 ≥4096 重读的占比：8-16K 8.3%、16-32K 9.2%、**32-64K 15.3%**、64K+ 4.7%。

线性拟合（prompt≥32K、read≤2048、n=5256）：`ms ≈ 983 + 2.123 × read`，
**固定开销 983 ms**。增量延迟按 read 分档（prompt≥32K）p50：

| read | 1-33 | 33-65 | 65-129 | 129-257 | 257-513 | 513-1025 | 1025-4097 |
|---|---:|---:|---:|---:|---:|---:|---:|
| p50 ms | 298 | 450 | 1630 | 1859 | 2157 | 2733 | 3663 |

也就是说：日常 agent 场景里真正贵的是**少量前缀失效导致的大段重读**，
而不是短增量本身。优化方向应该是“让前缀尽量不失效”，而不是再调 chunk 大小。

## 补充：内存压力（比显存更值得关注）

```
MemTotal 62.6 GiB / Available 18.1 GiB
Swap 8 GiB / SwapFree 604 kB      ← swap 已 100% 用满
strata-host RSS 52.5 GB, VmSwap 0 ← 不是它被换出
```

swap 被其他进程占用：Isolated Web Co 919 MB、java 869/858 MB、node 560 MB、cc-pocket 552 MB。

会话缓存 8 GiB 只够 2-3 个 76K 会话（实测 bytes/token p50 = 47238 → 76K ≈ 2.5-3.4 GiB，
128K ≈ 5.8 GiB）。park 数量分布：1 个 1456 次、2 个 497 次、4 个 345 次，
用户常开 4-6 个会话 → 8 GiB 会互相挤。

**但 Available 只剩 18 GiB 且 swap 已满，盲目调大 `--conversation-cache-mib` 有 OOM 风险。**
要加需先腾内存（关 java/node/cc-pocket 等），或小幅加到 10-12 GiB 观察。

## 补充：卡1 挤不出可观的专家（结论：不动）

当前 21569/22528（卡0）、21538/22528（卡1）。启动日志：卡0 剩 646 MiB、卡1 剩 584 MiB。

卡1 桌面进程约 230 MiB（Xorg 105 + awesun 11 + firefox 39 + gnome-shell 7），
即使全部移到卡0，按 2 MiB/专家算约 115 个槽，对 9178 个专家的缓存命中率影响 **<0.5%**。
引擎日志自身也建议：“Below 512 MiB the driver can move GPU memory to system RAM …
**If decode speed is normal, nothing needs to be done.**”

`--vram-reserve-mib` / `--vram-reserve-later-mib` 调低省下的空间会被 expert cache 立刻吃掉（已验证）。
**投入产出比极低，保持现状。**

## 补充：`--prefill auto:16384` 纠正上一版结论（2026-11-21 22:30-23:30 实测）

上一版写“16K chunk 会让借用槽涨到约 6800，RAM 副本补集跟不上，反而触发流式读”，
本轮在 `--resident-budget-gib 24` + `--vram-reserve-later-mib 512` 下实测**不成立**：

| chunk | 9K | 27K | 36K | 76K | 76K decode |
|---|---:|---:|---:|---:|---:|
| 8192（旧） | 1454.5 | 2150.5 | 2222.6 | 2395.9 | 61.5 |
| **16384（现）** | 1524.2 | 2129.1 | **2265.2** | **2585.7** | **63.8** |
| 32768 | 1091.4 | 1278.4 | 1371.2 | 2377.9 | 64.3 |

16384 让 76K 冷预填 +7.9%（2396→2586），decode 也微升；32768 因借用槽 9.77 GiB、
4483 槽无 RAM 副本，中小 prompt 崩到 1091-1371，**不可用**。
**结论：`STRATA_PREFILL_AUTO_MAX=16384` 是当前最优点，32768 是悬崖。**

96K 长 prompt 复测（3 次）：2649.8 / 2571.2 / 2576.4 tok/s，均值 2599。
117K 单轮冷预填 2522.3 tok/s、续写 2255.4 tok/s（235K 上下文）。

## 补充：`--layer-split` 与“卡0 塞更多专家”的实测边界

卡0 在 `--layer-split 30` 下已是 **9183/9183 满载**（该卡 30 层的全部 profile 专家都在显存），
所谓“卡0 没利用满”不成立；真正的缺口只在卡1（8320/9216）。
想把缺口补上只能把层挪给卡0：

| layer-split | 卡0 slots | 卡1 slots | 76K 预填 | decode | read=172 |
|---|---|---:|---:|---:|---:|
| **30（生产）** | 9183（满） | 8320/9216 | **2575.9** | **62.1-63.8** | 1151 ms |
| 31 | 9194（满） | 8251/8704（满） | 2550.3 | 56.0 | 1132 ms |
| 32 | 9066 | 8192/8192（满） | 2494.5 | 61.3 | 1032 ms |

31/32 确实让卡1 也 100% 常驻，但卡0 少 1-2 层、且跨卡流水线失衡，
decode 掉到 56.0（31）或 61.3（32），76K 预填也掉 1-3%。
**结论：`--layer-split 30` 仍是最优；显存“填满”不等于更快，瓶颈是流水线配平。**

`--vram-reserve-later-mib` 640→512 只多出 56 个槽（8259→8320），
实测 76K 预填 2583.0 vs 2585.7（噪声内），decode 63.4 vs 63.8（噪声内）。
保留 512 只为多 56 个专家，属无害微调。

## 补充：`--short-read` 与 `--prefill-pipe` 二次确认（本轮）

`--short-read 0`（全 batched）在 0.1.63 上**依然明显更差**：

| 增量 read | 46 | 61 | 108 | 172 | 271 |
|---|---:|---:|---:|---:|---:|
| short-read 64（默认） | 477 | 559 | 1053 | 1151 | 1244 ms |
| short-read 0 | 1528 | 1564 | 1636 | 1762 | 1891 ms |

0.1.63 的 batched 路径虽已从 2.5 s 固定开销降到约 5-10 ms/token，
但窗口路径（约 16 ms/token 含 miss）在 ≤271 token 仍全面胜出。**保持默认 64。**

`--prefill-pipe 1700`（按 b/a 实测拟合值下调）复测：27K 1979.1、36K 2085.1，
比 5000 慢 3-5%，说明真实 b/a 接近 5000 而非 1700。**保持 5000。**

## 补充：常用场景实测（2026-11-21 23:30）

| 场景 | 实测 | 说明 |
|---|---|---|
| 40K 冷建会话 | 2406.8 tok/s | 单轮 42.4 s，含 1342.8 ms park |
| 40K 续写 2.5K 增量 | 3710 ms | 677.3 tok/s，`reused=118464` |
| 40K 分支改写 2.5K | 3385 ms | 739.8 tok/s，与续写同量级 |
| 3×20K 并发 | 92.4 s 总墙钟 | 单个 31-92 s，串行化明显 |
| 117K 冷建 | 2522.3 tok/s | 单轮 46.7 s |

痛点结论：

1. **并发是最大痛点**：3 个 20K 并发总墙钟 92.4 s，接近串行（31+62+92），
   第二、三个请求在排队。多会话并行没有收益，`--pool-workers 15` 也救不了。
2. **分支改写已不是痛点**：40K 分支 3385 ms 与续写 3710 ms 同量级，
   `rewind-keep 4096` + 6 槽会话缓存生效，不再有整段重读。
3. **小 prompt 弱档仍在**：9K 1524 tok/s 是单 chunk 无流水重叠的物理结果，
   `pipe=1700` 反而更差，没有便宜的解法。
4. **上下文上限**：262144 是硬上限，3×18K 轮次即触顶返回 400，需客户端自己裁剪。

## 关键修复：短会话 agent 的 `min-tokens 2000` 门槛（2026-11-22）

用户反馈“agent 多步场景慢于 1500 ms”属实。根因**不是引擎退化，也不是
`short_read`/`pipeline-windows`/`layer-split`，而是 18:46 配置引入的
`--conversation-cache-min-tokens 2000`**：1.4-1.7K 的短会话被会话缓存排除，
日志反复出现 `skip parking (1826 tokens, below the 2000 minimum)`，
每一步都 `reused=0` 全量冷读，单步 1.5-1.9 s。

历史 0.1.56/0.1.63 稳定期没有这个参数（默认 0 = park any），
11-21 22:51 启动的短会话 `reused=0` 比例高达 100%，正是门槛造成。

改为 `--conversation-cache-min-tokens 0` 后的 5 步短会话 A/B（3 次 trial）：

| 配置 | 第 1 步 | 第 2-5 步 | `reused` |
|---|---:|---:|---:|
| 2000（旧） | 2632-2841 ms | 1089-1247 ms | 前 1 步 0，后续约 2500 |
| **0（现）** | **779-931 ms** | **1025-1194 ms** | 第 2 步起 2531 |

首步冷读降低 65%，多步 agent 不再出现 >1500 ms 的 prefilling；
长会话续写未退化（76K read=271 从 1244 → 1237 ms），
76K decode 66.9 tok/s（此前 62.1），冷预填 2608 tok/s（噪声内）。

08:54 最终状态：`--conversation-cache-min-tokens 0` 已写入生产 JSON。

否决的候选（均实测无效或有害，**不要再加**）：

- `--short-read 256` / `512`：read=172 从 1151 ms 涨到 1566/1600 ms，
  read=271 从 1244 ms 涨到 1290/2554 ms。窗口路径覆盖过长会拖慢。
- 运行时 `STRATA_PIPELINE_SWITCH=pw=0`：read=61 从 ~560 ms 涨到 790 ms。
- `--conversation-cache-slots 1`：对受控短会话无改善（read=61 反而 634 vs 560 ms）。

**生产配置最新差异**：仅把 `--conversation-cache-min-tokens 2000` 改为 `0`；
其余保持 `--vram-reserve-later-mib 512`、`STRATA_PREFILL_AUTO_MAX=16384`、
`--layer-split 30`、`--prefill-pipe 5000`、`--short-read` 默认。
