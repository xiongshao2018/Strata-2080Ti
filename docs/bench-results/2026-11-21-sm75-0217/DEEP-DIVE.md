# Strata 0.2.17 预填深挖：为什么 NVLink 搬运移植了仍不如 fork

日期：2026-11-20 深夜
工作区：`/home/xiongshao/projects/active/strata-0.2.17-src`（tag v0.2.17）
产物：`build-017-sm75/strata`（engine 0.1.63）

## 最终结论（源码级）

**fork 0.1.60 冷预填快约 6~8% 是构建期差异，不是运行时开关能补的。**
根因：官方从 **v0.2.16 起从源码树删除了 llama.cpp 的 MMQ prompt kernel**，
而 fork 0.1.60（源自 v0.2.15 时代）仍保留 MMQ。

- 关键提交：`3e26b4e15 [Engine] remove llama.cpp's MMQ build path and the vendored ggml-cuda (own-stack step 10)`
- `engine/CMakeLists.txt:105` 明确写：`llama.cpp's MMQ prompt kernels are no longer in the tree (0.2.16)`
- v0.2.15 有 `src/prefill/mmq_kq_lib.cu`（真实 MMQ kernel）
- v0.2.17 只剩 `mmq_absent.cpp` / `dense_mmq_notice.hpp`（占位/提示）
- 二进制符号数：fork 约 2500 个 `mmq` 符号，0.2.17 仅 309 个

要把 0.2.17 做到 fork 的预填速度，需要把 v0.2.15 的 MMQ kernel 与 ggml-cuda
vendored 路径整体移植回 0.2.17——大工程、高风险，**本次不做**。

## 决定性 A/B 数据（同一脚本、同一文本、同进程内、冷预填）

脚本 `/tmp/bench-long.sh <nonce> 40 <variant>`，3 个变体文本不同不复用 prompt cache，
约 41839 token，各自独立冷读。

| 引擎 / 配置 | 变体0 | 变体1 | 变体2 | 均值 |
|---|---:|---:|---:|---:|
| fork 0.1.60（生产） | 1945.6 | 1886.9 | 1905.5 | **1912.7 tok/s** |
| 0.2.17 + 全套补丁 + NVLink | 1799.5 | 1771.4 | 1759.4 | **1776.8 tok/s** |
| 0.2.17 + MTP_CACHE_ROOM 补丁 | 1742.4 | 1775.4 | 1775.0 | **1764.3 tok/s** |

fork 稳定快约 7.6%，每轮都快，非噪声。**没有发现任何让 0.2.17 反超的配置。**

## 逐项排除（实测，全部无效）

1. **NVLink / x1 bypass 搬运**：0.2.17 已移植 `h2d_expert` 两段式上传，日志确认
   `layer split hand-off over NVLink (peer access on)`，`STRATA_X1_BYPASS=1` 生效。
   搬运路径正确，不是差距来源。
2. **PCIe 探测速率**：发现 fork `g_last_h2d_gbps = max(...)`（取 x16=12.4），
   0.2.17 退化为 `= bw`（被 x1=0.8 覆盖）。加了 `STRATA_H2D_PROBE_MAX=1` 开关，
   规划值 0.8→12.4 GB/s，但预填 **1764.9 → 1768.3 tok/s**（噪声内，**无效**）。
3. **专家缓存 room / MTP 草稿头**：发现 0.2.17 `room = stage_room(...) - draft_head`
   （fork 不减），导致 CUDA1 缓存 7505 vs fork 7737 slots、resident RAM 21.68 vs
   22.70 GiB、预填每轮读 324 blobs 文件。加 `STRATA_MTP_CACHE_ROOM=1` 后
   CUDA1 缓存 7505→7571，但预填仍 1764 tok/s（**无效**，且末卡 VRAM 余量下降）。
4. **文件层 I/O**：0.2.17 首请求的 `decode waited 15985 ms` 只是冷缓存填充，
   预热后消失；稳态两边都是 `0 blob reads from the file`。非退化根因。

小结：这些现象（缓存少、读文件、PCIe 规划保守）都是共因的**表征**——新版
`strata_pf_own` prompt kernel 路径对专家缓存/调度需求不同，但都不是速度瓶颈本身；
真正的吞吐差来自 prompt kernel 实现（MMQ vs own-stack）。

## decode 结论（不变）

短请求 decode 两者都在噪声内（0.2.17 约 91.1，fork 约 92.0 tok/s）。
0.2.17 在 decode 上没有可拿的收益；历史 0.2.16 评估亦如此。
用户关心的 "decode 从 94+ 掉下来" 与引擎版本无关，是专家缓存命中率退化，
已由 `strata-expert-guard.service` 守护处理。

## 本次新增的源码修改（均在 0.2.17 工作区，未采用）

- `generate.cpp`: `STRATA_H2D_PROBE_MAX`（max vs last probe）
- `generate.cpp`: `STRATA_MTP_CACHE_ROOM`（draft head 是否扣缓存 room）
- 原有 0.2.16 sm75 补丁（x1 bypass + NVLink hand-off）
- 备份：`/home/xiongshao/backups/strata/generate.cpp.20261120-pre-mtpcache.patch`

这些改动都保留了 opt-in 开关，默认不改变官方行为，可作为以后移植 MMQ 的基础。

## 生产状态（已恢复）

- 引擎：fork 0.1.60（`components/engine/0.1.60-linux-x86_64-cuda/strata`）
- 端口 8000 / 0.0.0.0 / API key 开启 / health loaded:true
- `env.extra` 与 config 均为 `*-probeab-base` 干净态，无测试残留
- `strata-expert-guard.timer` active

## 若未来仍想升级 0.2.x 的建议

1. 把 v0.2.15 的 `src/prefill/mmq_kq_lib.cu` + `moe_mmq.cu` + vendored ggml-cuda
   移植到 0.2.17，恢复 MMQ prompt 路径；A/B 达标后再切。
2. 或者继续用 fork 0.1.60，它已是本机 sm75 + x1 NVLink 拓扑下的最优解，
   官方新版的价值在功能/修复，不在本机 decode/预填速度。
