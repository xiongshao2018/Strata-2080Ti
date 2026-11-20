# 0.2.17 更新思路 + 多步 agent 场景 TTFT/decode 实测

日期：2026-11-20

## 一、0.2.17 到底更新了什么（从 git log 读源码）

v0.2.16→0.2.17 没有独立 release notes，权威信息就是提交历史。设计主线印证了
"砍/绕开专家搬运"：**让专家少跨 PCIe、让空闲 CPU 分担、让第二张卡直接算路过的专家**，
目标机器是**显存小、专家放不全、PCIe 链路慢**的多卡配置，而不是本机这种
"22 GiB 专家基本都能常驻"的大 RAM + sm75 场景。

关键提交：

- `240aeb985` / `5b1725f0a`（PR #6023，0.2.16 起）：MMQ-free 构建上的
  NVLink P2P peer prefill、stage hand-off、P2P decode scatter、slot read-ahead/wide、
  MMQ tier sort（按 VRAM/pinned RAM/file 就绪度排序专家）。官方把这些改成
  `STRATA_*-HANDOFF / P2P-DECODE-SCATTER / SLOT-READ-*` opt-in 开关，因其收益
  只在 2x V100-SXM2 NVLink 上测过，默认全关。
- `38e6a0506`：双 CUDA 卡默认打开 CPU prefill share（空闲 CPU 池接管本该走
  PCIe 的专家），官方数据：2x RTX 5060 Ti 16G 上 32-token prompt 605→268 ms、
  566-token 3188→1574 ms；4096+ 超出 share 上限（+2%）。
- `a620ffb61` / `595d40922`：CPU share 的 limit/expert 宽度按 PCIe 链路自动算
  （只对 **cc 12.0** 卡默认开，`STRATA_PREFILL_CPU_SHARE_AUTO=0` 关）。
  RTX 5090 x2 官方数据：84-token prompt 843→327 ms；agent 场景 decode
  **136.3→138.2 tok/s**（+1.4%）。KL 检查：forced-token KL 0.98（有轻微答案差异）。
- `ca99e886f`：`STRATA_ADAPT_FETCH=2` 在 cc 12.x 且 ≤16 GiB 卡默认开，
  官方数据 decode +4~43%（中位 1.09~1.33x），但 layer split 下自动关。
- `a0a933ddf`（#6084，0.2.17）：MTP 草稿头在 stage0 时不该从末卡专家缓存里扣
  room——这就是我深挖时发现的 `- draft_head` 那段，官方 0.2.17 已修，需
  `STRATA_MTP_STAGE0=1`（HIP gfx12 默认）才触发；本机 sm75 不走。

小结：0.2.17 的更新对**本机 sm75 + 双 2080Ti + 22 GiB 专家常驻**几乎都不触发；
而且 v0.2.16 起官方删掉了 MMQ prompt kernel（见 deep-dive 文档），冷预填反而慢 7.6%。

## 二、多步 agent 场景实测（针对用户问题）

脚本 `/tmp/bench-agent.sh`：system ~2.6k token、连续 6 步、每步追加工具结果、
前缀高度复用（2646-2850 token reused，新增 50-60 token）、每步生成 120 token。
两边都是 `--prompt-cache 4 --conversation-cache 8192` 的生产参数，且预热到稳态。

| 引擎 | 样本数 | TTFT/prefill 均值 | decode 均值 | decode 中位 |
|---|---:|---:|---:|---:|
| fork 0.1.60（生产） | 11 | 496 ms（中位 417） | **105.4 tok/s** | 103.0 |
| 0.2.17 + 全套补丁 + NVLink | 6 | 479 ms | **113.5 tok/s** | 116.5 |

**结论**：

- **TTFT：两者接近**，0.2.17 略低约 3-8%（都在 400-700 ms 的噪声带），
  没有官方宣传的那种"砍专家"数量级收益——因为本机专家基本都在缓存里，
  没有多少专家可砍；agent 前缀复用已把 prefill 压到只算几十个新 token。
- **decode：0.2.17 高约 7.7%（113.5 vs 105.4）**，且更稳定（5/6 样本 ≥112），
  fork 波动大（91.7-129.6）。这与官方 `595d40922` 报告的 agent decode 小幅提升
  方向一致（它报 +1.4%，本机更大可能含 MTP 接受率/缓存状态差异）。
- 0.2.17 测试时 env 里开了 `STRATA_ROUTE_TAIL_SKIP=7`（跳过低排名未命中专家，
  答案会略有差异）；fork 生产是内置默认 tail skip。此开关可能贡献部分 decode 收益，
  但非版本本身独占——**要在生产采用需先确认答案质量可接受**。

## 三、综合建议（结合上一份 deep-dive）

- 冷预填/长上下文首 token：fork 快 7.6%（MMQ kernel），0.2.17 退化。
- 多步 agent 稳态 decode：0.2.17 快 ~8%，TTFT 持平。
- decode 收益是否值得切换，取决于负载：若绝大多数请求是**短 prompt + 多步复用 +
  高 decode 占比**，0.2.17 有小幅优势；若有**长 prompt 冷启动**，fork 明显更优。
- 官方新版的价值主要在小显存/慢 PCIe 机器和 bug 修复；本机采样结果不支持
  "全面升级收益"。**当前保持 fork 0.1.60**（长上下文与稳定性优先）。
  若以后 agent 吞吐成为首要目标，可再评估 0.2.17 + 全套补丁，但需验证
  `ROUTE_TAIL_SKIP=7` 的答案质量与长上下文退化。

## 四、状态

生产已恢复 fork 0.1.60，端口 8000、API key、专家守护均正常，无测试残留。
