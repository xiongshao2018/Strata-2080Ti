# Strata 0.2.17 / engine 0.1.63 on 2x RTX 2080 Ti (sm75, NVLink, x1 root port)

本目录记录把上游 v0.2.17 移植到本机双 2080 Ti sm75 + x1 NVLink 拓扑的补丁、
A/B 实测与生产配置。机器：SC117 IQ3_S（IQ3_S experts），GPU1 挂在 x1 根端口。

- [DEEP-DIVE.md](DEEP-DIVE.md)：为什么 fork 0.1.60 冷预填快约 7.6%——上游从 v0.2.16
  删除了 llama.cpp 的 MMQ prompt kernel；逐项排除 NVLink 搬运、PCIe 探测、专家缓存 room。
- [AGENT-AB.md](AGENT-AB.md)：多步 agent 场景实测——0.2.17 decode 113.5 vs fork 105.4
  tok/s（+7.7%），TTFT 持平。
- [PRODUCTION.md](PRODUCTION.md)：当前生产开关、长 prefill 优化结果、WMMA 在 sm75 上崩溃
  的禁用记录、短 prefill 固定开销探索与采样配置结论。
- [SAMPLING-CODE-AGENT.md](SAMPLING-CODE-AGENT.md)：采样参数与多步编程 agent——top_p/top_k 在 greedy 下无效、采样慢 10-15%；真正瓶颈是 thinking budget，建议编程请求传 reasoning_budget_tokens=128。

## 补丁要点（engine）

- `STRATA_X1_BYPASS=1`：fork 的两段式专家上传（host→GPU0 over x16→NVLink→GPU1），
  绕开 GPU1 的 x1 链路。0.2.17 上重新应用并实测生效。
- `STRATA_P2P_HANDOFF=1`：上游 #6023 的层间 hand-off 放在下一阶段 VRAM 经 P2P 写入。
- `STRATA_MMQ_TIER_SORT=1`：上游 #6023 的专家按就绪度排序，实测冷预填 +2.2%。
- `STRATA_MTP_CACHE_ROOM=1`（本次新增）：MTP 草稿头从 VRAM reserve 分配，不扣末卡
  专家缓存 room（fork 0.1.60 行为），末卡缓存 +66 slots。
- `STRATA_H2D_PROBE_MAX=1`（本次新增，默认关）：层间链路探测取各 stage 的最大值，
  避免被 x1 的 0.8 GB/s 覆盖规划；实测对预填速度无显著影响，保留供研究。

## 构建

```bash
CUDACXX=/usr/local/cuda-13.0/bin/nvcc cmake -G Ninja -S engine -B build-017-sm75 \
  -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF \
  -DSTRATA_EXPERIMENTAL_SM60=OFF -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_PORTABLE=ON \
  -DSTRATA_Q6K_EXPERTS=ON -DCMAKE_CUDA_ARCHITECTURES=75-real
cmake --build build-017-sm75 --target strata -j 12
```

警告：`STRATA_PF_WMMA=1` 在本机 sm75 上会崩溃（`prefill: the WMMA expert tiles
could not launch` 后反复重启），勿用。
