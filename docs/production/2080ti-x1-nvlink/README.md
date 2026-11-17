# 2x RTX 2080 Ti sm75 NVLink 生产配置

- **机器**: Intel i9-14900KF + 64 GiB RAM + 2x RTX 2080 Ti (sm75) + NVLink bridge
- **GPU1 硬件限制**: PCIe Gen2 x1 (0.8 GB/s), 无法修改——`STRATA_X1_BYPASS=1` 让数据走 GPU0 x16 → NVLink → GPU1
- **引擎**: 本地编译 fork (commit `3949f5238`, sm75-x1-bypass branch), engine 0.1.60
- **模型**: SC117 IQ3_S (Qwen3.8-Flash-Next, 282B MoE), 262144 context
- **Vision**: `gpu: false` (走 CPU, 不占 GPU 显存)

## 性能实测 (warm, greedy, temperature 0.0)

| 场景 | 速度 | TTFT | 备注 |
|---|---|---|---|
| 64K 冷 prefill | **2101 tok/s** | — | 目标 ≥ 1950，已达 |
| 2K decode | **76.7 tok/s** | 0.09s | 目标 ≥ 75，已达 |
| 8K decode | **77.4 tok/s** | 0.25s | 目标 ≥ 75，已达 |
| 32K decode | **69.0 tok/s** | 0.49s | draft_acc 73% |
| 64K decode | **69.0 tok/s** | 0.71s | draft_acc 77% |
| Agent 短增量 (每 turn +50 tok) | **~590ms** | TTFT | conversation cache 95%+ 复用 |

## env.extra 关键开关

| 开关 | 作用 | 实测收益 |
|---|---|---|
| `GGML_CUDA_P2P=1` | 双卡 P2P (NVLink 前提) | 必须 |
| `STRATA_SM75_PERF=1` | 全套 Turing 特化 kernel | decode +3.4-5.0%, prompt +4.2% |
| `STRATA_BF16_TC=1` | BF16→FP16 转换后跑 FP16 TC MMA | prompt **+15-18%** |
| `STRATA_X1_BYPASS=1` | GPU0→NVLink→GPU1 绕过 PCIe x1 | 必须，不然 GPU1 只能吃 0.8 GB/s |
| `STRATA_NVLINK_STAGE=1` | NVLink staging buffer | sm75-x1-bypass fork 内部使用 |
| `STRATA_EXPDEC=1` | expert 窗口化 decode kernel | bitwise 等价，减少等待 |
| `STRATA_SELECT_SIMT=1` | FP32 tiled block scorer | sm75 没有 TF32, 必须 |
| `STRATA_QSA_ROWS=1` | QSA row optimization | +3.2% (lingdeng 测) |
| `STRATA_EXP_MODE=8` | Volta expert 布局 (SwiGLU + q8_1 fused) | **decode +7-10%**, 刚刚验证 |
| `STRATA_SM70_TABLE=1` | 让 sm75 也能用 mode 8 | sm75 默认 mode 4 在大窗口自动回退 mode 0 |

## strata-sc117-iq3_s.json 关键参数

| 参数 | 值 | 说明 |
|---|---|---|
| sampling.temperature | **0.0** | greedy, agent 不需要多样性, draft acceptance +6% |
| sampling.top_k | **1** | greedy |
| sampling.min_p | 0.0 | agent 不需要 |
| --spec-min-p | **0.6** | draft acceptance 从 64% 升到 80-86% |
| --resident-experts | (开关) | 配合 --resident-budget-gib 16 |
| --resident-budget-gib | 16 | 16 GiB experts 常驻 page-locked RAM |
| --conversation-cache-mib | 8192 | agent 多轮会话复用 70%+ |
| --conversation-cache-slots | 4 | agent 多轮会话槽位 |
| --prompt-cache | 12 | 分段缓存 |
| --prompt-cache-every | 4096 | 每 4096 token 一个缓存点 |
| --expert-cache | auto | 引擎根据显存自适应 (8838 slots / 16.36 GiB) |
| --layer-split | 30 | GPU0 0-29, GPU1 30-47 |
| --kv | int8 | 一半 fp16 显存, 质量可接受 |
| --mtp | /home/xiongshao/strata-models/mtp/rt | MTP draft layer 开启 |
| --spec | 4 | spec decode 深度 |

## 已排除的方案 (实测无用 / 不适用)

| 方案 | 排除原因 |
|---|---|
| `STRATA_PF_FUSED_SM75=1` | 只对 Q2_0/Q2_0 expert pack 适用, IQ3_S 不适用 |
| `STRATA_PREFILL_HELP=1` | 2K decode 掉 4.5 tok/s, host RAM 被 resident experts 挤爆 |
| `STRATA_LOCAL_PREFILL_RELAY=1` | fork 自己的 h2d_expert 已经在做同一件事 |
| `--kv fp16` | KV cache 翻倍, GPU1 只剩 1.7 GiB free, 爆显存风险 |
| GPU 刷 BIOS 解锁 300W | 已有 280W (公版 max 250W), 刷砖风险高 |
| m16n8k16 FP16 | **硬件没有** (sm75 MMA 是 m16n8k8, k=8 被寄存器砍死) |

## systemd 开机启动

```bash
systemctl --user enable strata  # 已 enabled
systemctl --user status strata   # active (running)
```

## 服务命令

```bash
systemctl --user restart strata   # 重启 (约 50-70s ready)
systemctl --user stop strata
systemctl --user start strata
tail -f ~/.local/share/strata/install/strata-sc117-iq3_s.log
curl -s http://127.0.0.1:8081/health
```
