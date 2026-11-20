# 采样参数与多步编程 agent 场景优化（2026-11-21）

## 1. top_p / top_k 对速度没有收益（源码级确认）

引擎源码 `engine/src/program/generate.cpp:18147`：`s.greedy = req_temperature <= 0.0f`

- 只要 `temperature=0`，就走 greedy 路径，top_p / top_k / min_p 完全不参与。
- 当前生产 `sampling: {temperature:0, top_p:1, top_k:1, min_p:0}` 已是 greedy，
  top_p/top_k 取值对速度零影响。
- top_p/top_k 只在 `temperature>0` 的采样路径生效。

实测（多步编程 agent，外部 14-15 路并发，同一任务），decode tok/s（3 步）：

- greedy（temp=0, top_k=1）：76.8 / 78.5 / 82.1
- sampled（temp=1, top_p=0.95, top_k=20）：71.7 / 66.6 / 70.4

采样配置慢约 10-15%：温度>0 降低 MTP 草稿接受率，投机解码收益下降。
结论：速度优先就保持 temperature=0，不要动 top_p/top_k。

## 2. 编程 agent 的真正瓶颈：thinking 模型把 max_tokens 吃光

当前模型是 thinking 模型，输出内容全在 `reasoning_content` 字段。
生产 `reasoning_budget_tokens=0`（不限制思考长度）时：

- 多步编程 agent 测试中 `content` 全为空（0 chars），只有 reasoning；
- 3 步用了 320/320/200 token 后仍在推理，最终代码永远输出不出来；
- 例：greedy step2（200 token）的 reasoning 结尾还在纠结用不用 `heapq.nlargest`，
  没有进入代码块。

这才是编程场景该优化的点——不是采样参数，是 reasoning budget。

## 3. 关键发现：请求级 `reasoning_budget_tokens` 有效

实测 HTTP 请求带 `reasoning_budget_tokens: 256`：

```
content: '```python\nimport heapq\n\ndef topk_indices(values, k):\n
    return heapq.nlargest(k, range(len(values)), key=values.__getitem__)\n```'
usage: completion_tokens=310
```

引擎把生成分成两段：日志两行 `256 generated`（思考段）+ `36 generated`（代码段）。
限制思考后，模型正常输出最终代码，且代码正确（实测 `[3,1,5,5,2],k=2 -> [2,3]`、
`k=0 -> []`、空列表、k>n 全部通过）。

### budget 大小对比（merge_intervals 中等任务）

- budget=128：代码长度 337，含实现，completion token 211
- budget=256：代码长度 337，含实现，completion token 207
- budget=512：代码长度 337，含实现，completion token 207

代码完全相同。budget=128 足够。

### 复杂任务（LRU cache 装饰器）

- budget=32：代码长度 1178，完整实现，completion token 367
- budget=64：代码长度 858，完整实现，completion token 270
- budget=128：代码长度 747，完整实现，completion token 310
- budget=256：代码长度 811，完整实现，completion token 452

全部产出完整实现（OrderedDict/dict 方案）。budget 32-128 都可用。

## 4. 编程 agent 推荐配置

请求级（每个编程任务）：

```json
{
  "temperature": 0,
  "top_p": 1,
  "top_k": 1,
  "reasoning_budget_tokens": 128,
  "max_tokens": 800
}
```

- `temperature=0`：最快、MTP 接受率最高
- `reasoning_budget_tokens=128`：够用且省时间，避免思考吃光预算
- `max_tokens` 给足（≥800）：确保代码段有空间输出

不建议把生产全局 `reasoning_budget_tokens` 从 0 改成固定值——对话/推理类请求
可能需要更长思考。更好的做法是在编程 agent 的请求里按任务类型传该参数
（`anthropic_thinking: on_request` 已支持请求级控制）。

## 5. 还需要优化的（按优先级）

1. 编程 agent 客户端：在代码生成请求里传 `reasoning_budget_tokens=128`，
   这是投入产出比最高的改动（不改引擎、不改服务）。
2. max_tokens 与 thinking 的匹配：如果客户端没法传 budget，至少把编程任务的
   `max_tokens` 提到足够大（如 1024+），否则代码出不来。
3. 采样质量场景：若某些编程任务确实需要创造性（如算法设计），可用
   `temperature=0.6~0.8, top_p=0.9, top_k=20`，但要接受 decode 慢 10-15%。
4. 短 prefill 固定开销：见 `README-0.2.17-production-20261121.md`，
   本机可用开关无法削减，属架构固有成本。

测试期间外部 13-16 路并发持续占用服务，绝对 tok/s 偏低且有波动；
但 greedy vs sampled 的相对差异（10-15%）与 reasoning budget 的行为是确定的。
