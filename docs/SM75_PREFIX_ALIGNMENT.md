# 双 RTX 2080 Ti：工具前缀 8192 对齐与业务侧硬拦截

## 结论

在 Strata 0.1.56 / `--prompt-cache 12 --prompt-cache-every 1024` / 双卡 layer split 下，prefix checkpoint 只能完整复用 **8192 的整数倍**。固定工具前缀实测 30022 token 时，换正文只命中 24576；补齐到 **32768** 后，换正文稳定命中 32768/32768（100%）。

不改 C++ 源码、不改引擎参数，只在业务侧把固定前缀精确补齐到 32768，即可在 80K/100K 冷读换正文时省约 **6.9 s TTFT**。

## 实测数据

| 场景 | 固定前缀 | 命中 | TTFT |
| --- | ---: | ---: | ---: |
| 80K 冷读，工具前缀 30022 | 30022 | 24576 | 60250.7 ms |
| 80K 冷读，工具前缀 32768 | 32768 | 32768 | 53357.8 ms |
| 100K 冷读，工具前缀 30022 | 30022 | 24576 | 81450.1 ms |
| 100K 冷读，工具前缀 32768 | 32768 | 32768 | 74456.2 ms |

日志实锤：

```
strata serve: prompt 80052 tokens = 32768 reused + 47284 read
strata serve: prompt 100052 tokens = 32768 reused + 67284 read
```

## 致命坑：必须对齐「渲染后」的序列

不能只数工具定义正文或 `messages[0].content`。Strata 会应用 `chat_template.jinja`、合并 system/developer、插入 reasoning instructions、把 tools 序列化进 `<tools>` 块，并追加 `<|im_start|>assistant\n<think>\n`。实测 80000 字符的 raw prompt 被服务计为 `prompt_tokens=80052`，模板自身约 52 token。

因此 `tools/pad_prefix.py` 复用 `serve/frontend.py` 的 `openai_to_messages` + `ChatTemplate`，按 `Service.encode_prompt` 的 `parse_special=True` 口径，对「省略最后一条消息后的渲染文本」计数。补齐点放在 system/工具定义字符串内部，正文怎么变都不影响。

## 脚本

- `tools/pad_prefix.py`：CLI + 可 import 的 `align_request()` / `align_text()`。
- `tools/strata_prefix_proxy.py`：8001 反代，所有 `/v1/chat/completions`、`/v1/messages`、`/v1/responses` 请求强制经过补齐，再转发到 Strata 8000。

```bash
# 自检（12 项，含 30022→32768、幂等、32769→40960、JSON 合法、占位符）
/home/lingdeng/AI/strata/.venv/bin/python /home/lingdeng/AI/strata/tools/pad_prefix.py --self-test

# 作为库调用
from pad_prefix import Tokenizer, Renderer, align_request
request, report = align_request(request, tokenizer=tok, renderer=renderer)

# 命令行：JSON 请求对象
python tools/pad_prefix.py --input req.json --output padded.json --report-json report.json

# 命令行：已渲染纯文本（调用方自行保证已应用 Chat Template）
python tools/pad_prefix.py --format text --input prompt.txt --output padded.txt
```

边界规则：

- `30000 <= tokens <= 32768` → 补齐到 `32768`。
- `32768 < tokens <= 40960` → **stderr 报警**并兜底补齐到 `40960`；更大值按 `((n // 8192) + 1) * 8192`。
- 精确落在 8192 边界 → 不动（幂等）。
- `< 30000` → 默认不动，避免误补；`--force` 可强制补到 32768。
- `--strict` 在触发兜底报警时返回退出码 2，默认不中断业务。

## JSON 安全

脚本只修改**解码后**的字符串字段：`messages[].content`、`content[].text`、`instructions`、`input` 等；绝不把补齐文本拼到 JSON 外部。补齐内容是一段合法 HTML 注释：

```
<!-- STRATA_PREFIX_PAD: inert alignment padding, do not parse. abab... -->
```

可显式放占位符 `<!-- STRATA_PREFIX_PAD_HERE -->`，脚本会替换它（仅字符串内部），保证结构仍可 `json.loads()`。

## 硬拦截部署

1. `~/.config/systemd/user/strata-prefix-proxy.service`：开机自启，监听 `127.0.0.1:8001`，上游 `127.0.0.1:8000`。
2. magpie `providers.json` 中本地 Strata provider 的 `chat` 改为 `http://127.0.0.1:8001/v1`；`stash.json` 的 `codex.tables` 同步改到 8001。
3. Codex / magpie / VS Code 等经 3425 → 8001 → 8000 的流量全部强制补齐。

验证：

```bash
systemctl --user is-active strata-prefix-proxy.service   # active
curl -s http://127.0.0.1:8001/_prefix/status             # requests / padded / last_report
magpie providers                                         # 本地 provider 应显示 127.0.0.1:8001
```

反代状态里的 `last_report.final_tokens` 应为 32768；换正文请求在 Strata 日志里应为 `32768 reused`。

## 报警与兜底

- 工具定义膨胀导致渲染后固定前缀 > 32768：stderr 中文告警，自动对齐到下一 8192 边界；`/_prefix/status` 的 `alarms` 计数 +1。
- 补齐失败（字段形状异常、模板渲染失败）：反代打日志并**原样转发**，不吞请求。
- `127.0.0.1:8000` 仍开放；内部探针/脚本可直连，但所有 Agent 业务流量必须走 8001，否则绕过硬拦截。

## 回滚

```bash
# 1. magpie 改回直连
cp ~/.config/magpie/providers.json.bak-<stamp> ~/.config/magpie/providers.json
cp ~/.config/magpie/stash.json.bak-<stamp> ~/.config/magpie/stash.json

# 2. 停反代（保留脚本与文档）
systemctl --user disable --now strata-prefix-proxy.service
```

生产 Strata 配置全程不动：128K 上下文、12000 专家槽位、MTP、`--prefill auto`、KV k8v4 保持原样。
