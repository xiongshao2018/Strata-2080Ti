#!/usr/bin/env python3
"""把固定前缀精确补齐到 8192 边界，让 Strata 的 prefix checkpoint 100% 复用。

双 RTX 2080 Ti 实测（docs/SM75_PREFIX_ALIGNMENT.md）：30022-token 的工具前缀
只能复用 24576 token；补齐到 32768 后，换正文稳定复用 32768，80K/100K 冷读
约省 6.9 s TTFT。

致命坑：必须对齐「渲染后」的序列，不能只数工具定义正文。脚本复用主服务
serve/frontend.py 的 openai_to_messages + ChatTemplate 渲染，计数口径与
Service.encode_prompt 一致；补齐点放在 system/工具定义内，正文怎么变都不影响。
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import pathlib
import sys
from typing import Any

ENGINE = pathlib.Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

DEFAULT_TOKENIZER = pathlib.Path("/home/lingdeng/AI/models/Strata-SC117/sc117_pack/tokenizer")
DEFAULT_TEMPLATE = DEFAULT_TOKENIZER / "chat_template.jinja"
DEFAULT_SERVER_TEMPLATE = ENGINE / "serve" / "chat_template.jinja"
DEFAULT_PLACEHOLDER = "<!-- STRATA_PREFIX_PAD_HERE -->"

CHUNK = 8192
TARGET = 32768
NEXT = 40960
NEAR_MIN = 30000
NEAR_MAX = 32768
PAD_UNIT = "ab"
PAD_HEAD = "\n<!-- STRATA_PREFIX_PAD: inert alignment padding, do not parse. "
PAD_TAIL = " -->"


class PadError(Exception):
    pass


class Tokenizer:
    def __init__(self, directory: pathlib.Path):
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
        import strata_tokenizer as impl

        vocab = json.loads((directory / "vocab.json").read_text())
        tokens: list[Any] = [None] * len(vocab)
        for token, index in vocab.items():
            tokens[index] = token
        merges = (directory / "merges.txt").read_text().splitlines()
        types = json.loads((directory / "token_type.json").read_text())
        self.impl = impl.Tokenizer(tokens, merges, types)

    def count(self, text: str) -> int:
        return len(self.impl.encode(text, parse_special=True))


class Renderer:
    def __init__(self, template_path: pathlib.Path):
        from serve.frontend import ChatTemplate, openai_to_messages

        self.template = ChatTemplate(template_path)
        self.to_messages = openai_to_messages

    def fixed_text(self, request: dict) -> str:
        if "messages" not in request and ("input" in request or "instructions" in request):
            request = self._responses_to_chat(request)
        messages, tools, kwargs = self.to_messages(request)
        if not messages:
            raise PadError("请求缺少 messages，无法按渲染后模板计数")
        fixed = self._drop_empty_assistants(messages[:-1])
        return self.template.render(fixed, tools=tools, add_generation_prompt=False, **kwargs)

    @staticmethod
    def _responses_to_chat(request: dict) -> dict:
        """把 Responses API 请求折成 chat/completions 形状，仅用于计算固定前缀。"""
        instructions = request.get("instructions")
        messages = [{"role": "system", "content": instructions}] if isinstance(instructions, str) else []
        raw_input = request.get("input")
        if isinstance(raw_input, str):
            messages.append({"role": "user", "content": raw_input})
        elif isinstance(raw_input, list):
            for item in raw_input:
                if not isinstance(item, dict):
                    continue
                role = item.get("role") or "user"
                content = item.get("content")
                if isinstance(content, str):
                    messages.append({"role": role, "content": content})
                elif isinstance(content, list):
                    texts = [part.get("text", "") for part in content
                             if isinstance(part, dict) and part.get("type") in ("input_text", "text", None)]
                    messages.append({"role": role, "content": "".join(texts)})
        if not messages:
            messages = [{"role": "user", "content": ""}]
        tools = request.get("tools")
        if isinstance(tools, list):
            tools = [tool if isinstance(tool, dict) and tool.get("type") == "function"
                     else {"type": "function", "function": tool.get("function", tool)}
                     for tool in tools if isinstance(tool, dict)]
        chat = {"model": request.get("model"), "messages": messages, "tools": tools}
        if isinstance(request.get("reasoning"), dict) and request["reasoning"].get("effort"):
            chat["reasoning_effort"] = request["reasoning"]["effort"]
        return chat

    @staticmethod
    def _drop_empty_assistants(messages: list[dict]) -> list[dict]:
        from serve.frontend import _text_of

        if os.environ.get("STRATA_KEEP_EMPTY_TURNS") == "1":
            return messages
        return [
            message for message in messages
            if not (isinstance(message, dict)
                    and message.get("role") == "assistant"
                    and not _text_of(message.get("content")).strip()
                    and not message.get("tool_calls"))
        ]


def _candidate_strings(request: dict) -> list[tuple[Any, str]]:
    out: list[tuple[Any, str]] = []
    # OpenAI Responses API：input 可以是字符串或 content-part 数组。
    instructions = request.get("instructions")
    if isinstance(instructions, str):
        out.append((request, "instructions"))
    response_input = request.get("input")
    if isinstance(response_input, str):
        out.append((request, "input"))
    elif isinstance(response_input, list):
        for item in response_input:
            if not isinstance(item, dict):
                continue
            if isinstance(item.get("content"), str):
                out.append((item, "content"))
            elif isinstance(item.get("content"), list):
                for part in item["content"]:
                    if isinstance(part, dict) and isinstance(part.get("text"), str) \
                            and part.get("type") in ("input_text", "text", None):
                        out.append((part, "text"))
    messages = request.get("messages")
    if not isinstance(messages, list):
        return out
    for message in messages[:-1]:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            out.append((message, "content"))
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str) \
                        and part.get("type") in ("text", "input_text", None):
                    out.append((part, "text"))
        if isinstance(message.get("reasoning_content"), str):
            out.append((message, "reasoning_content"))
    return out


def _insert_target(request: dict, placeholder: str | None) -> tuple[Any, str, str]:
    candidates = _candidate_strings(request)
    if placeholder:
        for container, key in candidates:
            if placeholder in container[key]:
                return container, key, "replace"
    if "messages" not in request and isinstance(request.get("instructions"), str):
        return request, "instructions", "append"
    for container, key in candidates:
        parent = container
        if key == "content" and isinstance(parent, dict) and parent.get("role") in ("system", "developer"):
            return parent, key, "append"
    for container, key in candidates:
        return container, key, "append"
    raise PadError("找不到可安全补齐的字符串字段；请在 system/工具定义末尾放 %s" % DEFAULT_PLACEHOLDER)


def _choose_target(tokens: int, force: bool) -> tuple[int | None, str, bool]:
    if tokens > 0 and tokens % CHUNK == 0:
        return None, "aligned", False
    if NEAR_MIN <= tokens <= NEAR_MAX:
        return TARGET, "padded_to_32768", False
    if tokens > NEAR_MAX:
        target = NEXT if tokens <= NEXT else ((tokens // CHUNK) + 1) * CHUNK
        return target, "overflow_next_boundary", True
    if force:
        return TARGET, "force_padded_to_32768", False
    return None, "below_window", False


def _make_pad(need: int, tokenizer: Tokenizer) -> str:
    if need <= 0:
        return ""
    overhead = tokenizer.count(PAD_HEAD + PAD_TAIL)
    if need <= overhead:
        return PAD_UNIT * need
    units = need - overhead
    for _ in range(24):
        text = PAD_HEAD + PAD_UNIT * max(0, units) + PAD_TAIL
        actual = tokenizer.count(text)
        if actual == need:
            return text
        units += need - actual
        units = max(0, units)
    raise PadError("无法精确构造 %d token 的补齐文本" % need)


def align_request(request: dict, *, tokenizer: Tokenizer, renderer: Renderer,
                  force: bool = False, placeholder: str | None = None) -> tuple[dict, dict]:
    req = copy.deepcopy(request)
    if not isinstance(req, dict):
        raise PadError("请求必须是 JSON 对象")
    container, key, mode = _insert_target(req, placeholder)
    original = tokenizer.count(renderer.fixed_text(req))
    target, status, alarm = _choose_target(original, force)
    report = {
        "status": status,
        "alarm": alarm,
        "original_tokens": original,
        "target_tokens": target,
        "final_tokens": original,
        "inserted_tokens": 0,
        "insert_mode": mode,
        "count_basis": "rendered_fixed_prefix",
    }
    if target is None:
        return req, report
    need = target - original
    pad = _make_pad(need, tokenizer)
    if mode == "replace":
        container[key] = container[key].replace(placeholder, pad, 1)
    else:
        container[key] = container[key] + pad
    final = tokenizer.count(renderer.fixed_text(req))
    for _ in range(24):
        if final == target:
            break
        delta = target - final
        if delta <= 0:
            break
        extra = _make_pad(delta, tokenizer)
        if mode == "replace":
            container[key] = container[key] + extra
        else:
            container[key] = container[key] + extra
        final = tokenizer.count(renderer.fixed_text(req))
    if final != target:
        raise PadError("补齐后为 %d token，未精确命中 %d" % (final, target))
    report.update({"final_tokens": final, "inserted_tokens": final - original})
    if alarm:
        print("警告：渲染后固定前缀 %d token 超过 32768，已兜底对齐到 %d；请复核工具定义是否膨胀。"
              % (original, target), file=sys.stderr, flush=True)
    return req, report


def align_text(text: str, *, tokenizer: Tokenizer, force: bool = False,
               placeholder: str | None = None) -> tuple[str, dict]:
    original = tokenizer.count(text)
    target, status, alarm = _choose_target(original, force)
    report = {"status": status, "alarm": alarm, "original_tokens": original,
              "target_tokens": target, "final_tokens": original, "inserted_tokens": 0,
              "count_basis": "rendered_text"}
    if target is None:
        return text, report
    pad = _make_pad(target - original, tokenizer)
    out = text.replace(placeholder, pad, 1) if placeholder and placeholder in text else text + pad
    final = tokenizer.count(out)
    for _ in range(24):
        if final == target:
            break
        delta = target - final
        if delta <= 0:
            break
        out += _make_pad(delta, tokenizer)
        final = tokenizer.count(out)
    if final != target:
        raise PadError("补齐后为 %d token，未精确命中 %d" % (final, target))
    report.update({"target_tokens": target, "final_tokens": final,
                   "inserted_tokens": final - original})
    if alarm:
        print("警告：渲染后文本 %d token 超过 32768，已兜底对齐到 %d。" % (original, target),
              file=sys.stderr, flush=True)
    return out, report


def self_test(tokenizer: Tokenizer, renderer: Renderer) -> dict:
    checks: list[dict] = []

    def record(name: str, ok: bool, detail: Any = ""):
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    def synthesize(prefix_tokens: int, with_placeholder: bool = False) -> dict:
        req = {"model": "qwen3.8-flash-next-iq3_s",
               "messages": [{"role": "system", "content": ""},
                            {"role": "user", "content": "hello"}],
               "tools": [{"type": "function", "function": {
                   "name": "f", "description": "d",
                   "parameters": {"type": "object", "properties": {}}}}]}
        system = req["messages"][0]
        if with_placeholder:
            system["content"] = DEFAULT_PLACEHOLDER
        base = tokenizer.count(renderer.fixed_text(req))
        system["content"] = (system["content"] + "\n" if system["content"] else "") + \
            PAD_UNIT * max(0, prefix_tokens - base)
        actual = tokenizer.count(renderer.fixed_text(req))
        head = system["content"]
        for _ in range(24):
            if actual == prefix_tokens:
                break
            delta = prefix_tokens - actual
            if delta > 0:
                head += PAD_UNIT * delta
            else:
                head = head[:max(0, len(head) + delta * 2)]
            system["content"] = head
            actual = tokenizer.count(renderer.fixed_text(req))
        if actual != prefix_tokens:
            raise PadError("self_test 无法构造 %d token 前缀（得到 %d）" % (prefix_tokens, actual))
        return req

    near = synthesize(30022)
    out, rep = align_request(near, tokenizer=tokenizer, renderer=renderer)
    record("near_30022_to_32768", rep["final_tokens"] == TARGET and rep["status"] == "padded_to_32768", rep)
    record("json_still_valid", isinstance(json.loads(json.dumps(out)), dict))
    record("padding_in_system", "STRATA_PREFIX_PAD" in out["messages"][0]["content"])
    out2, rep2 = align_request(out, tokenizer=tokenizer, renderer=renderer)
    record("idempotent", rep2["inserted_tokens"] == 0 and rep2["final_tokens"] == TARGET, rep2)

    over = synthesize(32769)
    out3, rep3 = align_request(over, tokenizer=tokenizer, renderer=renderer)
    record("32769_to_40960", rep3["final_tokens"] == NEXT and rep3["alarm"] is True, rep3)
    record("json_valid_after_overflow", isinstance(json.loads(json.dumps(out3)), dict))

    exact = synthesize(32768)
    _, rep4 = align_request(exact, tokenizer=tokenizer, renderer=renderer)
    record("exact_32768_unchanged", rep4["inserted_tokens"] == 0, rep4)

    short = synthesize(1000)
    _, rep5 = align_request(short, tokenizer=tokenizer, renderer=renderer)
    record("below_window_untouched", rep5["inserted_tokens"] == 0 and rep5["status"] == "below_window", rep5)
    _, rep6 = align_request(short, tokenizer=tokenizer, renderer=renderer, force=True)
    record("force_below_window", rep6["final_tokens"] == TARGET, rep6)

    with_slot = synthesize(30022, with_placeholder=True)
    out7, rep7 = align_request(with_slot, tokenizer=tokenizer, renderer=renderer,
                               placeholder=DEFAULT_PLACEHOLDER)
    record("placeholder_removed", DEFAULT_PLACEHOLDER not in out7["messages"][0]["content"]
           and rep7["insert_mode"] == "replace", rep7)
    record("placeholder_json_valid", isinstance(json.loads(json.dumps(out7)), dict))

    text = "<|im_start|>system\n" + PAD_UNIT * 30000 + "<|im_end|>\n"
    padded, rep8 = align_text(text, tokenizer=tokenizer)
    record("plain_text_alignment", rep8["final_tokens"] == TARGET, rep8)

    ok = all(check["ok"] for check in checks)
    return {"ok": ok, "checks": checks}


def _load_objects(tokenizer_dir: pathlib.Path, template_path: pathlib.Path) -> tuple[Tokenizer, Renderer]:
    return Tokenizer(tokenizer_dir), Renderer(template_path)


def main() -> int:
    ap = argparse.ArgumentParser(description="把固定前缀精确补齐到 8192 边界")
    ap.add_argument("--input", "-i", help="输入文件；缺省读 stdin")
    ap.add_argument("--output", "-o", help="输出文件；缺省写 stdout")
    ap.add_argument("--in-place", action="store_true", help="原地改写输入文件")
    ap.add_argument("--format", choices=("auto", "json", "text"), default="auto")
    ap.add_argument("--placeholder", default=None, help="把补齐内容插到此占位符处")
    ap.add_argument("--force", action="store_true", help="低于 30000 也强制补齐到 32768")
    ap.add_argument("--strict", action="store_true", help="触发兜底报警时返回非零")
    ap.add_argument("--report-json", help="把结构化报告写到此文件")
    ap.add_argument("--self-test", action="store_true", help="运行内置单元测试")
    ap.add_argument("--tokenizer-dir", default=str(DEFAULT_TOKENIZER))
    ap.add_argument("--template", default=str(DEFAULT_TEMPLATE))
    args = ap.parse_args()

    template_path = pathlib.Path(args.template)
    if not template_path.exists():
        template_path = DEFAULT_SERVER_TEMPLATE
    tokenizer, renderer = _load_objects(pathlib.Path(args.tokenizer_dir), template_path)

    if args.self_test:
        result = self_test(tokenizer, renderer)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["ok"] else 1

    raw = pathlib.Path(args.input).read_text(encoding="utf-8") if args.input else sys.stdin.read()
    fmt = args.format
    if fmt == "auto":
        try:
            json.loads(raw)
            fmt = "json"
        except json.JSONDecodeError:
            fmt = "text"
    if fmt == "json":
        out, report = align_request(json.loads(raw), tokenizer=tokenizer, renderer=renderer,
                                    force=args.force, placeholder=args.placeholder)
        text = json.dumps(out, ensure_ascii=False, separators=(",", ":"))
    else:
        out, report = align_text(raw, tokenizer=tokenizer, force=args.force,
                                 placeholder=args.placeholder)
        text = out
    if args.report_json:
        pathlib.Path(args.report_json).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    json.dump({"final_tokens": report["final_tokens"], "status": report["status"]}, sys.stderr)
    sys.stderr.write("\n")
    dest = args.output or (args.input if args.in_place else None)
    if dest:
        pathlib.Path(dest).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
        if fmt == "text" and not text.endswith("\n"):
            sys.stdout.write("\n")
    return 2 if args.strict and report["alarm"] else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PadError as exc:
        print("pad_prefix: %s" % exc, file=sys.stderr, flush=True)
        raise SystemExit(1)
