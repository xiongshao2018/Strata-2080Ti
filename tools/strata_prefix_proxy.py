#!/usr/bin/env python3
"""业务侧硬拦截：所有 chat/responses/messages 请求先过 pad_prefix.py，再转发 Strata。

监听 127.0.0.1:8001；magpie/Codex 的 provider URL 指向 8001 后，请求无法绕过补齐。
生产 Strata 仍在 8000，配置、引擎、C++ 源码全都不动。
"""

from __future__ import annotations

import argparse
import http.client
import json
import pathlib
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

TOOLS = pathlib.Path(__file__).resolve().parent
ENGINE = TOOLS.parent
for path in (str(ENGINE), str(TOOLS)):
    if path not in sys.path:
        sys.path.insert(0, path)

from pad_prefix import DEFAULT_PLACEHOLDER, DEFAULT_TOKENIZER, DEFAULT_TEMPLATE, PadError, Renderer, Tokenizer, align_request

UPSTREAM_HOST = "127.0.0.1"
UPSTREAM_PORT = 8000
PAD_ENDPOINTS = {"/v1/chat/completions", "/v1/messages", "/v1/responses"}
HOP_BY_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
              "te", "trailers", "transfer-encoding", "upgrade", "host", "content-length"}


class ProxyState:
    def __init__(self, tokenizer: Tokenizer, renderer: Renderer):
        self.tokenizer = tokenizer
        self.renderer = renderer
        self.lock = threading.Lock()
        self.last_report: dict | None = None
        self.requests = 0
        self.padded = 0
        self.alarms = 0

    def pad(self, body: bytes) -> tuple[bytes, dict | None]:
        if not body:
            return body, None
        try:
            request = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return body, None
        try:
            padded, report = align_request(request, tokenizer=self.tokenizer,
                                           renderer=self.renderer,
                                           placeholder=DEFAULT_PLACEHOLDER)
        except PadError as exc:
            print("[prefix-proxy] 补齐失败，原样转发：%s" % exc, file=sys.stderr, flush=True)
            return body, None
        with self.lock:
            self.requests += 1
            self.last_report = report
            if report["inserted_tokens"]:
                self.padded += 1
            if report["alarm"]:
                self.alarms += 1
        if report["alarm"]:
            print("[prefix-proxy] 告警：%d -> %d token" %
                  (report["original_tokens"], report["target_tokens"]), file=sys.stderr, flush=True)
        return json.dumps(padded, ensure_ascii=False, separators=(",", ":")).encode(), report


STATE: ProxyState


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "StrataPrefixProxy/1.0"

    def log_message(self, fmt, *args):
        sys.stderr.write("[prefix-proxy] %s - %s\n" % (self.address_string(), fmt % args))

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            return self.rfile.read(length)
        if (self.headers.get("Transfer-Encoding") or "").lower() == "chunked":
            chunks = []
            while True:
                line = self.rfile.readline().strip()
                if not line:
                    break
                size = int(line.split(b";", 1)[0], 16)
                if size == 0:
                    self.rfile.readline()
                    break
                chunks.append(self.rfile.read(size))
                self.rfile.readline()
            return b"".join(chunks)
        return b""

    def _forward(self, method: str, body: bytes):
        parsed = urlsplit(self.path)
        endpoint = parsed.path.rstrip("/")
        if method == "POST" and endpoint in PAD_ENDPOINTS:
            body, _ = STATE.pad(body)
        conn = http.client.HTTPConnection(UPSTREAM_HOST, UPSTREAM_PORT, timeout=900)
        path = parsed.path + (("?" + parsed.query) if parsed.query else "")
        headers = {key: value for key, value in self.headers.items()
                   if key.lower() not in HOP_BY_HOP}
        headers["Host"] = "%s:%d" % (UPSTREAM_HOST, UPSTREAM_PORT)
        if body:
            headers["Content-Length"] = str(len(body))
        try:
            conn.request(method, path, body=body or None, headers=headers)
            upstream = conn.getresponse()
            content_length = upstream.getheader("Content-Length")
            streaming = (upstream.getheader("Content-Type") or "").startswith("text/event-stream") \
                or content_length is None
            self.send_response(upstream.status, upstream.reason)
            for key, value in upstream.getheaders():
                if key.lower() not in HOP_BY_HOP:
                    self.send_header(key, value)
            if streaming:
                self.send_header("Transfer-Encoding", "chunked")
            else:
                self.send_header("Content-Length", content_length or "0")
            self.end_headers()
            if not streaming:
                self.wfile.write(upstream.read())
                self.wfile.flush()
                return
            while True:
                chunk = upstream.read(4096)
                if not chunk:
                    break
                try:
                    self.wfile.write(("%X\r\n" % len(chunk)).encode() + chunk + b"\r\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    break
            try:
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
        except (BrokenPipeError, ConnectionResetError):
            pass
        except OSError as exc:
            print("[prefix-proxy] upstream %s: %s" % (path, exc), file=sys.stderr, flush=True)
            try:
                self.send_error(502, "Strata upstream unavailable: %s" % exc)
            except OSError:
                pass
        finally:
            conn.close()

    def do_POST(self):
        self._forward("POST", self._read_body())

    def do_GET(self):
        parsed = urlsplit(self.path)
        if parsed.path.rstrip("/") == "/_prefix/status":
            with STATE.lock:
                payload = {"requests": STATE.requests, "padded": STATE.padded,
                           "alarms": STATE.alarms, "last_report": STATE.last_report,
                           "upstream": "http://%s:%d" % (UPSTREAM_HOST, UPSTREAM_PORT),
                           "alignment": 32768}
            body = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self._forward("GET", b"")

    def do_PUT(self):
        self._forward("PUT", self._read_body())

    def do_DELETE(self):
        self._forward("DELETE", self._read_body())

    def do_HEAD(self):
        self._forward("HEAD", b"")

    def do_OPTIONS(self):
        self._forward("OPTIONS", self._read_body())


def main() -> int:
    global STATE
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", default="127.0.0.1:8001")
    ap.add_argument("--tokenizer-dir", default=str(DEFAULT_TOKENIZER))
    ap.add_argument("--template", default=str(DEFAULT_TEMPLATE))
    args = ap.parse_args()
    host, port = args.listen.rsplit(":", 1)
    template = pathlib.Path(args.template)
    if not template.exists():
        template = ENGINE / "serve" / "chat_template.jinja"
    STATE = ProxyState(Tokenizer(pathlib.Path(args.tokenizer_dir)), Renderer(template))
    server = ThreadingHTTPServer((host, int(port)), Handler)
    server.daemon_threads = True
    print("[prefix-proxy] listening on %s:%s -> Strata %s:%d" %
          (host, port, UPSTREAM_HOST, UPSTREAM_PORT), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
