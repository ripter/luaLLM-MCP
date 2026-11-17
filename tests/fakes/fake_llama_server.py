"""A stand-in for llama-server, with real sockets, for gateway tests.

Answers /health (503 while "loading"), /props, and /v1/chat/completions,
streaming or not, one token every FAKE_LLAMA_TOKEN_DELAY seconds. Between
tokens it checks whether its client has gone, the way llama.cpp stops a
generation whose connection closed, and appends what happened to
$FAKE_LUALLM_DIR/<model>.events as JSON lines:
  {"event": "request", "body": {...}}       the body it was sent
  {"event": "disconnected", "after": n}     client went away after n tokens
  {"event": "completed", "tokens": n}

Flags (from the fake luallm, i.e. the profile's args):
  --mmproj X        vision in /props
  -c / --ctx-size N n_ctx in /props
Environment:
  FAKE_LLAMA_LOAD_DELAY        seconds /health answers 503
  FAKE_LLAMA_TOKENS            tokens per reply (default 5)
  FAKE_LLAMA_TOKEN_DELAY       seconds between tokens (default 0)
  FAKE_LLAMA_REASONING_LEVELS  e.g. "low,medium,xhigh"; empty = no reasoning_effort
  FAKE_LLAMA_TEMPLATE          chat template text (default mentions enable_thinking)
  FAKE_LLAMA_WORD              the word each token is made of (default "t")
  FAKE_LLAMA_CRASH_AFTER       exit this many seconds after starting, while loading
"""

import argparse
import json
import os
import select
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, required=True)
parser.add_argument("--model", default="tiny")
parser.add_argument("--mmproj")
parser.add_argument("-c", "--ctx-size", type=int, default=4096)
opts, _unknown = parser.parse_known_args()

STARTED = time.monotonic()
LOAD_DELAY = float(os.environ.get("FAKE_LLAMA_LOAD_DELAY", "0"))
TOKENS = int(os.environ.get("FAKE_LLAMA_TOKENS", "5"))
TOKEN_DELAY = float(os.environ.get("FAKE_LLAMA_TOKEN_DELAY", "0"))
LEVELS = [l for l in os.environ.get("FAKE_LLAMA_REASONING_LEVELS", "low,medium,xhigh").split(",") if l]
TEMPLATE = os.environ.get(
    "FAKE_LLAMA_TEMPLATE",
    "{%- if enable_thinking is undefined or enable_thinking is true %}think{% endif %}")
WORD = os.environ.get("FAKE_LLAMA_WORD", "t")
CRASH_AFTER = os.environ.get("FAKE_LLAMA_CRASH_AFTER")
EVENTS = Path(os.environ["FAKE_LUALLM_DIR"]) / f"{opts.model}.events"


def record(event: dict) -> None:
    with open(EVENTS, "a") as f:
        f.write(json.dumps(event) + "\n")


def client_gone(sock: socket.socket) -> bool:
    readable, _, _ = select.select([sock], [], [], 0)
    if not readable:
        return False
    try:
        return sock.recv(1, socket.MSG_PEEK) == b""
    except OSError:
        return True


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _json(self, status: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        loaded = time.monotonic() - STARTED >= LOAD_DELAY
        if self.path == "/health":
            if loaded:
                self._json(200, {"status": "ok"})
            else:
                self._json(503, {"error": {"code": 503, "message": "Loading model"}})
        elif self.path == "/props":
            caps = {"supports_system_role": True,
                    "supports_reasoning_effort": bool(LEVELS),
                    "supports_preserve_reasoning": True}
            if LEVELS:
                caps["reasoning_effort_levels"] = LEVELS
            self._json(200, {
                "model_path": f"/models/{opts.model}.gguf",
                "modalities": {"vision": bool(opts.mmproj), "audio": False},
                "chat_template": TEMPLATE,
                "chat_template_caps": caps,
                "default_generation_settings": {"n_ctx": opts.ctx_size},
            })
        else:
            self._json(404, {"error": {"code": 404, "message": "not found"}})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        record({"event": "request", "body": body})
        if self.path != "/v1/chat/completions":
            self._json(404, {"error": {"code": 404, "message": "not found"}})
            return
        effort = body.get("reasoning_effort")
        if effort is not None and effort not in LEVELS:
            # What a chat template's raise_exception becomes in llama.cpp.
            self._json(500, {"error": {"code": 500, "type": "server_error",
                                       "message": f"Unexpected reasoning effort {effort}"}})
            return
        model = f"/models/{opts.model}.gguf"
        tokens = min(TOKENS, body.get("max_tokens") or TOKENS)
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

            def chunk(data: bytes) -> None:
                self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
                self.wfile.flush()

            for n in range(tokens):
                if client_gone(self.connection):
                    record({"event": "disconnected", "after": n})
                    self.close_connection = True
                    return
                event = {"id": "c1", "object": "chat.completion.chunk", "model": model,
                         "choices": [{"index": 0, "delta": {"content": f"{WORD}{n} "},
                                      "finish_reason": None}]}
                try:
                    chunk(b"data: " + json.dumps(event).encode() + b"\n\n")
                except OSError:
                    record({"event": "disconnected", "after": n})
                    self.close_connection = True
                    return
                time.sleep(TOKEN_DELAY)
            final = {"id": "c1", "object": "chat.completion.chunk", "model": model,
                     "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                     "usage": {"prompt_tokens": 3, "completion_tokens": tokens,
                               "total_tokens": 3 + tokens}}
            try:
                chunk(b"data: " + json.dumps(final).encode() + b"\n\n")
                chunk(b"data: [DONE]\n\n")
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except OSError:
                record({"event": "disconnected", "after": tokens})
                return
            record({"event": "completed", "tokens": tokens})
            return

        for n in range(tokens):
            if client_gone(self.connection):
                record({"event": "disconnected", "after": n})
                self.close_connection = True
                return
            time.sleep(TOKEN_DELAY)
        record({"event": "completed", "tokens": tokens})
        self._json(200, {
            "id": "c1", "object": "chat.completion", "model": model,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant",
                                     "content": " ".join(f"{WORD}{n}" for n in range(tokens))}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": tokens,
                      "total_tokens": 3 + tokens},
        })


if CRASH_AFTER:
    threading.Timer(float(CRASH_AFTER), lambda: os._exit(1)).start()
ThreadingHTTPServer.daemon_threads = True
ThreadingHTTPServer(("127.0.0.1", opts.port), Handler).serve_forever()
