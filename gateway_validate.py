"""Validation of /v1/chat/completions requests.

The gateway never forwards a client's body. It builds the upstream request
from the fields below, each checked, and refuses anything else by name.
That keeps out llama.cpp's own per-request controls (slot selection and
saving, LoRA switching, grammars, raw sampler chains), tools, and image
URLs llama-server would fetch from the network.

Checking happens in two steps: check_request() needs nothing but the body;
apply_capabilities() needs what the model's own server reported (see
gateway_config.capabilities_from_props), and runs again once the request's
own server is known, so a feature is only used where it was verified.
"""

import math
import re
from typing import Any

from gateway_config import Route

ROLES = {"system", "user", "assistant"}
_IMAGE_URL = re.compile(r"^data:image/(png|jpeg|jpg|webp|gif);base64,[A-Za-z0-9+/=\s]+$")
_TOOLS = {"tools", "tool_choice", "functions", "function_call", "parallel_tool_calls"}
MAX_MESSAGES = 2000
MAX_STOP = 8


class RequestError(Exception):
    def __init__(self, message: str, param: str | None = None,
                 code: str = "invalid_request", status: int = 400):
        super().__init__(message)
        self.message = message
        self.param = param
        self.code = code
        self.status = status


def _number(body: dict, key: str, lo: float, hi: float, *, integer: bool = False,
            lo_open: bool = False) -> None:
    if key not in body:
        return
    v = body[key]
    if isinstance(v, bool) or not isinstance(v, (int, float)) or (integer and not isinstance(v, int)):
        raise RequestError(f"'{key}' must be {'an integer' if integer else 'a number'}", key)
    if isinstance(v, float) and not math.isfinite(v):
        raise RequestError(f"'{key}' must be finite", key)
    if v > hi or v < lo or (lo_open and v == lo):
        bound = f"({lo}" if lo_open else f"[{lo}"
        raise RequestError(f"'{key}' must be in {bound}, {hi}]", key)


def _content(content: Any, role: str, path: str) -> tuple[Any, bool]:
    """Checked content, and whether it holds an image."""
    if isinstance(content, str):
        return content, False
    if not isinstance(content, list) or not content:
        raise RequestError(f"'{path}' must be a string or a non-empty list of parts", path)
    parts = []
    has_image = False
    for i, part in enumerate(content):
        p = f"{path}[{i}]"
        if not isinstance(part, dict):
            raise RequestError(f"'{p}' must be an object", p)
        kind = part.get("type")
        if kind == "text":
            if set(part) - {"type", "text"} or not isinstance(part.get("text"), str):
                raise RequestError(f"'{p}' must be {{\"type\": \"text\", \"text\": string}}", p)
            parts.append({"type": "text", "text": part["text"]})
        elif kind == "image_url":
            if role != "user":
                raise RequestError(f"'{p}': only user messages may hold images", p)
            image = part.get("image_url")
            if set(part) - {"type", "image_url"} or not isinstance(image, dict):
                raise RequestError(f"'{p}' must be {{\"type\": \"image_url\", \"image_url\": "
                                   f"{{\"url\": ...}}}}", p)
            if set(image) - {"url", "detail"}:
                raise RequestError(f"'{p}.image_url' allows only 'url' and 'detail'", p)
            url = image.get("url")
            if not isinstance(url, str) or not _IMAGE_URL.match(url):
                # Never a web address: llama-server would fetch it from here.
                raise RequestError(f"'{p}.image_url.url' must be a base64 data: URI "
                                   f"(png, jpeg, webp, or gif)", p)
            out = {"url": url}
            if "detail" in image:
                if image["detail"] not in ("auto", "low", "high"):
                    raise RequestError(f"'{p}.image_url.detail' must be auto, low, or high", p)
                out["detail"] = image["detail"]
            parts.append({"type": "image_url", "image_url": out})
            has_image = True
        else:
            raise RequestError(f"'{p}': part type {kind!r} isn't supported "
                               f"(text and image_url are)", p)
    return parts, has_image


def check_request(body: Any) -> dict:
    """Check everything that doesn't depend on the model. Returns the
    request rebuilt from checked fields only, plus "_needs" describing the
    features apply_capabilities must confirm."""
    if not isinstance(body, dict):
        raise RequestError("the request body must be a JSON object")
    tools = sorted(set(body) & _TOOLS)
    if tools:
        raise RequestError(f"tools aren't supported by this gateway ({', '.join(tools)})",
                           tools[0], code="unsupported_parameter")
    allowed = {"model", "messages", "stream", "stream_options", "max_tokens",
               "max_completion_tokens", "temperature", "top_p", "top_k", "min_p",
               "presence_penalty", "frequency_penalty", "repeat_penalty", "seed",
               "stop", "n", "response_format", "reasoning_effort", "chat_template_kwargs"}
    unknown = sorted(set(body) - allowed)
    if unknown:
        raise RequestError(f"unsupported parameter(s): {', '.join(unknown)}", unknown[0],
                           code="unsupported_parameter")

    if not isinstance(body.get("model"), str) or not body["model"]:
        raise RequestError("'model' is required", "model")

    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise RequestError("'messages' must be a non-empty list", "messages")
    if len(messages) > MAX_MESSAGES:
        raise RequestError(f"at most {MAX_MESSAGES} messages", "messages")
    out_messages = []
    has_image = False
    uses_reasoning_content = False
    for i, m in enumerate(messages):
        p = f"messages[{i}]"
        if not isinstance(m, dict):
            raise RequestError(f"'{p}' must be an object", p)
        role = m.get("role")
        if role not in ROLES:
            raise RequestError(f"'{p}.role' must be one of {sorted(ROLES)}", p)
        extra = set(m) - {"role", "content", "reasoning_content"}
        if extra:
            raise RequestError(f"'{p}' has unsupported field(s): {', '.join(sorted(extra))}", p)
        if "content" not in m:
            raise RequestError(f"'{p}.content' is required", p)
        content, img = _content(m["content"], role, f"{p}.content")
        has_image = has_image or img
        msg = {"role": role, "content": content}
        if "reasoning_content" in m:
            if role != "assistant" or not isinstance(m["reasoning_content"], str):
                raise RequestError(f"'{p}.reasoning_content' must be a string on an "
                                   f"assistant message", p)
            msg["reasoning_content"] = m["reasoning_content"]
            uses_reasoning_content = True
        out_messages.append(msg)

    out: dict = {"messages": out_messages}

    stream = body.get("stream", False)
    if not isinstance(stream, bool):
        raise RequestError("'stream' must be a boolean", "stream")
    out["stream"] = stream
    if "stream_options" in body:
        so = body["stream_options"]
        if not stream:
            raise RequestError("'stream_options' needs 'stream': true", "stream_options")
        if not isinstance(so, dict) or set(so) - {"include_usage"} or \
                not isinstance(so.get("include_usage", False), bool):
            raise RequestError("'stream_options' allows only include_usage (boolean)",
                               "stream_options")
        out["stream_options"] = dict(so)

    _number(body, "max_tokens", 1, 10_000_000, integer=True)
    _number(body, "max_completion_tokens", 1, 10_000_000, integer=True)
    mt, mct = body.get("max_tokens"), body.get("max_completion_tokens")
    if mt is not None and mct is not None and mt != mct:
        raise RequestError("'max_tokens' and 'max_completion_tokens' differ; send one",
                           "max_tokens")
    max_tokens = mt if mt is not None else mct

    _number(body, "temperature", 0, 5)
    _number(body, "top_p", 0, 1, lo_open=True)
    _number(body, "top_k", 0, 1000, integer=True)
    _number(body, "min_p", 0, 1)
    _number(body, "presence_penalty", -2, 2)
    _number(body, "frequency_penalty", -2, 2)
    _number(body, "repeat_penalty", 0, 5, lo_open=True)
    _number(body, "seed", -1, 2**32 - 1, integer=True)
    for key in ("temperature", "top_p", "top_k", "min_p", "presence_penalty",
                "frequency_penalty", "repeat_penalty", "seed"):
        if key in body:
            out[key] = body[key]

    if "n" in body and body["n"] != 1:
        raise RequestError("only 'n': 1 is supported", "n")

    if "stop" in body:
        stop = body["stop"]
        if isinstance(stop, str):
            stop = [stop]
        if not isinstance(stop, list) or len(stop) > MAX_STOP or \
                not all(isinstance(s, str) and 0 < len(s) <= 256 for s in stop):
            raise RequestError(f"'stop' must be a string or up to {MAX_STOP} strings", "stop")
        out["stop"] = stop

    if "response_format" in body:
        rf = body["response_format"]
        kind = rf.get("type") if isinstance(rf, dict) else None
        if kind in ("text", "json_object") and set(rf) == {"type"}:
            out["response_format"] = {"type": kind}
        elif kind == "json_schema" and set(rf) == {"type", "json_schema"} and \
                isinstance(rf["json_schema"], dict) and \
                isinstance(rf["json_schema"].get("schema"), dict) and \
                not set(rf["json_schema"]) - {"name", "schema", "strict", "description"}:
            out["response_format"] = {"type": "json_schema", "json_schema": rf["json_schema"]}
        else:
            raise RequestError("'response_format' must be text, json_object, or "
                               "json_schema with a 'schema' object", "response_format")

    effort = body.get("reasoning_effort")
    if effort is not None and not isinstance(effort, str):
        raise RequestError("'reasoning_effort' must be a string", "reasoning_effort")
    enable_thinking = None
    if "chat_template_kwargs" in body:
        kw = body["chat_template_kwargs"]
        if not isinstance(kw, dict) or set(kw) - {"enable_thinking"}:
            raise RequestError("'chat_template_kwargs' allows only enable_thinking",
                               "chat_template_kwargs", code="unsupported_parameter")
        if "enable_thinking" in kw:
            if not isinstance(kw["enable_thinking"], bool):
                raise RequestError("'chat_template_kwargs.enable_thinking' must be a boolean",
                                   "chat_template_kwargs")
            enable_thinking = kw["enable_thinking"]

    out["_needs"] = {"images": has_image, "reasoning_effort": effort,
                     "enable_thinking": enable_thinking, "max_tokens": max_tokens,
                     "reasoning_content": uses_reasoning_content}
    return out


def apply_capabilities(checked: dict, route: Route, caps: dict | None) -> dict:
    """Refuse features the model's server hasn't been verified to support,
    apply the output cap, and return the upstream body. caps None means
    nothing is verified yet, so only plain requests pass."""
    needs = checked["_needs"]
    body = {k: v for k, v in checked.items() if k != "_needs"}
    reasoning = (caps or {}).get("reasoning") or {}

    if needs["images"] and (caps or {}).get("images") is not True:
        why = "doesn't accept images" if caps and caps.get("images") is False else \
              "hasn't been verified to accept images"
        raise RequestError(f"'{route.name}' {why}", "messages", code="capability_missing")

    effort = needs["reasoning_effort"]
    if effort is not None:
        levels = reasoning.get("effort_levels") or []
        if effort not in levels:
            if reasoning.get("disabled_by_profile"):
                why = "has thinking turned off by its profile"
            elif caps is None:
                why = "hasn't been verified to accept reasoning_effort"
            elif not levels:
                why = "doesn't accept reasoning_effort"
            else:
                why = f"accepts reasoning_effort {', '.join(levels)}"
            raise RequestError(f"'{route.name}' {why}", "reasoning_effort",
                               code="capability_missing")
        body["reasoning_effort"] = effort

    thinking = needs["enable_thinking"]
    if thinking is not None:
        if reasoning.get("enable_thinking") is True:
            body["chat_template_kwargs"] = {"enable_thinking": thinking}
        elif thinking is False and reasoning.get("disabled_by_profile"):
            pass  # already off; asking for off changes nothing
        else:
            why = "has thinking turned off by its profile" if reasoning.get("disabled_by_profile") \
                else "hasn't been verified to read enable_thinking" if caps is None \
                else "has a chat template that doesn't read enable_thinking"
            raise RequestError(f"'{route.name}' {why}", "chat_template_kwargs",
                               code="capability_missing")

    if needs["reasoning_content"] and reasoning.get("preserve_reasoning") is not True:
        raise RequestError(f"'{route.name}' hasn't been verified to take reasoning_content "
                           f"in earlier messages", "messages", code="capability_missing")

    limits = [l for l in (route.max_output_tokens, (caps or {}).get("context_tokens")) if l]
    cap = min(limits) if limits else None
    requested = needs["max_tokens"]
    if requested is not None:
        if cap is not None and requested > cap:
            raise RequestError(f"'max_tokens' is more than '{route.name}' allows ({cap})",
                               "max_tokens")
        body["max_tokens"] = requested
    elif route.max_output_tokens is not None:
        body["max_tokens"] = route.max_output_tokens

    body["model"] = route.model
    return body
