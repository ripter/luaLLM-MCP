"""Gateway configuration, token, capabilities, and request validation."""

import json
import os

import pytest

from gateway_config import (CapabilityCache, ConfigError, Route, capabilities_from_props,
                            create_token, load_config, load_token)
from gateway_validate import RequestError, apply_capabilities, check_request
from profiles import Profile, ProfileRegistry


def _write(tmp_path, gateway=None, profiles=None):
    if gateway is not None:
        (tmp_path / "gateway.json").write_text(json.dumps(gateway))
    if profiles is not None:
        (tmp_path / "profiles.json").write_text(json.dumps({"profiles": profiles}))


def _load(tmp_path):
    return load_config(tmp_path, ProfileRegistry(state_dir=str(tmp_path)))


# -- configuration --

def test_config_resolves_routes_with_profile_snapshots(tmp_path):
    _write(tmp_path, {"models": {"qwen": {"model": "Q", "profile": "big",
                                           "max_output_tokens": 512}}},
           {"big": {"args": ["-c", "32768"], "preset": "throughput"}})
    config = _load(tmp_path)
    assert (config.host, config.port) == ("127.0.0.1", 8090)
    route = config.routes["qwen"]
    assert route.model == "Q"
    assert route.start_args == ["-c", "32768"]
    assert route.profile.preset == "throughput"
    assert route.max_output_tokens == 512
    assert config.warnings == []


def test_config_skips_broken_entries_and_says_why(tmp_path):
    _write(tmp_path, {"models": {
        "ok": {"model": "Q"},
        "no-profile": {"model": "Q", "profile": "missing"},
        "extra": {"model": "Q", "flags": ["-c", "1"]},
        "bad name!": {"model": "Q"},
        "bad-cap": {"model": "Q", "max_output_tokens": 0},
    }}, {})
    config = _load(tmp_path)
    assert list(config.routes) == ["ok"]
    text = " ".join(config.warnings)
    for fragment in ("missing", "flags", "bad name!", "max_output_tokens"):
        assert fragment in text


def test_config_default_model_names_a_served_model(tmp_path):
    _write(tmp_path, {"models": {"qwen": {"model": "Q"}}}, {})
    assert _load(tmp_path).default_model is None

    _write(tmp_path, {"default_model": "qwen", "models": {"qwen": {"model": "Q"}}}, {})
    config = _load(tmp_path)
    assert config.default_model == "qwen" and config.warnings == []

    # A default whose entry is broken, or missing, is no default: never another model instead.
    _write(tmp_path, {"default_model": "eyes", "models": {
        "qwen": {"model": "Q"}, "eyes": {"model": "Q", "profile": "missing"}}}, {})
    config = _load(tmp_path)
    assert config.default_model is None
    assert any("default_model 'eyes'" in w for w in config.warnings)

    _write(tmp_path, {"default_model": 3, "models": {}}, {})
    with pytest.raises(ConfigError, match="default_model"):
        _load(tmp_path)


def test_config_refuses_non_loopback_unless_asked(tmp_path):
    _write(tmp_path, {"listen": "0.0.0.0:8090", "models": {}})
    with pytest.raises(ConfigError, match="isn't loopback"):
        _load(tmp_path)
    _write(tmp_path, {"listen": "192.168.1.5:8090", "models": {}})
    with pytest.raises(ConfigError, match="isn't loopback"):
        _load(tmp_path)
    _write(tmp_path, {"listen": "0.0.0.0:8090", "allow_non_loopback": True, "models": {}})
    assert _load(tmp_path).host == "0.0.0.0"
    _write(tmp_path, {"listen": "localhost:9000", "models": {}})
    assert _load(tmp_path).port == 9000


def test_config_missing_or_broken_file_is_an_error(tmp_path):
    with pytest.raises(ConfigError, match="No gateway configuration"):
        _load(tmp_path)
    (tmp_path / "gateway.json").write_text("{nope")
    with pytest.raises(ConfigError, match="could not parse"):
        _load(tmp_path)


def test_fingerprint_changes_with_the_flags():
    a = Route("x", "Q", Profile("p", ["-c", "4096"]))
    b = Route("x", "Q", Profile("p", ["-c", "8192"]))
    assert a.fingerprint != b.fingerprint
    assert a.fingerprint == Route("x", "Q", Profile("p", ["-c", "4096"])).fingerprint


@pytest.mark.parametrize("args,off", [
    (["--reasoning", "off"], True), (["-rea", "off"], True), (["--reasoning=off"], True),
    (["--reasoning-budget", "0"], True), (["--reasoning-budget", "128"], False),
    (["--reasoning", "on"], False), ([], False)])
def test_reasoning_disabled_by_flags(args, off):
    assert Route("x", "Q", Profile("p", args)).reasoning_disabled_by_flags is off


# -- token --

def test_token_file_must_be_private_and_long(tmp_path, monkeypatch):
    monkeypatch.delenv("LUALLM_GATEWAY_TOKEN", raising=False)
    with pytest.raises(ConfigError, match="No token"):
        load_token(tmp_path)
    path, created = create_token(tmp_path)
    assert created and oct(path.stat().st_mode & 0o777) == "0o600"
    assert len(load_token(tmp_path)) >= 32
    assert create_token(tmp_path) == (path, False)
    path.chmod(0o644)
    with pytest.raises(ConfigError, match="readable by others"):
        load_token(tmp_path)
    monkeypatch.setenv("LUALLM_GATEWAY_TOKEN", "short")
    with pytest.raises(ConfigError, match="shorter"):
        load_token(tmp_path)


def test_token_errors_never_include_the_token(tmp_path, monkeypatch):
    monkeypatch.setenv("LUALLM_GATEWAY_TOKEN", "secret-but-short")
    with pytest.raises(ConfigError) as e:
        load_token(tmp_path)
    assert "secret-but-short" not in str(e.value)


# -- capabilities --

PROPS = {
    "modalities": {"vision": True},
    "chat_template": "{% if enable_thinking %}...{% endif %}",
    "chat_template_caps": {"supports_reasoning_effort": True,
                           "reasoning_effort_levels": ["low", "medium", "xhigh"],
                           "supports_preserve_reasoning": True},
    "default_generation_settings": {"n_ctx": 65536},
}


def test_capabilities_come_from_props():
    caps = capabilities_from_props(PROPS, Route("x", "Q", None, max_output_tokens=4096))
    assert caps == {"images": True, "context_tokens": 65536, "max_output_tokens": 4096,
                    "reasoning": {"disabled_by_profile": False,
                                  "effort_levels": ["low", "medium", "xhigh"],
                                  "enable_thinking": True, "preserve_reasoning": True}}


def test_capabilities_unknown_stays_unknown():
    caps = capabilities_from_props({}, Route("x", "Q", None))
    assert caps["images"] is None and caps["context_tokens"] is None
    assert caps["reasoning"]["effort_levels"] == []
    assert caps["reasoning"]["enable_thinking"] is False


def test_profile_turning_thinking_off_overrides_the_template():
    caps = capabilities_from_props(PROPS, Route("x", "Q", Profile("p", ["--reasoning", "off"])))
    assert caps["reasoning"] == {"disabled_by_profile": True, "effort_levels": [],
                                 "enable_thinking": False, "preserve_reasoning": True}


def test_capability_cache_forgets_entries_when_the_route_changes(tmp_path):
    cache = CapabilityCache(tmp_path)
    a = Route("x", "Q", Profile("p", ["-c", "4096"]))
    cache.put(a, {"images": True})
    assert cache.get(a)["capabilities"] == {"images": True}
    assert cache.get(Route("x", "Q", Profile("p", ["-c", "8192"]))) is None


# -- validation --

def _req(**extra):
    return {"model": "qwen", "messages": [{"role": "user", "content": "hi"}], **extra}


def _caps(**over):
    caps = capabilities_from_props(PROPS, Route("qwen", "Q", None))
    caps.update(over)
    return caps


def test_upstream_body_is_rebuilt_from_checked_fields_only():
    route = Route("qwen", "Qwen-file", None, max_output_tokens=256)
    body = apply_capabilities(check_request(_req(temperature=0.7, stop="END")), route, _caps())
    assert body == {"messages": [{"role": "user", "content": "hi"}], "stream": False,
                    "temperature": 0.7, "stop": ["END"], "max_tokens": 256, "model": "Qwen-file"}


@pytest.mark.parametrize("field,value", [
    ("id_slot", 0), ("cache_prompt", True), ("lora", [{"id": 0, "scale": 1}]),
    ("grammar", "root ::= x"), ("n_probs", 5), ("samplers", ["top_k"]), ("logit_bias", {}),
    ("user", "me"), ("logprobs", True)])
def test_llama_cpp_controls_and_unknown_fields_are_refused(field, value):
    with pytest.raises(RequestError) as e:
        check_request(_req(**{field: value}))
    assert e.value.code == "unsupported_parameter" and e.value.param == field


def test_tools_are_refused():
    with pytest.raises(RequestError, match="tools aren't supported"):
        check_request(_req(tools=[]))
    with pytest.raises(RequestError, match="role"):
        check_request({"model": "q", "messages": [{"role": "tool", "content": "x"}]})
    with pytest.raises(RequestError, match="unsupported field"):
        check_request({"model": "q", "messages": [
            {"role": "assistant", "content": "x", "tool_calls": []}]})


def test_image_urls_must_be_inline_data():
    def img(url):
        return _req(messages=[{"role": "user", "content": [
            {"type": "text", "text": "what"}, {"type": "image_url", "image_url": {"url": url}}]}])
    check_request(img("data:image/png;base64,iVBORw0KGgo="))
    for url in ("http://192.168.1.1/x.png", "file:///etc/passwd", "data:text/html;base64,AA=="):
        with pytest.raises(RequestError, match="data: URI"):
            check_request(img(url))


def test_images_need_verified_vision():
    req = check_request(_req(messages=[{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}]))
    route = Route("qwen", "Q", None)
    apply_capabilities(req, route, _caps())
    with pytest.raises(RequestError, match="doesn't accept images"):
        apply_capabilities(req, route, _caps(images=False))
    with pytest.raises(RequestError, match="hasn't been verified"):
        apply_capabilities(req, route, None)


def test_reasoning_effort_must_be_a_level_the_template_accepts():
    route = Route("qwen", "Q", None)
    ok = apply_capabilities(check_request(_req(reasoning_effort="xhigh")), route, _caps())
    assert ok["reasoning_effort"] == "xhigh"
    with pytest.raises(RequestError, match="low, medium, xhigh"):
        apply_capabilities(check_request(_req(reasoning_effort="high")), route, _caps())
    with pytest.raises(RequestError, match="hasn't been verified"):
        apply_capabilities(check_request(_req(reasoning_effort="low")), route, None)


def test_thinking_controls_respect_a_profile_that_turns_thinking_off():
    route = Route("qwen", "Q", Profile("p", ["--reasoning", "off"]))
    caps = capabilities_from_props(PROPS, route)
    with pytest.raises(RequestError, match="turned off by its profile"):
        apply_capabilities(check_request(_req(reasoning_effort="low")), route, caps)
    with pytest.raises(RequestError, match="turned off by its profile"):
        apply_capabilities(check_request(_req(chat_template_kwargs={"enable_thinking": True})),
                           route, caps)
    # Asking for "off" where it's already off changes nothing, so it's accepted
    # and not forwarded.
    body = apply_capabilities(check_request(_req(chat_template_kwargs={"enable_thinking": False})),
                              route, caps)
    assert "chat_template_kwargs" not in body


def test_enable_thinking_only_where_the_template_reads_it():
    route = Route("qwen", "Q", None)
    req = check_request(_req(chat_template_kwargs={"enable_thinking": False}))
    assert apply_capabilities(req, route, _caps())["chat_template_kwargs"] == {"enable_thinking": False}
    no_var = capabilities_from_props({**PROPS, "chat_template": "plain"}, route)
    with pytest.raises(RequestError, match="doesn't read enable_thinking"):
        apply_capabilities(req, route, no_var)
    with pytest.raises(RequestError, match="only enable_thinking"):
        check_request(_req(chat_template_kwargs={"reasoning_effort": "low"}))


def test_max_tokens_is_capped_and_defaulted():
    route = Route("qwen", "Q", None, max_output_tokens=1000)
    with pytest.raises(RequestError, match="1000"):
        apply_capabilities(check_request(_req(max_tokens=1001)), route, _caps())
    assert apply_capabilities(check_request(_req(max_completion_tokens=10)), route,
                              _caps())["max_tokens"] == 10
    unbounded = Route("qwen", "Q", None)
    assert "max_tokens" not in apply_capabilities(check_request(_req()), unbounded, _caps())
    with pytest.raises(RequestError, match="65536"):
        apply_capabilities(check_request(_req(max_tokens=70000)), unbounded, _caps())


@pytest.mark.parametrize("field,value", [
    ("temperature", -1), ("temperature", "hot"), ("temperature", True), ("top_p", 0),
    ("top_k", 1.5), ("n", 2), ("stop", ["a"] * 9), ("max_tokens", 0), ("stream", "yes"),
    ("response_format", {"type": "grammar"}), ("temperature", float("nan"))])
def test_values_are_checked(field, value):
    with pytest.raises(RequestError):
        check_request(_req(**{field: value}))


def test_structured_output_is_allowed():
    schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    body = check_request(_req(response_format={"type": "json_schema",
                                               "json_schema": {"name": "x", "schema": schema}}))
    assert body["response_format"]["json_schema"]["schema"] == schema


def test_reasoning_content_needs_preserve_support():
    req = check_request(_req(messages=[
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b", "reasoning_content": "thought"},
        {"role": "user", "content": "c"}]))
    route = Route("qwen", "Q", None)
    assert apply_capabilities(req, route, _caps())["messages"][1]["reasoning_content"] == "thought"
    caps = _caps()
    caps["reasoning"] = {**caps["reasoning"], "preserve_reasoning": False}
    with pytest.raises(RequestError, match="reasoning_content"):
        apply_capabilities(req, route, caps)
