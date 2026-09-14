"""结构化输出的 method 选择与降级。

背景:create_react_agent 的 response_format 内部固定调 with_structured_output(schema)
不带 method,langchain-openai 默认选 json_schema,而本网关的 OpenAI 端点回
400 "This response_format type is unavailable now";换 function_calling 又会撞上
"Thinking mode does not support this tool_choice"。所以两步拆开、method 可降级。
"""

import pytest

from app import llm


class _BadRequest(Exception):
    status_code = 400


class _ServerError(Exception):
    status_code = 500


class _FakeModel:
    """只记录被请求了哪些 method,并按预设抛错。"""

    def __init__(self, fail: dict, result="ok"):
        self.fail, self.result, self.tried = fail, result, []

    def with_structured_output(self, schema, method=None):
        self.tried.append(method)
        fail = self.fail.get(method)

        class _R:
            def __init__(self, result):
                self._result = result

            def invoke(self, messages):
                if fail:
                    raise fail
                return self._result

        return _R(self.result)


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.setattr("app.config.LLM_STRUCTURED_METHOD", "")
    monkeypatch.setattr("app.config.LLM_DISABLE_THINKING", True)
    yield


def test_is_protocol_error_only_for_400():
    assert llm._is_protocol_error(_BadRequest())
    assert not llm._is_protocol_error(_ServerError())
    assert not llm._is_protocol_error(ValueError("boom"))


def test_falls_back_to_next_method_on_400():
    """function_calling 被 thinking 拒掉时应自动退到 json_mode,而不是让整节点失败。

    这是实测的报错:400 "Thinking mode does not support this tool_choice"。
    """
    model = _FakeModel(
        fail={"function_calling": _BadRequest("Thinking mode does not support this tool_choice")},
        result="ok")
    llm.set_model(model, "openai")
    out = llm.structured_model(dict).invoke([])
    assert out == "ok"
    assert model.tried == ["function_calling", "json_mode"]


def test_function_calling_preferred_over_json_schema():
    """OpenAI 通道首选 function_calling:保留 tool-schema 抽取,比自由生成 JSON 可靠。"""
    model = _FakeModel(fail={})
    llm.set_model(model, "openai")
    llm.structured_model(dict).invoke([])
    assert model.tried == ["function_calling"]


def test_non_protocol_error_is_not_swallowed():
    """500/网络类错误照常抛出,不能靠换 method 掩盖真实故障。"""
    model = _FakeModel(fail={"function_calling": _ServerError("down")})
    llm.set_model(model, "openai")
    with pytest.raises(_ServerError):
        llm.structured_model(dict).invoke([])
    assert model.tried == ["function_calling"]


def test_all_methods_failing_raises_with_context():
    model = _FakeModel(fail={"function_calling": _BadRequest("a"),
                             "json_mode": _BadRequest("b")})
    llm.set_model(model, "openai")
    with pytest.raises(RuntimeError) as ei:
        llm.structured_model(dict).invoke([])
    assert "结构化输出" in str(ei.value)


def test_explicit_method_override_disables_fallback(monkeypatch):
    """LLM_STRUCTURED_METHOD 显式指定时只用它,不再自动降级。"""
    monkeypatch.setattr("app.config.LLM_STRUCTURED_METHOD", "json_mode")
    model = _FakeModel(fail={"json_mode": _BadRequest("nope")})
    llm.set_model(model, "openai")
    with pytest.raises(RuntimeError):
        llm.structured_model(dict).invoke([])
    assert model.tried == ["json_mode"]


def test_thinking_disabled_kwargs_per_transport():
    """两条通道关 thinking 的写法不同:Anthropic 顶层参数,OpenAI 放 extra_body。"""
    llm.config.LLM_DISABLE_THINKING = True
    assert llm._thinking_disabled_kwargs("anthropic") == {"thinking": {"type": "disabled"}}
    assert llm._thinking_disabled_kwargs("openai") == {
        "extra_body": {"thinking": {"type": "disabled"}}}
    llm.config.LLM_DISABLE_THINKING = False
    assert llm._thinking_disabled_kwargs("openai") == {}
    llm.config.LLM_DISABLE_THINKING = True
