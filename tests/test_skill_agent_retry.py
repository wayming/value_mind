"""run_skill_agent 拿不到结构化参数时的回喂重试。

背景(实测 2026-09-16,NAB 估值):08 综合节点给出的 4KB 中文散文参数里,模型在一处写了
未转义的英文双引号 `属"温和增价值"`,args 不是合法 JSON → langchain 把它放进
invalid_tool_calls 后既不解析也不报错、agent 直接返回 → structured_response 为 None。
这类失败必须在应用侧补一次回喂,否则整个节点白白作废。

这里把模型/MCP/skill 文档全换成桩,只让重试逻辑走真实代码。
"""

import asyncio
import types

import pytest
from langchain_core.messages import AIMessage
from langgraph.errors import GraphRecursionError

import app.nodes as nodes


class _FakeAgent:
    def __init__(self, out, states):
        self._out, self._states = out, states

    async def ainvoke(self, state, config=None):
        self._states.append(state)
        if isinstance(self._out, Exception):
            raise self._out
        return self._out


@pytest.fixture
def harness(monkeypatch):
    """记录每次 create_agent 的调用参数与 ainvoke 的消息,按序返回预置的 agent。"""
    created: list[dict] = []
    states: list[dict] = []
    outs: list[dict] = []

    def fake_create_agent(**kwargs):
        created.append(kwargs)
        return _FakeAgent(outs[len(created) - 1], states)

    async def fake_make_mcp_tools():
        return ["获取财务数据的工具"], {}

    monkeypatch.setattr(nodes, "load_skill", lambda sid: types.SimpleNamespace(body="技能正文"))
    monkeypatch.setattr(nodes, "make_mcp_tools", fake_make_mcp_tools)
    monkeypatch.setattr(nodes, "get_model", lambda: object())
    monkeypatch.setattr(nodes, "create_agent", fake_create_agent)
    return types.SimpleNamespace(created=created, outs=outs, states=states,
                                 prompts=lambda: [c["system_prompt"] for c in created])


def _out(structured=None, messages=None):
    return {"messages": messages or [], "structured_response": structured}


# 复刻实测的坏参数:中文散文里一处未转义的英文双引号(NAB 综合节点那份 4KB 参数的缩影)
_BROKEN_ARGS = '{"method_reconciliation_note": "' + "中文" * 20 + '"温和增价值"}'


def _langchain_error(args, pos):
    """langchain 塞进 invalid_tool_call.error 的原文(它会把整份 args 也带上)。"""
    return (f"Function SynthesisParams arguments:\n\n{args}\n\nare not valid JSON. "
            f"Received JSONDecodeError Expecting ',' delimiter: line 1 column {pos + 1} "
            f"(char {pos})\nFor troubleshooting, visit: "
            f"https://docs.langchain.com/oss/python/langchain/errors/OUTPUT_PARSING_FAILURE")


def _bad_json_message(args=None, error=None):
    """模型调了结构化工具,但 args 不是合法 JSON(未转义的双引号)。"""
    args = _BROKEN_ARGS if args is None else args
    if error is None:
        pos = args.index('"温和') if '"温和' in args else 4
        error = _langchain_error(args, pos)
    return AIMessage(content="", invalid_tool_calls=[{
        "type": "invalid_tool_call", "id": "call_1", "name": "SynthesisParams",
        "args": args, "error": error,
    }])


def test_invalid_tool_call_retries_with_escaping_hint(harness):
    """参数 JSON 不合法:回喂一次,并把"别用英文双引号"明确说给模型。"""
    harness.outs += [_out(None, [_bad_json_message()]), _out({"verdict": "合理"})]

    params, cache = asyncio.run(nodes.run_skill_agent("08_synthesize", {}, dict))

    assert params == {"verdict": "合理"}
    first, retry = harness.prompts()
    assert "英文双引号" in retry and "英文双引号" not in first
    assert retry.startswith(first)                   # 原 prompt 保留,只是追加回喂话术
    # 重试仍带着工具(与"步数耗尽降级"不同:那次模型已经有数据,这次是参数转义手滑,应允许它取数)
    assert harness.created[1]["tools"] == harness.created[0]["tools"] == ["获取财务数据的工具"]


def test_retry_echoes_the_broken_args_and_error_position(harness):
    """回喂"你上一轮的原样输出 + 解析报错 + 出错位置":让它改转义,而不是重做分析。"""
    harness.outs += [_out(None, [_bad_json_message()]), _out({"verdict": "合理"})]

    asyncio.run(nodes.run_skill_agent("08_synthesize", {}, dict))

    [_, retry_state] = harness.states
    ctx, echo = retry_state["messages"]
    assert ctx.type == "human" and echo.type == "human"
    assert _BROKEN_ARGS in echo.content                   # 原文完整回喂
    assert "出错位置附近的原文" in echo.content
    assert '属' not in echo.content or '"温和增价值"' in echo.content  # 窗口落在出错处
    assert "Received JSONDecodeError" in echo.content     # 只取关键那句
    assert "For troubleshooting" not in echo.content      # 不带 langchain 的整段文案


def test_oversize_args_fall_back_to_generic_hint(harness):
    """参数过大就不回喂(否则输入翻倍),但通用转义提示照给。"""
    big = '{"note": "' + "长" * 9000 + '"}'
    harness.outs += [_out(None, [_bad_json_message(args=big)]), _out({"verdict": "合理"})]

    asyncio.run(nodes.run_skill_agent("08_synthesize", {}, dict))

    assert len(harness.states[1]["messages"]) == 1        # 只有原始 ctx,没有回喂
    assert "中文引号" in harness.prompts()[1]


def test_missing_tool_call_gets_its_own_hint(harness):
    """压根没调工具(只在正文里分析)是另一种原因,话术不能混。"""
    harness.outs += [_out(None, [AIMessage(content="分析了一大段但没出参数")]),
                     _out({"verdict": "合理"})]

    asyncio.run(nodes.run_skill_agent("08_synthesize", {}, dict))

    retry = harness.prompts()[1]
    assert "没有输出结构化参数对象" in retry
    assert "英文双引号" not in retry


def test_gives_up_after_one_retry(harness):
    """回喂一次仍拿不到就报错(节点据此记 degraded),不能无限重试烧 token。"""
    harness.outs += [_out(None, [_bad_json_message()]), _out(None, [_bad_json_message()])]

    with pytest.raises(RuntimeError) as ei:
        asyncio.run(nodes.run_skill_agent("08_synthesize", {}, dict))

    assert len(harness.created) == 2
    assert "回喂重试后仍失败" in str(ei.value)


def test_success_does_not_retry(harness):
    """正常路径只有一次调用,不受重试逻辑影响。"""
    harness.outs.append(_out({"verdict": "合理"}))

    params, _ = asyncio.run(nodes.run_skill_agent("08_synthesize", {}, dict))

    assert params == {"verdict": "合理"}
    assert len(harness.created) == 1


def test_recursion_downgrade_still_retries_without_tools(harness):
    """既有的步数耗尽降级路径不能被改坏:禁止再用工具,并要求标注假设。"""
    harness.outs += [GraphRecursionError("step limit"), _out({"verdict": "合理"})]

    params, _ = asyncio.run(nodes.run_skill_agent("08_synthesize", {}, dict))

    assert params == {"verdict": "合理"}
    assert harness.created[1]["tools"] == []
    assert "不得再调用任何工具" in harness.prompts()[1]


def test_no_tools_node_gets_the_no_tools_note_not_the_tool_guide(harness):
    """无工具节点(08 综合)不能带"工具使用规范":那份规范在教它什么时候取数,而它没有工具。

    实测 08 就是带着这份规范去调 get_data_period/get_financials 各一次的(重复抓 5y)。
    """
    harness.outs.append(_out({"verdict": "合理"}))

    asyncio.run(nodes.run_skill_agent("08_synthesize", {}, dict, allow_tools=False))

    prompt = harness.prompts()[0]
    assert harness.created[0]["tools"] == []
    assert "不提供数据工具" in prompt
    assert "取数有硬额度" not in prompt and "list_metrics" not in prompt


def test_tool_node_gets_the_budget_rule(harness):
    """有工具节点必须看到硬额度与去重规则(账本会强制执行)。"""
    harness.outs.append(_out({"verdict": "合理"}))

    asyncio.run(nodes.run_skill_agent("03_dividends_growth", {}, dict))

    prompt = harness.prompts()[0]
    assert "最多 **2 次** `get_financials`" in prompt
    assert "by_year" in prompt                      # 逐年数据去哪找,而不是换 period 去抓
