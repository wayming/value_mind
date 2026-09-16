"""llm.out 全量对话日志。

不需要 LLM 与网络:直接把 langchain 的原始事件对象(消息、LLMResult、异常)喂给
handler,断言落盘内容与配对、脱敏、不打断估值这几条硬要求。
"""

import asyncio
import json
import threading
import uuid

import pytest
import yaml
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, LLMResult

import app.config as config
from app import llm, llm_log


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    """每条测试独立:日志打开、路径指向 tmp,并清掉模块级单例。"""
    monkeypatch.setattr(config, "LLM_LOG", True)
    monkeypatch.setattr(config, "LLM_LOG_PATH", str(tmp_path / "llm.out"))
    monkeypatch.setattr(llm_log, "_HANDLER", None)
    return tmp_path / "llm.out"


def _start(handler, messages, run_id=None, serialized=None, **kw):
    run_id = run_id or uuid.uuid4()
    handler.on_chat_model_start(serialized or {}, [messages], run_id=run_id, **kw)
    return run_id


def _end(handler, run_id, content="分析完成", tool_calls=None, usage=None):
    msg = AIMessage(content=content, tool_calls=tool_calls or [],
                    usage_metadata=usage)
    handler.on_llm_end(LLMResult(generations=[[ChatGeneration(message=msg)]]), run_id=run_id)


def _records(path):
    """文件里的全部记录(会话头是注释,self 会被 safe_load_all 跳过)。"""
    return [r for r in yaml.safe_load_all(path.read_text(encoding="utf-8"))
            if isinstance(r, dict)]


def test_full_input_and_output_written(_isolate):
    """文本不截断:超长的 skill 提示词、工具返回、响应正文都要原样在文件里。"""
    skill_text = "你是 DDM 估值分析师。" + "假设边界:" + "X" * 4000
    tool_payload = json.dumps({"roe": list(range(200)), "note": "工具返回" * 200})
    handler = llm_log.LLMLogHandler(path=_isolate)

    run_id = _start(handler, [
        SystemMessage(content=skill_text),
        HumanMessage(content='{"exchange":"NYSE","code":"WFC"}'),
        ToolMessage(content=tool_payload, tool_call_id="call_1"),
    ], metadata={"vm_skill": "04_ddm", "langgraph_node": "model", "ls_model_name": "deepseek-flash"})
    _end(handler, run_id, content="采用两阶段模型,终值派息率 65.12%",
         tool_calls=[{"name": "get_financials", "args": {"metrics": ["roe"]}, "id": "call_1"}],
         usage={"input_tokens": 1234, "output_tokens": 56, "total_tokens": 1290})

    text = _isolate.read_text(encoding="utf-8")
    assert "X" * 4000 in text                                    # 文件里没被截断
    [rec] = _records(_isolate)
    assert rec["event"] == "end"
    assert rec["skill"] == "04_ddm"                               # 归属靠 metadata.vm_skill
    assert rec["model"] == "deepseek-flash"
    assert rec["usage"] == {"input_tokens": 1234, "output_tokens": 56, "total_tokens": 1290}
    assert rec["request"]["messages"][0]["content"] == skill_text # 提示词全文
    assert rec["request"]["messages"][2]["content"] == tool_payload
    assert rec["request"]["messages"][2]["tool_call_id"] == "call_1"
    assert rec["response"]["generations"][0]["content"].startswith("采用两阶段模型")
    assert rec["response"]["generations"][0]["tool_calls"][0]["name"] == "get_financials"
    assert rec["elapsed_ms"] >= 0


def test_multiline_content_is_written_as_real_newlines(_isolate):
    """正文里的 \\n 落盘后是真换行,不是转义(所以用 YAML 字面块而不是 JSON)。"""
    handler = llm_log.LLMLogHandler(path=_isolate)
    run_id = _start(handler, [HumanMessage(content="第一问\n第二问:终值占多少")])
    _end(handler, run_id, content="结论一\n\n结论二:终值派息率 65.12%\n  - 缩进子项")

    text = _isolate.read_text(encoding="utf-8")
    assert "结论一\n\n" in text and "结论二:终值派息率 65.12%\n" in text   # 换行、空行是真的
    assert "\\n" not in text                                              # 没有转义换行
    [rec] = _records(_isolate)
    # 缩进与空行原样回来(字面块的内容相对块缩进保留)
    assert rec["response"]["generations"][0]["content"] == "结论一\n\n结论二:终值派息率 65.12%\n  - 缩进子项"
    assert rec["request"]["messages"][0]["content"] == "第一问\n第二问:终值占多少"


def test_pairing_is_by_run_id_even_out_of_order(_isolate):
    """并发扇出时 end 的到达顺序与 start 无关,各自必须配到自己那份输入。"""
    handler = llm_log.LLMLogHandler(path=_isolate)
    run_a, run_b = uuid.uuid4(), uuid.uuid4()
    _start(handler, [HumanMessage(content="A 的输入")], run_id=run_a)
    _start(handler, [HumanMessage(content="B 的输入")], run_id=run_b)
    _end(handler, run_b, content="B 的输出")
    _end(handler, run_a, content="A 的输出")

    recs = {r["run_id"]: r for r in _records(_isolate)}
    a, b = recs[str(run_a)], recs[str(run_b)]
    assert a["request"]["messages"][0]["content"] == "A 的输入"
    assert a["response"]["generations"][0]["content"] == "A 的输出"
    assert b["request"]["messages"][0]["content"] == "B 的输入"
    assert b["response"]["generations"][0]["content"] == "B 的输出"


def test_error_is_recorded_with_request(_isolate):
    """失败的那次对话最值得查:没有 response,但输入与报错都要留下。"""
    handler = llm_log.LLMLogHandler(path=_isolate)
    run_id = _start(handler, [HumanMessage(content="会失败的问题")])
    handler.on_llm_error(ValueError("Thinking mode does not support this tool_choice"),
                         run_id=run_id)

    text = _isolate.read_text(encoding="utf-8")
    [rec] = _records(_isolate)
    assert rec["event"] == "error" and rec["response"] is None
    assert rec["request"]["messages"][0]["content"] == "会失败的问题"
    assert "does not support this tool_choice" in rec["error"]
    assert "does not support this tool_choice" in text


def test_secrets_are_not_written(_isolate):
    """凭证字段不落盘(serialized 里可能带 api_key,metadata 里可能带 token)。"""
    handler = llm_log.LLMLogHandler(path=_isolate)
    run_id = _start(handler, [HumanMessage(content="hi")],
                    serialized={"id": ["langchain", "chat_models", "openai", "ChatOpenAI"],
                                "kwargs": {"model": "deepseek-flash", "api_key": "sk-LEAK",
                                           "temperature": 0}},
                    metadata={"vm_skill": "01_company_classifier", "access_token": "sk-LEAK"})
    _end(handler, run_id)

    text = _isolate.read_text(encoding="utf-8")
    assert "sk-LEAK" not in text
    [rec] = _records(_isolate)
    assert rec["invocation"]["model"] == "deepseek-flash"    # 白名单键仍在
    assert rec["metadata"]["vm_skill"] == "01_company_classifier"
    assert "access_token" not in rec["metadata"]


def test_write_failure_never_breaks_the_run(tmp_path):
    """日志写不进去(路径是目录)也不能抛:估值流程优先。"""
    handler = llm_log.LLMLogHandler(path=tmp_path)
    run_id = _start(handler, [HumanMessage(content="hi")])
    _end(handler, run_id)          # 不抛即通过


def test_disabled_writes_nothing(monkeypatch, tmp_path):
    """LLM_LOG=0:挂不上回调,即便手动调用 handler 也不落盘。"""
    monkeypatch.setattr(config, "LLM_LOG", False)
    monkeypatch.setattr(config, "LLM_LOG_PATH", str(tmp_path / "llm.out"))
    monkeypatch.setattr(llm_log, "_HANDLER", None)
    assert llm_log.llm_callbacks() == []

    handler = llm_log.LLMLogHandler()
    _end(handler, _start(handler, [HumanMessage(content="hi")]))
    assert not (tmp_path / "llm.out").exists()


def test_llm_callbacks_returns_singleton(monkeypatch, tmp_path):
    """回调是同一个实例(否则每次构造模型都会新增一个 handler)。"""
    first = llm_log.llm_callbacks()
    assert first == llm_log.llm_callbacks()
    assert isinstance(first[0], llm_log.LLMLogHandler)

    # 路径按 config 解析:写出来落在 LLM_LOG_PATH 指定的地方
    _end(first[0], _start(first[0], [HumanMessage(content="hi")]))
    assert (tmp_path / "llm.out").exists()


def test_model_construction_attaches_handler(monkeypatch):
    """挂载点:模型构造时的 callbacks —— agent 工具循环里的调用靠它才记得到
    (create_agent 内部 ainvoke 不带 config,见 app/llm_log.py 的说明)。"""
    monkeypatch.setattr(config, "LLM_API_KEY", "sk-test")
    model = llm._build_openai()
    assert any(isinstance(h, llm_log.LLMLogHandler) for h in (model.callbacks or []))


def test_agent_run_is_logged_with_skill_attribution(_isolate):
    """端到端:create_agent 内部的模型调用不传 config,靠模型构造时的回调才记得到
    (langchain/agents/factory.py 的 `await model_.ainvoke(messages)`);
    skill 归属只能由调用方经 config.metadata 注入 —— agent 内部的 langgraph_node
    恒为 "model",分不出是哪个节点。这里用假模型离线锁住这两条。"""
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain.agents import create_agent

    handler = llm_log.LLMLogHandler(path=_isolate)
    model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")] * 4),
                                 callbacks=[handler])
    agent = create_agent(model=model, tools=[], system_prompt="你是估值分析师")

    asyncio.run(agent.ainvoke({"messages": [HumanMessage(content="hi")]},
                              config={"metadata": {"vm_skill": "04_ddm"}}))

    [rec] = _records(_isolate)
    assert rec["skill"] == "04_ddm"
    assert rec["node"] == "model"
    assert rec["request"]["messages"][0]["content"] == "你是估值分析师"
    assert rec["response"]["generations"][0]["content"] == "ok"


def test_concurrent_nodes_do_not_interleave(_isolate):
    """四个方法节点并行扇出:每条记录必须自成一块,不能被别的记录劈开。"""
    handler = llm_log.LLMLogHandler(path=_isolate)

    def work(n):
        for i in range(5):
            rid = _start(handler, [HumanMessage(content=f"输入 {n}-{i}")],
                         metadata={"vm_skill": f"skill{n}"})
            _end(handler, rid, content=f"输出 {n}-{i}")

    threads = [threading.Thread(target=work, args=(n,)) for n in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]

    recs = _records(_isolate)
    assert len(recs) == 20                                   # 每条记录一个独立文档
    for rec in recs:
        n = rec["skill"][-1]
        tail = rec["request"]["messages"][0]["content"].split()[-1]
        assert rec["response"]["generations"][0]["content"].endswith(tail)   # 输入输出配对
        assert tail.startswith(n)
