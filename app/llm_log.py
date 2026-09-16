"""LLM 全量对话日志:每次模型往返的原始输入输出落一份到 llm.out。

为什么挂在模型实例的 callbacks 上,而不是各调用点:
    create_agent 内部是 `await model_.ainvoke(messages)`(不带 callbacks 参数),
    但 ChatModel 每次调用都会 `CallbackManager.configure(config.callbacks, self.callbacks, ...)`
    —— 只有构造时挂上的回调,才能覆盖 agent 工具循环里的每一次调用、结构化那一步
    和启动自检的 ping,且今后新增调用点不需要记得补日志。

**格式:YAML 多文档流**,一次模型往返 = 一条记录 = 一个 `---` 文档(不是 JSON:JSON 的
字符串里放不下字面换行,正文的 `\\n` 只能以转义形式出现,读起来是一整行;YAML 的
字面块 `|` 直接把换行写进文件,`yaml.safe_load_all(llm.out)` 又能原样读回)。
`width` 设为不折行,所以文件里出现的每一个换行都来自内容本身,不是排版。
日志不做任何截断:skill 提示词全文、全部消息(含 tool_calls / invalid_tool_calls)、
响应正文、usage 原样保留。

整条记录在"调用结束时"一次性写盘(并在写锁内 append),所以并发扇出的四个方法节点
各成一条 `---` 文档、不会互相交错(代价是进程被 Ctrl-C 打断时,那次调用的输入只在
内存里 —— 超时/报错这类失败会走 on_llm_error 正常落盘)。报错的那次对话同样落盘,
它往往正是要查的。

日志自己绝不能打断估值:写入全部吞异常,只在 value_mind.llm_log 上警告。
"""

import json
import logging
import os
import threading
import time
from datetime import datetime
from itertools import count
from pathlib import Path
from typing import Any

import yaml
from langchain_core.callbacks import BaseCallbackHandler

import app.config as config

logger = logging.getLogger("value_mind.llm_log")

_MAX_PENDING = 512         # 未配对的调用上限(父任务被取消时可能收不到 end)
_SECRET_HINTS = ("api_key", "apikey", "api-key", "auth", "secret", "password",
                 "bearer", "credential", "token")
# serialized 里只留这些构造参数(整体落盘既大又不稳,还夹着凭证)
_KEEP_INVOCATION = ("model", "model_name", "temperature", "max_tokens", "stop", "tool_choice")

_HANDLER: "LLMLogHandler | None" = None
_HANDLER_LOCK = threading.Lock()
_WRITE_LOCK = threading.Lock()   # 串行化 append:一条记录被写花,整个 YAML 流都会解析不了


# ---------------------------------------------------------------------------
# 序列化:YAML 多文档流(多行正文用字面块,文件里就是真换行)
# ---------------------------------------------------------------------------

class _Dumper(yaml.SafeDumper):
    """SafeDumper + 多行字符串强制字面块。

    默认风格下 PyYAML 对某些内容(带尾随空格、以空格开头等)会退回带引号并转义
    `\\n`,读起来又变成一整行——那些形状本来也不允许块标量,退回是对的,不值得为它
    牺牲可读性。这里只保证常见的中文散文正文是真换行。
    """

    def ignore_aliases(self, data: Any) -> bool:
        """不写锚点/别名。usage 与 generations[].usage_metadata 是同一个对象,
        默认会被写成 `usage: *id001`,读到那一行还得往上翻;日志里每个键都该写全。
        """
        return True


def _represent_str(dumper: yaml.SafeDumper, data: str):
    return dumper.represent_scalar("tag:yaml.org,2002:str", data,
                                   style="|" if "\n" in data else None)


_Dumper.add_representer(str, _represent_str)


def _dump(record: dict) -> str:
    """一条记录 → 一个 YAML 文档。width 不折行:文件里的换行只来自内容。"""
    return yaml.dump(record, Dumper=_Dumper, allow_unicode=True, sort_keys=False,
                     default_flow_style=False, width=1 << 30, explicit_start=True)


def log_path() -> Path:
    """llm.out 路径:LLM_LOG_PATH 优先,默认 <value_mind>/llm.out(与 report.py 同款口径)。"""
    if config.LLM_LOG_PATH:
        return Path(config.LLM_LOG_PATH)
    return Path(__file__).resolve().parent.parent / "llm.out"


def llm_callbacks() -> list[BaseCallbackHandler]:
    """给 ChatOpenAI / ChatAnthropic 的 callbacks 参数用;关闭时返回空表。"""
    if not config.LLM_LOG:
        return []
    global _HANDLER
    with _HANDLER_LOCK:
        if _HANDLER is None:
            _HANDLER = LLMLogHandler()
        return [_HANDLER]


# ---------------------------------------------------------------------------
# 脱敏 / 序列化helper
# ---------------------------------------------------------------------------

def _is_secret_key(key: str) -> bool:
    """凭证类字段名。tokens(复数)是计数不是凭证,max_tokens 不能被误伤。"""
    k = key.lower()
    if "token" in k and "tokens" in k:
        return False
    return any(h in k for h in _SECRET_HINTS)


def _plain(value: Any) -> Any:
    """转成纯 JSON 结构:pydantic/SDK 对象走 default=str,原始字段尽量保留。"""
    if value is None:
        return None
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, default=str))
    except Exception:  # noqa: BLE001 —— 序列化失败也要留下点什么
        return str(value)


def _no_secrets(d: dict) -> dict:
    return {k: v for k, v in d.items() if not _is_secret_key(k)}


def _message_dict(m: Any) -> dict:
    """消息 → 普通 dict(字段尽量原样保留,便于事后复盘)。"""
    if isinstance(m, dict):
        return _plain(m)
    d: dict[str, Any] = {"role": getattr(m, "type", type(m).__name__)}
    for k in ("content", "name", "id", "tool_call_id", "status"):
        v = getattr(m, k, None)
        if v not in (None, ""):
            d[k] = _plain(v)
    for k in ("tool_calls", "invalid_tool_calls", "additional_kwargs",
              "response_metadata", "usage_metadata"):
        v = getattr(m, k, None)
        if v:
            d[k] = _plain(v)
    return d


def _meta(metadata: dict | None) -> dict:
    """原样留 metadata(agent 内部节点名、模型名等都在里面),只去掉凭证字段。"""
    return _plain(_no_secrets(dict(metadata or {})))


def _invocation(serialized: Any, kwargs: dict) -> dict:
    """模型构造参数:只留白名单键。serialized 整体落盘既大又含 api_key。"""
    out: dict[str, Any] = {}
    skwargs = (serialized or {}).get("kwargs") if isinstance(serialized, dict) else None
    iparams = kwargs.get("invocation_params") or {}
    for key in _KEEP_INVOCATION:
        if isinstance(skwargs, dict) and key in skwargs:
            out[key] = skwargs[key]
        if key in iparams:
            out[key] = iparams[key]
    if isinstance(serialized, dict) and serialized.get("id"):
        out["id"] = serialized["id"]
    return _no_secrets(out)


def _usage(generations: list[dict], llm_output: dict | None) -> dict | None:
    """token 用量:消息级 usage_metadata → llm_output.token_usage → response_metadata.usage。"""
    for g in generations:
        if g.get("usage_metadata"):
            return g["usage_metadata"]
    for g in generations:
        usage = (g.get("response_metadata") or {}).get("usage")
        if usage:
            return usage
    if llm_output:
        for key in ("token_usage", "usage"):
            if llm_output.get(key):
                return llm_output[key]
    return None


def _response_parts(response: Any) -> tuple[list[dict], dict | None]:
    """LLMResult → (每条 generation 的消息 dict, llm_output)。"""
    generations: list[dict] = []
    if response is None:
        return generations, None
    for batch in getattr(response, "generations", None) or []:
        for gen in batch or []:
            msg = getattr(gen, "message", None)
            if msg is not None:
                d = _message_dict(msg)
            else:
                d = {"role": "text", "content": getattr(gen, "text", "")}
            info = _plain(getattr(gen, "generation_info", None))
            if info:
                d["generation_info"] = info
            generations.append(d)
    return generations, _plain(getattr(response, "llm_output", None))


# ---------------------------------------------------------------------------
# 回调处理器
# ---------------------------------------------------------------------------

class LLMLogHandler(BaseCallbackHandler):
    """一次模型往返写一块记录。

    run_inline=True:回调在当前事件循环线程里直接执行,不丢线程池 ——
    主路径(agent 的 ainvoke)与自检的同步 invoke 共用同一份同步实现。
    也不要实现 tap_output_iter/atap 之类:那会让 langchain 判定为流式回调,
    on_llm_end 收到的就不是完整响应了。
    """

    run_inline = True

    def __init__(self, path: Path | None = None):
        super().__init__()
        self._path = path            # 显式指定(测试用);None = 每次写盘时按 config 解析
        self._lock = threading.Lock()
        self._pending: dict[str, dict] = {}
        self._seq = count(1)
        self._session_open = False
        self._warned = False

    # ---- 回调入口 ----

    def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None,
                            tags=None, metadata=None, **kwargs) -> None:
        try:
            batch = messages[0] if messages else []
            entry = {
                "run_id": str(run_id),
                "parent_run_id": str(parent_run_id) if parent_run_id else None,
                "started": datetime.now(),
                "t0": time.monotonic(),
                # 必须当场深拷贝成普通 dict:传给模型的就是这批消息对象,
                # 后续 _format_for_tracing 会就地改写 content 块,留引用到 end 会读到改过的内容。
                "request": [_message_dict(m) for m in batch],
                "meta": _meta(metadata),
                "invocation": _invocation(serialized, kwargs),
                "tags": list(tags or []),
            }
            with self._lock:
                if len(self._pending) >= _MAX_PENDING:
                    self._pending.pop(next(iter(self._pending)), None)
                    self._warn("未配对的 LLM 调用超过 %d 条,丢弃最早一条", _MAX_PENDING)
                self._pending[str(run_id)] = entry
        except Exception as e:  # noqa: BLE001
            self._warn("on_chat_model_start 记录失败: %r", e)

    def on_llm_end(self, response, *, run_id, parent_run_id=None, tags=None, **kwargs) -> None:
        self._finish(str(run_id), response=response)

    def on_llm_error(self, error, *, run_id, parent_run_id=None, tags=None, **kwargs) -> None:
        self._finish(str(run_id), error=error)

    # ---- 落盘 ----

    def _finish(self, run_id: str, response=None, error: BaseException | None = None) -> None:
        try:
            with self._lock:
                entry = self._pending.pop(run_id, None)
                if entry is None:
                    # 没配对上(理论上不会):仍然把这一半记下来,别丢信息
                    entry = {"run_id": run_id, "parent_run_id": None,
                             "started": datetime.now(), "t0": time.monotonic(),
                             "request": [], "meta": {}, "invocation": {}, "tags": []}
                seq = next(self._seq)
            generations, llm_output = _response_parts(response)
            usage = _usage(generations, llm_output)
            record = {
                "seq": seq,
                "ts": entry["started"].strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                "event": "error" if error is not None else "end",
                "run_id": entry["run_id"],
                "parent_run_id": entry["parent_run_id"],
                # vm_skill 由 nodes.run_skill_agent 经 config.metadata 注入;
                # langgraph_node 是 agent 内部节点名(恒为 "model"),只作参考。
                "skill": entry["meta"].get("vm_skill"),
                "node": entry["meta"].get("langgraph_node"),
                "model": entry["meta"].get("ls_model_name") or entry["invocation"].get("model"),
                "provider": entry["meta"].get("ls_provider"),
                "elapsed_ms": round((time.monotonic() - entry["t0"]) * 1000),
                "tags": entry["tags"],
                "invocation": entry["invocation"],
                "metadata": entry["meta"],
                "request": {"messages": entry["request"]},
                "response": None if response is None else
                            {"generations": generations, "llm_output": llm_output},
                "usage": usage,
                "error": None if error is None else f"{type(error).__name__}: {error}",
            }
            self._write(_dump(record))
        except Exception as e:  # noqa: BLE001
            self._warn("LLM 日志写入失败: %r", e)

    def _write(self, text: str) -> None:
        if not config.LLM_LOG:      # 写盘时再看一次,便于运行时/测试里关掉
            return
        path = self._path or log_path()
        with _WRITE_LOCK:
            with self._lock:
                if not self._session_open:
                    self._session_open = True
                    text = (f"# value_mind llm.out 会话开始 {datetime.now():%Y-%m-%d %H:%M:%S} "
                            f"pid={os.getpid()} model={config.LLM_MODEL} "
                            f"provider={config.LLM_PROVIDER}\n"
                            f"# 格式:YAML 多文档流,一条记录一个 --- 文档;"
                            f"读取见 README「LLM 对话日志」\n") + text
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as f:
                    f.write(text)
                    f.flush()

    def _warn(self, msg: str, *args) -> None:
        if not self._warned:
            self._warned = True
            logger.warning(msg, *args)
