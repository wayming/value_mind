"""LLM 工厂与启动连通性自检。

两条候选连接路径(见 app/config.py):
- OpenAI 兼容端点(langchain-openai ChatOpenAI)
- Anthropic 兼容端点(langchain-anthropic ChatAnthropic,当前环境实测可用:
  ANTHROPIC_AUTH_TOKEN + api.deepseek.com/anthropic 网关,即宿主 Claude Code 所用网关)

check_llm() 按 LLM_PROVIDER 探测候选配置,成功即返回 (模型, 配置描述);
全部失败抛 RuntimeError 并给出诊断提示。
"""

import logging

import app.config as config

logger = logging.getLogger("value_mind.llm")

_MODEL = None       # 启动自检成功后注入,各 skill 节点共用
_TRANSPORT = None   # "openai" | "anthropic",决定结构化输出怎么走


def set_model(model, transport: str = "") -> None:
    global _MODEL, _TRANSPORT
    _MODEL = model
    if transport:
        _TRANSPORT = transport


def get_model():
    if _MODEL is None:
        raise RuntimeError("LLM 未初始化:请先调用 check_llm()")
    return _MODEL


def get_transport() -> str:
    return _TRANSPORT or "openai"


def _thinking_disabled_kwargs(transport: str) -> dict:
    """关掉 thinking 的通道相关写法(实测:开着 thinking 会拒绝强制 tool_choice)。"""
    if not config.LLM_DISABLE_THINKING:
        return {}
    if transport == "anthropic":
        return {"thinking": {"type": "disabled"}}
    # OpenAI 兼容端点没有顶层 thinking 参数,按该网关的约定放 extra_body
    return {"extra_body": {"thinking": {"type": "disabled"}}}


def _build_openai():
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=config.LLM_MODEL,
        base_url=config.LLM_BASE_URL,
        api_key=config.LLM_API_KEY,
        temperature=config.LLM_TEMPERATURE,
        timeout=config.LLM_TIMEOUT,
        max_retries=2,
        **_thinking_disabled_kwargs("openai"),
    )


def _build_anthropic():
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(
        model=config.LLM_MODEL,
        base_url=config.ANTHROPIC_LLM_BASE_URL,
        api_key=config.ANTHROPIC_LLM_API_KEY,
        temperature=config.LLM_TEMPERATURE,
        timeout=config.LLM_TIMEOUT,
        max_retries=2,
        **_thinking_disabled_kwargs("anthropic"),
    )


_DIAGNOSTIC = """诊断与修复:
- 指定 LLM_PROVIDER=openai,配置 LLM_BASE_URL / LLM_API_KEY(OpenAI 兼容端点);
- 或 LLM_PROVIDER=anthropic,配置 ANTHROPIC_LLM_BASE_URL / ANTHROPIC_LLM_API_KEY
  (Anthropic 兼容端点,本机 ANTHROPIC_AUTH_TOKEN 走 api.deepseek.com/anthropic 网关已验证可用);
- 模型名用 LLM_MODEL 指定(默认 deepseek-flash)。
- 注意:本机 DEEPSEEK_API_KEY 对 api.deepseek.com 的 OpenAI 端点实测认证失败,
  请用 --check-llm 确认实际生效的配置。"""


def check_llm():
    """按序探测候选配置,返回 (模型, 配置描述);全部失败抛 RuntimeError。"""
    candidates: list[tuple[str, object, str]] = []
    if config.LLM_PROVIDER in ("auto", "openai") and config.LLM_API_KEY:
        candidates.append(("openai 兼容端点", _build_openai, "openai"))
    if config.LLM_PROVIDER in ("auto", "anthropic") and config.ANTHROPIC_LLM_API_KEY:
        candidates.append(("anthropic 兼容端点", _build_anthropic, "anthropic"))
    if not candidates:
        raise RuntimeError(
            "未找到可用的 LLM 凭证:请设置 LLM_API_KEY 或 ANTHROPIC_LLM_API_KEY。\n"
            + _DIAGNOSTIC
        )

    errors = []
    for desc, build, transport in candidates:
        model = build()
        try:
            model.invoke([("user", "ping")], max_tokens=1)
            set_model(model, transport)
            logger.info("LLM 连通:%s", desc)
            return model, desc
        except Exception as e:  # noqa: BLE001 —— 自检要吞掉所有异常以便报告
            errors.append(f"  {desc}: {type(e).__name__}: {str(e)[:300]}")

    raise RuntimeError(
        "LLM 连通性自检失败,尝试过的配置:\n"
        + "\n".join(errors)
        + "\n"
        + _DIAGNOSTIC
    )


# 结构化输出的自动降级顺序:实测该网关
#   json_schema   → 400 "This response_format type is unavailable now"(OpenAI 端点不支持)
#   function_calling → 400 "Thinking mode does not support this tool_choice"(须先关 thinking)
#   json_mode     → 只要求提示词里出现 "json",不涉及 tool_choice,作为兜底
_DEFAULT_METHODS = {
    "openai": ["function_calling", "json_mode"],
    "anthropic": ["function_calling", "json_schema"],
}


def structured_model(schema):
    """返回能把对话收敛成 `schema` 的 runnable,自动跳过该端点不支持的方式。

    为什么不能直接用 create_react_agent 的 response_format:它内部固定调用
    model.with_structured_output(schema) 不带 method,langchain-openai 会默认选
    json_schema —— 在本网关上是 400。所以结构化那一步由这里自己发。
    """
    if config.LLM_STRUCTURED_METHOD:
        methods = [config.LLM_STRUCTURED_METHOD]
    else:
        methods = _DEFAULT_METHODS.get(get_transport(), ["function_calling"])
    return _FallbackStructured(get_model(), schema, methods)


class _FallbackStructured:
    """按 methods 顺序尝试 with_structured_output,失败换下一种。

    400 之类的协议级错误在这里是"这种 method 不能用",不是"这次调用失败",
    所以换一种重试;其余异常照常抛出,不掩盖真实问题。
    """

    def __init__(self, model, schema, methods: list[str]):
        self._model, self._schema, self._methods = model, schema, methods

    def invoke(self, messages):
        last: Exception | None = None
        for m in self._methods:
            try:
                return self._model.with_structured_output(self._schema, method=m).invoke(messages)
            except Exception as e:  # noqa: BLE001
                if not _is_protocol_error(e):
                    raise
                logger.warning("结构化输出 method=%s 不可用,尝试下一种:%s", m, str(e)[:160])
                last = e
        raise RuntimeError(f"结构化输出在所有方式下均失败:{last}") from last


def _is_protocol_error(e: Exception) -> bool:
    status = getattr(e, "status_code", None) or getattr(
        getattr(e, "response", None), "status_code", None
    )
    return status == 400
