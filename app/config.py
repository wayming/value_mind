"""全局配置:全部走环境变量,带默认值。

LLM 连接(两条候选路径,启动自检按序探测,见 app/llm.py):
  - OpenAI 兼容端点(langchain-openai):LLM_BASE_URL / LLM_API_KEY / LLM_MODEL
    默认 base_url=https://api.deepseek.com/v1,key 取 DEEPSEEK_API_KEY。
  - Anthropic 兼容端点(langchain-anthropic):ANTHROPIC_LLM_BASE_URL / ANTHROPIC_LLM_API_KEY
    默认复用宿主环境的 ANTHROPIC_BASE_URL / ANTHROPIC_AUTH_TOKEN。
  LLM_PROVIDER: auto | openai | anthropic,auto 时按上述顺序探测。
"""

import os


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


# ---- LLM ----
LLM_PROVIDER = _env("LLM_PROVIDER", "auto")
LLM_MODEL = _env("LLM_MODEL", "deepseek-flash")
LLM_BASE_URL = _env("LLM_BASE_URL", "https://api.deepseek.com")
LLM_API_KEY = _env("LLM_API_KEY", os.environ.get("DEEPSEEK_API_KEY", ""))
ANTHROPIC_LLM_BASE_URL = _env(
    "ANTHROPIC_LLM_BASE_URL",
    os.environ.get("ANTHROPIC_BASE_URL", "https://api.deepseek.com/anthropic"),
)
ANTHROPIC_LLM_API_KEY = _env(
    "ANTHROPIC_LLM_API_KEY", os.environ.get("ANTHROPIC_AUTH_TOKEN", "")
)

# ---- MCP ----
MCP_BASE_URL = _env("MCP_BASE_URL", "http://localhost:8081")

# ---- 行为 ----
# 该网关默认开启 thinking 模式,而 thinking 模式拒绝强制 tool_choice,导致结构化输出
# (以及绑定工具)失败;两条通道都要关掉它:Anthropic 走 thinking={"type":"disabled"},
# OpenAI 兼容端点走 extra_body={"thinking":{"type":"disabled"}}。实测关闭后均正常。
# 如你的网关不支持该参数,设为 0 关闭。
LLM_DISABLE_THINKING = _env("LLM_DISABLE_THINKING", "1") == "1"
# 结构化输出的实现方式,留空 = 按通道自动选(openai → function_calling,anthropic → 默认)。
# 可选:function_calling / json_schema / json_mode。自动模式下一次调用失败会降级重试。
LLM_STRUCTURED_METHOD = _env("LLM_STRUCTURED_METHOD", "")
# ---- LLM 对话日志 ----
# 每次模型往返的原始输入输出落一份到 llm.out(文本块 + 末行 @@JSON@@ 机读 JSON)。
# 内容不截断(含 skill 提示词全文与工具返回),文件会很大:估值一次几十 MB 量级。
LLM_LOG = _env("LLM_LOG", "1") == "1"                  # 0 = 关闭
LLM_LOG_PATH = _env("LLM_LOG_PATH", "")                # 空 = <value_mind>/llm.out

MAX_LLM_STEPS = int(_env("MAX_LLM_STEPS", "14"))       # 单节点内 LLM 工具调用步数上限
# 单节点内**去重后**的 get_financials 次数上限(mcp_tools 的账本强制执行)。实测 skill 03
# 会拿同一个指标清单换着 period 连抓 13 次(上下文涨到 31k token),提示词拦不住;
# 2 = 一次取全 + 一次补漏,再多的调用只会重复同一份数据。
MAX_FINANCIALS_CALLS = int(_env("MAX_FINANCIALS_CALLS", "2"))
LLM_TEMPERATURE = float(_env("LLM_TEMPERATURE", "0"))  # 估值分析用 0,保证可复现
LLM_TIMEOUT = float(_env("LLM_TIMEOUT", "120"))        # 秒
PARALLEL_TOOL_CALLS = _env("LLM_PARALLEL_TOOL_CALLS", "0") == "1"
REPORT_DIR = _env("REPORT_DIR", "")                    # 空 = <value_mind>/reports
