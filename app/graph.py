"""LangGraph 图组装。

拓扑:
  START → classify(01) → cost_of_equity(02) → dividends_growth(03)
        → [条件扇出 Send] → ddm(04) ┐
                           → fcfe(05) ┼→ synthesize(08) → report_writer → END
                           → excess(06)┘
                           → relative(07)┘

- 前三个节点串行:方法节点依赖上游产出的全局参数(COE、增长率、派息率)。
- 四个方法节点并行扇出:各自只需全局参数,互不依赖;Send 屏障保证 synthesize
  在全部扇出完成后只执行一次。
- 编排代码不决定何时调 MCP:方法节点完全由 LLM 自主决定数据获取,这里只调度 skill。
"""

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from app.nodes import (
    METHOD_SKILL,
    classify_node,
    cost_of_equity_node,
    ddm_node,
    dividends_growth_node,
    excess_returns_node,
    reg_capital_fcfe_node,
    relative_valuation_node,
    synthesize_node,
)
from app.state import ValuationState


def report_writer_node(state: ValuationState) -> dict:
    """纯 Python 节点(无 LLM):生成 markdown 报告并落盘。"""
    from app.report import write_report

    # 有任何 skill 报错即降级;状态须在生成报告前确定,否则报告里读到的是旧值
    status = "degraded" if state.get("errors") else "ok"
    try:
        path = write_report({**state, "status": status})
        return {"final_report_path": str(path), "status": status}
    except Exception as e:  # noqa: BLE001
        return {"status": "degraded", "errors": [f"报告生成失败:{e}"]}


# Send 的目标必须是图里的节点名(与方法名一致),不是 skill 目录名
def route_to_methods(state: ValuationState):
    """条件扇出:只对启用的方法发 Send(每个 Send 携带完整 state + method 标记)。

    启用的方法 = 分类结论 ∩ 有效方法名 − 用户 --skip 项。
    """
    skip = set(state.get("methods_skip") or [])
    enabled = [m for m in (state.get("methods_enabled") or [])
               if m in METHOD_SKILL and m not in skip]
    if not enabled:
        return [Send("synthesize", dict(state))]  # 无可用方法则直接综合
    return [Send(m, {**state, "method": m}) for m in enabled]


def build_graph():
    g = StateGraph(ValuationState)

    g.add_node("classify", classify_node)
    g.add_node("cost_of_equity", cost_of_equity_node)
    g.add_node("dividends_growth", dividends_growth_node)
    g.add_node("ddm", ddm_node)
    g.add_node("reg_capital_fcfe", reg_capital_fcfe_node)
    g.add_node("excess_returns", excess_returns_node)
    g.add_node("relative_valuation", relative_valuation_node)
    g.add_node("synthesize", synthesize_node)
    g.add_node("report_writer", report_writer_node)

    g.add_edge(START, "classify")
    g.add_edge("classify", "cost_of_equity")
    g.add_edge("cost_of_equity", "dividends_growth")

    method_nodes = ["ddm", "reg_capital_fcfe", "excess_returns", "relative_valuation"]
    g.add_conditional_edges("dividends_growth", route_to_methods,
                            method_nodes + ["synthesize"])
    for n in method_nodes:
        g.add_edge(n, "synthesize")
    g.add_edge("synthesize", "report_writer")
    g.add_edge("report_writer", END)

    return g.compile()
