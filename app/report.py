"""估值报告生成(纯 Python,无 LLM)。

所有数字都来自 state 中 Python 计算的结果,LLM 只贡献文字判断。
"""

from datetime import datetime
from pathlib import Path
from typing import Any

import app.config as config
import app.formulas as F

SKILL_ORDER = [
    ("classify", "公司分类"),
    ("cost_of_equity", "股权成本"),
    ("dividends_growth", "股息与增长"),
    ("ddm", "股息贴现模型(DDM)"),
    ("reg_capital_fcfe", "监管资本 FCFE"),
    ("excess_returns", "超额回报模型"),
    ("relative_valuation", "相对估值"),
    ("synthesize", "综合结论"),
]

METHOD_LABEL = {
    "ddm_per_share": "DDM(内在估值)",
    "fcfe_per_share": "监管资本 FCFE",
    "excess_per_share_perpetuity": "超额回报(永续,上界)",
    "excess_per_share_staged": "超额回报(均值回归,中央值)",
    "relative_pb_fair_price": "相对估值(PB 回归)",
}


def _fmt(v: Any, digits: int = 2) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        if abs(v) < 1 and v != 0:
            return f"{v:.4f}"
        return f"{v:,.{digits}f}"
    return str(v)


def _dict_table(d: dict, headers=("项目", "值")) -> list[str]:
    if not d:
        return ["(无)", ""]
    rows = [f"| {headers[0]} | {headers[1]} |", "|---|---|"]
    for k, v in d.items():
        if isinstance(v, (list, dict)) and not isinstance(v, str):
            v = f"`{str(v)[:120]}`"
        rows.append(f"| {k} | {_fmt(v)} |")
    return rows


def build_report(state: dict) -> str:
    skills = state.get("skills") or {}
    coef = state.get("coefficients") or {}
    val = state.get("valuation") or {}
    L: list[str] = []

    # ---- 标题 ----
    L.append(f"# {state.get('company_name') or state.get('code')} 估值报告")
    L.append("")
    L.append(f"- 交易所/代码:{state.get('exchange')} / {state.get('code')}")
    L.append(f"- 公司类型:{state.get('company_type', '未知')}")
    L.append(f"- 生成时间:{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    L.append(f"- 启用方法:{', '.join(state.get('methods_enabled') or [])}")
    L.append(f"- 运行状态:{state.get('status', 'unknown')}")
    L.append("")
    L.append("> 方法论来源:《估值》第九章 —— 反弹:对金融服务公司估值")
    L.append("> 所有公式由 Python 计算;参数判断由 LLM 依据 skills 分步分析产出。")
    L.append("")

    # ---- 估值结论(置顶)----
    synth = (skills.get("synthesize") or {}).get("params") or {}
    if synth:
        L.append("## 估值结论")
        L.append("")
        L.append(f"**每股估值区间:{_fmt(synth.get('value_range_low'))} ~ "
                 f"{_fmt(synth.get('value_range_high'))}**"
                 + (f",加权均值 **{_fmt(val.get('value_best'))}**(Python 计算)"
                    if val.get("value_best") is not None else ""))
        L.append("")
        L.append(f"- 判断:{synth.get('verdict', '—')}")
        L.append(f"- 置信度:{synth.get('conviction', '—')}")
        L.append("")
        synth_results = (skills.get("synthesize") or {}).get("results") or {}
        # 优先用归一后的权重:LLM 若用方法名当键,原样展示会与结果键对不上、显示为空
        weights = synth_results.get("weights_used") or synth.get("method_weights") or {}
        L.append("| 方法 | 每股价值 | 权重 | 隐含 PB |")
        L.append("|---|---|---|---|")
        implied = synth_results.get("implied_pb_by_method") or {}
        for key, label in METHOD_LABEL.items():
            if val.get(key) is None:
                continue
            w = weights.get(key)
            L.append(f"| {label} | {_fmt(val[key])} | "
                     f"{_fmt(w) if w is not None else '—'} | "
                     f"{_fmt(implied.get(key)) if implied.get(key) is not None else '—'} |")
        L.append("")

    # ---- 核心假设 ----
    L.append("## 核心假设")
    L.append("")
    L.append("| 参数 | 值 | 说明 |")
    L.append("|---|---|---|")
    L.append(f"| 股权成本(高增长期) | {F.pct(coef['coe_high'])} | Rf + β×ERP |" if coef.get("coe_high") is not None else "| 股权成本 | — | |")
    if coef.get("coe_terminal") is not None:
        L.append(f"| 股权成本(稳定期) | {F.pct(coef['coe_terminal'])} | β 趋 {_fmt(coef.get('terminal_beta'))} |")
    if coef.get("beta_effective") is not None:
        L.append(f"| β(有效) | {_fmt(coef['beta_effective'])} | β {_fmt(coef.get('beta'))} + 调整 "
                 f"{_fmt(coef.get('beta_risk_adjustment', 0.0))} |")
    if coef.get("rf") is not None:
        L.append(f"| 无风险利率 | {F.pct(coef['rf'])} | {coef.get('rf_tenor', '')} |")
    if coef.get("erp") is not None:
        erp_params = (skills.get("cost_of_equity") or {}).get("params") or {}
        L.append(f"| 股权风险溢价 | {F.pct(coef['erp'])} | {erp_params.get('erp_rationale', '')} |")
    if coef.get("roe_normalized") is not None:
        L.append(f"| 归一化 ROE | {F.pct(coef['roe_normalized'])} | 见股息与增长分析 |")
    if coef.get("growth") is not None:
        L.append(f"| 预期增长率 g | {F.pct(coef['growth'])} | ROE × (1 − 派息率) |")
    if coef.get("payout") is not None:
        payout_type = ((skills.get("dividends_growth") or {}).get("results") or {}).get("payout_type", "")
        L.append(f"| 派息率 | {F.pct(coef['payout'])} | {payout_type} |")
    if coef.get("shares_outstanding"):
        L.append(f"| 总股本 | {_fmt(coef['shares_outstanding'], 0)} | 用于每股换算 |")
    L.append("")

    # ---- 各 skill 明细 ----
    L.append("## 分析明细")
    L.append("")
    for key, label in SKILL_ORDER:
        s = skills.get(key)
        if not s:
            continue
        L.append(f"### {label}")
        L.append("")
        if s.get("skipped"):
            L.append(f"*已跳过:{s.get('skip_reason', '不适用')}*")
            L.append("")
            continue
        if s.get("error"):
            L.append(f"**错误:{s['error']}**")
            L.append("")
            continue
        if s.get("analysis"):
            L.append(str(s["analysis"]))
            L.append("")
        results = s.get("results") or {}
        if results:
            L.append("**Python 计算结果**")
            L.append("")
            for k, v in results.items():
                if k in ("sensitivity", "dividends", "excess_by_year", "bv_by_year",
                         "fcfe", "equity_path", "capital_ratio_path"):
                    continue
                L.append(f"- {k}: {_fmt(v)}")
            L.append("")
            if results.get("formula"):
                L.append(f"`{results['formula']}`")
                L.append("")
        grid = results.get("sensitivity")
        if grid:
            axes = [k for k in grid[0] if k != "param"]
            L.append("敏感性分析:")
            L.append("")
            L.append("| g \\ COE | " + " | ".join(axes) + " |")
            L.append("|---|" + "---|" * len(axes))
            for row in grid:
                L.append(f"| {row['param']} | " + " | ".join(_fmt(row.get(a)) for a in axes) + " |")
            L.append("")

    # ---- 三个估值驱动器 ----
    drivers = synth.get("drivers") or {}
    if drivers:
        L.append("## 三个估值驱动器")
        L.append("")
        for k, label in (("equity_risk", "① 股权风险"), ("growth_quality", "② 增长质量"),
                         ("capital_buffer", "③ 监管缓冲区")):
            if drivers.get(k):
                L.append(f"**{label}**")
                L.append("")
                L.append(str(drivers[k]))
                L.append("")
    notes = synth.get("investment_notes") or {}
    if notes:
        L.append("## 投资技巧评估")
        L.append("")
        for k, label in (("capital_buffer", "资本缓冲区"), ("operating_risk", "营运风险"),
                         ("transparency", "透明度"), ("entry_barriers", "进入壁垒")):
            if notes.get(k):
                L.append(f"- **{label}**:{notes[k]}")
        L.append("")
    if synth.get("key_risks"):
        L.append("## 主要风险")
        L.append("")
        for r in synth["key_risks"]:
            L.append(f"- {r}")
        L.append("")

    # ---- 数据质量与警示 ----
    warns = state.get("warnings") or []
    errs = state.get("errors") or []
    if warns or errs:
        L.append("## 数据质量与假设警示")
        L.append("")
        for w in warns:
            L.append(f"- ⚠️ {w}")
        for e in errs:
            L.append(f"- ❌ {e}")
        L.append("")

    cache = state.get("data_cache") or {}
    if cache:
        L.append("## 数据来源")
        L.append("")
        L.append(f"本次分析经 MCP 调用了 {len(cache)} 次数据接口:")
        L.append("")
        for k in sorted(cache):
            L.append(f"- `{k[:160]}`")
        L.append("")

    L.append("---")
    L.append("")
    L.append(f"*{synth.get('disclaimer', '本报告为方法演练输出,不构成投资建议。')}*")
    L.append("")
    return "\n".join(L)


def write_report(state: dict) -> Path:
    out_dir = Path(config.REPORT_DIR) if config.REPORT_DIR else Path(__file__).resolve().parent.parent / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"{state.get('exchange')}_{state.get('code')}_{ts}.md"
    path.write_text(build_report(state), encoding="utf-8")
    return path
