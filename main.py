#!/usr/bin/env python3
"""金融服务公司估值 agent 的 CLI 入口。

用法:
  python main.py SHA 600036                     # 完整估值
  python main.py NYSE WFC --period 5y --verbose
  python main.py SHA 600036 --beta 0.9 --rf 0.02   # 覆盖假设参数
  python main.py SHA 600036 --only cost_of_equity  # 单 skill 调试
  python main.py --check-llm                       # 只做连通性自检
"""

import argparse
import json
import logging
import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import app.config as config  # noqa: E402
from mcp_client import init_mcp_client, shutdown_mcp_client  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("value_mind.main")

OVERRIDE_MAP = {
    "beta": "beta", "rf": "rf", "erp": "erp", "roe": "roe_normalized",
    "growth": "g_high", "payout": "payout_ratio", "eps": "eps0",
}


def parse_args():
    p = argparse.ArgumentParser(description="金融服务公司估值 agent(第九章方法论)")
    p.add_argument("exchange", nargs="?", help="交易所代码:SHA / SHE / HKG / NYSE / NASDAQ / ASX")
    p.add_argument("code", nargs="?", help="股票代码,如 600036 / WFC / 0700")
    p.add_argument("--period", default="5y", choices=["1y", "2y", "5y", "all"],
                   help="建议 LLM 抓取的数据窗口(注入上下文,不强制)")
    p.add_argument("--beta", type=float, help="覆盖 β")
    p.add_argument("--rf", type=float, help="覆盖无风险利率(小数,如 0.02)")
    p.add_argument("--erp", type=float, help="覆盖股权风险溢价(小数,如 0.06)")
    p.add_argument("--roe", type=float, help="覆盖归一化 ROE(小数)")
    p.add_argument("--growth", type=float, help="覆盖高增长期增长率(小数)")
    p.add_argument("--payout", type=float, help="覆盖派息率(小数)")
    p.add_argument("--eps", type=float, help="覆盖 EPS0")
    p.add_argument("--skip", action="append", default=[],
                   choices=["ddm", "reg_capital_fcfe", "excess_returns", "relative_valuation"],
                   help="强制跳过某方法(可重复)")
    p.add_argument("--only", help="只运行单个 skill(调试用),如 cost_of_equity")
    p.add_argument("--check-llm", action="store_true", help="只做 LLM 连通性自检后退出")
    p.add_argument("--out", help="报告输出目录(默认 value_mind/reports)")
    p.add_argument("--verbose", action="store_true", help="打印每个节点的中间结果")
    return p.parse_args()


async def main() -> int:
    args = parse_args()

    if args.out:
        config.REPORT_DIR = args.out

    # ---- LLM 连通性自检 ----
    from app.llm import check_llm

    try:
        _, desc = check_llm()
        print(f"✅ LLM 连通:{desc}(模型 {config.LLM_MODEL})")
    except RuntimeError as e:
        print(f"❌ {e}", file=sys.stderr)
        return 2

    if args.check_llm:
        return 0

    if not args.exchange or not args.code:
        print("❌ 需要指定 exchange 和 code(或用 --check-llm 只做自检)", file=sys.stderr)
        return 2

    # ---- MCP ----
    mcp_ok = init_mcp_client(config.MCP_BASE_URL)
    if mcp_ok:
        print(f"✅ MCP 已连接:{config.MCP_BASE_URL}")
    else:
        print(f"⚠️  MCP 不可达({config.MCP_BASE_URL}),将以假设模式运行(结论降级)", file=sys.stderr)

    overrides = {v: getattr(args, v) for k, v in OVERRIDE_MAP.items()
                 if getattr(args, k, None) is not None}

    state = {
        "exchange": args.exchange.upper(),
        "code": args.code,
        "overrides": overrides,
        "period_hint": args.period,
        "methods_skip": args.skip,
        "skills": {}, "coefficients": {}, "data_cache": {}, "valuation": {},
        "warnings": [], "errors": [],
    }

    try:
        if args.only:
            return run_single_skill(args, state)

        from app.graph import build_graph

        graph = build_graph()
        print(f"\n▶ 开始估值:{state['exchange']} {state['code']}"
              f"{'(假设覆盖:' + str(overrides) + ')' if overrides else ''}\n")
        final = await graph.ainvoke(state, config={"recursion_limit": 60})

        print_summary(final)
        path = final.get("final_report_path")
        if path:
            print(f"\n📄 报告已保存:{path}")
        return 1 if final.get("errors") else 0
    finally:
        shutdown_mcp_client()


def run_single_skill(args, state: dict) -> int:
    """单 skill 调试:直接跑一个节点,打印参数与计算结果。"""
    import app.nodes as nodes
    from app.schemas import (
        ClassifyParams, CostOfEquityParams, DdmParams, DividendsGrowthParams,
        ExcessParams, FcfeParams, RelativeParams, SynthesisParams,
    )

    mapping = {
        "01_company_classifier": (nodes.classify_node, ClassifyParams),
        "02_cost_of_equity": (nodes.cost_of_equity_node, CostOfEquityParams),
        "03_dividends_growth": (nodes.dividends_growth_node, DividendsGrowthParams),
        "04_ddm": (nodes.ddm_node, DdmParams),
        "05_reg_capital_fcfe": (nodes.reg_capital_fcfe_node, FcfeParams),
        "06_excess_returns": (nodes.excess_returns_node, ExcessParams),
        "07_relative_valuation": (nodes.relative_valuation_node, RelativeParams),
        "08_synthesize": (nodes.synthesize_node, SynthesisParams),
    }
    key = args.only if args.only in mapping else next(
        (k for k in mapping if args.only in k), None)
    if key is None:
        print(f"❌ 未知 skill:{args.only};可选:{', '.join(mapping)}", file=sys.stderr)
        return 2

    fn, _ = mapping[key]
    print(f"▶ 单 skill 运行:{key}\n")
    out = fn(state)
    merged = {**state, **{k: v for k, v in out.items() if k not in ("skills",)}}
    merged["skills"] = {**(state.get("skills") or {}), **(out.get("skills") or {})}
    print(json.dumps(merged.get("skills"), ensure_ascii=False, indent=2, default=str))
    if out.get("coefficients"):
        print("\ncoefficients:", json.dumps(out["coefficients"], ensure_ascii=False, indent=2, default=str))
    if out.get("valuation"):
        print("\nvaluation:", json.dumps(out["valuation"], ensure_ascii=False, indent=2, default=str))
    if out.get("errors"):
        print("\nerrors:", out["errors"], file=sys.stderr)
    if out.get("warnings"):
        print("\nwarnings:", out["warnings"])
    return 1 if out.get("errors") else 0


def print_summary(final: dict) -> None:
    val = final.get("valuation") or {}
    print("\n" + "=" * 60)
    print("估值结果摘要")
    print("=" * 60)
    for k, v in val.items():
        print(f"  {k}: {v}")
    if final.get("warnings"):
        print("\n⚠️  警示:")
        for w in final["warnings"]:
            print(f"  - {w}")
    if final.get("errors"):
        print("\n❌ 错误:")
        for e in final["errors"]:
            print(f"  - {e}")


if __name__ == "__main__":
    asyncio.run(main())
