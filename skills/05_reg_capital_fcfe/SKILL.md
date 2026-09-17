---
name: financial-services-regulatory-capital-fcfe
description: 监管资本约束下的股权自由现金流(FCFE)模型:再投资=目标资本比率×新资产−现有股权,FCFE 替代股息折现。用于银行/保险公司的内在估值,特别关注资本缓冲区。
---

# 监管资本 FCFE 模型

## 目标

金融服务公司的净资本支出与营运资本无法定义,但**换一种方式定义再投资**后可以估算股权现金流:
在金融服务公司中,再投资是投入监管资本的钱——监管资本约束决定了未来增长的极限。
FCFE 可视为"潜在股息",替代 DDM 中的股息。

## 何时使用 / 不适用

- **适用**:银行、保险(受监管资本约束强)。
- **不适用**:投行/投资公司资本约束弱(除非有类似约束)→ fcfe_applicable=false + skip_reason。

## 数据获取(MCP 工具)

1. `list_metrics` 确认指标,重点:assets(总资产)、equity(股权账面价值)、netinc/netinccmn、grossLoans。
2. `get_financials` 取最近 2 年数据作为基数。注意:MCP 没有 CET1 等监管资本比率指标,
   用 equity/assets 近似资本比率,并明确标注为近似。

## 分析步骤

1. **确定基数**:assets(最新总资产)、equity_current(当前股权账面价值)、net_income(最近年度净收益)。
2. **目标资本比率(target_capital_ratio)**:受监管要求影响大,也反映管理层选择——
   保守银行保持高于监管要求的比率,进取的银行贴近下限。参考:巴塞尔 III 核心一级资本要求
   4.5%+缓冲,总资本 8% 以上;中国系统重要性银行要求更高。给出你的判断值。
3. **资产增速(assets_growth)**:公司贷款/资产扩张计划与历史趋势,保守取近 3–5 年平均。
4. **缓冲区判断(驱动器#3,关键)**:比较当前资本比率、目标比率与监管要求:
   - 缓冲区不足的银行 → 未来需要更多再投资把资本比率拉回目标 → FCFE 更小 → **价值更低**。
   - 缓冲区充足的银行 → 再投资压力小,能维持高派息 → 价值更高。
   这个判断写入 buffer_gap_note。
5. **净收益增速(ni_growth)**:与资产增速、ROE 保持自洽(≈ ROE × (1−派息率) 量级)。
6. **年限与终值**:projection_years 高增长期年限(如 5),g_terminal 永续增长率。

## 坑与注意点

- **再投资公式的直觉**:再投资 = 目标资本比率 × 新资产 − 现有股权。资产扩张越快、
  目标比率越高、现有股权越薄 → 再投资越大 → FCFE 越小。
- **书中例子(验算参考)**:1 亿贷款、增 10%、目标资本比率 7%、现有股权 600 万 →
  再投资 = 7%×1.1亿 − 600万 = 170 万;净收入 500 万 → FCFE = 330 万。
- **资本比率用 equity/assets 近似**,与监管口径(风险加权)有差异,标注即可,不影响方法框架。
- **期末 FCFE 为负(常见失败原因)**:发生在"资产增速 × 目标资本比率 > 净收益"时,即公司
  赚的钱不够支撑其扩张计划所需的资本,模型无法计算终值而报错。这**不是计算 bug,而是模型在
  指出假设不自洽**(按该扩张速度公司在烧资本)。补救方向:
  - 调低 `assets_growth` 至可持续水平(银行资产扩张受资本与市场容量约束,长期不应高于名义 GDP 增速);
  - 或调低 `target_capital_ratio` / `ni_growth` 使留存收益足以支撑扩张;
  - 或如实说明公司确实处于资本消耗状态(此时该法不适用,由其他方法承担估值)。
  注意 `ni_growth` 与资产增速、ROE 要联动:资产扩张快于利润增长会持续拉低资本比率。
- **稳定期同样要求 COE > g_terminal**,与 DDM 一致。
- 保险公司同样受监管资本约束,方法相同。

## 输出参数

按 schema(FcfeParams)输出:fcfe_applicable、skip_reason、assets、assets_growth、
target_capital_ratio、current_capital_ratio、equity_current、net_income、ni_growth、
projection_years、g_terminal、coe_terminal(可为 None,缺省沿用高增长期 COE)、buffer_gap_note、analysis。
assets / equity_current / net_income 取上下文「Python 判定口径后的年度数据」里的值(已年化),
不要用未换算的原始序列值。

## 公式(Python 计算,你不需要算)

- 再投资_t = 目标资本比率 × 资产_{t-1} × (1+资产增速) − 股权_{t-1}
- FCFE_t = 净收入_t − 再投资_t
- 股权_t = 股权_{t-1} + 再投资_t(留存收益增厚账面股权,资本比率路径同步给出)
- 终值 = FCFE_{T+1} / (COE_terminal − g_terminal);每股价值 = (Σ FCFE 现值 + 终值现值) / 股数
