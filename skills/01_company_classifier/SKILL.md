---
name: financial-services-company-classifier
description: 对公司进行金融服务行业分类(银行/保险/投行/投资公司/混业),判断第九章哪些估值方法适用。估值流程第一步必用。
---

# 金融服务公司分类与估值方法选择

## 目标

给一家金融服务公司估值前,先确定它属于哪一类,因为第九章的方法适用性依公司类型而不同:
错误的分类会导致用错模型(如给股息不稳定的投行用 DDM)。

## 金融服务业四大类(按赚钱方式分)

| 类别 | 赚钱方式 | 收入表特征指标 |
|---|---|---|
| 银行(bank) | 存贷款利差 | netInterestIncomeBank(净利息收入)、revenueBank |
| 保险(insurance) | 保费收入 + 投资组合收益 | 保费类收入指标、investmentSecurities 占比高 |
| 投资银行(investment_bank) | 咨询、承销、并购顾问 | nonInterestIncomeBank 占比高、手续费类收入 |
| 投资公司(investment_company) | 投资咨询费、管理费 | 咨询费/管理费类收入 |

**注意**:随着行业洗牌,越来越多公司混业经营(diversified)——一家公司同时做存贷、交易、投行、资管。
混业公司要分段评估不同业务的风险,不能只看报表名目。

## 数据获取(MCP 工具)

1. 先调 `list_metrics(exchange, code)` 了解该公司有哪些指标可用(按报表类型分组)。
2. 调 `get_data_period(exchange, code)` 确认数据覆盖时间范围。
3. 调 `get_financials(exchange, code, metrics=[...], period="2y")` 取关键分类指标,优先看收入表(income-statement)的构成,再辅以资产负债表(如 interestBearingDeposits、investmentSecurities、grossLoans)。
4. 数据够用即止,不要一次拉取全部指标。

## 分析步骤

1. **看收入构成**:净利息收入占比高 → 银行;保费+投资收益 → 保险;手续费/咨询费主导 → 投行或投资公司。
2. **看资产构成**:大量贷款(grossLoans)→ 银行;大量投资证券 → 保险或投资公司。
3. **判断混业程度**:若多项业务线收入都显著,归类为 diversified,并列出主要业务线。
4. **选择适用方法**(必须使用这四个精确名称:`ddm`、`reg_capital_fcfe`、`excess_returns`、`relative_valuation`):
   - 银行、保险:四法全开,监管资本约束强,股息通常较稳定。
   - 投行、投资公司:股息不稳定、资本约束弱 → 默认关闭 `ddm`(或仅作参考),以 `reg_capital_fcfe`(若仍有资本约束)或 `excess_returns`、`relative_valuation` 为主。
   - diversified:按主导业务线的风险档位决定,通常四法都开但提示混业风险。
5. **初判两个估值驱动器**(后续节点会细化):
   - 驱动器#1 股权风险:与同业相比,该公司业务组合风险如何?交易/证券化/投行业务占比高 → 风险高。
   - 驱动器#3 监管缓冲区:资本充足程度印象(粗判即可)。

## 坑与注意点

- **别只凭公司名分类**:很多"银行"实际是混业金融集团。
- **指标可能不存在**:list_metrics 里没有的指标不要硬要,用近似指标替代并说明。
- **保险公司的债务和股权更难区分**:保费收入是负债来源,资本定义同样要狭义化(只含产权资本)。
- 分类结论决定了后续所有节点的走向,宁可多抓一两个指标确认,不要草率。

## 何时不适用

- 非金融服务公司(制造业等)不适用本章方法——若数据表明该公司不在四大类之内,在 analysis 中明确说明,methods_enabled 只保留相对估值,其余排除并说明理由。

## 输出参数

按 schema(ClassifyParams)输出:company_type、business_lines、company_name、methods_enabled(按适用程度排序)、methods_disabled_reasons(每个被排除方法给理由)、risk_vs_peers、capital_buffer_note、analysis。

## 公式

本步骤无公式计算,全部是分类判断。
