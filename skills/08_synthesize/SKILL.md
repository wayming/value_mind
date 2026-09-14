---
name: financial-services-valuation-synthesis
description: 汇合 DDM、监管资本 FCFE、超额回报、相对估值的结果,结合三个估值驱动器与投资四技巧,给出最终估值区间与结论。估值流程最后一步必用。
---

# 综合估值结论

## 目标

把各方法的结果汇合成一个估值区间与结论。你收到的是各方法已经算好的每股价值
(Python 算的),你的工作是**解释分歧、定权重、回顾驱动因素、给出结论**。

## 输入(上下文中已提供,不必再抓数据)

- 各方法每股价值:ddm_per_share、fcfe_per_share、excess_per_share_perpetuity(上界)、
  excess_per_share_staged(中央值)、relative_pb_fair_price。
- 各方法的参数、警告、错误(skill 结果摘要)。
- 全局参数:COE、增长率、派息率、归一化 ROE。
- 公司分类与业务线。
- 数据覆盖情况。

## 分析步骤

1. **比较各方法结果**:谁高谁低?差异来自哪里?
   - 超额回报永续版是天然上界(假设超额回报永不消失)。
   - 相对估值受市场情绪影响,与内在估值分歧大时说明市场定价与基本面的偏差。
   - 某些方法报错/被跳过时,不参与区间,但要在结论中说明缺失了什么视角。
2. **定权重(method_weights)**:基于公司类型与数据质量——银行且股息可靠 → DDM/FCFE 权重高;
   投行 → 超额回报与相对估值为主;数据缺失多的方法降权。权重和必须为 1。
   **加权均值由 Python 用你给的权重计算,你不需要自己算,也不要写出这个数字**——
   若你认为某个方法离群,请通过调低它的权重来表达,而不是让权重与结论脱节。
   键名必须精确使用:ddm_per_share / fcfe_per_share / excess_per_share_perpetuity /
   excess_per_share_staged / relative_pb_fair_price,且只对已成功计算的方法给权重。
3. **三个估值驱动器回顾**(第九章的核心分析框架,逐个给结论):
   - 驱动器#1 股权风险:与行业平均相比,公司风险程度如何?(业务组合、β 判断)
   - 驱动器#2 增长质量:增长可以增加、减少或不影响价值——公司在追求增长时的 ROE 是多少?
     低于 COE 的增长毁价值,高于 COE 的增长增价值。
   - 驱动器#3 监管缓冲区:资本比率与监管要求(及公司自身规定)的差距;缓冲区不足 → 未来派息受限、价值低。
4. **投资四技巧评估**(第九章"估值技巧"):
   - 资本缓冲区:不仅达标,是否超标准满足要求?
   - 营运风险:风险 ≤ 平均且收益健康?
   - 透明度:报告是否提供营运细节与风险敞口?不透明可能是蓄意隐瞒风险。
   - 进入壁垒:高 ROE 是否来自对新进入者有重大障碍的领域?
5. **给区间与结论**:
   - value_range_low/high:覆盖多数方法结果(剔除被跳过的);数据缺失多 → 区间放宽。
     Python 会检查加权均值是否落在你给的区间内,不在区间内会告警。
   - verdict:低估/合理/高估 + 依据;conviction:数据与假设的可靠性。
   - 结论必须引用具体驱动因素,不得只报数字。

## 坑与注意点

- **区间必须覆盖主要方法结果**:某个方法的结果明显离群时,要么解释原因并降权,要么纳入区间下/上沿,不能悄悄丢掉。
- **错误的方法结果不得参与区间**。
- **不要编造没算过的东西**:所有数字来自上下文,你没有的都标"未知"。
- 免责声明必填。

## 输出参数

按 schema(SynthesisParams)输出:value_range_low、value_range_high、
method_weights(和=1)、method_reconciliation_note、drivers(equity_risk/growth_quality/
capital_buffer 三个键)、investment_notes(capital_buffer/operating_risk/transparency/
entry_barriers 四个键)、verdict、conviction、key_risks、disclaimer、analysis。

注意:不要输出 value_best —— 加权均值由 Python 计算(schema 里没有这个字段)。

## 公式(Python 计算,你不需要算)

- 加权均值由 Python 计算;你只需给权重与区间判断。
