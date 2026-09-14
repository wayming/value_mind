---
name: financial-services-relative-valuation
description: 金融服务公司相对估值:PE/PB 倍数分析,第九章 PB 回归公式(PB=1.527+8.63×ROE−2.63×σ),贷款损失准备金与业务多样化的坑。σ 由 Python 从 MCP 数据计算。
---

# 相对估值(PE / PB)

## 目标

用股权倍数(不是企业价值倍数)给金融服务公司做相对估值——与第九章"只对股权估值"一致。
提供:市场当前倍数、回归预测倍数、隐含倍数,三者对照判断高估/低估。

## 何时使用 / 不适用

- 有市场交易倍数(pe、pb 指标存在)即适用。
- 数据缺失时可只用可比公司知识判断,但结论降级。

## 数据获取(MCP 工具)

1. `get_financials` 取:ratios 组的 pe、pb、roe(取**全历史序列**,用于算 σ,窗口由你定 2y/5y/all)、
   peForward;income 组 epsBasic;balance 组 bvps。
2. 可比公司数据:MCP 没有同行清单接口。可以:
   - 用你的知识给出同类银行的 PB/PE 水平(注明来源与时间);
   - 或调 `get_financials` 抓 1–2 家同业代码的 pb/pe/roe 对比(先 get_data_period 确认有数据)。

## 分析步骤

1. **当前倍数**:pb_current、pe_current(最新值)。
2. **回归预测 PB**(第九章公式,由 Python 算):你只提供 roe(当前/归一化)和 roe_series_metric/roe_window,
   **σ 由 Python 从 MCP 缓存的 ROE 历史序列计算**——不要自己手算标准差。
   Python 按 skill 03 声明的序列口径年化,保证 σ 与回归里年度口径的 ROE 同单位;
   口径判断错了 σ 就会差 4 倍,直接扭曲预测 PB。
   只有在 ROE 序列确实取不到时,才填 `roe_std_dev_assumed` 作为降级假设(必须标注为假设,
   代码会在报告中告警);能取到数据时该字段留空。
3. **可比公司**:同类银行的 PB/PE 水平如何?这家公司相对同业的 ROE、风险(标准差)能否支撑更高倍数?
   写进 comparable_notes。
4. **PE 的特殊坑**:
   - **贷款损失准备金**:银行照例为不良贷款预留准备金,减少报告收益、抬高 PE。
     计提保守的银行报告的收益更低(PE 虚高),计提激进的银行收益更高(PE 虚低)。
     比较 PE 前必须评估计提政策差异 → loan_loss_provision_note。
   - **业务多样化**:投资者愿为商业贷款收益付的倍数 ≠ 为交易收益付的倍数。
     混业公司有多组风险/增长率/回报率,难找真正可比公司 → diversification_note。
5. **隐含倍数对照**:Python 会算隐含 PE(= 派息率×(1+g)/(COE−g))与隐含 PB(= 内在估值/BVPS),
   与市场倍数对照:市场倍数高于基本面支撑 → 高估倾向,反之低估。

## 坑与注意点

- **书中数字勘误**:第九章汤普金斯金融例,PB 回归代入结果印的是 1.95,但按书中系数实算
  = 1.527 + 8.63×0.2798 − 2.63×0.2789 = **3.2082**,结论方向相反(1.95 → 高估,3.21 → 低估)。
  以公式为准;且该回归 R² 仅 31%,**结果只作方向性 band,不作为精确公允价值**。
- **PB 的主导变量是 ROE**:同 ROE 下,风险更高的银行应有更低 PB(σ 项体现);增长潜力更高的
  银行应有更高 PB。别忽略 ROE 之外的基本面。
- 金融服务公司 PB 与 ROE 的关系比一般公司更强(账面股权更接近资产市价)。
- 当前 pe/pb 可能为负或异常(亏损、净资产异常),标注并降级处理。

## 输出参数

按 schema(RelativeParams)输出:applicable、pb_current、pe_current、bvps、eps、
roe_series_metric、roe_window、roe_std_dev_assumed(仅数据缺失时)、comparable_notes、loan_loss_provision_note、
diversification_note、analysis。

## 公式(Python 计算,你不需要算)

- PB 回归 = 1.527 + 8.63×ROE − 2.63×σ(ROE),R²=31%(σ:样本标准差,Python 从缓存序列算)
- 回归公允价值 = 预测 PB × BVPS
- 隐含 PE = 派息率 × (1+g) / (COE − g);隐含 PB = 内在估值每股 / BVPS
