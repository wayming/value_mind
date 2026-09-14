"""每个 skill 的结构化输出 schema(LLM 的判断产物,Python 计算的输入参数)。

约定:
- 百分比一律用小数(9.6% -> 0.096)。
- 金额字段单位为元/本币(与 MCP 返回口径一致)。
- 除明确标注外,None 表示"该字段不适用或数据缺失",须在相应 reason/analysis 字段说明。
- 所有 schema 的字段尽量扁平,保证结构化输出在 DeepSeek 上稳定。
"""

from pydantic import BaseModel, Field


class ClassifyParams(BaseModel):
    """公司分类(skill 01):确定公司属于哪类金融服务公司、哪些估值方法适用。"""

    company_type: str = Field(description="公司类型:bank(银行)/insurance(保险)/investment_bank(投行)/investment_company(投资公司)/diversified(混业),只能取其一")
    business_lines: list[str] = Field(description="主要业务线列表,如 ['零售银行','财富管理'],依据 MCP 收入表指标判断")
    company_name: str | None = Field(default=None, description="公司中文/英文全称(如可得)")
    methods_enabled: list[str] = Field(description="建议启用的估值方法,子集:ddm(股息贴现)/reg_capital_fcfe(监管资本FCFE)/excess_returns(超额回报)/relative_valuation(相对估值),按适用程度排序,必须使用这四个精确名称")
    methods_disabled_reasons: dict[str, str] = Field(default_factory=dict, description="被排除方法名 -> 排除理由,如 {'ddm': '投行业务股息不稳定'}")
    risk_vs_peers: str = Field(description="股权风险与同业对比初判(估值驱动器#1):higher/average/lower,并简述依据(业务组合、交易/证券化业务占比等)")
    capital_buffer_note: str = Field(default="", description="监管资本缓冲区初判(估值驱动器#3):公司资本比率相对监管要求的宽松程度印象(后续节点细化)")
    analysis: str = Field(description="分类分析过程简述:用了哪些指标、如何判断业务性质")


class CostOfEquityParams(BaseModel):
    """股权成本(skill 02):COE = Rf + β × ERP,由 Python 计算;这里输出判断参数。"""

    beta: float = Field(description="高增长期 β 系数,小数。MCP 无 beta 数据,基于行业知识与同业公司假设,理由写入 beta_source")
    beta_source: str = Field(description="β 的来源与依据:参照哪类同业平均(如大型货币中心银行约 1.2)、业务风险档位如何影响取值")
    beta_risk_adjustment: float = Field(default=0.0, description="业务风险调整量(可正可负,小数):证券化/交易/投行业务占比高则上调,传统存贷为主则 0 或下调")
    terminal_beta: float = Field(default=1.0, description="稳定期 β(第九章:稳定期趋 1)")
    rf: float = Field(description="无风险利率(小数),标注币种与期限一致性")
    rf_tenor: str = Field(description="无风险利率的期限与来源说明,如 '美国10年期国债收益率' 或 '中国10年期国债收益率'")
    erp: float = Field(description="股权风险溢价(小数),成熟市场约 0.05 量级,新兴市场上调")
    erp_rationale: str = Field(description="ERP 取值的理由:市场成熟度、历史水平、当前环境")
    analysis: str = Field(description="股权成本判断过程简述:高增长与低风险假设的联动(高增长应配高 β)、监管变化对 β 的影响")


class DividendsGrowthParams(BaseModel):
    """股息与增长(skill 03):增长率 g = ROE × (1 − 派息率),由 Python 计算。"""

    dividends_reliable: bool = Field(description="股息是否可靠可预测(不派息/波动大则 False,将触发 DDM 降级)")
    dps_by_year: dict[str, float] = Field(default_factory=dict, description="最近各年度每股股息(年度值),键为年份如 '2023',来自 dps 指标年度数据或计算")
    payout_ratio: float = Field(default=0.0, description="选定的派息率(小数):优先用年度 dps/eps 自算,不要直接用季报 payoutratio 指标")
    payout_basis: str = Field(description="派息率口径说明:用哪一年(或多年均值)的 dps/eps、为何不用 payoutratio 指标(季报口径失真)")
    include_buybacks: bool = Field(default=False, description="是否纳入股票回购计算复合派息率(回购年度波动大,须多年平均)")
    buyback_adjusted_payout: float | None = Field(default=None, description="复合派息率=(股息+回购)/净收益(小数),多年平均;不纳入回购则为 None")
    roe_normalized: float = Field(default=0.0, description="归一化 ROE(小数):当前 ROE 经危机/监管资本要求/周期调整后的可持续水平。必须是**年度**口径")
    roe_normalization_reason: str = Field(description="归一化理由:如监管资本比率要求提高约30%使 ROE 从 17.56% 降至 13.51%(第九章富国银行例)")
    roe_series_basis: str = Field(
        default="年度",
        description="数据源 roe 序列的口径(必答,决定 Python 是否年化):"
                    "『单季』= 每个点是一个季度的 ROE(值明显偏小,如银行 2–3%,×4 才到年度水平);"
                    "『半年度』= 每点是半年(×2);『年度』= 每点已是滚动年度/TTM 值(×1,如银行 10–12%);"
                    "『月度』= 每点是一个月(×12)。"
                    "判断方法:看序列数值本身是否已落在该公司年度 ROE 的合理量级,以及序列是否平滑"
                    "(TTM 平滑、单季锯齿)。日期间隔不能用来判断这一点——季度采样既可能采的是单季值,"
                    "也可能采的是滚动年度值。",
    )
    shares_outstanding: float | None = Field(default=None, description="总股本(股数),用于换算每股价值;取 sharesBasic 等指标最新值")
    analysis: str = Field(description="派息与增长判断过程简述:回购的可持续性、留存收益与监管资本约束的联动")


class DdmParams(BaseModel):
    """股息贴现模型(skill 04)参数,公式由 Python 计算。"""

    ddm_applicable: bool = Field(description="DDM 是否适用:依赖 skill 03 的 dividends_reliable,股息不可靠则为 False")
    skip_reason: str = Field(default="", description="不适用时必须填写理由")
    eps0: float | None = Field(default=None, description="最近年度每股收益(小数),用于生成高增长期股息序列")
    payout_high: float | None = Field(default=None, description="高增长期派息率(小数),通常沿用 skill 03 结论")
    g_high: float | None = Field(default=None, description="高增长期收益/股息增长率(小数),须与 ROE×(1−派息率)自洽")
    stage_years: int | None = Field(default=None, description="高增长期年限(如 5)")
    g_terminal: float | None = Field(default=None, description="稳定期永续增长率(小数),不超过无风险利率量级")
    roe_terminal: float | None = Field(default=None, description="稳定期 ROE(小数),理论上趋近稳定期股权成本")
    coe_terminal: float | None = Field(default=None, description="稳定期股权成本(小数,β 趋 1);None 则用高增长期 COE")
    analysis: str = Field(description="阶段假设的判断过程:高增长期能持续多久、终值假设的合理性")


class FcfeParams(BaseModel):
    """监管资本 FCFE 模型(skill 05)参数,公式由 Python 计算。"""

    fcfe_applicable: bool = Field(description="监管资本 FCFE 是否适用(银行/保险通常适用;投行/投资公司资本约束弱,可能不适用)")
    skip_reason: str = Field(default="", description="不适用时必须填写理由")
    assets: float | None = Field(default=None, description="总资产(最新,本币)")
    assets_growth: float | None = Field(default=None, description="资产年增速假设(小数)")
    target_capital_ratio: float | None = Field(default=None, description="目标股权/资产资本比率(小数):监管要求+管理层选择,保守银行更高")
    current_capital_ratio: float | None = Field(default=None, description="当前股权/资产资本比率(小数),与目标比较得到缓冲区缺口")
    equity_current: float | None = Field(default=None, description="当前股权账面价值(本币)")
    net_income: float | None = Field(default=None, description="最近年度净收益(本币)")
    ni_growth: float | None = Field(default=None, description="净收益年增速假设(小数),与资产增速、ROE 保持自洽")
    projection_years: int | None = Field(default=None, description="高增长期预测年限")
    g_terminal: float | None = Field(default=None, description="稳定期永续增长率(小数)")
    coe_terminal: float | None = Field(default=None, description="稳定期股权成本(小数,β 趋 1);None 则沿用高增长期 COE")
    buffer_gap_note: str = Field(default="", description="监管缓冲区缺口判断(估值驱动器#3):资本比率与监管要求及公司自身目标的差距,不足则未来再投资大、价值低")
    analysis: str = Field(description="资本约束判断过程:监管要求、管理层资本策略、资产扩张计划")


class ExcessParams(BaseModel):
    """超额回报模型(skill 06)参数,公式由 Python 计算。"""

    applicable: bool = Field(description="超额回报模型是否适用(股权账面价值可观察即适用,几乎总是 True)")
    skip_reason: str = Field(default="", description="不适用时必须填写理由")
    bv_equity: float | None = Field(default=None, description="当前股权账面价值(本币,equity 指标)")
    bvps: float | None = Field(default=None, description="每股账面价值(bvps 指标),用于换算每股")
    roe_current: float | None = Field(default=None, description="当前/归一化 ROE(小数),与 skill 03 的 roe_normalized 一致")
    roe_terminal: float | None = Field(default=None, description="终期 ROE(小数):超额回报终会消失,理论上均值回归到股权成本")
    years_high: int | None = Field(default=None, description="高 ROE 维持年限")
    years_fade: int | None = Field(default=None, description="ROE 线性衰减到终值的年限(0 表示立即回归)")
    payout: float | None = Field(default=None, description="派息率(小数),用于逐年增厚股权账面价值")
    mean_reversion_rationale: str = Field(default="", description="均值回归依据:竞争、监管、业务结构(如高 ROE 往往伴随高风险业务,第九章)等")
    analysis: str = Field(description="超额回报判断过程:超额回报可持续性、风险/回报权衡(账目两边:活动产生的 ROE 与带来的风险)")


class RelativeParams(BaseModel):
    """相对估值(skill 07)参数,公式与统计量由 Python 计算。"""

    applicable: bool = Field(description="相对估值是否适用(有市场交易倍数即适用)")
    skip_reason: str = Field(default="", description="不适用时必须填写理由")
    pb_current: float | None = Field(default=None, description="当前市净率(pb 指标最新值,小数)")
    pe_current: float | None = Field(default=None, description="当前市盈率(pe 指标最新值,小数)")
    bvps: float | None = Field(default=None, description="每股账面价值(小数,本币)")
    eps: float | None = Field(default=None, description="最近年度每股收益(小数,本币)")
    roe_series_metric: str = Field(default="roe", description="用于回归的 ROE 历史序列指标名(通常为 roe);σ 由 Python 从 MCP 缓存序列计算,LLM 不手算标准差")
    roe_window: str = Field(default="5y", description="ROE 序列窗口,如 2y/5y/all")
    roe_std_dev_assumed: float | None = Field(default=None, description="仅在 ROE 序列无法从数据取得时使用的假设标准差(小数,如 0.28);由 Python 优先用真实序列计算,此值只是数据缺失时的降级假设,必须在 comparable_notes 或 analysis 中标注为假设")
    comparable_notes: str = Field(default="", description="可比公司说明:同类银行的 PB/PE 水平(LLM 知识或从 MCP 抓取同业数据),注明来源")
    loan_loss_provision_note: str = Field(default="", description="贷款损失准备金影响说明:计提保守的银行报告收益被压低、PE 虚高,反之亦然")
    diversification_note: str = Field(default="", description="业务多样化影响说明:混业公司不同业务风险/增长率不同,难找真正可比公司,市场倍数可比性下降")
    analysis: str = Field(description="相对估值判断过程")


class SynthesisParams(BaseModel):
    """综合(skill 08):汇合各方法结果,输出最终估值结论。"""

    value_range_low: float = Field(description="估值区间下沿(每股,本币)。应覆盖多数有效方法的结果;数据缺失多或方法分歧大时放宽")
    value_range_high: float = Field(description="估值区间上沿(每股,本币)")
    method_weights: dict[str, float] = Field(default_factory=dict, description="各方法权重(0~1,和为 1)。键必须使用以下精确名称(只对已成功计算的方法给权重):ddm_per_share(股息贴现)、fcfe_per_share(监管资本FCFE)、excess_per_share_perpetuity(超额回报永续)、excess_per_share_staged(超额回报均值回归)、relative_pb_fair_price(相对估值)。加权均值由 Python 用你给的权重计算,你不需要自己算,也不要猜这个数")
    method_reconciliation_note: str = Field(description="各方法结果分歧的解释与权重依据:公司类型、数据质量、假设可靠性")
    drivers: dict[str, str] = Field(default_factory=dict, description="三个估值驱动器回顾:equity_risk(股权风险)/growth_quality(增长质量:增长的ROE)/capital_buffer(监管缓冲区),值为结论文本")
    investment_notes: dict[str, str] = Field(default_factory=dict, description="第九章投资四技巧评估:capital_buffer/operating_risk/transparency/entry_barriers,值为结论文本")
    verdict: str = Field(description="估值结论:低估/合理/高估,附一句话依据")
    conviction: str = Field(description="结论置信度:high/medium/low + 理由(数据缺失多则降级)")
    key_risks: list[str] = Field(default_factory=list, description="主要风险点列表")
    disclaimer: str = Field(default="本报告为方法演练输出,不构成投资建议。", description="免责声明")
    analysis: str = Field(description="综合判断过程简述")
