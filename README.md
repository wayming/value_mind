# value_mind — 金融服务公司估值 AI Agent

给定一家金融服务公司(银行 / 保险 / 投行 / 投资公司),按《估值》第九章
「反弹:对金融服务公司估值」的方法论自动完成估值并输出 markdown 报告。

```bash
python3 main.py SHA 600036          # 招商银行
python3 main.py NYSE WFC --period 5y
python3 main.py --check-llm         # 只做连通性自检
```

## 设计:四条硬约束怎么落的

| 要求 | 实现 |
|---|---|
| **LangGraph 编排** | [app/graph.py](app/graph.py):串行前三步 → `Send` 条件扇出四个方法节点并行 → 屏障汇合 → 综合 → 写报告 |
| **公式不是 skill,固定计算由 Python 做** | [app/formulas.py](app/formulas.py) 是零 IO 的纯函数库(21 个测试锁住书中数字);LLM 只输出参数,没有算术 |
| **LLM 负责不可量化的判断** | 每个 skill 写清"这个公式的坑 / 适用方向 / 数据陷阱";LLM 据此自主抓数、判断适用性、给参数 |
| **MCP 只注册给 LLM,何时调用由 LLM 决定** | [app/mcp_tools.py](app/mcp_tools.py) 把三个接口包成 LangChain 工具;编排代码里没有任何一处硬编码"该调哪个接口" |
| **按 skill 切成小块,不一次性喂给 LLM** | 每个节点只加载自己那一份 `SKILL.md` 当 system prompt([app/skill_loader.py](app/skill_loader.py)),节点之间靠 state 传结论 |

### 关键分界:**Python 算数,LLM 判事**

```
LLM 判断                           Python 计算
─────────────────────────────      ────────────────────────────
公司归为 bank 还是 diversified  →   —
β / Rf / ERP 取多少             →   COE = Rf + β×ERP
股息可靠吗、派息率用哪个口径      →   g = ROE × (1 − payout)
高增长期几年、终值假设           →   DDM 折现 + 终值 + 敏感性网格
资产增速、目标资本比率           →   再投资 = Δ目标资本 − 现有股权,FCFE 折现
超额回报持续多久、如何衰减       →   (ROE−COE)×BV 逐年折现
PB 回归参数是否可信              →   公平价 = 预测PB × BVPS
各方法权重、估值区间             →   加权均值(LLM 只给权重)
```

**修复回路**:计算发生在 LLM 输出参数*之后*,所以 LLM 看不到自己的假设不自洽
(如"期末 FCFE 为负""COE ≤ g")。`compute_with_repair()`
([app/nodes.py](app/nodes.py))捕获公式报错,把报错原文回喂给 LLM 让它改参数、
重算一次(修正轮禁用工具,防止它跑去重新抓数据)。这是闭环成立的必要结构补丁。

综合节点还有第二处同类回路 `_reconcile_range()`:加权均值由 Python 从 LLM 的权重算出,
若落在 LLM 自己给的区间之外(报告会印成"区间 53~70,均值 71.4"),回喂一次让它
**自己决定**是放宽区间还是调权重——这是估值判断,代码不替它选,但必须二者自洽。

第三处在参数**进入** Python 之前:`run_skill_agent()` 的回喂。结构化输出走的是
ToolStrategy(schema 绑成工具),模型把 4KB 中文散文塞进参数时会写未转义的英文双引号
(实测 NAB 综合节点 `属"温和增价值"`),args 不是合法 JSON —— 此时 langchain 只把它
扔进 `invalid_tool_calls` 就返回,**既不报错也不重试**,`structured_response` 静默为 None。
代码把模型自己的原文 + JSONDecodeError 那句 + `char N` 附近的窗口一起回喂,让它改转义
而不是重做整份分析;只回喂一次,仍失败才抛。参数超过 8000 字符则退回通用提示词
(回喂会让输入翻倍)。

**权重键归一** `normalize_weight_keys()`([app/state.py](app/state.py)):LLM 常把方法名
(`reg_capital_fcfe`)或拼写变体(`excess_share_perpetuity`,少了个 s 又漏了 per)当成结果键
写进 `method_weights`,导致该方法的结果静默拿不到权重——实测 WFC 上 FCFE 84.11、
超额回报 64.45/55.78 都因此掉出过加权均值。三级判断从严到宽:已是规范键 → 查方法名别名表
→ 剥掉 `per`/`share` 这类通用词后**剩余词集完全相等且只能命中一个规范键**才归位。
裸 `excess_returns` 对应两个结果键,**不猜**,原样落进告警——归一与判断的分界线就在这里。

**两条 LLM 通道,结构化输出不能想当然**([app/llm.py](app/llm.py)):`check_llm()`
按序探测 OpenAI 兼容端点与 Anthropic 兼容端点。结构化那一步**没有**用
`create_react_agent` 的 `response_format`——它内部固定调 `with_structured_output(schema)`
不带 `method`,langchain-openai 会默认选 `json_schema`,而本网关的 OpenAI 端点直接回
400 `This response_format type is unavailable now`;换成 `function_calling` 又会撞上
`Thinking mode does not support this tool_choice`。所以工具循环与结构化拆成两步,
`structured_model()` 自己发结构化调用并按 400 自动降级(`function_calling` → `json_mode`)。

**口径统一:LLM 判断口径,Python 做算术**。数据源的比率类指标(ROE/ROA)口径
**因公司而异**——实测同一份数据源里,ASX NAB 的 `roe` 是滚动年度值(0.099–0.124,平滑),
NYSE WFC 的是单季值(0.014–0.038,锯齿),**而两者的日期间隔都是 91 天**。
日期只能说明多久采一次,说明不了每个点代表多长期间的比率,所以口径判断交给 LLM
(`roe_series_basis` 字段,依据是数值量级与平滑度),`ratio_series_stats()`
([app/formulas.py](app/formulas.py))按声明的口径把 latest/mean/min/max/**σ** 一起年化,
`basis_conflict()` 只兜一种硬矛盾(间隔一年却声称每点是单季值)。

判错方向的代价很直接:ROE 年化而 σ 没年化,回归里 `−2.63×σ` 项被缩小到 1/4,
预测 PB 机械性偏高;把年度值当单季值则 σ 高估 4 倍。

**序列必须属于一家公司,且来自单次抓取**(`extract_series()`
[app/mcp_tools.py](app/mcp_tools.py))。这里出过一次最隐蔽的错:skill 07 会抓同业做
PE/PB 对比,而旧实现只按日期把缓存里所有 `get_financials` 结果的同名指标合并——
ASX NAB 报告里印的"数据最新 ROE 0.0949"其实是**西太平洋银行的**,NAB 自己是 0.09908;
两家澳洲银行报告日完全相同(3/31、9/30),逐日覆盖,报告里看不出任何异常,而 σ 是两家
公司的混合体,直接喂进了 PB 回归。改成按 `exchange`/`code` 限定,并**取单次抓取而非合并**
(1y/2y/5y/all 在重叠日期上取值一致,合并不出错值,但 `all` 比 `5y` 多 20 年历史、含
2020 年 ROE 腰斩那段,σ 会随 LLM 恰好调过哪些 period 而变)。选不中声明的 `roe_window`
时取信息量最大的一份,并把**实际用的** period 回传、写进告警与报告
(`roe_std_source: 数据序列(roe, period=5y, n=20, 年度口径年化)`),让 σ 可复算、可追溯。

**缓存里存的必须是 structuredContent,不是工具返回的原始块**([app/mcp_tools.py](app/mcp_tools.py))。
换成 langchain-mcp-adapters 后,工具的 `ainvoke()` 只回 content 块
(`[{"type":"text","text":"{...}"}]`),structuredContent 被丢在 artifact 里拿不到,
于是缓存从 dict 变成了 list:`extract_series()` 里读 `raw["data"]` 直接
`'list' object has no attribute 'get'`——07 相对估值每次运行都失败,03 的 ROE 交叉校验
则静默跳过(那处 except 只记 debug,报告上完全看不出来)。`_structured()`
把文本块还原成 structuredContent(三个工具实测与 structuredContent 逐字节相同),
缓存存 dict、给 LLM 的仍是压缩后的文本(未压缩的一次 `get_financials` 上万 token,
长序列还会把关键的头尾挤出视野)。这类"形状"错误自己不会叫,所以 `_points_of()`
现在遇到非 dict 直接抛,而不是继续往下走。

## 目录

```
main.py              CLI 入口
mcp_client.py        MCP HTTP 客户端(原有文件,未改动)
chapter9.md          方法论原文
app/
  graph.py           图组装、条件扇出、并行屏障
  nodes.py           8 个节点 + run_skill_agent + 修复回路
  formulas.py        全部公式(纯函数,零 IO)
  schemas.py         8 个 Pydantic 模型(LLM 结构化输出契约)
  state.py           ValuationState + merge_dicts reducer + 规范键名表
  mcp_tools.py       MCP → LangChain 工具(结果压缩、错误不抛出)
  skill_loader.py    SKILL.md 加载(YAML frontmatter + 正文)
  report.py          纯 Python markdown 报告
  llm.py             双通道 LLM(auto 探测 openai / anthropic)
  llm_log.py         全量对话日志(llm.out,YAML 多文档流)
skills/<NN_name>/SKILL.md   每个节点一份中文分析指导
tests/               85 个测试
reports/             输出报告
```

## 图拓扑

```
START → classify(01) → cost_of_equity(02) → dividends_growth(03)
      → [Send 扇出] → ddm(04) / reg_capital_fcfe(05) / excess_returns(06) / relative_valuation(07)
      → synthesize(08) → report_writer(纯 Python) → END
```

前三步串行(方法节点依赖它们产出的全局参数 COE / g / payout);四个方法互不依赖,
并行跑,每个节点自己决定抓什么数据。`--skip` 可从扇出中剔除方法。

## Skills

每份 `SKILL.md` 是独立的分析指导,结构固定:

```
## 目标           这一步要产出什么
## 输入           上下文里已经有什么、还缺什么
## 分析步骤       分步怎么做(含该抓哪些指标)
## 坑与注意点     公式的适用边界、数据陷阱、书中特别提醒
## 输出参数       对应哪个 schema 的哪些字段
## 公式           Python 算,你不需要算
```

加一个新方法 = 加 `skills/NN_xxx/SKILL.md` + 在 [app/schemas.py](app/schemas.py)
加参数模型 + 在 [app/formulas.py](app/formulas.py) 加公式函数 + 在
[app/nodes.py](app/nodes.py) 加节点 + 在 [app/graph.py](app/graph.py) 注册。
公式和 skill 严格分开:skill 里不写死数字,公式里不做判断。

## 覆盖的方法(第九章)

1. **DDM 多阶段** — 高增长期 + 稳定期终值,含终值占比与敏感性网格
2. **监管资本 FCFE** — 再投资 = 目标资本比率 × 资产增量 − 现有股权
3. **超额回报模型** — 股权价值 = 账面股权 + Σ (ROE−COE)×BV 的现值;永续版(上界)与均值回归版
4. **PE 相对估值** — 含准备金扭曲、业务多样化的定性调整
5. **PB 相对估值** — 主导变量 ROE;书中回归 `PB = 1.527 + 8.63×ROE − 2.63×σ`

外加三个**估值驱动器**(股权风险 / 增长质量 / 监管缓冲区)和**投资四技巧**
(资本缓冲区 / 营运风险 / 透明度 / 进入壁垒),在综合节点逐条给结论。

## MCP 接口(LLM 自主调用)

`list_metrics` / `get_data_period` / `get_financials` @ `http://localhost:8081`
(可用 `MCP_BASE_URL` 覆盖)。工具结果做了压缩(系列最多 12 点、单次最多 30000 字符),
报错以字符串返回而不是抛出,避免一次失败打断整条链。所有调用都记进 `data_cache`,
报告末尾列出来源。prompt 里统一要求"一次批量抓取",减少往返。

## 配置

| 环境变量 | 默认 | 说明 |
|---|---|---|
| `LLM_PROVIDER` | `auto` | auto / openai / anthropic,auto 时按序探测 |
| `LLM_MODEL` | `deepseek-flash` | |
| `LLM_BASE_URL` / `LLM_API_KEY` | api.deepseek.com/v1 | OpenAI 兼容通道 |
| `ANTHROPIC_LLM_BASE_URL` / `ANTHROPIC_LLM_API_KEY` | 复用 `ANTHROPIC_BASE_URL` / `ANTHROPIC_AUTH_TOKEN` | Anthropic 兼容通道 |
| `LLM_DISABLE_THINKING` | `1` | 该网关 thinking 模式拒绝强制 tool_choice,导致结构化输出失败。两条通道关法不同:Anthropic 传顶层 `thinking`,OpenAI 端点放 `extra_body` |
| `LLM_STRUCTURED_METHOD` | 空(自动) | `function_calling` / `json_schema` / `json_mode`。留空按通道自动选并降级 |
| `MAX_LLM_STEPS` | `14` | 单节点 LLM 步数上限;超限自动降级为"无工具直接出参数"重试 |
| `LLM_TEMPERATURE` | `0` | 估值分析要可复现 |
| `LLM_LOG` | `1` | 每次模型往返追加一条记录到 `llm.out`;`0` 关闭 |
| `LLM_LOG_PATH` | `<repo>/llm.out` | 日志路径 |
| `MCP_BASE_URL` | `http://localhost:8081` | |
| `REPORT_DIR` | `<repo>/reports` | |

CLI 覆盖:`--beta --rf --erp --roe --growth --payout --eps` 直接钉住关键假设;
`--skip ddm,relative_valuation` 排除方法;`--only cost_of_equity` 单 skill 调试。

## LLM 对话日志(llm.out)

每次模型往返往 `llm.out` 追加一条记录,**内容不截断**:skill 提示词全文、完整消息列表
(含工具调用与工具返回)、响应正文与 tool_calls、usage、response_metadata 原样保留。
挂载点在模型实例的 callbacks 上([app/llm.py](app/llm.py)),不在各调用点 ——
agent 工具循环里的每一次调用、结构化输出那一步、启动自检的 ping 都会被记到,
以后新增调用点也不会漏。

**格式是 YAML 多文档流,一条记录一个 `---` 文档**(不是 JSON:JSON 的字符串里放不下
字面换行,正文只能以 `\n` 转义出现,读起来是一整行;YAML 的字面块 `|` 把换行直接写进
文件,`yaml.safe_load_all` 又能逐字节读回):

```yaml
---
seq: 22
ts: '2026-09-16 11:36:38.976'
event: end
skill: 07_relative_valuation
model: deepseek-flash
usage: {input_tokens: 13774, output_tokens: 2483, total_tokens: 16257}
request:
  messages:
  - role: system
    content: |
      # 相对估值(PE / PB)

      ## 目标

      用股权倍数(不是企业价值倍数)给金融服务公司做相对估值。
response:
  generations:
  - role: ai
    content: |-
      1) 当前倍数:pb=1.892、pe=18.994。

      σ 由 Python 从 5 年 ROE 序列计算。
```

取记录用 `safe_load_all`(会话头是注释,会被自动跳过):

```bash
python3 -c "
import yaml
for r in yaml.safe_load_all(open('llm.out', encoding='utf-8')):
    if r and r['event'] == 'end':
        print(r['seq'], r['skill'], r['elapsed_ms'], 'ms', r['usage'])"

# 只看某次对话的完整输入输出(正文是真换行,直接读)
python3 -c "
import yaml
for r in yaml.safe_load_all(open('llm.out', encoding='utf-8')):
    if r and r.get('skill') == '07_relative_valuation' and r['event'] == 'end':
        print(r['request']['messages'][-1]['content'][:500])"
```

两点格式上的取舍:多行正文用字面块,但**碰到块标量不允许的内容**(行尾空格、以空格
开头等)PyYAML 会退回带引号并转义——那类内容本来就是块标量的禁区,不值得为它牺牲
其余部分的可读性;**不写 YAML 锚点/别名**(`usage: &id001` 那种),`usage` 与
`generations[].usage_metadata` 是同一个对象,默认会被写成别名,读到那一行还得往上翻。

排查时最常用的两个字段:

- `skill` —— 这次对话属于哪个节点。只能由调用方注入(`app/nodes.py` 里经
  `config.metadata.vm_skill` 传),因为 agent 内部的 `langgraph_node` 恒为 `"model"`,
  光靠它分不出 ddm 还是 fcfe。
- `event` —— `end` / `error`。失败的那次对话同样落盘(输入 + 报错原文),
  自检时哪个候选端点失败、为什么失败,都在这里。

两点代价:文件长得很快(一次完整估值 = 8 个 skill × 最多 14 步工具循环,几十 MB 量级);
SDK 内部重试(`max_retries=2`)发生在一次 `_generate` 里,回调看不到,一次调用仍只记一条。

## 测试

```bash
python3 -m pytest tests/ -q
```

- [tests/test_formulas.py](tests/test_formulas.py) — 29 个,用书中数字锁死公式
  (COE 9.6%、g 6.13%、终值派息率 65.12%、再投资 170 万、超额回报 28.38/股 …),
  以及口径年化(含"σ 不年化会抬高预测 PB"的反向偏差、口径不能由日期推断的反例)
- [tests/test_pipeline_offline.py](tests/test_pipeline_offline.py) — 9 个,打桩 LLM 输出跑通整图(节点全异步,故走 `ainvoke`):
  含两处修复回路(公式报错、区间与权重不自洽)、权重键归一与不可归一时的等权降级、
  单方法报错不拖垮全图
- [tests/test_state_keys.py](tests/test_state_keys.py) — 8 个,权重键归一的边界:
  拼写变体可归、歧义键不猜、别名目标无结果时不假装有权重
- [tests/test_llm_structured.py](tests/test_llm_structured.py) — 7 个,两条通道的结构化输出:
  method 选择与 400 降级、非协议错误不掩盖
- [tests/test_extract_series.py](tests/test_extract_series.py) — 7 个,序列归属与取数选择:
  同业公司的同名指标不得并进目标公司、声明的窗口有对应抓取就必须用它、全空抓取不靠点数胜出
- [tests/test_mcp_tools.py](tests/test_mcp_tools.py) — 7 个,打真实 MCP(5 个)+
  工具返回值形状的离线回归(2 个:文本块还原成结构化结果供缓存用、纯文本错误原样透传)
- [tests/test_llm_log.py](tests/test_llm_log.py) — 11 个,llm.out 全量日志(离线喂原始事件):
  不截断、**正文的 `\n` 落盘是真换行(含空行与缩进)且能原样读回**、按 run_id 配对
  (含 end 乱序到达)、失败落盘、凭证不落盘、写不进去也不抛、并发扇出不交错,以及
  "agent 内部调用靠模型构造时的回调才记得到 + skill 归属靠 `config.metadata` 注入"
  这两条挂载前提
- [tests/test_skill_agent_retry.py](tests/test_skill_agent_retry.py) — 7 个,坏 JSON 的回喂:
  回喂模型自己的原文与出错位置、只取 JSONDecodeError 关键句、过大参数不翻倍输入、
  "压根没调工具"是另一种话术、只试一次,以及既有的步数耗尽降级不被改坏

## 两处书中勘误(已在公式与 skill 中按正确值实现)

1. **超额回报现值**:书中印的 58.22B / 105.85B 与自身假设不自洽,重算为书中逻辑应有的值。
2. **PB 回归**:书中印的代入结果 1.95 与给出的系数不符 —— 用书中系数
   `1.527 + 8.63×ROE − 2.63×σ` 代入富国银行的实际 ROE/σ,得 3.2082,方向相反。
   实现以系数为准,并在 skill 里提醒 LLM 不要被印出来的数字带偏。

---

*本工具为方法论演练,输出不构成投资建议。*
# value_mind
