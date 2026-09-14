# 529M / F2 后续训练：独立审计与可执行研究方案

**研究截止：2026-09-15。对象：`yinli-systems/l20-1b-pretraining`，固定 commit `eb5eded979ba148b3dc06052d436eabc583bf3fc`。**

本报告完整依据用户研究委托，交叉阅读固定 commit 下的模型、CPT runner、实际 Slurm 启动脚本、训练与评测回执，并核验公开一手模型卡、数据卡和论文。没有在本会话中运行 GPU 训练、下载实际权重或重新评测竞品。文中的 GPU 时间是预算估算，不是已经完成的运行。所有没有足够证据的效应、上限、版本或结果标为 **unknown**。

标记：**[仓库事实]** 是已读固定 commit 的记录，不等于本会话独立重跑；**[官方事实]** 是一手公开材料；**[计算]** 是由这些数字推导；**[建议]** 是待预注册的实施选择；**[假设]** 是可以被实验推翻的解释。数字化验收门槛不是收益预测。

## 1. 结论先行：继续项目，但不要把“继续 F2 原配比”当作唯一主线

**推荐：保留 immutable parent 与 F2 两种子作为不可覆盖基线；从同一 parent 比较五个 fresh-corpus 配比，以新建英文能力代理集和 retention 约束选择；先 0.537B pilot，再 2.147B 两种子 confirmation，只有通过后才启动总计约 20B 的长 horizon。F3 留作多语种研究分支，模型融合只做低成本旁路，不作为主训练方案。**

F2 的已知七项均值是 48.5311%，相对 matched base +0.5359 pp；两个种子的接近，不能消除题目采样不确定性。正式 `formal_promotion=false` 应保留。距离 50 还有 1.4689 pp。只恢复 WinoGrande 的 1.6969 pp 与 ARC-Easy 的 1.0101 pp，七项均值只增加 `(1.6969+1.0101)/7=0.3867 pp`，约到 **48.9179%**；仍缺约 **1.0821 pp**。因此需要新增能力，不只是修复退化。[R1] [R5]

**F2 还能提高多少：unknown。达到 50 的概率：unknown。20B 是否足以超过现代强 0.5B–1B base：unknown。** 当前证据支持继续做受控实验，不支持把每 2B 的 +0.54 pp 线性外推，也不能凭更低 PPL 推断最终 accuracy。

### 1.1 先修正四个审计口径

| 项目 | 核验结果 | 必须采取的动作 |
|---|---|---|
| 架构 | 附件的 18×1536 / FFN4096 / Q12-KV4 按仓库公式是 **530,271,744** 参数，不是 528,748,800。实际 confirmation 脚本传 `--architecture deep`，即 **26×1280 / FFN3584 / Q20-KV5 / head64**，计数才是 **528,748,800**。 | 以实际 checkpoint 的 tensor shapes、run manifest、严格 state load 为最终证据；二进制独立验证仍为 **unknown**，不能把代码默认 wide 配置用于续训。[R2] [R3] [R7] |
| 验证 token | held-out 回执写 `validation_prediction_tokens=16,678,912`，但五域 `target_tokens_by_domain` 合计为 **10,103,815**。 | 分开记 packed/processed positions 与实际 mask-selected loss targets；不能将两者都称有效预测 token。[R4] |
| 运行上限 | runner 的 `pilot_steps` 每次启动上限 2048，但 `max_steps=min(total_steps, step+pilot_steps)`，支持固定总 horizon 的 `--resume`。 | 20B 可以分段，不需要重置 LR。每段退出 0/`QUALIFICATION_COMPLETED` 不等于整个 20B 完成；必须核对全局 step。[R3] |
| 七项口径 | PIQA 的冻结指标是 **acc_norm**；不是想当然的 acc。 | 完整保留 task 配置、metric、normalizer 实现、样本数和 seeds；版本未锁定不能宣称 exact matched。[R5] [R6] |

### 1.2 路径选择

| 路径 | 结论 | 预期增益 / 失败信号 / 回滚 |
|---|---|---|
| F2 原配比继续 | 必须保留控制组，但不是默认赢家 | 增益 **unknown**；新代理停滞或 retention 失败，回 parent/F2，不延长同一配比硬赌 |
| F3 转正为英文主线 | 不推荐 | 多语种 LM loss 已改善，不等于英文七项更强；英文代理退化则只保留独立多语种身份 |
| parent→新配比 | **主推荐**，因果比较最清楚 | 幅度 **unknown**；两种子同预算确认；失败保留全部负面结果 |
| F2→新配比 warm-start | 仅在主配比选择后做配对初始化对照 | 可能节省重复学习，幅度 **unknown**；必须计入前 2.147B 的 source exposure，重置 optimizer 不叫 exact resume |
| 权重插值 / 融合 | 便宜旁路，不替代数据实验 | `parent + α(F2-parent)`，α预注册 0.25/0.5/0.75；同架构、同 tokenizer、同 key、FP32 加权、再导出。只用新 dev，预算≤1 GPUh **是上限而非实测**；alias/logit/retention 任一失败即弃用。[S28] |

## 2. 竞品定义与官方研究表

**融合补充：** 可另预注册F2两种子50/50权重平均，作为一个新单模型与两个原模型分别评估；它不是“两种子均值”的同一统计对象。F2/F3各50%融合只在独立多语分支的Pareto目标下考虑，不因F3的五域loss更低而默认进入英文主线。所有融合收益与是否存在低损失连通路径均为 **unknown**。[S28]

### 2.1 比较集合不能只按名字里的“1B”定义

**主集合 C_strict：** 截止研究日已公开、明确 base/PT、可获得权重和 tokenizer、文本 likelihood 可实现、**实际唯一可学习参数 300,000,000–1,100,000,000**，包含 embedding，tied 参数只计一次。需要推理的额外组件也要计入，不用“active parameters”偷换 total parameters。每个模型还须通过许可证与运行器准入。

**历史参照 C_anchor：** TinyLlama 名义 1.1B 的两个检查点、OLMo 名义 1B 等，实际计数尚未独立确认的边界模型先放这里，不悄悄扩张严格上界。若实际计数越界，继续报告，但不计入 strict 冠军集合。

**分离赛道：** 自定义许可证、非商业研究许可、混合架构和含视觉组件分别打标签。Qwen3.5 的语言部分标称 0.8B，不能据此假定总参数合规。OLMo-2 的“1B”命名也不能替代实际参数计数。**“同规模开放权重”不等于“同规模、同许可证、同训练成本”。**

以下全部是官方参考，**没有一行是本项目的新 matched rerun**。精确 checkpoint revision、权重 SHA、作者评测 commit 不能从模型名字推导，未回收的项均为 **unknown**；`competitor_registry.json` 对每行保留这些字段。

| 开放 checkpoint | 参数口径 | 训练 tokens | 原生 context | 数据 / 许可 | 官方七项或最接近结果；工具；能否复跑 |
|---|---:|---:|---:|---|---|
| EleutherAI/pythia-410m | 标称410M；精确计数准入时核对 | 系列约300B | 2048 | The Pile；Apache-2.0 | 七项 **unknown**；lm-eval 系列、精确版本 **unknown**；权重公开，可条件复跑。[S2] |
| EleutherAI/pythia-1b | 标称1B | 300B参考口径 | 2048 | The Pile；Apache-2.0 | TinyLlama 官方对照表 48.30；不是本协议数值；条件复跑。[S1] [S2] |
| keeeeenw/MicroLlama | 标称300M | 50B | 2048 | SlimPajama；Apache-2.0 | 当前官方卡未独立核验附件中的42.36；七项 **unknown**；条件复跑。[S3] |
| HuggingFaceTB/SmolLM2-360M | 标称360M | 4T | 8192，官方 base config | FWE/DCLM/math/code；Apache-2.0 | Hella54.5、ARC平均53.0、PIQA71.7、WG52.5、OBQA37.4；缺完整七项；lighteval，commit **unknown**；可条件复跑。[S4] [S29] |
| Qwen/Qwen2.5-0.5B | 官方0.49B | 单模型精确消费量 **unknown** | 32768 | 通用/数学/代码/多语；Apache-2.0 | SmolLM2 对照中的局部任务数不是本项目七项；exact author tool/version **unknown**；可条件复跑。[S4] [S5] |
| Qwen/Qwen3-0.6B-Base | 官方0.6B | 卡中36T是系列声明；此 checkpoint完整账本 **unknown** | 32768 | 三阶段、多语/STEM/code/synthetic；Apache-2.0 | 七项 **unknown**；需支持该架构的固定 runtime；可条件复跑。[S6] |
| google/gemma-3-1b-pt | 标称1B | 官方2T | 32768 | web/code/math/多语；Gemma terms | PT Hella62.3是10-shot；BoolQ63.2与PIQA73.8为0-shot；不能混算零样本七项；工具版本 **unknown**；接受许可后可条件复跑。[S7] |
| openbmb/MiniCPM5-1B-Base | 官方 **1,080,632,832** | base消费量 **unknown** | 131072 | Ultra-FineWeb/L3/Math等；Apache-2.0 | 精确 base 七项 **unknown**；不能把最终 post-trained 模型成绩转给此 base；条件复跑。[S11] |
| LiquidAI/LFM2.5-350M-Base | 官方标称350M；实际总数准入核对 | base卡报告28T | 32768 | 精确语料配比 **unknown**；lfm1.0 | 混合卷积/GQA；七项 **unknown**；HF支持公开，但 likelihood adapter 必须验证，单列混合架构赛道。[S12] |
| facebook/MobileLLM-R1-360M-base / 950M-base | 官方359M / 949M | 论文 base recipe约4.2T采样、约2T unique；各权重消费账本 **unknown** | 4096，勿用最终模型32K混淆 | 高质量语料与蒸馏；FAIR非商业研究许可 | 950M base有数学/代码参考成绩，但不是七项；研究许可赛道，条件复跑。[S10] [S30] [S31] |
| TinyLlama/...step-1195k-2.5T | 名义1.1B；边界计数须核对 | 2.5T | 2048 | SlimPajama+StarCoder；Apache-2.0 | 官方七项 **53.86**；lm-eval commit **unknown**。必须纳入，不能只挑较弱最终版。[S1] |
| TinyLlama/...step-1431k-3T | 名义1.1B | 3T | 2048 | 同上 | 官方七项 **52.99**；条件复跑，不等于本协议成绩。[S1] |
| allenai/OLMo-1B | 名义1B；实际计数 **unknown** | 约3T | 2048 | Dolma；Apache-2.0 | 本报告按官方七个条目重算 **56.3886**，不是原表包含其他任务的平均；OLMo-Eval，版本 **unknown**。[S8] |
| allenai/OLMo-1B-0724-hf | 名义1B；实际计数 **unknown** | 官方 **3.05T** | 4096 | Dolma1.7；Apache-2.0 | 官方七个条目重算 **58.4143**；是更强历史参照。卡中部分退火描述针对7B，不能直接套给1B。[S9] |
| Qwen/Qwen3.5-0.8B-Base | 官方语言部分0.8B；含视觉总数 **unknown** | **unknown** | 262144 | 精确数据 **unknown**；Apache-2.0 | 扩展多模态/混合赛道；七项与适配器验证 **unknown**，绝不能引用 instruct/thinking 分数。[S13] |

**市场结论：** 50 是第一个工程里程碑，不是现代小模型领先门槛。公开参考已存在明显更高的分数，但不同协议不允许直接计算本项目落后多少、已经超过谁，或者计算严格的 token-efficiency 倍数。现代竞品训练 token、蒸馏成本和 tokenizer 不一致，须同时报告训练 FLOPs、unique/consumed tokens、teacher 成本与参数量。[S1] [S4] [S9] [S10]

## 3. 根因分析：能证明什么，怎样证伪

### 3.1 F2 / F3 的 loss 分歧已经能做定量解释

从独立 held-out JSON 重算，四个英文域平均 NLL：F2 **2.308450**，F3 **2.320517**；多语域：F2 **3.605908**，F3 **2.730973**。在五域等权目标中，F3 的多语变化单独贡献约 **−0.174987**，其余四域合计抵消 **+0.009654**，净效应约 **−0.165333**。因此 F3 的 equal-domain 优势主要来自被赋予20%权重的多语域，而英文七项不测这个收益。[R4]

这不是证明 F3“虚假改善”，也不是证明 F2 全面更好，而是目标函数不同。保留五域原始 loss，并预注册四英文域辅助指标；以后不得为了某个候选临时改变汇总权重。多语生成能力仍为 **unknown**。

### 3.2 可证伪的假设表

| 假设及可信边界 | 独立实验，不读取旧七项题面/labels | 支持 / 推翻信号 | 回滚点 |
|---|---|---|---|
| [假设] F2 的 math+code 35%挤压通用叙事、指代和初等常识覆盖 | 同 parent、相同新 corpus pool，对比 R0/R1/R2/R3；新 coreference、叙事、基础科学家族分别报数 | repair在新家族改善且高级能力保留则支持；所有组一样或更差则反对 | 返回 R0/F2，不能按 WG 某类旧错题造训练数据 |
| [假设] PDF 的教育分不等于连续正文质量；抽取噪声破坏语言/指代线索 | 最佳配比内，只替换 PDF quality stratum：原准入 vs 连贯 span。token、长度、领域、LR匹配 | 新阅读/指代改善而 NLL也不恶化则支持；只有长度变化收益需重新配对 | 退回原合格 PDF pack；不得把被过滤行静默补成其他源 |
| [假设] 学习率重新加热造成部分能力迁移/遗忘 | R3跑3e-5/6e-5/1e-4，固定 warmup和token；比较新proxy的早期/结束曲线及梯度 | 低LR改善retention但不损失增益则支持；只压低loss无能力改善则不支持 | 选择已确认 LR；历史1e-4不是已证明最优。[S22] |
| [假设] NLL改善主要来自格式、频繁token，未学到选择正确答案所需margin | 新 held-out families 同时测 NLL、正确项概率margin、acc、校准；按长度/词频分层 | NLL降但margin/acc不动说明loss代理不足 | 不用NLL作单一promotion指标 |
| [假设] BF16近似、normalization或prompt边界影响近似平局题 | 在全新可公开诊断题上，BF16/FP32 likelihood、选项前缀空格与EOS处理做一致性审计；不能挑旧七项较高实现 | 大量预测翻转说明数值/实现敏感；无翻转则不支持 | 回到冻结正确实现，不择高分实现 |
| [假设] 2K packing切断连贯性而非模型真的缺知识 | 相同文档、相同2K长度、相同tokens，比较任意切片与连续段落切片；不能同时变attention mask或ctx | 新跨句推理改善支持；同分则反对 | 回旧 pack；4K实验另立，不混到此归因中 |
| [unknown] 回退是否统计显著 | 只由独立审计者使用已归档结果作回顾性统计，不把题面/错题反馈给训练选择；新代理承担开发 | paired差异CI，而非两组独立误差条 | 若证据不足写inconclusive，不强称遗忘或修复 |

SmolLM2 的官方消融支持“不同 web 来源对应不同能力收益”，但其规模/训练阶段不同，不能据此证明本报告 R3 配比最优。DataDecide 也支持用有限预算研究数据选择，不能用其结果保证小 pilot 完全预测本项目20B能力。[S14] [S15]

**特别注意 F1：** 已提供证据中，F1的英文下游表现是 **unknown**。它仅在旧 loss 筛选中落后，而当前项目已经证明 loss winner 和 accuracy winner 不一致。因此 F1 必须重返候选，而不是永久淘汰。[R1]

## 4. 五个 CPT 候选配比与下一批数据

### 4.1 五组均按 prediction-token quota 精确归一到100%

全部从同一 parent 开始，使用相同冻结的 fresh-corpus source pool、相同最低准入过滤和两种子网格。**R0是“F2比例的新数据对照”，不是历史F2完全复现**；历史 F2 两个 checkpoint另外保留。

| recipe | FWE新文档 | DCLM | 英文PDF | Math | Code | 多语 | Synthetic | 用途 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| R0 F2 fresh control | 25 | 15 | 25 | 20 | 15 | 0 | 0 | 原 reasoning 比例控制 |
| R1 F1 reconsider | 35 | 20 | 20 | 15 | 10 | 0 | 0 | 纠正仅按loss淘汰F1的盲区 |
| R2 retention repair | 35 | 25 | 20 | 10 | 10 | 0 | 0 | 最强通用英文保留对照 |
| **R3 balanced** | **30** | **25** | **20** | **15** | **10** | **0** | **0** | **首选待验证假设，不是已知最优** |
| R4 code retention guard | 30 | 20 | 20 | 15 | 15 | 0 | 0 | 对照R3，检验DCLM↔code 5pp交换 |

R3 的细项是 FWE30、DCLM25、PDF20、FineMath11.25、InfiWebMath3.75、Python5、JavaScript2、TypeScript1、C++1、Java1，合计100。新的关键变化是把通用英文/阅读份额从F2的65%增到75%，而非继续加大“reasoning”标签来源。

### 4.2 Source、revision、质量与许可

沿用附件六个 revision，不自动切换到main；完整40位 revision写在 `research_plan.json`。公开卡只能核验来源说明，**新 shard实际bytes、许可回执、剩余可用tokens仍为unknown**，须在本地准入后才能训练。

| source / revision前缀 | 准入与 quality stratum | 许可与provenance | repeat |
|---|---|---|---|
| FineWeb-Edu `87f09149…` | 优先父训练未用、family-disjoint的新文档；保留score3广覆盖与score≥4层，不全变最高分“教材” | 官方ODC-By；同时记录URL、dump、内容hash及原始权利信息。classifier源见官方卡。[S16] | 新池每lineage≤2.0；默认旧FineWeb份额0 |
| DCLM100BT `01022d37…` | 广泛英文叙事、说明、日常因果；保留预注册fastText分层，去网页模板和重复段 | 官方ODC-By；URL/原dataset字段/内容hash；不得拿其他转载集偷偷补料。[S17] | ≤2.0 |
| FinePDFs-Edu EN `9cfabe21…` | 同时核对整篇与逐页语言；过滤抽取失败、严重截断、重复页眉；保留连续正文span | 官方ODC-By及CommonCrawl条款；canonical PDF URL、文档hash、页码/offset、抽取版本。其top10%筛选不能写成所有score≥3。[S18] | ≤2.0；PDF镜像归同family |
| FineMath / Infi `e92b25a6…` | 4plus内部75/25；按短解释、初等/中等推理、较复杂推理分层记录，不按旧benchmark模板筛 | 官方ODC-By及CC条款；source URL/hash/原始文档family；公开tokens不能当本tokenizer的tokens。[S19] | ≤2.0；层内也要检查 |
| Stack-Edu `eeec5caa…` | Py/JS/TS/C++/Java=50/20/10/10/10；官方质量阈值通常3、Java2。增加parse、minified/generated/vendor检查，不要求所有预训练代码都能独立运行 | 官方给SWHIDs，不是完整代码；按blob/repo恢复并核对detected_licenses和license_type。**unknown license排除；不能说Stack-Edu所有行已有许可**。[S20] | 每repo/fork family分组；每子源≤2.0 |
| FineWeb2-HQ `c0c06e94…` | 主线0；仅独立多语分支，按各语言quality校准 | 官方ODC-By及CC条款；语言、URL和family需保留。[S21] | ≤2.0 |

**原计划的旧FWE cap2.2不能重置。** 若以后加入旧数据，`prior_lineage_exposure + planned_exposure <= cap × unique_eligible_tokens`，而且这一平均条件不能替代每文档/子源检查。F2 warm-start要加上它已经消费的source quota。不同独立seed可共享训练集，但不能把它们的GPU消费从实验总成本中删除。

### 4.3 新数据量与可执行准入

20B R3约需FWE6B、DCLM5B、PDF4B、math3B、code2B **consumed prediction tokens**。在全新池、均匀采样、cap2且无祖先消费这一理想条件下，各类unique至少约3/2.5/2/1.5/1B，总计10B；这是必要条件，不是现有200k文档已经满足的证明。建议实际准备≥12B本tokenizer有效unique tokens以留筛选和分层余量，能否获得为 **unknown**。

流水线：固定source revision和classifier revision → 逐文档许可/provenance → exact/near/cross-source去重 → parent/benchmark family排除 → 先分train/dev/confirmation/sealed → tokenize →连续span pack →精确quota → admission receipt。`unknown`许可、family或cap不允许被填成零。source exhaustion直接停止。

去重初始实现可用normalized exact hash、canonical URL/PDF家族/code仓库与fork家族、MinHash候选加精确相似度复核；阈值先在训练语料人工校准，不用七项测试校准。污染筛查覆盖题干、选项、答案、释义以及镜像，短文本需避免common-phrase大量误杀。**筛查通过不等于证明绝对零污染；parent既有污染也不能靠只过滤CPT修复。**

Cosmopedia-v2主线继续0%。只有恢复原parent document identity、派生链、parent级去重和与测试家族隔离，才开放独立≤5%替换消融，且teacher文本必须做质量与事实验证。**排除synthetic是provenance决定，不是synthetic无效的研究结论。**

### 4.4 Curriculum

第一轮所有组static，防止同时改变比例、数据质量、LR和顺序无法归因。选定配比后才比较static与70/20/10：前70%固定广覆盖；中20%提高**各source内部**高质量层占比；末10%相同LR衰减。总source百分比、总tokens、repeat cap和长度分布保持可比。每层重复率也需预算。若只降低loss、不改善新能力代理或造成retention退化，退回static。预期收益 **unknown**。[S14] [S15]

## 5. 多保真训练矩阵、系统与磁盘

### 5.1 预算与公平性

| 阶段 | run数 | 每run有效prediction tokens / steps | 配置 / 目的 | 4卡纯训练GPUh估算 |
|---|---:|---:|---|---:|
| P0资格测试 | 先1组，含连续64步与32+32恢复对照 | 独立资格token计入总预算 | 新source快照、fixed horizon、full checkpoint、实际MFU；不测旧七项 | 取决于测试布局，**unknown**，预留≤8 GPUh而非保证 |
| P1 mixture pilot | 5 | 536,870,912 /256 | 五组共同LR6e-5、seed20260916；只淘汰崩溃/明显退化，不据微小loss下定论 | 合计18.41 |
| P1 LR pilot | 2 | 同上 | R3另跑3e-5与1e-4；R3已有6e-5 | 合计7.36 |
| P1补充LR资格 | 最多2，条件触发 | 同上 | 若胜出非R3，对该组补另两个LR；此后固定，不无界搜索 | ≤7.36 |
| P2 confirmation | 2配比×2seed=4 | 2,147,483,648 /1024 | 两个joint recipe分别锁LR；seed20260916/17；从parent重启，不假装已decay的pilot是相同horizon前缀 | 合计58.92 |
| P3 long | 1配比×2seed=2 | **20,000,538,624 /9537** | P2通过才开启；8.5899B是长horizon的中间gate，不是另一个完整结束run | 合计274.36 |
| P3独立seed | 1，条件触发 | 同long | seed20260918在recipe冻结后启用，不能择优丢弃 | 137.18 |

P1+两次补充+P2+两seed long约 **366.4 GPUh**纯训练；预留25%非训练/吞吐浮动后约458 GPUh，再另计资格测试、代理评测、数据过滤、teacher生成、失败重跑和磁盘IO。不是固定总费用承诺。8卡可并行两个4卡seed；这通常比把8卡并成单run更适合本轮实验吞吐。[R1]（按历史速度计算）

### 5.2 4 / 8 / 16 卡单run速度换算

| 每run tokens | 4卡 /162k tps | 8卡 /290,323 tps | 16卡 /467,716 tps |
|---|---:|---:|---:|
| 0.537B | 0.92h | 0.51h | 0.32h |
| 2.147B | 3.68h | 2.05h | 1.28h |
| 8.590B | 14.73h | 8.22h | 5.10h |
| 20.00054B | **34.29h** | **19.14h** | **11.88h** |

这些是历史吞吐外推，**新配比实测吞吐unknown**。16卡wall-clock快，但每20B约190.05 GPUh，劣于4卡137.18 GPUh。两组4卡总吞吐约324k tps是理想并发算术，真实共享IO与节点隔离须测，不能保证优于单8卡。[R1]

### 5.3 LR、batch、checkpoint

首选待确认LR **6e-5**，grid **3e-5/6e-5/1e-4**；warmup **67,108,864 tokens=32 steps**，对全部新组一致。继续使用fused AdamW、β=(0.9,0.95)、eps1e-8、wd0.1、clip1.0；这些是控制变量，不是声称已找到最优。历史8-step warmup保留在历史复现记录，不混称新试验完全复现。[R3]

固定global2,097,152 tokens/step，micro4时4/8/16卡分别acc64/32/16；不靠扩大batch悄悄改变优化问题。只有在新固定horizon确认后才允许wd0.01对照，不能同时再扫β、batch、curriculum。默认最后10% cosine趋近0，保持当前runner实际实现；不要把每个分段当新warmup/decay。[R3]

full checkpoint每128步（4卡历史速度约27.6分钟）保存；五域masked loss每256步及段末；外部能力proxy只在预注册里程碑评估。P3可在2.147/8.590B检查趋势，但对照horizon必须相同。长run中途stable checkpoint不等于末尾退火checkpoint；需要decay probe时，从副本分支另记token/成本，不污染主线cursor。

### 5.4 MFU严格门槛

按当前deep模型与冻结仓库公式，2K每processed token约 **3,989,975,040 FLOPs**。历史协议使用每5090 **209.5 TFLOP/s dense BF16 / FP32 accumulation**分母；本审计成功核验了仓库取值，独立NVIDIA官方硬件文档复核仍为 **unknown**。不得换用FP4稀疏AI TOPS或GPU utilization百分比。[R2] [R3] [R7]

`MFU = tokens_per_second × FLOPs_per_token / (GPU数 × 209.5e12)`。同口径MFU>0.50要求4/8/16卡分别严格高于约 **105,013 /210,026 /420,053 tps**。4/8卡有较大历史余量；16卡历史0.5567余量有限。换源、checkpoint策略或kernel后均须重新资格验证。

默认沿用已恢复验证的DDP布局、native GQA、compile、gradient_as_bucket_view与25MB bucket；不要未经测试改static_graph/find_unused_parameters或NCCL拓扑策略。固定GPU UUID、独占节点、CPU/NUMA亲和，预tokenize并本地预取，去掉训练critical path上的在线解压/classifier/网络拉取。显存峰值、电源/温度、IO等待与通信时间一起记录。

**执行约束而非保证硬件永不变慢：** grace后每10-step rolling median必须>0.50；任何失败run不准promotion，保留失败窗口、停止原因与checkpoint。跨launch不能只看最后10步，更不能利用反复重启重置grace掩盖慢段；增加全轨迹及跨段审计。原runner按每launch维护MFU样本，因此此跨段审计是额外必要工作。新数据的“实际通过”在测量前是 **unknown**。[R3]

同时报告包含validation、checkpoint和重编译的end-to-end有效tokens/s。MFU窗口通过不等于任务墙钟效率优良。

### 5.5 磁盘与恢复

| 类别 | 可审计口径 | 20B规划值，非实测 |
|---|---|---|
| 原始/清洗语料 | 从1%样本测compressed-byte与本tokenizer有效token比 | **unknown**；临时预留100–200GB，不满足则分批stream，不全量缓存 |
| packed shard | 实测dtype与token槽位数；uint32时4bytes/槽，uint16时2 | 若10–12B unique且uint32，约40–48GB；若物化20B消费序列，约80GB；两版并存约80–160GB |
| full optimizer checkpoint | FP32 weights+Adam两moment仅参数主项约6.345GB，另有buffers/RNG/reader等 | 每份总bytes **unknown**；首次full save实测。预留2个保留版本+1个atomic写入slot |
| 历史model-only | 仓库记录每份约2.3725GB | 每seed保留parent/F2/最终与必要里程碑；不能当BF16纯权重尺寸 |
| HF export | BF16参数主项约1.0575GB，另有格式/配置/可能buffers | 建议每active run预留3–5GB；实测receipt为准 |
| 竞品cache / sample logs / compile cache | 逐模型测、按hash共享、日志压缩 | 临时预留30–60GB；raw logits全词表不得默认保存 |

建议第一轮共享工作盘先保证约 **400–600GB** 可用、每active run单独保留 **≥50GiB** checkpoint/export空间；这是带假设的容量规划，不是数据集已知大小。扩到50/100B前重算；raw/token比 **unknown** 时不得无条件批准下载。

原runner的full模式初始磁盘检查以init checkpoint文件大小估计；若从model-only初始化，它不能可靠代表更大的full optimizer checkpoint。外层应按实测full bytes×3+exports+安全余量做额外检查。[R3]

只删除已写入receipt且非保留对象的文件：dry-run清单 → 路径realpath在允许根内 → 拒绝symlink越界 → UID/ownership → Slurm active job及跨节点进程引用检查 → 验证替代副本hash → 人工批准 → 删除并记receipt。不得按文件名猜“旧”就rm。交付的launcher不执行任何清理。

## 6. 新开发集、独立确认与封存评测

**旧七项已经看过，不能被重新命名成“未见测试集”。** 今后不根据其题面、label、单题margin、错误类别细节、prompt改写结果选择数据、LR、checkpoint或merge系数。已知宏观退化用于提出一次新研究问题，但不是无限使用旧答案的许可证。

建议四层：T0旧七项，仅最后历史审计；D新proxy，可开发；C新confirmation，冻结候选后一次选择确认；S新sealed，独立保管、最终一次揭盲。开发者只获得S的任务规格、统计计划和污染排除接口，不读内容；污染工具由保管者返回需剔除的训练family IDs，不泄露题面。若无法实现角色隔离，封存强度 **unknown**，报告必须写明。

D至少五个英文能力家族：指代/跨句一致性、叙事/日常因果、初等科学事实与推理、passage-grounded阅读、困难多选/可靠性。每家族先准备约2,000个独立题/簇作为开发起点；C/S按功效计算增加，不拿2,000当充分统计保证。公共CB/WiC/COPA/SciQ等只能作辅助参考；数据许可、split、潜在训练暴露核验前为 **unknown**，新私有题不能只是旧七项paraphrase。

### 6.1 统计与promotion规则

所有model按同一item ID对齐；家族内对独立document/problem/repository family进行paired cluster bootstrap，**10,000**次，每次对所有模型/seed重用同样采样索引。先算各任务accuracy，再任务等权；不能按样本数把HellaSwag权重偷偷放大。两训练seed对同一道题不是两道独立题；报告条件于固定seed的题目CI，seed均值/范围另报。仅2seed不足以精确估计训练随机性总体方差。

开发选择：先满足各关键家族相对parent的non-inferiority，再比较worst seed代理分数，最后看均值。建议margin **1.0 pp**；保留F2已有优势的家族同时对F2作NI约束。多家族使用预注册Holm检验或保守Bonferroni simultaneous lower bounds，不拿七个未校正95% CI当联合95%结论。

功效示例是计算假设，不是现有数据事实：若paired discordance q=0.10、NI margin δ=0.01、5个家族Bonferroni单侧α=0.05/5、power80%，独立样本近似 `n ≈ (z(0.99)+z(0.8))² × q/δ² ≈ 10,000/家族`。若family内相关，需要design-effect上调。真实q、ICC、最终所需样本为 **unknown**，先用D估计再冻结S规模。CI太宽是**inconclusive/HOLD**，不是模型无效，也不是自动通过。

### 6.2 唯一一次final何时允许

必须先冻结：候选recipe、LR、全部seed、固定最终step、checkpoint hashes、运行代码/环境、tokenizer、D/C/S和污染回执、competitor registry和所有pins、统计计划、NI margins、任务权重。D/C达到预注册实用阈值（建议代理aggregate≥+0.5 pp且CI lower>0，所有关键家族NI通过）后才可申请final。阈值是预算决策门槛，收益预测仍为 **unknown**。

final将parent、F2两个seed、新候选全部预注册seed与全体准入竞品一起跑。T0和S都一次揭盲。T0仍称**已暴露历史suite的锁定复跑**；其bootstrap不能消除过去adaptive选择偏差。真正新增的泛化证据来自S。

失败后发布失败，不继续拿同一S调参。新一轮研究必须新建独立seal、重新登记研究cycle；不能挑“更友好”的现有suite替换失败结果。

## 7. >50 的 go / no-go 决策树

| 阶段 | 预期七项增益范围 / 置信度 | 量化决策，不是预测 | 失败与停止 |
|---|---|---|---|
| P0审计 | **unknown**；主要减少无效结论风险 | 实际deep checkpoint/schema、source可用量、exact segmented recovery、MFU、eval custody全PASS | 任一unknown未解决不启动正式训练 |
| 0.537B pilot | **unknown**；不能把loss差转成pp | 排除非有限loss、明显新proxy退化或cap/MFU失败；选最多2组继续 | 无可用组则修数据/系统，不扩大tokens |
| 2.147B两seed | **unknown** | 新proxy aggregate建议≥+0.5pp、paired lower>0；关键家族NI≥−1pp；两个seed方向一致 | gain微小且CI宽：HOLD补确认样本；明确无增益或退化：只允许一次预注册LR/quality救援 |
| 8.590B长horizon检查点 | **unknown** | 同horizon的2.147→8.590B proxy能力趋势为正、关键NI不破坏；检查是否只是LM loss改善 | 两seed都只有loss下降、能力停滞：不自动批准后续50/100/300B |
| 20.00054B结束 | **unknown** | recipe固定final通过D/C后才开S/T0；不按旧七项最优挑checkpoint | 未过D/C则不浪费final；单次救援亦失败则停止该CPT路线 |
| 锁定final | 新候选真实分数 **unknown** | 两seed七项各自>50、两seed均值>50，附条件采样CI；新S正增益与家族NI同时通过；建议追加冻结第三seed才称更强稳定性 | 任一seed≤50则“均值50+”不能说两seed稳定50+；S失败不能说泛化确认 |
| 竞品超越 | **unknown** | 对预登记集合所有成员进行同时控制的paired检验，详见第8节 | 只胜旧模型便只点名旧模型，不能宣传同规模全面领先 |

要求旧七项aggregate采样LCB也>50可以作为更严格工程门槛，但必须注明该CI仍不能撤销历史adaptive bias。**50.01不等于有可靠余量；所需余量由实际误差决定，unknown，不能预先保证“51就够”。**

### 停止CPT与下一代from-scratch的触发

不是因为“529M一定到顶”而停止；其理论/经验能力上限 **unknown**。建议在以下组合出现时停止**当前路线**：两个独立fresh tranche都无新proxy能力进展；一次受控LR/quality救援无效；学习曲线只剩NLL改善；可用合法新数据不足且必须越过repeat cap；或每次修复通用能力都不可接受地牺牲已验证能力。

下一代再比较tokenizer、depth/width、GQA/head_dim、context设计、数据与训练horizon；保持总参数和训练FLOPs约束，从小规模资格与统一新suite开始。不能在当前checkpoint上直接改变层数、宽度、vocab或embedding tied策略，却仍叫同一CPT。当前26层深模型并不因文档写错就自动需要替换。

## 8. 超过竞品的 exact matched rerun

冻结历史 protocol ID `p500m-english-base-v5`，protocol SHA `a71ce3c8cc85cac6fe6751cb46f7e50dfb78d266602d6377cb2e65249f11c761`。已核验实际启动参数：HF backend、BF16、max_length2048、zero-shot、2GPU、`--seed 42,42,42,1234`、`--log_samples --show_config`，没有chat template。[R5] [R6]

| task | 样本数 | 必须使用的metric |
|---|---:|---|
| hellaswag | 10,042 | acc_norm |
| piqa | 1,838 | **acc_norm** |
| winogrande | 1,267 | acc |
| openbookqa | 500 | acc_norm |
| arc_easy | 2,376 | acc_norm |
| arc_challenge | 1,172 | acc_norm |
| boolq | 3,270 | acc |
| 总计 / aggregate | **20,465** | 七项等权，不按样本量加权 |

**尚未完整回收的lm-eval commit、datasets/transformers版本、task源码hash、样本缓存hash为unknown。** 现有环境中的匹配记录可信度不等于第三方已经可以无条件exact复现。先从冻结runtime、pip freeze、源码目录、eval-cache-manifest及results/show_config恢复，不直接装“最新版”冒充旧协议。[R6] [S27]

每个竞品先下载固定revision、检查base阶段、计数参数、核验许可证和tokenizer文件hash。保持相同可见UTF-8 prompt、题目/选项与选择规则；每模型使用原生tokenizer及合规BOS/EOS策略，不能强制套F2 tokenizer。特殊token处理、首token前缀、padding、truncation与normalizer必须输出trace。normalization直接调用冻结task实现，不自行换成按token数除。

2K是共同上限，不给长context模型额外材料。先检查每一prompt+choice在所有tokenizer下是否都放得下；若有超长，预注册统一可见文本截断与统计完整性规则，并同时报告旧协议full set和共同可容纳subset。禁止看到结果后丢掉难题。预计本任务多数prompt很短，但实际overlength计数 **unknown**。

新混合/VLM架构可能需更新Transformers/harness。此时建立**v6桥接协议**，把parent、F2和旧竞品也重跑，不把v5、v6分数混为一个榜。新adapter先在独立toy文本上对齐full-sequence teacher-forced loglikelihood、batch1/batchN、KV cache状态、padding和FP32求和；若未通过则标“not admitted / unknown”，不能用聊天API或生成A/B概率替代。

统计上对同题配对，候选两seed均值相对每个已冻结competitor做paired family bootstrap，控制全体比较的多重检验；报告每task、aggregate、NI、seed差异、CI和失败模型。若宣称胜过整个集合，**所有预登记比较均须成立**，不是仅胜过平均竞品或挑弱模型。registry不得因为某模型太强就事后删除。

允许措辞：“在protocol X、明确checkpoint集合Y、共同2K零样本七项下，本checkpoint两seed均值为Z，相对A/B的paired CI为…；新sealed suite结果为…。”

禁止措辞：“超过所有0.5B/1B模型”“SOTA”而不定义集合；“50+所以超过现代强base”；“与instruct/thinking在不同shot数对比也算base领先”；“token少100倍所以严格效率高100倍”；“从未见测试”指旧七项；“显著提升”只有点估计没有适用检验。

## 9. base通过后的post-training顺序、预算与停止条件

顺序：**冻结base release → 可选独立2K→4K → SFT → 一种preference方法 → 小规模RLVR**。Domain-CPT若改变主能力，应在SFT前分叉。SFT后分数不能冒充base分数。Tülu3提供了可复现阶段化范式，但主要是更大模型，不能把其收益量迁移给529M。[S23]

### 9.1 Context extension单独实验

有明确需求才做4K。比较RoPE position interpolation / YaRN小倍率方案，保存原2K分支；从完整长文档重新pack，建议50%短context replay、50%长文档训练tokens。Pilot 0.537B，必要时2.147B两seed确认；LR3e-5与6e-5候选，其他优化器先固定。**预算与比例均为建议，最优值unknown**。[S25]

测1K/2K/4K位置扫掠、needle之外的多跳/聚合与完整长文阅读，短context能力NI保持≤1pp损失；不以一个检索例子宣布4K能力。RULER说明“能检索needle”不等于真实有效长context。[S26] 当前CPT runner固定2K，没有可直接传入的`--context-length 4096`；修改RoPE/packing/model buffers/恢复指纹后必须独立资格与MFU验证。8K在4K确认前no-go。

### 9.2 SFT

先50k样本pilot，assistant target budget **10–20M**；正式起点约150k样本、**60M assistant targets**，不是固定“每样本400token”的事实。实际去重后长度分布决定样本数；同时记录全部processed tokens。通常1epoch，上限2，按family验证控制，不因为loss仍降而无限repeat。

| assistant target类别 | 配比 | 数据实现 |
|---|---:|---|
| 通用指令/对话/有依据知识 | 35% | 经逐例筛选的官方SmolTalk/Tülu类候选池或有授权自建对话；公开池的具体revision/teacher条款/provenance未核验前是unknown，不默认准入 |
| 简短可验证数学 | 20% | 独立problem family、可执行答案核验；先教会短推理，再增加长度；不从旧七项改题 |
| 经过测试的代码与解释 | 20% | 合法repo/函数family；隐藏单元测试、随机性质测试；解释须与可执行程序一致 |
| 结构化输出/工具 | 15% | 自有schema与工具模拟器，状态转移/参数/权限验证；不能只验证JSON能parse |
| 多语 | 5% | 独立语言组核验与双语质量抽检；10%多语是另一个受控产品实验，不默认挤压英文 |
| 校准/安全 | 5% | 可回答/不可回答成对、证据不足、恶意工具输出、正常敏感请求；同时控制过拒绝 |

LR试 **5e-6/1e-5/2e-5**，warmup3%，短pilot先固定wd0.01、clip1；这不是当前CPT runner现成支持的assistant masking配置。需支持turn-aware assistant loss mask、EOS正确监督、packing样本隔离的独立SFT实现。长答案要完整容纳、重新分段成语义完整turn或拒收，不能把被截掉的答案尾部当负面模型表现。

准入：teacher checkpoint/revision、生成参数、prompt与parent provenance、校验器版本都保留；teacher自评不是事实验证。安全/许可未知的数据不因“质量高”就自动通过。**SFT带来多少七项或真实对话改善unknown。**

SFT gate：新指令遵循、答案正确率与格式执行必须改善；对应base代理NI；校准评估Brier/log-loss及risk-coverage而不只ECE；良性拒答率不得比控制增加超过预注册2pp；安全攻击成功率不得显著上升。绝对安全阈值由产品场景另定，目前baseline与可行阈值均 **unknown**。失败回滚到base或已通过SFT，不用preference掩盖缺失基础能力。

### 9.3 Preference optimization

只选一个默认方法：**DPO**。先10k pair，正式≤30k pair；两候选回答的assistant targets合计预算 **10–30M**，实际长度unknown。每pair需有可核查偏好依据，去掉正确性未定、两个都错、只靠长短差异的pairs。

LR候选 **5e-7/1e-6/2e-6**，β候选 **0.05/0.1**，冻结reference为已通过SFT；从小2×2或顺序bounded grid开始，不跑全排列无界挑选。IPO/ORPO只作有明确失败假设的替代，**不把DPO→IPO→ORPO全串起来**。单epoch起步；win rate需长度控制与人工/验证器复核。准确率、校准、benign refusal、base retention或语言质量退化则回SFT。效应 **unknown**。

### 9.4 RLVR

先有非零、可重复成功再做。使用约5k独立problem families，难度分层以已冻结SFT策略pass@1约10–60%为初始目标区间；这是挑可学任务的操作标准，不是预测模型已经达到。每prompt4–8 rollouts，最大output约512–1024按任务限制，试验总**generated-token cap 8–20M**，另记参与optimizer的tokens与prompt开销；达到cap即停，不能只按“训练步很少”隐去rollout成本。

数学用受限表达式/符号等价与随机数值复核，代码用断网沙箱、CPU/内存/超时约束、隐藏测试与property tests，工具用状态机验证。不能执行不受控字符串eval，不能由同一teacher出题又单独担任真值裁判。奖励以真实正确性为主，解析失败不能误判正确，单独格式分不得压过错误答案。

LR起点范围 **5e-7–2e-6**，较保守clipping/KL预算由pilot冻结；确切最佳系数 **unknown**。必须同时报告pass@1、pass@k、长度、KL、all-zero/all-one groups、verifier误接收率、OOD family与基础retention。若all-zero groups>80%是预注册停止信号之一：回更容易题与teaching data，不无限加rollouts。发现reward exploit立即停止、修验证器并重新登记试验，不把修前分数当有效成绩。

现有RLVR负面研究提醒：pass@1提高可能主要是更高效采样已有解法，不能仅据此宣布扩展推理能力上限；这不是“任何RL都绝不可能创造新能力”的定理。小模型蒸馏/推理训练也有积极官方结果，但规模、数据和预算不等于本项目。[S24] [S10]

**MFU边界：** 不能把CPT的MFU直接套到自回归rollout。SFT/DPO也要以全部实际processed tokens/操作计算FLOPs，不能只用assistant目标tokens作吞吐分子。若硬要求覆盖“端到端RLVR（含生成）始终MFU>0.50”，可达性 **unknown**，RLVR在实测资格前no-go；不得只报告teacher-forcing更新kernel的MFU来冒充整个RL阶段通过。

## 10. 五条domain-specific路线与能力上限

这些都是独立分支，不同时混进英文base冠军声明。每个数值化能力ceiling当前均为 **unknown**；有效上限只能在定义好的任务分布、context、采样预算、工具权限和污染规则下测量。

| 方向 | 最先做的数据与预算 | 可执行验证 / 实用边界 | 停止条件 |
|---|---|---|---|
| Math | base gate后，可选0.537B domain-CPT pilot，约50%通用replay+50%合格数学；≤2.147B确认；再短解SFT | 自动可校验算术/代数/文字问题，按family和难度留出，pass@1/pass@8及长短解分别测；公开GSM/MATH只能另作污染说明后的参考 | 新难度成功为零、只是更长思考或OOD退化则回teaching，不承诺AIME水平 |
| Code | 可选0.537B→2.147B，50%通用replay+50%许可代码；先Py函数与小程序，再仓库修改 | 代码编译、隐藏单测、变异/性质测试，repo/fork与时间划分；HumanEval/MBPP等只做锁定额外测试；pass@1与cost一起报 | verifier漏洞、测试污染、代码格式记忆无执行收益即停止；不能由LM loss宣布coding强 |
| Knowledge | 可选0.537B→2.147B，50%replay+50%干净知识材料；有明确新知识目标才做 | 闭卷事实与给证据阅读分开，按文档/时间家族留出，引用支持率与拒答校准；RAG单列系统路线 | 事实幻觉上升、只在来源复述中好则不宣传世界知识超越；动态知识优先外部检索 |
| Multilingual | 保留F3；单独以10–20%目标语言份额+英文replay做0.537B pilot，最佳后2.147B确认 | 按每语言报告阅读、生成、翻译、代码切换与英文retention；测token inflation与上下文利用 | NLL改善但生成/语义一致性不改，或少数语言完全失效，则不称“支持多语”；不能用英文aggregate掩盖 |
| Tool use | 优先SFT，而非先大规模CPT；20k–50k状态机任务、5–15M assistant targets上限 | JSON Schema、参数语义、工具可用性、拒绝非法权限、观察结果后的下一步；执行成功率而不只是格式率 | 可解析却参数错误、伪造工具结果、越权、循环调用则no-go；不可拿静态MC accuracy推断agent能力 |

上述预算是待确认实验上限，吞吐、gain与最终ceiling均 **unknown**。每条分支须对未改动的base/已通过instruct保留NI检查，并分别发布失效案例。最具现实价值的产品定位可以是“小型、可验证、受工具约束的专用模型”，但是否胜过本表竞品同任务必须另行matched验证。

## 11. 优先级、前72小时与后2–6周

按照预期信息/能力收益、GPU小时与风险排序；收益幅度未知，因此不用虚构“每GPUh涨多少分”。

| 优先级 | 工作 | GPU负担 | 收益判断 / 风险 |
|---|---|---|---|
| P0 | 修正架构与token口径，锁runtime/源hash，补新evaluation custody与repeat账本 | 低，GPU资格另计 | 避免把无效实验当进步；必须先做 |
| P1 | 扩充合法fresh英文/FWE/DCLM/PDF，重建proxy；复活F1与retention组 | 主要CPU/IO/标注；训练约25.8 GPUh初筛 | 与现证据最直接相关；数据瓶颈真实存在 |
| P1 | bounded LR probe、两个recipe×两seed确认 | ~58.9 GPUh确认，补probe≤7.4 | 区分偶然与可复制；不读旧七项 |
| P2 | 小规模parent-F2插值旁路 | 上限≤1 GPUh预留；实测unknown | 廉价但不应抢主线精力；共享权重不保证低loss连通 |
| P2 | winner两seed20B fixed-horizon | ~274.4 GPUh训练外推 | 只在D/C门槛通过后；绝不自动开50/100/300B |
| P3 | 独立4K、SFT、偏好和RLVR | 独立预算；当前实测unknown | base gate后按实际产品需求启动，不一次堆满 |
| P4 | 16GPU扩展、from-scratch新架构 | 高风险独立项目 | 当前质量瓶颈不应优先靠卡数或架构掩盖 |

**前72小时（工作顺序，不保证完成日期）：** 0–12h整理checkpoint/schema/runtime/metrics口径；并行审计新source容量、许可证及评测隔离。12–36h完成独立proxy第一批、fresh pack、P0恢复和MFU资格，满足后开始7个pilot。36–72h只在数据/评测已就绪时推进最多2组两seedconfirmation。若标注/许可/容量为unknown，停在相应gate；不要为赶72h拿旧七项代替新dev。

**第2周：** 完成确认与一次受控quality/curriculum假设验证；选一个joint recipe，或明确no-go。**第3–4周：** winner两seed20B、轨迹审计、第三seed资格；冻结registry与所有checkpoint。**第5–6周：** 仅达到门槛才开一次S/T0/竞品final、发布正负结果和model card；base达标后开始4K或SFT pilot。该顺序取决于GPU可用性、数据合法可用量和标注吞吐，实际周期 **unknown**。

## 12. 最终推荐manifest与训练参数

可直接使用 `research_plan.json` 作为**研究计划源文件**，`audit_plan.py`生成精确block quota；它明确不是已经含shard paths/hashes的packed manifest，也不会伪造corpus admission。`R3_20B_quota_plan.json`是已经本地计算的配额，不是已训练模型。

```json
{
  "status": "PROPOSED_NOT_TRAINED",
  "formal_promotion": false,
  "initialization": "immutable_parent",
  "architecture": "deep: 26x1280, ffn3584, Q20/KV5, head64",
  "parameters": 528748800,
  "context": 2048,
  "recipe_hypothesis": "R3_balanced",
  "prediction_token_percent": {
    "fineweb_edu_fresh": 30,
    "dclm": 25,
    "finepdfs_english": 20,
    "finemath_4plus": 11.25,
    "infiwebmath_4plus": 3.75,
    "code_python": 5,
    "code_javascript": 2,
    "code_typescript": 1,
    "code_cpp": 1,
    "code_java": 1
  },
  "multilingual_percent": 0,
  "synthetic_percent": 0,
  "cpt_target_prediction_tokens": 20000538624,
  "global_tokens_per_step": 2097152,
  "steps": 9537,
  "peak_lr_hypothesis": 0.00006,
  "warmup_tokens": 67108864,
  "seed_grid": [20260916, 20260917],
  "audit_seed": 20260918,
  "ready_for_training": false,
  "remaining_blockers": "see full JSON; unknown fields are hard gates"
}
```

从parent启动该20B run，最终累计tokens约 **35,999,711,232**；若另行批准F2 warm-start再训练20B，则约 **38,147,194,880**。这两个初始化与训练成本不同，不能混成同一个数据效率结果。

### 12.1 与真实runner对应的参数表

| 真实参数 | 推荐初始值 | 说明 |
|---|---|---|
| `--architecture` | `deep` | 与运行脚本/参数计数匹配；binary形状仍要确认 |
| `--target-tokens` | pilot536870912 / confirm2147483648 / long20000538624 | 每run的全局horizon；resume不得改变 |
| `--pilot-steps` | pilot256 / confirm1024 / long每段≤2048 | long9537步分为2048×4+1345；不是每段新schedule |
| `--microbatch --accumulation` | `4 64` | 默认单node4GPU；global2Mi tokens |
| `--peak-lr` | `6e-5`待pilot确认 | 控制grid3e-5/6e-5/1e-4 |
| `--warmup-tokens` | `67108864` | 32 global steps |
| `--save-every --validate-every` | `128 256` | full resume保存，外部proxy另调度 |
| `--checkpoint-mode` | `full` | 原model-only-final不能exact resume |
| `--dense-bf16-tflops-per-gpu` | `209.5` | 保持历史MFU口径与其未独立核验边界 |
| `--min-mfu --mfu-window --mfu-grace-steps` | `0.50 10 5` | ≤0.50失败；跨launch额外审计 |
| `--deterministic --validate-before-training` | 开启 | 与指纹一起固定；不得只在某些seed开启 |
| `--mixture-manifest` / validation / admission / protocol | 新准入文件和实际SHA | 全部unknown未解决前no-go；五域masked validation不是旧七项 |
| 新启动 / 续段 | `--init-model-checkpoint --expected-init-sha256` / `--resume` | 二者互斥；不传不存在的resume重置参数 |

### 12.2 本地执行入口

```bash
# 无GPU：校验计划并生成精确配额。此文件不是packed manifest。
python audit_plan.py --tokens 20000538624 --output my_R3_20B_quotas.json

# 填好真实unique/prior exposure后强制检查预算，unknown会返回非零退出码。
python audit_plan.py --budget source_budget.json --require-budget

# 仅对自己信任且hash已固定的checkpoint执行shape/alias核验；不会宣称export parity通过。
python audit_checkpoint.py \
  --model-source "$REPO/pretrain500m/model.py" \
  --checkpoint "$PARENT" \
  --expected-sha256 13aa21721e15c48cdfdafe30d8fdd41d9c95c90be766327661d96af1e90dd6cf \
  --trusted-own-checkpoint

# 默认只打印实际runner命令；路径必须真实存在并与固定commit一致。
python launch_segmented.py \
  --repo "$REPO" --init-checkpoint "$PARENT" \
  --train-manifest "$TRAIN_MANIFEST" \
  --validation-manifest "$VAL_MANIFEST" --validation-tokens "$VAL_PACKED_BUDGET" \
  --admission-receipt "$ADMISSION" --protocol-file "$PROTOCOL" \
  --output-dir "$OUT" --tokens 536870912 --seed 20260916
```

正式执行增加 `--execute --preflight-receipt <独立产生的资格回执>`。该回执必须将所有PASS证据绑定到真实输入hash和run参数；交付包不会生成虚假的PASS。长run把tokens改为20000538624，固定horizon自动分段，任何错误不自动绕过。wrapper只支持默认**单node4GPU**；8/16GPU的多节点launcher须沿用项目已经资格验证的Slurm/torchrun布局，不把单机nproc=8当两节点。

**本会话已完成：** 附件阅读、一手检索、固定commit关键源码与结果审计、参数/配额/预算计算、脚本语法和无GPU逻辑测试。**unknown / 未执行：** checkpoint二进制校验、真实数据下载和准入、新proxy/seal建立、任何GPU训练或竞品matched rerun。没有修改远端GitHub内容，没有删除用户文件，没有把未验证方案写成“已到50”。

---

## 关键unknown登记

| unknown | 消除方式 | 阻塞的claim |
|---|---|---|
| 新配比对旧七项的真实收益、到50概率、领域能力上限 | 新dev/confirmation选择后一次冻结final | 任何“保证50”“保证超越”的说法 |
| checkpoint实际schema与数值export parity | hash→tensor audit→native/HF新文本logit gate | 直接开始训练与严谨参数声明 |
| 新source unique、剩余额度、细粒度许可证与provenance | shard/row/lineage admission receipt | 20B可实际执行、商业许可/无污染等过强主张 |
| 历史harness及所有依赖精确版本 | 冻结runtime、results metadata、cache manifest回收 | “第三方exact复跑已经完备” |
| competitor revision/权重hash/实际参数/adapter pass | 逐个下载固定revision并做load/likelihood资格 | strict集合最终成员与matched排行 |
| 新seal的独立性、q/ICC与所需样本 | 独立保管、D上功效估计、预注册 | 真正新泛化与联合NI确认 |
| 新数据上MFU、checkpoint峰值、4/8/16性能 | P0资格、全轨迹日志、首次full-save实测 | 未来实际MFU>0.50已经保证 |
| RLVR rollout含生成的端到端MFU | 独立全阶段计量 | RL阶段满足同一硬效率门槛 |

## 一手来源与固定仓库证据

这些链接用于核验原文；官方卡中的数字不自动成为本项目matched成绩。访问失败或没有读取完整内容的字段，在正文保留unknown。

[R1]: https://github.com/yinli-systems/l20-1b-pretraining/blob/eb5eded979ba148b3dc06052d436eabc583bf3fc/pretrain500m/RESULTS.md
[R2]: https://github.com/yinli-systems/l20-1b-pretraining/blob/eb5eded979ba148b3dc06052d436eabc583bf3fc/pretrain500m/model.py
[R3]: https://github.com/yinli-systems/l20-1b-pretraining/blob/eb5eded979ba148b3dc06052d436eabc583bf3fc/pretrain500m/posttrain/cpt-confirmation-v1/train_cpt.py
[R4]: https://github.com/yinli-systems/l20-1b-pretraining/blob/eb5eded979ba148b3dc06052d436eabc583bf3fc/pretrain500m/reports/long-confirmation-heldout-results-1590660.json
[R5]: https://github.com/yinli-systems/l20-1b-pretraining/blob/eb5eded979ba148b3dc06052d436eabc583bf3fc/pretrain500m/reports/seven-task-confirmation-2gpu-results-1590907.json
[R6]: https://github.com/yinli-systems/l20-1b-pretraining/blob/eb5eded979ba148b3dc06052d436eabc583bf3fc/pretrain500m/posttrain/seven-task-confirmation-2gpu-v1/run.sbatch
[S1]: https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T
[S2]: https://huggingface.co/EleutherAI/pythia-1b
[S3]: https://huggingface.co/keeeeenw/MicroLlama
[S4]: https://huggingface.co/HuggingFaceTB/SmolLM2-360M
[S5]: https://huggingface.co/Qwen/Qwen2.5-0.5B
[S6]: https://huggingface.co/Qwen/Qwen3-0.6B-Base
[S7]: https://ai.google.dev/gemma/docs/core/model_card_3
[S8]: https://huggingface.co/allenai/OLMo-1B
[S9]: https://huggingface.co/allenai/OLMo-1B-0724-hf
[S10]: https://arxiv.org/abs/2509.24945
[S11]: https://huggingface.co/openbmb/MiniCPM5-1B-Base
[S12]: https://huggingface.co/LiquidAI/LFM2.5-350M-Base
[S13]: https://huggingface.co/Qwen/Qwen3.5-0.8B-Base
[S14]: https://arxiv.org/html/2502.02737v1
[S15]: https://arxiv.org/abs/2504.11393
[S16]: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
[S17]: https://huggingface.co/datasets/HuggingFaceFW/dclm_100BT
[S18]: https://huggingface.co/datasets/HuggingFaceFW/finepdfs-edu
[S19]: https://huggingface.co/datasets/HuggingFaceTB/finemath
[S20]: https://huggingface.co/datasets/HuggingFaceTB/stack-edu
[S21]: https://huggingface.co/datasets/epfml/FineWeb2-HQ
[S22]: https://arxiv.org/abs/2308.04014
[S23]: https://arxiv.org/abs/2411.15124
[S24]: https://arxiv.org/abs/2504.13837
[S25]: https://arxiv.org/abs/2309.00071
[S26]: https://arxiv.org/abs/2404.06654
[S27]: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md
[S28]: https://arxiv.org/abs/2203.05482

[R7]: https://github.com/yinli-systems/l20-1b-pretraining/blob/eb5eded979ba148b3dc06052d436eabc583bf3fc/pretrain500m/posttrain/confirmation-runner-v1/run.sbatch

[S29]: https://huggingface.co/HuggingFaceTB/SmolLM2-360M/blob/main/config.json

[S30]: https://huggingface.co/facebook/MobileLLM-R1-360M-base

[S31]: https://huggingface.co/facebook/MobileLLM-R1-950M-base

### 可见来源索引

- **R1** — [https://github.com/yinli-systems/l20-1b-pretraining/blob/eb5eded979ba148b3dc06052d436eabc583bf3fc/pretrain500m/RESULTS.md](https://github.com/yinli-systems/l20-1b-pretraining/blob/eb5eded979ba148b3dc06052d436eabc583bf3fc/pretrain500m/RESULTS.md)
- **R2** — [https://github.com/yinli-systems/l20-1b-pretraining/blob/eb5eded979ba148b3dc06052d436eabc583bf3fc/pretrain500m/model.py](https://github.com/yinli-systems/l20-1b-pretraining/blob/eb5eded979ba148b3dc06052d436eabc583bf3fc/pretrain500m/model.py)
- **R3** — [https://github.com/yinli-systems/l20-1b-pretraining/blob/eb5eded979ba148b3dc06052d436eabc583bf3fc/pretrain500m/posttrain/cpt-confirmation-v1/train_cpt.py](https://github.com/yinli-systems/l20-1b-pretraining/blob/eb5eded979ba148b3dc06052d436eabc583bf3fc/pretrain500m/posttrain/cpt-confirmation-v1/train_cpt.py)
- **R4** — [https://github.com/yinli-systems/l20-1b-pretraining/blob/eb5eded979ba148b3dc06052d436eabc583bf3fc/pretrain500m/reports/long-confirmation-heldout-results-1590660.json](https://github.com/yinli-systems/l20-1b-pretraining/blob/eb5eded979ba148b3dc06052d436eabc583bf3fc/pretrain500m/reports/long-confirmation-heldout-results-1590660.json)
- **R5** — [https://github.com/yinli-systems/l20-1b-pretraining/blob/eb5eded979ba148b3dc06052d436eabc583bf3fc/pretrain500m/reports/seven-task-confirmation-2gpu-results-1590907.json](https://github.com/yinli-systems/l20-1b-pretraining/blob/eb5eded979ba148b3dc06052d436eabc583bf3fc/pretrain500m/reports/seven-task-confirmation-2gpu-results-1590907.json)
- **R6** — [https://github.com/yinli-systems/l20-1b-pretraining/blob/eb5eded979ba148b3dc06052d436eabc583bf3fc/pretrain500m/posttrain/seven-task-confirmation-2gpu-v1/run.sbatch](https://github.com/yinli-systems/l20-1b-pretraining/blob/eb5eded979ba148b3dc06052d436eabc583bf3fc/pretrain500m/posttrain/seven-task-confirmation-2gpu-v1/run.sbatch)
- **R7** — [https://github.com/yinli-systems/l20-1b-pretraining/blob/eb5eded979ba148b3dc06052d436eabc583bf3fc/pretrain500m/posttrain/confirmation-runner-v1/run.sbatch](https://github.com/yinli-systems/l20-1b-pretraining/blob/eb5eded979ba148b3dc06052d436eabc583bf3fc/pretrain500m/posttrain/confirmation-runner-v1/run.sbatch)
- **S1** — [https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T](https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T)
- **S2** — [https://huggingface.co/EleutherAI/pythia-1b](https://huggingface.co/EleutherAI/pythia-1b)
- **S3** — [https://huggingface.co/keeeeenw/MicroLlama](https://huggingface.co/keeeeenw/MicroLlama)
- **S4** — [https://huggingface.co/HuggingFaceTB/SmolLM2-360M](https://huggingface.co/HuggingFaceTB/SmolLM2-360M)
- **S5** — [https://huggingface.co/Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B)
- **S6** — [https://huggingface.co/Qwen/Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base)
- **S7** — [https://ai.google.dev/gemma/docs/core/model_card_3](https://ai.google.dev/gemma/docs/core/model_card_3)
- **S8** — [https://huggingface.co/allenai/OLMo-1B](https://huggingface.co/allenai/OLMo-1B)
- **S9** — [https://huggingface.co/allenai/OLMo-1B-0724-hf](https://huggingface.co/allenai/OLMo-1B-0724-hf)
- **S10** — [https://arxiv.org/abs/2509.24945](https://arxiv.org/abs/2509.24945)
- **S11** — [https://huggingface.co/openbmb/MiniCPM5-1B-Base](https://huggingface.co/openbmb/MiniCPM5-1B-Base)
- **S12** — [https://huggingface.co/LiquidAI/LFM2.5-350M-Base](https://huggingface.co/LiquidAI/LFM2.5-350M-Base)
- **S13** — [https://huggingface.co/Qwen/Qwen3.5-0.8B-Base](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base)
- **S14** — [https://arxiv.org/html/2502.02737v1](https://arxiv.org/html/2502.02737v1)
- **S15** — [https://arxiv.org/abs/2504.11393](https://arxiv.org/abs/2504.11393)
- **S16** — [https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
- **S17** — [https://huggingface.co/datasets/HuggingFaceFW/dclm_100BT](https://huggingface.co/datasets/HuggingFaceFW/dclm_100BT)
- **S18** — [https://huggingface.co/datasets/HuggingFaceFW/finepdfs-edu](https://huggingface.co/datasets/HuggingFaceFW/finepdfs-edu)
- **S19** — [https://huggingface.co/datasets/HuggingFaceTB/finemath](https://huggingface.co/datasets/HuggingFaceTB/finemath)
- **S20** — [https://huggingface.co/datasets/HuggingFaceTB/stack-edu](https://huggingface.co/datasets/HuggingFaceTB/stack-edu)
- **S21** — [https://huggingface.co/datasets/epfml/FineWeb2-HQ](https://huggingface.co/datasets/epfml/FineWeb2-HQ)
- **S22** — [https://arxiv.org/abs/2308.04014](https://arxiv.org/abs/2308.04014)
- **S23** — [https://arxiv.org/abs/2411.15124](https://arxiv.org/abs/2411.15124)
- **S24** — [https://arxiv.org/abs/2504.13837](https://arxiv.org/abs/2504.13837)
- **S25** — [https://arxiv.org/abs/2309.00071](https://arxiv.org/abs/2309.00071)
- **S26** — [https://arxiv.org/abs/2404.06654](https://arxiv.org/abs/2404.06654)
- **S27** — [https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md)
- **S28** — [https://arxiv.org/abs/2203.05482](https://arxiv.org/abs/2203.05482)
- **S29** — [https://huggingface.co/HuggingFaceTB/SmolLM2-360M/blob/main/config.json](https://huggingface.co/HuggingFaceTB/SmolLM2-360M/blob/main/config.json)
- **S30** — [https://huggingface.co/facebook/MobileLLM-R1-360M-base](https://huggingface.co/facebook/MobileLLM-R1-360M-base)
- **S31** — [https://huggingface.co/facebook/MobileLLM-R1-950M-base](https://huggingface.co/facebook/MobileLLM-R1-950M-base)
