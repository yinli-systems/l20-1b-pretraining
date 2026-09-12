# 1.1B 英文模型的最后 2B tokens：TinyLlama 对比与续训方案

## 判断与范围

额外 2B tokens 有机会改善现有模型，并尝试在预先固定的任务集合上超过旧版 TinyLlama。现有证据不足以承诺全面超过 TinyLlama 3T 或 v1.1；尤其不能承诺 HellaSwag 在这一小段续训中补齐十几个百分点。最有价值的目标是保留 20B checkpoint，做受控的继续预训练，得到可以复核的同协议比较。

这里的预算指所有新增训练实验累计不超过 2,000,000,000 个有效预测 tokens。旧数据重放、失败试验、小试、重跑和任何训练式 warmup 都收费；下载、评测与数据筛选另计 GPU 时间，但不是训练 tokens。模型保持现有参数、tokenizer 与 2048 上下文，从自己的 20B checkpoint 继续；没有加载别人的预训练权重。这仍然是从零预训练项目的延续。

研究核对日期为 2026-09-11。既有实测来自本仓库 `12168e72411f26461c33165c6711a88f9649690c` 的结果文件，以及当天读取的 GPU 主机原始结果。BoolQ 和两个 TinyLlama 的同协议补评测已全部完成，见 [实测对照报告](tinyllama-comparison-20260911.md)。下表中的 TinyLlama 数字是作者公布值，不能当成本次重测结果。本方案形成时尚未执行；后续已启动隔离的数据准备并通过 GPU forward-only preflight，详见 [执行协议](continuation-execution-protocol.md)。数据准备不代表已经开始反向传播。

## 现有模型与真实比较边界

现有模型为 1,100,048,384 参数，完成 19,999,703,040 tokens；最终验证 loss 为 2.4247145653，PPL 为 11.2990036011。训练配置为 BF16、micro-batch 6、累积 85、global batch 510，原学习率从 4e-4 衰减至 4e-5。[本地指标](../metrics/final-benchmarks.json)、[评测收据](../receipts/final-evaluation-receipt.json)、[训练代码](../../run_pretrain.py)。

| 任务 / 指标（%） | 我们 20B 实测 | TinyLlama 3T 公布值 | TinyLlama v1.1 公布值 |
|---|---:|---:|---:|
| HellaSwag acc_norm | 45.13 | 59.20 | 61.47 |
| OpenBookQA acc_norm | 35.00 | 36.00 | 36.80 |
| Winogrande acc | 52.17 | 59.12 | 59.43 |
| ARC-Challenge acc_norm | 33.62 | 30.12 | 32.68 |
| ARC-Easy acc_norm | 64.14 | 55.25 | 55.47 |
| PIQA acc_norm | 69.59 | 73.29 | 73.56 |
| 上述六项等权均分 | 49.9411 | 52.1633 | 53.2350 |

TinyLlama 公布值来自官方评测页和 v1.1 模型卡。[1][2] 两项六任务均分差距分别为 2.2222 与 3.2939 个百分点。这一差距不能外推成“整体能力只差两三分”：六项未包含 BoolQ，HellaSwag 和 Winogrande 的弱项会被 ARC 的优势部分抵消。不同评测版本、精度、提示和 tokenizer 行为也可能改变这些差距。

官方 GPT4All 集合有七项，包括 BoolQ。本次 BoolQ 实测为 59.1743%（3,270 题），补齐后的七项等权均分为 51.2601%。旧版官方分项相加得到约 52.9729%，v1.1 约 53.6286%；与之相比的差距只是跨协议参考。不能引用我们自行选择的八项均分 47.85 与对方七项均分比较。

2026-09-11 同协议重测的旧版 TinyLlama 3T 七项均分为 52.7763%，我们落后 1.5162 个百分点。10,000 次逐题配对、按任务分层的 bootstrap 给出差值（我们减对方）的 95% 区间约 [-2.4063, -0.6633] 个百分点。数据、提示、答案哈希以及任务配置均通过一致性核对。这是测题抽样不确定性，不是多次训练种子的波动范围。

同协议 v1.1 的七项均分为 53.5722%，我们落后 2.3121 个百分点。其中我们的 ARC-E 领先约 8.38 分，但 HellaSwag 落后约 16.31 分，Winogrande 落后约 7.02 分。ARC-C 对 v1.1 仅领先 0.68 分，逐题差异区间跨过零；不能把它与对旧版 3T 的 ARC-C 优势混为一谈。

BoolQ 的验证集含 2,033 个 yes 与 1,237 个 no；全部回答 yes 即可得到 62.1713%，高于当前模型的 59.1743%。因此不能把此项相对 TinyLlama 的小幅优势单独解释为强阅读理解。当前模型选择 yes 的比例为 2,384/3,270，存在明显答案偏好。

1.5162 分的七项差距意味着：若另外六项完全不变，单靠 HellaSwag 需要增加约 10.61 个百分点才能刚好追平旧版。分散改善也需要各项涨幅总和约 10.61 分，并且不能被其他项目的退步抵消。“均分只差一点”不等于只需让某一项涨一点。

v1.1 是重新训练的版本，作者记录了基础训练、继续预训练和 cooldown，不能把旧版 3T checkpoint 当作整个 TinyLlama 家族的上限。[2] 基座和 Chat 模型也需要分开比较。

这轮固定比较的是 3T 最终版和标准 v1.1，不包括其他中间 checkpoint、Math&Code 或 Chat 变体。官方表中的中间训练点也并非单调变好。[1][2] 将来即使超过这两个指定对象，也只能据此陈述指定协议下的胜出，不能升级为超过 TinyLlama 全家族或所有 1B 模型。

MMLU 当前为 25.31%，GSM8K 的 flexible extraction 为 1.67%，strict match 为 0.53%。MMLU 部分 few-shot 输入被左截断；两种 GSM8K 抽取指标必须一起保留。当前证据反映了知识选择题、常识续写和数学生成之间的不均衡，不能仅以验证 PPL 的降低代表这些能力都提高。

补测的 TinyLlama 3T / v1.1 MMLU 分别为 25.30% / 26.07%；GSM8K flexible 分别为 1.90% / 2.50%，strict 为 1.06% / 1.74%。我们与旧版在 MMLU 上只差 2 题；三者在此 GSM8K 协议下都很弱。这些结果没有提供“只差一个小技巧就有强推理”的证据。

对 MMLU 14,042 个样本重新从逐选项似然还原预测，模型选择 A/B/C/D 的次数分别为 2,851/8,591/2,056/544；B 占约 61.18%。真实标签中 D 最多，一律选 D 可得 26.89%。这是当前评分协议下明显的答案偏好和弱表现，不足以区分知识不足、提示格式与概率校准各自贡献。格式训练可能有帮助只是待测假设，不能把它当作已证明能恢复的隐藏能力，也不能把 SFT 后分数混入未做 SFT 的基座对照。

## 文献给出的有效线索

### 数据覆盖比继续提高教育门槛更值得测试

FineWeb 的阈值消融在 1.71B 模型、28B tokens 设置下发现，3+ 的总体效果最佳；作者选择它是为了平衡知识型任务与 HellaSwag。[3] 因此，现有 4+ 筛选有选择性，不能据此认定它对所有能力都是最优，也不能把降低到经验证的 3+ 简化为降低数据质量。

SmolLM2 的同条件对照显示，FineWeb-Edu 更有利于 ARC 等教育型任务，DCLM 更有利于 HellaSwag 和 CommonsenseQA；混合可以兼顾二者。但其主要 web 消融为 350B tokens，专门的 math/code annealing 分别达到 100B/200B，而且起点已训练 3T。[4] 这些结果支持调整方向，不能给我们的 2B 续训提供具体涨分保证。

现有 20B 已包含 42.5% DCLM，所以“第一次加入 DCLM 就大幅提升”的理由不成立。候选方向是增加新文档、保留更广的非教材叙事、检查来源多样性，并适量放宽教育分数门槛。当前分数与语料特点相符只是相关性；没有我们自己的消融，不能宣称某个数据源已经被证明造成某项优势。

### Replay 与温和的学习率设计

Ibrahim 等人的继续预训练研究表明，学习率重新升温、再次衰减和旧数据重放能平衡适应与遗忘；同时，重新升温本身即使在同分布数据上也可能引起 loss 上升。[5] 他们使用 405M/10B 模型及远长于 2B 的继续训练，因此不照搬其峰值学习率或 replay 比例。

本项目已完成一次 cosine 衰减。Hägele 等人的结果支持 cooldown 与同一轨迹权重平均的价值，同时指出 cosine 结束后的曲线不能直接按原趋势外推。[6] 我们不能再把“第一次 cooldown 的全部收益”算作尚未兑现的增益。

SmolLM3、OLMo 2 也采用分阶段语料或后期专门数据。[7][8] 这支持后期数据设计值得尝试；它们的模型规模、基础训练量和后期预算都不同，不能把论文的最终涨分按 2B 比例换算。

### 小 batch 是候选，不是已经验证的加速

2025 年小 batch 论文在含 1.3B 模型的实验中展示了适当调参后的效果，特别强调 Adam 二阶动量的 token 半衰期。[9] 对我们，global batch 从 510 改成 126、micro-batch 保持 6，可把累积从 85 改为 21，每个更新由 1,044,480 tokens 降为 258,048。

若按同一 token 半衰期调整，候选 beta2 为 `0.95^(126/510) = 0.9874075`。这只是该规则的数值应用；学习率与 weight decay 也要审视。论文并未支持“只缩小 batch，其他都不动”或“tok/s 一定更高”。优化器更新更频繁可能降低吞吐，训练阶段也会影响最优 batch。[9][10]

默认实验先保持原 batch，以便用有限预算辨别数据变化的价值。小 batch 作为明确的后续选项；若在本轮加入，必须替换原预算中的候选并提前写明，不能额外偷偷花一轮 2B。

### 合成、筛选和权重平均

PreSelect 在从零训练的 1B/3B 对照中展示了有选择的数据可以大幅优于未筛选基线。[11] 它需要自己的选择器和数据池；我们现有语料已经经过筛选。不能把“某实验的 10 倍效率”解释为最后 2B 等效于任意 20B 训练。

FinePhrase 是可用的 2026 年合成资源，包含 FAQ、tutorial 等改写。其 `text` 是原文，生成内容在 `rollout_results[*].text`；模型卡明确记录幻觉与截断限制。[12] 它适合抽样审计后的小比例实验，不足以支持把剩余预算全部换成合成文本。合成材料需要同时对原文和生成文本去污染；生成器及其隐含知识来源也应公开记录。

IMU-1 提供了另一条样本效率线索：其 430M、72B 模型报告末期 checkpoint EMA 改善均分，但同时改变了架构、优化器与训练阶段。[13] 对我们可测试的部分是同一续训轨迹的少量末期权重平均；不据此临时更换架构或 Muon，也不保证平均一定优于最后 checkpoint。

## 推荐数据配方

以下是需要实测的候选 B，不是论文证明的最优百分比。按本模型 tokenizer 计算有效预测 tokens；保留原始来源、revision、过滤记录和重复曝光计数。

| 成分 | 目标比例 | 目的与约束 |
|---|---:|---|
| 原训练集 replay | 30% | 内部按原 42.5/42.5/3/12 比例抽样；仅使用原 train |
| 新 DCLM 文档 | 35% | 提高自然语言覆盖；保持语言与内容过滤，排除近重复 |
| 新 FineWeb-Edu 3+ | 20% | 增加教材之外的表达，保留教育型能力 |
| 英文连贯叙事 | 5% | 候选为经清理的 Common Pile Gutenberg；按书去重与划分 |
| 新 FineMath-4+ | 5% | 保留基础数学学习信号，避免转成专门数学模型 |
| 新 Stack-Edu | 5% | 维持代码接触，沿用原许可和质量条件 |

在 replay 展开后，候选约为 32.75% FineWeb-Edu、47.75% DCLM、5% 叙事、5.9% 数学、8.6% 代码。原训练集的 replay 不是额外免费的数据：每一次参与反向传播都计入预算。

叙事源需要检查英文、OCR、章节边界、重复版本与现代语言覆盖，不能把全部古典书籍直接混入。Common Pile 提供文档来源与许可元数据，也区分 raw 与 filtered 版本。[14] 这个 5% 是覆盖实验，不是已经证实能提升 HellaSwag 的特效数据；若合格样本不足，必须明确修改并冻结配方，而非在后台换成未知来源。

候选 A 为控制组：维持原四源比例，也使用 30% 旧 train replay 与 70% 新四源文档。这样 A/B 都有新数据，不会把“新文档对反复重放旧文档”的效果误归因于新的比例。两组共享可共享的数据、顺序规则、tokenizer、学习率计划和计算预算。

本轮默认不额外混入 benchmark 的训练题、测试题、答案解释或以这些题为种子的合成改写。常识弱项可指导选择广泛的内容类型，但不能用评测题训练一个选题器来挑中相似问题后宣称纯外部泛化。

## 两个小试与完整预算

两组都从冻结的同一个 20B checkpoint 起跑。小试使用独立开发集选方向；最终完整 benchmark 在候选冻结后评估。小试阶段没有显著信号时不把小幅噪声当成获胜，优先保留控制组或停止这次扩展。

| 阶段 | 优化器步数（原 global batch） | 有效训练 tokens |
|---|---:|---:|
| A：原配方控制 | 190 | 198,451,200 |
| B：候选配方 | 190 | 198,451,200 |
| 继续合格候选 | 1,534 | 1,602,232,320 |
| 总实验费用 | 1,914 | 1,999,134,720 |

剩余 865,280 tokens 作为严格总额内的零头，不主动消费。获选模型沿自己的分支新增 1,800,683,520 tokens，累计为 21,800,386,560；另一分支的 198,451,200 tokens 仍然花掉了。因此这套设计不能被写成“最终模型训练满 22B 且小试免费”。如果重跑或额外训练式校验花费了 token，要从主段扣除。

190 步只适合发现明显遗忘、异常数据和较强方向性信号；不足以排除延迟收益，也不能可靠判定细小的 benchmark 差异。两个小试都按照完整分支约 1.8B 的计划运行，不能先为 200M 冷却到零，再选择赢家重新升温。

这个小试也只检验所登记的学习率条件；若较低学习率下暂时没有收益，不能据此证明配方本身无效。没有出现可靠信号时，停止或保留控制组是预算决策，不是宣称已经穷尽所有可行方法。

默认保留 BF16、micro-batch 6、global batch 510、AdamW betas=(0.9,0.95)、weight decay=0.1、clip=1。建议作为保留能力优先的初始试验：延续 4e-5，前约 10% 分支预算维持这一量级，随后按分支局部 token 进度平滑衰减至 4e-6。具体数值是待验证工程选择；如果数据适应不足，不能在没有新小试的情况下自动跳回原峰值 4e-4。

每约 50M–100M tokens 读取固定开发集与分源 loss，检查原能力；最后约 400M tokens 保留至少四个独立 checkpoint，额外生成一个预先定义的等权平均候选。权重平均只在同一架构、同一 tokenizer、同一分支上进行，不平均 optimizer，也不与 TinyLlama 权重合并。平均候选仍要过同样的评测门槛。[6]

主机当前还保留原训练的 `step-00018500`、`step-00019000` 和 `final`。因此也存在一个不消耗新增训练 tokens 的后续实验：对这些同一轨迹权重做预先定义的平均，再用开发集检查。它仍需要模型转换、存储和完整评测时间，本轮没有创建或评测这个新权重候选；也不能把其他论文的 EMA 涨分直接加到当前成绩上。

## 保留现有结果的具体含义

已有 20B 原始 checkpoint、HF 转换产物、tokenizer、配置、评测 JSON 与哈希收据作为永久基线。新的输出使用独立目录，并把原始模型哈希写入派生记录。这样即使候选退步，现有可用结果仍能保持。

“新候选绝对不遗忘任何能力”无法事先保证。正式选择时，至少要求同一固定验证集的 loss/PPL 不高于基线，ARC-E/ARC-C 和预先登记的保留任务点估计不下降，主集合平均分提高，并报告逐题配对差异的不确定性。若达不到，继续保留原模型；不能因为平均分提高就隐去变差的项目。

可以把 PPL 恶化 1% 设为中途诊断警戒：11.2990 对应约 11.4120，loss 约 2.434665。这是中止检查用的容差，不是“无损”的定义，也不是最后晋级的宽松替代条件。

开发用数据与最终测试用数据分开，样本 ID 先固定。已反复查看的公开评测集不能被描述为此前完全未见；最后另留未参与配方选择的能力检查。不同任务的差异按样本配对计算，均分按任务等权聚合，避免 ARC-E 的样本数支配结果。较小的 OpenBookQA 和 Winogrande 差异应报告置信区间。

代码生成能力尚无 HumanEval/MBPP 的实测证据。代码占比仍在，不等于已证明保留了代码能力；若将“代码无退步”纳入正式目标，需要单独配置代码执行隔离和固定解码协议后补测，不能以代码验证 loss 代替。

## 评测协议和数据去污染

同协议比较固定 `lm_eval==0.4.9`、BF16、`auto:4`、2048 上下文，不使用 chat template；随机、NumPy、Torch 种子为 42，而 few-shot 种子为 1234。这来自原始 JSON 和 LitGPT 包装器核对，不能简单写为所有 seed=42。[15]

主集合为 HellaSwag / OpenBookQA / Winogrande / ARC-C / ARC-E / BoolQ / PIQA 七项。相应使用 acc_norm 或原任务 acc，并保留 acc 与 acc_norm 原字段。额外完整报告 LAMBADA、TruthfulQA、MMLU 5-shot、GSM8K 5-shot，以便看清知识、续写和数学的不同变化。

固定参照模型 revision：

| 参照 | Hugging Face ID | revision |
|---|---|---|
| 旧版 3T | `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T` | `59f6f375b26bde864a6ca194a9a3044570490064` |
| v1.1 | `TinyLlama/TinyLlama_v1.1` | `ff3c701f2424c7625fdefb9dd470f45ef18b02d6` |

本次补测由 [evaluate_comparison.py](../../evaluate_comparison.py) 生成独立收据；复用已有结果前校验三个原始 SHA-256。样本级 doc_hash、提示/答案哈希、task config、样本数和 n-shot 已在全部比较中核对通过。跨 tokenizer 的 PPL 不直接作为排行榜，文本似然比较若有需要另用同文本 bits-per-byte 等明确指标。

Harness 之后的版本存在可能改变分数的修复。[16] 为复现旧结果，本轮保持旧协议；若定位到适用于本任务的实质 bug，则另建修订协议重跑全部比较对象，不能只给一个模型打补丁后混表。

现有 `build_decontam.py` 未包含 BoolQ；现有 13-word 匹配也不能证明不存在短题、转述或上游污染。新数据至少补齐 BoolQ 和所有将使用的评测项，分别处理题干、选项与短题完整归一化匹配，并检查源文档/近重复关系。旧 20B 的 BoolQ 去污染覆盖不足应如实记录，不能靠给后续 2B 添加过滤就追溯性宣称旧数据已全部合格。

新料应同时对原训练文档、原验证文档和新料内部去重。replay 为刻意重复，要有独立来源标记；原验证集始终排除。合成数据必须按原始文档或种子族分组去重、划分，不能只比较生成文本表面的哈希。

## 现有代码在续训前需要的变化

这里是代码审计得出的实施要求，不代表这些训练改造已经完成。

1. `run_pretrain.py` 固定了 20B 与 `full` 输出目录，并使用 `resume="auto"`；需要独立 continuation 模式、显式来源 checkpoint 和追加 token 计数。恢复原全局步数后把 `max_tokens` 设为 2B 会直接触发停止，不能用它表示追加 2B。
2. 原 LitGPT 在训练循环里根据 `iter_num` 与 `max_iters` 重算 LR；改变总预算会同时改变已处于末期的调度。[17] 要保留来源累计 tokens，并为新阶段定义局部调度，不能把重启调度说成平滑续训。
3. 原 checkpoint 保存逻辑包含模型、优化器和 dataloader 状态；应验证实际文件内容后保留权重与动量，对新数据显式重建 loader/局部计数。来自旧数据的游标不能误接到新语料。权重初始化与完整状态恢复是不同操作。[17]
4. `data_module.py` 的 `TRAIN_WEIGHTS` 同时被 train 与 val 使用。仅修改这个常量会连验证配比一起改，从而让 PPL 不可比。必须分离训练采样与固定原验证集。
5. 默认每 500 步保存/验证，相当于约 522M tokens，频率不足以检查本次 200M 小试；原来只保留最近两个 checkpoint，也不足以支撑末期平均。新分支需要独立留存策略。
6. 新 `run_manifest` 必须记录父模型、源数据/配置哈希、optimizer 是否继承、各分支费用以及实际有效 tokens。原 launcher 固定的 `initial_checkpoint=None` 不能出现在续训收据中。

执行前需要验证加载后未训练模型与冻结基线输出在登记容差内一致、追加预算不会立即退出、学习率边界连续、train/val 分离、分支输出不影响原文件，以及断点恢复不会漏计重复的反向传播。失败后重算必须占用实验 token 预算。这些是保护已有结果所必需的回归检查。

原训练验证设置为最多 100 个 batch，并非声明遍历整个 holdout。保留原口径的检查应固定实际样本或 token 序列清单；若另增更大、分源的新验证集，其结果单列，不能把换了验证范围后的 loss/PPL 直接与 11.2990 相减。

## 时间和投入

2026-09-11 的主机检查显示剩余磁盘约 57 GiB；每个原生完整 checkpoint 为 13,200,829,231 bytes（约 13.20 GB），HF 导出约 5.32 GB。若小试、续训与末期平均都保留完整 optimizer 状态，空间会不足。实施前必须按实际峰值做容量检查：用于平均的末期点可只保存 FP32 模型权重，完整可恢复 checkpoint 另设明确数量上限，并考虑原子保存的临时双份空间、NumPy/packed 两份 token 数据及原始下载缓存。必要时先扩容或经批准转存，不以删除现有基线来默认解决。

主机物理内存约 15 GiB，完整 checkpoint 本身已约 12.3 GiB。续训还要验证加载时的 CPU 峰值内存，避免同时保留多个完整状态副本；内存映射或分阶段加载方案需要实际验证，不能只检查 GPU 显存是否足够。

按已有基准约 12.5k–12.9k tokens/s，全部 2B 反向传播计算约需 43–44.5 小时；若有效速度降至 10k，则约 55.6 小时。历史基准不是本次续训速度承诺；下载、数据筛选、评测、保存和失败重跑还需额外时间。[历史 GPU 基准](../receipts/gpu-benchmark.json)

因此可按约 2–3 天安排有准备的数据与稳定主机条件下的实验，但不应给尚未开跑的任务一个保证完成日。HF token 可以影响某些访问限制；它本身不会提高已在本地训练的 tok/s。优先下载预先筛好的公开数据、CPU 完成文本清理、复用本地打包缓存，把 GPU 时间留给模型计算。

## 能够成立的最终结论

如果同协议七项均分及预先登记的保留检查都通过，可以陈述“在所列协议和任务集合上超过指定 TinyLlama checkpoint”，并附分项和不确定性。若仅 ARC 胜出，则只宣称 ARC 的优势。若数学或聊天后训练改善交互体验，也不能自动改写基座常识成绩。

现阶段合理的投入理由是以很小的追加预算检验数据覆盖、保留能力和末期优化的收益；不以“必然远超 TinyLlama”作为决策前提。固定规模再多 10% tokens 的普通继续训练没有已知定律保证跨越当前常识差距。同协议基线已经完成；下一步需要两组受控小试的真实收益和保留能力证据，而不是继续用论文涨分代替本项目实测。

## 来源

所有网络来源于 2026-09-11 核对。以下为作者论文、官方仓库、官方模型/数据卡或发布记录；方法结论与本报告提出的具体实验参数已区分。

1. TinyLlama authors. [EVAL.md](https://github.com/jzhang38/TinyLlama/blob/main/EVAL.md). 官方 GPT4All 分项、指标和旧版模型记录。
2. TinyLlama authors. [TinyLlama v1.1 model card](https://huggingface.co/TinyLlama/TinyLlama_v1.1). 2024；2T 三阶段训练、cooldown 和分项对照。
3. Penedo et al. [The FineWeb Datasets](https://arxiv.org/html/2406.17557v2). 2024；§4、附录 F.2，教育阈值与分布取舍。
4. Ben Allal et al. [SmolLM2: When Smol Goes Big](https://arxiv.org/html/2502.02737v1). 2025；§3.1–3.2、§4，web 配比和 annealing 实验尺度。
5. Ibrahim et al. [Simple and Scalable Strategies to Continually Pre-train Large Language Models](https://arxiv.org/html/2403.08763v2). 2024；§6–7，replay、重新升温和遗忘。
6. Hägele et al. [Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations](https://arxiv.org/html/2405.18392v1). 2024；学习率期限、cooldown、权重平均。
7. Hugging Face. [SmolLM3](https://huggingface.co/blog/smollm3). 2025-07-08；分阶段 web/math/code 训练与实际预算。
8. Team OLMo et al. [2 OLMo 2 Furious](https://arxiv.org/html/2501.00656v1). 2024/2025；后期专门语料与完整开放训练。
9. [Small Batch Size Training for Language Models](https://arxiv.org/html/2507.07101v2). 2025；§4，Adam 半衰期与 1.3B 实验。
10. [How Does Critical Batch Size Scale in Pre-training?](https://arxiv.org/html/2410.21676v2). 2024/2025；训练规模、batch、Adam 超参数的联动。
11. [Predictive Data Selection](https://arxiv.org/html/2503.00808v2). 2025；1B/3B 从零数据选择实验与适用范围。
12. HuggingFaceFW. [FinePhrase dataset card](https://huggingface.co/datasets/HuggingFaceFW/finephrase). 2026；生成内容字段、来源和局限。
13. Grigorev. [IMU-1](https://arxiv.org/html/2602.02522v1). 2026；430M/72B 实验与附录 checkpoint EMA。
14. Common Pile. [Project Gutenberg dataset](https://huggingface.co/datasets/common-pile/project_gutenberg). 模型训练候选叙事源，原始与过滤版本说明。
15. Lightning AI. [LitGPT evaluation wrapper](https://github.com/Lightning-AI/litgpt/blob/7bf2960dfb26bae8e815c9a16a22732974824ac1/litgpt/eval/evaluate.py). 与环境收据一致的 commit；核对 seed 传递。
16. EleutherAI. [LM Evaluation Harness releases](https://github.com/EleutherAI/lm-evaluation-harness/releases). 评测变更和版本固定的必要性。
17. Lightning AI. [LitGPT pretrain.py](https://github.com/Lightning-AI/litgpt/blob/7bf2960dfb26bae8e815c9a16a22732974824ac1/litgpt/pretrain.py). 原环境 commit；状态恢复、局部预算与 checkpoint 保存。
