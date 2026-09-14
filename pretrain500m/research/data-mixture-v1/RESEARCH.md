# 529M 模型的数据配比、架构与后训练研究

研究日期：2026-09-13。对象：当前 528,748,800 参数、2,048 上下文的文本模型。用户目标是全面提升能力，并与不同大小的模型比较。本文区分**公开实验结果、当前项目实测、待验证的建议**；配比数字不是已经找到的最优解。

## 1. 决策

下一笔训练预算应优先用于**新数据的准入、多领域验证和配比对照**。当前模型只在一个 FineWeb-Edu 子集上训练过；它缺少显式配置的数学、代码、多语言和指令数据，也缺少足以支持全面能力判断的评测。因此，继续重复同一份网页数据、只观察一个验证 loss，不是合理的主要优化路线。

推荐先测试的全面能力配方是：**20% FineWeb-Edu、15% DCLM、20% FinePDFs-Edu、15% 数学、15% 代码、5% 合成教材、10% 多语言**。这些比例全部按清洗、去重、分割、tokenization 后实际进入 loss 的 prediction tokens 计算。它是有依据的起点，最终权重必须由当前模型的对照实验决定。

保留现有模型作为继续预训练的起点。另设等参数量的架构实验；不要同时更换语料、tokenizer、层数和优化器。先进架构不等于适合我们的硬件或预算，更不等于已经证明效果更好。

“全面超过其他大小的模型”应保留为研究目标，不能写成结果承诺。当前没有证据支持 0.53B 模型全面超过所有更大模型。合理的验证路线是：先与相近量级做完整的同协议比较，再向 1–2B、3–8B 扩展，报告逐项胜负、误差范围和推理成本。文本、视觉、音频能力需要分别评价，当前模型没有后两者的资格证据。

## 2. 当前起点与真正的瓶颈

| 项目 | 已验证状态 | 对决策的影响 |
|---|---|---|
| 模型 | 26 层，宽度 1280，FFN 3584，20 Q / 5 KV，head dimension 64，GQA、SwiGLU、RMSNorm、RoPE、共享词嵌入 | 属于合理的小型 dense Transformer 起点 |
| 数据 | 15.999B prediction tokens；8.377B unique prediction tokens，约 1.910 轮；只有 FineWeb-Edu sample-10BT | 新的有效信息和领域覆盖比继续盲目重复更值得测试 |
| 基础评测 | 固定七项平均 47.9568%，bootstrap 95% 区间 47.0329%–48.8761% | 只覆盖部分常识、阅读、科学选择题，不能推断代码、数学解题、中文或指令能力 |
| 训练效率 | 已完成正式训练的 MFU 中位数约 54.54%；4 卡 CPT pilot 约 77.5% | 满足此前 >50% 的要求，但新数据管线必须重新验证 |
| 当前 LR pilot | 三组均完成；每个候选约 0.537B token；验证只看 64 条序列 | 小于 0.003 的 loss 变化不足以支持大规模投入决策 |
| 后续恢复 | v1 在 resume 路径上没有恢复初始化 checkpoint 身份；本地 v2 已持久保存该身份并通过 CPU 恢复对照 | 529M 模型的真实 GPU 恢复一致性仍待验证 |

项目结果依据：[正式审计](../../reports/formal-v5/final-audit-v5.json)、[固定评测汇总](../../reports/formal-v5/seven-task-aggregate.json)、[现有 CPT 协议](../../posttrain/cpt-pilot-protocol-v1.json)。这些结果优先于早期设计文档中的预估数据量和旧 GPU 配置。

本轮最终刷新确认 Slurm 1587857 的三组均 `COMPLETED`。初始验证 loss 同为 2.698995；3e-5、1e-4、3e-4 的最终值分别为 2.696879、2.696430、2.702541。1e-4 的相对 loss 改善约 0.095%，三个候选的稳态 MFU 均约 77.5%。旧选择器的 `PASS_CPT_PILOT` 只表示通过其旧的探索条件，**不代表新协议允许正式晋级**。见 [实时记录](lr-pilot-live-receipt.json)、[探索选择记录](lr-pilot-selection-exploratory-v1.json)。

当前底座 checkpoint 的 SHA-256 为 `13aa21721e15c48cdfdafe30d8fdd41d9c95c90be766327661d96af1e90dd6cf`。后续所有 CPT 对照应从这个相同底座重新开始，不能把一个候选训练后的权重作为另一个候选起点。

强小模型的公开训练量通常远高于当前预算：SmolLM2-360M 的模型卡报告 4T；MobileLLM-R1 报告约 2T 的数据池、4.2T 的训练；LFM2.5 报告 28T。这说明 16B 仍是早期训练规模；这些数字不构成“我们也必须训练同样多”的结论。[SmolLM2-360M](https://huggingface.co/HuggingFaceTB/SmolLM2-360M)、[MobileLLM-R1](https://arxiv.org/abs/2509.24945)、[LFM2.5-350M](https://huggingface.co/LiquidAI/LFM2.5-350M)。

## 3. 配比研究：哪些证据可以迁移，哪些不能

| 研究 / 发布时间 | 可以借鉴的结果 | 本项目的适用边界 |
|---|---|---|
| [DataDecide，2025](https://arxiv.org/abs/2504.11393) | 25 种数据配方、多个模型规模和三个种子；150M 的排序可较好预测 1B；连续 likelihood 指标有用 | 约 80% 的预测正确率不是保证；数学的生成准确率尤其不能仅由小模型 likelihood 推断 |
| [SmolLM2，2025](https://arxiv.org/abs/2502.02737) | 网页、数学、代码及退火阶段共同设计；后期引入更好的专门数据 | 论文主要配方针对 1.7B、约 11T 训练，不能直接复制到 529M、几十 B 的 CPT |
| [DoReMi，2023](https://arxiv.org/abs/2305.10429) | 使用参考模型与 domain excess loss 学习权重 | 不能简单地给原始 loss 最大的领域更多权重；噪声和高熵文本也可能有高 loss |
| [RegMix，ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/5f67d864aae6115374fed7beddd119e0-Abstract-Conference.html) | 用小模型实验训练 mixture→performance 回归器，领域间存在相互作用 | 四组实验不足以拟合可靠的高维“最优配比函数” |
| [CLIMB，2025](https://arxiv.org/abs/2504.13161) | 先按语义聚类，再迭代探索数据权重 | 数据集名称不是语义领域；FineWeb、PDF 中同样有数学、代码和新闻 |
| [RegMix-D，2026-06](https://arxiv.org/abs/2606.18663) | 用 proxy 的 loss 轨迹预测阶段性配比，而非只用最终 loss | 主要是 1B、25B Pile 实验，目标仍是单个 pile-cc 验证 loss；并非每项任务都提高 |
| [Mixture Experiment，2026-08](https://arxiv.org/abs/2608.23922) | 用响应面和实验设计更有效地选择候选配比 | 部分节约实验数量的结论来自校准模拟，应作为后续方法候选 |

SmolLM2 的一个具体例子是：中期使用 75% 英语网页、20% 代码、5% 数学；最后退火阶段使用约 58% 网页、24% 代码、14% 数学、4% Cosmopedia。它支持“保留广泛语言覆盖，同时逐步补强数学代码”的思路；它没有证明这四个比例在其他模型上最优。[原论文第 4 节](https://arxiv.org/html/2502.02737v1)。

需要特别纠正“50/30/20 是最佳配方”的说法。该经验最初来自作者在约 70M 模型上的特定实验；2026 年 Hugging Face 的 SmolData 发布了对应的约 100B token 混合版本。发布数据集、给出一个有效案例，都不等于建立了通用最优性。我们应把它作为一个 PDF 权重较高的对照，而不是最终答案。[原作者实验](https://huggingface.co/blog/codelion/optimal-dataset-mixing)、[官方混合数据卡](https://huggingface.co/datasets/HuggingFaceFW/finepdfs_edu_50BT-dclm_30BT-fineweb_edu_20BT)。

另外，FinePDFs 官方实验在其评测设置中发现教育版与 HTML 语料混合后表现更好。这是把 PDF 纳入候选池的重要依据，但不能把该结论扩展为它胜过所有最新语料、所有模型和所有任务。[FinePDFs 技术报告](https://huggingfacefw-finepdfsblog.hf.space/)。

## 4. 找到的数据：主池、替补池与后训练池

已保存 28 个数据集入口和 8 个模型入口的公开元数据、固定 revision、README 快照及可取得的配置。入口之间有别名和重叠，不能把它们当成 28 份独立信息。完整记录见 [source-inventory.json](source-inventory.json) 和 [DATA-CATALOG.md](DATA-CATALOG.md)。公开数量采用各作者的 tokenizer，正式 token 配额必须用我们的 tokenizer 重新计算。

| 数据来源 | 推荐用途与相对优势 | 现状与限制 |
|---|---|---|
| [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | 教育网页与底座能力保持 | 已有远端正式训练子集；新内容需排除旧数据重复，不能把 sample 子集和全集累计成独立规模 |
| [DCLM-Baseline](https://github.com/mlfoundations/dclm) / [官方 100BT 样本](https://huggingface.co/datasets/HuggingFaceFW/dclm_100BT) | 增加日常语言、讨论、文档和网页多样性 | 本地取得固定版本正文样本；原始大 row group 超出采样预算，改用官方分片版本成功 |
| [FinePDFs-Edu](https://huggingface.co/datasets/HuggingFaceFW/finepdfs-edu) | 教材、解释性长文、知识结构；官方报告 350B+ tokens、69 种语言 | 本地取得英语样本；需页面级语言识别、公式/OCR 和页眉页脚检查；350B+ 是多语总量 |
| [FineMath 4+ / InfiWebMath 4+](https://huggingface.co/datasets/HuggingFaceTB/finemath) | 数学讲解与完整推导；公开规模约 10B / 8.5B | 两者均取得样本；要核算交集、模板重复和错误解答，不能视为全部已验证答案 |
| [Stack-Edu](https://huggingface.co/datasets/HuggingFaceTB/stack-edu) | 约 125B token 的教育代码索引，多语言覆盖 | 数据文件只有 SWH 内容 ID；已从公开 S3 取回 14 份代码并校验原始内容 SHA-1；大规模取回仍未做 |
| [Cosmopedia v2](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) | 补充适合小模型的教材式表达与解释 | 已有正文样本；`cosmopedia-v2` 入口会指向 SmolLM corpus，不能重复计入；建议初始上限 5% |
| [FineWeb2-HQ](https://huggingface.co/datasets/epfml/FineWeb2-HQ) / [FineWeb2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) | 多语言覆盖，按语言分层 | 已取得中文、西语样本；HQ 不等于教育或事实核验，不能认为 10% 混合足以学好所有语言 |
| [FineWiki](https://huggingface.co/datasets/HuggingFaceFW/finewiki) | 知识与多语言补充候选 | 元数据固定；需选定语言、版本、重复控制，再与 PDF/网页对照 |
| [Nemotron-CC-v2.1](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2.1) | 更新的网页与多种合成改写候选 | 访问受限，初始配方不依赖它；公开卡中的 TB 级规模不等于当前可以下载 |
| [Nemotron-CC-Math-v1](https://huggingface.co/datasets/nvidia/Nemotron-CC-Math-v1) | 后续强数学候选；3plus 133B、4plus 52B、MIND 73B | 有访问门槛；3plus 是 `3` 和 `4plus` 的并集，不能重复相加；作者优势来自 8B 级实验 |
| [Nemotron-Pretraining-Specialized-v1](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Specialized-v1) | 新的教材、问答、科学代码、Wiki 改写替补 | 可公开访问；所选 Parquet row group 超过 48MiB 读取上限，本轮只确认 schema，未取得正文样本 |
| [MegaMath](https://huggingface.co/datasets/IFM/MegaMath) | 数学网页、数学代码和合成混合候选 | `LLM360/MegaMath` 为重定向入口；与 Stack/FineMath 等来源的交集必须核算 |
| [Dolma 3 Dolmino](https://huggingface.co/datasets/allenai/dolma3_dolmino_mix-10B-1025) | 研究成熟中期混合配方，补充知识、阅读、数学和代码 | 属于已经混合的语料；不建议作为额外黑箱再混入主池，先拆清来源、权重和评测重叠 |

Nemotron 数学的强公开证据值得重视，但应正确理解：其官方比较是 8B 模型的 mid-training，不能直接推算 529M 的收益；正文可访问性和协议也是两个独立问题。[NVIDIA 原始实验](https://huggingface.co/blog/nvidia/nemotron-cc-math)。

代码只接纳已识别且允许使用的来源许可证，保存 repo、path、blob ID 和许可证字段；`no_license` 不能因为外层数据卡写着“可用于训练”就自动视作获准。样本的 14/14 校验成功证明了这条取回路径可用，不证明整个代码池已经齐备、许可证完备或代码语义正确。

后训练的最新候选包括 [Nemotron-SFT-Math-v3](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Math-v3)、[Open-SWE-Traces](https://huggingface.co/datasets/nvidia/Open-SWE-Traces)、[Nemotron Instruction Following](https://huggingface.co/datasets/nvidia/Nemotron-Instruction-Following-Chat-v1)、[Dolci Instruct SFT](https://huggingface.co/datasets/allenai/Dolci-Instruct-SFT)、[SmolTalk2](https://huggingface.co/datasets/HuggingFaceTB/smoltalk2)。它们都已固定数据卡版本。最新不等于适合当前小模型：长推理轨迹、多轮工具状态和难题需要长度、正确性、难度及能力前提检查。

## 5. 真实样本检查改变了哪些决定

本次下载 1,422 条记录：其中 256 条只是代码索引，1,152 条为正文或对话，另外 14 条是成功取回的代码正文。代码正文与索引存在对应关系，不能当成互不重叠的独立样本。所有样本均未准入正式训练。

采样方法为每个指定子集选一个固定 shard、一个 row group，并在该 group 内均匀取最多 128 行。它可以验证取回路径、schema 和明显失败模式，**不能估计整个语料库的合格率**。下载脚本限制每个来源最多 48MiB 读取量；不执行数据集脚本或下载的代码。记录见 [sample-audit.json](sample-audit.json)、[补充记录](supplemental-sample-audit.json)、[代码取回记录](code-rehydration-receipt.json)。

实际检查产生四个决策：

1. **PDF 需要切分，不能仅取开头。** 128 篇英语分区样本有 41.4% 超过 2K；如果每篇只取前 2K，会丢掉该样本中 58.1% 的 token。按章节/页面切分并保留完整文本覆盖；需要跨长距离的任务单独留给上下文扩展实验。
2. **教育评分仍会漏过噪声。** 选定的 FineMath 样本包含自动生成的数字属性页面、破损公式的预览文本以及讨论中的错误尝试；英语 PDF 分区有非英语开头。准入不能只写 `score >= 4`。
3. **多语言权重必须与 tokenizer 一起检查。** 当前 tokenizer 在中文样本上约 1.27 token/Unicode 字符，英语网页约 0.21。字符跨语言不具备语义等价性，不能把这解读为“中文差六倍”，但它说明文档数和字符数都不适合控制训练配额。
4. **代码可取回，但不能只看语法。** 14 份正文与 SWH 原始内容 hash 一致，14/14 能被 Python 3 AST 解析；这不代表依赖存在、程序能运行或解题正确。代码评测必须另用隔离执行器和测试。

上述数值由当前固定 tokenizer 计算，见 [tokenizer-sample-audit.json](tokenizer-sample-audit.json)。对话长度统计暂未包括最终 chat template 的特殊 token，不能直接当作 SFT 打包预算。30 条定点人工式检查的观察保存在 [manual-sample-review.json](manual-sample-review.json)。

## 6. 建议测试的配方与质量控制

| 有效 token 比例 | M0 原始对照 | M1 通用英语 | M2 推理加强 | M3 全面覆盖候选 |
|---|---:|---:|---:|---:|
| FineWeb-Edu | 100% | 30% | 20% | 20% |
| DCLM | 0 | 20% | 15% | 15% |
| FinePDFs-Edu | 0 | 20% | 25% | 20% |
| 数学 | 0 | 15% | 20% | 15% |
| 代码 | 0 | 10% | 15% | 15% |
| 合成教材 | 0 | 5% | 5% | 5% |
| 多语言 | 0 | 0 | 0 | 10% |

M3 更符合用户的全面能力目标；M1、M2 用于量化广度与专门推理之间的变化。四组都是复合候选，只能判断整体配方优劣，不能把其中差异归因于某一种数据。若 M3 有优势，再做删去 PDF、改变代码占比、改变中文比例的单因素实验。

数学槽位先用 FineMath 4+ / InfiWebMath 4+ 的 75/25，这是待测内部比例。代码槽位先测试 Python 50%、JavaScript 20%、TypeScript/Cpp/Java 各 10%；这是小模型入门覆盖方案，不是所有编程语言的最优分布。多语言槽位先用中文 50%，西语、法语、德语、日语、阿语各 10%；也即总训练 token 中中文 5%、其余五种各 1%。如果目标要求更多语言，需要新增对应验证，不能只扩展数据列表。

需要把“优质”拆成可检验的条件：

- 正文可读、语言符合预期；保留数学符号、缩进、表格关系，避免通用字符过滤器误删数学和代码。
- 依据 URL 家族、模板、仓库和内容近重复簇控制集中度；先识别真实长尾，再决定是否限制一个来源的比例。
- 按来源记录下载 token、过滤后 token、独特 token、进入训练的 token 和累计重复次数。重复率按每个模型训练分支单独计算，不能把独立试验的读取次数都算成一个模型的训练轮数。
- 合成数据保留父文档、生成 prompt、生成模型、验证结果。改写相同父文档不等于增加同样多的新知识。
- 评测去污染覆盖旧的七项测试，以及新增数学、代码、指令和多语言测试；新验证集也应从训练中排除。

跨来源去重必须先测交集，再固定策略。FineMath 卡报告过一个值得注意的反例：删掉 FineMath 与 InfiWebMath 的交集，某些实验反而下降。这说明重复有时起到了隐式加权作用。推荐移除非预期重复，并把希望增加的权重显式写入采样器；如果保留重复更好，也必须通过对照并计入重复预算，不能隐藏。[FineMath 数据卡](https://huggingface.co/datasets/HuggingFaceTB/finemath)。

清洗顺序建议为：固定来源 → 完整解码 → 质量/语言审计 → 文档与仓库分组 → 近重复聚类 → 基准去污染 → train/dev/confirmation 分割 → tokenize → 按有效 token 配额打包。分割必须按文档/仓库/问题家族进行，避免同一本 PDF 的相邻块或同一代码仓库进入训练与验证两侧。

现有 reader 只打乱文件顺序，文件内顺序读取。把不同来源的文件丢进同一个目录，会出现长段的单一领域训练，不能保证期望配比。新管线需要预混合到固定长度块，或实现带完整状态的多来源采样器；恢复时要同时恢复每个来源的 cursor、epoch、shuffle、mixture schedule 和随机状态。

## 7. 如何用实验找到更好的配比

机器可读设计见 [experiment-design.json](experiment-design.json)。设计已写出，数据准入和 GPU 实验尚未完成。`plan_tokens.py` 能生成精确 block/token 配额，不能用来宣称训练已启动。

**第一轮：筛选明显的坏配方。** 使用当前完整 529M 底座，4 个配方 × 2 个数据顺序种子 × 536,870,912 prediction tokens，共 4.295B tokens。保持模型、LR、schedule、有效全局 batch、上下文和评估样本一致；从同一底座开始。LR 1e-4 只是起始假设，来自旧数据的 LR 结果不能证明它对新混合最优。配方排名与 LR 比较分开进行。

**第二轮：确认结果。** 取两个候选和 M0，每个两个种子，重新从底座训练 2.147B tokens，总计 12.885B。使用未参与第一轮选择的 confirmation dev 家族；先在固定配方上复核 LR，再固定确认协议。M0 的旧数据累计曝光约 2.17 轮，应显式登记为对照重复量，不能被“新语料最多两轮”的规则悄悄排除。

验证至少覆盖五个领域：通用网页、知识阅读、数学、代码、多语言。每个领域至少 1,000 个独立文档/问题家族及约 1M prediction tokens，分成探索与确认部分。还需要独立的生成正确性和短指令任务，避免只优化语言建模 loss。

预声明的候选晋级标准是：各领域相对于同一底座的 loss 变化等权汇总，平均相对改善至少 0.5%，配对区间排除零；每个领域相对 loss 退化不超过 0.5%，准确率按任务族允许的退化不超过 1 个百分点，并做同时置信区间。两个训练种子应有同向结果。**这些阈值是本项目的初始效应门槛，不是论文给定常数**；若验证集精度不足以判断，应增加验证样本或报告不确定，不能把“未显著变差”当作已证明不退化。

Bootstrap 要按独立文档、仓库或问题家族抽样；不能把同一文档的 token 当成独立观察。两个训练种子不能精确刻画种子分布，它们只是最低限度的稳健性检查。固定测试只在确认阶段通过后用于正式比较，不能不断回看测试分数调参。

有稳定方向后，再做约 20B 的增量训练，按 50B、100B、300B 累计研究节点评估边际收益。并非一次承诺跑到 300B。动态配比先与静态胜出配方对照：可测试前 70% 使用广泛配方、后 20% 平滑增加优质子集、最后 10% 退火；不能把文献中的 T 级阶段机械缩成几个小时。[Smol Training Playbook](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook)。

重复预算初始按新来源不超过两轮规划，再按实际收益调整。旧研究发现一些设置下四轮重复仍有价值，但不是所有来源的上限保证；2026 年的新工作仍在研究数据受限条件下的正则化和收益衰减。因此优先衡量独特信息与跨域收益，而不是用重复次数制造“大数据量”。[数据受限缩放研究](https://arxiv.org/abs/2305.16264)、[2026 年后续研究](https://arxiv.org/abs/2606.06888)。

## 8. 架构：保留什么，单独测试什么

当前 529M 结构没有发现一个足以解释全面能力差距的明显设计错误。约 12.2% 参数在词嵌入、20.1% 在注意力投影、67.7% 在 MLP；GQA 和共享词嵌入节约了小模型很宝贵的参数与 KV cache。提升架构的价值应由相同数据与计算预算下的质量、训练吞吐、推理延迟共同决定。

| 路线 | 已核查的结构事实 | 建议 |
|---|---|---|
| 当前 dense 模型 | 26×1280，FFN 3584，Q20/KV5/head64，528.749M | 当前 CPT 的控制架构，保存已训练权重 |
| 等参数 deep/thin 候选 | 32×1152，FFN 3200，Q18/KV6/head64，约 525.138M | 新初始化对照；参数匹配不等于计算与速度匹配 |
| 当前结构 + QK-Norm | 若每层各给 Q/K 一个共享 head-dimension RMSNorm scale，增加 3,328 参数 | 仅在独立候选中测试；不要直接改已有底座的注意力尺度 |
| Qwen3-0.6B | 配置为 28×1024，Q16/KV8，**head dimension 128** | 不可用 hidden_size / heads 推导为 64；复刻还要包含 QK-Norm 等细节 |
| Qwen3.5-0.8B | 文本骨干 24×1024，每四层一个 full attention，其余 linear attention，另有视觉模块 | 当前小模型中值得研究的混合路线；不能直接套 dense MFU 公式或把视觉能力归给文本底座 |
| LFM2 / LFM2.5 | 卷积块与少量 GQA 交错；硬件参与架构搜索 | 适合低延迟推理方向，需单独测本机训练内核和目标推理设备 |
| Falcon-H1-0.5B | Transformer 与 Mamba 混合 | 已公开的近量级对照候选，不是可无成本替换的 layer |

结构来源：[MobileLLM](https://arxiv.org/abs/2402.14905)、[Qwen3 固定配置](https://huggingface.co/Qwen/Qwen3-0.6B-Base/blob/da87bfb608c14b7cf20ba1ce41287e8de496c0cd/config.json)、[Qwen3.5 固定配置](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/2fc06364715b967f1860aea9cf38778875588b17/config.json)、[LFM2 原论文](https://arxiv.org/abs/2511.23404)、[Falcon-H1](https://huggingface.co/tiiuae/Falcon-H1-0.5B-Base)。

两项容易误读的配置：LFM2 的 FFN 有自动宽度调整，不能直接将 config 的 nominal intermediate size 当成最终矩阵宽度；其公开模型卡的 32K 上下文，也不能因为 config 写了更大 max_position 就变成已验证的更长能力。Qwen3.5 的 vocabulary、head dimensions 和线性注意力维度也与普通 Llama 不同。

2026-08-31 的 Qwen3.8-Next 报告进一步研究了混合注意力、稀疏注意力、门控残差、主存 n-gram 表与 Muon。其公开对象是 125B 总参数、6B 激活参数，外加 51B 的 n-gram 表；不能将“6B 激活”理解为一个轻量的 6B 存储模型。我们可以借鉴联合评估质量、稳定性、训练和推理成本的方法，但没有依据立刻给 0.53B 加一套主存参数系统。[Qwen3.8-Next 原论文](https://arxiv.org/abs/2608.30320)。

当前架构实验排序：先固定新数据测试 QK-Norm 与等参数形状；再做 2K→4K 的上下文扩展；只有目标推理延迟/内存明确受限时，再投入 hybrid 或 MoE。MoE 增加了总参数、路由和通信成本，低激活参数不能自动满足 >50% MFU。

tokenizer 保持现状用于 CPT，避免破坏底座。若未来从头训练多语言模型，应比较 32K/50K/64K 词表在目标数据上的压缩率、词嵌入占比、软最大计算量和下游效果。当前 uint16 存储只能容纳 0–65535 的 token ID，任何更大词表必须升级数据格式，不能只改 config。

上下文扩展必须重新按原文打包，保留完整的问题、答案和工具结果；不能简单拼接已有无关的 2049-token 块。先做 4K 的真实训练与短上下文回归，再决定 8K。更大的 position 配置或 RoPE theta 不等于模型学会了长文理解。

## 9. Post-training 应如何接上

第一阶段建议使用 10万–30万条经过检查的短、完整、多样指令，先测试一个 epoch，并按 assistant 目标 token 控制配比：通用对话/知识 30%、数学 20%、代码 20%、结构化输出/工具 15%、多语言 10%、校准与安全 5%。这是初始实验设计，样本数、完整上下文 token 和有监督目标 token 都要分别记录。

通用起点可采用 [Smol-SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk) 的小模型友好部分，并与 [Dolci Instruct SFT](https://huggingface.co/datasets/allenai/Dolci-Instruct-SFT) 的短样本对照。对已混合的 SFT 数据，应根据原始来源重新分组，而非给整个包再指定一个无法解释的权重。

数学方面，Nemotron-SFT-Math-v3 包含有/无 Python 工具的推理，其最终答案经过参考答案核对，且 2026-04-27 修复过格式问题。必须锁定修复后版本。正确的最终答案不保证中间推导全部正确；先选难度适中、完整且适配长度的样本，再逐步提高难度。[数据卡](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Math-v3)。

代码/agent 方面，Open-SWE-Traces 的当前卡记录了 2026 年的新轨迹和移除 git hacking 行为的更新，说明固定版本和行为审计都很重要。其轨迹存在成功、失败和未知结果，不能把整包当成正确示范；需要保留完整 tool schema、观测与动作，排除 benchmark/repository 泄漏。当前 2K 模型应先学短工具调用和局部代码修复，长轨迹留给上下文扩展后的对照。[官方数据卡](https://huggingface.co/datasets/nvidia/Open-SWE-Traces)。

SFT 后再比较偏好优化和可验证奖励训练。偏好对必须检查被选答案确实更好，避免只偏爱更长回答。数学、代码和结构化输出可以用可验证信号；若同一问题的多个 rollout 都失败，增加 rollout 数通常没有有效学习信号，应先补教学数据或降低难度。任何 teacher 生成和 verifier 的计算成本都纳入总预算。

最终全面评测至少包含：知识与阅读、常识推理、数学生成、代码执行、多语言、指令与结构化输出、工具调用、长上下文，以及校准/安全。base 对 base、instruct 对 instruct；分别报告无工具与有工具、短回答与长推理预算。匹配训练规模的比较和匹配参数量的比较应分别展示，不能用速度优势抵消能力退化后宣称“全面更强”。

## 10. 资源与实际执行状态

本轮实时检查：ParaCloud 登录节点访问 Hugging Face 仍返回 `Network is unreachable`；共享 `/ssd` 约剩 39GB；本地磁盘约剩 15GiB。对 corpus/pretraining2 的三层目录检查未找到可直接替代这些新来源的目录；这不是对整个账户存储的穷尽盘点。本地样本与索引已取得，远端新语料尚未备齐，新的多域训练没有启动。

采用现有实测吞吐作条件估算，假设模型、context 和 batch 不变、数据已在训练节点就绪：

| token 工作量 | 4×5090，约 163K token/s | 16×5090，假设维持约 460K token/s |
|---|---:|---:|
| 4.295B 筛选总量 | 7.3 小时 | 2.6 小时等价计算量 |
| 12.885B 确认总量 | 22.0 小时 | 7.8 小时等价计算量 |
| 20B 增量训练 | 34.1 小时 | 12.1 小时 |
| 100B | 7.1 天 | 2.5 天 |
| 300B | 21.3 天 | 7.5 天 |
| 1T | 71.0 天 | 25.2 天 |

以上不含排队、数据传输、预处理、编译、评测与保存，也不保证 16 卡当前可用。筛选设计是单次 4 卡、最多同时两组；不能把 16 卡等价时间理解为该筛选配置已经测过。新混合的 I/O、长上下文和 teacher 会改变吞吐，需要重新实测。

100B uint16 token 仅 token 数组就约 200GB，不适合一次写进剩余 39GB 的共享盘。采用有持久清单的分段处理：共享盘保存底座、审计和可恢复 checkpoint，节点 scratch 缓存有限的 packed segments；删除缓存前必须能从固定清单恢复。完整 optimizer checkpoint 约 6.35GB，原子写入时至少预留新旧两份，加上下一段数据和安全余量。模型评测副本可以仅保存模型权重，但不能替代 optimizer resume。

旧自动规则已被研究优先规则替代：现有 LR pilot 正常结束并保留探索记录；不会因为 64 条验证序列上的微小正变化自动启动 8B FineWeb-only 训练。下一阶段必须先达到数据、验证、恢复和 MFU 条件。

研究阶段完成了版本索引、可复用的限量下载脚本、真实样本审计和实验设计。随后新增的 [v2 候选实现](../../posttrain/v2/README.md) 接入了确定性的整数 token 配额采样，并修复恢复时丢失底座身份的问题。31 项 CPU 测试通过，包括四个真实实验配方的配额一致性、多卡分配模拟，以及小型 CPU 网络连续/恢复训练的逐步 loss、权重与 AdamW 状态完全一致。该实现尚未部署，CPU 测试不能替代 529M 模型的 CUDA/NCCL 实验。

尚未完成的是大规模语料取回与去污染、数学/代码/多语言内部子配比准备、训练站点数据就绪、多域验证集构建、v2 的真实 GPU 恢复与 MFU 验收，以及正式配比实验。v2 入口仅报告资格测试结果，不能自动晋级正式训练。详见 [本地验证记录](../../posttrain/v2/local-verification.json) 和 [执行状态](execution-readiness.json)。

## 11. 交付文件与复现

| 文件 | 用途 |
|---|---|
| [DATA-CATALOG.md](DATA-CATALOG.md) / [source-inventory.json](source-inventory.json) | 来源、固定 revision、访问状态与本地快照 |
| [collect_sources.py](collect_sources.py) | 公开元数据/数据卡的限量收集，不接受访问协议 |
| [acquire_samples.py](acquire_samples.py) | pinned Parquet range 采样，有读取上限和 hash 记录 |
| [rehydrate_code_samples.py](rehydrate_code_samples.py) | 代码索引→正文，保留许可证信息并校验内容身份 |
| [tokenizer-sample-audit.json](tokenizer-sample-audit.json) | 当前 tokenizer 下的长度、截断损失和字符/token 统计 |
| [experiment-design.json](experiment-design.json) | 配比、种子、token 预算、晋级条件、后训练起点 |
| [plan_tokens.py](plan_tokens.py) / [token-budget-plan.json](token-budget-plan.json) | 整数 block 配额，无 GPU 启动副作用 |
| [execution-readiness.json](execution-readiness.json) | 下一阶段仍待满足的条件 |

所有派生配额均通过总量、舍入误差、零权重和无效输入检查。数据下载可复用，但研究样本不是训练数据认证；文献支持的是可测试假设，最终能力结论必须由本项目的正式对照给出。
