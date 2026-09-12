# 有限预算续训的实验协议与研究复核

本项目的首要目标是在保留原 20B 基线的条件下，检验新增不超过 2B 预测 tokens 能否改善 1.1B 英文基座模型。比较对象固定为 TinyLlama 3T 最终版和标准 v1.1；目标指标为同协议七任务等权均分。超过指定模型的指定均分、超过每个分项、以及超过整个模型家族，是三种不同的结论。

2026-09-11 已完成原模型和两个对照的同协议评测。当前七项均分分别为 51.2601%、52.7763%、53.5722%。相对 3T 的配对差值为 −1.5162 个百分点，95% bootstrap 区间为 [−2.4063, −0.6633]；相对 v1.1 为 −2.3121，区间 [−3.1978, −1.4233]。这些区间描述固定模型的测题抽样不确定性，不包括训练种子方差。[完整结果与限制](tinyllama-comparison-20260911.md)。

本协议承接 [完整文献综述与 2B 方案](2b-continuation-plan.md)，补充会改变实施决策的研究、资源审计、停止条件和执行口径。原模型的权重、tokenizer、评测产物和原数据均保留不变。续训继承自己的随机初始化训练轨迹，不加载 TinyLlama 权重，也不更换 tokenizer。

## 研究补充及取舍

### 教育质量与语言覆盖

FineWeb 的 1.71B、28B 消融选择 3+ 教育门槛作为知识任务与 HellaSwag 的较好折中。SmolLM2 的 web 消融显示，教育型文本和更广的 DCLM 文本各有优势，混合有价值。它们支持测试覆盖面，不能证明某个具体新配方会弥补本项目约 14–16 分的 HellaSwag 差距。[^1][^2]

当前模型已经使用 42.5% DCLM，不能把“添加 DCLM”描述为首次引入新能力。实验应区分新文档、来源比例、教育阈值、叙事覆盖和优化器变化。A/B 保持架构、tokenizer、batch、优化器与学习率相同，只改变登记的数据配方。

### 续训学习率并无通用升温规则

Ibrahim 等人在较长的分布变化实验中验证了 replay、重新升温和衰减的作用，同时也观察到重新升温造成的遗忘。NVIDIA 的《Reuse, Don’t Retrain》报告则在自身设置中发现，不升温、从原末端学习率继续 cosine 衰减优于所试升温方案，WSD 也未胜出。两者研究条件不同，不能由任何一篇推导出普遍最优规则。[^3][^4]

这轮选择保留 Adam 动量，以 4e-5 开始，前 172 个更新保持该学习率，随后按完整分支长度衰减至 4e-6。约 200M 的 pilot 停点不改变完整分支学习率曲线，因此不会把“先冷却一次、获选后再次升温”的影响混入数据比较。这一保守取舍用于降低破坏现有模型的风险，不代表已经证明它对最后 2B 最优；适应不足仍是需要实测的风险。

### 梯度对齐值得研究，但本轮不同时加入

2025 年预印本《Revisiting Replay and Gradient Alignment》考察了每阶段 100B tokens、跨语言的连续训练。其 1B 对照中，50% replay 的英文均分为 64.0%，加入 Reptile 后为 63.8%；25% replay 的这项均分在加入 Reptile 后保持 61.5%。其他规模和指标有收益，但这不是对任意 1B、2B tokens 续训的统一改进保证。[^5]

因此本轮不同时改用 Reptile、Muon、小 batch 或新架构。它们不是被证明无效，而是不适合与数据变化混在这两个有限预算小试里。若未来重新研究这些方法，需要独立预算、优化器状态处理规则与消融。

### 叙事来源采用已过滤版本，仍需本地审阅

Common Pile 明确区分 raw 和 filtered 数据。论文中的过滤包括语言、OCR、部分来源的 PII 和模板清理，之后做跨来源模糊去重。Project Gutenberg 子集依据英文和公共领域元数据收集；这些上游说明有助于审计，但不意味着任何书籍都适合本模型的常识短期目标。[^6]

候选叙事源采用 `common-pile/project_gutenberg_filtered`，而非将 raw 版本直接当作已清洗语料。2026-09-11 查得 revision 为 `3cdf6879c807f4e4e063f2ceb23bc268d8c29ab7`。仍需检查实际字段、重复版本、过长书籍、OCR 残留和叙事占比，之后才允许加入候选 B。[^7]

## 冻结预算与配方

训练 token 指参与目标模型反向传播的有效预测位置，不包括每个样本的额外输入位置。序列长 2048，micro-batch 为 6，累积 85，global batch 为 510；每个优化器更新为 1,044,480 预测 tokens。replay、失败、从旧 checkpoint 重算的更新和目标模型的训练式 warmup 都计费。

| 阶段 | 更新数 | 新增预测 tokens |
|---|---:|---:|
| A 控制组 | 190 | 198,451,200 |
| B 覆盖组 | 190 | 198,451,200 |
| 获选分支的后续段 | 最多 1,534 | 最多 1,602,232,320 |
| 无失败重算时合计 | 1,914 | 1,999,134,720 |

总上限为 2,000,000,000，而不是“每个实验 2B”。获选轨迹最多新增 1,800,683,520，累计 21,800,386,560；另一个 pilot 消耗的 tokens 不属于获选模型的有效训练历程。失败重算会减少可用主段，不能假装免费。

A 为 30% 原 train replay 加 70% 新文档；内部沿用 web/DCLM/math/code 的 42.5/42.5/3/12 比例。B 为 30% 同比例 replay、35% 新 DCLM、20% 新 web、5% 经审阅的叙事、5% 新 FineMath-4+、5% 新 Stack-Edu。比例是实验假设而非文献中的最优常数。

**启动前配方修订：** 两组 fresh web 均改用 SmolLM corpus 中 FineWeb-Edu-Dedup 的 3+、英文语言置信度至少 0.9，并共同追加域名来源限制。原 20B replay 仍保持原来的 4+ 数据，不重新筛选或改写。因此 A 是相同比例、更新过滤方式的对照，不是完全复刻原分布；B 的差异集中在比例与叙事，而不再额外改变 web 教育分数门槛。修订发生在任何目标模型反向传播之前。这个取舍受 FineWeb 3+ 消融和本地抽样共同影响，不是声称论文证明了本项目的域名白名单最优。

每个 pilot 预先生成 96,900 个训练块的顺序。各成分使用最大余数法分配整数块，成分内部无放回抽样，之后固定随机顺序。这样有可复核的实际 token 配比，不依赖短跑中随机 source 抽样恰好接近目标。replay 的源标签单列，绝不把旧文档写成新料。

## 数据质量与污染控制

原文档 SHA-256 去重库以只读方式使用，同时覆盖已登记的旧 train 和 val 文档。新源使用独立游标副本，新文档哈希、ID、token 数与来源存入独立数据库。不会延长原训练文件、覆盖原游标或把原 val 当成 replay。

原训练的全部 NumPy shard 在建立索引前核对原 manifest SHA-256。旧 train 采用内容定义的 13-token 指纹、1/1024 抽样；新文档若有至少两个命中且覆盖其至少一半抽样指纹，则按近复制品拒绝。旧 val 使用全部 32-token 指纹，命中则拒绝。新文档之间应用相同的抽样 span 重复检查。

这个方法是内存有界的概率性跨度审计，不是完整 MinHash 去重，也不识别所有转述。短文档可能没有足够抽样指纹；64-bit 指纹也不是密码学证明。文档级 SHA-256、上游去重和跨度审计构成互补检查，不能据此声称“绝对无重复、绝对无污染”。

新的 benchmark 过滤库保留原 13-word 索引，并从本次实际评分的原始样本补齐字段内 13-word 匹配和 BoolQ。长度 5–12 词的完整短题使用归一化单词精确匹配；不对所有短答案做匹配，以免大量正常文本被无意义拒绝。新训练数据不会主动混入 benchmark 数据集或以其题目生成训练文本。

旧 20B 的 BoolQ 去污染覆盖不足仍然存在于历史证据边界中。新过滤不能追溯消除旧污染，也不能证明上游网站没有转述过测试题。已查看过的公开 benchmark 不能重新包装为从未查看的隐藏测试集。

每源保存过滤计数、实际训练块哈希和主机内文本样本。进入训练前需要独立质量审阅收据，绑定最终 manifest；任一源不足、去污染缺失、样本质量不符或哈希变化都会阻止训练。文本样本、原始语料和大体积数据库不进入 Git。

### 本地语义抽查导致的实际修订

最初四类新池约 163M 输入 tokens，完成去重、跨度与 benchmark 检查后，并未直接获准训练。最先查看的 20 个数学样本仍有题解农场、词语替换拼接和错误解释。这是有序便利样本，不能把其中问题数直接外推为整个 FineMath 的坏数据率。

v2 先用数学来源白名单和文本结构规则重筛。对首批数学池的 7,614 个完整文档逐篇解码、核对归一化 SHA-256 后，保留 1,830 篇、约 2.34M 输入 tokens；其余大多因为来源不在白名单被排除，**不等于证明其余文档都是坏文档**。另外补充约 7.97M 同规则数学 tokens，避免靠重复少量文本填配额。

随后对每个完整源池按文档 SHA-256 优先级取样，而非只看前 20 篇。web 中仍看到搜索摘要拼接和不自然的同义词替换，因此 v2 web 未获放行。恢复原 parquet 的 URL 元数据后，新 web 仅接纳登记的教育/政府域名，以及百科和科普出版方；同时排除聚合路径。缓存中的 4+ 新页经这些规则和旧数据排除后只有 5,319,204 输入 tokens，未满足配额，构建任务按预期失败，保留产物而不降低门槛冒充完成。

基于这个审计结果和前述 FineWeb 消融，正式候选改用统一 3+ 加来源限制，独立输出 `focused-web-score3-v4`。3+ 是新的、明示的实验选择，不是把 3 分文档标成 4 分。两组使用同样的新过滤；保留 DCLM 的非教育文体覆盖。白名单可能漏掉高质量个人博客，也不能保证机构网站每篇内容正确，适用性最终仍需用下游能力检验。

DCLM 中的色情/侮辱性俚语词典样本促使新增独立内容过滤；不会因为政治观点不同或新闻涉及争议而将其标为低质量。数学额外排除自动生成的论坛摘要、目录页和明显缺失公式的模板。正常题目讨论仍可能含学生的错误尝试和后续纠正；数学白名单不是逐题正确性认证。代码维持上游 permissive 许可和语言教育分数门槛，没有逐仓库执行全部代码，也未声称代码注释全部为英文。

FineMath 数据卡本身提示高分长文仍可能质量不均、图片公式可能丢失；FineWeb-Edu 分类器也明确说明高分可能偏向“看起来学术”的文本。因此本轮没有把再次调用同一种教育分数分类器当作独立语义验真。[^9][^10] 另审阅了 DCLM 官方 fastText 管线和发布模型，但没有把尚未在本地做误差检验的第二个分类器分数直接作为质量保证。[^11]

所有重筛基于文档边界和已记录长度，完整文档的 token 解码必须匹配原 SHA-256。原始已拼接池的最后一个部分文档不接纳；二次打包可能丢掉不足一个训练块的尾部，须与 manifest 的丢弃 token 数精确相符。这个尾部边界在 DCLM 重筛中触发过失败，已保留失败产物并补回归测试，不跳过哈希错误后继续训练。

## 验证口径和保留门槛

原验证过程每次最多 100 个 batch，不是全量 holdout。现已把这一口径重新生成的 600 个序列固定为 token 数组并记录 SHA-256，另外固定四个分源 probe，每源 96 个序列。新分源 loss 单列，不替换原混合验证口径。

2026-09-11 的 forward-only preflight 实测如下；所有数值来自 [GPU 收据](../receipts/continuation-preflight-20260911.json)，不是预测。

| 固定 probe | loss | PPL | 预测 tokens |
|---|---:|---:|---:|
| 原混合 100 batch | 2.424699545 | 11.29883412 | 1,228,800 |
| web | 2.432228565 | 11.38422432 | 196,608 |
| DCLM | 2.815621853 | 16.70355971 | 196,608 |
| math | 2.142814875 | 8.52339618 | 196,608 |
| code | 1.094964504 | 2.98907658 | 196,608 |

原最终混合 loss 为 2.424714565；重测只差 −0.00001502，不是训练进步。后续统一相对本次冻结 probe 和重测 baseline 比较。不同源的 PPL 不代表同一难度，也不能拿这些 probe 给不同 tokenizer 排名。

每 50 个优化器更新及 pilot 终点执行固定验证。若原混合 loss 高于基线加 `log(1.01)`，则保存诊断 checkpoint 并停止。这是 PPL 变坏超过约 1% 的中途警戒，不是最终“无退步”的定义。

pilot 结束不是晋级。主段开始前需检查源数据、曲线和独立开发信号；最终冻结候选后，重新运行七任务及已登记辅助任务，报告点估计、配对区间和各分项退步。验证 PPL 的下降不能替代 HellaSwag、MMLU 或生成质量的提高。若保留条件不通过，原 20B 仍是可用基线。

## 恢复、精度与容量

原 checkpoint 的 SHA-256 为 `7a3ebb609c2d79b1b6370531bed6fd173466d23f4847c1b8c589cc969790a933`，大小 13,200,829,231 bytes。实读确认含 model、optimizer、train_dataloader、iter_num 和 step_count；步数为 19,148。Adam 一阶和二阶动量为 FP32，所有参数状态计数一致。

加载采用 CPU mmap、meta 初始化后在 GPU 分配参数，逐张量核对与 native checkpoint 完全相等。mmap 减少一次性 CPU 内存分配，但仍需留出页缓存和暂存峰值。PyTorch 官方文档推荐类似的内存有界加载模式，并强调参数对象改变时优化器创建顺序的重要性。[^8]

GPU preflight 中模型及动量加载后的分配约 12.295 GiB。BF16 eager 与 compiled 输出并非逐位相同：RMSE 0.02011，最大绝对差 0.65625，argmax 一致率 99.4222%。这是明确记录的数值差异；通过的是预先设置的数值容差，不是声称二者完全一致。原权重逐张量检查则是完全相等。

主机物理内存约 15 GiB，GPU 为 L20 46,068 MiB；启动审计时磁盘约 55 GiB 空闲。每次 checkpoint 保存前检查原子临时文件所需空间；只写入独立 continuation 目录。数据准备也预留 checkpoint 空间，不通过删除原 20B checkpoint 来默认腾位置。

预算账本使用 SQLite 事务，在每个更新的第一次反向传播前预记整步费用。崩溃时可能保守多计未完成的一步，但不会漏计；从旧 checkpoint 重算相同 step 会产生新的费用记录。模型恢复进度和实验已收费 tokens 分开保存。完整 checkpoint 以临时文件、fsync 和原子替换发布，并核对 receipt；不盲目覆盖无法验证的残留临时文件。

首个正式启动通过权重与输入哈希校验，但在为 1,251 个分片建立 mmap 时碰到进程默认 1,024 个文件句柄软上限。失败发生在预算账本创建和反向传播之前，新增训练 tokens 为零。独立 launcher 仅把训练子进程的 `RLIMIT_NOFILE` 软上限设为 4,096，保留 1,048,576 的原硬上限；不修改系统全局设置。launcher 先映射全部实际分片并读取一个真实 batch 验证，再 exec 已冻结的训练器。此资源修复不改变模型、Adam 状态、数据顺序、学习率或预算。失败日志与 receipt 均保留。

## 执行状态和后续边界

**A 小试已真实开始。** 2026-09-11 11:59:45 UTC 完成第一个优化器更新，新增 1,044,480 预测 tokens；首步耗时 84.371 秒，约 12,379.7 tok/s，训练 loss 2.472564、裁剪前梯度范数 0.11994，峰值 PyTorch 分配 40.275 GiB。该步已完成反向传播和 Adam 更新，不再只是准备或 forward-only preflight。首步可能含编译/启动开销，不能作为长期稳定速度承诺。GPU 同期观测为 100% 利用率、43,751 MiB 占用。

随后复核已完成第 6 步，累计新增 6,266,880 预测 tokens。第 2–6 步各约 80.34–80.50 秒、12,976–13,000 tok/s；第 6 步 training loss 为 2.449813。同期 GPU 100%、43,751 MiB、约 352 W，未见 OOM 或数值异常。这仅证明早期运行与吞吐稳定，尚无本轮固定验证结果，不能由训练 batch 的 loss 波动推断下游能力涨分。按此速度，A 的 190 步约需 4–5 小时（含验证和保存的粗估），不是完成时刻保证。

完整配置见 [launch receipt](../receipts/continuation-launch-A-20260911.json)，数据与语义审核边界见 [quality review](../receipts/continuation-quality-review-A-20260911.json)，启动进度见 [dated snapshot](../receipts/continuation-pilot-A-start-20260911.json)。全仓本地测试 36 passed / 1 CUDA skip，目标 GPU 主机 continuation 测试 15 passed；另实际验证全部 1,251 个分片的映射和首个真实 batch。数据准备期间没有目标模型训练式 warmup。

当前后台任务为 A 的 190 步小试，不自动追加约 1.6B 主段。原 20B 模型、tokenizer、Adam 父状态和历史评测产物仍保留。预算在每步开始前预记，因此查询账本时通常会比最后一个已完成更新多一笔正在执行的费用；进度与预记费用必须分别报告。新的固定验证第一次在第 50 步，不将新配方的 training loss 与原 validation loss 直接比较。

B 的叙事来源审阅及语料、小量数学/DCLM 补充和完整小试仍需完成。后期权重平均、最终 benchmark、候选是否保留旧能力以及是否超过 TinyLlama，均未得到本轮续训结果证明。已开始并不等于已完成全部 2B 实验。

全部 2B 的历史吞吐估算约 43–56 小时纯训练，准备、评测、保存和重算另计。本轮已获得真实首步速度，但尚不能由单步承诺最终完成日期，或把 GPU 100% 利用率写成 MFU 100%。

## 来源

以下 primary sources 均于 2026-09-11 复核。版本和实验尺度是结论的一部分；论文中的方法收益没有直接记入本项目的预计涨分。

[^1]: Penedo et al. *The FineWeb Datasets*, 2024，附录 F.2。[论文](https://arxiv.org/html/2406.17557v2)。
[^2]: Ben Allal et al. *SmolLM2: When Smol Goes Big*, 2025，§3.2 与后期训练设置。[论文](https://arxiv.org/html/2502.02737v1)。
[^3]: Ibrahim et al. *Simple and Scalable Strategies to Continually Pre-train Large Language Models*, 2024。[论文](https://arxiv.org/html/2403.08763v2)。
[^4]: Parmar et al., NVIDIA. *Reuse, Don’t Retrain: A Recipe for Continued Pretraining of Language Models*, 2024，学习率消融与附录 B.2。[论文](https://arxiv.org/html/2407.07263v1)。
[^5]: Abbes et al. *Revisiting Replay and Gradient Alignment for Continual Pre-Training of Large Language Models*, 2025 预印本，1B 结果与 100B/阶段设置。[论文](https://arxiv.org/html/2508.01908v1)；[PMLR 出版记录](https://proceedings.mlr.press/v330/abbes26a.html)。
[^6]: Kandpal et al. *The Common Pile v0.1*, 2025，§4.1、Project Gutenberg 来源说明。[论文](https://arxiv.org/html/2506.05209v1)。
[^7]: Common Pile. *Project Gutenberg filtered*，数据卡与 revision 文件清单。[数据集](https://huggingface.co/datasets/common-pile/project_gutenberg_filtered/tree/3cdf6879c807f4e4e063f2ceb23bc268d8c29ab7)。许可元数据仍可能存在错误，不能将上游标签视作逐文档法律保证。
[^8]: PyTorch. *Tips for Loading an nn.Module from a Checkpoint*，mmap、meta device 与参数/优化器引用说明。[官方文档](https://docs.pytorch.org/tutorials/recipes/recipes/module_load_state_dict_tips.html)。实际执行版本固定为主机的 torch 2.12.1+cu130，没有因文档显示更新版本而升级运行环境。
[^9]: HuggingFaceTB. *FineMath*，Other Known Limitations。[数据卡](https://huggingface.co/datasets/HuggingFaceTB/finemath)。
[^10]: HuggingFaceFW. *FineWeb-Edu classifier*，Limitations。[模型卡](https://huggingface.co/HuggingFaceFW/fineweb-edu-classifier)。
[^11]: ML Foundations. *DCLM baselines*，fastText filtering、独立打分和阈值设置。[官方实现](https://github.com/mlfoundations/dclm/blob/main/baselines/README.md)；[OH/ELI5 分类器](https://huggingface.co/mlfoundations/fasttext-oh-eli5)。该分类器未用于当前已构建候选，不把文献结果当作本项目的新实测。
