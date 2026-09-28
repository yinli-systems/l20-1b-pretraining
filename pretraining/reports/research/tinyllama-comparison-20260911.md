# 20B 模型与 TinyLlama：同协议补评测

## 状态与结论

2026-09-11：本轮七个评测任务全部完成，包括补测我们的 BoolQ，以及两个 TinyLlama 的九项基础对照、MMLU 5-shot 和 GSM8K 5-shot。已有的三套原始评测经 SHA-256 校验后复用，没有重跑或覆盖。

当前模型在标准七项集合上的均分为 51.26%，低于同机重测的 TinyLlama 3T（52.78%）和 v1.1（53.57%）。ARC-E 是明确优势；HellaSwag、Winogrande、LAMBADA 是明显短板。不能据此声称已经全面超过 TinyLlama。

## 实测结果

以下 TinyLlama 数字均为本轮重测，不是从模型卡复制。百分数保留两位小数；主集合使用 TinyLlama 官方 GPT4All 的七项定义，混用的 acc/acc_norm 已明确列出。

| 任务 / 指标 | 题数 | 我们 20B | TinyLlama 3T | TinyLlama v1.1 |
|---|---:|---:|---:|---:|
| HellaSwag acc_norm | 10,042 | 45.13 | 59.09 | 61.44 |
| OpenBookQA acc_norm | 500 | 35.00 | 35.80 | 36.00 |
| Winogrande acc | 1,267 | 52.17 | 58.88 | 59.19 |
| ARC-Challenge acc_norm | 1,172 | 33.62 | 30.29 | 32.94 |
| ARC-Easy acc_norm | 2,376 | 64.14 | 55.26 | 55.77 |
| BoolQ acc | 3,270 | 59.17 | 57.37 | 56.06 |
| PIQA acc_norm | 1,838 | 69.59 | 72.74 | 73.61 |
| 七项等权均分 | — | 51.26 | 52.78 | 53.57 |
| LAMBADA acc | 5,153 | 46.26 | 58.88 | 56.10 |
| TruthfulQA MC2 | 817 | 36.91 | 37.56 | 35.06 |
| MMLU 5-shot acc | 14,042 | 25.31 | 25.30 | 26.07 |
| GSM8K 5-shot flexible extraction | 1,319 | 1.67 | 1.90 | 2.50 |
| GSM8K 5-shot strict match | 1,319 | 0.53 | 1.06 | 1.74 |

MMLU 汇总按各学科样本数加权；GSM8K 的两行是同一批生成结果的不同答案抽取规则。扩展任务不混入七项均分。跨 tokenizer 的 PPL 不直接排序，原始字段另行完整保存。

## 怎样理解差距

对每道题按 doc_id、doc_hash 对齐，并核对 prompt_hash、target_hash、任务配置、任务版本和 n-shot。汇总前还重算逐题指标均值，检查是否与原始结果一致；不满足就拒绝生成比较。

七项均分使用逐题配对 bootstrap，按任务分层后等权聚合，共 10,000 次，种子 20260911。区间反映测试样本不确定性，不代表重新训练或其他提示下的波动；没有作多重比较校正。

对旧版 3T，我们落后 1.52 分，95% 配对区间约为落后 0.66–2.41 分。对 v1.1，落后 2.31 分，95% 区间约为落后 1.42–3.20 分。较小的 OpenBookQA、ARC-C 对 v1.1 等差异不支持强胜出结论；尤其不能只保留某个正向点估计。

MMLU 对 3T 仅多答对 2 题，差值区间约 [-0.78, 0.80] 分；对 v1.1 少答对 107 题，差值区间约 [-1.78, 0.24] 分。两者都不足以作强优劣判断。GSM8K flexible 分别答对 22/25/33 题，三者都尚未展示可靠的数学推理能力。

## 答案偏好与去污染边界

BoolQ 有 2,033 个 yes 和 1,237 个 no；一律回答 yes 即可达到 62.17%。三款模型在此协议下均低于这一朴素基线。我们的模型预测 yes 共 2,384 次（72.91%），所以 BoolQ 相对领先不能单独说明强阅读理解。

当前模型在 MMLU 的 A/B/C/D 预测次数为 2,851/8,591/2,056/544，约 61.18% 选择 B。真实标签中 D 最多，一律选 D 可得 26.89%，同样高于我们的 25.31%。这反映了当前协议下的弱表现与答案偏好；仅靠这些结果不能分离知识不足、提示格式、截断和概率校准的贡献，也不能承诺换个格式就能解决。

原 20B 的去污染清单未显式包括 BoolQ。它的 13-word 匹配也不能保证排除短问题、转述与上游污染。新增的 BoolQ 评测不能被描述为已经补上了旧训练数据的去污染。分析文件另给出去除 BoolQ 的六项敏感性结果，但不会在看到结果后拿它替换正式七项集合。

## 复现协议

- 模型：我们的冻结 20B 最终 HF 导出；两个 TinyLlama 均固定 revision。
- 环境：`lm_eval==0.4.9`、BF16、2048 上下文、`batch_size=auto:4`，不用 chat template。
- 软件：Python 3.12.3、PyTorch 2.12.1+cu130、Transformers 4.56.2、Datasets 3.6.0、Tokenizers 0.22.2、NumPy 2.5.2。原结果与新结果内嵌的 PyTorch/CUDA/Python 环境信息一致。
- 种子：Python/NumPy/Torch 为 42，few-shot 为 1234；后者与旧 LitGPT 包装器实际默认值一致。
- 基础九项为 0-shot；MMLU、GSM8K 为 5-shot。全部样本，无 `limit`。
- GSM8K 使用原任务的确定性生成和原停止条件，两种抽取指标同时报告。
- 2048 是 token 上限，不同 tokenizer 的左截断可能不同；MMLU 已观察到长输入截断。

| 模型 | 固定 revision |
|---|---|
| `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T` | `59f6f375b26bde864a6ca194a9a3044570490064` |
| `TinyLlama/TinyLlama_v1.1` | `ff3c701f2424c7625fdefb9dd470f45ef18b02d6` |

执行脚本：[evaluate_comparison.py](../../evaluate_comparison.py)；分析脚本：[analyze_comparison.py](../../analyze_comparison.py)。原有三套结果先校验 SHA-256，再复用；不会覆盖原最终 checkpoint 或原始评测。新评测使用独立目录、单写者锁、原子落盘和逐任务收据，可以校验后跳过已完成任务。

证据文件：[完整紧凑指标与配对区间](../metrics/tinyllama-comparison-20260911.json)、[评测收据](../receipts/tinyllama-comparison-20260911.json)。摘要还记录了模型/tokenizer 的评测后哈希、分析软件版本与收据哈希；不把评测后采集的哈希冒充评测前已经登记的哈希。含题目文本和逐题输出的大 JSON 留在 GPU 主机，不加入普通 Git 历史。

交付验证：7/7 评测任务完成；全部逐题与配置对齐通过；传回本地的两份 JSON 与主机 SHA-256 一致；22 项本地测试及空白检查通过。本轮新增报告和脚本尚未 commit/push，也没有将本地测试描述成 GitHub CI 结果。

这轮只做评测，没有继续训练，没有消耗额外 2B 训练 token 预算，也没有加载 TinyLlama 权重到我们的模型中。HumanEval/MBPP、聊天质量和新权重平均候选不在这轮已完成范围内。

结论只针对上表两款固定 checkpoint，不覆盖 TinyLlama 的全部中间 checkpoint、Math&Code 或 Chat 变体；也不代表当前所有 1B 级模型的上限。

## 对后续 2B 的影响

优先测试广泛语言覆盖与原能力保留，而不是为某个评分偏好定制答案。保留 20B 基线，用严格计费的 A/B 小试决定后续；“保持已有文件”可以保证，“新模型不遗忘任何能力”不能事先保证。详细语料、学习率、分支 token 算术、磁盘约束和研究来源见 [2B 继续预训练研究](2b-continuation-plan.md)。

官方主集合定义见 [TinyLlama EVAL.md](https://github.com/jzhang38/TinyLlama/blob/main/EVAL.md)；v1.1 的版本与训练阶段说明见 [官方模型卡](https://huggingface.co/TinyLlama/TinyLlama_v1.1)。两者的公布分数仅作为历史参考，本报告比较采用本机实测。
