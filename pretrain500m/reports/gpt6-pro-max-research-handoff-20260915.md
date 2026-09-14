# 给 GPT-6 Pro（Max reasoning）的 529M / F2 后续训练研究委托

日期：2026-09-15
代码仓库：`yinli-systems/l20-1b-pretraining`
工作分支：`codex/529m-results`

## 你的任务

请把下面材料当作一个需要独立审计的模型研发项目。先认真联网研究截至
2026-09-15 的一手资料、论文、官方模型卡、数据集卡和训练报告，再给出可以直接
执行的下一阶段方案。硬目标是：

1. 先把当前 528.75M base model 在完全匹配的七项零样本评测上的两种子均值从
   `48.5311%` 稳定提高到 `>50%`；
2. 再用匹配协议超过有代表性的 0.3B--1.1B 开放 base models；
3. continued pretraining、context extension、SFT、preference optimization、RLVR
   和 domain-specific training 都要有清楚的先后顺序、数据预算和停止条件；
4. 实际训练 rolling median MFU 必须严格 `>0.50`，优先使用 4/8 张 RTX 5090；
5. 不要承诺无法证实的“超过所有模型”。请定义比较集合，并把开发结果、封存测试、
   同协议竞品复跑和市场结论分开。

请重点回答：F2 是否仍有上升空间；应继续 F2、转向 F3、做权重插值/模型融合，还是
从同一个 immutable parent 训练新配比；下一批数据该是什么、每类占比多少、多少
prediction tokens、什么 LR/decay/curriculum；怎样修复 WinoGrande 和 ARC-Easy 回退而
保留 BoolQ、ARC-Challenge、PIQA 和 HellaSwag 的增益；怎样避免因我们已经查看过这
七项测试结果而产生自适应过拟合。

## 1. 模型与不可变父检查点

- decoder-only、Llama-compatible、RoPE、SwiGLU、RMSNorm、GQA、tied embeddings；
- `528,748,800` parameters；
- vocabulary `50,280`；context length `2,048`；
- hidden size `1,536`，intermediate size `4,096`，18 layers，12 query heads，
  4 KV heads，head dim 128，RoPE theta 10,000；
- parent step `7,629`，parent prediction tokens `15,999,172,608`；
- parent checkpoint SHA-256：
  `13aa21721e15c48cdfdafe30d8fdd41d9c95c90be766327661d96af1e90dd6cf`；
- parent 的正式七项零样本 unweighted mean 是 `47.9568%`（4-GPU historical run）；
  本次完全匹配的 2-GPU base rerun 是 `47.9952%`，两者只差 `+0.0384 pp`；
- BF16 Hugging Face export 通过 bitwise state parity 和 bounded-logit-drift gate；
- 当前 checkpoint 的架构不能通过 continued pretraining 改动。架构优化必须作为下一代
  from-scratch 模型单独研究；2K -> 4K context extension 也应作为独立实验。

## 2. 已完成的数据工程

三批原始数据审计覆盖 16 类来源、`199,407` rows、`199,372` byte-identical unique
documents。质量和 family gates 后保留 `180,758` rows；最终 selected-source pack 包含
`180,637` documents，并在 packing 前完成 train/development/confirmation 分割。

已实施或冻结的控制包括：source revision pin、内容 hash、exact dedup、cross-source
duplicate audit、canonical URL/document family/code repository 分组、family-disjoint split、
benchmark contamination screening、quality filtering、deterministic token-mixture sampler、
rank-specific RNG、reader cursor 和 exact-resume 验证。每个非旧 FineWeb source 的累计
epoch cap 是 2.0；旧 FineWeb-Edu（父训练已用 15.999B tokens）cap 是 2.2。

主要来源与 pinned revision：

| 类别 | 一手数据源 | pinned revision |
| --- | --- | --- |
| educational web | HuggingFaceFW/fineweb-edu | `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` |
| broad web | HuggingFaceFW/dclm_100BT / DCLM baseline | `01022d378d944de6deeb1c79d08fecb4d27b2c6f` |
| knowledge/PDF | HuggingFaceFW/finepdfs-edu, English | `9cfabe2127faca99b3d5c4dc6d1fcb397399ebde` |
| math | FineMath `finemath-4plus` + `infiwebmath-4plus` | `e92b25a616738fe95dc186b64dfb19f9c8525594` |
| code | Stack-Edu; Software Heritage blob rehydration | `eeec5caac5cc3758a18f1d3ba4416837a9ba814c` |
| multilingual | FineWeb2-HQ | `c0c06e94fd3a44ae9e802b2b0fc533817601eb5e` |

Math 内部配比为 FineMath-4plus 75%、InfiWebMath-4plus 25%。Code 内部配比为
Python 50%、JavaScript 20%、TypeScript 10%、C++ 10%、Java 10%。Multilingual 内部
配比为中文 50%，西班牙语/法语/德语/日语/阿拉伯语各 10%。未知许可证 code 不准入。

Cosmopedia-v2 synthetic data 原设上限是 5%，但因 seed-source label 不能充分证明
parent-document identity，冻结实验中全部排除。这是保守的 provenance 决策，不表示
synthetic data 本身无效；若建议重新加入，必须给出 parent-level 去重与独立 ablation。

## 3. 实际训练配比

所有百分比是 filtering、dedup、split、tokenization 和 packing 之后的 prediction-token
quota；不存在 source exhaustion 后的静默重归一化。

| Recipe | FineWeb-Edu | DCLM | FinePDFs | Math | Code | Multilingual | Synthetic |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| F0 existing-web control | 100% | 0 | 0 | 0 | 0 | 0 | 0 |
| F1 broad English | 35% | 20% | 20% | 15% | 10% | 0 | 0 |
| F2 reasoning | 25% | 15% | 25% | 20% | 15% | 0 | 0 |
| F3 broad multilingual | 25% | 15% | 20% | 15% | 15% | 10% | 0 |

F2 的 `262,144` blocks 精确 quota 为：FineWeb-Edu 65,536；DCLM 39,321；PDF
65,536；FineMath 39,322；InfiWebMath 13,107；Python 19,661；JavaScript 7,865；
TypeScript/C++/Java 各 3,932。F3 使用相同总 block 数，把 PDF 从 25% 降到 20%、
math 从 20% 降到 15%，腾出的 10% 给 multilingual。

## 4. 已完成的训练与系统验证

### 4.1 Exact recovery 与扩展效率

4-GPU restart test 对 model、optimizer、reader cursor、origin、run fingerprint 和所有
rank RNG state 做到 exact recovery。两条 32-step 分支的 final-10-step median MFU 为
`0.77514` 和 `0.77531`。

| GPUs | Nodes | median tokens/s | steady MFU | 结论 |
| ---: | ---: | ---: | ---: | --- |
| 4 | 1 | ~162,000 | ~0.775 | 当前最稳健的训练配置 |
| 8 | 2 | 290,323 | 0.6912 | 可更快，仍明显超过 0.50 gate |
| 16 | 4 | 467,716 | 0.5567 | 总吞吐更高但扩展效率较差 |

因此“至少 50 MFU”已经满足。后续研究请同时优化 wall-clock、tokens/s、checkpoint
频率和本地/共享盘占用，不要仅以卡数最大化为目标。

### 4.2 两种子短筛选

每个 run 从同一 parent 开始，使用 `536,870,912` prediction tokens、context 2,048、
global prediction tokens/step `2,097,152`、4 张 RTX 5090。候选选择规则是在 equal-domain
development loss 上先比较 worst seed，再比较两种子均值。

| Recipe | mean equal-domain loss | worst seed | seed spread |
| --- | ---: | ---: | ---: |
| F0 | 3.312786 | 3.315484 | 0.005397 |
| F1 | 2.702542 | 2.703114 | 0.001144 |
| F2 | 2.689510 | 2.689540 | 0.000059 |
| F3 | **2.571669** | **2.572069** | 0.000799 |

### 4.3 两种子长 confirmation

F2、F3 各两种子，每个 run `1,024` steps、`2,147,483,648` prediction tokens、4 张
独立 RTX 5090。四个 run 共 `8,589,934,592` prediction tokens，全部 Slurm `COMPLETED
0:0`。

| Recipe | seed | job | final-10 MFU | end validation loss | checkpoint SHA prefix |
| --- | ---: | ---: | ---: | ---: | --- |
| F2 | 20260914 | 1589567 | 0.766436 | 2.542319 | `69d9d8d3...a085` |
| F2 | 20260915 | 1590004 | 0.766533 | 2.546584 | `f6a4ff43...1cbc` |
| F3 | 20260914 | 1590005 | 0.775366 | 2.378190 | `194e1da1...222e` |
| F3 | 20260915 | 1589570 | 0.776009 | 2.379313 | `8ee55130...9a21` |

训练采用同 recipe/schedule、公平 seed grid、从同一 parent 重新开始；最终 model-only
checkpoint 各约 2.37 GB。当前 long-confirmation 配置是 BF16 autocast、fused AdamW
(`betas=(0.9,0.95)`, `eps=1e-8`, weight decay 0.1)、peak LR `1e-4`、16,777,216-token
warmup、warmup-stable-cosine-decay（最后 10% decay）、per-GPU microbatch 4、gradient
accumulation 64、global tokens/step 2,097,152、global grad-norm clipping 1.0；PyTorch
compile + DDP native GQA，DDP bucket 25 MB、gradient-as-bucket-view。推荐方案必须重新
检查数据变化后的 LR，不能把当前 LR 假定为新配比的最优值。

## 5. 独立 held-out loss 结果

Slurm job `1590660` 在五个冻结 domain 上评估 base 和精确的 F2/F3 两种子网格，
每个模型 `16,678,912` validation prediction tokens。result SHA-256 是
`32d19eae4cf49b522a954bd9ab083466058f5386ed177ddf009c96da0bb070ef`；selection
SHA-256 是 `478086a7661899be25c7ff1843f1feb91700c6fb40be26d89ff8750a888409bc`。

| Checkpoint family | two-seed equal-domain loss | relative reduction vs base |
| --- | ---: | ---: |
| immutable base | 3.304217 | -- |
| F2 | 2.567941 | 22.28% |
| F3 | **2.402608** | **27.29%** |

F3 loss 比 F2 低 6.44%。F2 在 code、general web、knowledge reading、math 上略好；
F3 的 multilingual loss 比 F2 低约 24.26%，所以 equal-domain loss 明确选 F3。这个结论
只说明 held-out language modeling loss，不等于下游能力。

## 6. 完全匹配的七项零样本结果

Slurm job `1590907` 用两张独立 RTX 5090 对 immutable base、F2 两种子、F3 两种子进行
相同 lm-eval-harness BF16、context 2,048、zero-shot、相同 task versions/prompts/seeds/
sample counts 的复跑；28m38s 完成，exit `0:0`。summary SHA-256：
`7eddae38ce9b1b002ede388dc73c55e078ccbeb38269c5e893609d997e31ab99`。

| Task | matched Base | F2 two-seed mean | F2 delta | F3 two-seed mean | F3 delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| HellaSwag | 42.3422 | 42.9347 | +0.5925 pp | 42.8102 | +0.4680 pp |
| PIQA | 66.8662 | 67.6551 | +0.7889 pp | 67.3830 | +0.5169 pp |
| WinoGrande | 53.6701 | 51.9732 | **-1.6969 pp** | 51.3023 | **-2.3678 pp** |
| OpenBookQA | 33.2000 | 33.7000 | +0.5000 pp | 33.4000 | +0.2000 pp |
| ARC-Easy | 55.2609 | 54.2508 | **-1.0101 pp** | 54.5455 | **-0.7155 pp** |
| ARC-Challenge | 28.3276 | 29.4795 | +1.1519 pp | 29.5222 | +1.1945 pp |
| BoolQ | 56.2997 | 59.7248 | +3.4251 pp | 59.6942 | +3.3945 pp |
| **unweighted mean** | **47.9952** | **48.5311** | **+0.5359 pp** | **48.3796** | **+0.3844 pp** |

F2 两个 seed 分别为 `48.5191%`、`48.5432%`，spread 仅 `0.0240 pp`；F3 为
`48.2961%`、`48.4631%`，spread `0.1670 pp`。按“更高 worst seed，再看均值”的冻结规则，
F2 排第一。但正式 promotion 仍是 `false`：总增益小于当前统计不确定性，且 WinoGrande、
ARC-Easy 回退。当前最合理的结论是 F2 有稳定的正方向信号，不能宣称整体能力已显著提升。

更关键的是：F3 是 held-out loss winner，F2 是七项 accuracy winner。这是下一阶段应研究
的真实 trade-off，不能只看一个 aggregate。

## 7. 公开竞品参照（需你重新联网核验）

下面数字用于设定研究范围，不是 matched rerun：

- TinyLlama 官方模型卡在相近七项表中报告：Pythia-1B（300B tokens）48.30，
  TinyLlama-1.1B（503B）48.28，1.007T checkpoint 50.22，最终 3T checkpoint 52.99：
  <https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T>
- MicroLlama-300M（50B tokens）官方卡相近七项平均约 42.36：
  <https://huggingface.co/keeeeenw/MicroLlama>
- SmolLM2-360M（4T tokens）官方卡 zero-shot 表中 HellaSwag 54.5、ARC average 53.0、
  PIQA 71.7、WinoGrande 52.5、OpenBookQA 37.4：
  <https://huggingface.co/HuggingFaceTB/SmolLM2-360M>
- 同一 SmolLM2 表中的 Qwen2.5-0.5B 为 HellaSwag 51.2、ARC average 45.4、PIQA 69.9、
  WinoGrande 54.1、OpenBookQA 37.4；Qwen 系列公开训练规模很大，但 0.5B 精确 token
  口径需要核查：<https://qwenlm.github.io/blog/qwen2.5/>
- OLMo-1B 官方模型卡给出的相近七项数值换算平均约 56.4，训练约 3T tokens：
  <https://huggingface.co/allenai/OLMo-1B>

这些表可能使用不同 harness/version/normalization/ARC aggregation，所以只能说明：当前
F2 以约 18.15B cumulative prediction tokens（parent 15.999B + CPT 2.147B）达到接近早期
1B 模型的水平，token efficiency 有研究价值；它尚未超过现代强 0.5B--1B base models。
请寻找截至 2026-09-15 更强且可复跑的同规模开放模型，并设计 exact matched protocol。

## 8. 已知负面结果、失败与约束

1. 初始只用 development loss 会选 F3；下游 accuracy 却选 F2，说明 loss 不能替代能力。
2. F2/F3 都损失 WinoGrande 和 ARC-Easy，平均分会掩盖 domain regression。
3. 两种子 aggregate gain 只有 0.38--0.54 pp，不能忽略 benchmark sampling uncertainty。
4. 七项 final set 已被查看，今后不能继续用这些 label/prompt 做 mixture、LR 或 curriculum
   selection；需要新的 contamination-screened proxy development families，最终只做一次
   独立验证，最好再增加未见过的 sealed suite。
5. Synthetic source 因 parent identity 不足被移除；不能把它当作现成高质量数据。
6. 当前只有 base-model multiple-choice accuracy 和 LM loss；instruction following、safety、
   calibration、tool use、code execution、multilingual generation、long context 尚未验证。
7. SFT、DPO/IPO/ORPO、RLVR 都还没有开始。当前所谓 `posttrain` 目录中的工作实质上是
   continued pretraining 与 base-model evaluation。
8. 16-GPU MFU 虽通过 >0.50，但通信损失明显；当前训练应优先 4 或 8 GPU。
9. 磁盘曾是现实约束：方案必须按阶段估计原始数据、packed data、optimizer checkpoint、
   model-only checkpoint、HF export、eval sample logs 和 cache 的峰值空间；删除必须绑定
   active-job/ownership/path 检查，保留 hashes/receipts/最终 checkpoints。

## 9. 现有下一阶段假设（请批判，不要照抄）

原计划是先做 20B prediction-token tranche，若多域能力与 retention 通过，再考虑 50B、
100B、300B milestones；source repeat 初始 cap 2.0。候选 curriculum 是前 70% broad
static mixture，中间 20% 增加高质量 strata，最后 10% LR decay，并和 static control 比较。
context extension 是先 2K -> 4K，在完整长文档上重 pack，验证 retrieval 和短上下文 retention，
通过后才考虑 8K。

初始 SFT 假设是 100k--300k samples，按 assistant target tokens 计量：general dialogue /
knowledge 30%、short verified math 20%、tested code explanation 20%、structured output/tools
15%、multilingual 10%、calibration/safety 5%。长答案不允许截断；teacher outputs 必须验证。
之后才做 checked preference pairs 和在 base success 非零时的 RLVR pilot。如果 reward 全空，
返回 teaching data/easier problems，不盲目增加 rollouts。

这些只是预注册起点。请用最新证据重新决定 token budget、配比、来源、质量分类器、退火策略、
LR、warmup、weight decay、batch、seed 数和每个 gate。

## 10. 你必须交付的结果

请输出一份工程可执行、证据可审计的研究报告，至少包括：

1. **结论先行**：F2 还能提高多少、最大不确定性是什么、为什么；
2. **竞品定义与联网研究表**：0.3B--1.1B base models，参数、tokens、context、数据、七项或
   最接近任务、评测工具/版本、官方链接、可否 exact rerun；区分 apples-to-apples 与参考值；
3. **根因分析**：解释 F2/F3 loss/accuracy 分歧及 WinoGrande、ARC-Easy 回退的最可能原因，
   给出可证伪实验；
4. **三到五个候选 CPT 配比**：prediction-token 百分比精确到总和 100%，明确每个 source、
   quality stratum、revision/license/provenance、去重与 repeat cap；至少包含 F2 control、
   retention-repair arm 和你认为最强的新 arm；
5. **多保真训练矩阵**：pilot、两种子 confirmation、长 tranche 各用多少 tokens/GPU/time，
   LR range、warmup/decay、checkpoint cadence、预估峰值磁盘；在 4/8/16 张 5090 下保证
   MFU >0.50 的具体策略；
6. **开发与封存评测**：禁止用已查看的七项 final set 调参；建立未见 proxy suites、paired
   bootstrap、per-family non-inferiority、contamination audit，并定义何时允许唯一一次 final；
7. **>50 的 go/no-go tree**：每阶段预期增益范围和置信度；若 50 不可由现 checkpoint 合理
   达到，要清楚说明何时停止 CPT 并启动 next-from-scratch 架构；
8. **超过竞品的 matched rerun**：列出具体开放 checkpoints、精确 protocol、tokenizer/context
   差异处理、统计检验和不允许使用的宣传措辞；
9. **后续 post-training**：只有 base gate 通过后，给出 SFT -> preference -> RLVR 的数据配比、
   样本/target-token budget、验证器、safety/calibration/retention gates；
10. **domain-specific 极致路线**：math、code、knowledge、multilingual、tool use 各给出独立
    capability ceiling、数据和可执行验证，不以 aggregate 掩盖退化；
11. **优先级清单**：按“预期收益 / GPU 小时 / 风险”排序，前 72 小时和后 2--6 周分别做什么；
12. **最终推荐配置**：给一份可直接转成 manifest 的 JSON-like 配比和一份训练命令参数表。

每个关键建议必须说明证据来源、预期收益范围、失败信号与回滚点。若公开资料不足，明确写
“unknown”，不要补造数字。目标是形成能推动 F2 真实进步的研究方案，而不是把 48.53 包装成
已经超过竞品。

## 11. 可审计仓库证据

- `pretrain500m/RESULTS.md`
- `pretrain500m/model.py`
- `pretrain500m/research/data-mixture-v1/RESEARCH.md`
- `pretrain500m/research/data-mixture-v1/experiment-design.json`
- `pretrain500m/posttrain/intake-expansion-v1/F2-screen-manifest.json`
- `pretrain500m/posttrain/intake-expansion-v1/F3-screen-manifest.json`
- `pretrain500m/reports/frozen-two-seed-screen-selection-v1.json`
- `pretrain500m/reports/long-confirmation-training-20260914.json`
- `pretrain500m/reports/long-confirmation-heldout-results-1590660.json`
- `pretrain500m/reports/long-confirmation-heldout-selection-1590660.json`
- `pretrain500m/reports/seven-task-confirmation-2gpu-results-1590907.json`
