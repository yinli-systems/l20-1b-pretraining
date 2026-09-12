# 单张 NVIDIA L20 上的 1.1B 高质量后训练路线

## 结论

最可靠的路线不是立即复制大集群的完整 SFT→DPO→两轮 2M-episode RLVR，而是先完成基座选择，再按可中止的小规模阶梯推进：高质量 reasoning bootstrap、双模式 SFT、同数据 DPO/APO 小试、最多 10k→50k→100k episode 的单卡 RLVR、成功轨迹再蒸馏，最后才做 checkpoint soup 与少量 base merge。

公开证据支持这些方法值得测试，但不支持“必然超过所有 1B 模型”。尤其是 OLMo 2 1B 的每轮 2M episode RLVR 使用 16 张 H100 约 43 小时；这相当于每轮约 688 H100 GPU-hours。把它原样搬到一张 L20 上既不是两天任务，也不是本项目当前最优的边际投入。[^1]

当前最先要解决的仍是 Stage 0。continuation A 的七任务均分较 20B 基座下降 0.2549 个百分点，配对区间跨零，未获晋级；B 曾按用户要求在第 61 步安全暂停。2026-09-12 本次检查时，Tailscale 地址返回 502/超时，公网 SSH 入口也超时，因此 B 的现时进程、GPU、checkpoint 与账本无法重新核验。本报告不把历史状态冒充实时状态，也不授权无验证自动恢复。

## 公开证据能说明什么

### OLMo 2 1B：多阶段后训练有效，但算力尺度不能忽略

OLMo 2 1B 是最直接的同规模公开参照。其官方模型卡报告，SFT、DPO、RLVR 三阶段的十项平均分从 36.9 到 40.6，再到 42.7；GSM8K 为 52.1→59.0→68.3，IFEval 为 50.5→67.1→70.1，MATH 为 13.2→14.1→20.7。官方 open-instruct 配置给出 SFT 学习率 3e-5、DPO 2.5e-6，以及 RLVR 5e-7、16 rollouts、2M episodes 等起始参考。[^1][^2]

这些数字证明阶段化训练可以累积收益，但不能直接预测本模型。OLMo 2 的基座训练量、tokenizer、数据、上下文和后训练算力不同；其 SFT 用 8×H100 约 9 小时，DPO 用 8×H100 约 2 小时，每轮 RLVR 用 16×H100 约 43 小时。对单张 L20，正确做法是复用实验结构和监控指标，而非复刻总 episode 数。

### reasoning distillation：对小模型优先级高于从弱 policy 冷启动 RL

DeepSeek-R1 官方报告明确区分了两件事：大模型可以通过大规模 RL 涌现推理行为；对小模型，将强模型的 reasoning patterns 蒸馏进去，比让小模型自行通过 RL 发现这些模式更有效。其公开蒸馏模型以已有 Qwen/Llama 基座为起点，因此证据支持“先教会，再探索”，但不证明任意 1B from-zero base 会得到相同成绩。[^3]

OpenThoughts3-1.2M 的官方数据卡说明，它通过系统实验比较问题来源、筛选和回答生成，并用于 OpenThinker3 系列。它适合作为候选池，不应整体无筛选导入。每个候选样本仍要保留 revision、许可、问题来源、teacher、verifier、长度和 problem-family 信息；最终答案通过之外，还要排除截断、循环、伪证明、答案泄漏与公开 benchmark family 重合。[^4]

NVIDIA Nemotron Post-Training Dataset v1 覆盖 math、code、STEM、general reasoning 与 tool calling，采用 CC BY 4.0，数据卡同时声明它是合成数据。数据规模约 25.7M 行、203 GB，远超本项目需要，也强化了“按领域与验证质量抽取”而不是整库下载训练的必要性。[^5]

2026-09-12 通过 Hugging Face 官方 API 冻结的候选 revision 分别为：OpenThoughts3 `61bcf9d4eb38b30295efc2021227a63cc5bb34c8`（Apache-2.0）、Nemotron v1 `74e23eb6f830fef4a9e96a92f6f6262214cbb9a8`（CC BY 4.0）、Tülu 3 OLMo 2 mixture `d91a0785ade02942520280fb484866fce41e448f`（数据集级未声明统一 license）、RLVR mixed constraints `7dbd180f5440c0b90f2944e6efea934b85437a95`（ODC-BY）。因此 Tülu mixture 当前保持阻断，必须逐组件完成许可与 provenance 审计；revision pin 也不等于质量入场。

### 双模式与 replay：是待验证的保留设计，不是免费增益

Qwen3 的官方 post-training 路线为 long-CoT cold start、reasoning RL、thinking/non-thinking fusion、general RL；这支持同时教 reasoning 和 direct-answer 行为。SmolLM3 也提供 `/think` 与 `/no_think`，其 SFT 数据为 1.8B tokens、训练约四个 epoch，并在 post-training 后通过 checkpoint soup 加 10% mid-training checkpoint 恢复长上下文能力。[^6][^7]

本项目上下文只有 2048，不能宣称复制 SmolLM3 的 merge 就能恢复 128K 能力。可迁移的原则是：分别评测 think/no-think，保留 5% causal base replay，预注册 5/10/15% base merge，并只在开发集选比例。任何 merge 都必须重跑最终协议，不能因为权重插值“零训练成本”就视作零验证成本。

### DPO/APO：必须同数据比较

APO 论文显示，偏好对越具有真正对比性，学习信号越好，并给出可控制模型相对参考策略移动方向的目标；TRL 当前也原生提供 `apo_zero` 和 `apo_down`。SmolLM3 报告其内部 ablation 中 APO 更稳定且下游更好，但这来自 3B 模型和其自有数据。[^7][^8][^9]

因此本项目只预注册 `DPO vs APO-zero` 同数据 pilot，不能先从不同来源各找一套数据再比较算法。chosen 应为验证通过的高质量答案；rejected 优先取模型自己的“接近正确但存在可定位错误”的回答。完全垃圾的 rejected 容易制造风格捷径。对于无法确定 chosen 是否优于当前 policy 默认答案的数据，不应机械使用 APO-zero。

### 单卡 RLVR：生成是瓶颈，先验证信号密度

TRL 的 GRPO 文档支持 vLLM colocate 模式；它能避免独立 GPU server，但会争用训练显存。sleep mode 可降低显存压力，却增加权重搬运延迟。TRL 还明确指出应按实际模型调整 vLLM memory utilization。[^10]

在 46 GB L20 上，1.1B 全参数 policy、优化器、梯度、rollout engine 和 KV cache 理论上可通过 BF16、gradient checkpointing、colocate 与受控长度组合尝试，但“能放下”不等于“吞吐高”。第一轮必须实测 10k episodes，4 rollouts/prompt、最多 1024 completion tokens；同时记录生成 tokens/s、训练 tokens/s、端到端 episodes/s、显存、功耗、reward、KL、entropy、response length 与 verifier pass rate。只有 held-out pass rate 上升且未发生长度投机、格式投机或能力遗忘，才扩大到 50k，再考虑 100k。

## 本项目的阶段协议

机器可读协议位于根目录 `posttraining_protocol.json`，并由 `validate_posttraining_protocol.py` fail closed 校验。它刻意把文学路线图改成实验合同。

| 阶段 | 首轮预算 | 晋级依据 |
|---|---:|---|
| 0 基座选择 | 完成 B 既定 pilot 与评测 | B 必须通过冻结的 PPL、七任务、扩展任务、逐样本 CI 与身份核验；否则回退 20B |
| 1 reasoning bootstrap | 3 个 LR arm，各 20M target tokens | GSM8K/MATH/code/IF 开发集改善，base retention 合格，无污染 |
| 2 dual-mode SFT | 300M–600M target tokens | think 与 no-think 均可用，IFEval/数学/代码改善，base suite 不发生预注册的实质退步 |
| 3 preference | DPO/APO 各 10k pairs | 同数据、同计算比较；held-out preference、IFEval、长度与 retention 决定赢家 |
| 4 RLVR | 10k→50k→100k episodes | reward 与 held-out verifier 同升，KL/entropy/长度稳定，非 RL 任务保留 |
| 5 再蒸馏 | 10k–50k 成功轨迹 | 以更低服务成本复现 RL 收益，或进一步改善保留能力 |
| 6 soup/merge | base 权重 5/10/15% | 只用开发集选，最终集一次性评估 |

Stage 1 的 1e-5/2e-5/3e-5 和 Stage 3 的 1e-6/2.5e-6/5e-6 是 pilot 网格，不是预判答案。Stage 2 的 300M–600M target tokens 低于原文的 1–2B，是单卡约束下更合理的第一轮：先估计每 1M target tokens 带来的真实 capability gain，再决定是否继续。训练日志还应同时报告总输入 tokens、有效 target tokens 与 replay tokens，避免用不同分母包装效率。

## 数据设计

### reasoning bootstrap

候选数据从 OpenThoughts3 与 Nemotron 的 math/code/STEM/general reasoning 子集中抽取，但采用两级 gate。第一层是确定性检查：许可与 revision 可追溯、答案格式可解析、math/code verifier 通过、无截断/循环、长度 256–1500 target tokens 为主。第二层是 problem-family 去重与污染检查：不仅比较最终文本 hash，还追踪原题来源、模板、数字替换家族和 synthetic seed。

原文建议 100k–300k traces、300M–800M tokens 可作为候选池上限，不应提前视作训练配额。1.1B 模型是否能吸收超长 teacher trace 必须通过长度桶消融确认；首轮将大多数 target 限制在 2048 context 内，过长答案不能静默截断后仍标记“verified”。

### dual-mode SFT

首个完整候选配比为 30% general instruction、25% math/STEM reasoning、15% code reasoning、10% general reasoning、8% rewrite/summarization/QA、5% tool/structured output、2% multi-turn、5% raw causal replay。chat 数据只对 assistant/completion 计算 loss；replay 使用独立 causal-LM loader，不把原始网页伪装成对话。

TRL SFTTrainer 支持 packing、assistant-only loss 与 completion-only loss；其当前 chunked NLL 可减少大词表 logits 的显存峰值。是否采用 Liger、activation offloading 或 gradient checkpointing 必须由本机吞吐基准决定，因为节省显存与提高单卡 tok/s 不是同一个结论。[^11][^12]

### preference 与 RL 数据

偏好 prompt 与 SFT 训练题按 problem family 分组切分。每个 prompt 初始生成 4 个候选；数学使用 exact/symbolic checker，代码运行在禁网、限时、限内存的容器，instruction constraints 使用确定性规则。teacher judge 只能作为辅助标签，必须保存模型、prompt、sampling 参数与原始判断；teacher 一票通过不能替代可执行验证。

RLVR 只从当前 policy 成功率约 10%–60% 的题池采样。接近 0% 的题几乎没有正轨迹，接近 100% 的题学习信号不足。采样器需要定期重估难度，且 held-out verifier 集不可用于挑训练题或调 reward。

## 评测与统计

Base 论文继续使用冻结的 HellaSwag、ARC-E、ARC-C、PIQA、Winogrande、OpenBookQA、BoolQ 协议；post-training 主表使用 IFEval、GSM8K、MATH500、BBH、HumanEval、MBPP，并把 MMLU、GPQA、LiveCodeBench 与一个未参与调参的 instruction 集留到最终阶段。

报告必须给出解码配置、样本数、prompt template、task revision、执行器版本和逐样本结果。accuracy 类任务使用配对 bootstrap 或适当的二项区间；同一模型多个随机生成的 pass@k 不能假装成训练种子不确定性。至少三次训练种子之前，不能把单次 checkpoint 的 CI 描述为训练可复现性。

Stage gate 不使用“总平均略涨”作为唯一条件。必须单列：instruction、math、code、base retention、长度、拒答/格式、污染审计、吞吐与总 compute。最终模型若只在 reasoning benchmark 上胜出，就只发布 reasoning-specific claim；不得改写成“超过所有 1B 模型”。

## 时间与算力边界

20B 预训练曾在 L20 上测得约 12.8k prediction tok/s，这只是同一训练栈的点时基准。SFT 的 masking/packing、DPO 的参考策略前向、GRPO 的自回归生成都会改变瓶颈，不能直接复用 71.17% MFU 或 12.8k tok/s。

纯 SFT 若实测 6k–10k target tok/s，300M–600M target tokens 约为 8–28 小时，不含数据构建、评测和 checkpoint。偏好与 RL 必须先测 wall-clock；尤其 OLMo 的公开尺度已经说明两轮 2M episodes 对单卡并不现实。单卡首版完整 pipeline 更合理的规划单位是数天到数周，而非承诺两天完成“远超所有 1B”。

## 当前执行边界

本次已经完成研究核验、机器可读协议和协议测试；没有启动新训练。原因不是方案缺失，而是两个已知 SSH 入口均超时，无法验证当前 GPU、B 的 step-61 checkpoint、预算账本、磁盘余量与是否存在其他 CUDA writer。恢复连接后的第一条安全路径是只读检查五类证据：进程、GPU、B status/checkpoint、预算 SQLite、磁盘；然后仅恢复 B pilot 到 190，完成冻结评测后选 parent。

在 Stage 0 通过前，不下载 203 GB Nemotron 全库，不生成 teacher 数据，不更改 tokenizer，不删除 20B/A/B 证据，也不启动 SFT。这样做保护现有可复现结果，并避免把后训练数据选择反向泄漏到基座比较。

## Sources

[^1]: Allen Institute for AI. [OLMo 2 training commands](https://github.com/allenai/open-instruct/blob/main/docs/olmo2.md). Official repository; SFT, DPO and two-stage RLVR configurations and hardware/time.
[^2]: Allen Institute for AI. [OLMo-2-0425-1B-Instruct model card](https://huggingface.co/allenai/OLMo-2-0425-1B-Instruct). Official model card; stage-by-stage benchmark table.
[^3]: DeepSeek AI. [DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1). Official repository and paper links; cold start, RL and small-model distillation claim.
[^4]: OpenThoughts. [OpenThoughts3-1.2M dataset card](https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M). Official dataset card; generation and selection design.
[^5]: NVIDIA. [Nemotron Post-Training Dataset v1](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v1). Official dataset card; domains, scale, synthetic characterization and CC BY 4.0 license.
[^6]: Qwen Team. [Qwen3: Think Deeper, Act Faster](https://qwenlm.github.io/blog/qwen3/). Official release; four-stage post-training and thinking/non-thinking fusion.
[^7]: Hugging Face. [SmolLM3: smol, multilingual, long-context reasoner](https://huggingface.co/blog/smollm3). Official technical blog; dual mode, SFT scale, APO and merge recipe.
[^8]: D'Oosterlinck et al. [Anchored Preference Optimization and Contrastive Revisions](https://arxiv.org/abs/2408.06266). 2024; APO and contrastive preference evidence.
[^9]: Hugging Face. [TRL DPO configuration](https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_config.py). Official implementation; supported APO objectives and reference-logprob options.
[^10]: Hugging Face. [TRL GRPO Trainer](https://huggingface.co/docs/trl/grpo_trainer). Official documentation; colocated vLLM, memory contention and sleep mode.
[^11]: Hugging Face. [TRL SFT Trainer](https://huggingface.co/docs/trl/sft_trainer). Official documentation; packing and assistant/completion-only losses.
[^12]: Hugging Face. [TRL reducing memory usage](https://huggingface.co/docs/trl/reducing_memory_usage). Official documentation; chunked NLL, activation offloading and checkpointing tradeoffs.
[^13]: Chen, Gao, and Wu. [Towards Revealing the Effectiveness of Small-Scale Fine-tuning in R1-style Reinforcement Learning](https://arxiv.org/abs/2505.17988). 2025; re-distillation experiments and scope.
