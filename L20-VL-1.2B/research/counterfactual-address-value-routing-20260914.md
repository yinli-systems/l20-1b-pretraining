# 1.1B / 49-token VLM：从负结果到下一条可证伪路线

研究日期：2026-09-14。范围：L20-VL-1.2B 的对象—属性绑定；单张 NVIDIA L20；不打开 final test。本文区分已有工作、当前证据和候选假设，不将检索未发现表述为全球首创证明。

## 结论

目前最强的证据不是“残差还不够大”，而是：**视觉局部属性在 196→49→语言投影后仍近乎完美线性可解码，但现有 1.1B 语言路径不会根据问题稳定寻址并读取它。**

因此下一步不应继续堆叠 query-blind residual。最值得测试的候选是 **Counterfactual Address–Value Routing（CAVR，反事实地址—值路由）**：

1. 保持 49 个视觉 token 不变；
2. 用问题生成一个很小的空间路由分布，先决定“读哪个对象”；
3. 将“地址”与“对象属性值”分开，在配对属性交换中要求地址保持、值和答案按交换变化；
4. 同时训练语言 LoRA，因为当前语言路径即使看到精确结构化场景，也被固定的 yes 偏置支配。

“问题引导视觉 token”“注意力监督”“反事实交换”“binding ID”分别都已有工作。潜在研究贡献只能是：**在固定 49-token、小语言模型、受控属性交换下，先实证区分信息存在与寻址失败，再用地址不变/值等变的干预目标实现可迁移修复，并在同数据同计算强基线上显著取胜。** 这仍是候选，不是已证明创新。

## 当前新增证据

### E1：低秩 residual 与 IIBR 均未修复

冻结评测摘要：[binding-e1-evaluation-v1-summary.json](../evidence/binding-e1-evaluation-v1-summary.json)，SHA256 `15a7a265415ac7c62f98a963616afb1ee5b1dd3f95b69a10d6bc48dec53263db`。

| mechanism-dev | residual CE | IIBR |
|---|---:|---:|
| 受影响问题，交换前后 joint | 4.6875% | 6.2500% |
| 真图−错配图 | -1.5625pp，95% CI [-4.6875, +1.5625] | -3.1250pp，95% CI [-7.1875, +0.9375] |
| 六题全对的 scene pair | 0% | 0% |

IIBR−residual CE 为 `+1.5625pp`，95% CI `[-1.875, +5.3125]`，不显著。IIBR 还多一次语言模型计算，连 step-matched 优势都没有，更不需要补 GPU-time-matched 胜利对照。selection-dev 上 IIBR 的受影响 joint 为 7.5%，真图−错配图为 -1.875pp，95% CI [-5.625, +1.875]。final test 未生成、未使用。

### 冻结表示探针：信息没有在压缩时消失

结果：[binding-representation-probe-v1.json](../evidence/binding-representation-probe-v1.json)，SHA256 `af303c0843672ebd8e062f362b514bc4073a729db1db107dad57b140a98cb1a3`。

线性 probe 只得到目标位置 oracle，不得到答案；800 个 train scene pair 拟合，160 mechanism-dev 与 160 selection-dev 分组评测。六分类随机水平为 16.67%。

| 目标位置颜色解码 | mechanism-dev | selection-dev |
|---|---:|---:|
| SigLIP2 196-token local | 100.00% | 100.00% |
| 49-token compressed local | 100.00% | 99.6875% |
| 49-token projected local | 100.00% | 99.6875% |

mechanism-dev 上两个阶段差值均为 0；selection-dev 上压缩相对 196 仅 -0.3125pp，95% CI [-0.9375, 0]，投影相对压缩为 0。局部 probe 比全局平均 probe 高约 49–59pp，且所有 CI 下界远高于 0。

这支持“局部信息存在但没有被正确查询”的诊断，不支持“49-token 无损”“probe 表示足够因果使用”或“真实图也会相同”。

### 文本 oracle：语言路径有弱方向信号，但输出被答案先验压倒

结果：[binding-text-oracle-v1.json](../evidence/binding-text-oracle-v1.json)，SHA256 `d869252b49a77f70b0a7ce579477470dd45291c5a348ab2c116ee57b8f499d8e`。

把每个对象的精确颜色、形状、x/y 坐标作为文本提供，纯 base 与 Stage-D adapter 都对 2,880 个变体预测 `yes`，所以受影响问题 joint 均为 0%，普通单题准确率均为 50%。中性提示的先验校准仍不能翻转这一行为。

但交换前后 yes−no 分数的变化方向不是完全随机：

| 文本路径 | 方向正确率 | 相对 50% 的 cluster-bootstrap 95% CI | 有符号 margin 变化均值，95% CI |
|---|---:|---:|---:|
| language base | 59.0625% | [+3.75, +14.375]pp | 0.01131 [0.00616, 0.01638] |
| Stage-D adapter | 62.5000% | [+7.1875, +17.5]pp | 0.01437 [0.00970, 0.01899] |

所以语言模型并非对记录完全无感，而是信号太弱，无法克服约 1.7–2.1 的 prompt-specific yes 偏置。文本 oracle 只有一种结构化序列化，失败也可能部分来自提示格式；它不是自然语言能力总评。

## 文献查重后，哪些不能叫创新

- [QMoP](https://arxiv.org/html/2603.21232) 已根据图像和文本查询路由 pooling、resampler、pruning 三种 projector；“query-guided projector/router”不是新点。
- [FlashVLM](https://arxiv.org/html/2512.20561) 已用投影图像 token 与文本 embedding 的显式相似度做 text-guided token selection，并保留背景多样性；“文本引导选 token”不是新点。
- [LLaVA-Mini](https://arxiv.org/abs/2501.03895) 已在 LLM 前把视觉信息预融合进文本 token，并将视觉流压到一个 token；“pre-fusion/readout token”不是新点。
- [LVPruning](https://aclanthology.org/2025.findings-naacl.242/) 和 [PuMer](https://aclanthology.org/2023.acl-long.721/) 已覆盖 language-guided pruning 及 text-informed pruning/merging。
- [SwapMix](https://openaccess.thecvf.com/content/CVPR2022/html/Gupta_SwapMix_Diagnosing_and_Regularizing_the_Over-Reliance_on_Visual_Context_in_CVPR_2022_paper.html) 已用对象/属性 feature swap 诊断并训练 VQA；“属性交换增强”不是新点。
- [V-SEAM](https://aclanthology.org/2025.emnlp-main.880/) 已用语义视觉编辑定位对象、属性、关系相关 attention head，并通过调制改善 VQA；“编辑后定位并调 attention”不是新点。
- [视觉—语言 Binding ID 研究](https://openaccess.thecvf.com/content/CVPR2025W/MIV/html/Saravanan_Investigating_Mechanisms_for_In-Context_Vision_Language_Binding_CVPRW_2025_paper.html) 已证明视觉对象 token 与文本指代可携带共享 binding ID；不能声称首次提出视觉 binding code。
- [MATE](https://aclanthology.org/2025.findings-acl.965/) 已专门评测跨模态 entity linking，并发现对象增多时 VLM 明显退化；“发现实体属性难绑定”也不是新点。
- [Attention Mask Consistency](https://openaccess.thecvf.com/content/CVPR2023/html/Yang_Improving_Visual_Grounding_by_Encouraging_Consistent_Gradient-Based_Explanations_CVPR_2023_paper.html) 已用区域标注约束解释/grounding 一致性；简单的 attention-to-box loss 不是新点。

这个方向并不冷门。相对窄、仍值得查验的空白是：**固定低 token 预算的小型 decoder-only VLM 中，能否把 query-to-object 地址与 object-to-attribute 值分开干预，并证明这种因果分解比 query-aware compression、普通 grounding supervision 和高学习率 CE 都更有效。**

## 候选方法：CAVR

令现有 bridge 输出 `Z ∈ R^(49×2048)`，文本 embedding 为 `T`。新增一个极小的 query pooler 和单头 reader：

```text
q = TextPool(T) Wq
a(q,Z) = softmax(q (Z Wk)^T / sqrt(d))       # 49 个地址概率
r(q,Z) = sum_i a_i(q,Z) (Z_i Wv)             # 被读取的属性值
image_end' = image_end + Wo r(q,Z)            # 复用已有分隔 token
```

这样不增加 49 个视觉 token、LLM 序列长度或 KV token 数。额外开销是一次很小的 query pooling 和 49-way attention；真实延迟必须实测，不能从公式直接宣称端侧加速。

对颜色交换 pair `(Z_b, Z_e)`，问题指向同一对象但值改变。训练目标按最小必要集合固定：

1. `L_answer`：所有 base / edited / invariant 的正常答案 CE；
2. `L_address`：只在有明确目标对象的问题上，让 `a` 对齐目标 7×7 cell；oracle 位置只用于训练；
3. `L_address-pair`：同一问题在 base/edited 间地址分布保持一致；
4. `L_value-swap`：使用 base 的地址与 edited 的 value，答案必须变为 edited；反向同理；
5. 训练已有 language LoRA，而不再冻结全部语言路径。

关键点不是“又一个 cross-attention”，而是**可单独反驳的地址/值干预**：地址应对属性交换不变，读出的值及答案应等变。若普通 query reader + CE 一样好，CAVR 的方法贡献失败；模型可采用更简单方案。

## 必须先过的强基线

E0 的 LoRA 学习率只有 `1e-5`，bridge 为 `2e-5`，且只有一遍数据；最终 batch 仍约 50% 并不能代表普通 CE 的能力上限。进入新模块前，必须运行一个更合理但仍很小的学习率/更新数 screen。

最小三臂顺序：

| 臂 | 作用 |
|---|---|
| tuned plain CE | 排除原 E0 只是欠训练 |
| query reader + CE | 排除收益只是 query-aware pre-fusion |
| CAVR full | 检验地址/值反事实目标是否有独立价值 |

同数据、同父模型、同 trainable LoRA、同候选答案、同评测器。既报告相同步数，也报告 GPU-time-matched plain CE；CAVR 有额外分支，不能只做 step-matched 比较。

### tuned plain CE 已完成：更强更新仍没有解决绑定

冻结摘要：[binding-tuned-plain-ce-evaluation-v1/summary.json](../evidence/binding-tuned-plain-ce-evaluation-v1/summary.json)，SHA256 `c7c2ed0e78604694314984dc35ef6d45067946d31273bc743780379460645792`。8 个 checkpoint 只在 40 个 mechanism-dev scene pair 上筛选，step 200 的 affected joint 最高（10.0%），随后冻结并跑完整 development 评测。

| split | affected joint | 真图−错配图，95% CI | invariant joint | 六题全对 |
|---|---:|---:|---:|---:|
| mechanism-dev | 10.3125% | -8.4375pp [-13.75, -3.125] | 50.78125% | 0% |
| selection-dev | 14.0625% | -1.5625pp [-6.25, +3.125] | 44.21875% | 0% |

mechanism-dev 上 tuned CE 相对 E0 的 affected joint 仅 `+0.625pp`，paired cluster-bootstrap 95% CI `[-3.75, +5.00]`，不显著。虽然真图相对 no-image 为正，但错配图仍不差于真图，说明答案训练增强了“有视觉前缀”响应，却没有学到正确图像—问题绑定。协议决策为 `insufficient_tuned_plain_ce`；final test 未使用。

还需要四个机制对照：oracle address 上界、query-blind reader、随机地址标签、交换无关对象的 no-op。它们分别回答“可达上限”“是否真的需要问题”“监督本身是否只是正则化”“干预是否具有选择性”。

## 决策阈值

开发阶段继续条件：

- mechanism-dev 受影响 joint ≥70%；
- 真图−错配图的 95% CI 下界 >0；
- 六题全对 scene-pair 明显离开 0；
- CAVR 相对 tuned plain CE 至少 +5pp，配对 95% CI 下界 >0；
- invariant controls 不因追求 flip 而明显下降。

只有开发阶段固定方法、超参数和 checkpoint 后，才生成并一次性打开 final test。更强的研究结论还需要多适配种子，以及至少一项外部真实图 entity/attribute binding 评测；MATE 是优先候选，但必须先核对数据许可、可获取性、prompt 适配和训练重合。

“惊人效果”不能定义为某个合成题从 6% 涨到 30%。可信目标是：在合成机制集进入可靠区间、显著超过强普通训练、对错配图真正敏感、迁移到真实图，同时维持固定 49-token 成本。任何一项不成立，都要缩小结论。

## 追加查重：为什么“routing failure”和“interchange training”也不能单独申新

- [CARD](https://arxiv.org/html/2608.20763v1) 在四类开放 VLM 上区分“表示可线性解码”与“表示没有驱动下游预测”，并将其明确定义为 cross-axis routing failure。我们的“视觉属性存在但答案不读取”诊断与其概念高度相邻；不能声称首次发现 VLM routing failure。
- [Interchange Intervention Training](https://arxiv.org/abs/2112.00826) 已提出把高层因果变量对齐到神经表示，并用 source-input 表示交换训练 counterfactual 行为。CAVR 的 value-source exchange 属于这一方法族；不能把“交换 latent value 并监督反事实答案”本身称为首创。
- [Visual Instruction Tuning Aligns Modalities through Abstraction](https://arxiv.org/abs/2606.03871) 用 probe 与因果干预指出视觉指令微调主要在语言模型中间语义层形成跨模态对齐；如果入口 reader 失败，下一步应比较中层写入，而不能预设输入层就是最佳位置。
- [Layer-wise Vision Injection with Disentangled Attention](https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_Layer-wise_Vision_Injection_with_Disentangled_Attention_for_Efficient_LVLMs_ICCV_2025_paper.html) 与 [DeepInsert](https://aclanthology.org/2026.eacl-long.332/) 已覆盖 layer-wise / middle-layer multimodal injection；中层写入架构也不是新点。
- [Cross-Modal Projection in Multimodal LLMs Doesn’t Really Project Visual Attributes to Textual Space](https://aclanthology.org/2024.acl-short.60/) 说明微调获得领域视觉能力时，projection 未必显式编码对应属性。我们的 probe 只能证明局部线性可解码，不能直接推出模型已形成可用、因果的 object-value code。
- [Investigating Mechanisms for In-Context Vision Language Binding](https://arxiv.org/abs/2505.22200) 已在合成 3D 视觉—文本任务中用因果干预研究 VLM 的 Binding ID；“视觉 object-attribute binding 的机制研究”本身不是空白。
- [Multi-Attribute Interactions Matter for 3D Visual Grounding](https://openaccess.thecvf.com/content/CVPR2024/papers/Xu_Multi-Attribute_Interactions_Matter_for_3D_Visual_Grounding_CVPR_2024_paper.pdf) 已用 counterfactual attention 和 exchanging-based fusion 改进属性级 3D grounding；因此“反事实 attention loss”或“交换属性 token”也不能单独作为新颖性主张。它与 CAVR 的差别需要严格限定为固定 KV 预算下、问题条件化 address/value source 的成对 latent interchange 及其选择性验证。
- [Quantifying and Mitigating Unimodal Biases in MLLMs](https://arxiv.org/abs/2403.18346) 已用 image/question controlled interventions 区分敏感性与鲁棒性；我们的 true/random/no-image 门控属于同一证据传统，而不是新的评测思想。
- [Adaptive Information Flow](https://arxiv.org/abs/2604.15809) 也观察到 VLM 能定位相关区域却给错答案，并通过抑制无关视觉 token 的信息流改进结果。这进一步否定“知道看哪里但答不出来”作为独立创新点，同时把 irrelevant-token masking 确立为后续真实图评测必须比较的 inference-time baseline。

因此更严谨的候选贡献被进一步收窄为：**在不增加视觉 token/KV 长度的 1.1B decoder-only VLM 中，对同一问题显式分离 location-address 与 attribute-value source，并通过 paired value-source interchange 检验和训练选择性读出；同时用 query-reader CE、tuned plain CE、随机地址、无关对象交换和外部真实图迁移排除简单解释。** 这仍只能称为“未在本轮检索中发现完全同构实现”的候选，不能称为世界首次。

## V2 结果：CAVR 学到了可干预的地址—值机制

第一版未归一化 dot-product reader 在 Query-CE step 46 出现 attention collapse 与 non-finite router gradient；失败证据保存在 `evidence/cavr-v1-stability-failure/`，没有覆盖。V2 使用 FP32 L2-normalized query/key、有界可学习 logit scale，并将 reader 学习率降为 `1e-4`。Query-CE 与 CAVR 的 64-step stability smoke 均通过；正式训练均由同一个 Stage-D parent 开始，parent 哈希前后不变。

400-step 开发比较按 40 个 mechanism-dev scene pair 选 checkpoint。Query-CE 与 CAVR 都选中 step 300。完整结果如下：

| 冻结臂 / split | affected joint | 真图−错配图，95% CI | invariant joint | strict 六题全对 |
|---|---:|---:|---:|---:|
| Query-CE / mechanism | 4.6875% | -1.5625pp [-5.00, 1.875] | 49.21875% | 0% |
| Query-CE / selection | 5.9375% | -5.625pp [-10.3125, -0.9375] | 70.00% | 0% |
| CAVR / mechanism | 100.00% | +85.625pp [81.25, 89.375] | 71.40625% | 21.875% |
| CAVR / selection | 90.625% | +73.75pp [67.50, 79.6875] | 70.00% | 11.875% |

mechanism-dev 上 CAVR 相对 Query-CE 的 affected joint 差为 `+95.3125pp`，95% CI `[92.8125, 97.50]`；相对 tuned plain CE 为 `+89.6875pp`，`[86.25, 92.8125]`。所有置信区间均按 scene pair 做 10,000 次 paired cluster bootstrap。final test 未生成、未使用。

这说明“普通 query-aware readout”不足以解释结果，但还不能证明外部迁移、种子稳健性或全局新颖性。

## GPU-time-matched 强控制：多算一倍 CE 仍未学会绑定

CAVR 每 batch 有两次 language-model forward，400 steps 用时 546.87 秒；原 Query-CE 只有一次 forward，400 steps 用时 281.39 秒。为排除算力混淆，额外冻结了一个两遍重复相同 800 个训练 scene pair 的 Query-CE：800 个 LM forwards、800 steps、562.64 秒，和 CAVR 的 wall-time ratio 为 `1.0288`，位于预注册 `[0.9, 1.1]`。

该控制在完整 mechanism-dev 上 affected joint 只有 7.50%，strict 仍为 0%，真图−错配图反而为 `-7.1875pp`，95% CI `[-11.5625, -2.8125]`；selection-dev affected joint 为 5.9375%，strict 0%，真图−错配图 `-7.50pp`，`[-12.1875, -2.8125]`。更多普通 CE 将 invariant joint 提高到 86.875% / 85.3125%，却没有让答案随正确图像中的属性改变。

CAVR 相对同 GPU 时间 Query-CE 的 affected joint 差为 `+92.50pp`，95% CI `[89.375, 95.3125]`。因此，在这个单种子合成开发实验里，CAVR 的优势不能由多一倍 forward 或 wall time 解释。

## 潜变量机制评测：不是只把答案背对了

独立机制评测直接读取地址 attention，并把同一问题的 address source 与另一图像的 value source 交叉组合。CAVR step 300 在 mechanism-dev 上达到：

- affected address all variants 100%；目标 cell 概率均值 99.06%；
- 正常 base/edit joint 100%，交换 value source 后 base/edit joint 100%；
- invariant 在 normal / swapped 下均正确且稳定 100%；
- 六项同时成立的 causal six-way 100%，cluster-bootstrap 95% CI `[100,100]`。

在未参与 checkpoint 选择的 selection-dev 上，CAVR 的 address all variants 为 88.125%，normal base/edit 90.3125%，swapped base/edit 92.8125%，causal six-way 88.75%，95% CI `[84.0625, 92.8125]`。

同 GPU 时间 Query-CE 的 causal six-way 在两个 split 都是 0。CAVR−control 的 causal six-way 差在 mechanism-dev 为 `+100pp [100,100]`，selection-dev 为 `+88.75pp [84.0625,92.8125]`。这支持“显式地址监督与 value-source interchange 形成了可干预机制”，但仍局限于合成开发域。

## 外部真实图评测：未证明迁移

[NaturalBench](https://arxiv.org/abs/2410.14669) 用两张自然图与两道互补问题构成一个组，专门压制只靠文本或答案先验的解法。官方代码定义四个指标：单题 `Acc`、同问题跨两图全对的 `Q_Acc`、同图两问全对的 `I_Acc`，以及四个组合全对的 `G_Acc`。本项目固定 Hugging Face revision `c8180c14e7408287a385bd7851ef8816965f638d`，共 1,900 组 / 7,600 图问组合；1,450 组 Yes/No、450 组 A/B，来源为 DOCCI 与 Flickr。官方计分代码固定到 commit `53e684f83822341e71b4515d711156b619fe065b`。

正式协议一次性比较四个已冻结模型：Stage-D parent、step-matched Query-CE、GPU-time-matched Query-CE、CAVR；提示直接使用数据中已附官方后缀的问题，以候选答案 token log-probability 打分，排除 EOS。主指标为最严格的原始 `G_Acc`，主比较为 CAVR−GPU-time-matched Query-CE；不得用该 benchmark 调 prompt、选 checkpoint 或改架构。只有差值至少 +2pp 且 paired 95% CI 下界大于 0，才能称“在 NaturalBench 上观察到外部迁移”。

V1（`naturalbench_external_evaluation_protocol_v1.json`）在加载任何模型、占用 GPU 或产生预测之前被 16GB 主机的 OOM killer 终止：评测器把全部解码图像物化为 Python list，RSS 达约 15.1GB。失败日志以 SHA256 `9ba819ba77cd762367bbd8a0f520554570c5b38e84f3d9a6924adb492062b55f` 保留。V2（`naturalbench_external_evaluation_protocol_v2.json`）只把数据访问改为 Arrow memory-map + `Image(decode=False)` + 每批惰性解码；模型、数据、prompt、候选评分、顺序、统计与决策阈值均未改变。真实数据内存回归约 3.2GiB；正式 v2 峰值 RSS 6.52GiB、总评测 wall time 672.17 秒，正常退出。

NaturalBench 论文同时指出，答案先验可能让绝对阈值下的结果远低于模型对八个 image-question-answer triples 的相对排序能力。因此协议还预注册了 Section-5-style 的 sample-balanced likelihood ranking：每组强制比较两个应选第一候选与两个应选第二候选的 margin，另报 `Balanced_Q_Acc`、`Balanced_I_Acc`、`Balanced_G_Acc`。它只作为偏置诊断，不能替代原始 `G_Acc` 主指标，也不能用于回头调校阈值。

V2 完整结果保存在 `evidence/naturalbench-external-zero-shot-v2/result.json`，SHA256 `4f3d65011b279fe37c40674a17b8adbdf786e3d12e18133c0c457413a8375ad0`；运行日志 SHA256 `466c27e87bb3ba421e534158569f5c64b0262105071c1959974d65c484688499`。

| 冻结臂 | raw Acc | raw Q-Acc | raw I-Acc | raw G-Acc | Balanced G-Acc | score ties |
|---|---:|---:|---:|---:|---:|---:|
| Stage-D parent | 48.7500% | 6.7632% | 6.6316% | 0% | 2.0526% | 140 |
| Query-CE | 46.3026% | 2.0526% | 5.8684% | 0% | 0.5263% | 531 |
| GPU-time-matched Query-CE | 49.8026% | 0.6053% | 2.8684% | 0% | 1.5263% | 34 |
| CAVR | 49.5526% | 4.9474% | 7.3947% | 0.0526%（1/1900） | 2.4211% | 84 |

预注册主比较 CAVR−GPU-time-matched Query-CE 的 raw `G_Acc` 为 `+0.0526pp`，paired cluster-bootstrap 95% CI `[0, +0.1579]`：既没有达到 +2pp，也没有让下界大于 0；candidate-score zero-tie gate 同样失败。因此正式判定是 `external_transfer_not_demonstrated`。CAVR 的 raw `Acc` 反而为 `-0.25pp [-0.7105,+0.2237]`；Balanced `G_Acc` 为 `+0.8947pp [+0.0526,+1.7382]`，只能说明组内相对排序有一个很小的诊断性改善，不能覆盖 raw 主指标失败。四个模型都远低于四个二元组合随机独立全对的 6.25% G-Acc 期望，说明当前系统尚未形成可用的自然图像问答能力，而不只是 CAVR 外部增益不够大。

所以当前证据支持两句同时成立：CAVR 在合成域形成了普通同算力 CE 没有形成的可干预地址—值机制；这个机制没有零样本迁移成 NaturalBench 上的有效自然图像能力。后续不得在 NaturalBench 上调 prompt、阈值或 checkpoint。

## 追加效率与新颖性查重

- [LEO-Mini](https://aclanthology.org/2025.emnlp-main.368/) 已把文本/视觉/learned-query 相似度驱动的 conditional token reduction 与多模态 LoRA experts 结合；所以“query-conditioned visual compression + expert routing”也不是可申新的核心。
- [AdaV](https://aclanthology.org/2025.findings-acl.258/) 已将 text-guided visual attention redirection 放到 pre-LLM 阶段，并以 training-free pruning 改善有限视觉 token 预算下的鲁棒性；未来必须把这类 inference-time redirection 当作基线。
- [ViPE](https://aclanthology.org/2025.emnlp-main.897/) 已把压缩后的视觉内容通过 LoRA-like 参数路径注入 LLM；如果 CAVR 改成中层/参数写入，不能把“视觉写入参数空间”本身当创新。
- [ARPGrounding](https://openaccess.thecvf.com/content/CVPR2024/html/Zeng_Investigating_Compositional_Challenges_in_Vision-Language_Models_for_Visual_Grounding_CVPR_2024_paper.html) 已系统评测 attribute、relation、priority 的组合 grounding，并提供 composition-aware fine-tuning；真实图上的属性定位必须与它区分。
- [SWIM](https://openaccess.thecvf.com/content/CVPR2026/papers/Sun_See_What_I_Mean_Aligning_Vision_and_Language_Representations_for_CVPR_2026_paper.pdf) 已把文本中的对象名精确链接到 mask，并监督多个 LLM 层的 text-to-vision attention；所以 CAVR 的地址监督本身不新，差异只能来自固定瓶颈下可交换的 value source 与因果验收。
- [ViLLA](https://openaccess.thecvf.com/content/ICCV2023/html/Varma_ViLLA_Fine-Grained_Vision-Language_Representation_Learning_from_Real-World_Data_ICCV_2023_paper.html) 已从真实图文数据构造 region-attribute pairs 做细粒度表示学习；若进入真实图 post-training，必须与 region-attribute supervision 比较，而不能把对象—属性配对数据当成新贡献。
- [MACCO](https://aclanthology.org/2026.acl-long.1490/) 已通过跨模态 masked compositional concept reconstruction 改善 object-attribute binding、关系与词序；“遮住属性再跨模态重建”不应成为 CAVR 的新颖性主张。

由此，当前仍可研究的窄贡献不是一个新 projector 名称，而是：**固定 49-token / 固定 KV 成本下，显式可交换的 location-address / attribute-value 因果接口，是否能以远少于大规模 multimodal post-training 的数据，在小型 decoder-only VLM 上产生可复现、可迁移且可解释的绑定增益。** 该命题只有在 NaturalBench、第二项外部 benchmark、多种子和未见 final test 上连续成立后才有论文级含金量。

## NaturalBench 之后的冻结决策树

### 若外部主指标通过

1. 不再改当前方法或 prompt，先补到至少 5 个 adapter / router 初始化种子；报告跨种子均值、标准差与 paired seed effect，而不是继续放大单种子 bootstrap。
2. 在所有种子完成前冻结新的合成 final-test 生成协议；只开一次，避免用已反复查看的 development 代替确认性证据。
3. 再选一项不同构造的真实图 benchmark。NaturalBench 主要检验 paired discrimination；第二项应带对象区域或属性链接标注，以区分“答对”与“地址真的对”。
4. 只在以上结果稳定后做端侧 profiling：同 batch、同回答长度、分别预热，报告 vision encode、prefill、decode、峰值显存与端到端时间；49 tokens 只能直接说明 KV/token 成本固定，不能代替真实延迟。

### 若外部主指标不通过

1. 把结论收缩为“合成域机制成立”，不在 NaturalBench 上改 prompt、选 checkpoint 或反复调权重。
2. 下一阶段改用**真实图训练开发集**，而不是把 NaturalBench 变成训练集。优先数据结构是带 bbox/region 与 object attributes 的 GQA / Visual Genome 类 scene graph；它们能给 address 监督，也能构造同对象类别、不同属性值的跨图 value-source pair。
3. 新方法需加入三类真实图强基线：相同数据的 Query-CE、bbox attention supervision、region-attribute hard-negative training。CAVR 只有在同 GPU 时间下同时改善正常答案、错配图敏感性和区域定位，才保留。
4. NaturalBench 从此只作为已见 regression；新的 MATE、region-grounded VQA 或人工封存自然图 pair 承担确认性评测。

这两条路径都不会直接扩大到通用 multimodal pretraining。先证明一项明确能力能在小模型上以很少数据被因果地修复，比堆更多模态、更多 loss 和更多公开 benchmark 更有研究价值。

## NaturalBench 之后已执行：真实图 posture hard-negative pilot

NaturalBench 主指标失败后，没有在该 benchmark 上改提示或选 checkpoint，而是从 Open Images 的 boxable attributes 与 Localized Narratives 构建了新的真实图开发材料。数据源本身是公开的大规模图像标注资源；本项目另外保存了每张图的原始来源、作者、许可链接、文件 SHA256、目标框与人工审核决定。[Open Images 官方下载页](https://storage.googleapis.com/openimages/web/download_v6.html)；[Localized Narratives](https://google.github.io/localized-narratives/)。

第一轮 v2 候选对 192 个目标逐张人工审核，只接受 125 个（65.10%）。Boy/standing、Man/standing、Woman/standing 分别只有 10、10、11 个，低于冻结的每 stratum 16 个门槛，因此整个 v2 pool 按协议拒绝，没有为了凑数量放宽标准。

随后用 [ViTPose](https://arxiv.org/abs/2204.12484) 对全部 192 个目标做 replay，而不是先看分数再挑阈值。目标三个 standing strata 的 `mean_lower_body_score` 区分人工 accept/reject 的 pooled AUROC 为 `0.8883`，stratified bootstrap 95% CI `[0.8072, 0.9552]`；因此它只获准作为新候选池的软排序器，不作为自动 hard gate。排序后另取与 v1/v2 image ID 完全不重叠的 v3 standing supplement 144 张并全部人工审核：Boy 29/48、Man 24/48、Woman 26/48 通过，共 79/144，三个 stratum 均超过冻结的 20 个门槛。

最终只合并人工 accept 的 v2/v3 目标，并做跨 pilot exact SHA 与 dHash≤4 去重。结果为 77 个类别内 sitting/standing hard-negative pairs、154 张唯一图片：Boy 18 对、Girl 16 对、Man 23 对、Woman 20 对；拒绝样本使用数为 0。冻结产物为 [pair manifest](../evidence/openimages-posture-pairs-final-v1.jsonl)（SHA256 `959514828d1be086c46c48176f77c6a68b80947fb0b1d89f6f08448d05394b10`）及 [finalization receipt](../evidence/openimages-posture-final-pairs-v1-build.json)（SHA256 `d2580724291c3f54c1bab8b334cb492331e1a8b663923e142dcd9f35a7207881`）。

这些 pair 是“同人物类别、相反姿态、不同真实场景”的 hard negatives，**不是**仅改变姿态、其余像素保持不变的因果 counterfactual，不能把跨图 value swap 直接解释为 address/value 因果交换。为防止同一 pair 或同一图泄漏，新增了 5-fold pair-atomic crossfit：fold 大小为 16/15/16/15/15，每个类别的 fold 计数差最多 1，77 个 pair ID、154 个 image ID 和 154 个 image SHA 均唯一。冻结 [crossfit manifest](../evidence/openimages-posture-crossfit-v1.jsonl) 的 SHA256 为 `ad96b6716b4032ecaeb41450433f61beeb6ad94b0268974afedbddfad5d6bf16`；当前只完成数据划分，没有授权或执行参数更新、checkpoint 选择或模型成绩声明。

## 2026 追加查重：下一步不能靠换名字制造“创新”

- [AdaptVision](https://arxiv.org/abs/2512.03794) 已让 VLM 从低分辨率开始并按需调用 bbox crop，以 RL 同时优化答案与视觉 token 成本；“模型自己决定何时放大”不是空白。
- [Token-Efficient VLM](https://research.nvidia.com/labs/lpr/publication/tevlm2025/) 已用 dynamic region proposal 做高分辨率、少 token 视觉获取；“保留全局、补关键局部”也不是独立新点。
- 2026 年的 [CORAL counterfactual grounding](https://arxiv.org/abs/2607.03647) 已用 blank、pixel-shuffled、image-absent 与检索 hard negatives 检验视觉依赖，并训练图像替换敏感性；“错配图下降”应作为必要控制，不再可单独申新。
- [SigLIP 2](https://arxiv.org/abs/2502.14786) 已提供同为 ViT-B/16 的 NaFlex 版本，支持多分辨率和原始长宽比，并专门改善 localization/dense features。它是当前 224 固定输入最直接的视觉编码器对照，但换 encoder 后必须重新对齐 bridge。
- [LightOnOCR-1B](https://huggingface.co/blog/lightonai/lightonocr) 的公开消融中，两阶段“先冻 backbone 训 projector、再全量训练”平均分反而低于一阶段端到端训练 1.4 分；因此常见两阶段 recipe 也必须实测，不能当定律。其 2026 后续 [LightOnOCR-2-1B](https://huggingface.co/blog/lightonai/lightonocr-2) 把成功归因于 16M 级高质量文档页、标注一致性、分辨率和专门鲁棒性训练，进一步说明 77 对 pilot 只能筛机制，不能直接造出通用强模型。
- [PixMo](https://docs.allenai.org/training_data/pixmo) 把 dense human captions、QA 和 pointing 分开建设；[MolmoPoint](https://allenai.org/blog/molmopoint) 又以每图高密度 point 标注提高样本打包效率。对本项目，优先扩大“一个图像对应多个可验证 region/attribute/relation 问题”，比堆低密度长 caption 更匹配单卡预算。
- [SAIL-VL](https://arxiv.org/abs/2501.05952) 在 2B 模型上仍观察到随高质量多模态预训练数据扩大而持续提升，并报告课程难度比 all-in-one 更好；这否定了“用几十或几百张图即可把 1B 做到通用极致”的不现实预期，也支持先做质量、覆盖和课程，再做规模。

所以最可信的研究主线进一步收敛为：**在固定 49-token 推理模式下，比较普通答案监督、真实 bbox/point 地址监督、同类 hard-negative 图像敏感性和合成 latent interchange；再引入 NaFlex/原图裁块作为可购买的额外证据，报告同一质量下的视觉 token、prefill 与端到端成本。** 方法价值必须来自真实图 out-of-fold 增益、定位正确、错配图敏感、成本受控四项同时成立，而不是任一单项。

## 冻结真实图零样本诊断：父模型有弱感知，合成 CAVR 没有迁移

在任何真实图参数更新之前，固定同一组 77 对/154 张图，一次性评测四个既有 checkpoint：Stage-D parent、step-matched Query-CE、GPU-time-matched Query-CE、CAVR。每个 checkpoint 均跑原图、红框目标图、目标裁块，以及同类反姿态 paired-swap 和 uniform blank 控制；候选答案固定为 `sitting/standing`，按答案 token log-probability 计分并排除 EOS，没有选 checkpoint、调 prompt 或阈值。完整结果为 [openimages-posture-zero-shot-v1.json](../evidence/openimages-posture-zero-shot-v1.json)，SHA256 `e135fbf6ff02747d3cc4a1f5a9cfb27e6ab52c777900394d511903e44f473a81`；154/154 张图片重新验 SHA 全通过。

| 冻结臂 / true view | 单图准确率，95% CI | pair joint | true−paired-swap，95% CI | strict bidirectional flip |
|---|---:|---:|---:|---:|
| Stage-D parent / full | 53.25% [48.70,57.79] | 11.69% | +8.44pp [0.00,17.53] | 11.69% |
| Stage-D parent / marked | 57.14% [51.30,62.99] | 22.08% | +14.94pp [3.25,26.62] | 22.08% |
| Stage-D parent / crop | 60.39% [55.19,65.58] | 24.68% | +20.78pp [10.39,31.17] | 24.68% |
| GPU-time Query-CE / crop | 50.00% [50.00,50.00] | 0% | 0pp [0,0] | 0% |
| CAVR / marked | 49.35% [46.75,51.30] | 1.30% | -1.30pp [-6.49,2.60] | 1.30% |
| CAVR / crop | 48.70% [44.81,51.95] | 3.90% | -2.60pp [-10.39,3.90] | 3.90% |

Stage-D parent 的 crop 与 marked 条件都对正确图显著敏感，证明现有 SigLIP2→bridge→language 路径尚有真实姿态信号，不是完全随机。红框比 full 提高 3.90pp、crop 比 full 提高 7.14pp，但三个 view 使用不同冻结 prompt，因此这些跨 view 差值只能用于失败定位，不能作为严格方法比较。

CAVR 相对 Stage-D parent 在 crop 上为 `-11.69pp`，95% CI `[-18.18,-5.19]`；marked 为 `-7.79pp [-14.29,-1.30]`。它与 GPU-time Query-CE 在三个 view 上都无显著差异。CAVR 的红框 bbox-region attention top-1 从 full 的 50.65% 增至 59.09%，但 center-cell top-1 仅 3.25%，答案仍不随正确图稳定变化。这说明合成训练出来的路由分布会被标记区域影响，却没有迁移为真实姿态读取；attention overlap 不能替代答案因果证据。

因此后续真实图开发必须从 Stage-D parent 初始化，而不是从 CAVR checkpoint 初始化。最便宜的下一个诊断是对冻结 SigLIP2 的 full/global、bbox-ROI 与 target-crop 表示做 pair-atomic out-of-fold 线性 probe：若 crop/ROI 可分而 VLM 不行，优先修 bridge/语言读取；若视觉表示本身也不可分，再投入 NaFlex、高分辨率或视觉层适配。77 对仍只够做机制筛选，不够训练或宣称通用能力。

## 当前执行边界

截至冻结零样本诊断：V1 稳定性失败、V2 正式训练、标准输出评测、潜变量机制评测、GPU-time-matched CE 控制、NaturalBench V1 OOM 和 V2 正式结果均保留了哈希与原始记录。新增真实图流水线保留了候选、ViTPose replay、完整人工决定、最终 pair、无泄漏 crossfit receipts 与四臂冻结零样本评测；crossfit splitter 的 3 项、本轮零样本 evaluator 的 4 项本地及远端测试均通过。合成 final test 仍缺席。该时间点最强可信结论是“单种子合成 development 中，CAVR 形成了同等 GPU 时间普通 CE 没有形成的可干预地址—值机制，但它没有迁移到 NaturalBench 或新的姿态 hard negatives；相反，pre-CAVR Stage-D parent 在目标裁块上保留了显著但较弱的真实姿态信号”。后续真实图参数更新见下节；这些结果仍不能称 1.1B 模型已成为通用强 VLM。

## 冻结视觉探针：姿态信息在 SigLIP2 中，主要瓶颈在读出

在更新任何模型参数前，对同一 77 对/154 张图的冻结 SigLIP2-B/16 表示做固定超参数、pair-atomic 5-fold 线性探针。每个 pair 只由其余四折拟合的 logistic probe 评分一次；没有调 C、选 checkpoint 或使用正式 test。完整结果为 [openimages-posture-siglip2-probe-v1.json](../evidence/openimages-posture-siglip2-probe-v1.json)，SHA256 `6c31a97f7788092d7f79f2084a51bd045c2431e811ce60f86bea9819f8fa4b85`。

| 冻结表示 | 样本外单图准确率，95% CI | pair joint，95% CI | pair 内 standing score > sitting |
|---|---:|---:|---:|
| full global | 94.81% [90.91,98.05] | 89.61% [81.82,96.10] | 97.40% [93.51,100] |
| full bbox ROI | 94.81% [91.56,98.05] | 89.61% [83.12,96.10] | 100% [100,100] |
| marked global | 94.16% [90.26,97.40] | 88.31% [80.52,94.81] | 98.70% [96.10,100] |
| crop global | 97.40% [94.81,99.35] | 94.81% [89.61,98.70] | 97.40% [93.51,100] |

这与完整 Stage-D VLM 的 53.25% / 57.14% / 60.39% 形成明显差距：姿态在冻结视觉特征中高度线性可分，但没有稳定驱动语言答案。它支持把下一步计算投入 bridge/readout，而不是立即更换视觉编码器。这个结论只定位当前小型开发集上的瓶颈；不同场景 pair 可能仍含背景相关性，不能把 97.40% 当作模型 benchmark 成绩。近期研究也观察到视觉表征存在但跨模态链接失败的系统性缺陷，并用 probe 检测失败；因此“可线性解码但答案不会用”不是独立新颖性主张（[Ashok et al., EMNLP Findings 2025](https://aclanthology.org/2025.findings-emnlp.850/)）。

## 真实图 readout 5-fold 开发：简单 CE 胜出，显式地址监督只改善注意力

基于上述诊断，冻结同一个 Stage-D parent、1.1B language model、已有 LoRA、SigLIP2、49-token compressor/projector 和 image boundary tokens。每折只新建并训练一个 rank-64 query readout，共 `659,521` 个参数；训练折与测试折按 pair 完全隔离。两个同容量 arm 只差 loss：`query_ce` 使用答案 CE；`grounded_query` 另加 audited bbox region attention 与 pair-ranking。主训练输入为带红框的无歧义 target；full 与 crop 只作迁移。完整 5 折运行在单张 NVIDIA L20 上耗时 `227.46s`，1,920 optimizer steps，峰值 allocated `4.06GiB`，没有 checkpoint selection。

完整结果和全部 10 折权重/日志在 `runs/openimages-posture-readout-crossfit-v1/`；总结果 SHA256 `0a8162b319c78033aeddd860424a6e641eb181ccdc856b95cf15dc646b39b6a4`。独立审计收据为 [openimages-posture-readout-crossfit-v1-analysis.json](../evidence/openimages-posture-readout-crossfit-v1-analysis.json)，SHA256 `9a1f206470021edb752157f564558ef934a92298f82f8cfdcf317c255879d615`，确认每个 pair 每个 arm 恰好样本外评分一次、无 pair 泄漏。

| arm / true view | 单图准确率，95% CI | pair joint，95% CI | true−paired-swap，95% CI |
|---|---:|---:|---:|
| query-CE / marked | 94.81% [91.56,98.05] | 89.61% [83.12,96.10] | +89.61pp [83.12,96.10] |
| query-CE / full | 88.31% [83.77,92.86] | 76.62% [67.53,85.71] | +76.62pp [67.53,85.71] |
| query-CE / crop | 87.01% [81.82,91.56] | 74.03% [63.64,83.12] | +74.03pp [63.64,83.12] |
| grounded / marked | 90.26% [85.71,94.81] | 80.52% [71.43,89.61] | +80.52pp [71.43,89.61] |
| grounded / full | 81.82% [75.97,87.01] | 64.94% [54.55,75.32] | +63.64pp [51.95,74.03] |
| grounded / crop | 83.77% [77.27,90.26] | 72.73% [62.34,81.82] | +67.53pp [54.55,80.52] |

query-CE 相对未适配 Stage-D parent 的 pair-joint 增益为：marked `+67.53pp [55.84,77.92]`，full `+64.94pp [51.95,76.62]`，crop `+49.35pp [35.06,62.34]`。这证明当前冻结的视觉/语言主干只需很小的真实图读出适配即可把 latent posture signal 转成答案；它仍是同一 method-development 集上的 crossfit，不是新数据集确认。

显式区域监督把 marked bbox-region top-1 相对 query-CE 提高 `+46.75pp [37.66,55.84]`、region mass 提高 `+48.33pp [42.54,54.11]`；但 marked 答案 pair-joint 反而为 `-9.09pp [-19.48,+1.30]`，full 为 `-11.69pp [-23.38,0]`。因此当前数据上，干净的 attention 对齐不能替代答案质量，`grounded_query` 不进入下一轮；冻结选择 `query_ce` 进入全新、未见确认集。这个负结果也与 instruction-grounded projector、空间 token、grounding-feedback 等近期工作相邻，不能因为注意力更可解释就声称方法优越（[Instruction-Grounded Visual Projectors](https://openaccess.thecvf.com/content/ICCV2025/html/Jin_Instruction-Grounded_Visual_Projectors_for_Continual_Learning_of_Generative_Vision-Language_Models_ICCV_2025_paper.html)、[SVP](https://arxiv.org/abs/2501.04568)）。

当前冻结决策是：不在这 77 对上继续调 epoch、LR 或 loss 权重；下一份证据必须来自新获取、图像不重叠、目标描述无歧义、prompt 与超参数预先冻结的 real-image confirmation。只有确认复现后，才把同一 readout recipe 扩展到 color/material/action/relation、pointing、OCR 和多图任务。
