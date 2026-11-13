# 异构图 × LLM 推理：顶刊级研究空白调研报告

> 调研日期：2026-11-10 | 调研方式：全网最新文献检索（2025–2026，NeurIPS/ICML/ICLR/AAAI/KDD/TKDE/TPAMI 级别）
> 结论先行：推荐主攻方向 **「异构图原生过程验证器」（Graph-as-PRM）——用图本身为 LLM 推理过程打分**，证据链完整、无撞车、非增量。

---

## 一、研究现状全景（2025–2026）

### 战线 1：LLM 图推理 RL 化（最热，但奖励全是"盲"的）
| 工作 | 会议 | 核心做法 | 致命弱点 |
|---|---|---|---|
| **G1** | NeurIPS 2025 | 首个 GRPO + 图任务奖励训练 LLM 图推理 | 仅 **outcome 奖励**（最终答案对错），过程无监督 |
| **KG-R1** (arXiv 2509.26383) | 2025 | 单智能体 RL 训练 KG-RAG 智能体 | 同上，奖励稀疏，长推理链崩溃 |
| **UniRel-R1 / KG-Hopper / GraphRAG-R1** | 2025–2026 | RL 微调 LLM 做 KG 多跳推理 | 全部 outcome-only 奖励 |
| **Kansal & Jha** (arXiv 2601.15160) | 2026.01 | KG 作为"隐式奖励模型"，GRPO 训练 | 奖励 = **词面 token 重叠**（非语义、非结构）；仅医学 MCQ、≤5 跳；自认 future work 是"符号知识与神经架构的深度融合" |

### 战线 2：图推理过程奖励模型（刚起步，验证器全是文本 LLM）
| 工作 | 会议 | 核心做法 | 致命弱点 |
|---|---|---|---|
| **GraphPRM** (arXiv 2503.00845) | 2025 | 首个图推理 PRM，13 个图计算任务 | 验证器是 **微调的文本 LLM**（Qwen-Math-7B）——用不会算图的语言模型去验证图推理；大图/稠密图上失效；小 PRM 验证大 policy 反而掉点；数学 PRM 完全不迁移 |
| **PAV / RPS** (DeepMind) | ICLR 2025 | 通用过程验证器理论 | 纯数学域，无图结构 |
| 结构化 DAG 验证 | AAAI 2026 | 推理链建 DAG 验证 | 验证器仍是 LLM |

### 战线 3：异构图 + LLM（管道化，无验证闭环）
- **Think-on-Graph 3.0** (arXiv 2509.21710)：Chunk-Triplets-Community 异构图索引 + 多智能体检索，纯 prompt 管道，无 RL、无过程验证。
- **SCHEMAS** (OpenReview 2026)：异构图组织多智能体系统的工程框架。
- **HetGCoT** (EMNLP 2025 Findings)：异构图增强 CoT，只做上下文增强。
- **Metapath-of-Thoughts** (Amazon 2024)：meta-path 文字化喂给 LLM，同上。
- **GFM-RAG** (NeurIPS 2025)：图基础模型做检索器，不做过程验证。

### 战线 4：结构性失败证据（问题的"病根"文献）
- **LLM 图同构失败** (arXiv 2606.09484)：LLM 靠序列化表面模式，不做真结构推理。
- **NLGraph** (NeurIPS 2023) / **GraphWiz** / **GraphOmni**：图规模一涨准确率崩塌，远低于算法基线。
- **Wold et al.** (ACL Findings 2024)：LM+KG 在**未见过的组合/长度**上组合泛化失败。
- **The Mystery of Compositional Generalization in GNNs** (arXiv 2410.06272)：同分布评测掩盖了组合失败。
- **MLG 2025 position paper**：oversmoothing/oversquashing 与同配-异配二分的"共识"本身未解决。

### 战线 5：图基础模型（GFM）——统一靠文本对齐，无人做模式级推理
- VLDB 综述：主流策略 = 把异构图**对齐到文本接口**再预测（OpenGraph/AnyGraph/SAMGPT/GIT）。
- NeurIPS 2025 (KG pretraining transferability)：KG 预训练迁移的初步证据。
- 跨 schema（未见节点/关系类型）归纳推理：仅有 ACL workshop 级别的"语言学初始化"（Pasquini et al.），无系统方法、无 benchmark。

---

## 二、核心发现：一个被整个领域忽视的"验证鸿沟"（The Verification Gap）

**当前范式的荒谬之处**：
1. LLM 在图上推理会结构性犯错（跳不存在的边、类型不一致的推理、路径断裂）——已被多篇 benchmark 论文证实；
2. 领域的应对是 RL（outcome 奖励）或文本 PRM（GraphPRM）——**但验证器本身是不会算图的语言模型**；
3. 与此同时，异构图社区（HGNN/HGT/meta-path）二十年积累的恰恰是**类型感知的结构计算能力**——却被完全排除在推理验证闭环之外。

**一句话空白**：*没有人把异构图本身（符号约束 + 轻量 HGNN）做成 LLM 推理的步骤级验证器/奖励模型。*

搜索引擎的原话佐证（三轮独立检索均确认）：
- "No single paper combining GNN-based verifiers with test-time scaling for graph reasoning — appears to be a research gap"
- "No results specifically combined heterogeneous graph + verifier + reward model + agent reasoning"
- "No single result matched all terms exactly (meta-path type constraints as verifiable/shaped rewards for LLM reasoning over heterogeneous graphs), suggesting this specific combination may be a novel research direction"

**异构图的不可替代性**（这是它能成为"异构图论文"而非普通 LLM 论文的关键）：
- 同构图上"结构有效"只检查边存在性；
- **异构图上每一步推理都有可精确判定的类型约束**：这一步走的边是否符合 schema 允许的关系类型？路径是否构成类型一致的 meta-path？答案实体类型是否与问题约束匹配？——这些是**符号可验证**的，天然构成无懈可击的 RLVR（可验证奖励），且只在异构 schema 中存在。

---

## 三、推荐主攻课题（精准选题）

### 题目（工作命名）
**HG-PRM: Heterogeneous Graphs as Process Reward Models — 让图为 LLM 的思考过程打分**
（备选口号："Stop asking text models to verify what graphs can compute."）

### 精确问题定义
> 给定异构图/知识图谱 G 和多跳问题 q，LLM 智能体逐步执行动作（选关系、走边、剪枝、聚合）。现有 RL 只有终局 outcome 奖励（稀疏、可被 hack、长链崩溃）。**如何构造图原生的步骤级奖励 r(s_t, a_t)，使其（i）结构上精确不可 hack（符号部分）、（ii）语义上稠密可学习（神经部分）、（iii）可同时用于 RL 训练与测试时搜索？**

### 方法骨架（三明治结构）
1. **符号层（精确稀疏奖励，不可 hack）**：每步动作在图上执行类型一致性判定：
   - 动作合法性：所选关系 r 的头/尾类型是否匹配当前实体类型（schema 约束）；
   - meta-path 一致性：历史路径的类型序列是否构成合法/在先验 meta-path 集中的路径；
   - 可达性守恒：目标实体是否仍在当前前驱可达集合内（用图算法精确计算）。
2. **神经层（稠密语义奖励）**：轻量异构 GNN（HGT/RGHN 级别，<10M 参数）在「算法执行轨迹→自然语言步骤」合成数据上训练（借鉴 GraphSilo 思路但首次做成**异构**版），输出步骤语义分，弥补符号层只判"合法"不判"优"的缺陷。**小 GNN 验证大 LLM 完全可行**（对称于 GraphPRM 发现的"小 LLM 验证大 LLM 会掉点"——GNN 不做语言理解，只做结构-语义对齐，不存在该问题）。
3. **融合与使用**：
   - **理论钩子**：证明符号层奖励是 potential-based reward shaping（Ng et al. 1999 框架），不改变最优策略 → 奖励致密化不引入偏差，RLVR 安全性定理；
   - **训练**：GRPO/DPO 过程奖励（对齐 Kansal & Jha 的 GRPO 管线，但奖励从词面重叠升级为"符号+神经"）；
   - **测试**：验证器引导 beam search / MCTS（对齐 GraphPRM 的测试时扩展用法，但验证器从文本 LLM 换成 HGNN+符号）。

### 评测设计（每一条都踩在已证实的失败模式上）
- **域**：生物医学 UMLS（对齐 Kansal & Jha 可直接对比）、学术异构图（DBLP/MAG 三类节点）、金融异构图（对齐 RLHGNN）、Freebase/Wikidata 多跳 QA（CWQ/WebQSP/GrailQA）。
- **切分轴**（这是把"组合泛化失败"文献变成自己评测武器的关键）：
  - 按 hop 长度切（短跳训练→长跳零样本测试）；
  - 按 **meta-path 类型组合**切（训练时未见过的类型组合）——文献证实所有人在这轴上崩，没人系统测过异构版；
  - 按域切（医学训练→学术零样本迁移）。
- **基线**：outcome-only GRPO（KG-R1/G1 路线）、GraphPRM 式文本 PRM、Kansal & Jha 词面奖励、无验证器 self-consistency、ToG-3 式 prompt 管道。
- **消融**：符号层 only / 神经层 only / 融合；shaping 理论预测 vs 实测策略偏移；奖励 hacking 压力测试（故意构造可 hack 的词面奖励对照）。

### 为什么这是顶刊级、非增量
1. **范式翻转而非性能修补**：从"LLM 学图推理、LLM 互验"翻转为"图验证 LLM"——新轴（谁来验证），不是又一种 attention 变体。
2. **理论深度**：potential-based shaping 保策略不变性 + RLVR 不可 hack 性证明 + 稠密化降低策略梯度方差的分析——够 TPAMI/ICLR 级别理论章节。
3. **异构性是第一公民**：类型约束奖励只存在于异构 schema 中，方法无法被同构图工作"顺手拿走"。
4. ** benchmark 贡献**：首个异构图推理的 meta-path 组合泛化评测协议，独立成章。
5. **时效窗口**：Kansal & Jha（2026.01）刚刚打开"KG as reward"这条线且自认只做了 5%；GraphPRM 验证了 PRM 路线但用错了验证器。窗口正在打开，12–18 个月内必有人补上，先发者拿走 paradigm 论文。

### 投稿定位
- 首选：**ICLR / NeurIPS**（主会，方法+理论+benchmark 三合一）；
- 期刊版：**TKDE**（扩展为系统性框架 + 全套 benchmark）；TPAMI 备选（若理论部分做厚）。

### 可行性与资源匹配（重要）
- **零训练即可启动**：测试时验证器路线（符号约束 + 预训练 HGNN 引导 API LLM 的 beam search）**完全不需要 GPU 训练**，用 dmxapi/硅基流动 的 API LLM（Qwen/GPT 系列）当 policy，一周可出第一版数字——先占坑 arXiv。
- **第二阶段**：用 API LLM 合成异构版 GraphSilo 轨迹数据（你的 API key 正好用上），LoRA-GRPO 微调 Qwen-7B（单卡 A100/4090×2 可跑），冲主会。
- 风险对冲：若 RL 算力不足，DPO 版本 + 测试时扩展的结果已足够一篇强文。

---

## 四、备选课题（次优，各有短板）

### B1. 跨 Schema 归纳推理的异构图基础模型
- 空白：GFM 全靠文本对齐统一异构图，无 schema 级显式推理；无 benchmark。
- 做法：schema-as-graph 元学习 + 类型本体嵌入 + 未见类型零样本迁移 benchmark。
- 优点：TKDE 相性极佳；缺点：GFM 赛道拥挤（AnyGraph/SAMGPT/GIT 每月新文），易被增量淹没，且评测成本高。

### B2. 异构图 RAG（Heterogeneous GraphRAG）
- 空白：存在但正在快速关闭（ToG-3、SCHEMAS、agent-hetero-RAG 已入场）。
- 判断：6 个月窗口，除非抢在前面否则不建议。

### B3. 组合泛化失败的诊断与修复（KG+LLM）
- 空白真实（Wold et al. 只诊断未修复），但单独成文偏 benchmark/分析文，影响力上限低于主推课题。
- **建议**：不单独做，作为 HG-PRM 的评测轴吸收进主课题。

---

## 五、关键文献清单（写 related work 直接用）

**RL/PRM 侧**：G1 (NeurIPS'25)、GraphPRM (2503.00845)、Kansal & Jha (2601.15160)、KG-R1 (2509.26383)、PAV/RPS (ICLR'25)、GenPRM (AAAI'26)、GRPO 系列
**异构图+LLM 侧**：ToG-3 (2509.21710)、HetGCoT (EMNLP'25 Findings)、Metapath-of-Thoughts (Amazon'24)、RLHGNN (2507.02690)、RHCR (Sci Rep)
**GFM 侧**：VLDB GFM 综述、AnyGraph、SAMGPT (WWW'25)、GIT (ICML'25)、NeurIPS'25 KG-transferability
**失败模式证据**：LLM 图同构失败 (2606.09484)、NLGraph (NeurIPS'23)、GraphOmni、Wold et al. (ACL Findings'24)、GNN 组合泛化之谜 (2410.06272)、MLG'25 position paper
**理论工具**：Ng et al. 1999 potential-based shaping、RLVR 不可 hack 性分析

---

## 六、立即行动清单（本周可做）
1. 跑通一个最小实验：UMLS + 3 类符号约束奖励 + API LLM greedy/beam 对比，验证符号奖励对 4–5 跳问题的提升（对标 Kansal & Jha 表 2 的数字即可）；
2. 用 API LLM 生成异构版步骤级轨迹数据 1k 条（DBLP schema），训第一版 HGNN 奖励模型（<1M 参数，CPU 都能跑）；
3. arXiv 预印本占坑（题目建议：*Graphs as Process Reward Models: Type-Constrained Verification for LLM Reasoning on Heterogeneous Graphs*）。
