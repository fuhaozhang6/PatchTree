# PatchTree：双证据约束的分层修复融合算法

> 历史设计稿说明：当前运行时代码已经移除 support self-check 与 leaf
> self-check，并将 PatchRecord 精简为数据集 prompt 直接生成的类型、修正机制、
> 条件、边界和 patch。节点采用 `shared_core + conditional_residuals`，并统一编译
> 为现有 edits。当前实现契约以 `docs/type_guided_v23_light.md` 为准。

> 文档性质：方法设计与实现规格。
>
> 本文档给出 PatchTree 的完整算法定义。方法保留真实的树状抽象结构，但不对所有树节点执行 rollout。其核心是使用两类含义不同的证据：训练支持样本上的局部修复证据用于约束抽象是否保留已有能力，验证集上的泛化证据用于判断候选 Skill 是否能够写回。最终算法在有限评测预算内，从根节点向下搜索同时满足两类证据的最高抽象树切面。

阅读导航：

- 第 1--3 节：研究问题、总体设计和数据协议；
- 第 4--9 节：从训练轨迹到 certified leaf 和修复证书；
- 第 10--13 节：PatchTree、Internal/Root 和树切面编译；
- 第 14--17 节：局部可代替性、val 泛化和预算约束切面搜索；
- 第 18--22 节：完整伪代码、决策表、复杂度、超参数和实验产物；
- 第 23--26 节：当前代码对齐差距、消融实验、论文定位和实现禁区。

---

## 1. 研究问题

给定冻结的执行模型 \(M\)、第 \(t\) 轮的自然语言 Skill \(S_t\)、训练集 \(\mathcal D_{\mathrm{train}}\)、验证集 \(\mathcal D_{\mathrm{val}}\) 和测试集 \(\mathcal D_{\mathrm{test}}\)，目标是在不更新模型参数的情况下，根据执行轨迹迭代优化 Skill：

\[
S_{t+1}=\operatorname{Update}(S_t,\mathcal D_{\mathrm{train}},\mathcal D_{\mathrm{val}}).
\]

这一过程存在两组核心矛盾。

### 1.1 决策证据与评测成本的矛盾

树中的叶节点、内部节点和根节点都可以被视为候选 Skill 修改。若对每个节点分别进行支持集评估和验证集评估，节点得分会更充分，但 rollout 数量会随树节点数迅速增长。对于交互式环境或长轨迹任务，节点评测往往比节点生成更昂贵。

因此，算法必须明确：

- 哪些判断必须依靠真实 rollout；
- 哪些判断可以通过结构约束完成；
- 哪些候选值得消耗验证集评测预算。

### 1.2 局部具体性与全局泛化性的矛盾

样本级或簇级 Patch 通常更具体，容易修复产生它的训练样本，但覆盖范围有限。父节点越抽象，越可能覆盖更多修复机制、产生更短的 Skill，同时也越可能丢失子节点的适用条件和关键操作。

因此，树节点存在两个不同的评价角度：

1. **局部可代替性**：父节点或树切面能否保留叶节点已经证明有效的修复能力；
2. **全局泛化性**：该树切面编译成 Skill 后，能否提升未参与 Patch 生成的验证样本。

PatchTree 不把这两种证据简单加权成一个节点分数，而是采用约束式决策：局部可代替性是抽象成立的必要条件，验证集提升是候选写回的必要条件。在二者均满足时，优先选择抽象程度最高、表示最紧凑的树切面。

---

## 2. 方法总览

PatchTree 的完整流程为：

```text
训练样本多次执行
    -> 失败或不稳定样本诊断
    -> 样本级 PatchRecord
    -> 按修复机制形成叶簇
    -> 叶 Patch 生成
    -> 抽样 leaf self-check
    -> 一次最小修改重试
    -> 局部修复证书
    -> 稀疏 PatchTree 构建
    -> Internal 节点结构检查
    -> 从 Root 开始的树切面搜索
    -> 切面局部可代替性检查
    -> 切面 val 泛化检查
    -> 最高抽象可行切面写回
```

三个层级承担不同职责：

| 层级 | 核心问题 | 是否执行 rollout | 使用的数据 |
|---|---|---:|---|
| 叶节点 | 当前 Patch 是否真的能修复产生它的局部问题 | 是，低成本抽样 | train support samples |
| Internal 节点 | 多个局部 Patch 是否存在结构上合理的共享抽象 | 否 | 子节点元数据与规则文本 |
| Root/树切面 | 当前抽象是否既能替代局部 Patch，又能在未见样本上泛化 | 是，预算受限 | repair certificates + val |

---

## 3. 符号与数据职责

### 3.1 主要符号

| 符号 | 含义 |
|---|---|
| \(M\) | 冻结的执行模型 |
| \(S_t\) | 第 \(t\) 轮当前 Skill |
| \(x_i\) | 一个训练、验证或测试样本 |
| \(\tau_{ik}\) | 样本 \(x_i\) 的第 \(k\) 条执行轨迹 |
| \(r(x,\tau)\) | 任务奖励或完成质量 |
| \(R_i\) | 样本级 PatchRecord |
| \(C_k\) | 第 \(k\) 个修复机制叶簇 |
| \(P_k\) | 叶簇 \(C_k\) 对应的叶 Patch |
| \(\Gamma_k\) | 叶节点的局部修复证书 |
| \(T\) | PatchTree |
| \(\mathcal C\) | PatchTree 的一个完整切面 |
| \(S(\mathcal C)\) | 将切面 \(\mathcal C\) 编译后得到的候选 Skill |
| \(B_{\mathrm{cut}}\) | 一轮最多允许评估的树切面数量 |

### 3.2 数据使用协议

训练、验证和测试数据承担严格分离的职责：

\[
\mathcal D_{\mathrm{train}}
\Rightarrow
\text{轨迹生成、Patch 生成、叶节点修复检查和可代替性检查},
\]

\[
\mathcal D_{\mathrm{val}}
\Rightarrow
\text{有限预算下的树切面泛化选择},
\]

\[
\mathcal D_{\mathrm{test}}
\Rightarrow
\text{方法与超参数冻结后的最终性能报告}.
\]

需要特别强调：

- val 样本不参与 PatchRecord 生成；
- val 失败轨迹不反馈给 LLM 用于修改 Root；
- test 不参与树深、阈值、Prompt、切面和训练轮数选择；
- 如果根据逐轮 test 结果修改方法，则该 test 已经成为事实上的验证集，必须重新保留一个未使用的最终测试集。

---

## 4. Step 1：训练样本多次执行

### 4.1 目的

多次执行用于区分稳定成功、稳定失败和随机不稳定。单次失败可能来自采样波动，不应自动被解释为 Skill 缺陷。

### 4.2 操作

对于当前 batch 中的训练样本 \(x_i\)，使用当前 Skill \(S_t\) 独立执行 \(K\) 次：

\[
\tau_{ik}\sim M(\cdot\mid x_i,S_t),
\qquad k=1,\ldots,K.
\]

记录每条轨迹的奖励：

\[
r_{ik}=r(x_i,\tau_{ik})\in[0,1].
\]

计算经验成功率：

\[
q_i^0=\frac{1}{K}\sum_{k=1}^{K}r_{ik}.
\]

### 4.3 样本状态划分

可使用以下三种状态：

```text
stable_success: q_i^0 >= tau_success
stable_failure: q_i^0 <= tau_failure
unstable:       tau_failure < q_i^0 < tau_success
```

- `stable_success` 不产生修复 Patch；
- `stable_failure` 和 `unstable` 进入轨迹诊断；
- 如果所有样本均稳定成功，则本轮直接返回 \(S_{t+1}=S_t\)。

### 4.4 可复用信息

本阶段产生的 \(q_i^0\)、轨迹、错误位置和奖励必须缓存。后续 leaf self-check 直接复用它们作为 baseline，不重新运行当前 Skill。

---

## 5. Step 2：样本级 PatchRecord 生成

### 5.1 目的

PatchRecord 将具体轨迹错误转化为可聚合的最小修复证据。它不是完整 Skill，也不是长篇反思，而是后续构造叶节点所需的结构化记录。

### 5.2 PatchRecord 定义

对于失败或不稳定样本，生成：

\[
R_i=(z_i,d_i,\ell_i,p_i,b_i,I_i,q_i^0),
\]

其中：

- \(z_i\)：题目或行为需求的简短类型描述；
- \(d_i\)：修复机制，即为什么当前 Skill 会失败、需要补充什么行为；
- \(\ell_i\)：目标 Skill 区域；
- \(p_i\)：最小 Patch；
- \(b_i\)：适用边界和不适用条件；
- \(I_i\)：后续抽象必须保留的关键操作；
- \(q_i^0\)：原始经验成功率。

推荐结构：

```json
{
  "record_id": "R0007",
  "sample_id": "sample-42",
  "question_type": "evidence-grounded answering",
  "repair_mechanism": "verify every conclusion against an observable evidence span",
  "target_region": "verification rules",
  "patch": {
    "op": "add",
    "content": "Before answering, verify that each conclusion is supported by a located evidence span."
  },
  "applicable_condition": "the task requires evidence-grounded conclusions",
  "inapplicable_boundary": "do not invent a span when evidence is missing",
  "must_preserve": [
    "locate evidence before forming the final conclusion",
    "do not replace missing evidence with an unsupported guess"
  ],
  "baseline_q": 0.0
}
```

### 5.3 生成约束

PatchRecord 应满足：

1. 不包含原题答案、具体数字或样本特定实体；
2. `repair_mechanism` 描述功能机制，而不是 add/delete/replace 操作名；
3. `patch` 尽量最小，只修复当前识别的问题；
4. `inapplicable_boundary` 明确规则可能被误用的场景；
5. `must_preserve` 使用可检查的动作描述，而不是“提高鲁棒性”等空泛表达。

---

## 6. Step 3：按修复机制形成叶簇

### 6.1 叶簇的含义

叶簇不是题目领域聚类，也不是表面语义聚类。它要寻找的是：

> 哪些样本可以由同一条具体、可执行的局部规则共同修复？

### 6.2 聚类依据

聚类的主要依据为：

\[
\operatorname{LeafKey}(R_i)
=
(d_i,\ell_i,b_i).
\]

即：

1. 修复机制兼容；
2. 目标 Skill 区域兼容；
3. 适用边界不冲突。

`question_type` 只作为辅助信息，不是硬边界。不同题型若需要相同操作规则，可以进入同一叶簇；相同题型若失败机制不同，则必须进入不同叶簇。

### 6.3 不允许的合并

以下记录不应因为文本相似而合并：

- 修改位置不同且不能由同一编辑表达；
- 一个要求“始终执行”，另一个要求“仅在条件成立时执行”；
- `must_preserve` 之间存在直接冲突；
- 一个修复推理过程，另一个只修复输出格式；
- 为满足目标簇数而强行合并不同机制。

### 6.4 低支持处理

每个正式叶簇至少需要 `min_support` 个独立样本。低支持记录进入 tail bank：

```text
current-step singleton
    -> tail bank
    -> 后续步骤出现同机制证据
    -> 跨步骤合并为正式叶簇
```

tail bank 必须记录生成它时的 Skill hash 或 step，以减少旧 Skill 状态下的过时证据污染。

---

## 7. Step 4：叶 Patch 生成

### 7.1 输入

对每个叶簇 \(C_k\)，输入：

- 簇内全部 PatchRecord；
- 当前 Skill 的目标区域；
- 每条记录的适用边界；
- 每条记录的 `must_preserve`；
- 支持样本数量和原始成功率。

### 7.2 输出

生成叶 Patch：

\[
P_k=\operatorname{LeafMerge}(C_k).
\]

推荐结构：

```json
{
  "leaf_id": "L2",
  "repair_mechanism": "evidence verification",
  "support_sample_ids": ["s3", "s8", "s21"],
  "support_count": 3,
  "patch": {
    "op": "add",
    "target": "verification rules",
    "content": "Verify that each conclusion is supported by a located observation before finalizing the answer."
  },
  "applicable_condition": "a conclusion must be grounded in provided or observed evidence",
  "inapplicable_boundary": "when evidence is unavailable, report uncertainty instead of fabricating support",
  "must_preserve": [
    "explicitly locate supporting evidence",
    "do not infer unsupported evidence"
  ]
}
```

### 7.3 叶 Patch 质量要求

- 必须能够应用到当前 Skill；
- 不重复当前 Skill 已有规则；
- 不包含簇外的新修复机制；
- 不把条件规则改写成无条件全局规则；
- 不为了覆盖所有记录而生成无法执行的长篇原则。

---

## 8. Step 5：抽样 leaf self-check

### 8.1 目的

leaf self-check 只回答：

> 当前生成的叶 Patch 是否至少能够修复产生它的局部错误？

它不证明泛化性，也不使用 val。

### 8.2 支持样本抽样

对叶簇 \(C_k\)，选择：

\[
A_k\subseteq C_k,
\qquad
|A_k|=\min(m_{\mathrm{leaf}},|C_k|).
\]

推荐 \(m_{\mathrm{leaf}}=3\)，样本组成如下：

1. 一个最接近簇机制中心的典型样本；
2. 一个边界、最困难或原始成功率最低的样本；
3. 一个随机样本。

如果簇只有两个样本，则检查两个；如果只有一个低支持样本，默认进入 tail bank，而不是单独形成正式叶节点。

### 8.3 第一次检查

将叶 Patch 应用到当前 Skill：

\[
S_k^{\mathrm{leaf}}=S_t\oplus P_k.
\]

每个抽样支持样本先只执行一次：

\[
y_{ik}^{\mathrm{leaf}}
\sim
M(\cdot\mid x_i,S_k^{\mathrm{leaf}}).
\]

定义叶 Patch 的样本级改进：

\[
\delta_{ik}^{\mathrm{leaf}}
=
r(x_i,y_{ik}^{\mathrm{leaf}})-q_i^0.
\]

定义被成功修复的证据集合：

\[
D_k^+
=
\{x_i\in A_k:\delta_{ik}^{\mathrm{leaf}}>0\}.
\]

叶节点通过条件为：

\[
\frac{|D_k^+|}{|A_k|}\ge \eta_{\mathrm{leaf}},
\]

并且 Patch 应用过程没有产生无效编辑或破坏 Skill 结构。

### 8.4 一次最小修改重试

若第一次检查未通过，允许一次修正：

1. 输入原始叶 Patch；
2. 输入失败支持样本的原始轨迹和当前检查结果；
3. 只允许修改当前叶 Patch；
4. 不允许引入新修复机制；
5. 不允许扩大到簇外问题；
6. 对同一抽样支持集合重新执行一次。

若重试后通过，则使用修正后的叶 Patch。若仍未通过，则丢弃**当前叶 Patch 候选**，并记录：

```text
status: rejected_after_one_retry
reason: no observed repair on sampled support
```

这里不能声称“该问题无法被 Skill 修复”。正确含义是：在当前生成器、当前 Patch 形式和一次修正预算下，没有得到可验证的局部修复。

### 8.5 自适应重复

为节省资源，不对所有样本固定重复多次。只有以下情况才追加一次 rollout：

- 第一次结果处于通过阈值边缘；
- 模型或环境随机性较强；
- 某个代表样本与簇内其他样本结果相反。

明显通过或明显失败时不追加采样。

---

## 9. Step 6：构造局部修复证书

通过 leaf self-check 的节点被视为具有经验局部修复能力，但其结论只覆盖抽样支持数据。

为每个叶节点构造修复证书：

\[
\Gamma_k
=
(P_k,D_k^+,Y_k^+,B_k,I_k),
\]

其中：

- \(P_k\)：最终叶 Patch；
- \(D_k^+\)：该 Patch 实际修复成功的抽样支持样本；
- \(Y_k^+\)：叶 Patch 在这些样本上的奖励或输出摘要；
- \(B_k\)：适用条件和不适用边界；
- \(I_k\)：后续父节点必须保留的关键动作。

推荐存储结构：

```json
{
  "leaf_id": "L2",
  "status": "certified",
  "patch_hash": "...",
  "checked_sample_ids": ["s3", "s8", "s21"],
  "repaired_sample_ids": ["s3", "s21"],
  "leaf_scores": {
    "s3": 1.0,
    "s21": 1.0
  },
  "baseline_scores": {
    "s3": 0.0,
    "s21": 0.0
  },
  "must_preserve": [
    "locate supporting evidence",
    "do not fabricate missing support"
  ],
  "retry_used": false
}
```

后续树切面可代替性检查只需要使用这些已修复证据，不需要重新使用整个训练簇。

---

## 10. Step 7：构建稀疏 PatchTree

### 10.1 树结构

通过修复证书检查的叶节点构成：

\[
\mathcal P_{\mathrm{leaf}}
=
\{P_1,\ldots,P_K\}.
\]

推荐使用三层稀疏树：

```text
Certified Leaf Patches
        -> Mechanism-level Internal Nodes
        -> Root Candidate
```

### 10.2 Planner 的职责

Planner 只负责提出树拓扑和抽象计划，不执行环境评测。它根据以下信息决定哪些叶节点优先合并：

- 修复机制是否共享高层原则；
- `must_preserve` 是否兼容；
- 适用条件能否被一个共同规则表达；
- 不适用边界是否冲突；
- 一个规则是否是另一个规则的特殊情况；
- 是否可以使用“共享核心 + 条件残差”表示差异。

### 10.3 树边语义

PatchTree 的边不表示一般语义相似，而表示修复可替代性假设：

> 父节点尝试使用一个更高层规则替代子节点，同时保留子节点已经证明有效的关键修复动作。

对于内部节点 \(v\)，其覆盖叶集合为：

\[
L(v)=\bigcup_{u\in\operatorname{Child}(v)}L(u).
\]

### 10.4 稀疏性约束

为控制生成和评测成本：

- 每轮最多保留 \(K_{\max}\) 个 certified leaves；
- 每个 Internal 节点建议覆盖 2--4 个孩子；
- 默认深度为 3，即 leaf -> internal -> root；
- 不枚举所有节点对；
- 同一层节点在一个批量调用中生成；
- 无法合理合并的叶节点可以直接成为 Root 的孩子。

---

## 11. Step 8：Internal 节点生成与结构检查

### 11.1 Internal 节点输出

对于孩子集合 \(\operatorname{Child}(v)\)，生成：

\[
P_v
=
\operatorname{AbstractMerge}
(\{P_u:u\in\operatorname{Child}(v)\}).
\]

推荐输出：

```json
{
  "node_id": "M1",
  "child_ids": ["L1", "L2", "L3"],
  "covered_leaf_ids": ["L1", "L2", "L3"],
  "shared_core": "Verify every conclusion against an observable source before finalizing it.",
  "conditional_residuals": [
    {
      "condition": "the source is a document",
      "rule": "locate the supporting text span"
    },
    {
      "condition": "the source is a table",
      "rule": "locate the supporting cell or range"
    }
  ],
  "inapplicable_boundary": "when no observation is available, express uncertainty",
  "preserved_constraints": {
    "L1": ["..."],
    "L2": ["..."],
    "L3": ["..."]
  },
  "unresolved_conflicts": []
}
```

### 11.2 Internal 节点不执行 rollout

Internal 节点生成后只做结构检查：

1. **覆盖完整性**：所有孩子都出现在 `covered_leaf_ids`；
2. **约束保留**：每个孩子的 `must_preserve` 都有明确映射；
3. **边界一致性**：父节点没有删除或反转孩子的适用边界；
4. **冲突显式化**：无法统一的差异进入 `conditional_residuals`；
5. **位置合法性**：父 Patch 可以应用到当前 Skill；
6. **非样本化**：父节点不包含具体题目、答案、数字或实体；
7. **非重复性**：不重复当前 Skill 已有规则。

结构检查未通过时允许重新生成一次；再次失败则不建立该父节点，保留其孩子作为更细粒度的切面元素。

这一步不消耗环境 rollout，因为 Internal 节点是否具有真实行为效果，将在它参与完整树切面时统一检查。

---

## 12. Step 9：Root Candidate 生成

Root 尝试对当前所有 Internal 节点和未合并叶节点进行最高层抽象。

Root 必须优先输出：

```text
Shared Core
    所有孩子真正共享的高层原则

Conditional Residuals
    不能安全上升为全局规则的局部差异

Conflict Boundaries
    规则不能被应用的条件
```

Root 不是所有孩子文本的摘要，也不是必须消除全部条件分支。Root 的目标是在不丢失关键机制的前提下，尽量提高共享程度和压缩程度。

Root 生成后执行与 Internal 节点相同的结构检查，但不立即断言其有效。Root 的真实可替代性和泛化性由后续树切面检查决定。

---

## 13. Step 10：树切面定义与 Skill 编译

### 13.1 树切面

树切面 \(\mathcal C\) 是一组互不重叠的节点，并且每个 certified leaf 恰好被其中一个节点覆盖。

例如：

\[
\mathcal C=\{M_1,M_2,L_6\}.
\]

表示：

- \(M_1\) 替代其覆盖的多个叶节点；
- \(M_2\) 替代另一组叶节点；
- \(L_6\) 尚不能继续抽象，因此保留叶级规则。

### 13.2 切面编译

切面必须被编译为一个 Skill，而不是运行时多 Skill 路由：

\[
S(\mathcal C)
=
S_t\oplus\operatorname{Compile}(\mathcal C).
\]

推荐输出结构：

```text
General Principles
- 多个切面节点共同支持的共享原则。

Conditional Strategies
- When condition A holds: apply ...
- When condition B holds: apply ...

Conflict Boundaries
- Do not apply rule A when ...

Final Verification
- 在最终回答或行动前执行统一检查。
```

### 13.3 防止 Skill 无限增长

编译器不能简单拼接切面节点，而应：

1. 提取节点之间的共享核心；
2. 只把差异写成条件残差；
3. 删除重复或被新规则覆盖的旧规则；
4. 使用 replace/merge 更新目标区域，而不是持续 append；
5. 限制条件分支数量；
6. 限制本轮新增 token 数；
7. 超出长度预算时拒绝候选或保留更高层节点。

定义 Skill 复杂度：

\[
\Omega(S(\mathcal C))
=
\alpha|\mathcal C|
+
\beta\operatorname{Tokens}(S(\mathcal C))
+
\gamma\operatorname{Branches}(S(\mathcal C)).
\]

越靠近 Root 的切面通常节点更少、文本更短；向叶节点展开意味着使用更具体但更复杂的表示。

---

## 14. Step 11：树切面的局部可代替性检查

### 14.1 检查对象

不对 Internal 节点逐个 rollout，而只检查当前完整切面编译出的候选 Skill \(S(\mathcal C)\)。

### 14.2 检查数据

构造叶节点已经成功修复的证据集合：

\[
\mathcal D_{\mathrm{repair}}
=
\bigcup_{k=1}^{K}D_k^+.
\]

为了进一步节约资源，第一次切面检查可以从每个叶证书中最多取一个代表性修复样本：

\[
\widetilde{D}_k^+\subseteq D_k^+,
\qquad |\widetilde{D}_k^+|\le 1.
\]

只有结果处于阈值边缘时，才追加该叶证书中的其他修复样本。

### 14.3 可代替性定义

对于来自叶节点 \(k\) 的修复样本 \(x\)，叶 Patch 已经记录了奖励 \(r_k^{\mathrm{leaf}}(x)\)。将切面候选运行在同一样本上，得到：

\[
r_{\mathcal C}(x)=r(x;S(\mathcal C)).
\]

定义总体能力保留率：

\[
R_{\mathrm{retain}}(\mathcal C)
=
\frac{
\sum_k\sum_{x\in \widetilde{D}_k^+}
\mathbf 1
\left[
r_{\mathcal C}(x)
\ge
r_k^{\mathrm{leaf}}(x)-\epsilon
\right]
}{
\sum_k|\widetilde{D}_k^+|
}.
\]

可同时记录每个叶证书的保留结果，避免总体平均掩盖某一类修复能力被完全破坏：

\[
R_k(\mathcal C)
=
\frac{1}{|\widetilde{D}_k^+|}
\sum_{x\in \widetilde{D}_k^+}
\mathbf 1
\left[
r_{\mathcal C}(x)
\ge
r_k^{\mathrm{leaf}}(x)-\epsilon
\right].
\]

推荐通过条件：

\[
R_{\mathrm{retain}}(\mathcal C)
\ge
\eta_{\mathrm{retain}},
\]

并且不存在被完全破坏的高支持叶节点。

### 14.4 检查结果的解释

- 通过：当前切面至少保留了主要局部修复能力，可以进入 val；
- 失败：当前抽象不能替代其覆盖的部分叶节点，应向下展开；
- 局部失败集中在某些叶证书：优先展开覆盖这些叶子的祖先节点；
- 随机或边缘失败：只追加失败叶证书中的一个支持样本，不立即评估整个簇。

该检查使用训练支持数据，只证明可代替性，不作为泛化证据。

---

## 15. Step 12：树切面的 val 泛化检查

### 15.1 进入条件

只有通过局部可代替性检查的切面才进入 val。这样可以避免把验证预算浪费在已经丢失叶节点修复能力的候选上。

### 15.2 基线缓存

当前 Skill 在验证集上的结果只计算一次并缓存：

\[
J_{\mathrm{val}}(S_t).
\]

对候选切面计算：

\[
J_{\mathrm{val}}(S(\mathcal C)).
\]

### 15.3 接受条件

核心接受条件采用简单的增量 gate：

\[
J_{\mathrm{val}}(S(\mathcal C))
\ge
J_{\mathrm{val}}(S_t)+\tau_{\mathrm{val}}.
\]

不需要为每个节点定义复杂的“增益减负迁移减长度”分数。局部能力丢失已经由可代替性约束控制，Skill 复杂度则由自顶向下的搜索顺序和长度预算控制。

### 15.4 小验证集上的随机波动

若 val 很小，单个样本就可能造成显著分数变化。推荐：

- 明显高于阈值：直接通过；
- 明显低于当前 Skill：直接失败；
- 只多答对一个样本或差值处于预设边界：重复评估一次；
- 重复评估也计入 rollout 预算；
- 不根据 val 的具体失败内容重新生成 Patch，以避免验证泄漏。

---

## 16. Step 13：预算约束的最高抽象切面搜索

### 16.1 搜索目标

定义可行切面集合：

\[
\mathcal F(T)
=
\left\{
\mathcal C\in\operatorname{Cuts}(T):
R_{\mathrm{retain}}(\mathcal C)\ge\eta_{\mathrm{retain}},
\ J_{\mathrm{val}}(S(\mathcal C))\ge J_{\mathrm{val}}(S_t)+\tau_{\mathrm{val}}
\right\}.
\]

最终选择：

\[
\mathcal C^\star
=
\arg\min_{\mathcal C\in\mathcal F(T)}
\Omega(S(\mathcal C)),
\]

满足：

\[
\operatorname{EvalCost}(\mathcal C)\le B_{\mathrm{cut}}.
\]

如果复杂度相同，则选择 val 分数更高的切面。

这个目标精确表达了方法意图：在保留局部修复能力且具有验证集收益的条件下，选择最高、最紧凑的抽象层级。

### 16.2 自顶向下搜索

初始化最高抽象切面：

\[
\mathcal C_0=\{\operatorname{Root}(T)\}.
\]

每轮搜索：

1. 编译当前切面；
2. 检查局部可代替性；
3. 若局部检查失败，展开覆盖失败叶证书的节点；
4. 若局部检查通过，执行 val 检查；
5. 若 val 通过，立即接受当前切面；
6. 若 val 失败，进入更具体的预定义下一层切面；
7. 达到切面评测预算后停止。

### 16.3 推荐的三层搜索顺序

```text
C0 = {Root}
    |
    | failure
    v
C1 = {Root 的直接孩子，即 Internal/Leaf 混合切面}
    |
    | failure and budget remains
    v
C2 = 对失败证书对应的高风险 Internal 节点进行局部展开
```

默认 \(B_{\mathrm{cut}}=2\)。资源允许时可设为 3，但不执行无限深搜索。

### 16.4 Root 失败后的重新抽象

Root 失败不一定说明抽象文本有问题，也可能来自局部能力丢失、规则冲突、val 随机波动或候选本身无泛化收益。因此不无限重新生成 Root。

Root 失败后的直接孩子切面编译，本身就是一次受约束的保守重新抽象：

```text
保留 Root 中确实共享的核心
+ 恢复孩子节点的条件残差
+ 恢复被 Root 删除的适用边界
```

重新编译时只使用孩子节点、修复证书和结构约束，不把 val 题目、答案或失败轨迹反馈给 LLM。

若保守切面仍然失败，则放弃本轮更新，而不是继续在同一 val 上生成大量候选。

---

## 17. Step 14：Skill 写回与状态更新

### 17.1 成功写回

若找到可行切面 \(\mathcal C^\star\)，则：

\[
S_{t+1}=S(\mathcal C^\star).
\]

同时保存：

- 完整树结构；
- 所有 certified leaves；
- 被丢弃叶节点及原因；
- 每个切面的编译结果；
- 可代替性检查结果；
- val 分数；
- 最终被选择的切面；
- Skill 长度变化；
- 本轮 rollout 成本。

### 17.2 本轮失败

若在预算内没有切面同时通过两类检查，则：

\[
S_{t+1}=S_t.
\]

失败不意味着叶节点全部无效。可以保留：

- certified leaf artifacts；
- 修复证书；
- 未解决的冲突；
- 失败切面的结构诊断。

但这些内容不能未经新一轮检查直接写入 Skill。

### 17.3 tail bank 更新

以下证据进入 tail bank：

- 支持数不足的 PatchRecord；
- 叶 Patch 生成失败但机制标签可靠的记录；
- 因本轮叶节点数量上限而未处理的记录。

以下内容不应进入 tail bank：

- 明确包含样本答案或实体的 Patch；
- 与当前 Skill 无关的执行随机失误；
- 已经证明与其他修复机制冲突且没有适用边界的规则。

---

## 18. 完整伪代码

### 18.1 叶节点生成与认证

```text
Algorithm 1: BuildCertifiedLeaves(S_t, PatchRecords)

Input:
    current Skill S_t
    PatchRecords R
    leaf sample budget m_leaf
    one-retry budget retry_max = 1

1. clusters <- ClusterByRepairMechanism(R)
2. certified_leaves <- []
3. tail_records <- []

4. for each cluster C_k in clusters:
5.     if IndependentSupport(C_k) < min_support:
6.         tail_records.append(C_k)
7.         continue

8.     P_k <- GenerateLeafPatch(S_t, C_k)
9.     A_k <- SelectRepresentativeSupport(C_k, max_items=m_leaf)
10.    report <- EvaluateLeafPatchOnce(S_t, P_k, A_k)

11.    if not PassLeafRepair(report):
12.        P_k <- ReviseLeafPatchOnce(P_k, report, C_k)
13.        report <- EvaluateLeafPatchOnce(S_t, P_k, A_k)

14.    if not PassLeafRepair(report):
15.        SaveRejectedLeaf(P_k, report)
16.        continue

17.    Gamma_k <- BuildRepairCertificate(P_k, report)
18.    certified_leaves.append((P_k, Gamma_k))

19. return certified_leaves, tail_records
```

### 18.2 树构建

```text
Algorithm 2: BuildSparsePatchTree(S_t, CertifiedLeaves)

Input:
    current Skill S_t
    certified leaves {(P_k, Gamma_k)}
    maximum depth D = 3

1. plan <- PlanSparseMergeTopology(CertifiedLeaves)
2. internal_nodes <- BatchGenerateInternalNodes(plan, CertifiedLeaves)
3. internal_nodes <- StructuralCheckAndRepairOnce(internal_nodes)
4. replace invalid internal nodes with their original children
5. root <- GenerateRoot(internal_nodes and unmerged leaves)
6. root <- StructuralCheckAndRepairOnce(root)
7. if root remains structurally invalid:
8.     use its direct children as the highest available cut
9. return PatchTree(root, internal_nodes, certified leaves)
```

### 18.3 双证据树切面搜索

```text
Algorithm 3: SearchHighestFeasibleCut(S_t, T, Certificates, D_val)

Input:
    current Skill S_t
    PatchTree T
    leaf repair certificates Gamma
    validation set D_val
    cut evaluation budget B_cut

1. baseline_val <- CachedEvaluate(S_t, D_val)
2. frontier <- HighestAvailableCut(T)       # normally {Root}
3. best_candidate <- None

4. for budget_index in 1..B_cut:
5.     candidate <- CompileCut(S_t, frontier, token_budget)
6.     if candidate is invalid or exceeds token_budget:
7.         frontier <- ExpandRiskyNodes(frontier)
8.         continue

9.     retain_report <- EvaluateRetention(candidate, Gamma)
10.    if not PassRetention(retain_report):
11.        frontier <- ExpandNodesCoveringFailedCertificates(
12.            frontier, retain_report.failed_leaf_ids
13.        )
14.        continue

15.    candidate_val <- Evaluate(candidate, D_val)
16.    if IsBorderline(candidate_val, baseline_val):
17.        candidate_val <- RepeatOnceAndAggregate(candidate, D_val)

18.    if candidate_val >= baseline_val + tau_val:
19.        best_candidate <- candidate
20.        break                            # first pass = highest feasible cut

21.    frontier <- NextMoreSpecificCut(frontier, T)

22. return best_candidate                   # None means keep S_t
```

### 18.4 外层训练循环

```text
Algorithm 4: DualEvidencePatchTreeOptimization

Input:
    frozen model M
    initial Skill S_0
    D_train, D_val

1. S <- S_0
2. cache baseline val results for S

3. for each optimization step t:
4.     trajectories <- RepeatedRollout(M, S, TrainBatch_t)
5.     records <- GeneratePatchRecords(trajectories)

6.     if records is empty:
7.         continue

8.     leaves, tail <- BuildCertifiedLeaves(S, records)
9.     UpdateTailBank(tail)

10.    if leaves is empty:
11.        continue

12.    T <- BuildSparsePatchTree(S, leaves)
13.    candidate <- SearchHighestFeasibleCut(S, T, leaves.certificates, D_val)

14.    if candidate is not None:
15.        S <- candidate
16.        update cached val baseline

17. return S
```

---

## 19. 决策表

### 19.1 叶节点

| 第一次检查 | 一次修正后 | 决策 |
|---|---|---|
| 通过 | 不执行 | 生成修复证书 |
| 失败 | 通过 | 使用修正后叶 Patch，生成修复证书 |
| 失败 | 失败 | 丢弃当前叶 Patch 候选 |
| 边缘 | 重复一次 | 根据聚合结果决定 |

### 19.2 树切面

| 局部可代替性 | val 泛化 | 决策 |
|---|---|---|
| 通过 | 通过 | 接受当前切面 |
| 失败 | 不评估 | 展开破坏修复证书的节点 |
| 通过 | 失败 | 尝试更具体的切面 |
| 失败 | 即使 val 偶然较高 | 不作为合法 PatchTree 抽象接受 |
| 通过 | 边缘 | 允许一次重复 val |
| 多个切面均失败 | 失败 | 保持当前 Skill |

其中，“局部失败但 val 较高”仍不接受，是因为该候选无法被解释为叶节点修复证据的有效抽象。如果希望接受这种候选，应将其作为另一个独立的全局 Skill 优化方法，而不能声称它是 PatchTree 的父节点替代结果。

---

## 20. 评测预算与复杂度

设：

- certified leaf 数为 \(K\)；
- 每个叶节点最多检查 \(m_{\mathrm{leaf}}\) 个样本；
- 切面搜索预算为 \(B_{\mathrm{cut}}\)；
- 每个切面初始使用每叶一个修复证书样本；
- val 大小为 \(|\mathcal D_{\mathrm{val}}|\)。

叶节点检查成本近似为：

\[
O(Km_{\mathrm{leaf}}),
\]

最坏情况下的一次修正重试将该部分至多放大一倍。

切面局部可代替性检查成本近似为：

\[
O(B_{\mathrm{cut}}K),
\]

val 成本上界为：

\[
O(B_{\mathrm{cut}}|\mathcal D_{\mathrm{val}}|),
\]

但只有通过局部检查的切面才消耗 val 预算。

与逐节点评估相比，该方法的关键节省来自：

- 不对每个 Internal 节点 rollout；
- 不在 val 上逐个筛选所有孩子；
- 不枚举所有树切面；
- 同层节点批量生成；
- 原 Skill 的训练与 val baseline 全部缓存；
- 支持检查采用代表性抽样和边缘自适应重复。

---

## 21. 推荐初始超参数

以下数值用于第一轮可行性实验，不应被写成普适常数：

| 参数 | 建议初值 | 含义 |
|---|---:|---|
| `rollout_repeats` | 3 | 生成 PatchRecord 前的训练样本重复执行次数 |
| `min_support` | 2 | 正式叶簇所需独立样本数 |
| `max_leaf_groups` | 6 | 每轮最多 certified leaves |
| `leaf_self_check_samples` | 3 | 每个叶节点首次检查的最大支持样本数 |
| `leaf_retry_max` | 1 | 叶 Patch 最大修改重试次数 |
| `leaf_repair_threshold` | 2/3 | 叶 Patch 抽样修复通过阈值 |
| `tree_depth` | 3 | leaf -> internal -> root |
| `internal_fanout` | 2--4 | 一个 Internal 节点覆盖的孩子数 |
| `retain_samples_per_leaf` | 1 | 每个切面首次可代替性检查使用的证书样本数 |
| `retain_threshold` | 0.8--1.0 | 切面局部能力保留率阈值 |
| `cut_search_budget` | 2 | 每轮最多评估的树切面数 |
| `root_retry_max` | 0 | 不进行无约束 Root 重写；使用孩子切面作为保守重新抽象 |
| `skill_token_delta_budget` | 任务相关 | 单轮允许增加的 Skill token 上限 |

当 val 较小时，`tau_val` 不应简单设为“任何正提升”。可以要求至少净增加两个正确样本，或对只增加一个正确样本的候选重复评估一次。

---

## 22. 必须保存的实验产物

为了复现、诊断和论文分析，每轮至少保存：

```text
patch_records.jsonl
leaf_clusters.json
leaf_patches.json
leaf_self_check/<leaf_id>/summary.json
rejected_leaves.json
repair_certificates.json
patch_tree.json
internal_structural_check.json
root_candidate.md
cut_00_root/
    compiled_skill.md
    retention_report.json
    val_summary.json
cut_01_children/
    compiled_skill.md
    retention_report.json
    val_summary.json
selected_cut.json
selected_skill.md
cost_report.json
```

`selected_cut.json` 应记录：

```json
{
  "cut_id": "cut_01_children",
  "node_ids": ["M1", "M2", "L6"],
  "covered_leaf_ids": ["L1", "L2", "L3", "L4", "L5", "L6"],
  "retention_score": 0.9,
  "val_baseline": 0.42,
  "val_candidate": 0.48,
  "skill_tokens_before": 820,
  "skill_tokens_after": 960,
  "rollout_cost": {
    "leaf_self_check": 14,
    "cut_retention": 6,
    "cut_validation": 32
  }
}
```

---

## 23. 与当前代码路径的关系

当前仓库已经具备部分基础：

- 样本级 PatchRecord；
- 全局聚类 Planner；
- leaf -> root 和 leaf -> mid -> root 两种深度；
- 基于 `support_sample_ids` 的训练支持样本评估；
- `valid_seen` 上的候选 gate；
- Root 失败后的 child fallback；
- tail bank 和缓存。

但要严格对齐本文档，还需要以下语义调整。

### 23.1 叶节点聚类

当前类型信息可以继续保留，但聚类主轴应明确改为：

```text
repair mechanism
+ target compatibility
+ boundary compatibility
```

题目类型只作为辅助信号。

### 23.2 support self-check

当前支持样本检查更接近本文的叶节点检查，但本文要求新增：

- 每叶支持样本抽样上限；
- 典型/困难/随机的抽样策略；
- 一次最小修改重试；
- 叶节点可行性判断；
- 修复成功样本集合 \(D_k^+\)；
- repair certificate artifact。

当前仅记录 support 指标、不参与叶节点可行性决策的行为，不能完全代表本文算法。

### 23.3 validation leaf self-check

本文不建议在 `valid_seen` 上逐个筛选所有叶节点。叶节点使用 train support samples 检查；val 只评估通过局部可代替性检查的完整树切面。

### 23.4 Internal 节点

Internal 节点不需要独立 rollout，但需要显式保存：

- `must_preserve` 映射；
- `conditional_residuals`；
- 冲突边界；
- 结构检查结果。

### 23.5 Root fallback

当前逐个评估 child、筛选 child、再评估 child 组合的路径，应改为：

```text
Root 失败
    -> 编译完整直接孩子切面
    -> 对整个切面执行一次可代替性检查
    -> 通过后对整个切面执行一次 val 检查
```

不在 val 上逐个选择孩子。

### 23.6 新增切面搜索预算

需要显式配置：

```text
cut_search_budget
retain_samples_per_leaf
retain_threshold
leaf_self_check_samples
leaf_retry_max
skill_token_delta_budget
```

并在 artifact 中记录实际消耗。

---

## 24. 消融实验建议

为验证每个设计是否必要，建议至少进行以下消融。

### 24.1 叶节点修复证书

```text
无 leaf self-check
vs.
抽样 leaf self-check
vs.
抽样 self-check + 一次修改重试
```

关注：certified leaf 比例、最终 test、每轮 rollout 成本。

### 24.2 双证据约束

```text
仅 val gate
vs.
仅局部可代替性
vs.
局部可代替性 + val gate
```

关注：val-test gap、局部修复保留率、最终 test。

### 24.3 树切面策略

```text
固定 Root
vs.
Root -> direct children
vs.
预算约束的三层切面搜索
```

关注：被接受的切面深度、Skill 长度、评测成本和 test。

### 24.4 逐 child 评估与整体切面评估

```text
逐 child val 筛选再组合
vs.
直接编译完整 child cut 后整体评估
```

关注：val 调用数、val-test gap、最终 test 和 Skill 长度。

---

## 25. 方法定位与贡献表述

PatchTree 不应被描述为“先聚类，再构建一棵树”。更准确的表述是：

> PatchTree 将 Skill 更新建模为一个双证据约束的分层抽象问题。方法首先从训练轨迹中构造能够被局部验证的叶级修复假设，并将叶节点实际修复成功的支持样本保存为修复证书；随后，方法通过稀疏树逐级提出更高层的共享规则，但不对所有内部节点进行昂贵评测。对于每个候选树切面，方法先检查其是否能够替代叶节点已经证明有效的局部修复能力，再使用验证集判断其是否具有全局泛化收益。最终，算法在有限评测预算内选择同时满足局部可代替性和验证集提升的最高抽象切面，并将其编译为一个结构化全局 Skill。

最凝练的形式为：

\[
\boxed{
\text{PatchTree}
=
\text{Certified Local Repair}
+
\text{Hierarchical Abstraction}
+
\text{Budgeted Feasible-Cut Search}.
}
\]

三项核心贡献可以概括为：

1. **修复证书**：用低成本训练支持检查证明叶 Patch 具有经验局部修复能力；
2. **可代替性树**：树边表示父规则能否替代子规则，而不是普通语义相似；
3. **双证据切面选择**：局部可代替性作为抽象成立约束，val 提升作为全局写回门槛，在评测预算内寻找最高抽象可行切面。

---

## 26. 实现时必须避免的误解

1. 叶 Patch 重试失败不表示问题原则上不可修复，只表示当前候选在当前预算下未被验证；
2. leaf self-check 不能使用 val，否则 val 会参与 Patch 构造；
3. train support 上修复成功不表示具有泛化性；
4. val 提升不表示父节点保留了叶节点修复能力；
5. Internal 节点不需要逐个 rollout；
6. 切面必须整体编译、整体评估，不能先在 val 上逐个筛选孩子再组合；
7. Root 失败后只允许预算内的保守条件化编译，不进行无限 Root 重写；
8. 子节点切面不能原样拼接，必须使用共享核心与条件残差控制 Skill 长度；
9. test 不能用于树切面选择或超参数调节；
10. “最高抽象切面”指满足双证据约束后的最高层级，不是无条件选择 Root。
