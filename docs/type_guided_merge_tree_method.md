# 3 方法

## 3.1 问题定义与方法概述

给定冻结语言模型 \(\pi\)、当前自然语言 Skill \(S_t\)、训练集 \(\mathcal D_{\mathrm{train}}\)、验证集 \(\mathcal D_{\mathrm{val}}\)、测试集 \(\mathcal D_{\mathrm{test}}\)，以及任务奖励函数 \(r(x,y)\)，本文目标是在不更新模型参数的条件下，通过执行反馈迭代优化 Skill，使其在验证集上获得更高任务完成质量，并最终在测试集上提升泛化性能。

自然语言 Skill 的更新需要解决一个核心问题：来自不同样本的失败轨迹可能共享某种可泛化的修复方向，也可能只是局部偶然错误。若直接将所有失败样本的 Patch 合并，容易得到过度宽泛的规则；若只保留单样本 Patch，又会导致 Skill 记忆训练样本，缺少迁移性。因此，Skill 优化不仅要生成候选编辑，还要判断哪些样本属于同一类可合并修复，并进一步决定这些修复应写回到什么抽象层级。

本文提出 **Type-Guided Merge-Tree Skill Optimization**。方法首先在当前 Skill 下对训练样本进行多次采样，并从失败或不稳定样本中生成样本级 Patch 记录。每个记录由题目类型、修正类型、目标位置和最小 Patch 组成：

\[
R_i=(\alpha_i,\beta_i,\ell_i,p_i).
\]

其中，\(\alpha_i\) 表示题目类型，即对题目输入需求结构的类型化抽象；\(\beta_i\) 表示修正类型，即对 Patch 功能性修改意图的类型化抽象；\(\ell_i\) 表示目标 Skill 位置；\(p_i\) 表示具体的最小 Patch 建议。用于样本分类和叶子簇归纳的核心签名为：

\[
s_i=(\alpha_i,\beta_i).
\]

题目类型和修正类型共同决定样本属于哪一类可合并修复。题目类型不是普通领域标签，例如数学题、写作题或代码题，而是题目对模型行为提出的结构性需求；修正类型也不是 add、delete 或 replace 这样的编辑操作，而是对修复意图的类型化概括，例如约束验证、格式约束、证据检查或步骤分解。

随后，方法根据类型签名生成叶子簇。叶子簇表示一组具有相近题目类型和相同或相近修正类型的样本，因此可以合并为同一最低层 Skill 编辑。为了控制簇粒度，LLM 在生成叶子簇时不仅依据类型兼容性，还根据当前 batch 中失败或不稳定样本数量确定目标簇数，使每个簇大约包含 6--8 个样本。

每个叶子簇具有一个簇级类型原型：

\[
P_j=(A_j,B_j),
\]

其中 \(A_j\) 是簇级题目类型，\(B_j\) 是簇级修正类型。簇级类型原型是对簇内样本类型签名的一步抽象，代表一类数据及其对应的一类局部修复方向。

簇内样本 Patch 被合并为叶节点编辑，并在其支持样本上执行 self-check。若初始叶节点无法修复自身支持样本，算法允许一次最小自修正；修正后仍失败的簇被丢弃。通过 self-check 的叶节点构成合并树的叶子。随后，LLM Planner 根据叶节点摘要规划合并路径，决定哪些叶节点应合并，哪些应保持分离，以及每个内部节点对应怎样的更抽象编辑。由此得到一棵合并树：

\[
\text{Sample Patch}
\rightarrow
\text{Leaf Edit}
\rightarrow
\text{Internal Merge Nodes}
\rightarrow
\text{Root Candidate}.
\]

最后，算法在验证集上执行受限的自顶向下剪枝：优先评估根候选；若根候选通过，则写回最高层合并结果；若根候选失败，则只回退到根的直接子节点，评估并组合验证集收益为正的子节点；若子节点组合仍失败，则放弃本轮更新。

整体流程为：

\[
\boxed{
\text{多采样 Patch 记录生成}
\rightarrow
\text{类型与数量感知的叶子簇归纳}
\rightarrow
\text{叶节点自检与一次修正}
\rightarrow
\text{LLM 规划合并树}
\rightarrow
\text{验证集剪枝写回}.
}
\]

---

## 3.2 数据使用协议

本文采用 Train/Val/Test 三部分数据划分：

\[
\mathcal D
=
\mathcal D_{\mathrm{train}}
\cup
\mathcal D_{\mathrm{val}}
\cup
\mathcal D_{\mathrm{test}}.
\]

三者承担不同职责。

训练集 \(\mathcal D_{\mathrm{train}}\) 用于执行采样、生成样本级 Patch 记录、归纳叶子簇、构造叶节点编辑，并进行簇内 self-check 与一次自修正。训练集上的评估只用于判断某个叶节点是否至少能修复其支持样本，不作为泛化性能证据。

验证集 \(\mathcal D_{\mathrm{val}}\) 用于合并树的抽象层级选择和本轮候选 Skill 的接受判断。具体而言，验证集用于评估 Root Candidate；若 Root Candidate 失败，则用于评估根的直接子节点及其组合。验证集不参与样本 Patch 生成、叶子簇归纳或叶节点自修正。

测试集 \(\mathcal D_{\mathrm{test}}\) 不参与任何优化、剪枝或选择过程，只用于最终性能报告。

因此，数据职责可以概括为：

\[
\mathcal D_{\mathrm{train}}
\Rightarrow
\text{生成与簇内可靠性检查},
\]

\[
\mathcal D_{\mathrm{val}}
\Rightarrow
\text{合并树剪枝与写回选择},
\]

\[
\mathcal D_{\mathrm{test}}
\Rightarrow
\text{最终评估}.
\]

---

## 3.3 多采样样本级 Patch 记录生成

在第 \(t\) 轮优化中，对于训练样本 \(x_i\in\mathcal D_{\mathrm{train}}\)，使用当前 Skill \(S_t\) 独立采样 \(K\) 条执行轨迹：

\[
\tau_{ik}
\sim
\pi(\cdot\mid x_i,S_t),
\qquad
k=1,\ldots,K.
\]

每条轨迹获得任务奖励：

\[
r_{ik}=r(x_i,\tau_{ik})\in[0,1].
\]

样本在当前 Skill 下的经验成功率为：

\[
q_i^0
=
\frac{1}{K}
\sum_{k=1}^{K}r_{ik}.
\]

若样本稳定成功，即：

\[
q_i^0\geq \tau_{\mathrm{succ}},
\]

则该样本不产生修复 Patch。若样本失败或表现不稳定，则分析器联合读取该样本的多条轨迹，生成样本级 Patch 记录：

\[
R_i=(\alpha_i,\beta_i,\ell_i,p_i).
\]

其中，\(\alpha_i\) 和 \(\beta_i\) 构成用于分类和聚簇的类型签名：

\[
s_i=(\alpha_i,\beta_i).
\]

题目类型 \(\alpha_i\) 表示题目的结构性需求类型。它应概括该样本要求模型处理的输入结构、约束结构或作答结构，而不是描述具体题面内容。例如：

```text
explicit-constraint following
format-controlled generation
multi-step reasoning
evidence-grounded answering
comparison-and-selection
ambiguous-intent handling
tool-use decision
```

修正类型 \(\beta_i\) 表示 Patch 的功能性修复意图。它应概括该样本需要补充、约束或修改哪一类 Skill 行为，而不是描述具体编辑操作。例如：

```text
constraint-verification
format-enforcement
step-decomposition
evidence-checking
calculation-verification
ambiguity-clarification
overgeneralization-control
answer-completeness-check
```

目标位置 \(\ell_i\) 表示该 Patch 应写入 Skill 的哪个部分，用于后续合并兼容性检查；最小 Patch \(p_i\) 表示具体编辑建议，采用 add、delete 或 replace 形式。注意，\(\ell_i\) 和 \(p_i\) 不作为主要分类依据，而是在叶节点编辑构造阶段使用。

例如，一个样本记录可以写作：

```text
question_type: explicit-constraint following
revision_type: constraint-verification
target: verification rules
patch: add a rule requiring all explicit constraints to be checked before final answer
```

类型签名必须短且稳定。它不包含原题中的具体实体、数字、答案或样本特定事实。其作用不是解释完整失败轨迹，而是为后续聚簇提供类型级分类依据。

---

## 3.4 类型与数量感知的叶子簇归纳

样本记录中的类型签名用于形成叶子簇。叶子簇的目标不是发现题目领域类别，也不是发现表面文本相似性，而是发现一组可以合并成同一最低层 Skill 编辑的样本。因此，叶子簇表示一种 **type-compatible local repair**。

首先，收集当前轮所有失败或不稳定样本的 Patch 记录，记为：

\[
\mathcal R_t=
\{R_i:q_i^0<\tau_{\mathrm{succ}}\}.
\]

其数量为：

\[
B_t=|\mathcal R_t|.
\]

若 \(B_t=0\)，说明当前 batch 中没有失败或不稳定样本，算法不生成新的候选编辑，并直接返回 \(S_{t+1}=S_t\)。

为了控制叶子簇粒度，本文设定每个簇的目标样本数约为 6--8。实际实现中取目标簇大小：

\[
s_{\mathrm{cluster}}=7,
\]

并根据当前 batch 中失败或不稳定样本数量计算目标簇数：

\[
M_{\mathrm{target}}
=
\max
\left(
1,
\operatorname{round}
\left(
\frac{B_t}{s_{\mathrm{cluster}}}
\right)
\right).
\]

例如，当当前 batch 中包含 \(B_t=32\) 个失败或不稳定样本时，有：

\[
M_{\mathrm{target}}
=
\operatorname{round}
\left(
\frac{32}{7}
\right)
=
5.
\]

因此，LLM 在该 batch 上默认生成约 5 个叶子簇，使每个簇平均包含 6--8 个样本。

为了减少上下文长度，聚类阶段可以先聚合完全相同的类型签名。设去重后的类型签名集合为：

\[
\widehat{\mathcal S}_t
=
\{(\widehat s_m,n_m,I_m)\}_{m=1}^{N_t},
\]

其中 \(\widehat s_m=(\widehat\alpha_m,\widehat\beta_m)\) 表示一种唯一类型签名，\(n_m\) 是该签名对应的样本数量，\(I_m\) 是对应样本编号集合。LLM 读取这些去重后的类型签名及其数量信息，并在 Prompt 中被要求遵循两个优先级：第一，根据题目类型和修正类型判断样本是否属于同一类可合并修复；第二，在类型兼容的前提下参考 \(M_{\mathrm{target}}\) 控制簇数量，使每个簇大约包含 6--8 个样本。

也就是说，类型兼容性是主要分类依据，簇数量是粒度控制信号。数量要求不强制均匀分配样本，也不允许 LLM 仅为了满足目标簇数而合并明显不兼容的样本。

LLM 输出一组叶子簇原型：

\[
\mathcal P_t
=
\{P_1,\ldots,P_M\},
\]

其中 \(M\) 应接近 \(M_{\mathrm{target}}\)。每个叶子簇原型采用二元类型结构：

\[
P_j=(A_j,B_j).
\]

其中，\(A_j\) 是簇级题目类型，表示簇内样本共享的输入需求结构；\(B_j\) 是簇级修正类型，表示簇内样本共享的修复方向。例如：

```text
Leaf Prototype L1
question_type: explicit-constraint following
revision_type: constraint-verification
```

随后，每个样本记录被分配到一个叶子簇原型：

\[
z_i\in\{1,\ldots,M\}.
\]

每个叶子簇定义为：

\[
C_j
=
\{R_i:z_i=j\}.
\]

与基于最小支持数的硬过滤不同，本文不再因为某个簇大小低于阈值而直接丢弃该簇。LLM 生成的每个非空叶子簇都会进入后续叶节点编辑构造。簇的规模信息保留在叶节点摘要中，供后续合并树规划参考。

该阶段可以对应如下 Prompt 约束：

```text
Given all failed or unstable samples in the current batch, cluster them into leaf clusters.

Each sample has:
- question_type
- revision_type
- target
- patch

Primary criterion:
Cluster samples by compatible question_type and revision_type.

Secondary criterion:
The target cluster size is about 6-8 samples.
Generate approximately M_target clusters, where M_target is computed from the batch size.

Do not force equal cluster sizes.
Do not merge clearly incompatible types only to match the target number.
Output each cluster with:
- cluster_question_type
- cluster_revision_type
- assigned sample ids
```

---

## 3.5 叶节点编辑构造与 Self-Check

对于每个非空叶子簇 \(C_j\)，合并其中的样本 Patch，得到初始叶节点编辑：

\[
e_j^{(0)}
=
\operatorname{MergeLeaf}
\left(
\{p_i:R_i\in C_j\}
\right).
\]

叶节点编辑是最低可写回单位。它比单样本 Patch 更抽象，因为它由同一类型簇内的多个样本支持；同时又比高层合并节点更保守，因为它只对应一个局部类型原型 \(P_j=(A_j,B_j)\)。

为了过滤错误合并和噪声簇，算法在叶节点自己的支持样本上执行 self-check。由于 \(C_j\) 是 Patch 记录集合，记其对应的训练样本集合为：

\[
X(C_j)=\{x_i:R_i\in C_j\}.
\]

定义 Skill \(S\) 在簇 \(C_j\) 上的平均成功率为：

\[
V_{C_j}(S)
=
\frac{1}{|X(C_j)|}
\sum_{x_i\in X(C_j)}
q_i(S),
\]

其中 \(q_i(S)\) 由少量重复采样估计。初始叶节点编辑的簇内收益为：

\[
\Delta_j^{(0)}
=
V_{C_j}(S_t\oplus e_j^{(0)})
-
V_{C_j}(S_t).
\]

若：

\[
\Delta_j^{(0)}>\tau_{\mathrm{self}},
\]

则该叶节点通过 self-check，可以进入合并树。若：

\[
\Delta_j^{(0)}\leq\tau_{\mathrm{self}},
\]

则说明合并编辑即使在其支持样本上也没有产生足够正向效果，可能存在类型归纳过宽、目标位置错误、Patch 冲突或编辑表达不清等问题。此时允许一次叶节点自修正。

---

## 3.6 一次叶节点自修正

当初始叶节点未通过 self-check 时，算法利用 self-check 的对照信息执行一次最小修正。对簇内样本 \(x_i\in X(C_j)\)，定义加入初始叶节点编辑前后的变化：

\[
d_i
=
q_i(S_t\oplus e_j^{(0)})
-
q_i(S_t).
\]

分析器读取改善样本、退化样本和无变化样本的轨迹摘要与 Patch 摘要，检查初始叶节点编辑中的冲突、错误边界或表述问题，并生成一次修正版编辑：

\[
e_j^{(1)}
=
\operatorname{ReviseLeaf}
\left(
e_j^{(0)},C_j,\{d_i\}_{x_i\in X(C_j)}
\right).
\]

修正受到两个限制。第一，修正后的编辑不得引入样本特定实体、数字或答案。第二，修正必须保持最小，优先修改适用条件、边界描述或含混指令，而不是重写整个规则。

修正后再次在同一支持簇上执行 self-check：

\[
\Delta_j^{(1)}
=
V_{C_j}(S_t\oplus e_j^{(1)})
-
V_{C_j}(S_t).
\]

最终叶节点编辑定义为：

\[
e_j=
\begin{cases}
e_j^{(0)},&
\Delta_j^{(0)}>\tau_{\mathrm{self}},\\[1mm]
e_j^{(1)},&
\Delta_j^{(0)}\leq\tau_{\mathrm{self}}
\land
\Delta_j^{(1)}>\tau_{\mathrm{self}},\\[1mm]
\varnothing,&
\text{otherwise}.
\end{cases}
\]

若：

\[
e_j=\varnothing,
\]

则该叶子簇不进入合并树。每个叶子簇最多允许一次自修正，以避免在训练簇上反复优化。

通过 self-check 的叶节点集合记为：

\[
\mathcal U_t
=
\{u_j=(e_j,P_j,C_j):e_j\neq\varnothing\}.
\]

这些节点构成后续合并树的叶子。若 \(\mathcal U_t=\varnothing\)，说明当前轮没有可靠叶节点，算法直接拒绝本轮更新并返回 \(S_{t+1}=S_t\)。

---

## 3.7 LLM 规划合并树

给定通过 self-check 的叶节点集合 \(\mathcal U_t\)，算法由 LLM Planner 构造一棵合并树 \(T_t\)。每个叶节点以简洁卡片形式输入：

```text
leaf_id: L_j
question_type: ...
revision_type: ...
edit summary: ...
support size: ...
self-check gain: ...
target: ...
```

Planner 的任务不是直接生成最终 Skill，而是规划叶节点之间的合并路径。具体而言，它需要决定：

1. 哪些叶节点可以合并；
2. 哪些叶节点应保持分离；
3. 每个合并步骤的合并意图是什么；
4. 每个内部节点应表达怎样的更抽象编辑；
5. 哪些合并会违反目标位置或适用边界，应被禁止。

Planner 输出一个有向树结构：

\[
T_t=(\mathcal V_t,\mathcal E_t),
\]

其中叶节点为 \(\mathcal U_t\)，内部节点表示由若干子节点合并得到的更抽象编辑。对任一内部节点 \(v\)，其编辑由子节点编辑合并得到：

\[
e_v
=
\operatorname{MergeNode}
\left(
\{e_c:c\in\operatorname{child}(v)\}
\right).
\]

每个内部节点同时保留一个合并说明：

\[
m_v=(\text{merge intent},\text{applicability},\text{boundary}).
\]

合并树的根节点 \(r\) 对应本轮最高层抽象候选编辑 \(e_r\)，从而得到 Root Candidate：

\[
\widetilde S_t^{\mathrm{root}}
=
S_t\oplus e_r.
\]

若通过 self-check 的叶节点只有一个，则该叶节点同时作为根节点，Root Candidate 直接由该叶节点编辑得到。为了控制复杂度，合并树受到浅层约束。每个内部节点最多合并 \(b\) 个子节点，树深度不超过 \(H_{\max}\)。若 Planner 判断某些叶节点语义冲突、目标位置不兼容或适用边界不一致，则这些节点可以保持分离，不被强行纳入高层合并。

---

## 3.8 验证集上的自顶向下剪枝

合并树构造完成后，算法在验证集上选择写回的抽象层级。首先评估 Root Candidate：

\[
\Delta_{\mathrm{root}}^{\mathrm{val}}
=
V_{\mathcal D_{\mathrm{val}}}
(\widetilde S_t^{\mathrm{root}})
-
V_{\mathcal D_{\mathrm{val}}}
(S_t).
\]

若：

\[
\Delta_{\mathrm{root}}^{\mathrm{val}}>\tau_{\mathrm{val}},
\]

则提交 Root Candidate：

\[
S_{t+1}=\widetilde S_t^{\mathrm{root}}.
\]

这表示最高层合并编辑在验证集上具有正向泛化效果。

若 Root Candidate 未通过验证集检查，则算法不递归搜索整棵树，而只回退到根节点的直接子节点：

\[
\operatorname{child}(r)=\{v_1,\ldots,v_m\}.
\]

若根节点没有直接子节点，则说明当前合并树没有可回退的子层候选，算法直接拒绝本轮更新：

\[
S_{t+1}=S_t.
\]

若根节点存在直接子节点，则对于每个直接子节点 \(v_j\)，构造候选 Skill：

\[
S_t\oplus e_{v_j},
\]

并在验证集上评估：

\[
\Delta_{v_j}^{\mathrm{val}}
=
V_{\mathcal D_{\mathrm{val}}}
(S_t\oplus e_{v_j})
-
V_{\mathcal D_{\mathrm{val}}}
(S_t).
\]

保留验证集收益为正的直接子节点：

\[
\mathcal C^+
=
\{v_j:\Delta_{v_j}^{\mathrm{val}}>\tau_{\mathrm{child}}\}.
\]

若 \(\mathcal C^+=\varnothing\)，则没有子节点候选通过验证集检查，算法拒绝本轮更新：

\[
S_{t+1}=S_t.
\]

否则，由保留节点组合得到子节点候选 Skill：

\[
\widetilde S_t^{\mathrm{child}}
=
S_t\oplus
\{e_{v_j}:v_j\in\mathcal C^+\}.
\]

再评估组合效果：

\[
\Delta_{\mathrm{child}}^{\mathrm{val}}
=
V_{\mathcal D_{\mathrm{val}}}
(\widetilde S_t^{\mathrm{child}})
-
V_{\mathcal D_{\mathrm{val}}}
(S_t).
\]

若：

\[
\Delta_{\mathrm{child}}^{\mathrm{val}}>\tau_{\mathrm{val}},
\]

则提交子节点组合：

\[
S_{t+1}=\widetilde S_t^{\mathrm{child}}.
\]

否则放弃本轮更新：

\[
S_{t+1}=S_t.
\]

该剪枝过程只允许从根节点回退到直接子节点，不继续展开到更深层叶节点。这样既保留了合并树提供的自然 fallback 结构，又避免在验证集上进行深层搜索。

---

## 3.9 完整算法

```text
Algorithm: Type-Guided Merge-Tree Skill Optimization

Input:
    current skill S_t
    training set D_train
    validation set D_val
    rollout number K
    success threshold tau_succ
    self-check threshold tau_self
    validation thresholds tau_val, tau_child
    target cluster size s_cluster, usually set to 7
    maximum merge branching factor b
    maximum tree depth H_max

1. Multi-sample patch record generation
    For each sample x_i in D_train:
        run K rollouts under S_t
        compute q_i^0
        if q_i^0 < tau_succ:
            generate record R_i = (question_type, revision_type, target, patch)

2. Target cluster number computation
    collect all failed or unstable sample records R_t
    compute batch size B_t = |R_t|
    if B_t = 0:
        return S_{t+1} = S_t
    set M_target = max(1, round(B_t / s_cluster))

3. Type- and size-aware leaf clustering
    aggregate identical type signatures if needed
    provide type signatures and their counts to the LLM
    ask the LLM to generate approximately M_target leaf clusters
        primary criterion: compatible question_type and revision_type
        secondary criterion: each cluster should contain about 6-8 samples
    assign each record R_i to one generated non-empty leaf cluster

4. Leaf cluster construction
    For each generated cluster C_j:
        merge sample patches in C_j into initial leaf edit e_j^(0)

5. Leaf self-check and one-step revision
    For each initial leaf edit e_j^(0):
        evaluate S_t ⊕ e_j^(0) on its supporting cluster C_j
        if self-check gain is greater than tau_self:
            keep e_j^(0) as leaf edit e_j
        else:
            revise e_j^(0) once using self-check comparison
            obtain e_j^(1)
            evaluate S_t ⊕ e_j^(1) on C_j
            if revised gain is greater than tau_self:
                keep e_j^(1) as leaf edit e_j
            else:
                discard the leaf cluster

6. Merge-tree planning
    collect all surviving leaf edits
    if no leaf edit survives:
        return S_{t+1} = S_t
    use all surviving leaf edits as leaves
    LLM Planner generates a bounded merge tree:
        decide which leaves or subtrees to merge
        decide which nodes remain separate
        generate an edit for each internal merge node
        produce the root candidate edit

7. Root validation
    construct root candidate skill S_root = S_t ⊕ e_root
    evaluate S_root on D_val
    if S_root improves over S_t by more than tau_val:
        return S_{t+1} = S_root

8. Root-to-children fallback
    if S_root fails:
        if the root has no direct children:
            return S_{t+1} = S_t
        evaluate each direct child of the root on D_val
        keep children with validation gain greater than tau_child
        if no child is kept:
            return S_{t+1} = S_t
        combine kept children into S_child
        evaluate S_child on D_val
        if S_child improves over S_t by more than tau_val:
            return S_{t+1} = S_child
        else:
            reject this round and return S_t

Output:
    updated skill S_{t+1}
```

---

## 3.10 计算开销

该方法的额外开销主要来自三部分：多采样执行、叶节点 self-check、以及验证集上的有限剪枝。

首先，Patch 记录只从训练集中失败或不稳定样本生成，稳定成功样本不进入合并树。其次，叶子簇归纳基于短类型签名 \((\alpha_i,\beta_i)\)，而不是长轨迹全文或两两相似度矩阵。若当前 batch 中失败或不稳定样本数为 \(B_t\)，目标簇大小为 \(s_{\mathrm{cluster}}\)，则目标簇数近似为：

\[
M_{\mathrm{target}}
\approx
\frac{B_t}{s_{\mathrm{cluster}}}.
\]

因此，随着失败样本数增长，叶子簇数量按 batch 大小线性调整，但每个簇的平均规模保持在约 6--8 个样本。该设计避免了过多小簇造成的碎片化，也避免了过少大簇造成的过度抽象。

最后，合并树剪枝只在验证集上评估根节点和根的直接子节点，不递归搜索整棵树。叶节点 self-check 只发生在其训练支持簇内部，并且最多允许一次自修正。因此，方法在引入显式合并树结构的同时，将额外搜索控制在浅层范围内。

---

## 3.11 方法性质

Type-Guided Merge-Tree Skill Optimization 的核心不是生成更多候选编辑，而是将局部失败证据组织为一棵可选择抽象层级的合并树。类型签名 \((\text{Question Type},\text{Revision Type})\) 为样本提供短而稳定的分类依据，使算法可以判断哪些样本属于同一类可合并修复，而不需要在聚类阶段读取冗长的失败轨迹。

其中，题目类型抽象题目的输入需求结构，修正类型抽象 Patch 的功能性修复意图。叶子簇原型沿用同样的二元结构，是对簇内样本类型的一步抽象。为了控制聚类粒度，算法根据当前 batch 中失败或不稳定样本数量自适应计算目标簇数，使每个簇大约包含 6--8 个样本。这样既避免 LLM 自由生成过多碎片化小簇，也避免将过多异质样本强行合并到同一个簇。

叶节点编辑由同一类型簇内的多个样本 Patch 合并得到，并通过簇内 self-check 验证其最低层可靠性。随后，LLM Planner 将可靠叶节点组织为多层合并树，验证集上的 Root-to-Children 剪枝决定最终写回最高层规则还是保守子规则。

该方法形成如下约束：

\[
\boxed{
\text{单样本 Patch 不直接写回，避免样本记忆；}
}
\]

\[
\boxed{
\text{样本通过题目类型和修正类型进行分类，避免长上下文聚类；}
}
\]

\[
\boxed{
\text{叶子簇数量由 batch 大小控制，使每簇约包含 6--8 个样本；}
}
\]

\[
\boxed{
\text{叶节点必须通过簇内 self-check，保证底层可靠性；}
}
\]

\[
\boxed{
\text{合并路径由 LLM 显式规划，而不是固定层级聚合；}
}
\]

\[
\boxed{
\text{验证集只用于 Root 和直接子节点选择，避免深层搜索；}
}
\]

\[
\boxed{
\text{测试集只用于最终报告，保持评估独立。}
}
\]

因此，本文方法将 Skill 更新过程表述为：

\[
\boxed{
\text{局部 Patch 证据}
\rightarrow
\text{题目类型与修正类型}
\rightarrow
\text{类型与数量感知的叶子簇}
\rightarrow
\text{可靠叶节点}
\rightarrow
\text{LLM 规划合并树}
\rightarrow
\text{验证集剪枝写回}.
}
\]
