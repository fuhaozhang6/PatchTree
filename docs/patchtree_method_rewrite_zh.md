# 3 方法

本节给出 PatchTree 的方法定义。与直接把样本级反馈总结成一条全局规则不同，PatchTree 将 Skill 更新拆解为两个连续的结构化选择问题：首先判断哪些样本级 Patch 可以共享同一条局部规则，然后判断这些局部规则应当停留在叶节点层级，还是可以继续合并为更抽象的高层规则。前者对应融合对象选择，后者对应融合尺度选择。

## 3.1 问题定义与方法概述

给定冻结执行模型 \(\pi\)、优化器模型 \(O\)、当前 Skill \(S_t\)、训练集 \(\mathcal D_{\mathrm{train}}\)、验证集 \(\mathcal D_{\mathrm{val}}\) 和任务奖励函数 \(r(x,\tau)\in[0,1]\)，我们的目标是在不更新执行模型参数的条件下，通过修改外部 Skill 文档得到下一轮 Skill：

\[
S_{t+1}=S_t\oplus e_t .
\]

其中 \(e_t\) 是本轮最终写回的结构化编辑，\(\oplus\) 表示将编辑应用到 Skill 文档。PatchTree 关心的不是如何生成单个样本的局部修复，而是如何从一组局部修复中选择合适的跨样本更新。因此，本轮优化可以写成：

\[
e_t
=
\operatorname{SelectLevel}
\left(
\operatorname{BuildTree}
\left(
\operatorname{ClusterPatch}(\mathcal R_t)
\right),
\mathcal D_{\mathrm{val}}
\right).
\]

这里 \(\mathcal R_t\) 是当前训练 batch 中失败或不稳定样本产生的 Patch 记录集合。`ClusterPatch` 负责把样本级 Patch 归纳为类型兼容的叶节点，`BuildTree` 负责构造从局部规则到高层规则的候选空间，`SelectLevel` 则利用验证集决定最终写回哪个抽象层级。

这个分解对应两个设计动机。第一，单样本 Patch 往往包含局部失败细节，直接写回容易造成样本记忆；但把所有 Patch 一次性总结为全局规则，又容易丢失适用边界。第二，高层规则虽然覆盖范围更大，却更容易引入负迁移；局部规则更保守，但可能无法消除重复和碎片化。PatchTree 因此先用类型签名约束“哪些 Patch 可以合并”，再用验证集选择“合并到什么程度”。

数据划分承担不同职责。训练集只用于产生轨迹、Patch 记录、叶节点与合并树候选；验证集只用于选择写回层级和接受或拒绝候选；测试集不参与任何优化或选择，仅用于最终报告。这样可以避免把训练样本上的可修复性误当成泛化收益。

## 3.2 样本级 Patch 记录

在第 \(t\) 轮优化中，对于训练样本 \(x_i\)，执行模型在当前 Skill \(S_t\) 下独立采样 \(K\) 条轨迹，并估计该样本的经验成功率：

\[
q_i(S_t)=\frac{1}{K}\sum_{k=1}^{K} r(x_i,\tau_{ik}),
\qquad
\tau_{ik}\sim \pi(\cdot\mid x_i,S_t).
\]

若 \(q_i(S_t)\ge \tau_{\mathrm{succ}}\)，说明当前 Skill 已能稳定处理该样本，本轮不为其生成修复。若 \(q_i(S_t)<\tau_{\mathrm{succ}}\)，优化器读取输入、重复轨迹、成功或失败片段以及验证器反馈，生成一个样本级 Patch 记录：

\[
R_i=(\alpha_i,\beta_i,\ell_i,p_i).
\]

其中，\(\alpha_i\) 是题目类型，描述输入要求 Agent 完成的结构性行为，例如显式约束遵循、多步依赖推理、证据支持回答、格式受控生成、比较与选择或工具调用决策。题目类型不是领域标签；它刻画的是任务对行为流程的要求。

\(\beta_i\) 是修正类型，描述该 Patch 试图补充或约束的功能性机制，例如约束核验、步骤分解、证据检查、格式强制、计算复核、歧义澄清或过度泛化控制。修正类型不是 `add`、`delete` 或 `replace` 这类表面编辑动作；它刻画的是失败需要怎样的行为修复。

\(\ell_i\) 是目标 Skill 位置，\(p_i\) 是可执行的最小编辑。目标位置和编辑内容用于后续融合时判断是否冲突，但不作为主要聚类键。用于判断融合兼容性的核心签名为：

\[
s_i=(\alpha_i,\beta_i).
\]

该签名必须短、稳定并去实例化，不能包含原题中的实体、数值、选项标签或标准答案。换言之，Patch 记录不是对失败轨迹的完整解释，而是把一次样本反馈压缩成一个可聚合的行为证据单元。

## 3.3 类型引导的叶节点构造

收集当前 batch 中所有失败或不稳定样本的 Patch 记录：

\[
\mathcal R_t=\{R_i:q_i(S_t)<\tau_{\mathrm{succ}}\}.
\]

若 \(\mathcal R_t\) 为空，本轮不产生候选更新。否则，PatchTree 将 \(\mathcal R_t\) 聚合为若干叶节点。叶节点是最低层跨样本更新单元：它比单样本 Patch 更抽象，因为它由多个记录共同支持；又比根节点更保守，因为它只覆盖一个局部修复方向。

聚类阶段遵循“类型兼容性优先、数量控制为辅”的原则。设期望簇大小为 \(s_{\mathrm{cluster}}\)，当前待聚合记录数为 \(B_t=|\mathcal R_t|\)，目标簇数为：

\[
M_{\mathrm{target}}
=
\max\left(1,\operatorname{round}\left(B_t/s_{\mathrm{cluster}}\right)\right).
\]

\(M_{\mathrm{target}}\) 只提供粒度先验，防止簇过碎或过宽；它不强制均匀划分，也不能迫使语义不兼容的记录合并。聚类器主要依据题目类型 \(\alpha_i\)、修正类型 \(\beta_i\)、目标位置 \(\ell_i\) 和 Patch 摘要判断记录是否描述同一类 type-compatible repair。每个叶簇 \(C_j\) 对应一个簇级类型原型：

\[
P_j=(A_j,B_j),
\]

其中 \(A_j\) 概括簇内共享的输入需求结构，\(B_j\) 概括簇内共享的修复机制。

对于每个叶簇，融合器将成员 Patch 合成为簇级编辑：

\[
e_j=\operatorname{MergeLeaf}\left(P_j,\{(\ell_i,p_i):R_i\in C_j\}\right).
\]

叶节点表示为：

\[
u_j=(e_j,P_j,C_j).
\]

簇级融合需要满足三个约束。第一，编辑只表达簇内记录共同支持的修复机制，不写入只在单个样本中出现的实体或答案细节。第二，编辑必须保留适用条件和不适用边界，避免把条件化规则改写为无条件规则。第三，当成员 Patch 指向不同 Skill 位置时，只有在行为目标和执行顺序兼容时才形成复合编辑；否则应保留为不同叶节点。

对于暂时缺少共同支持的 singleton 记录，PatchTree 不立即写回。它们进入 epoch 级长尾缓冲区，并在 epoch 结束时与其他 batch 的 singleton 重新聚合。若长尾记录在更大时间窗口中形成多样本簇，则按相同流程生成叶节点；若仍无共同支持，则从候选集合中移除。长尾机制的作用不是放宽证据要求，而是避免低频但重复的修复模式被 batch 边界切碎。

## 3.4 叶节点支持检查

叶节点来自训练样本，因此训练集上的检查不能作为泛化证据。但它可以回答一个更基础的问题：该叶节点是否至少修复了产生它的支持样本。如果一个簇级编辑连自己的支持簇都不能改善，它通常意味着聚类过宽、Patch 冲突、目标位置错误或编辑表达过于空泛。

对叶簇 \(C_j\)，记其支持样本集合为：

\[
X(C_j)=\{x_i:R_i\in C_j\}.
\]

定义叶节点在支持样本上的修复收益：

\[
\Delta^{\mathrm{sup}}_j
=
V_{X(C_j)}(S_t\oplus e_j)-V_{X(C_j)}(S_t).
\]

支持检查只用于刻画叶节点的本地可靠性。通过检查的叶节点可以获得更高的合并可信度；未通过的叶节点可以被记录、降权或在保守设置下过滤。它与验证集选择不同：支持检查回答“这个叶节点是否修复了自己的证据来源”，验证集选择回答“这个候选是否能泛化到未参与生成的样本”。这两个信号必须分开记录，不能用训练支持收益替代验证集收益。

最终进入合并树的叶节点集合记为 \(\mathcal U_t\)。若 \(\mathcal U_t=\varnothing\)，说明当前 batch 没有可靠的跨样本候选，本轮保持 \(S_{t+1}=S_t\)。

## 3.5 Patch 合并树

给定叶节点集合 \(\mathcal U_t\)，合并规划器构造一棵有界 Patch 合并树：

\[
T_t=(\mathcal V_t,\mathcal E_t).
\]

叶节点来自 \(\mathcal U_t\)，内部节点表示若干子节点进一步融合得到的更高层编辑。对任一内部节点 \(v\)，其编辑为：

\[
e_v=\operatorname{MergeNode}(\{e_c:c\in\operatorname{child}(v)\}).
\]

每个内部节点不仅保存摘要，还保存可以直接应用到 Skill 的实际编辑 \(e_v\)，以及该编辑的合并意图、适用条件和边界。这样，合并树中的每一层都是可验证的候选 Skill 更新：靠近叶节点的候选更具体、更保守；靠近根节点的候选覆盖更多叶节点，也承担更大的过度泛化风险。

根节点 \(r\) 对应本轮最大范围的融合候选：

\[
\widetilde S^{\mathrm{root}}_t=S_t\oplus e_r.
\]

规划器允许不兼容叶节点保持为不同分支，而不是强行合并为一条全局规则。为控制搜索规模，树的分支数和深度受到上界约束。实际系统可以采用两层树，即叶节点直接作为根的子节点；也可以在需要时增加中间层。无论树深如何，后续剪枝只使用根节点及其直接子节点，从而避免在验证集上进行深层搜索。

## 3.6 验证集剪枝与写回

验证阶段选择最终写回的抽象层级。首先比较根候选与当前 Skill 在验证集上的表现：

\[
\Delta^{\mathrm{val}}_{\mathrm{root}}
=
V_{\mathcal D_{\mathrm{val}}}(\widetilde S^{\mathrm{root}}_t)
-
V_{\mathcal D_{\mathrm{val}}}(S_t).
\]

若 \(\Delta^{\mathrm{val}}_{\mathrm{root}}>\tau_{\mathrm{val}}\)，说明最大范围融合在未参与生成的样本上产生正收益，算法接受根候选：

\[
S_{t+1}=\widetilde S^{\mathrm{root}}_t.
\]

若根候选未通过验证，算法回退到根的直接子节点 \(\operatorname{child}(r)\)。对每个直接子节点 \(v\)，构造候选 \(S_t\oplus e_v\)，并计算验证收益。保留收益超过阈值的子节点：

\[
\mathcal C^+
=
\{v\in\operatorname{child}(r):
V_{\mathcal D_{\mathrm{val}}}(S_t\oplus e_v)
-
V_{\mathcal D_{\mathrm{val}}}(S_t)
>
\tau_{\mathrm{child}}\}.
\]

若 \(\mathcal C^+=\varnothing\)，本轮拒绝更新。否则，将保留子节点作为条件化编辑共同写回，得到组合候选：

\[
\widetilde S^{\mathrm{child}}_t
=
S_t\oplus\{e_v:v\in\mathcal C^+\}.
\]

组合候选仍需再次通过整体验证：

\[
V_{\mathcal D_{\mathrm{val}}}(\widetilde S^{\mathrm{child}}_t)
-
V_{\mathcal D_{\mathrm{val}}}(S_t)
>
\tau_{\mathrm{val}}.
\]

只有当组合后的整体 Skill 仍然改善验证性能时，算法才接受子节点组合；否则保持原 Skill。该复验步骤用于避免多个单独有效的局部编辑共同写入后产生交互退化。

因此，验证集在 PatchTree 中承担两个角色。它不仅判断本轮是否应该更新 Skill，也决定更新应停留在根节点的高层抽象，还是回退到更保守的直接子节点组合。剪枝不递归展开整棵树，也不枚举任意 Patch 子集，从而在保留回退能力的同时控制验证成本。

## 3.7 完整算法

```text
Algorithm 1: PatchTree Skill Optimization

Input:
  current Skill S_t; executor model pi; optimizer O
  training split D_train; validation split D_val
  rollout count K; thresholds tau_succ, tau_val, tau_child
  target cluster size s_cluster; tree bounds b and H_max

1. R_t <- empty set
2. For each training sample x_i in the current batch:
3.     Run K rollouts under S_t and compute q_i(S_t)
4.     If q_i(S_t) < tau_succ:
5.         Generate PatchRecord R_i = (alpha_i, beta_i, ell_i, p_i)
6.         Add R_i to R_t
7. If R_t is empty, return S_t

8. Compute M_target = max(1, round(|R_t| / s_cluster))
9. Cluster R_t into type-compatible leaf clusters C_t
10. For each non-singleton cluster C_j:
11.     Merge member patches into leaf edit e_j
12.     Build leaf node u_j = (e_j, P_j, C_j)
13.     Optionally run support check on X(C_j)
14. Add singleton clusters to the epoch tail buffer
15. If no usable leaf node remains, return S_t

16. Build a bounded merge tree T_t from the leaf nodes
17. Evaluate the root candidate on D_val
18. If root gain > tau_val, accept the root candidate
19. Otherwise evaluate each direct child of the root on D_val
20. Keep children whose gain > tau_child
21. If no child is kept, return S_t
22. Combine kept children as condition-preserving edits
23. Accept the combination only if its joint validation gain > tau_val
24. Otherwise return S_t

At epoch end:
25. Re-cluster singleton tail records across batches
26. Build new leaves from multi-record tail clusters
27. Apply the same merge-tree and validation-pruning procedure
```

PatchTree 的核心约束可以概括为：样本级 Patch 不直接写回，避免样本记忆；题目类型和修正类型共同定义融合兼容性，避免仅按表面相似度聚类；叶节点保留适用条件和边界，避免局部修复被过度抽象；合并树显式保存多个抽象层级的可执行候选；验证集从根到直接子节点选择写回尺度，并用组合复验防止局部有效编辑之间产生负交互。
