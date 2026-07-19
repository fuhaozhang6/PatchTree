# OfficeQA PatchTree 结构消融实验方案

日期：2026-07-19

## 1. 实验目标

本实验不只比较最终 TEST 分数，而是依次回答三个问题：

1. 一次训练运行中，样本如何逐层变成 PatchRecord、语义簇、Leaf、Mid 和
   Root？
2. 一个具有代表性的 step 中，各层究竟保留、抽象或丢失了什么修正机制？
3. 最终收益来自“多一层生成”“节点压缩”，还是来自正确的语义分组与
   `shared_core + conditional_residuals`？

OfficeQA 本地划分为：

| Split | 总数 | Easy | Hard |
|---|---:|---:|---:|
| train | 50 | 23 | 27 |
| val | 24 | 11 | 13 |
| test | 172 | 79 | 93 |

已有 observed taxonomy 表明 OfficeQA 同时包含高频共享机制和条件差异：

| 主要修正机制 | 已观察数量 |
|---|---:|
| operand verification | 80 |
| source/period alignment | 32 |
| calculation verification | 21 |
| evidence localization | 17 |
| answer format control | 17 |

因此它适合观察“多个具体修正能否被抽象成共享机制”。但 train 和 val 都较小，
最终结论必须结合多 seed，而不能只依据单次最高分。

## 2. 初次机制实验：一个 epoch、一次更新

初次实验的目标是证明树机制至少能在一个真实更新中产生可观测价值，而不是要求
第一次就形成形态完美的树。因此不设置 multi-Leaf coverage、压缩率或
shared-core 激活率等准入门槛，也不因为树较简陋而丢弃该 step。

推荐把 OfficeQA 完整 train50 放入一个 batch：

| 参数 | 初次实验值 |
|---|---:|
| epochs | 1 |
| batch size | 50 |
| accumulation | 1 |
| optimization steps | 1 |
| rollout repeats | 3 |
| max PatchRecords | 50 |
| min support | 1 |
| max Leaf groups | 50（不截断） |
| clustering | on，target/max=2/4 |
| tail bank | off |
| selection | 完整 val24 |
| final test | 完整 test172 |
| seed | 42 |

一次大 batch 有三个优点：

1. 50 个训练样本提供尽可能多的候选修正，最容易形成哪怕很粗糙的层级；
2. 只有一次更新，所有对照都从完全相同的 initial skill 出发；
3. 可以缓存一次 rollout/PatchRecords，然后重放不同聚合结构，避免多次 rollout
   的随机差异被误认为树的效果。

初次实验只比较三个机制：

| ID | 机制 | 要回答的问题 |
|---|---|---|
| P0 Flat | PatchRecords 直接形成 Root 更新，不构造 Leaf/Mid 树 | 没有树时一次更新能做到什么？ |
| P1 Bottom-up | PatchRecords→Cluster→Leaf→Mid→Root，只使用 Root gate | 上层抽象是否比 Flat/Leaf 更能泛化？ |
| P2 Full tree | 与 P1 复用完全相同的树，再加入 Root→Mid→Leaf 的自顶向下选择与重组 | 当 Root 过度合并或失败时，树能否恢复局部有效 patch？ |

P1 和 P2 必须复用同一个 Tree artifact，不能重新调用 planner 生成另一棵树。
P0/P1/P2 也必须复用相同的 initial skill、50 个样本、150 条 rollout trajectory
和 PatchRecords。初次实验不加入 random tree、强制 2–3 Leaf、tail bank 或多
seed；这些属于确认实验，而不是第一步。

为了避免一次 Root 恰好通过、导致观察不到 top-down 行为，初次实验应当无论
Root gate 是否通过，都以 diagnostic 模式评估所有 Mid 和 Leaf；实际 Full-tree
策略仍然只有在 Root 失败时才使用后代重组。

### 2.1 初次实验的成功证据

不要求所有条件同时满足，也不预设树必须很规整。分别寻找三类直接证据：

1. **树相对 Flat**
   - P1 或 P2 在 held-out val24 / test172 上优于 P0；
   - 或在相同编辑预算下，树候选覆盖更多能改善验证样本的 PatchRecord。
2. **上层抽象相对叶节点**
   - 至少存在一个 Mid 覆盖多个 Leaf；
   - 它的文本确实提取了更通用机制并保留条件差异；
   - 它在 val24 上的改善样本集合或分数优于单个子 Leaf，并接近多个子 Leaf
     改善集合的并集。
3. **自顶向下容错**
   - Root 失败但至少一个 Mid/Leaf 通过，记为 recovery opportunity；
   - 后代重组候选最终通过完整 val24，记为 recovery success；
   - 记录被 Root 丢失、但被有效后代重新带入候选的 PatchRecord 数。

若一个 step 没有发生 Root failure，只能说明该 step 没有触发 recovery，不能
据此否定 top-down。此时再补一个 seed 或把 train50 分成两个 batch25，增加触发
机会即可。

## 3. 后续正式训练的统一配置

只有在初次机制实验观察到正向信号后，才进入端到端、多 step 和多 seed 的正式
训练。所有正式结构消融保持以下参数完全相同：

| 参数 | 建议值 | 原因 |
|---|---:|---|
| train / val / test | 50 / 24 / 172 | 使用完整官方划分 |
| batch size | 25 | 每个 step 有足够 PatchRecords 构树；每 epoch 正好 2 step |
| epochs | 8 | 共 16 step，可统计结构分布而不过度增加运行数量 |
| rollout repeats | 3 | 保留稳定成功、波动和失败判定 |
| max PatchRecords | 25 | 不截断单 step 的有效样本修正 |
| min support | 2 | 避免以单样本噪声作为主要 Leaf |
| max Leaf groups | 12 | support≥2 时理论最多 12 组，因而基本不产生额外截断 |
| clustering | on | 使用语义机制簇形成 Leaf 输入 |
| cluster target / max | 2 / 4 | 保留足够细粒度的 OfficeQA 修正簇 |
| fallback | off | 避免 child fallback 干扰树结构比较 |
| tail bank | off | 避免 epoch 级长尾更新干扰主实验 |
| val gate | 完整 val24 | 所有候选使用相同 selection 集 |
| test | 完整 test172 | 每次训练默认测试 best skill |
| seed | 42/43/44 | 最终确认使用三个 seed |

优化器、目标模型、模型温度、数据顺序、worker 数、Skill 初始版本和 prompt
版本都必须固定。主指标使用 TEST hard score，同时保留 soft、mixed gate score、
训练耗时和 token 数。

## 4. 实验一：运行级结构统计

### 4.1 每个 step 必须统计的原始量

| 层级 | 指标 |
|---|---|
| Sample | unique train samples、rollout trajectories、stable success、unstable、failure |
| PatchRecord | PatchRecord 数、PatchRecord/sample 产出率、question/revision type 分布 |
| Cluster | 原始 cluster 数、cluster size 分布、未分配和无效记录数 |
| Leaf | kept/dropped group 数、Leaf 数、Leaf support 分布、非空 shared core 比例 |
| Mid | Mid 数、multi-Leaf Mid 数、singleton/passthrough Mid 数、每个 Mid 的 Leaf 数 |
| Root | root child 数、root edits 数、候选是否通过 gate、val 分数变化 |

推荐逐 step 输出以下表格：

| step | samples | trajectories | records | clusters | kept leaves | dropped | mids | multi mids | singleton mids | root edits | action | Δval |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|

### 4.2 运行级派生指标

必须从上述原始计数计算：

- PatchRecord yield = `PatchRecords / unique samples`
- Leaf retention = `kept Leaf groups / clusters`
- Leaf compression = `Leaf nodes / PatchRecords`
- Mid compression = `Mid nodes / Leaf nodes`
- Multi-Leaf Mid rate = `multi-Leaf Mids / all Mids`
- Multi-Leaf coverage = `被 multi-Leaf Mid 覆盖的 Leaf / all Leaves`
- Singleton rate = `singleton or passthrough Mids / all Mids`
- Shared-core activation = `shared_core 非空的 multi-Leaf Mids / multi-Leaf Mids`
- Gate acceptance rate = `accepted root candidates / all non-empty candidates`

下列结构健康条件只用于后续正式确认，不作为初次机制实验的限制。它们用于防止
多 seed 实验再次“名义 D3、实际 singleton D3”：

- 至少一半有 `Leaf>=4` 的 step 产生 multi-Leaf Mid；
- 总体 multi-Leaf coverage 不低于 30%；
- semantic 模式的 `Mid/Leaf` 中位数小于 0.8；
- 所有 singleton 必须标记为 `passthrough`，不能计作抽象节点。

初次实验即使不满足也保留并完整分析；正式多 seed 运行若不满足，则不能把它
解释为对理想 PatchTree 的否定。

### 4.3 数据来源

当前代码已经保存主要原始 artifact：

```text
steps/step_XXXX/step_record.json
steps/step_XXXX/type_guided_v2_patch_records.json
steps/step_XXXX/type_guided_v2_rollouts.json
steps/step_XXXX/type_guided_v2_clustering.json
steps/step_XXXX/type_guided_v2_leaf_clusters.json
steps/step_XXXX/type_guided_v2_mid_nodes.json
steps/step_XXXX/type_guided_v2_root.json
steps/step_XXXX/type_guided_v2_merge_artifact.json
steps/step_XXXX/merged_patch.json
```

后续统计程序应直接读取 JSON，而不是从控制台日志正则解析。

## 5. 实验二：典型 step 的逐层抽象分析

### 5.1 典型 step 的选择规则

不能根据“分数最好”主观挑选。先限定：

1. `Leaf count >= 4`；
2. 至少存在一个 multi-Leaf Mid；
3. PatchRecords、Leaf、Mid 数均非异常极值。

在符合条件的 step 中，选择其结构向量
`[PatchRecords, clusters, Leaves, Mids, multi-Leaf coverage]` 与全运行中位数
标准化距离最小者。性能分数不参与选择。

如果整个运行没有符合条件的 step，报告应明确写成“本次运行没有产生可分析的
典型抽象 step”，而不是选择 singleton step 代替。

### 5.2 从样本到 Root 的追踪表

为所选 step 生成完整映射：

| Sample ID | rollout outcome | PatchRecord ID | cluster ID | Leaf ID | Mid ID | Root edit |
|---|---|---|---|---|---|---|

逐层展示：

1. **Sample → PatchRecord**
   - 原问题、失败轨迹、诊断；
   - `question_type`、`revision_type`、`repair_signature`；
   - 具体 patch 和适用条件。
2. **PatchRecord → Cluster**
   - cluster 成员；
   - 为什么认为它们共享修正机制；
   - 是否混入语义不同的记录。
3. **Cluster → Leaf**
   - Leaf `shared_core`；
   - Leaf `conditional_residuals`；
   - 哪些 Record 细节被保留或丢失。
4. **Leaf → Mid**
   - Mid planner 选择了哪些 Leaf；
   - Mid `shared_core` 相对每个 Leaf 新增了什么抽象；
   - 条件差异是否完整进入 residuals；
   - 是否只是同义改写或拼接。
5. **Mid → Root**
   - Root 保留了哪些 Mid 机制；
   - 哪些规则被合并、改写或删除；
   - Root 的适用边界是否比子节点更宽。

### 5.3 抽象质量人工审计

对每个 multi-Leaf Mid 按下面 rubric 标注 `yes / partial / no`：

| 维度 | 判断问题 |
|---|---|
| Faithfulness | shared core 是否被每个子 Leaf 支持？ |
| Generalization | 是否比任何单个 Leaf 更一般，而不是换一种说法？ |
| Conditionality | 子 Leaf 的关键差异是否进入 residuals？ |
| Boundary safety | 是否保留“不应应用”的边界？ |
| Non-redundancy | Mid 是否减少重复内容？ |
| Root retention | Root 是否真正保留了该 Mid 的抽象？ |

同时计算文本以外的行为指标。在同一个 val24 上分别评估：

- current skill；
- 每个 Leaf patch 单独应用后的 skill；
- 每个 Mid patch 单独应用后的 skill；
- Root candidate。

记录每个节点的 `Δhard / Δsoft / Δmixed`，以及在 24 个 val 样本上：

- 改善样本集合；
- 伤害样本集合；
- Mid 相对各 Leaf 是否扩大有效覆盖；
- Root 是否保留子节点收益。

这个 counterfactual node evaluation 是判断“抽象是否真的有用”的核心证据。

## 6. 实验三：树结构消融

### 6.1 第一阶段：固定输入的机制级消融

选定同一个 step，冻结：

- step 开始前的 current skill；
- 相同 PatchRecords；
- 相同 clusters；
- 相同 Leaf patches；
- 相同 val24。

只改变 Mid 和 Root 的构造：

| ID | 结构 | 目的 |
|---|---|---|
| A Flat | `Leaf → Root` | 无 Mid 的主基线 |
| B Singleton | `每个 Leaf → singleton Mid → Root` | 测量单纯多一层生成的影响 |
| C Shuffled | 保留 D 的 Mid 数和 group size，但确定性打乱 Leaf 成员 | 控制树形、压缩率和生成调用，破坏语义选择 |
| D Semantic | 语义 planner 分组，`shared_core + conditional_residuals` | 完整方法 |
| E Residual-only（可选） | 沿用 D 的语义分组，但禁用 shared core，只保留条件规则 | 单独验证抽象 core 的贡献 |

C 必须使用与 D 相同的 Mid 数量和 group-size multiset，只打乱成员，不能任意生成
另一棵大小不同的随机树。随机打乱使用固定 seed，并建议重复 5 次，报告均值和
范围。

每个候选都在同一个 val24 上评估。固定输入实验可以直接回答：

- B vs A：额外抽象调用本身是收益还是噪声？
- C vs A：单纯分层和压缩是否有效？
- D vs C：语义节点选择是否优于随机选择？
- D vs E：shared core 是否提供了超越条件规则拼接的收益？

### 6.2 第二阶段：端到端训练消融

端到端主实验建议先运行四组：

| ID | Mid mode | 其他设置 |
|---|---|---|
| A | off，depth=2 | Flat Leaf→Root |
| B | singleton | 每个 Leaf 独立经过 Mid |
| C | semantic-shuffled | 与语义树相同 shape，打乱成员 |
| D | semantic constrained | 完整语义树 |

D 中应使用硬结构约束：

- multi-Leaf Mid 必须包含 2–3 个 Leaf；
- multi-Leaf Mid 的 shared core 必须非空；
- 无法安全合并的 Leaf 标记为 passthrough；
- planner 全部输出 singleton 时，重试一次；仍无兼容组则自适应退化为 Flat；
- passthrough 不计作抽象 Mid。

当前代码只原生支持 A（depth=2）和未硬约束的 semantic depth=3。为了得到可解释
的 B/C/D，需要先新增显式实验参数，例如：

```text
type_guided_mid_plan_mode:
  off | singleton | semantic_shuffled | semantic_constrained
```

以及可选的：

```text
type_guided_mid_compile_mode:
  shared_conditional | residual_only
```

在这些模式实现前，不建议直接把当前 depth=3 当作 D，因为它可能再次退化成
singleton tree。

## 7. Seed、资源和运行顺序

### 7.1 结构先导

先执行第 2 节的一次更新机制实验：

- 1 张 GPU；
- 1 epoch、batch50、共 1 step；
- 复用一次 rollout 和 PatchRecords 得到 P0/P1/P2；
- 分析实际形成的树，不设置形态准入门槛；
- 完整执行节点级 val24 counterfactual 和三个最终候选的 test172。

只要观察到树、上层抽象或 top-down recovery 中任一明确正向信号，就可以围绕该
信号进入正式实验。没有触发 Root failure 时补一个 seed 或改成 batch25 两个
step，而不是立刻扩大全部消融。

### 7.2 四卡第一轮

四张卡分别运行 A/B/C/D，统一 seed42、8 epochs。完成后：

1. 汇总完整结构统计；
2. 选择典型 step；
3. 做固定输入 counterfactual；
4. 检查是否值得扩展多 seed。

### 7.3 多 seed 确认

正式结论至少使用 seed42/43/44：

- 4 个结构 × 3 seeds = 12 runs；
- 有 4 张 GPU 时分 3 轮完成；
- 如果资源受限，先保留 A/C/D，得到 9 runs；
- 报告 mean、standard deviation 和每个 seed 原始值，不能只报最好 seed。

TEST172 每个样本对应约 0.58 个百分点，因此小于 1 个百分点的单次差异不应被
解读为稳定优势。

## 8. 最终报告结构

最终实验记录至少包含：

1. 数据、模型、prompt 版本、代码版本和完整启动参数；
2. 每个 run 的原始日志和 output 路径；
3. 每 step 的 Sample→Record→Cluster→Leaf→Mid→Root 计数；
4. 结构派生指标及其跨 step 分布；
5. 典型 step 完整映射和节点文本；
6. Leaf/Mid/Root counterfactual val24 结果；
7. A/B/C/D 每个 seed 的 best-val、TEST、耗时和 token；
8. 失败运行、无 multi-Leaf 运行和中断运行，不得从汇总中静默删除。

只有当 D 在结构健康检查中确实形成 multi-Leaf abstraction 后，D vs A/C 的最终
性能对比才有资格被解释为对 PatchTree 创新点的验证。
