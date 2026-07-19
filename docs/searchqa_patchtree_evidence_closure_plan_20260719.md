# SearchQA PatchTree 树机制裁决计划

日期：2026-07-19

状态：取代 OfficeQA 版本的裁决计划。后续不再使用 OfficeQA 做本轮树机制实验。

## 1. 总目标

本轮不再搜索最优超参数，只裁决三个机制问题：

1. **Leaf 是否有用**：相同 PatchRecords 下，先按可合并类型形成 Leaf，是否比
   PatchRecords 直接进入 Root 更有效或更紧凑？
2. **Mid 是否有用**：相同 Leaf 下，多 Leaf 形成 Mid，是否比 Leaf 直接进入
   Root 具有更好的抽象、压缩或泛化？
3. **Top-down 是否有用**：当 Root 自然失败时，直接 child 中是否存在能够被
   找回、并最终通过完整验证的有效局部更新？

这三个问题分别对应：

```text
Flat:       PatchRecord → Root
Leaf-only:  PatchRecord → Leaf → Root
Full tree:  PatchRecord → Leaf → Mid → Root
Top-down:   Root reject → shared-subset child evaluation → reconcile → full val
```

所有结构性能比较必须共享同一份 PatchRecords、同一个 parent skill、同一组
val/test 样本和同一个 target 进程。

实验组总表：

| 组 | 唯一用途 | 核心验收 |
|---|---|---|
| G0 Parent | 共同未更新基线 | Skill、records、评估样本和请求参数 hash 完全一致 |
| G1 Flat | 无树对照 | 真实 `depth=1`，Leaf/Mid 均为0，candidate 正常生成 |
| G2 Leaf-only | 验证局部类型聚合 | 相对 Flat 显著提高≥14/1400，或非劣且压缩≥20% |
| G3 Full tree | 验证 Mid 抽象与主树贡献 | 相对 Leaf/Flat 显著提高≥14/1400，或非劣且层级压缩≥30% |
| G4 Top-down | 验证 Root 失败后的 child rescue | 自然 Root reject，至少2个互补 child，reconcile 通过 full val 且 TEST 优于 Root |
| G5 Artifact audit | 验证树是否普遍真实形成 | ≥50%有效 step 出现 multi-Leaf Mid，provenance 完整 |

## 2. 为什么选择 SearchQA P6

固定证据源：

```text
/ai-app-vepfs/zhangfuhao/skill/PatchTree/outputs/
searchqa_tree_shape_d3_clustering_on_min_support1_all_clusters_tail_off_fallback_off_seed42_20260718_165434/
searchqa/
```

该运行使用：

| 参数 | 值 |
|---|---:|
| train / val / test | 400 / 200 / 1400 |
| batch size | 40 |
| steps | 10 |
| rollout repeats | 3 |
| min support | 1 |
| max leaf groups | 40 |
| clustering | true |
| cluster target / max | 2 / 4 |
| tree depth | 3 |
| fallback | false |
| target | Qwen/Qwen3.5-4B |

已有结果：

```text
initial TEST hard = 0.6779
best TEST hard    = 0.7093
delta             = +0.0314，约 +44/1400
```

这个端到端增益只能说明 P6 产生了有效 Skill，不能单独归因于树结构，但它使 P6
成为比低分 OfficeQA 更合适的固定证据源。

P6 中还有两个重要的自然步骤：

| Step | 结构与结果 | 用途 |
|---:|---|---|
| 8 | 14 records → 7 leaves → Mid → Root；candidate 被接受并成为 best | Flat / Leaf / Tree 主比较候选 |
| 6 | 18 records → 13 leaves → 3 mids；Root mixed 0.7393，低于 parent 0.7521，被自然拒绝 | Top-down rescue 裁决 |

Step 6 的 `mid workers=3`，而 worker 上限为4，因此其 Mid 数就是3；13个 Leaf
汇成3个 Mid，已经确认不是 singleton Mid 退化。

## 3. 实验前的固定规则

### 3.1 Accepted step 的选择

主比较不根据 TEST 结果挑 step。下载 P6 全部10个 step artifact 后，在
`accept_new_best` 的 step 中按以下顺序选择：

1. 至少有2个 multi-Leaf Mid；
2. 进入 multi-Leaf Mid 的 Leaf 覆盖率最高；
3. 若并列，选择更早的 step。

Step 8 是当前优先候选；如果 artifact 证明它仍以 singleton Mid 为主，则按上述
规则在已接受的 step 1、2、4、8 中重新选择，而不是看到 TEST 后换 step。

### 3.2 Root-reject step 的选择

Top-down 组只从原运行中已经自然 rejected 的 step 选择：

1. 至少有2个 multi-Leaf Mid；
2. `leaf_count / mid_count` 最大；
3. 若并列，选择更早的 step。

根据现有日志，Step 6 是优先候选。

### 3.3 统一评估

```text
target model              = Qwen/Qwen3.5-4B
target temperature        = 0
thinking                  = false
max completion tokens     = 4096
val                       = full 200
test                      = full 1400
same vLLM process         = true
same workers              = true
separate output directory = required for every skill
```

所有 comparison 保存逐样本结果，并按 sample ID 做 paired comparison。平均分
只作为汇总，不作为唯一证据。

统一统计规则：

- TEST 主指标为 hard exact match，soft 作为辅助描述；
- 报告 `n01 = A对/B错`、`n10 = A错/B对` 和净增益 `n01-n10`；
- 主比较使用双侧 exact McNemar test；
- 同时给出 paired bootstrap 95% confidence interval；
- 1个百分点即14/1400，作为最小实际效应；
- 非劣必须同时满足：观察值少正确不超过7/1400，且单侧95%置信下界
  大于 `-1.5` 个百分点。

## 4. 共享对照组 G0：Parent Skill

### 输入

如果选用 Step 8：

```text
parent skill = skills/skill_v0007.md
evidence     = steps/step_0008/type_guided_v2_patch_records.json
```

如果选用其他 step `N`，parent 固定为 `skills/skill_v(N-1).md`。

### 目标

提供所有候选共同的未更新基线，排除不同训练阶段 Skill 的影响。

### 验收标准

G0 不承担“提升”要求，但必须满足：

- parent skill SHA256 在所有组中完全一致；
- PatchRecords JSON SHA256 在 G1/G2/G3 中完全一致；
- val/test item ID 和顺序集合完全一致；
- 不复用其他 skill 的 `results.jsonl`；
- target 请求参数完全一致。

任意一项不满足，则 G1/G2/G3 整组无效，不解释得分。

## 5. 实验组 G1：Flat，验证直接合并基线

### 唯一结构

```text
PatchRecords → Root，tree_depth=1
```

不构建 Leaf 和 Mid。沿用原 step 的 edit budget、ranker 和 patch apply 规则。

### 验证目标

建立“没有层级抽象”时，同一批失败证据能够产生的最好直接更新，作为 G2/G3 的
因果对照。

### 记录指标

- records 数；
- Root 输出 edits 数；
- rank 前后 edits 数；
- applied/skipped/error edits；
- 新增 Skill 字符数；
- val200 hard/soft/mixed；
- test1400 hard/soft；
- 相对 G0 的逐样本 `fixed / broken` 数。

### 验收标准

G1 是对照而不是待证明的方法，因此不要求一定超过 G0。其有效性要求：

- 输入 hash 与 G2/G3 一致；
- `tree_depth=1` artifact 中 Leaf/Mid 数均为0；
- Root 的 provenance 能追溯到输入 records；
- candidate 可以被 patch apply 正常生成。

若 Flat 自身生成失败，本轮不能据此宣称 Tree 更强，必须先修复编译错误。

## 6. 实验组 G2：Leaf-only，验证局部类型聚合

### 唯一结构变化

```text
PatchRecords → Leaf → Root，tree_depth=2
```

与 G1 相比，只增加 Leaf 分组与 Leaf 内共享/条件细节编译，不构建 Mid。

G2 不重新生成 clusters 或 Leaf。它直接读取 G3 原 artifact 中冻结的
`clustering`、`kept_groups` 和 `leaf_patches`，再以这些 Leaf 作为 children
新编译一个 Leaf Root。这样 G2 与 G3 之间的唯一结构差异才是 Mid。

### 验证目标

验证“先把相近问题形成局部修正规则”是否比所有 records 直接进入 Root 更好。

### 结构前提

- 至少50%的 PatchRecords 进入 support≥2 的 Leaf；
- dropped record 比例不超过10%；
- Leaf provenance 无重复分配、无未知 record ID。

不满足结构前提时，G2 记为“Leaf 未形成”，不能解释准确率。

### 成功标准

满足以下任一条即可认为 Leaf 有价值：

1. **准确率成功**：
   - G2 比 G1 在 test1400 多正确至少14题（+1.0个百分点）；
   - paired McNemar exact test `p < 0.05`；
   - G2 相对 G0 满足非劣：观察值少正确不超过7题，单侧CI下界
     大于 `-1.5` 个百分点。
2. **非劣压缩成功**：
   - G2 相对 G1 少正确不超过7题（0.5个百分点非劣界）；
   - 单侧 paired 95% 置信下界大于 `-1.5` 个百分点；
   - 从 records 总文本到 Root 输入/输出的有效字符或 token 至少减少20%；
   - 主要 condition/boundary provenance 保留率不少于90%。

### 失败标准

- G2 比 G1 少正确至少14题且 McNemar `p < 0.05`；或
- Leaf 大量丢记录；或
- 既没有准确率收益，也没有达到非劣与压缩标准。

未达到显著胜/负且没有压缩证据时记为“不成立/证据不足”，不再追加 seed。

## 7. 实验组 G3：Full Tree，验证 Mid 抽象

### 唯一结构变化

```text
PatchRecords → Leaf → Mid → Root，tree_depth=3
```

G3 与 G2 使用相同 records、相同 parent，以及物理上完全相同的
`clustering + kept_groups + leaf_patches`；唯一新增机制是多 Leaf 到 Mid 的
抽象。不能让 G2/G3 各自重新调用 Leaf merger。

优先复用 P6 原 step 已保存的 Tree artifact、ranked edits 和 candidate，不重新
生成 Tree。只新编译 G1 Flat 和 G2 Leaf-only，避免第二次随机生成 Tree。

### 验证目标

验证 Mid 是否：

1. 提取多个 Leaf 的共同修正原则；
2. 用 `conditional_residuals` 保留差异条件；
3. 在不损害泛化的前提下减少 Root 面对的重复信息。

### 结构与抽象前提

- 至少2个 Mid 各自包含不少于2个 Leaf；
- 至少50%的 Leaf 进入 multi-Leaf Mid；
- singleton Mid 比例不高于50%；
- Root 覆盖不少于90%的入树 records；
- 至少80%的 multi-Leaf Mid，其 `shared_core` 可追溯到2个以上 children；
- 至少80%的主要差异条件进入 `conditional_residuals`、boundary 或等价字段；
- `source_child_ids` 不得包含越界或不存在的 child。

### 成功标准

满足以下任一条即可认为 Mid 有价值：

1. **准确率成功**：
   - G3 比 G2 在 test1400 多正确至少14题；
   - 与 G2 vs G1 一起做 Holm 校正后，McNemar `p < 0.05`；
   - G3 相对 G0 满足非劣。
2. **非劣抽象成功**：
   - G3 相对 G2 少正确不超过7题；
   - 单侧 paired 95% 置信下界大于 `-1.5` 个百分点；
   - Leaf 总内容到 Mid 总内容至少压缩30%；
   - G2 Root 输入到 G3 Root 输入至少压缩30%；
   - provenance/condition 保留满足上述结构前提。

### 主结论：树相对 Flat

除 G3 vs G2 外，同时报告 G3 vs G1：

| 结果 | 裁决 |
|---|---|
| G3 比 G1 多正确≥14，McNemar `p<0.05`，且 G3 对 G0 满足非劣 | Green：树的泛化贡献成立 |
| G3 相对 G1/G2 均少正确≤7、单侧CI下界>-1.5pp，且层级压缩≥30%、provenance 保留≥90% | Yellow：树的抽象/压缩贡献成立，性能按非劣表述 |
| G3 比 G1 少正确≥14且`p<0.05`，或 G3 比 G0 少正确≥14且`p<0.05` | Red：树有显著伤害，不能作为性能贡献 |
| 差值处于灰区且无压缩证据 | Gray：只保留结构案例，不再声称性能收益 |

安全检查优先于组间胜负：如果 G3 相对 G0 显著少正确至少14题，即使 G1 更差，
也不能宣称 Full Tree 成功。

若 G3 的最终 Root patch provenance 没有覆盖任何 non-singleton Mid，则即使
G3 得分提高，也不能把提高归因于 Mid。

## 8. 实验组 G4：Top-down，验证 Root 失败后的局部恢复

### 固定证据

优先使用 P6 Step 6：

```text
parent skill = skills/skill_v0005.md
records      = steps/step_0006/type_guided_v2_patch_records.json
tree         = steps/step_0006/type_guided_v2_merge_artifact.json
root         = steps/step_0006/candidate_skill.md
```

原运行已经自然观察到：

```text
parent val mixed = 0.7521
root val mixed   = 0.7393
result           = Root rejected
13 leaves → 3 mids
```

不人为破坏 Root，也不为了触发 fallback 重跑训练。

### 唯一机制变化

1. Root 仍先经过 full val200；
2. Root reject 后，3个直接 Mid children 使用完全相同的固定 subset40；
3. 按现有 child gate 选择通过者；
4. 使用 deterministic reconcile；
5. current、Root、入选 children 和 reconciled candidate 在 full val200 上按
   同一协议复核，其中 reconciled candidate 接受与否仍由正常 gate 决定；
6. 只有 full val 通过，才运行一次 full test1400。

subset40 通过 sample ID 的确定性 hash 从 val200 产生，并将 ID 写入 manifest。
Root、所有 children 和 current skill 在 subset40 上使用完全相同的题目。

### 验证目标

验证层级结构是否把被 Root 过度抽象或错误合并的局部有效 patch 保留下来，并让
它们重新进入更新候选视野。

### 成功标准

必须依次满足：

1. temperature=0 的固定协议下，Root 在 full val200 上仍不超过 parent；如果
   Root 不再失败，只能记录 descendant potential，不能算 end-to-end top-down；
2. 至少2个 Mid children 在 subset40 上各自相对 parent 净增益大于0；
3. 两个入选 children 各自至少有1个对方没有覆盖的独有修复样本；
4. reconciled candidate 在 full val200 上超过 parent，能够被正常 gate 接受；
5. 在 test1400 上，reconciled candidate 比 rejected Root 多正确至少14题，
   且 McNemar `p < 0.05`；
6. reconciled candidate 相对 parent 满足非劣：少正确不超过7题，且单侧
   paired 95% 置信下界大于 `-1.5` 个百分点。

全部满足时，才能声称“top-down 提高了 Root 失败时的容错率”。

### 部分结果与失败标准

| 观察 | 结论 |
|---|---|
| 只有1个 child 通过，或children没有互补修复 | 存在局部候选，但不足以证明树形协调 |
| child subset 通过，但 reconcile/full val 不通过 | 存在局部多样性，但当前 top-down 选择机制无效 |
| full val 通过，但 TEST 不优于 Root且不满足非劣 | val 过拟合，top-down 性能主张不成立 |
| 固定协议下 Root 不再 reject | 只能报告 rescue potential，不能声称实际 fallback 成功 |
| 没有2个互补 child | 没有充分 rescue 证据，删除 top-down 容错主张 |
| 只有单 Leaf Mid 或 provenance 错误 | 结构前提失败，本组无效 |

G4 失败后不再更换 subset、seed、top-k 或阈值重试。

## 9. Artifact 结构审计组 G5

G5 不调用 target，不是额外训练实验。它使用 P6 全部10个 step 的现有 artifact。

### 验证目标

证明一次完整 SearchQA 执行中树究竟形成了多少次、以什么形状形成，而不是只展示
一个最好案例。

### 输出

- 每 step 的 samples、trajectories、PatchRecords；
- clusters、kept/dropped groups；
- Leaf、multi-Leaf Mid、singleton Mid、Root 数；
- 每个 Mid 的 children 数；
- record/leaf/mid/root provenance 覆盖；
- accepted/rejected 与树形的对应关系；
- 一条 accepted step 的典型路径；
- 一条 rejected step 的反例路径；
- shared core、conditional residual、boundary、unresolved conflict；
- Record → Leaf → Mid → Root 的字符/token 压缩率。

### 验收标准

若要声称“SearchQA 中树机制被普遍激活”，必须满足：

- 至少50%的有效 step 出现 multi-Leaf Mid；
- 全部 step 合计至少50%的 Leaf 进入 multi-Leaf Mid；
- Root provenance 覆盖率平均不少于90%；
- 未知/越界 child ID 为0；
- 至少抽查2个 multi-Leaf Mid，shared core 和 residual 均能追溯到 children。

如果只在少数 step 形成真实树，则论文必须表述为“条件性触发的层级聚合”，不能
写成每次更新都会形成有效抽象树。

## 10. 资源和执行顺序

### 阶段 A：CPU 审计

下载 P6 的：

```text
config.json
history.json
skills/
steps/step_0001 ... step_0010/
```

每个 step 至少保留：

```text
type_guided_v2_patch_records.json
type_guided_v2_clustering.json
type_guided_v2_leaf_clusters.json
type_guided_v2_mid_nodes.json
type_guided_v2_root.json
type_guided_v2_merge_artifact.json
merged_patch.json
ranked_edits.json
candidate_skill.md
step_record.json
```

先完成 G5，并按预注册规则确定主 accepted step 和 root-reject step。若结构前提
失败，直接停止相应组，不浪费 GPU。

### 阶段 B：编译

对选定 accepted step：

- G1 Flat 新编译一次；
- G2 复用 G3 已冻结的 clusters/Leaf，只新编译一次 Leaf Root；
- G3 Tree 直接复用；
- 记录输入、代码、prompt、输出 SHA256；
- 沿用该 step 实际 edit budget，不能统一写死为4。以 Step 8 为例，实际
  select budget 是2。

当前 DeepSeek/OpenAI optimizer backend 没有显式 temperature 参数，因此不能
把“optimizer temperature=0”写成已满足条件。控制办法是冻结已有 Tree 和 Leaf，
只新增 Flat Root 与 Leaf Root 两条必要编译链，并完整保存请求/响应 artifact。

### 阶段 C：一张 GPU 统一评估

同一个 vLLM 进程依次评估：

```text
G0 parent
G1 Flat
G2 Leaf-only
G3 Full tree
```

共 `4 × 200` 条 val 和 `4 × 1400` 条 test rollout。所有 skill 使用独立
`out_root`，最后做逐样本配对统计。

然后执行 G4：

- 先做 `3 children × subset40`；
- 再按同一协议复核 current/Root/selected children/reconciled 的 full val200；
- 只有 full val 接受后才追加 Root/Top-down 的 test1400 对比。

执行前比较 Step 6 `skill_v0005.md` 与主实验 parent 的 SHA256。若二者相同，
G4 复用 G0 的 parent TEST；若不同，必须把 G4 parent 作为独立 skill 评估，不能
借用 G0 分数。最坏情况下统一 TEST 总量为6个 skill × 1400，仍只启动一个
vLLM 进程。

全程只使用一张 H20 或 L20，不重新跑 train400，不重新生成 PatchRecords，不追加
seed。

## 11. 最终停止规则

实验结束后只允许以下四种结论：

- **Green**：Full Tree 在相同证据下显著优于 Flat/Leaf，保留树为性能创新；
- **Yellow**：性能非劣但抽象/压缩显著，树定位为结构化候选编译与压缩机制；
- **Red**：Tree 落后且 top-down 无 rescue，树退出性能主线；
- **Gray**：只证明局部案例，论文仅做定性机制展示，不再追加随机消融。

无论得到哪一种结果，本轮 SearchQA 树机制实验到此结束。
