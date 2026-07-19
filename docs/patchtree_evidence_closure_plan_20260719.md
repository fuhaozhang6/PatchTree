# PatchTree 树机制证据收束计划

日期：2026-07-19

> 状态：本 OfficeQA 版本已被
> `docs/searchqa_patchtree_evidence_closure_plan_20260719.md` 取代，保留本文件
> 仅作为历史设计记录。

## 1. 研究决策

从现在开始停止以下实验：

- 不再继续扫描采样次数、batch size、tree depth；
- 不再用不同 rollout 生成的 PatchRecords 比较 Flat 与 Tree；
- 不再仅凭不同进程中的绝对 TEST 分数判断结构优劣；
- 不再为了触发 fallback 人为增加 epoch 或随机 seed。

后续只回答一个问题：

> 在完全相同的失败证据和评估条件下，层级树编译是否比直接 Flat 编译产生更有
> 泛化性、更紧凑，或更有容错空间的 Skill 更新？

这是一次可证伪的机制裁决，不是新一轮参数搜索。

## 2. 当前已有结论

OfficeQA 的三次先导实验已经足以确认：

1. P1 的 29 个 PatchRecords 形成了 15 个 Leaf 和 7 个 Mid，确实发生了
   multi-Leaf 汇聚，树结构不是空实现；
2. Flat 在各自 TEST 中减少 2 个正确样本，两个 Tree 分别增加 1、2 个正确样本，
   方向上有弱正信号；
3. 三次 initial TEST 相差 8 个样本，且 PatchRecords 数量为 29、35、36，
   因此三组最终绝对分数不能作为严格的结构因果证据；
4. P2 的 Root 直接通过 gate，top-down fallback 没有触发，所以这一次实验没有
   验证 top-down 容错。

因此，现有实验不是“没有结果”，而是已经明确指出：下一步必须固定输入，而不是
继续扩大消融表。

## 3. 最终目标与验收标准

### 目标 A：树真实形成

从一个固定 step 的 artifact 中报告：

- 样本数、trajectory 数、PatchRecord 数；
- Leaf、Mid、Root 数；
- 每个 Mid 包含的 Leaf 数；
- singleton Mid 比例；
- 每层覆盖的 record/sample provenance；
- 一条完整的 Record → Leaf → Mid → Root 典型路径。

最低成立条件：

- 至少存在 2 个由多个 Leaf 合并而成的 Mid；
- Root 覆盖不少于 90% 的入树 PatchRecords；
- 不把大量未进入 Root 的记录误报为“被抽象”。

该目标只读取 artifact，不使用 GPU，不重新训练。

### 目标 B：上层节点确有抽象作用

对同一条典型路径检查：

- `shared_core` 是否来自两个或以上子节点的共同修正原则；
- `conditional_residuals` 是否保留子节点间不同的适用条件和边界；
- Mid 是否减少重复规则，而不是简单拼接；
- Root 是否继续保留关键边界，而不是只留下空泛表述；
- 从子节点到上层节点的 edit/token 压缩率和 provenance 保留率。

最低成立条件：

- 上层节点相对其子节点总文本至少压缩 30%；
- 每个被审计的 `shared_core` 可追溯到至少 2 个子节点；
- 主要差异条件进入 `conditional_residuals` 或等价边界描述；
- 不出现“压缩很多但丢失主要修正条件”的情况。

这一步主要是 artifact 审计，也不重新训练。

### 目标 C：树结构具有实际效用

只做一次 fixed-input paired replay：

1. 固定使用 P2 的同一份 `type_guided_v2_patch_records.json`；
2. 固定使用 P2 step 进入前的 `skills/skill_v0000.md`；
3. 直接复用 P2 已保存的 `Tree(depth=3)` artifact、ranked edits 和 candidate，
   避免重新生成 Tree；
4. 只从同一份 PatchRecords 新编译一次 `Flat(depth=1)`；
5. 两个候选使用同一个 vLLM 进程、同一 OfficeQA TEST、temperature=0、
   thinking=false、max completion tokens=4096，各评估一次；
6. 保存逐样本结果，进行 paired comparison，而不仅比较平均分。

当前 DeepSeek/OpenAI optimizer backend 没有显式传递 temperature，因此计划不把
“optimizer temperature=0”写成无法保证的控制条件。复用原 Tree、只新建 Flat，
比重新随机生成两棵候选更稳妥。

判定标准：

| 结果 | 判定 |
|---|---|
| Tree 比 Flat 多正确至少 5/172，且 hard 不净退化超过 2 个 | 树的泛化效用成立 |
| Tree 与 Flat 相差不超过 2 个，但 edits/token 减少至少 30% | 树的压缩效用成立，性能按非劣处理 |
| Tree 的 Root 不优于 Flat，但某个 Mid/child 在共享 val 上比 Root 多正确至少 2/24，或 children 并集新增至少 3/24 | 层级候选有容错价值，保留 top-down |
| Tree 比 Flat 少正确至少 3 个，且没有 child rescue、没有显著压缩优势 | 停止把树作为性能贡献 |
| 差值落在 -2 到 +4，且没有压缩或 child coverage 证据 | 证据不足，但不再追加随机消融 |

“多 5 个样本”是预先设定的实际效应阈值，不在看到结果后修改。
child rescue 的 2/24 与 coverage 的 3/24 同样是预先设定的阈值。

## 4. 唯一一次剩余实验

### 4.1 固定证据来源

优先使用：

```text
/ai-app-vepfs/zhangfuhao/skill/PatchTree/outputs/
officeqa_tree_mechanism_p2_full_tree_top_down_seed42_20260719_143349/
officeqa/steps/step_0001/
```

需要取回：

```text
config.json
skills/skill_v0000.md
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

并取回该 run 根目录的：

```text
best_skill.md
```

### 4.2 固定输入重放

重放不是重新训练：

- 不跑 train rollout；
- 不重新生成 PatchRecord；
- 不改变数据、采样次数、batch size 或 seed；
- 复用 P2 已有的 Tree artifact、ranked edits 和 candidate；
- 只调用聚合器把同一份 records 新编译成一次 Flat；
- Flat 若超过 `edit_budget=4`，沿用原训练的 `rank_and_select`，这属于同一条
  Flat 编译链，不另做采样或候选搜索。

输出必须包含：

```text
replay_manifest.json
flat_merge_artifact.json
flat_merged_patch.json
flat_ranked_edits.json
flat_candidate_skill.md
flat_apply_report.json
tree_merge_artifact.json
tree_ranked_edits.json
tree_candidate_skill.md
artifact_metrics.json
```

### 4.3 一次统一评估

在一个 H20 或 L20 上只启动一个 vLLM，统一评估：

1. initial skill；
2. fixed-input Flat；
3. fixed-input Tree。

统一设置：

```text
target = Qwen/Qwen3.5-4B
temperature = 0
thinking = false
max completion tokens = 4096
test = OfficeQA full 172
same docs / same offline tools / same workers
```

三个 skill 必须使用不同的新 `out_root`，防止 `results.jsonl` 的 resume 逻辑复用
其他 skill 的旧结果。评估顺序不是实验变量，最终按样本 ID 做配对比较。

### 4.4 Top-down 不再单独重跑

top-down 的证据分两部分取得：

1. 从已有 LiveMath、SearchQA、OfficeQA 日志中离线统计
   `root fail → child pass` 的真实次数；
2. 在 fixed-input Tree 中，让 Root 和直接 child 在同一组 val24 上各评估一次，
   仅作为诊断，不重新训练、不写回 Skill。

报告：

- Root 单独通过数；
- 每个 child 的增益；
- Root 错但 child 对的样本数；
- Root 与 children 的正确样本并集；
- 是否真实存在可被 top-down 保留的有用局部 patch。

如果既有历史日志和本次诊断都没有 child rescue，则删除或弱化
“top-down 提高容错率”的主张，不再专门制造 fallback 实验。

## 5. 资源与成本上限

| 工作 | 资源 | 上限 |
|---|---|---|
| artifact 下载与结构统计 | CPU | 不调用模型 |
| 典型路径审计 | CPU/人工 | 1 个典型 Mid，必要时再看 1 个反例 |
| fixed-input 编译 | DeepSeek optimizer | 只新增 1 条 Flat 编译链，Tree 直接复用 |
| val 节点诊断 | 1 张 H20/L20 | 只跑 val24 |
| initial/Flat/Tree TEST | 同一张 H20/L20、同一 vLLM | 每个 skill 只跑一次 |

总预算固定为一张卡的一次任务；不再开启四卡并行消融，不再追加 seed。

## 6. 最终交付

实验结束只产出一份裁决报告，包含：

1. 结构计数表；
2. 一条 Record → Leaf → Mid → Root 的完整内容对照；
3. Flat 与 Tree 的同输入编译差异；
4. TEST 逐样本配对表；
5. Root/child coverage 表；
6. 按预注册阈值得出的 Green / Yellow / Red 判定；
7. 对论文主张的保留、弱化或删除建议。

不再输出新的超参数排行榜。

## 7. 无论结果如何都能收束

- **Green**：树在准确率上超过 Flat，保留“层级抽象提升泛化”的核心主张；
- **Yellow**：准确率非劣但明显更紧凑，改写为“层级压缩与候选组织”，
  top-down 只在有 rescue 证据时保留；
- **Red**：树明显落后且没有 rescue，停止把树包装成性能创新，将论文主线收缩为
  PatchRecord、类型化修正和受验证更新。

这三个结果都可以形成确定结论。计划的目标不是保证树胜出，而是用一次严格、
可解释的比较决定它应当处于论文的什么位置。
