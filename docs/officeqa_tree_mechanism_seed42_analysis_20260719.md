# OfficeQA 一次更新树机制实验分析

日期：2026-07-19

## 1. 原始日志

| ID | 结构 | 日志 |
|---|---|---|
| P0 | PatchRecord → Root | `logs/officeqa_tree_mechanism_p0_flat_records_to_root_seed42_20260719_143311/officeqa.log` |
| P1 | Cluster/Leaf → Mid → Root，fallback off | `logs/officeqa_tree_mechanism_p1_bottom_up_tree_seed42_20260719_143327/officeqa.log` |
| P2 | Cluster/Leaf → Mid → Root，fallback on | `logs/officeqa_tree_mechanism_p2_full_tree_top_down_seed42_20260719_143349/officeqa.log` |

三组均完整结束，使用 OfficeQA train50、val24、test172，一次 step、三次
rollout repeat。

## 2. 最重要的有效性问题

虽然三组使用相同 seed42 和相同 initial skill，但 initial skill 的重复评估结果
明显不同：

| Run | initial val | initial TEST | TEST correct |
|---|---:|---:|---:|
| P0 | 0.4167 | 0.2384 | 41/172 |
| P1 | 0.2500 | 0.1919 | 33/172 |
| P2 | 0.3750 | 0.2035 | 35/172 |
| 极差 | 0.1667 | 0.0465 | 8 samples |

这说明 seed42 只固定了数据顺序，并没有固定 Qwen/tool rollout。三组之间的
initial TEST 波动 4.65 个百分点，而三个最终 Skill 的最大差距只有 2.90 个
百分点。因此不能直接把三个最终绝对分数解释为结构差异。

三组训练 rollout 和 PatchRecord 也不是同一份输入：

| Run | repeat hard | stable | candidates | PatchRecords | no-patch |
|---|---|---:|---:|---:|---:|
| P0 | 0.24 / 0.20 / 0.20 | 8 | 42 | 36 | 6 |
| P1 | 0.22 / 0.20 / 0.30 | 9 | 41 | 29 | 12 |
| P2 | 0.26 / 0.26 / 0.32 | 9 | 41 | 35 | 6 |

PatchRecord 数量在 29–36 之间波动。P1/P2 即使 tree depth 相同，也不是同一棵
树、同一批 Leaf 或同一个 Root 输入。因此本轮属于端到端随机先导，尚不是严格
的固定输入结构消融。

## 3. Val 和 TEST 原始结果

| Run | val before | val after | Δval | TEST before | TEST after | ΔTEST |
|---|---:|---:|---:|---:|---:|---:|
| P0 Flat | 10/24 = 0.4167 | 12/24 = 0.5000 | +2 samples | 41/172 = 0.2384 | 39/172 = 0.2267 | -2 samples |
| P1 Bottom-up | 6/24 = 0.2500 | 9/24 = 0.3750 | +3 samples | 33/172 = 0.1919 | 34/172 = 0.1977 | +1 sample |
| P2 Full tree | 9/24 = 0.3750 | 11/24 = 0.4583 | +2 samples | 35/172 = 0.2035 | 37/172 = 0.2151 | +2 samples |

三个候选都被 val gate 接受，但只有树结构的 P1/P2 在各自 TEST 中得到正增量；
Flat 在 TEST 中净损失两个样本。这是方向一致的弱正信号，但增量只有 1–2 个
样本，仍小于当前重复评估噪声，不能作为显著结论。

按难度拆分：

| Run | easy before → after | Δeasy | hard before → after | Δhard |
|---|---:|---:|---:|---:|
| P0 | 28/79 → 24/79 | -4 | 13/93 → 15/93 | +2 |
| P1 | 20/79 → 25/79 | +5 | 13/93 → 9/93 | -4 |
| P2 | 24/79 → 26/79 | +2 | 11/93 → 11/93 | 0 |

P1 的三个 Root edits 明显偏向 easy，并在 hard 上发生回退；P2 的单一 Root edit
在该次评估中增加两个 easy 样本且没有净伤害 hard。若确定性复评后仍保持这一
形态，P2 会是“更高层抽象减少过拟合”的有价值案例。

## 4. 结构计数

| Run | samples | trajectories | records/edits | Leaf groups | Mid nodes | Root edits | selected edits | Skill length |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P0 | 50 | 150 | 36 | 0 | 0 | 11 | 4 | 887 → 3135 |
| P1 | 50 | 150 | 29 | 15 | 7 | 3 | 3 | 887 → 3823 |
| P2 | 50 | 150 | 35 | 26 | ≥8 | 1 | 1 | 887 → 2734 |

P1 中 15 个 Leaf 被规划成 7 个 Mid，因此这次 OfficeQA 实验确实发生了
multi-Leaf 汇聚，而不是 SearchQA P5 中的全 singleton 退化。P1 的：

- Record → Leaf ratio：15/29 = 0.517；
- Mid → Leaf ratio：7/15 = 0.467；
- Root 将 7 个 Mid 编译成 3 个 edits。

P2 日志只打印 `mid merges parallel workers=8`。由于 worker 上限也是 8，只能
证明 Mid 数不少于 8；要得到精确 Mid 数、每个 Mid 的 Leaf 成员和 shared core，
必须读取远端 `type_guided_v2_mid_nodes.json`。

P2 从 35 个记录、26 个 Leaf、至少 8 个 Mid 最终压缩成 1 个 Root edit。单从
结构上看，它形成了非常强的抽象/压缩；但只有拿到节点 artifact，才能判断这一
edit 是真正抽象还是信息丢失。

## 5. Top-down 没有被验证

P2 的 Root 在 val24 上从 0.3750 提高到 0.4583并直接通过 gate。因此：

- fallback 没有触发；
- 没有 Mid child evaluation；
- 没有 root-fail/child-pass；
- 没有 descendant reconciliation；
- P2 的 +2 TEST samples 只能归因于该次 bottom-up Root candidate，不能归因于
  top-down 容错。

P1 和 P2 的结果差异也不能用来间接证明 fallback，因为两者使用了不同 rollout、
PatchRecords、Leaf 分组和 Root。

## 6. 低分的主要来源

### 6.1 OfficeQA + Qwen3.5-4B 的初始能力较低

三次 initial TEST 均值约为 0.211。hard 部分只有约 9.7%–16.1%，而 TEST 中
hard 占 93/172。OfficeQA 需要长文档检索、表格定位、时期对齐和数值计算，当前
4B target 的主要瓶颈首先是任务执行能力，不是树结构。

### 6.2 vLLM 存在真实的上下文长度失败

三组 `vllm_qwen.log` 中都出现：

```text
max context = 65536
requested output = 16384
input >= 49153
input + output >= 65537
```

| Run | context-length validation errors | non-2xx requests |
|---|---:|---:|
| P0 | 3 | 3 |
| P1 | 6 | 6 |
| P2 | 4 | 4 |

当前为每个请求预留 16384 个输出 token，对 OfficeQA 的长工具轨迹过大。少数长
prompt 被服务端直接拒绝，经过重试后仍可能变成错误样本。这会同时降低分数并
增加跨运行随机性。后续建议把 target max completion tokens 降到 4096，至少
不应继续使用 16384。

vLLM 多次达到约 98%–99% KV cache 使用率并出现少量 waiting，但没有观察到
OOM；这主要影响吞吐，不是当前分数差异的首要解释。

### 6.3 val24 太小且 gate 噪声较大

val 中一个样本对应 4.17 个百分点。三个候选都因为改善 2–3 个 val 样本被接受，
但 P0 在 TEST 上反而下降。这表明当前一次随机 val rollout 很容易把推理波动或
局部过拟合当成真实提升。

TEST 中一个样本对应约 0.58 个百分点。本轮 +1/+2/-2 个样本的变化不足以超过
initial skill 重复评估的 8-sample 极差。

## 7. 时间与成本

| Run | aggregate | step wall | total wall | total tokens |
|---|---:|---:|---:|---:|
| P0 | 28.6s | 4542.1s | 11917s | 75.58M |
| P1 | 123.9s | 4426.7s | 12324s | 73.97M |
| P2 | 138.0s | 4765.3s | 13449s | 72.56M |

树聚合相对 Flat 多约 95–109 秒，但相对三到四小时的完整运行很小。主要成本来自
OfficeQA tool rollout，以及 initial/best 两次完整 TEST；不是 Mid 聚合本身。

## 8. 当前可以和不可以下的结论

可以确认：

1. OfficeQA P1 真实形成了 multi-Leaf Mid，树机制至少被激活；
2. Flat 在该次 TEST 中下降2个样本，两个树候选分别增加1、2个样本；
3. P2 产生了高度压缩的单 edit Root，并在该次评估中没有净伤害 hard；
4. 树聚合的额外时间不是主要成本。

不能确认：

1. P2 的绝对分数高于 P1 是因为树或 fallback；
2. top-down 提高了容错率——本轮没有触发 fallback；
3. Mid 的 shared core 具有更高抽象性——缺少 output artifact；
4. +1/+2 TEST 样本超过了随机推理噪声。

## 9. 建议的最小下一步

不建议立刻重跑更多 epoch。先做两件更便宜且更有判别力的事：

1. 下载三个远端 output，至少包括：
   - `steps/step_0001/type_guided_v2_patch_records.json`
   - `type_guided_v2_clustering.json`
   - `type_guided_v2_leaf_clusters.json`
   - `type_guided_v2_mid_nodes.json`
   - `type_guided_v2_root.json`
   - `type_guided_v2_merge_artifact.json`
   - `merged_patch.json`
   - `ranked_edits.json`
   - `candidate_skill.md`
   - `best_skill.md`
2. 在同一个 vLLM 进程中，以 temperature=0、max completion tokens=4096，
   重新评估 initial、P0、P1、P2 四个 Skill 的完整 test172。这样可以先判断
   现有三个候选的真实排序，而不必重新训练。

随后用 P2 的同一批 PatchRecords 固定输入重放 Flat 与 Tree，并对 Leaf/Mid/Root
在同一 val24 上做节点级评估，才是验证抽象性的关键实验。
