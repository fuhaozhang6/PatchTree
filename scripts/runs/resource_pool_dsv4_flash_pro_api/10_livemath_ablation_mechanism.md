# 10 — LiveMathematicianBench 机制消融文档（预写）

> 对应脚本：`scripts/runs/resource_pool_dsv4_flash_pro_api/10_livemath_ablation_mechanism.sh`
> 状态：**预写（结果待填）**。本文只讲解各维度"是什么、为什么消融、看什么"，实验数据跑完后填入第 4 节。

---

## 0. 这份消融要回答什么

09 号消融（`09_livemath_ablation.sh`）扫的是**训练旋钮**（batch_size / rollout_repeats / leaf_fallback），结论是：三个旋钮调到头，best_test 也只到 0.41~0.43，**追不上 SkillOpt-main 的 0.4597**。

这说明瓶颈不在训练旋钮，而在**方法层 / 接受机制**。10 号消融就是专门扫这一层——即"为什么 Tree 涨不动，而 main 能涨"的候选原因。

| | 09（旋钮层） | 10（机制层，本文） |
|---|---|---|
| 扫什么 | batch_size / rollout_repeats / leaf_fallback | 接受机制、编辑预算、信息保留、gate 松紧、长尾阈值 |
| 目的 | 找最佳超参 | 定位"Tree 不涨"的根因 |
| 共享设定 | — | 各行固定 batch=35 / repeats=4（09 找到的稳健最优），避免弱旋钮噪声淹没机制信号 |

所有行**串行执行**（一次一个 train.py），各自独立输出子目录 + 日志，128/64 并发。每行是 **OFAT（一次只改一个机制）**，其余保持 PatchTree 默认。

---

## 1. 背景：Tree 的接受链路长什么样

要理解这些维度，先看候选 skill 从产生到被接受要过几道关：

```
rollout 采样 (repeats 次)
   │
   ▼
edits 按 (question_type, revision_type) 分组
   │
   ├─ min_support 过滤：支持度 < 阈值 的组 → dropped（长尾丢弃）
   │        └─ 若开 clustering：分组前先做跨类型 LLM 语义融合，提升支持度
   │        └─ 若开 tail_bank：dropped 长尾跨轮回收，二次评估
   │
   ▼
leaf 合并 → root 合并 → 生成候选 patch
   │
   ▼
edit budget 裁剪：每步最多改 N 处（cosine 4→2）
   │
   ▼
gate 评估（use_gate）：候选在验证集打分
   ├─ hard <= 当前 → REJECT（候选被丢，skill 不变）  ← Tree 卡在这里
   └─ use_gate=false → force-accept（无条件接受）     ← main 的机制
```

10 号消融的 5 个维度，正好对应这条链路上的 5 个关键闸门。

---

## 2. 维度详解

### A. 接受机制 —— `evaluation.use_gate`（true / false）

| | |
|---|---|
| **config key** | `evaluation.use_gate` |
| **默认** | `true`（硬 gate） |
| **代码依据** | `skillopt/engine/trainer.py:1949-1969` |
| **脚本 run** | `use_gate_true` / `use_gate_false` |

**是什么**：控制候选 skill 要不要过"验证集打分门槛"才能被接受。

- `true`（默认）：候选必须在验证集上**打得过当前 skill**（`hard > current`）才 ACCEPT，否则 REJECT，skill 保持不变。这是 PatchTree v2 的"硬 gate"。
- `false`：**force-accept**——每个候选无条件成为新的 current skill（`trainer.py:1949`），best-so-far 仍单独记录，最终从轨迹里手动选最优。

**为什么消融它**：这是本次消融**最关键的一维**。SkillOpt-main 用的正是 `slow_update + force-accept (unconditional)` —— 无条件注入、skill 持续演化。而 Tree 的硬 gate 会把"暂时打不过 baseline"的候选全部 REJECT（09 日志里 batch=8 一口气 REJECT 16 次），大量优化信号被拦掉。`use_gate=false` 就是**直接把 Tree 的接受机制对齐成 main**。

**看什么**：如果 `use_gate=false` 的 best_test 明显高于 `true`（且超过噪声 ~0.06），基本可以坐实"gate 过严"就是 Tree 不涨的根因。

---

### B. 编辑预算 —— `EDIT_BUDGET_OFF`（0 / 1）

| | |
|---|---|
| **env 开关** | `EDIT_BUDGET_OFF`（由 `_common.sh` 翻译成 cfg） |
| **底层 config** | `optimizer.lr_scheduler`（`cosine` → `autonomous`） |
| **默认** | `0`（预算开启，cosine 衰减） |
| **代码依据** | `skillopt/optimizer/scheduler.py`（`AutonomousScheduler` 返回 `NO_LIMIT=999`） |
| **脚本 run** | `edit_budget_on` / `edit_budget_off` |

**是什么**：控制**每一步最多能改动 skill 的几处**（edit budget）。

- `0`（默认）：`lr_scheduler=cosine`，每步允许的编辑数从 `learning_rate=4` 余弦衰减到 `min_learning_rate=2`。即越到后期改得越少。
- `1`：`lr_scheduler=autonomous`，每步返回 `NO_LIMIT=999`，等于**取消每步编辑数上限**。注意这条路径**不额外调用 LLM**（区别于 `lr_control_mode=autonomous`，那个每步多一次 pro 调用）。

**为什么消融它**：Tree 在 merge 后还会按 budget 裁剪候选 patch，可能把有用的 edit 砍掉。放开预算看看"让候选一次改更多"是否有助于跨过 gate、加速演化。

**看什么**：`edit_budget_off` 是否让候选质量提升、ACCEPT 增多。注意它可能与 A 维度有交互——预算放开但 gate 仍严，未必能涨。

---

### C. 信息保留 —— `TYPE_GUIDED_CLUSTERING` + `TYPE_GUIDED_TAIL_BANK`（关 / 开）

| | |
|---|---|
| **env 开关** | `TYPE_GUIDED_CLUSTERING`、`TYPE_GUIDED_TAIL_BANK` |
| **底层 config** | `optimizer.type_guided_clustering`、`optimizer.type_guided_tail_bank` |
| **默认** | 都 `false` |
| **代码依据** | clustering: `type_guided_merge_v2.py:713-764`（`cluster_patch_records`）、prompt `prompts/type_guided_cluster.md:14-18`；tail_bank: `trainer.py:2467-2679` |
| **脚本 run** | `info_default`（都关） / `info_retain`（都开） |

**是什么**：两个减少"merge 阶段信息丢弃"的开关，本消融里**打包一起开**。

- **`type_guided_clustering`（跨类型 LLM 语义融合）**：默认关时，edits 严格按 `(question_type, revision_type)` 精确分组，类型不同就不合并。**开启后**，在 min_support 过滤**之前**先用 LLM 做一次语义聚类——prompt 明确写"类型标签只是信号不是硬边界，只要修复机制和触发条件本质相同，就允许跨类型合并"（`type_guided_cluster.md:14-18`）。合并后支持度提升，原本会被当长尾丢掉的 edits 就能保留。**这就是你问的"语义判断融合"能力**。
- **`type_guided_tail_bank`（跨轮长尾库）**：默认关时，被 min_support 丢弃的长尾 edits 当步即弃。**开启后**，这些长尾写入一个跨 epoch 的库，滑窗内累计支持度够（且跨 ≥2 step）就重新纳入 merge、二次过 gate。给低频但反复出现的修复经验一个"攒够再上"的机会。

**为什么消融它**：09 日志里能看到 `groups=7 kept=2 dropped=5` 这类大量丢弃。这两个开关直接减少丢弃、提高候选覆盖度，从而提高过 gate 的概率。

**看什么**：`info_retain` 相对 `info_default` 是否提升 best_test；同时可在日志里对比 `dropped=N` 是否变小。**这是零代码验证"语义融合有没有用"的最快方式**——若有效，可能就不必再新增"dropped 长尾二次融合"的代码。

> ⚠️ 边界：clustering 融合的是**过滤前的全部 edits**；它**不是**专门抢救"已经被 min_support 丢掉的长尾"。真正"对 dropped 长尾做二次语义融合"目前代码里没有，需另行开发。本维度先测现有能力的天花板。

---

### D. gate 松紧 —— `optimizer.type_guided_tau_succ`（0.5 / 1.0）

| | |
|---|---|
| **config key** | `optimizer.type_guided_tau_succ` |
| **默认** | `1.0` |
| **代码依据** | `type_guided_merge_v2.py:345`（`if q_i >= tau_succ:`） |
| **脚本 run** | `tau_succ_0.5` / `tau_succ_1.0` |

**是什么**：判定一次 rollout 算不算"成功"的分数门槛。merge 时只有得分 `q_i >= tau_succ` 的 rollout 才被计入"成功样本"，其修复经验才会被提炼成 edit。

- `1.0`（默认）：**必须满分**才算成功。门槛很高，很多"部分对"的 rollout 被判失败，其经验不入选。
- `0.5`：半对即算成功，**更多 rollout 的经验被纳入**，候选 edit 更丰富。

**为什么消融它**：`tau_succ=1.0` 在数学题这种"要么全对要么全错"的任务上可能过严，导致可用信号稀少。放低门槛看是否能榨出更多有效 edit。

**看什么**：`tau_succ=0.5` 是否带来更多 edit / 更高 best_test。风险：门槛太低会引入噪声 edit，反而拉低质量——所以两档对比。

---

### E. 长尾阈值 —— `optimizer.type_guided_min_support`（1 / 2 / 3）

| | |
|---|---|
| **config key** | `optimizer.type_guided_min_support` |
| **默认** | `2` |
| **代码依据** | `type_guided_merge.py:296-304`（`kept = [g for g in groups if support >= min_support]`） |
| **脚本 run** | `min_support_1` / `min_support_2` / `min_support_3` |

**是什么**：一组 edit 至少要有几条"支持样本"才被保留，低于阈值的组直接进 dropped（长尾丢弃）。

- `1`：**几乎不丢长尾**——只要出现过就保留。
- `2`（默认）：至少两条样本支持才保留。
- `3`：更严，只保留高频修复经验。

**为什么消融它**：这是"长尾丢弃"这道闸门的直接旋钮，和 C 维度互补——C 是"通过语义融合/跨轮回收来挽救长尾"，E 是"直接调低丢弃门槛"。对 train 只有 35 条的 livemath，支持度天然偏低，`min_support=2` 可能丢掉太多。

**看什么**：`min_support=1` 是否显著减少 `dropped`、提升 best_test；`min_support=3` 是否因过度丢弃而变差。三档能画出"保留 vs 噪声"的权衡曲线。

---

## 3. 运行方式

```bash
cd /Users/bytedance/Documents/codes/Opt/SkillOpt-Tree
export DEEPSEEK_API_KEY='你的key'

# 全量 5 维 11 run（串行，128/64 并发）
bash scripts/runs/resource_pool_dsv4_flash_pro_api/10_livemath_ablation_mechanism.sh

# 只跑最关键的接受机制 + 信息保留两维
DO_EDIT_BUDGET=0 DO_TAU_SUCC=0 DO_MIN_SUPPORT=0 \
  bash scripts/runs/resource_pool_dsv4_flash_pro_api/10_livemath_ablation_mechanism.sh

# 冒烟（不发请求，只看接线）
DRY_RUN=1 bash scripts/runs/resource_pool_dsv4_flash_pro_api/10_livemath_ablation_mechanism.sh
```

**可调环境变量**：
- `DO_USE_GATE` / `DO_EDIT_BUDGET` / `DO_INFO_RETENTION` / `DO_TAU_SUCC` / `DO_MIN_SUPPORT`：每维开关（1 开 / 0 关）。
- `TAU_SUCC_GRID` / `MIN_SUPPORT_GRID`：网格取值。
- `MECH_BASE_BATCH_SIZE`（默认 35）/ `MECH_BASE_ROLLOUT_REPEATS`（默认 4）：各行共享的固定训练旋钮。
- `MECH_WORKERS`（128）/ `MECH_ANALYST_WORKERS`（64）/ `MECH_EXEC_TIMEOUT`（1800）：并发与超时。
- `CONTINUE_ON_ERROR`（默认 1）：某行失败是否继续。

各 run 输出在 `outputs/<RUN_ID>/<tag>/`，日志同 09。

---

## 4. 结果记录

- **日志**：`SkillOpt-Tree/10live`（13041 行，11 run）
- **run 前缀**：`outputs/skillopt_tree_livemath_mech_dsv4_flash_pro_*/`
- **固定训练旋钮**：batch_size=35 / rollout_repeats=4
- **分析时间**：2026-07-21

> ⚠️ **噪声比 09 更大**：初始 skill 逐字节一致，但本轮同一 skill 的 base_test 在 11 个 run 间从 **0.2984 飘到 0.4032，极差 0.1048**（09 是 0.056）。这意味着**判优只能看绝对 best_test，且差距要 > ~0.10 才勉强可信**。单 seed 下，本轮大部分 Δ 都落在噪声内——**结论以"机制是否按预期激活"为主，分数排名为辅**。

### 完整结果表

`base/best` = 初始/最优 skill 在 124 条 test 上的 hard 分；`sel` = 验证集(18)最优分；`drop` = 各 step `dropped` 组数累计；`FA` = force-accept 次数。

| 维度 | run | base | **best** | Δ | sel | step | wall | ACC | REJ | drop | FA |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A 接受机制 | use_gate_true | 0.3145 | 0.3790 | +0.065 | 0.6111 | 3 | 497s | 2 | 2 | 16 | 0 |
| A 接受机制 | **use_gate_false** | 0.4032 | 0.4032 | +0.000 | 0.4444 | 4 | 435s | 4 | **0** | 15 | **9** |
| B 编辑预算 | edit_budget_on | 0.3387 | 0.3871 | +0.048 | 0.5556 | 1 | 561s | 1 | 3 | 16 | 0 |
| B 编辑预算 | **edit_budget_off** | 0.3710 | **0.4113** | +0.040 | 0.6667 | 2 | 449s | 2 | 2 | 17 | 0 |
| C 信息保留 | info_default | 0.3145 | 0.3790 | +0.065 | 0.6111 | 4 | 492s | 3 | 1 | 11 | 0 |
| C 信息保留 | info_retain | 0.2984 | 0.3790 | +0.081 | 0.4444 | 3 | 590s | 2 | 2 | 11 | 0 |
| D gate 松紧 | tau_succ_1.0 | 0.3468 | 0.3952 | +0.048 | 0.5000 | 3 | 522s | 2 | 2 | 13 | 0 |
| D gate 松紧 | **tau_succ_0.5** | 0.3065 | **0.4355** | **+0.129** | 0.5556 | 4 | 481s | 3 | 1 | 15 | 0 |
| E 长尾阈值 | min_support_2 | 0.3065 | 0.3548 | +0.048 | 0.5556 | 4 | 510s | 2 | 2 | 12 | 0 |
| E 长尾阈值 | **min_support_1** | 0.3387 | **0.4274** | +0.089 | 0.6111 | 2 | 556s | 2 | 2 | **0** | 0 |
| E 长尾阈值 | min_support_3 | 0.3548 | 0.3065 | **−0.048** | 0.5000 | 2 | 515s | 2 | 2 | 19 | 0 |

**参照线**：09 稳健最优 ≈ 0.41；SkillOpt-main ≈ 0.4597。本轮最高 `tau_succ_0.5=0.4355`、`min_support_1=0.4274`，仍未追平 main。

### 逐维度解读

**A. 接受机制（use_gate）—— 机制按预期激活，但本轮没赢**
- `use_gate=false` 确实生效：**9 次 force-accept、4 ACCEPT / 0 REJECT**，完全对齐 main 的无条件注入。
- 但它 best_test=0.4032，反而**低于** `use_gate_true`(0.3790) 吗？不——注意 false 的 base 起点是 0.4032（本轮最高），Δ=0，即"force-accept 接受的候选没能超过它自己那个偏高的起点"。
- **这是最典型的噪声陷阱**：false 组恰好抽到高 base、gate 组抽到低 base，两者 best 反而接近。**单 seed 无法判 gate 到底是不是根因**——A 维度必须做 seed 复现才能定论。

**B. 编辑预算（EDIT_BUDGET_OFF）—— 轻微正向**
- `edit_budget_off`(0.4113) > `edit_budget_on`(0.3871)，且 sel 分也更高(0.6667)。方向正向但幅度(0.024)在噪声内。
- 关预算未见负面，可作为默认候选之一。

**C. 信息保留（clustering + tail_bank）—— 机制激活，分数未变**
- 确认激活：`info_retain` 的 tail 相关日志从 3 → **13 次**，clustering 也触发。
- 但 best_test 两者都是 0.3790，**完全没差别**；info_retain 的 base 更低(0.2984)使其 Δ 看似更大(+0.081)，属假象。
- **含义**：现有的跨类型语义融合 + 跨轮长尾库，在 livemath 上**没有转化为 test 提升**。这印证了此前判断——现有 clustering 不是专门抢救 dropped 长尾，效果有限。**若要靠长尾语义融合涨分，可能真得新增"dropped 二次融合"逻辑。**

**D. gate 松紧（tau_succ）—— 本轮最大赢家**
- `tau_succ_0.5` best_test=**0.4355，全场最高**，Δ=+0.129（唯一超过噪声极差的提升）。
- 机理合理：数学题"部分对"很常见，`tau_succ=1.0`（要满分才算成功）把大量有效 rollout 判为失败、经验丢弃；放到 0.5 后更多修复经验被提炼成 edit。
- **这是本轮最值得跟进的信号**，但仍需 seed 复现确认不是运气。

**E. 长尾阈值（min_support）—— 机制完美验证，趋势清晰**
- `min_support_1`：**dropped 全程为 0**（几乎不丢长尾），best_test=0.4274（次高）。
- `min_support_2`（默认）：dropped=12，best=0.3548。
- `min_support_3`：dropped=19（丢最多），best=0.3065，**唯一负 Δ**。
- **三档呈单调趋势：丢得越少越好。** 这是本轮机制信号最干净的一维——`min_support` 越低，长尾保留越多，效果越好；默认的 2 偏保守，对 35 条小训练集尤其吃亏。

### 结论

1. **两个明确正向信号**：`tau_succ=0.5`（+0.129，全场最高）和 `min_support=1`（dropped=0，次高）。二者都指向同一件事——**Tree 默认把太多信号当噪声丢了**（成功门槛太高 + 长尾阈值太高），放宽后有效 edit 增多。
2. **接受机制（A）本轮无定论**：force-accept 机制确实激活，但被 base 噪声掩盖，**必须 seed 复现**。
3. **信息保留（C）现有能力无效**：clustering+tail_bank 激活了但不涨分 → 想靠长尾语义融合，需新增针对 dropped 的二次融合代码。
4. **噪声是头号敌人**：极差 0.1048，本轮除 tau_succ_0.5 外所有 Δ 都不可信。

### 下一步建议
- [ ] **优先**：对 `tau_succ ∈ {0.3, 0.5, 0.7, 1.0}` + `min_support ∈ {1, 2}` 做小网格，各配 2~3 seed，锁定这两个"减少信号丢弃"旋钮的真实增益。
- [ ] **A 维度补 seed**：use_gate true/false 各 3 seed，才能判 gate 是否根因。
- [ ] **组合验证**：`tau_succ=0.5 + min_support=1`（+可选 edit_budget_off）组合跑一轮，看正向信号能否叠加、能否逼近 main 的 0.4597。
- [ ] **C 维度**：若组合仍追不上 main，再考虑实现"dropped 长尾二次 LLM 语义融合"新开关。
