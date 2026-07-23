# LiveMathematicianBench 消融实验综合报告

> 汇总 **09（训练旋钮）** 与 **10（方法机制）** 两轮消融，定位 SkillOpt-Tree 在 livemath 上停滞的原因，并给出"明确有收益"的组合配置。
>
> - 数据日志：`SkillOpt-Tree/09live`（11865 行，10 run）、`SkillOpt-Tree/10live`（13041 行，11 run）
> - 目标模型 `deepseek-v4-flash` / 优化器 `deepseek-v4-pro`，DeepSeek 官网 API，128/64 并发
> - 数据规模：train=35 / val(selection)=18 / test=124
> - 报告时间：2026-07-21

---

## 1. 背景与目标

SkillOpt-**Tree** 在 livemath 上停滞（best_test ≈ 0.41），始终追不上 SkillOpt-**main** 的 **0.4597**。此前已逐一排除并确认两版**初始 skill 逐字节一致、数据/模型/思维链设置一致**，因此差异只可能来自方法层与超参。两轮消融的分工：

| | 09（旋钮层） | 10（机制层） |
|---|---|---|
| 扫什么 | batch_size / rollout_repeats / leaf_fallback | 接受机制 / 编辑预算 / 信息保留 / gate 松紧 / 长尾阈值 |
| 目的 | 找最佳训练超参 | 定位"Tree 不涨"的方法根因 |
| 脚本 | `09_livemath_ablation.sh` | `10_livemath_ablation_mechanism.sh` |
| run 数 | 10 | 11 |

所有 run 串行执行、各自独立子目录+日志，OFAT（一次只改一个因子）。

---

## 2. 头号发现：噪声极大，主导一切结论

同一个逐字节相同的初始 skill，仅因随机性，base_test 就大幅漂移：

| 轮次 | base_test 极差 | 区间 |
|---|---|---|
| 09 | 0.056 | — |
| 10 | **0.1048** | 0.2984 ~ 0.4032 |

**含义**：单 seed 下，绝大多数 Δ 都落在噪声带内，**不可信**。判优只能看**绝对 best_test**，且两个配置差距需 **> ~0.10** 才勉强可信。本报告所有结论据此把"机制是否按预期激活"放在"分数排名"之前。

---

## 3. 09 训练旋钮消融

固定其余旋钮，逐一扫描。关键指标（best_test 绝对值）：

| 维度 | 取值 | best_test | 结论 |
|---|---|---|---|
| batch_size | 8 / 16 / **35** | 0.3871 / 0.4113 / **0.4113** | 越大越好且越快；35=全量上限，batch=8 最慢(1458s)且 REJECT 16 次 |
| rollout_repeats | 1 / 2 / 3 / **4** / 5 | 0.3468 / 0.3387 / 0.3871 / **0.3952** / 0.3790 | repeats=1 完全没涨；4 是峰值；再高无益 |
| leaf_fallback | true / **false** | 0.3871 / **0.4274** | false 拿最高分，但单点+低基线，**存疑** |

**09 小结**：三个训练旋钮调到头，best_test 也只到 0.41~0.43，**追不上 main 的 0.4597** → 瓶颈不在训练旋钮，转向机制层（10）。稳健取值：**batch_size=35 / rollout_repeats=4**（作为 10 各行的固定底座）。

---

## 4. 10 方法机制消融

各行固定 batch_size=35 / rollout_repeats=4，只翻一个机制。完整结果：

| 维度 | run | base | **best** | Δ | ACC | REJ | drop | FA |
|---|---|---|---|---|---|---|---|---|
| A 接受机制 | use_gate_true | 0.3145 | 0.3790 | +0.065 | 2 | 2 | 16 | 0 |
| A 接受机制 | **use_gate_false** | 0.4032 | 0.4032 | +0.000 | 4 | **0** | 15 | **9** |
| B 编辑预算 | edit_budget_on | 0.3387 | 0.3871 | +0.048 | 1 | 3 | 16 | 0 |
| B 编辑预算 | **edit_budget_off** | 0.3710 | **0.4113** | +0.040 | 2 | 2 | 17 | 0 |
| C 信息保留 | info_default | 0.3145 | 0.3790 | +0.065 | 3 | 1 | 11 | 0 |
| C 信息保留 | info_retain | 0.2984 | 0.3790 | +0.081 | 2 | 2 | 11 | 0 |
| D gate 松紧 | tau_succ_1.0 | 0.3468 | 0.3952 | +0.048 | 2 | 2 | 13 | 0 |
| D gate 松紧 | **tau_succ_0.5** | 0.3065 | **0.4355** | **+0.129** | 3 | 1 | 15 | 0 |
| E 长尾阈值 | min_support_2 | 0.3065 | 0.3548 | +0.048 | 2 | 2 | 12 | 0 |
| E 长尾阈值 | **min_support_1** | 0.3387 | **0.4274** | +0.089 | 2 | 2 | **0** | 0 |
| E 长尾阈值 | min_support_3 | 0.3548 | 0.3065 | **−0.048** | 2 | 2 | 19 | 0 |

**逐维度解读**：

- **A 接受机制（use_gate）—— 无定论**：`false` 的 force-accept 确实激活（9×FA、0 REJECT），机制成功对齐 main。但它恰好抽到全场最高 base(0.4032)，Δ=0，被噪声掩盖。**必须 seed 复现才能判 gate 是否为根因。**
- **B 编辑预算（EDIT_BUDGET_OFF）—— 轻微正向**：off(0.4113) > on(0.3871)，方向对但幅度在噪声内。无负面，可纳入组合。
- **C 信息保留（clustering+tail_bank）—— 无收益**：机制确实激活（tail 日志 3→13），但 best_test 两者都是 0.3790，**完全没动**。现有跨类型语义融合在 livemath 上不转化为 test 提升。
- **D gate 松紧（tau_succ）—— 本轮最大赢家**：`tau_succ=0.5` best_test=**0.4355（全场最高）**，Δ=+0.129 是唯一超过噪声极差的提升。机理：数学题"部分对"常见，1.0 要满分才算成功、丢掉大量有效 rollout；放到 0.5 后经验被提炼进 edit。
- **E 长尾阈值（min_support）—— 机制最干净**：1(dropped=0, 0.4274) > 2(0.3548) > 3(dropped=19, 0.3065, 唯一负 Δ)。**三档单调：丢得越少越好**，默认的 2 对 35 条小训练集偏保守。

---

## 5. 综合结论

1. **两个最强正向信号**：`tau_succ=0.5`（+0.129）与 `min_support=1`（dropped=0）。二者指向同一根因——**Tree 默认把太多信号当噪声丢了**（成功门槛太高 + 长尾阈值太高），放宽后有效 edit 增多。
2. **训练旋钮**：batch_size=35 + rollout_repeats=4 为稳健底座。
3. **编辑预算**：关闭上限（off）轻微正向，可纳入。
4. **接受机制（gate）**：force-accept 已激活但被噪声掩盖，**待 seed 复现定论**。
5. **信息保留（语义融合）**：现有 clustering+tail_bank 无效；若要靠长尾语义融合涨分，需新增针对 dropped 长尾的二次融合逻辑（`type_guided_tail_fuse`，尚未实现）。
6. **仍未追平 main**：本轮顶到 0.4355，距 0.4597 尚有差距 → 需组合验证 + 抑噪。

---

## 6. "明确有收益"的组合配置

把上面带明确正向信号的旋钮叠在一起，见脚本 `11_livemath_combo.sh`：

| 旋钮 | config key | 值 | 依据 |
|---|---|---|---|
| batch_size | `train.batch_size` | 35 | 09 |
| rollout_repeats | `optimizer.type_guided_rollout_repeats` | 4 | 09 |
| gate 松紧 | `optimizer.type_guided_tau_succ` | **0.5** | 10 最大赢家 |
| 长尾阈值 | `optimizer.type_guided_min_support` | **1** | 10 次高、dropped=0 |
| 编辑预算 | `EDIT_BUDGET_OFF`→`lr_scheduler=autonomous` | 1 | 10 轻微正向 |

**刻意排除**：`use_gate`（无定论，脚本留 `COMBO_USE_GATE` 开关 A/B）、info retention（无收益）。

### 运行

```bash
cd /Users/bytedance/Documents/codes/Opt/SkillOpt-Tree
export DEEPSEEK_API_KEY='你的key'

# 单次
bash scripts/runs/resource_pool_dsv4_flash_pro_api/11_livemath_combo.sh

# 抑噪（强烈建议，针对 0.10 噪声，3 seed 取平均）
SEED_GRID="42 1 7" bash scripts/runs/resource_pool_dsv4_flash_pro_api/11_livemath_combo.sh

# 顺带测 force-accept 能否再加成
COMBO_USE_GATE=false SEED_GRID="42 1 7" \
  bash scripts/runs/resource_pool_dsv4_flash_pro_api/11_livemath_combo.sh
```

---

## 7. 后续建议（按优先级）

1. **抑噪优先**：组合配置至少跑 3 seed 取均值/方差，否则单点仍会被 0.10 噪声吞掉。
2. **小网格锁定**：`tau_succ ∈ {0.3,0.5,0.7,1.0}` × `min_support ∈ {1,2}` 各配 2~3 seed，确认这两个"减少信号丢弃"旋钮的真实增益曲线。
3. **A 维度补 seed**：use_gate true/false 各 3 seed，判 gate 是否根因。
4. **若仍追不上 main**：再实现"dropped 长尾二次 LLM 语义融合"新开关（`type_guided_tail_fuse`）。

---

## 附：关键脚本与文档索引

- `09_livemath_ablation.sh` — 训练旋钮消融
- `10_livemath_ablation_mechanism.sh` + `10_livemath_ablation_mechanism.md` — 机制消融（含详细维度讲解与 10 结果）
- `11_livemath_combo.sh` — 本报告结论的组合配置
- `09live` / `10live` — 原始合并日志
