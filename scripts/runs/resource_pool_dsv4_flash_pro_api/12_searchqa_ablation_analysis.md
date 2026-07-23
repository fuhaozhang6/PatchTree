# SearchQA 消融实验结果分析

> 数据来源：`SkillOpt-Tree/12_search`（21.8 MB / 282,719 行合并日志），脚本 `12_searchqa_ablation.sh`。
> 目标模型 deepseek-v4-flash / 优化器 deepseek-v4-pro，DeepSeek 官网 API，128/64 并发，exec_timeout=600s。
> 数据规模：train=400 / val(selection)=200 / test=1400。指标为 test（1400 条）上的 overall（hard）。
> 各 run 串行、一次只改一个因子（OFAT）。

---

## 1. 做了什么

在 SearchQA 上对以下维度做机制/旋钮消融，每个 run 相对共享基线只改一个因子：

- **H 批大小** `train.batch_size`：20 / 40 / 80
- **A 接受机制** `evaluation.use_gate`：true / false（false = force-accept）
- **B 编辑预算** `EDIT_BUDGET_OFF`：0（默认上限）/ 1（`lr_scheduler=autonomous`，放开上限）
- **D gate 松紧** `optimizer.type_guided_tau_succ`：0.5 / 1.0
- **E 长尾阈值** `optimizer.type_guided_min_support`：1（min_support_2/3 未在本日志中）

共享固定旋钮：batch_size=40、rollout_repeats=3。

---

## 2. 结果总表

日志中已完成 9 个 run（`min_support_1` 为日志末尾正在进行的 run，未打印最终 test 结果，故不计入）。

| run | base_test | best_test | Δtest | sel(best_score) | best step | wall | ACC | REJ | FA | dropped |
|---|---|---|---|---|---|---|---|---|---|---|
| batch_size_20 | 0.7264 | 0.7700 | +0.0436 | 0.7900 | 30 | 3304s | 5 | 75 | 0 | 146 |
| batch_size_40 | 0.7393 | 0.7929 | +0.0536 | 0.8000 | 19 | 3013s | 9 | 31 | 0 | 75 |
| batch_size_80 | 0.7386 | 0.7764 | +0.0378 | 0.7900 | 3 | 2628s | 3 | 17 | 0 | 49 |
| use_gate_true | 0.7350 | 0.8000 | +0.0650 | 0.7900 | 15 | 3496s | 7 | 33 | 0 | 74 |
| use_gate_false | 0.7321 | 0.7886 | +0.0565 | 0.7850 | 15 | 2321s | 40 | 0 | 81 | 53 |
| edit_budget_on | 0.7329 | 0.7700 | +0.0371 | 0.7850 | 34 | 3279s | 2 | 38 | 0 | 102 |
| edit_budget_off | 0.7429 | 0.7886 | +0.0457 | 0.8000 | 16 | 2846s | 5 | 35 | 0 | 94 |
| tau_succ_0.5 | 0.7471 | 0.7893 | +0.0422 | 0.7950 | 7 | 2368s | 6 | 33 | 0 | 82 |
| tau_succ_1.0 | 0.7371 | 0.8050 | +0.0679 | 0.7950 | 16 | 3288s | 5 | 35 | 0 | 79 |

**统计（9 个完成的 run）**：
- base_test：mean=0.7368，max=0.7471，min=0.7264，**spread=0.0207**
- best_test：mean=0.7868，max=**0.8050**（tau_succ_1.0），min=0.7700，spread=0.0350
- 所有 run 的 Δtest 均为正（+0.0371 ~ +0.0679）

---

## 3. 逐维度结果

### H. 批大小（train.batch_size）
| 取值 | best_test | wall | best step |
|---|---|---|---|
| 20 | 0.7700 | 3304s | 30 |
| 40 | 0.7929 | 3013s | 19 |
| 80 | 0.7764 | 2628s | 3 |

batch_size=40 的 best_test 最高（0.7929）；batch_size=80 在第 3 步即达到最佳并最快结束（2628s）；batch_size=20 用时最长（3304s）且 REJECT 最多（75）、dropped 最多（146）。

### A. 接受机制（evaluation.use_gate）
| 取值 | best_test | ACC | REJ | FA |
|---|---|---|---|---|
| true | 0.8000 | 7 | 33 | 0 |
| false | 0.7886 | 40 | 0 | 81 |

use_gate=false 触发 force-accept（81 次 force-accept、0 次 REJECT、40 次 ACCEPT），wall 最短（2321s）；其 best_test（0.7886）低于 use_gate=true（0.8000）。

### B. 编辑预算（EDIT_BUDGET_OFF）
| 取值 | best_test |
|---|---|
| on（默认上限） | 0.7700 |
| off（autonomous） | 0.7886 |

edit_budget_off 的 best_test（0.7886）高于 on（0.7700）。

### D. gate 松紧（type_guided_tau_succ）
| 取值 | best_test |
|---|---|
| 0.5 | 0.7893 |
| 1.0 | 0.8050 |

tau_succ=1.0 的 best_test 为全表最高（0.8050）；tau_succ=0.5 为 0.7893。

### E. 长尾阈值（type_guided_min_support）
min_support_1 为日志末尾正在进行的 run，未打印最终 test 结果（当前 dropped=0，ACC=6/REJ=11）；min_support_2、min_support_3 未包含在本日志中。

---

## 4. 与主表对照

`paper/实验.xlsx` 中 deepseek-flash 系列 SearchQA 得分：

| 方法 | SearchQA |
|---|---|
| No skill | 74.57 |
| TextGrad | 68.5 |
| GEPA | 74.21 |
| SkillOpt | 81.29 |
| PatchTree | 81.57 |

本轮消融 base_test 约 0.7264~0.7471（72.6%~74.7%，与 No skill 量级一致），best_test 最高 0.8050（80.5%）。

---

## 5. 各维度含义（备查）

- **batch size**：每步梯度所用训练样本数（train=400）。
- **use_gate**：true 时用验证集 gate 判定是否接受候选；false 时 force-accept（无条件接受候选为新 current，best-so-far 单独追踪）。
- **EDIT_BUDGET_OFF**：off 时每步编辑数按调度递减；on(=1) 时切 `lr_scheduler=autonomous`，放开每步编辑上限。
- **tau_succ**：rollout 判成功的分数门槛，1.0 需满分、0.5 半对即算成功。
- **min_support**：一个 (question_type, revision_type) 分组被保留所需的最小样本数，低于该值的分组计入 dropped。
- **dropped**：因未达 min_support 而被丢弃的分组累计数。
- **FA**：force-accept 次数。

---

## 附：run 与日志

- 脚本：`scripts/runs/resource_pool_dsv4_flash_pro_api/12_searchqa_ablation.sh`
- 日志：`SkillOpt-Tree/12_search`
- 已完成 run：batch_size_{20,40,80}、use_gate_{true,false}、edit_budget_{on,off}、tau_succ_{0.5,1.0}
- 未完成/缺失：min_support_1（进行中）、min_support_2、min_support_3
