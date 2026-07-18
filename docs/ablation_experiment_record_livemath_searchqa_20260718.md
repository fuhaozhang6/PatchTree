# PatchTree 消融实验原始记录：LiveMath 与 SearchQA

最后更新：2026-07-18  
记录范围：LiveMath core-8、LiveMath follow-up-4、SearchQA fallback pilot、SearchQA tree-shape pilot。  
记录原则：本文件以实际训练/测试日志为准，保存参数、原始指标、运行状态和路径；不承担方法优劣分析。

## 1. 路径约定与状态

本地代码与日志根目录：

```text
/Users/bytedance/Documents/codes/Opt/SkillOpt-Tree
```

正式训练机器上的项目与输出根目录：

```text
/ai-app-vepfs/zhangfuhao/skill/SkillOpt-Tree
```

模型：

```text
optimizer: deepseek-v4-pro
target:    Qwen/Qwen3.5-4B
model:     /ai-app-vepfs/modelscope/models/Qwen/Qwen3.5-4B
```

状态说明：

- `有效训练`：日志包含 `[exit] 0`，且生成了 best skill。
- `有效 TEST`：完整测试集完成，结果可用于记录。
- `失效 TEST`：测试服务报错，分数不得用于实验比较。
- `未执行`：只有脚本或 dry-run，没有正式训练结果。

## 2. LiveMath 消融实验

### 2.1 数据与共同参数

实际正式训练日志中的共同参数：

| 参数 | 值 |
|---|---:|
| train / val / test | 35 / 18 / 124 |
| epochs | 4 |
| seed | 42 |
| optimizer | `deepseek-v4-pro` |
| target | `Qwen/Qwen3.5-4B` |
| target temperature | 0.2 |
| target request timeout | 300 s |
| target max tokens | 16384 |
| edit budget / minimum | 4 / 2 |
| LR scheduler | cosine |
| gate | mixed，hard/soft 权重 0.5/0.5 |
| min support | 2 |
| max leaf groups | 8 |
| leaf fallback | true |
| fallback top-k | 4 |
| clustering | false |
| tail bank | false |
| max PatchRecords | 32 |
| target workers | 48 |
| analyst workers | 16 |
| PatchRecord workers | 16 |
| training-time TEST | false，后续统一 eval-only |
| split | `/ai-app-vepfs/zhangfuhao/skill/SkillOpt-Tree/data/livemathematicianbench_split` |

说明：上述并发值来自 2026-07-16/17 正式日志中的实际命令，不使用脚本后来更新后的默认值代替历史值。

### 2.2 LiveMath 训练与 TEST 总表

分数均为日志原始小数。LiveMath 此处 hard 与 soft 相同；正确数由完整 TEST 的
`score × 124` 回算。

| 实验 | repeats | batch | depth | steps | baseline val | best val | best step | accept / reject / skip | TEST hard | TEST 正确数 | best skill 字符数 |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| `r1_b8_d2` | 1 | 8 | 2 | 20 | 0.2778 | 0.6111 | 16 | 4 / 15 / 1 | 0.3306 | 41/124 | 8925 |
| `r2_b8_d2` | 2 | 8 | 2 | 20 | 0.2778 | 0.8333 | 15 | 3 / 17 / 0 | 0.4274 | 53/124 | 6714 |
| `r3_b8_d2` | 3 | 8 | 2 | 20 | 0.2222 | 0.6111 | 3 | 3 / 17 / 0 | 0.3710 | 46/124 | 9569 |
| `r4_b8_d2` | 4 | 8 | 2 | 20 | 0.2778 | 0.7778 | 3 | 3 / 16 / 1 | 0.4919 | 61/124 | 11958 |
| `r3_b4_d2` | 3 | 4 | 2 | 36 | 0.0556 | 0.8333 | 24 | 6 / 30 / 0 | 0.4032 | 50/124 | 13606 |
| `r3_b16_d2` | 3 | 16 | 2 | 12 | 0.1667 | 0.7222 | 8 | 2 / 10 / 0 | 0.4355 | 54/124 | 8928 |
| `r3_b32_d2` | 3 | 32 | 2 | 8 | 0.1111 | 0.3889 | 3 | 2 / 6 / 0 | 0.3306 | 41/124 | 6196 |
| `r3_b8_d3` | 3 | 8 | 3 | 20 | 0.2778 | 0.7222 | 12 | 5 / 15 / 0 | 0.4677 | 58/124 | 12336 |
| `r5_b8_d3` | 5 | 8 | 3 | 20 | 0.1667 | 0.6667 | 12 | 4 / 16 / 0 | 0.4113 | 51/124 | 10959 |
| `r8_b8_d3` | 8 | 8 | 3 | 20 | 0.2778 | 0.6111 | 7 | 4 / 4 / 12 | 0.4355 | 54/124 | 15150 |
| `r4_b16_d2` | 4 | 16 | 2 | 12 | 0.2222 | 0.5556 | 7 | 3 / 6 / 3 | 0.4274 | 53/124 | 9131 |
| `r4_b16_d3` | 4 | 16 | 3 | 12 | 0.2222 | 0.6111 | 5 | 3 / 9 / 0 | 0.3952 | 49/124 | 11745 |

### 2.3 LiveMath 训练成本原始记录

这些记录来自训练日志 Final Summary；均不包含后续独立 TEST。

| 实验 | wall 秒 | prompt tokens | completion tokens | total tokens | calls |
|---|---:|---:|---:|---:|---:|
| `r1_b8_d2` | 12428 | 2,281,484 | 3,108,945 | 5,390,429 | 862 |
| `r2_b8_d2` | 14116 | 3,200,231 | 3,873,805 | 7,074,036 | 977 |
| `r3_b8_d2` | 16381 | 5,342,561 | 5,199,179 | 10,541,740 | 1227 |
| `r4_b8_d2` | 14521 | 6,012,265 | 5,244,933 | 11,257,198 | 1242 |
| `r3_b4_d2` | 20118 | 6,409,876 | 6,197,556 | 12,607,432 | 1515 |
| `r3_b16_d2` | 12212 | 3,903,600 | 4,482,342 | 8,385,942 | 1083 |
| `r3_b32_d2` | 7067 | 3,376,698 | 2,966,356 | 6,343,054 | 853 |
| `r3_b8_d3` | 16710 | 5,053,717 | 5,449,248 | 10,502,965 | 1293 |
| `r5_b8_d3` | 17108 | 7,644,829 | 6,618,919 | 14,263,748 | 1547 |
| `r8_b8_d3` | 7941 | 4,036,929 | 2,638,658 | 6,675,587 | 774 |
| `r4_b16_d2` | 7323 | 3,030,691 | 2,537,797 | 5,568,488 | 769 |
| `r4_b16_d3` | 10708 | 5,637,245 | 3,995,899 | 9,633,144 | 1169 |

### 2.4 LiveMath best-val 接受轨迹

格式为 `step:best_val`。未列出的普通 step 为 reject；skip 单独列出。

| 实验 | 接受轨迹 | skip steps |
|---|---|---|
| `r1_b8_d2` | `2:0.3333, 6:0.5000, 9:0.5556, 16:0.6111` | 15 |
| `r2_b8_d2` | `1:0.4444, 2:0.7222, 15:0.8333` | — |
| `r3_b8_d2` | `1:0.3333, 2:0.3889, 3:0.6111` | — |
| `r4_b8_d2` | `1:0.6111, 2:0.7222, 3:0.7778` | 5 |
| `r3_b4_d2` | `1:0.1111, 2:0.5000, 4:0.5556, 6:0.6667, 20:0.7778, 24:0.8333` | — |
| `r3_b16_d2` | `1:0.5556, 8:0.7222` | — |
| `r3_b32_d2` | `1:0.2778, 3:0.3889` | — |
| `r3_b8_d3` | `3:0.4444, 7:0.5000, 9:0.5556, 11:0.6667, 12:0.7222` | — |
| `r5_b8_d3` | `1:0.4444, 2:0.5000, 3:0.6111, 12:0.6667` | — |
| `r8_b8_d3` | `1:0.3889, 2:0.4444, 3:0.5556, 7:0.6111` | 9–20 |
| `r4_b16_d2` | `1:0.3889, 4:0.5000, 7:0.5556` | 9、11、12 |
| `r4_b16_d3` | `1:0.2778, 2:0.5000, 5:0.6111` | — |

### 2.5 LiveMath 原始路径索引

#### Core-8

| 实验 | 启动脚本 | 本地训练日志 | 远端 best skill |
|---|---|---|---|
| `r1_b8_d2` | `scripts/runs/ablations/livemath_core8/run_01_r1_r4.sh` | `/Users/bytedance/Documents/codes/Opt/SkillOpt-Tree/logs/livemath_core8_pairs/r1_r4_seed42_20260716_161711/r1_b8_d2/livemath.log` | `/ai-app-vepfs/zhangfuhao/skill/SkillOpt-Tree/outputs/livemath_core8_pairs/r1_r4_seed42_20260716_161711/r1_b8_d2/livemath/best_skill.md` |
| `r4_b8_d2` | `scripts/runs/ablations/livemath_core8/run_01_r1_r4.sh` | `/Users/bytedance/Documents/codes/Opt/SkillOpt-Tree/logs/livemath_core8_pairs/r1_r4_seed42_20260716_161711/r4_b8_d2/livemath.log` | `/ai-app-vepfs/zhangfuhao/skill/SkillOpt-Tree/outputs/livemath_core8_pairs/r1_r4_seed42_20260716_161711/r4_b8_d2/livemath/best_skill.md` |
| `r2_b8_d2` | `scripts/runs/ablations/livemath_core8/run_02_r2_base.sh` | `/Users/bytedance/Documents/codes/Opt/SkillOpt-Tree/logs/livemath_core8_pairs/r2_base_seed42_20260716_162410/r2_b8_d2/livemath.log` | `/ai-app-vepfs/zhangfuhao/skill/SkillOpt-Tree/outputs/livemath_core8_pairs/r2_base_seed42_20260716_162410/r2_b8_d2/livemath/best_skill.md` |
| `r3_b8_d2` | `scripts/runs/ablations/livemath_core8/run_02_r2_base.sh` | `/Users/bytedance/Documents/codes/Opt/SkillOpt-Tree/logs/livemath_core8_pairs/r2_base_seed42_20260716_162410/base_r3_b8_d2/livemath.log` | `/ai-app-vepfs/zhangfuhao/skill/SkillOpt-Tree/outputs/livemath_core8_pairs/r2_base_seed42_20260716_162410/base_r3_b8_d2/livemath/best_skill.md` |
| `r3_b4_d2` | `scripts/runs/ablations/livemath_core8/run_03_b4_b32.sh` | `/Users/bytedance/Documents/codes/Opt/SkillOpt-Tree/logs/livemath_core8_pairs/b4_b32_seed42_20260716_162509/r3_b4_d2/livemath.log` | `/ai-app-vepfs/zhangfuhao/skill/SkillOpt-Tree/outputs/livemath_core8_pairs/b4_b32_seed42_20260716_162509/r3_b4_d2/livemath/best_skill.md` |
| `r3_b32_d2` | `scripts/runs/ablations/livemath_core8/run_03_b4_b32.sh` | `/Users/bytedance/Documents/codes/Opt/SkillOpt-Tree/logs/livemath_core8_pairs/b4_b32_seed42_20260716_162509/r3_b32_d2/livemath.log` | `/ai-app-vepfs/zhangfuhao/skill/SkillOpt-Tree/outputs/livemath_core8_pairs/b4_b32_seed42_20260716_162509/r3_b32_d2/livemath/best_skill.md` |
| `r3_b16_d2` | `scripts/runs/ablations/livemath_core8/run_04_b16_d3.sh` | `/Users/bytedance/Documents/codes/Opt/SkillOpt-Tree/logs/livemath_core8_pairs/b16_d3_seed42_20260716_162509/r3_b16_d2/livemath.log` | `/ai-app-vepfs/zhangfuhao/skill/SkillOpt-Tree/outputs/livemath_core8_pairs/b16_d3_seed42_20260716_162509/r3_b16_d2/livemath/best_skill.md` |
| `r3_b8_d3` | `scripts/runs/ablations/livemath_core8/run_04_b16_d3.sh` | `/Users/bytedance/Documents/codes/Opt/SkillOpt-Tree/logs/livemath_core8_pairs/b16_d3_seed42_20260716_162509/r3_b8_d3/livemath.log` | `/ai-app-vepfs/zhangfuhao/skill/SkillOpt-Tree/outputs/livemath_core8_pairs/b16_d3_seed42_20260716_162509/r3_b8_d3/livemath/best_skill.md` |

Core-8 有效 TEST：

```text
测试脚本：
/Users/bytedance/Documents/codes/Opt/SkillOpt-Tree/scripts/runs/ablations/livemath_core8/run_test_core8_best_skills.sh

本地测试日志目录：
/Users/bytedance/Documents/codes/Opt/SkillOpt-Tree/logs/livemath_core8_test_20260717_001343

远端测试输出目录：
/ai-app-vepfs/zhangfuhao/skill/SkillOpt-Tree/outputs/livemath_core8_test_20260717_001343
```

Core-8 TEST 服务实际参数：

```text
max_model_len=65536
max_num_batched_tokens=65536
max_num_seqs=128
enable_prefix_caching=true
enable_chunked_prefill=true
test items=124
```

#### Follow-up-4

| 实验 | 启动脚本 | 本地训练日志 | 远端 best skill |
|---|---|---|---|
| `r5_b8_d3` | `scripts/runs/ablations/livemath_followup4/run_01_r5_r4b16d3.sh` | `/Users/bytedance/Documents/codes/Opt/SkillOpt-Tree/logs/livemath_core8_pairs/r5_r4b16d3_seed42_20260717_023532/r5_b8_d3/livemath.log` | `/ai-app-vepfs/zhangfuhao/skill/SkillOpt-Tree/outputs/livemath_core8_pairs/r5_r4b16d3_seed42_20260717_023532/r5_b8_d3/livemath/best_skill.md` |
| `r4_b16_d3` | `scripts/runs/ablations/livemath_followup4/run_01_r5_r4b16d3.sh` | `/Users/bytedance/Documents/codes/Opt/SkillOpt-Tree/logs/livemath_core8_pairs/r5_r4b16d3_seed42_20260717_023532/r4_b16_d3/livemath.log` | `/ai-app-vepfs/zhangfuhao/skill/SkillOpt-Tree/outputs/livemath_core8_pairs/r5_r4b16d3_seed42_20260717_023532/r4_b16_d3/livemath/best_skill.md` |
| `r8_b8_d3` | `scripts/runs/ablations/livemath_followup4/run_02_r8_r4b16d2.sh` | `/Users/bytedance/Documents/codes/Opt/SkillOpt-Tree/logs/livemath_core8_pairs/r8_r4b16d2_seed42_20260717_023553/r8_b8_d3/livemath.log` | `/ai-app-vepfs/zhangfuhao/skill/SkillOpt-Tree/outputs/livemath_core8_pairs/r8_r4b16d2_seed42_20260717_023553/r8_b8_d3/livemath/best_skill.md` |
| `r4_b16_d2` | `scripts/runs/ablations/livemath_followup4/run_02_r8_r4b16d2.sh` | `/Users/bytedance/Documents/codes/Opt/SkillOpt-Tree/logs/livemath_core8_pairs/r8_r4b16d2_seed42_20260717_023553/r4_b16_d2/livemath.log` | `/ai-app-vepfs/zhangfuhao/skill/SkillOpt-Tree/outputs/livemath_core8_pairs/r8_r4b16d2_seed42_20260717_023553/r4_b16_d2/livemath/best_skill.md` |

Follow-up-4 有效 TEST：

```text
测试脚本：
/Users/bytedance/Documents/codes/Opt/SkillOpt-Tree/scripts/runs/ablations/livemath_followup4/run_test_followup4_best_skills.sh

本地有效测试日志目录：
/Users/bytedance/Documents/codes/Opt/SkillOpt-Tree/logs/livemath_followup4_test_20260717_142624

远端有效测试输出目录：
/ai-app-vepfs/zhangfuhao/skill/SkillOpt-Tree/outputs/livemath_followup4_test_20260717_142624
```

有效重测服务实际参数：

```text
gpu_memory_utilization=0.85
max_model_len=65536
max_num_batched_tokens=32768
max_num_seqs=64
available KV cache memory=25.21 GiB
GPU KV cache size=206448 tokens
test items=124
```

### 2.6 LiveMath 失效 TEST 记录

第一次 follow-up-4 测试必须保留为事故记录，但不能使用其中分数：

```text
本地失效测试日志：
/Users/bytedance/Documents/codes/Opt/SkillOpt-Tree/logs/livemath_followup4_test_20260717_140120

远端失效测试输出：
/ai-app-vepfs/zhangfuhao/skill/SkillOpt-Tree/outputs/livemath_followup4_test_20260717_140120
```

日志中出现：

```text
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.25 GiB.
GPU total capacity: 44.53 GiB
EngineDeadError
HTTP 500 Internal Server Error
```

该次日志中的原始输出为：

| 实验 | 失效运行输出 |
|---|---:|
| `r5_b8_d3` | 0.3548 |
| `r4_b16_d2` | 0.0000 |
| `r4_b16_d3` | 0.0000 |
| `r8_b8_d3` | 0.0000 |

这些数字全部标记为无效；正式记录采用 `20260717_142624` 重测结果。

## 3. SearchQA 消融实验

### 3.1 数据与共同参数

| 参数 | 值 |
|---|---:|
| train / val / test | 400 / 200 / 1400 |
| epochs | 1 |
| batch size | 40 |
| rollout repeats | 3 |
| steps | 10 |
| seed | 42 |
| optimizer | `deepseek-v4-pro` |
| target | `Qwen/Qwen3.5-4B` |
| target temperature | 0.2 |
| target timeout | 300 s |
| target max tokens | 16384 |
| edit budget / minimum | 4 / 2 |
| gate | mixed，hard/soft 权重 0.5/0.5 |
| min support | 2 |
| max leaf groups | 8 |
| max PatchRecords | 40 |
| target workers | 128 |
| analyst workers | 48 |
| PatchRecord workers | 40 |
| fallback sample | 40 个 val items |
| fallback top-k | 4 |
| fallback reconcile | deterministic |
| tail bank | false |
| split | `/ai-app-vepfs/zhangfuhao/skill/SkillOpt-Tree/data/searchqa_split` |

### 3.2 SearchQA 配置、训练与 TEST 总表

表中的 TEST hard/soft 来自 eval 日志；TEST mixed 按同一 gate 定义
`(hard + soft) / 2` 计算。

| ID | depth | fallback | clustering | cluster target/max | baseline val H/S/M | best val H/S/M | best step | A/R/S | TEST H/S/M | TEST 正确数 | skill 字符数 |
|---|---:|---|---|---|---|---|---:|---|---|---:|---:|
| P1 | 2 | off | off | 4/8（未启用） | 0.6750 / 0.7518 / 0.7134 | 0.7750 / 0.8183 / 0.7967 | 3 | 2/8/0 | 0.7207 / 0.7987 / 0.7597 | 1009/1400 | 4394 |
| P2 | 2 | on | off | 4/8（未启用） | 0.6700 / 0.7582 / 0.7141 | 0.7000 / 0.7772 / 0.7386 | 10 | 3/7/0 | 0.6843 / 0.7694 / 0.7269 | 958/1400 | 4802 |
| P3 | 3 | off | off | 4/8（未启用） | 0.6800 / 0.7595 / 0.7198 | 0.7200 / 0.7941 / 0.7570 | 6 | 3/7/0 | 0.6979 / 0.7844 / 0.7412 | 977/1400 | 5257 |
| P5 | 3 | off | on | 2/4 | 0.6800 / 0.7538 / 0.7169 | 0.7250 / 0.7967 / 0.7608 | 10 | 3/7/0 | 0.7121 / 0.7881 / 0.7501 | 997/1400 | 4180 |

P5 同次运行还测试了 initial skill：

```text
initial TEST hard = 0.6786 = 950/1400
best TEST hard    = 0.7121 = 997/1400
hard delta        = +0.0336 = +47 correct items
```

P1/P2/P3 的训练阶段 `eval_test=false`，TEST 来自后续统一 eval-only。P5 的 `eval_test=true`，initial 与 best TEST 在训练进程末尾完成。

### 3.3 SearchQA 训练成本原始记录

| ID | wall 秒 | prompt tokens | completion tokens | total tokens | calls | 计费范围 |
|---|---:|---:|---:|---:|---:|---|
| P1 | 3542 | 7,491,145 | 757,198 | 8,248,343 | 3593 | 训练，不含后测 |
| P2 | 6156 | 11,016,856 | 1,223,666 | 12,240,522 | 5207 | 训练，不含后测 |
| P3 | 2901 | 7,294,033 | 663,545 | 7,957,578 | 3439 | 训练，不含后测 |
| P5 | 1343 | 10,845,165 | 454,079 | 11,299,244 | 6421 | 训练 + initial TEST1400 + best TEST1400 |

### 3.4 SearchQA best-val 接受轨迹

| ID | 接受轨迹 |
|---|---|
| P1 | `1:0.7799, 3:0.7967` |
| P2 | `1:0.7222, 5:0.7342, 10:0.7386` |
| P3 | `1:0.7388, 3:0.7485, 6:0.7570` |
| P5 | `4:0.7308, 5:0.7456, 10:0.7608` |

### 3.5 P2 fallback 原始计数

P2 的 7 个 rejected-root step 均进入 fallback。日志中逐次记录为：

| fallback 次序 | 子节点总数 | subset40 通过数 | full-val 组合是否通过 |
|---:|---:|---:|---|
| 1 | 4 | 4 | 否 |
| 2 | 3 | 0 | 否 |
| 3 | 1 | 0 | 否 |
| 4 | 3 | 1 | 否 |
| 5 | 3 | 3 | 否 |
| 6 | 4 | 0 | 否 |
| 7 | 2 | 2 | 否 |
| 合计 | 20 | 10 | 0 次通过 |

补充原始计数：

```text
fallback 触发次数：7
child subset 评估：20 children × 40 items
subset 通过 children：10
进入 full val200 的组合：4
最终 fallback accept：0
```

### 3.6 P5 clustering 与树形原始计数

| Step | PatchRecords/edits | clusters | kept | dropped | mid nodes | merged edits | action | val mixed |
|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 1 | 11 | 10 | 1 | 9 | 0 | 3 | reject | 0.6885 |
| 2 | 7 | 4 | 3 | 1 | 3 | 7 | reject | 0.7090 |
| 3 | 12 | 9 | 3 | 6 | 3 | 8 | reject | 0.6827 |
| 4 | 9 | 6 | 3 | 3 | 3 | 9 | accept | 0.7308 |
| 5 | 13 | 11 | 2 | 9 | 2 | 2 | accept | 0.7456 |
| 6 | 15 | 12 | 3 | 9 | 3 | 9 | reject | 0.7446 |
| 7 | 14 | 10 | 3 | 7 | 3 | 10 | reject | 0.7346 |
| 8 | 16 | 9 | 4 | 5 | 4 | 15 | reject | 0.7407 |
| 9 | 8 | 7 | 1 | 6 | 0 | 3 | reject | 0.7252 |
| 10 | 12 | 9 | 3 | 6 | 3 | 3 | accept | 0.7608 |
| 合计 | 117 | 87 | 26 | 61 | 24 | 69 | 3 accept | — |

P5 step 1、9 只有一个 kept leaf，因此实际为 `Leaf → Root`。其他 8
步虽然执行了 `Leaf → Mid → Root`，但每一步的 `mid nodes` 都恰好等于
`kept`（3/3、3/3、3/3、2/2、3/3、3/3、4/4、3/3），说明这 24 个
Mid 全部是单 Leaf Mid；本次实验没有观测到多个 Leaf 汇聚成一个 Mid。

### 3.7 SearchQA 原始路径索引

| ID | 启动脚本 | 本地训练日志 | 远端 best skill |
|---|---|---|---|
| P1 | `scripts/runs/ablations/searchqa_fallback_pilot/run_01_d2_fallback_off.sh` | `/Users/bytedance/Documents/codes/Opt/SkillOpt-Tree/logs/searchqa_fallback/searchqa_fallback_pilot_p1_d2_fallback_off_seed42_20260717_175305/searchqa.log` | `/ai-app-vepfs/zhangfuhao/skill/SkillOpt-Tree/outputs/searchqa_fallback_pilot_p1_d2_fallback_off_seed42_20260717_175305/searchqa/best_skill.md` |
| P2 | `scripts/runs/ablations/searchqa_fallback_pilot/run_02_d2_fallback_on.sh` | `/Users/bytedance/Documents/codes/Opt/SkillOpt-Tree/logs/searchqa_fallback/searchqa_fallback_pilot_p2_d2_fallback_on_seed42_20260717_175923/searchqa.log` | `/ai-app-vepfs/zhangfuhao/skill/SkillOpt-Tree/outputs/searchqa_fallback_pilot_p2_d2_fallback_on_seed42_20260717_175923/searchqa/best_skill.md` |
| P3 | `scripts/runs/ablations/searchqa_fallback_pilot/run_03_d3_fallback_off.sh` | `/Users/bytedance/Documents/codes/Opt/SkillOpt-Tree/logs/searchqa_fallback/searchqa_fallback_pilot_p3_d3_fallback_off_seed42_20260717_175939/searchqa.log` | `/ai-app-vepfs/zhangfuhao/skill/SkillOpt-Tree/outputs/searchqa_fallback_pilot_p3_d3_fallback_off_seed42_20260717_175939/searchqa/best_skill.md` |
| P5 | `scripts/runs/ablations/searchqa_fallback_pilot/run_05_d3_clustering_on_tail_off_fallback_off.sh` | `/Users/bytedance/Documents/codes/Opt/SkillOpt-Tree/logs/searchqa_tree_shape_d3_clustering_on_tail_off_fallback_off_seed42_20260718_144046/searchqa.log` | `/ai-app-vepfs/zhangfuhao/skill/SkillOpt-Tree/outputs/searchqa_tree_shape_d3_clustering_on_tail_off_fallback_off_seed42_20260718_144046/searchqa/best_skill.md` |

P1/P2/P3 有效后测：

```text
测试脚本：
/Users/bytedance/Documents/codes/Opt/SkillOpt-Tree/scripts/runs/ablations/searchqa_fallback_pilot/run_test_three_best_skills_h20.sh

本地测试日志目录：
/Users/bytedance/Documents/codes/Opt/SkillOpt-Tree/logs/searchqa_fallback/searchqa_fallback_pilot_three_test_h20_20260718_003135

远端测试输出目录：
/ai-app-vepfs/zhangfuhao/skill/SkillOpt-Tree/outputs/searchqa_fallback_pilot_three_test_h20_20260718_003135
```

P1/P2/P3 后测服务实际参数：

```text
max_model_len=32768
max_num_batched_tokens=65536
max_num_seqs=256
reasoning_parser=qwen3
enable_prefix_caching=true
enable_chunked_prefill=true
test items=1400
```

P5 集成 TEST 输出：

```text
initial:
/ai-app-vepfs/zhangfuhao/skill/SkillOpt-Tree/outputs/searchqa_tree_shape_d3_clustering_on_tail_off_fallback_off_seed42_20260718_144046/searchqa/test_eval_baseline

best:
/ai-app-vepfs/zhangfuhao/skill/SkillOpt-Tree/outputs/searchqa_tree_shape_d3_clustering_on_tail_off_fallback_off_seed42_20260718_144046/searchqa/test_eval
```

P5 训练与集成 TEST 共用的 vLLM 实际参数：

```text
gpu_memory_utilization=0.85
max_model_len=65536
max_num_batched_tokens=32768
max_num_seqs=128
available KV cache memory=68.33 GiB
GPU KV cache size=559680 tokens
```

P5 详细树 artifact 位于远端每个 step：

```text
.../searchqa/steps/step_XXXX/type_guided_v2_patch_records.json
.../searchqa/steps/step_XXXX/type_guided_v2_clustering.json
.../searchqa/steps/step_XXXX/type_guided_v2_leaf_clusters.json
.../searchqa/steps/step_XXXX/type_guided_v2_mid_nodes.json
.../searchqa/steps/step_XXXX/type_guided_v2_root.json
.../searchqa/steps/step_XXXX/type_guided_v2_merge_artifact.json
```

### 3.8 SearchQA 未执行与 dry-run 记录

P4 配置：

```text
ID: P4
depth=3
fallback=true
clustering=false
tail_bank=false
script:
/Users/bytedance/Documents/codes/Opt/SkillOpt-Tree/scripts/runs/ablations/searchqa_fallback_pilot/run_04_d3_fallback_on.sh
```

截至本文件更新时间，P4 没有正式训练结果。以下目录只是命令构造 dry-run，不得计为实验：

```text
/Users/bytedance/Documents/codes/Opt/SkillOpt-Tree/logs/searchqa_fallback_pilot_p4_d3_fallback_on_seed42_20260717_165111
/Users/bytedance/Documents/codes/Opt/SkillOpt-Tree/logs/searchqa_fallback_pilot_p4_d3_fallback_on_seed42_test_default_check
```

同一批本地 dry-run 目录还包括 P1/P2/P3 的 `20260717_165111` 版本；正式记录只采用 `logs/searchqa_fallback/` 下的远端训练日志。

## 4. 原始数据复核入口

优先级从高到低：

1. 每次训练日志第 2 行 `[cmd]`：该次实际参数。
2. 训练日志末尾 Final Summary：steps、accept/reject/skip、best score、wall、tokens、calls。
3. eval-only 日志末尾 `Results`：正式 TEST hard/soft。
4. `best_skill.md`：用于后测的具体 skill。
5. 远端 `summary.json`、`history.json` 和 step artifact：结构化训练明细。

本文件没有复制每条 rollout prediction；逐样本预测应从对应远端测试输出目录的 `results.jsonl` 或 prediction 文件读取，日志路径和输出路径已在上文逐项保留。
