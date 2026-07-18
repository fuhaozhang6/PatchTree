# PatchTree 资源分配与训练注意事项

最后更新：2026-07-18。本文包含 2026-07-16 吞吐压测结论、资源分配方案、
正式训练参数基线、数据集专项配置、冒烟测试判定规则和故障排查记录。

本文汇总 2026-07-16 的两轮 OpenAI-compatible API 吞吐压测，用于后续
PatchTree 训练的 GPU、vLLM、云端优化器和进程并发分配。结论只适用于本次
测试使用的模型、账号配额、vLLM 版本和请求长度分布；供应商配额或模型版本
变化后应重新压测。

本文是当前训练脚本的统一操作基线。新建或复制脚本前应先核对第 8–15 节，
不能只参考并发数字。

## 1. 可直接采用的结论

| 资源 | 最快稳定点 | 日常建议 | 不建议 | 结论 |
|---|---:|---:|---:|---|
| 单张 L20，本地 Qwen3.5-4B/vLLM | 128 | 总在途 96–128 | 192、256 | `--max-num-seqs=128`；追求最短总耗时使用 128，96 是低延迟保守点 |
| DeepSeek 官网 V4 Pro | 256 | 192–256（账号总并发） | 384 | 256 全成功且吞吐最高；384 出现超时和吞吐坍塌 |
| 火山 Ark V4 Pro | 128 曾稳定，但跨轮波动很大 | 长任务先按 64，监控后再升至 128 | 192 | 第二轮 192 触发 TPM 429；不宜作为大规模同步消融的唯一优化器源 |

资源分配原则：

1. 每张 GPU 只启动一个 vLLM，多训练进程共享同一端点。
2. 单 L20 的所有训练进程合计在途请求尽量落在 96–128，不要追求 192。
3. DeepSeek 官网优先承担需要可比性的消融实验；同一组消融不要混用官网和
   Ark，以免服务端模型版本及随机性成为额外变量。
4. 云端并发按同一账号下所有机器的总和计算，不按单脚本分别计算。
5. `API_MAX_CONCURRENCY` 在当前 shell 启动器中只是参数合法性检查，并不是
   跨进程或跨机器的全局限流器；正式大规模运行仍需按 worker 总量人工规划。

### 实验 TEST 默认规则

1. 除纯 smoke、吞吐压测或明确标记为“仅训练”的临时运行外，实验脚本必须默认
   `EVAL_TEST=true`。不能因为完整 test 较慢而静默关闭。
2. 若为节省训练墙钟时间而显式设置 `EVAL_TEST=false`，实验在补完 held-out
   test 前统一标记为 `test_pending`，不得只根据 train/val 指标下结论。
3. 补测应使用 `scripts/cli/eval_only.py`，评估最终 `best_skill.md`，并保留
   `eval_summary.json`、逐样本 `results.jsonl`、汇总表和运行日志。
4. 同组消融的 test 必须使用相同模型、temperature、thinking、数据全集和
   scoring 配置；可以改变 GPU 型号与并发以缩短墙钟时间，但不能改变样本集合。
5. 启动前同时核对脚本打印值和最终 `[cmd]`。`TEST_ENV_NUM=0` 表示完整 test，
   不是关闭 test；真正的开关是 `EVAL_TEST`。

## 2. 测试条件

- 请求输入长度选项：约 512、1024、2048、4096 tokens，均值 1920。
- 请求输出上限选项：256、512、1024、2048 tokens，均值 960。
- 每个并发档的请求数：第一轮通常为 `2 × concurrency`；第二轮高并发复测
  使用约 `1 × concurrency`。
- thinking：关闭。
- 稳定判定：成功率至少 99%。
- 本地模型：`Qwen/Qwen3.5-4B`，单 L20，BF16，tensor parallel 1。
- 本地 vLLM：0.19.1，`max_model_len=65536`、prefix cache 开启、chunked
  prefill 开启、`max_num_batched_tokens=65536`、GPU memory utilization 0.90。
- 本地服务压测时 `max_num_seqs=256`，以便观察 128 之后的退化区间；这不代表
  正式训练也应设置为 256。

原始记录：

- [第一轮总表](../outputs/throughput_v4pro_l20_20260716_121035/comparison.md)
- [第二轮总表](../outputs/throughput_v4pro_l20_20260716_141506/comparison.md)
- [L20/vLLM 明细](../outputs/throughput_v4pro_l20_20260716_121035/local_l20_vllm/report.md)
- [DeepSeek 官网第一轮](../outputs/throughput_v4pro_l20_20260716_121035/deepseek_official/report.md)
- [DeepSeek 官网第二轮](../outputs/throughput_v4pro_l20_20260716_141506/deepseek_official/report.md)
- [Ark 第一轮](../outputs/throughput_v4pro_l20_20260716_121035/ark/report.md)
- [Ark 第二轮](../outputs/throughput_v4pro_l20_20260716_141506/ark/report.md)

## 3. 单 L20 本地 Qwen/vLLM

| 并发 | req/s | 总 tok/s | TTFT p95 | 延迟 p95 | 相对峰值吞吐 |
|---:|---:|---:|---:|---:|---:|
| 32 | 0.921 | 2716 | 5.37s | 54.36s | 58.3% |
| 64 | 1.248 | 3683 | 11.04s | 85.78s | 79.1% |
| 96 | 1.450 | 4280 | 16.42s | 113.91s | 91.9% |
| 128 | **1.578** | **4657** | 22.06s | 142.01s | **100%** |
| 192 | 1.489 | 4394 | 32.82s | 227.99s | 94.3% |
| 256 | 1.459 | 4306 | 93.42s | 268.16s | 92.4% |

从 96 提升到 128，req/s 只增加约 8.8%，但仍是完成固定总请求数最快的点。
从 128 增加到 192 后，吞吐下降约 5.7%，TTFT 和尾延迟明显恶化；256 更差。
因此：

- 追求最短训练墙钟时间：`VLLM_MAX_NUM_SEQS=128`，客户端合计并发尽量靠近
  128，但不应超过。
- 更看重单请求延迟或长输出稳定性：客户端合计并发采用 96。
- 不要把“256 请求全部成功”误解成“256 更快”；成功率稳定但队列已经拥塞。

vLLM 日志显示本次服务约有 192K token KV cache。压测在 192/256 并发时 KV
cache 接近满载并出现大量等待请求，这与吞吐回落相符。若实际训练的平均输出
显著长于本次压测，安全并发应从 96 起步，而不是直接使用 128。

## 4. DeepSeek 官网 V4 Pro

高并发复测结果：

| 并发 | 成功率 | req/s | 总 tok/s | TTFT p95 | 延迟 p95 | 判断 |
|---:|---:|---:|---:|---:|---:|---|
| 128 | 100% | 2.68 | 7878 | 1.11s | 42.79s | 稳定 |
| 192 | 100% | 3.89 | 11442 | 1.72s | 43.26s | 稳定 |
| 256 | 100% | **5.29** | **15536** | 1.42s | 44.01s | 最快稳定点 |
| 384 | 99.0% | 0.62 | 1840 | 3.73s | 45.79s | 4 个传输超时，吞吐坍塌 |

第一轮 128 并发曾达到 4.52 req/s，第二轮同档为 2.68 req/s，说明云服务吞吐
存在时段波动；但两轮 128 均为 100% 成功，高并发复测中的 192 和 256 也均
100% 成功。后续分配时：

- 最快模式：同一账号合计不超过 256。
- 留余量模式：同一账号合计约 192–224。
- 不使用 384；该档不是“稍慢”，而是出现排队/超时后的整体坍塌。

## 5. 火山 Ark V4 Pro

第一轮从 8 到 128 均为 100% 成功，128 达到 3.76 req/s；但第二轮出现明显
账号级限流：

| 轮次 | 并发 | 成功率 | req/s | 主要现象 |
|---|---:|---:|---:|---|
| 第一轮 | 128 | 100% | 3.76 | 当时稳定且吞吐最高 |
| 第二轮 | 128 | 100% | 0.43 | 无失败，但墙钟时间约 300s，疑似服务端节流/排队 |
| 第二轮 | 192 | 75.5% | 0.47 | 47 个 HTTP 429，明确命中账号 TPM 限额 |

因此 Ark 的上限不是纯并发上限，还受账号 TPM、时段和共享配额影响。建议：

- 长时间训练先以账号总并发 64 启动，观察 429、TTFT 和分钟吞吐。
- 确认无 TPM 限流后可逐步升至 96 或 128。
- 不使用 192。
- 若与 DeepSeek 官网同时可用，Ark 更适合承接独立数据集任务，不用于同一组
  需要严格可比的消融实验。

## 6. LiveMath core-8 四张 L20 的资源审计

LiveMath 的 repeated rollout 会被扁平为单批请求，因此每个实验单步的目标侧
峰值任务数为 `batch_size × rollout_repeats`。训练集共 35 条，4 epochs。

| 脚本 | 并发实验 | 单步 target 峰值 | 每 epoch 步数 | 4 epoch target 请求量 | 每卡合计峰值 |
|---|---|---:|---:|---:|---:|
| `run_01_r1_r4.sh` | r1/b8/d2 + r4/b8/d2 | 8 + 32 | 5 + 5 | 140 + 560 | 40 |
| `run_02_r2_base.sh` | r2/b8/d2 + r3/b8/d2 | 16 + 24 | 5 + 5 | 280 + 420 | 40 |
| `run_03_b4_b32.sh` | r3/b4/d2 + r3/b32/d2 | 12 + 96 | 9 + 2 | 420 + 420 | **108** |
| `run_04_b16_d3.sh` | r3/b16/d2 + r3/b8/d3 | 48 + 24 | 3 + 5 | 420 + 420 | 72 |

配对方式无需调整：采样最重的 r4 与最轻的 r1 配对；步数最多的 b4 与步数最少
的 b32 配对；depth=3 与较少步数的 b16 配对。四张卡的 4-epoch 目标请求量为
700、700、840、840，已经比较均衡，同时每卡峰值都低于本地最佳点 128。

本次审计发现并修正了两个客户端瓶颈：

1. 原先每实验只有 48 个 target workers。`b32 × r3` 会产生 96 个扁平任务，
   因而必须分两轮；现在改为 96，使其可一次提交。最重配对的总峰值为
   `96 + 12 = 108`，仍低于 vLLM 128。
2. 原先每实验只有 16 个 PatchRecord workers。单个 b32 步骤最多产生 32 个
   candidate records，可能分两轮；现在改为 32。

当四个脚本同时运行时，PatchRecord 阶段的理论最大官网并发约为：

`(8+8) + (8+8) + (4+32) + (16+8) = 92`

这低于 DeepSeek 官网验证过的 256 稳定点。叶节点合并每实验最多 8 并发，八个
实验合计最多 64，也低于官网上限。因此这组实验不会被官网 API 并发上限卡住；
真正的长尾主要来自 b4 的 36 个训练步骤、depth=3 的额外融合以及模型输出长度。

## 7. 后续分配速查

| 场景 | 单 L20 训练进程数 | vLLM max seqs | 客户端合计目标并发 | 云端建议 |
|---|---:|---:|---:|---|
| 单个大批量数据集 | 1 | 128 | 96–128 | 官网 64–128，按实际 analyst 数量 |
| 两个中小数据集共享 L20 | 2 | 128 | 合计 96–128 | 官网账号总量不超过 256 |
| LiveMath core-8 当前方案 | 2 | 128 | 每卡自然峰值 40/40/108/72 | 8 实验全用官网，理论峰值约 92 |
| Ark 长任务 | 1–2 | 128 | 由本地 L20 决定 | Ark 账号先 64，确认无 429 后逐级增加 |

### 7.1 当前四张 L20 正式资源池配置

当前 `scripts/runs/resource_pool_4x_l20` 的分配已经按上述压测结果收束：

| GPU 任务组 | 单卡训练进程 | batch/repeats | target workers 合计 | vLLM max seqs | optimizer analyst 合计 | 判断 |
|---|---:|---:|---:|---:|---:|---|
| SearchQA / DeepSeek 官网 | 1 | 40/3 | 128；单步自然峰值约 120 | 128 | 48 | 接近本地最快点 |
| ALFWorld / DeepSeek 官网 | 1 | 8/3 | 24 | 96 | 16 | 长序列环境，主动保守 |
| DocVQA + SpreadsheetBench / Ark | 2 | 各 8/3 | 48+48=96 | 128 | 16+16=32 | 为图像和长代码任务留 32 条余量 |
| LiveMath + OfficeQA / Ark | 2 | 各 8/3 | 48+48=96 | 128 | 16+16=32 | 利用充分且不超过本地稳定区 |

四张卡同时运行时：

- 两个 DeepSeek 官网脚本的 analyst 上限合计约 64。
- 两个 Ark 脚本的 analyst 上限合计约 64。
- 每张卡只有一个 vLLM，成对数据集只是共享 endpoint，不会加载两份
  Qwen3.5-4B。
- 两数据集正式训练阶段的自然 target 峰值通常为
  `2 × batch_size × repeats = 48`；完整 selection/test 阶段最多由两个
  48-worker 进程共同形成约 96 个在途请求。
- `max_num_seqs=128` 留出的 32 条余量对 DocVQA 图像输入、SpreadsheetBench
  长输出和 OfficeQA 多轮工具调用是必要的，不建议为了“看起来满载”把 workers
  提到每进程 64。

该配置的目标是最短整体墙钟时间，不要求每个阶段都持续保持 128 个 running
request。PatchRecord、树融合、gate 和环境执行阶段本来就不全是 target 模型
密集型任务；只提高 workers 无法消除这些串并行阶段，反而可能增加长尾和显存
压力。

### 7.1.1 全量类型轨迹调查的四张 H20 配置

`scripts/runs/analysis/observed_taxonomy_h20` 使用四张 H20，但仍沿用单 L20
压测得到的 96–128 保守区间，不以更大的显存为理由直接使用 192/256 并发。
四个任务全部使用 DeepSeek 官网，避免同一 taxonomy 混入不同服务端版本。

| H20 任务 | target workers 合计 | vLLM max seqs | DeepSeek analyst 上限 |
|---|---:|---:|---:|
| SearchQA shard 0 | 128 | 128 | 64 |
| SearchQA shard 1 | 128 | 128 | 64 |
| ALFWorld | 24（单批自然峰值 8） | 96 | 16 |
| Spreadsheet + OfficeQA + DocVQA + LiveMath | 24+16+28+28=96 | 128 | 16+16+24+24=80 |

四张卡同时运行时，DeepSeek 官网理论 analyst 总上限为 `224`，位于已验证的
192–256 稳定区间内；每张本地 endpoint 的客户端在途上限不超过 128。两个
SearchQA 分片必须使用同一个官网模型，不能为了分流把其中一个临时改到 Ark。

### 7.2 冒烟脚本资源语义

冒烟脚本每数据集使用 `batch_size=2`、`repeats=2`，单数据集一个训练 rollout
最多产生 4 个自然 target 请求；两个数据集共享一张 L20 时合计约 8 个。虽然
仍以 `max_num_seqs=128` 启动 vLLM，但这不是为了压满 GPU，而是为了复用正式
训练完全相同的服务配置并验证一个完整 epoch。

因此冒烟脚本的 `workers=8`、`analyst_workers=4` 不应提高。冒烟的判定目标是：
数据加载、模型调用、PatchRecord、融合、gate 和结果落盘全部正确，而不是测
吞吐或 GPU 利用率。历史真实 smoke 未出现 CUDA OOM；最重的
DocVQA+SpreadsheetBench 日志中观察到的 KV cache 峰值约为 8.5%。

正式运行后应同时观察：vLLM 的 Running/Waiting、KV cache usage、GPU utilization、
请求 p95 延迟、云端 429/timeout，以及每个 PatchTree 阶段的 timing。若本地
Waiting 长期为 0 且 GPU utilization 偏低，说明任务本身并发不足；若 Waiting
持续增长且吞吐不再上升，应降低客户端并发，而不是继续增加 worker。

## 8. 正式训练的通用参数基线

下面是当前 DeepSeek 优化器 + 本地 Qwen3.5/vLLM 组合的推荐基线。新建脚本时
应从这组参数开始，只对数据集确实需要的部分做覆盖。

| 参数 | 推荐值 | 原因与注意事项 |
|---|---:|---|
| `DEEPSEEK_THINKING` | `disabled` | 关闭优化器额外思考，保持吞吐、成本和消融条件一致 |
| `REASONING_EFFORT` | 空字符串 | 覆盖部分数据集 YAML 中遗留的 `medium`，避免向 DeepSeek 发送额外 reasoning 参数 |
| `QWEN_CHAT_ENABLE_THINKING` | `false` | 关闭共享 Qwen thinking 默认值 |
| `TARGET_QWEN_CHAT_ENABLE_THINKING` | `false` | 明确关闭 target 角色 thinking；最终请求会带 `chat_template_kwargs.enable_thinking=false` |
| `TARGET_QWEN_CHAT_TEMPERATURE` | `0.2` | 降低训练 rollout 方差，同时保留少量非确定性 |
| `QWEN_CHAT_TIMEOUT_SECONDS` | `300` | 本地 128 并发压测 latency p95 已达到约 142 秒，不能使用很短的网络超时 |
| `TARGET_QWEN_CHAT_TIMEOUT_SECONDS` | `300` | 与 target Qwen 后端保持一致；长工具任务可提高到 600 |
| `TARGET_QWEN_CHAT_ROLLOUT_RETRIES` | `2` | 本地服务失败时允许一次重试；避免默认 5 次把单题拖到外层 watchdog 之后 |
| `QWEN_CHAT_MAX_TOKENS` | `16384` | SearchQA、DocVQA、LiveMath、OfficeQA 的通用安全上限 |
| `TARGET_QWEN_CHAT_MAX_TOKENS` | `16384` | 必须与环境的 completion 上限协调，避免后端先截断 |
| `VLLM_MAX_NUM_SEQS` | `128` | 单 L20 压测的最快稳定点 |
| `VLLM_MAX_NUM_BATCHED_TOKENS` | `65536` | 与本轮 L20 压测条件一致 |
| `MAX_MODEL_LEN` | `65536` | 通用长上下文配置；ALFWorld 可用 32768 |
| `GPU_MEMORY_UTILIZATION` | `0.90` | 已验证的安全值，不要未经压测直接提高以追求 KV cache |
| `VLLM_ENABLE_CHUNKED_PREFILL` | `1` | 长输入和混合长度请求必须保留 |
| `VLLM_TOOL_CALL_PARSER` | `qwen3_coder` | Qwen3.5 工具调用解析器；不要沿用 Qwen2.5 常见的 `hermes` |
| `VLLM_REASONING_PARSER` | `qwen3` | Qwen3.5 服务端输出解析器；设置它不会自动开启 thinking |
| `TARGET_MAX_COMPLETION_TOKENS` | `16384` | 通用环境上限；ALFWorld 当前推荐 2048 |

推荐 shell 基线：

```bash
export DEEPSEEK_THINKING=disabled
export REASONING_EFFORT=''
export QWEN_CHAT_ENABLE_THINKING=false
export TARGET_QWEN_CHAT_ENABLE_THINKING=false
export TARGET_QWEN_CHAT_TEMPERATURE=0.2

export QWEN_CHAT_TIMEOUT_SECONDS=300
export TARGET_QWEN_CHAT_TIMEOUT_SECONDS=300
export TARGET_QWEN_CHAT_ROLLOUT_RETRIES=2
export QWEN_CHAT_MAX_TOKENS=16384
export TARGET_QWEN_CHAT_MAX_TOKENS=16384
export TARGET_MAX_COMPLETION_TOKENS=16384

export VLLM_MAX_NUM_SEQS=128
export VLLM_MAX_NUM_BATCHED_TOKENS=65536
export MAX_MODEL_LEN=65536
export GPU_MEMORY_UTILIZATION=0.90
export VLLM_ENABLE_CHUNKED_PREFILL=1
export VLLM_TOOL_CALL_PARSER=qwen3_coder
export VLLM_REASONING_PARSER=qwen3
```

注意两种“思考”必须分别关闭：`DEEPSEEK_THINKING` 控制云端优化器，
`TARGET_QWEN_CHAT_ENABLE_THINKING` 控制本地 target。只关闭其中一个不够。
当前资源池公共脚本已经默认将 DeepSeek thinking 设为 `disabled`，但部分旧的
16-epoch 脚本仍写过 `enabled`；复制旧脚本时必须重新检查。

## 9. 超时分层：不要只改一个 timeout

一次 rollout 至少涉及以下几层时间限制。外层比内层短时，即使底层模型仍在
正常生成，任务也会被外层提前标记失败。

| 层级 | 参数或位置 | 建议 |
|---|---|---|
| vLLM 冷启动 | `VLLM_WAIT_SECONDS` | 已缓存模型至少 600 秒；新机器、慢挂载或首次编译使用 1200 秒 |
| Qwen HTTP 后端 | `TARGET_QWEN_CHAT_TIMEOUT_SECONDS` | 通用 300 秒；工具/代码任务可用 600 秒 |
| 环境单次模型调用 | YAML `env.exec_timeout` 或 SpreadsheetBench `env.llm_timeout` | 高并发通用至少 300 秒 |
| 单题完整任务 | SpreadsheetBench `env.exec_timeout` | multi 模式使用 1200 秒，必须覆盖多轮生成与执行 |
| batch 任务看门狗 | rollout 中的 `task_timeout` | 必须大于内部完整任务；SpreadsheetBench 当前为 1200+300=1500 秒 |
| OfficeQA 单次搜索 | `env.search_timeout_seconds` | 20 秒只控制一次搜索工具调用，不等于整题 timeout |

尤其要注意：SearchQA 和 DocVQA 曾经默认使用 120 秒，而单 L20 在 128 并发下
测得的 latency p95 约为 142 秒。环境会把 `exec_timeout` 作为显式请求 timeout
传给 Qwen，它会覆盖后端配置的 300 秒。当前默认配置和资源池脚本均已改为：

```yaml
env:
  exec_timeout: 300
```

SpreadsheetBench 的 multi 模式在单 L20/Qwen3.5-4B 上实测出现 472–546 秒才完成
一条任务，另外三条在 600 秒被看门狗提前标记 timeout，因此当前训练和 smoke
基线使用单次模型请求 300 秒、单题完整任务 1200 秒、外层 batch watchdog
1500 秒。SearchQA、DocVQA、LiveMath 的单次请求为 300 秒，外层单题 watchdog
为 660 秒，可容纳一次重试。不要把 `/models` 健康检查所用的 5 秒 timeout
当成生成 timeout；健康检查短是正常的。

出现 timeout 时先区分：

1. vLLM 尚未 ready：查看 `vllm_qwen.log` 和启动等待时间。
2. 单请求生成超时：查看 prediction 的 `fail_reason`，提高 `env.exec_timeout`。
3. batch task timeout：检查是否大量请求在 vLLM Waiting 队列，必要时降低总并发。
4. 云端 optimizer 超时或 429：按账号总并发降载，不能只提高本地 timeout。

## 10. 常见参数语义和运行纪律

### 10.1 数值 0 通常不是“关闭”

| 参数 | `0` 的实际含义 | 正确关闭方式 |
|---|---|---|
| `TRAIN_SIZE=0` | 使用 split 中完整训练集 | 用正数限制训练量 |
| `LIMIT=0` | 不限数据，使用完整 split | 用正数做 smoke test |
| `SEL_ENV_NUM=0` | 使用完整 validation split | 不建议关闭 gate；需缩小则设正数 |
| `TEST_ENV_NUM=0` | 使用完整 test split | 用 `EVAL_TEST=false` 关闭 test evaluation |
| `TYPE_GUIDED_FALLBACK_TOP_K=0` | 不截断候选，可能评估所有 child | 用 `TYPE_GUIDED_LEAF_FALLBACK=false` 关闭 fallback |
| `TYPE_GUIDED_FALLBACK_SEL_ENV_NUM=0` | 复用完整 selection split | 设正数限制 fallback 评估样本 |

因此 `TEST_ENV_NUM=0 EVAL_TEST=true` 会跑完整测试集，并不是“不跑测试”。
消融实验也默认使用该组合。只有明确的 smoke 或仅训练运行才设置
`EVAL_TEST=false`，并必须标记为 `test_pending`、随后用 `eval_only.py` 补测。

### 10.2 输出目录和缓存必须与参数绑定

多个环境的 rollout 都支持从 `results.jsonl` 续跑，已经出现的 ID 会被直接跳过；
PatchTree 还会使用 `type_guided_cache`。修改模型、thinking、tools、数据路径、
timeout 或超参数后，不要继续使用被旧配置写过的输出目录，否则旧的失败结果和
PatchRecord 可能被当成有效缓存复用。

- 每次新配置使用新的 `TS`、`OUT_BASE` 和 `LOG_DIR`。
- 只有完全相同的模型、数据和参数才能续跑同一输出目录。
- 错误运行的目录应归档后换新目录，不要直接在原目录上重试并期待重新计算。
- 两个训练进程必须使用不同的输出子目录，但可以共享同一个 Qwen endpoint。

### 10.3 vLLM 启动规则

- 一张 GPU 只能由一个脚本负责启动 vLLM；同卡第二个训练进程设置
  `START_VLLM=0` 并复用相同 `QWEN_CHAT_BASE_URL`。
- 同一台机器的多个 vLLM 必须使用不同端口；不同资源池机器可以使用相同端口。
- 当前 vLLM 版本已经移除旧参数 `--disable-log-requests`，不要再复制到新脚本。
- Qwen3.5 的 OfficeQA 工具模式必须带
  `--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3`。
  `hermes` 不适用于本项目当前的 Qwen3.5；它会在 vLLM 日志中产生
  `hermes_tool_parser.py` / `JSONDecodeError`，即使客户端文本兜底偶尔仍能回答。
  参考 [Qwen3.5-4B 官方 vLLM Tool Call 启动示例](https://huggingface.co/Qwen/Qwen3.5-4B)。
- 工具启动 smoke 必须得到结构化 `message.tool_calls`。只看到正文中的
  `<tool_call>` 不能证明 vLLM parser 正常，不应绕过后继续正式训练。
- 普通数据集不依赖工具解析；多个数据集共享同一个 vLLM 时可以保留上述参数。
- dry-run 只证明命令能够生成，不证明模型路径、GPU、文档目录和真实 API 可用。

### 10.4 API 来源不能被遗留环境变量污染

DeepSeek 官网应明确设置 endpoint `https://api.deepseek.com` 和模型
`deepseek-v4-pro`；Ark 应明确设置自己的 endpoint 和
`deepseek-v4-pro-260425`。不要只改模型名却沿用上一任务留下的
`AZURE_OPENAI_ENDPOINT` 或 key。同一组消融固定使用一个来源。

## 11. 各数据集参数与高频故障

| 数据集 | 必须保留的设置 | 高频错误与后果 |
|---|---|---|
| LiveMath | `max_turns=1`、`exec_timeout=300`、`shuffle_choices=true`、`use_theorem=false`、`use_sketch=false` | split 未物化出 question/choices/correct_choice；高并发时 timeout 过低 |
| SearchQA | `max_turns=1`、`exec_timeout=300`、完整 `searchqa_split` | 旧默认 120 秒在 L20 满载时偏低；完整 val/test 很慢，smoke test 应显式限制样本 |
| DocVQA | vision-capable Qwen、`data/docvqa/splits`、`image_detail=auto`、`exec_timeout=300` | 错用仅含 ID 的 split、图片路径不存在或使用纯文本模型会导致空答/全 0 |
| OfficeQA | `use_local_tools=true`、`search_mode=offline`、官方 docs 目录、`max_tool_turns=24`、`qwen3_coder` tool parser、`qwen3` reasoning parser | 关闭 tools、文档目录错误、使用 `hermes` 或工具解析失败时常出现整批 hard=0 |
| SpreadsheetBench | 正确 `data_root`、`max_turns=30`、multi 模式 `exec_timeout=1200`、确认 Adapter 接收 `limit` | 把 max_turns 改成 1 会破坏多步代码任务；600 秒会产生假 timeout；只看 CLI 有 `--limit` 而不看 DataLoader 计数可能误跑完整训练集 |
| ALFWorld | `max_steps=50`、较小的环境/API worker、`MAX_MODEL_LEN=32768`、completion 2048 | 任务是长序列交互，盲目提高并发或缩短 step budget 会增加失败而非提速 |

DocVQA 使用的是物化后的 `data/docvqa/splits`，不是只保存样本 ID 的 manifest。
运行前至少抽查一个 split item 的图片路径确实存在。SpreadsheetBench 同时需要
split 和 `data/spreadsheetbench_verified_400` 数据根目录，二者缺一不可。
SpreadsheetBench 启动后还必须检查日志中的实际计数；冒烟配置应出现
`[SpreadsheetBenchDataLoader] train=2 val=2 test=2` 和
`steps/epoch=1`，不能只凭最终命令包含 `--limit 2` 判断限制已生效。

## 12. OfficeQA 全 0 专项排查

OfficeQA 的数值/表格问题依赖 Treasury bulletin 文档检索。若连续一批样本全部
`hard=0`，尤其 `predicted_answer` 为空，不应继续跑完整训练，应立即按以下顺序
排查。

### 12.1 正确基线

```bash
export OFFICEQA_USE_LOCAL_TOOLS=true
export OFFICEQA_SEARCH_MODE=offline
export OFFICEQA_DOCS_DIR=/path/to/data/officeqa_docs_official
export VLLM_ENABLE_AUTO_TOOL_CHOICE=auto
export VLLM_TOOL_CALL_PARSER=qwen3_coder
export VLLM_REASONING_PARSER=qwen3
export TARGET_QWEN_CHAT_ENABLE_THINKING=false
```

同时确认最终 `cfg-options` 中存在：

```text
env.data_dirs=<officeqa_docs_official>
env.use_local_tools=true
env.search_mode=offline
```

保留 YAML 中的 `env.max_tool_turns=24`。OfficeQA 的关键参数是
`max_tool_turns`，不要用其他数据集常见的 `--max_turns 1` 替代它。

### 12.2 启动前检查

1. `OFFICEQA_SPLIT_DIR/{train,val,test}` 都存在且包含 items 文件。
2. `OFFICEQA_DOCS_DIR` 存在并包含完整文档；当前官方目录约有 855 个文件，
   不能把只有 split JSON 的目录当作文档目录。
3. vLLM 启动命令包含 auto tool choice、`qwen3_coder` tool parser 和
   `qwen3` reasoning parser，且日志中没有 Hermes parser traceback。
4. 启动器的 Qwen tool smoke test 必须观察到 structured tool call；只有文本
   `<tool_call>` 说明客户端兜底可能生效，但服务端 parser 仍未验证通过。
5. 先用新的输出目录跑少量样本，确认 conversation 中出现 tool call 和工具结果。

当前 `OfficeQAAdapter.setup()` 还会抽样解析 train/val/test 的
`source_files/source_docs`。目录存在但样本证据一个都解析不到时，训练会在首个
rollout 前直接失败，并打印 `[OfficeQA preflight]`，避免静默跑出整批 0。
运行中若 vLLM auto tool choice 生成了不合法的参数 JSON，错误会作为 tool result
返回给模型下一轮修正，不再直接把整条样本异常终止。

### 12.3 从结果判断故障位置

检查 `results.jsonl` 和单样本 `conversation.json`：

| 现象 | 重点字段 | 常见原因 |
|---|---|---|
| `predicted_answer` 为空 | `fail_reason`、`last_finish_reason`、`n_turns` | 模型没有返回最终答案、tool parser 失败或输出被截断 |
| 没有任何 tool call | `use_local_tools`、conversation | tools 被关闭、vLLM 未启用 parser、模型未收到 tool schema |
| 找不到文档 | `resolved_source_paths`、`oracle_parsed_pages_included` | docs 路径错误、source file 无法解析 |
| 有答案但全部 EM=0 | `predicted_answer` 与 `ground_truth` | 检索/数值提取质量问题，或答案包含额外文本导致 exact match 失败 |
| 修正参数后仍复现旧结果 | 输出目录中的既有 `results.jsonl` | resume 机制跳过已完成 ID；必须换新 `OUT_BASE` |

区分两类“全 0”：如果答案为空、没有工具调用或路径无法解析，这是运行配置故障；
如果答案非空、工具证据正常但数值错误，则属于模型/skill 质量问题，才应进入
PatchTree 优化。

## 13. 一轮完整 epoch 冒烟测试：判定规则与已发现回归

2026-07-17 使用 2 条训练样本、batch size 2、1 epoch 对 SearchQA、DocVQA、
SpreadsheetBench、LiveMath 和 OfficeQA 做了真实 smoke。ALFWorld 已由独立脚本
验证。该轮结果说明：分数为 0、训练失败和脚本未完成是三种不同状态，不能只看
`best_score`。

### 13.1 最小完整 epoch 的正确配置

```bash
export NUM_EPOCHS=1
export TRAIN_SIZE=0
export LIMIT=2
export BATCH_SIZE=2
export EVAL_TEST=false
```

`TRAIN_SIZE=0` 表示使用 DataLoader 实际加载的训练 split；`LIMIT=2` 应先把
train/val/test 各截为 2 条，因此训练器解析出的 `train_size` 才是 2，最终得到
一个完整 epoch、一个 step。单独设置 `TRAIN_SIZE=2` 不能代替 `LIMIT=2`，因为
训练器要求显式 train size 与 DataLoader 实际加载的 split 大小一致；在
`LIMIT=2` 已生效时再写 `TRAIN_SIZE=2` 只是冗余。

启动后必须在日志中看到：

```text
[<Dataset>DataLoader] train=2 val=2 test=2
[config] train_size=2
[config] steps/epoch=1 ... total_steps=1
[STEP 1/1]
```

结束时还必须同时具备：

- 训练日志包含 `[exit] 0`。
- `completed.tsv` 中存在该数据集且退出码为 0。
- 输出目录存在 `summary.json`，其中 `total_steps=1` 且只包含 epoch 1。
- `steps/step_0001/rollout/results.jsonl` 存在并包含真实模型结果。

仅有启动命令、部分 step 日志或另一个并发数据集已经完成，都不代表该数据集
通过。若 `completed.tsv` 没有该数据集且日志没有 `[exit]`，应判断为仍在运行或
被中断。

### 13.2 如何解释 smoke 中的 0

本轮 selection 只有 1 条题，因此 baseline/gate 分数只能是 0 或 1，统计意义
很弱。冒烟测试的目标是验证数据、模型调用、PatchRecord、树融合和 gate 链路，
不是评估准确率。

| 现象 | 判定 | 处理 |
|---|---|---|
| `hard=0`，但 `agent_ok=true`、答案非空、证据和输入正常 | 正常答错 | smoke 可通过；不要据此判断代码错误 |
| `best_score=0`，训练 rollout 有成功样本且 epoch 完成 | 单条 validation 未答对 | smoke 可通过；正式效果需更大验证集 |
| `skip_no_patches`，训练样本均为 stable success | 正常无候选 patch | 训练流程已完成，不是 optimizer 没启动 |
| `hard=0` 且答案为空、`agent_ok=false`、timeout、HTTP error | 运行故障 | smoke 必须失败并先修配置 |
| OfficeQA 有答案但无工具调用/文档证据 | 工具链故障 | 检查 parser、tools、docs 路径 |
| SpreadsheetBench `phase=timeout/error` | 基础设施或时限故障 | 不应当作普通答错；提高 task timeout 或降低负载 |

如果需要做精度 sanity check，应另开一组 8–32 条样本的测试，并扩大 selection；
不要为了让最小 smoke 的数字“好看”而挑选简单样本。

### 13.3 2026-07-17 实测结果

| 数据集 | 实际结果 | 结论 |
|---|---|---|
| DocVQA | baseline=1，训练 4/4 rollout 正确，stable success，无 patch | 训练链路通过 |
| SearchQA | baseline=0，但训练 4/4 rollout 正确，stable success | 训练链路通过；validation 的 0 是单题答错 |
| LiveMath | baseline=0，产生 patch，candidate validation=1，ACCEPT | 完整 PatchTree 更新链路通过 |
| OfficeQA | baseline=0，训练 2/4 正确，产生 patch 后 gate REJECT | 主训练链路完成，但 vLLM Hermes parser 报错，修正后需复测 |
| SpreadsheetBench | 实际加载 train=80、steps/epoch=40；600 秒出现 3/4 timeout，日志停在 step 2 | smoke 未完成，不能判定通过 |

### 13.4 本轮发现并修复的代码/配置错误

1. **SpreadsheetBench 的 `LIMIT` 未生效。**
   [SpreadsheetBenchAdapter](../skillopt/envs/spreadsheetbench/adapter.py) 构造函数
   没有接收并转发 `limit`，导致 CLI 虽然显示 `--limit 2`，DataLoader 仍加载
   80 条训练数据并生成 40 steps。已补齐参数并增加回归测试。
2. **SpreadsheetBench 600 秒任务看门狗过短。** 单条 multi-turn 任务实测需要
   472–546 秒，多个任务在 602 秒被外层统一标记 timeout。当前 smoke 基线调整为
   1200 秒；timeout 结果在验证器中按基础设施失败处理，不能作为普通 hard=0。
3. **OfficeQA 使用了错误的 Hermes parser。** Qwen3.5 应使用
   `qwen3_coder` tool parser 和 `qwen3` reasoning parser。Hermes 产生连续
   `JSONDecodeError`；客户端的文本 tool-call 兜底可能掩盖这个错误。当前启动
   smoke 要求结构化 tool call，否则禁止开始正式训练。
4. **SpreadsheetBench smoke 验证字段假设错误。** codegen 结果主要使用
   `llm_ok`、`phase`、`code_ok`，不保证有其他 QA 环境的 `agent_ok` 和文本答案
   字段。验证器已按数据集结果结构检查，同时显式拒绝 timeout/error。
5. **vLLM 的 `Unknown environment variable` 警告不等于 CLI 参数失效。** 本项目
   会把部分启动器变量导出到 vLLM 进程，同时也用正式 CLI 参数传入配置；应以
   vLLM 最终启动命令和 engine config 为准。该警告可以清理，但不是本轮分数为
   0 的原因。
6. **单次请求 timeout 与 batch watchdog 曾过于接近。** SearchQA、DocVQA、
   LiveMath 现在使用 300 秒单请求、最多 2 次 rollout 尝试、660 秒外层 watchdog；
   SpreadsheetBench 使用 300/1200/1500 秒三层时限，避免外层提前写入假
   `task-timeout` 后内部线程仍继续占用 vLLM。
7. **OfficeQA 文档目录“存在”不代表可检索。** Adapter 现在在 setup 阶段抽样
   验证 split 引用能否解析为 candidate file 或 oracle page；同时将畸形工具
   参数反馈给模型自修复，减少 parser 参数错误造成的空答案和全 0。

修复后重新运行必须使用新的 `TS/OUT_BASE/LOG_DIR`，并在 SpreadsheetBench
开始第一个 rollout 前确认 `train=2`、`steps/epoch=1`。OfficeQA 则应确认
`vllm_qwen.log` 中没有 `hermes_tool_parser.py`，启动 smoke 返回结构化
`tool_calls`。

### 13.5 2026-07-17 二次代码、脚本与资源复查

二次复查覆盖：

- `scripts/runs/smoke/resource_pool_epoch1` 三组完整 epoch 冒烟脚本。
- `scripts/runs/resource_pool_4x_l20` 四组正式资源池脚本。
- 通用本地 Qwen/vLLM 并发启动器及六个数据集的最终命令展开。
- SearchQA、DocVQA、LiveMath、OfficeQA、SpreadsheetBench 的 rollout timeout
  和错误结果生成路径。

复查后的明确结论：

1. **普通单轮数据集不再使用过短 timeout。** SearchQA、DocVQA、LiveMath 均为
   300 秒单请求，`TARGET_QWEN_CHAT_ROLLOUT_RETRIES=2`，外层单题 watchdog
   660 秒。真实服务异常仍可能产生 timeout，但不应再因 120 秒旧默认或内外层
   几乎同时到期产生假失败。
2. **SpreadsheetBench 使用三层时限。** `env.llm_timeout=300` 控制一次模型
   请求，`env.exec_timeout=1200` 控制完整多轮单题，batch future watchdog 在
   1500 秒才兜底。外层不再在内部 1200 秒截止点同时写入 TIMEOUT。
3. **OfficeQA 不再只检查目录是否存在。** Adapter setup 会抽样验证 train、
   val、test 的 `source_files/source_docs` 能否解析为 candidate file 或 oracle
   page。本地当前 split/docs 实测 `evidence_resolved=4/4`，抽样 oracle context
   约为 5K–12K 字符。
4. **OfficeQA 畸形工具参数可自恢复。** 结构化 tool call 的 arguments 先严格
   JSON 解析，再尝试安全修复；仍不合法时把错误写成 tool result 交给下一轮，
   不再由最外层异常处理直接把整条样本变成空答案和 hard=0。
5. **OfficeQA parser 启动检查是硬门槛。** Qwen3.5 使用
   `qwen3_coder` tool parser 和 `qwen3` reasoning parser；启动 smoke 必须得到
   `message.tool_calls`。正文里只有 `<tool_call>` 不算验证通过。
6. **当前正式资源没有发现 OOM 配置错误。** 单卡只有一个 vLLM，
   `GPU_MEMORY_UTILIZATION=0.90`、`MAX_MODEL_LEN=65536`、
   `VLLM_MAX_NUM_BATCHED_TOKENS=65536`，SearchQA 上限 128，其余成对任务上限
   96。历史日志无 CUDA OOM。
7. **当前资源是“高利用率并留长任务余量”，不是每阶段硬凑 128 并发。**
   SearchQA 可接近 120–128；两数据集共享卡的训练自然峰值约 48、评估 worker
   上限约 96；ALFWorld 因 50-step 长交互保持 24 workers。

本轮本地验证结果：

| 检查 | 结果 |
|---|---|
| OfficeQA 真实 split/docs 预检 | 通过，4/4 抽样证据可解析 |
| 三组 smoke + 四组正式脚本 dry-run | 全部通过 |
| 相关 shell `bash -n` | 全部通过 |
| Python `compileall` | 通过 |
| 全量 pytest | 193 passed，2 skipped |
| 历史 smoke CUDA OOM 检索 | 未发现 |

上述检查不能代替 L20 上的修复后真实 smoke。尤其 OfficeQA parser 和
SpreadsheetBench 1200 秒任务必须在远端新运行中确认；历史输出目录记录的是
旧 Hermes/600 秒配置，不得作为修复后的通过证据。

## 14. 修复后 smoke 的重跑与验收

每次修复 timeout、parser、数据路径、worker 或验证器后，都必须：

1. 使用新的 `TS`、`OUT_BASE`、`LOG_DIR`，不得复用旧 `results.jsonl`。
2. 先 dry-run，检查最终命令包含 `timeout=300`、thinking 关闭以及数据集专属
   `cfg-options`。
3. OfficeQA 启动时确认：
   - `[OfficeQA preflight] ... evidence_resolved=N/M` 且 `N>0`；
   - vLLM 启动参数是 `qwen3_coder`/`qwen3`；
   - tool smoke 返回结构化 `tool_calls`；
   - `vllm_qwen.log` 没有 `hermes_tool_parser.py`。
4. SpreadsheetBench 启动时确认：
   - DataLoader 显示 train/val/test 均为 2；
   - `steps/epoch=1`；
   - 最终命令含 `env.exec_timeout=1200 env.llm_timeout=300`；
   - 结果中没有 `phase=timeout/error`。
5. smoke 完成后检查 `[exit] 0`、`completed.tsv`、`summary.json` 和 rollout
   `results.jsonl`；任何一个缺失都不能判定通过。
6. OfficeQA 若全部为 0，先区分：
   - 空答案、无证据、tool error：运行故障；
   - 有答案、有 oracle/tool 证据但 EM=0：模型或 skill 质量问题。

## 15. 每次正式启动前的最小检查清单

1. 确认 optimizer endpoint、model、key 属于同一来源。
2. 确认 optimizer 和 target thinking 都关闭，`REASONING_EFFORT` 没有被 YAML
   中的 `medium` 重新带回。
3. 确认每张 GPU 只有一个 vLLM owner，其他训练进程使用 `START_VLLM=0`。
4. 确认 `VLLM_MAX_NUM_SEQS`、各进程 workers 和账号总 analyst 并发符合本文上限。
5. 确认普通数据集采用 300/660 秒请求与 watchdog，SpreadsheetBench 采用
   300/1200/1500 秒三层时限，并且 rollout retries 为 2。
6. 确认 train/val/test split、图片、OfficeQA docs、Spreadsheet data root 均存在。
7. 确认 `0` 的语义，特别是 `TEST_ENV_NUM=0` 不代表关闭 test。
8. 使用唯一的 `TS/OUT_BASE/LOG_DIR`，避免复用错误缓存。
9. 先执行 dry-run 检查最终 `train.py` 命令，再跑 2–8 个样本的真实 smoke test。
10. smoke 启动后核对 DataLoader 实际计数与 `steps/epoch`，不能只核对 CLI 参数。
11. smoke 结束后同时检查 `[exit] 0`、`completed.tsv`、`summary.json` 和 rollout
    结果；并发组中只完成一个数据集不算整组通过。
12. 将普通答错与系统失败分开：非空答案的 `hard=0` 可以接受，timeout、空答、
    `agent_ok=false`、缺少图片/文档证据不能接受。
13. OfficeQA 确认结构化 tool call、`evidence_resolved>0`、文档路径和证据；
    SpreadsheetBench 确认没有 `phase=timeout/error`；DocVQA 确认
    `image_paths` 非空。
14. smoke 通过后再启动完整训练，并为正式运行使用新的输出目录。
