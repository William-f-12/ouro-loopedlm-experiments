# Thinking BF16 / PTQ 基线实验

## Colab 操作

1. 本项目目录内运行 `python scripts/package_for_colab.py`，使用打印的新 ZIP。
2. 在 [Colab](https://colab.research.google.com/) 上传 `colab/Ouro_Thinking_Baselines.ipynb`。
3. 选择原生 BF16 GPU，例如 L4 / A100。T4 不满足精度要求，会明确报错。
4. 从上到下运行：上传 ZIP、安装依赖、挂载 Drive、检查源码、设置本次批次、跑实验。
5. 当前 notebook 默认 MATH-500 分为 5 批，每批 100 题；每次只运行一批。两组 × 两阈值，共 400 次生成，每次最多 2048 新 tokens。
6. 每批看表格/图、复查逐题输出、下载结果 ZIP。五批完成后运行第 11 节合并，第 12 节下载完整 500 题结果。

修改本地代码后重新打包、上传新 ZIP，用新 `RUN_NAME` 开始。每次运行启动新进程；
模型类从上传源码直接 import，权重和 tokenizer 从固定版本的 Hub 下载。
两组只在 GPU 加载一份模型：先 BF16，再原地量化 MLP 后跑 PTQ。

### MATH-500 分五批运行

新版 notebook 用一个布尔开关控制两种退出行为；上传新版 notebook 和源码 ZIP 后即可使用。
本地修改不会自动同步到已打开的 Colab。新会话从第 1 节开始，使用同一份源码 ZIP 完成全部五批。
第 6 节默认参数：

```python
DATASETS = ["math500"]
LIMIT = 100
BATCH_ID = 1
ENABLE_EARLY_EXIT = True  # False = 跑满后选择
EXECUTION_MODE = "early_exit" if ENABLE_EARLY_EXIT else "simulated"
RUN_PREFIX = f"baseline-math500-100q-{EXECUTION_MODE.replace('_', '-')}"
RESUME = False
```

本轮沿用阈值 `[1.0, 0.5]`、原始 4-loop 配置（`LOOPS=None` 读取本地配置）、
`MAX_NEW_TOKENS=2048`、`MAX_TOTAL_TOKENS=4096`、`SEED=42`，以及相同 BF16/PTQ 实现。

- seed 42 固定打乱整个数据集；`OFFSET=(BATCH_ID-1)*LIMIT` 取连续切片。不要通过换 seed 来分批。
- 五批的 `OFFSET` 分别为 0、100、200、300、400，对应打乱后第 1–100、101–200、201–300、301–400、401–500 题，恰好覆盖 500 题且不重复。
- 依次运行第 6 节（参数）→ 第 7 节（实验）。第 6 节应打印本批 offset、`400` 次生成及独立结果目录。
- 完成后运行第 8 节（比较）、可选第 9 节（单题详情）、第 10 节（下载结果）。
- 下一批只改 `BATCH_ID=2`（随后 3、4、5），设 `RESUME=False`，重跑第 6–10 节；无需重新上传或安装。不会自动接着跑下一批。
- 批内中断：保持同一 `BATCH_ID`、`RUN_PREFIX`、源码、参数及运行环境，改为 `RESUME=True`；已完成记录会跳过。换下一批时改回 False。
- 若运行时已重置，重新运行第 1–5 节准备相同环境，再选择本批并续跑。新 GPU 型号或依赖版本不一致时，严格续跑检查会拒绝混合结果。
- 第 4 节默认 `USE_GOOGLE_DRIVE=True`，每批写入 `MyDrive/ouro-experiments/results/baseline-math500-100q-early-exit-batch-01`（直到 `05`）。确认 Drive 挂载后再启动实验。
- 每次第 7 节会在新进程中重新加载原始模型，先跑 BF16 再跑 PTQ；上一批的量化不会累积到下一批。

### 用一个开关对比两种退出行为

模型循环都在 `ouro_local/modeling_ouro.py`，生成流程都在 `experiments/generation.py`；
不需要维护两套源码。在第 6 节只修改 `ENABLE_EARLY_EXIT`：

| 开关 | 实验模式 | threshold=0.5 | threshold=1 |
|---|---|---|---|
| True | early_exit | 首次达到累计阈值即停止循环 | 固定 4 轮 |
| False | simulated | 跑满 4 轮，再按累计阈值选择输出 | 固定 4 轮 |

`EXECUTION_MODE` 和结果前缀由开关自动生成，传给同一份 benchmark 代码。
`True` 保存到 `baseline-math500-100q-early-exit-batch-XX`；
`False` 保存到 `baseline-math500-100q-simulated-batch-XX`。
每种模式的阈值和精度轨迹均独立生成，缓存不会共用。

完成一种模式的某批后，保持相同 BATCH_ID、题数、seed、阈值和预算，切换开关，
设 `RESUME=False`，重跑第 6–10 节即可运行另一种模式。恢复已有未完成的同模式批次
才设 `RESUME=True`。对比准确率、token 数、实际深度和时间时，保留 `execution_mode` 列。

每种模式自己的五批分别合并；合并前在第 6 节选回对应开关并运行参数单元格。
两种模式的汇总可以并排比较，但不要把它们混成一个整体平均值。
True 的 KV 补齐会改变后续轨迹，因此 threshold=0.5 的文本和准确率可能不同。
False 仍计算全部轮次，其 selected 深度不能解释成已经节省的计算量。

### 合并五批统计

五批均完成后，运行第 11 节合并、第 12 节下载。每批应有 400 条记录，合并后为 500 道题、2000 条记录、4 个实验组摘要和 2 个阈值的配对比较。
合并只需要 CPU。也可在本地用 Python 标准库合并五个下载 ZIP：

```bash
python scripts/merge_results.py --inputs batch1-results.zip batch2-results.zip batch3-results.zip batch4-results.zip batch5-results.zip --dataset math500 --expected-examples 500 --output-dir outputs/math500-full
```

`--inputs` 接受结果目录或 notebook 下载的 ZIP；替换为实际路径。合并器严格比较除题目列表/切片信息外的实验协议，包括权重、源码哈希、量化方式、生成参数、seed、评分和运行环境。
它会拒绝重复题、漏掉任何精度/阈值组合的未完成批次、记录与 manifest 不匹配，以及不足 500 题的输入；不会静默去重或覆盖已有合并目录。

所有统计都从合并后的逐题记录重新计算，**不要直接平均五份 summary**，尤其是 token 加权深度、双方共同答对题上的计算膨胀比例和标准误。
合并输出为 `results.jsonl`、`summary.csv/json`、`merge_manifest.json` 和 `provenance/` 下保存的各批原始 manifest/量化清单。记录保留原批次协议哈希；合并目录只用于统计，不作为 benchmark 的续跑目录。

分批本身不减少总计算量。真实退出改变执行量和生成轨迹，此前跑满 4 轮的耗时估计不直接适用；先用独立 pilot 前缀、`LIMIT=2` 测量，再恢复 `LIMIT=100` 和正式前缀。比较 `generation_seconds`，不能直接用循环数比例推算加速。

## 对照协议

| 项目 | BF16 | PTQ |
|---|---|---|
| 起点 | Ouro-1.4B-Thinking | 相同 checkpoint |
| 核心 MLP gate/up/down projection | 原始 BF16 | NVFP4 RTN 后反量化至 BF16 |
| Attention、退出 gate、embedding、输出头等 | BF16 | BF16 |
| 激活与原始算子的内部精度 | 原始行为 | 相同 |
| 题目、prompt、阈值、生成上限 | 两组相同 | 两组相同 |
| 解码 | 单次 greedy，batch=1 | 同左 |

PTQ 是数值模拟（QDQ）：E2M1 四位权重，沿每行输入维度每 16 个权重一组，
E4M3FN 组 scale，FP32 矩阵级 scale；最近舍入、等距取偶。scale 仅由权重统计决定，
不做校准训练，不使用测试答案调 scale。不是 bitsandbytes NF4。

**量化值反量化后仍用 BF16 存储和矩阵乘法，不压缩显存，不调用真实 NVFP4 kernel。**
当前适合研究精度和生成行为，不能用耗时声称真实低比特部署加速。已实现单样本推理提前退出；尚不包括 QAT、RL。

## 退出策略和循环指标

默认 `ENABLE_EARLY_EXIT=True`（自动映射为 `EXECUTION_MODE="early_exit"`），两阈值从同一题目分别独立生成，各自新建 KV cache：

- `threshold=1.0`：强制执行并选择最后一轮（默认 4），保留原先固定深度参考。
  即使 BF16 累积概率提前舍入到 1，也不会提前退出。
- `0<threshold<1`：每完成一轮计算累计退出概率，首次达到阈值就立刻停止当前 token 的循环。
  没达到则继续，最多执行 `LOOPS` 轮。0.5 仍是可修改的实验设置，并非调优后的默认值。
- 阈值依据条件 hazard：第 i 轮概率是 `sigmoid(gate_i) × 前面各轮未退出概率`；
  比较这些概率的累计和，而非只比较当前 gate。最后一轮强制接收剩余概率。

**跳过轮次的 KV 补齐：** 若当前 token 在第 d 轮退出，将本 token 在第 d 轮每一层
已经计算好的 K/V 追加到所有未执行轮次的对应层缓存。以前 token 的缓存不改动，
K 已经包含 RoPE，不会重复旋转。这样下一 token 即使需要更深的循环，也有完整长度的历史。
这是明确的 last-exit fill 近似，`manifest.json` 中记录为 `last_executed_loop_kv_fill_v1`。
它与旧版“全部计算再选输出”的深层 KV 不同，因而阈值 0.5 的后续文本、token 数、准确率可能变化。

自适应模式先满深度处理 prompt 的前 P−1 个 token，再让最后 prompt token 的 forward
也能提前退出并预测第一个输出 token。后续逐 token 使用同一策略。阈值 1 保留原先整个
prompt 一次满深度 prefill 的路径。**prompt 前缀本身不做自适应退出。**

| 输出字段 | 含义 |
|---|---|
| `execution_mode` | `early_exit` 真实停止；`simulated` 跑满后选择 |
| `generated_tokens` / N | 含 reasoning 和终止 EOS 的生成 token 数；不含 prompt |
| `selected_depths` | 每个生成决策选中的 gate 深度，从 1 开始 |
| `executed_depths` | 每个生成决策真实执行的循环数；真实退出时等于 selected |
| `selected_core_steps` | Σ selected depth；模拟模式中仅是代理 |
| `executed_core_steps` / C | Σ executed depth；不含 prompt 前缀 |
| `actual_forward_token_loops` | 包含 prompt 前缀的 token×循环次数 |
| `mean_executed_depth`（CSV） | 所有生成 token 加权的实际平均深度，等于 ΣC/ΣN |
| `exit_pdfs` | 已执行轮的退出概率；未执行轮为 null，不是零或补足的概率 |
| `truncated` / `stop_reason` | 达到长度上限，还是遇到 EOS 正常停止 |

第一个输出 token 的预测只计一次，最后一个输出 token 不再 forward。
两种模式都满足 `actual_forward_token_loops = loops × (prompt_tokens − 1) + executed_core_steps`。
此计数不是精确 FLOPs；prefill/decode 的 attention 成本不同。缓存仍保留全部循环的槽位，
不会按退出深度压缩 KV 显存。门控判断的 GPU 同步和缓存复制也有开销，实际速度需在 Colab 测量。

需要跑满后选择时设 `ENABLE_EARLY_EXIT=False`（自动映射 `EXECUTION_MODE="simulated"`，CLI `--execution-mode simulated`），
此时仍跑满再选。普通 `inference.py` / 单条演示 notebook 默认仍沿用上游模拟选择。
底层 `true_early_exit=True` 仅支持 eval/no-grad、batch=1、单 token、缓存开启的推理；
不支持训练、weighted exit、静态或 offloaded cache。基线入口负责 prompt prefill。

模式写入协议、CSV 和 JSON，结果前缀自动随开关变化。**不同模式、源码或缓存策略不能混合续跑或合并。** 每轮完整五批必须固定同一份源码包和配置。

## 数据和评分

- GSM8K main/test，1319 题；MATH-500 test，500 题，均固定 revision。
- seed 决定固定打乱顺序，offset/limit 决定切片；manifest 保存切片信息、选中的题目、标准答案和数据 SHA256。
- Thinking chat template，`enable_thinking=True`；统一要求最终答案写在 `\boxed{}` 中。
- 评分保留原始特殊 token，只读取 `</think>` 后的最终回答。未闭合 thinking 的回答
  标记为 `unfinished_thinking`，不能把推理中间的 boxed 数字当作最终答案。
- 单次 greedy 作答准确率，不是多次采样 pass@k，也不直接复现论文所有评测设置。
- GSM8K：提取最终 boxed 或明确 answer marker，用有理数精确比较。不会从长推理的
  最后一个数字猜答案。解析失败计为未答对，并单列 `parse_failure_rate`。
- MATH-500：Math-Verify 0.9.0 对抽取出的最终答案做符号等价比较。Colab/Linux 上
  解析和验证各有 5 秒超时；原始输出和提取方式保留供复查。
- 截断题仍计入分母，按已生成的最终答案评分，同时报告截断率。截断率高时，
  token 均值不能解释为自然完成回答的长度，准确率受输出预算限制。

报告提供逐题均值/标准误、token 加权深度、深度直方图、配对差值。
在双方都答对的题目上计算 `C_ptq/C_bf16 - 1`，分别报告逐题比例均值和总和之比。
共同答对为零时为 null，不是 0；不足两题时标准误为 null。小样本不能支持显著性结论。

## 文件与续跑

- `manifest.json`：权重版本、配置、源码哈希、题目、prompt、精度和运行环境。
- `results.jsonl`：每个问题/实验组/阈值一行，含展示文本、保留特殊 token 的
  `generated_text_raw`、评分、token IDs 和逐 token 深度。
- `quantization.json`：实际量化模块、scale、量化误差。
- `summary.csv`：表格比较；`summary.json`：完整统计和配对结果。

每题完成后即保存。新实验换 `RUN_NAME`；中断后相同名字并设置 `RESUME=True`。
源码、参数、题目或环境变化会拒绝续跑，避免混合实验。默认临时磁盘可能随会话删除；
可启用 Drive 持久保存。若 JSONL 最后一行写入中断，先备份，再只移除程序指出的
不完整行并续跑，该题会重算。

主要修改点：`experiments/quantization.py`（量化）、`generation.py`（生成和深度）、
`data.py`（题目与 prompt）、`scoring.py`（评分）、`reporting.py`（统计）、
`benchmark.py`（实验入口）。`ouro_local/modeling_ouro.py` 仍是可修改的官方源码。

## Linux 云 GPU 命令行

```bash
python -m pip install -r requirements-bench.txt
python benchmark.py --datasets gsm8k math500 --limit 8 --thresholds 1.0 0.5 --output-dir outputs/pilot-001
python benchmark.py --datasets gsm8k math500 --limit 8 --thresholds 1.0 0.5 --output-dir outputs/pilot-001 --resume
python benchmark.py --datasets math500 --limit 100 --offset 0 --thresholds 1.0 0.5 --output-dir outputs/math500-batch-01
python benchmark.py --datasets math500 --limit 100 --offset 100 --thresholds 1.0 0.5 --output-dir outputs/math500-batch-02
```

后续三批使用 offset 200、300、400 和独立目录；完整测试集仍可使用 `--limit 0 --offset 0`。

保留已有 CUDA PyTorch 2.6–2.x，Transformers 固定 4.57.1。普通单条推理 notebook
允许 T4 自动 FP16，但不能作为本 BF16 基线。`--allow-cpu-test` 仅供小模型离线检查，
强制 FP32 并标记 `research_valid=false`；该模式关闭数学评分超时以支持 Windows，
不得引用为正式 BF16 结果。完整模型 GPU 运行尚需用户在 Colab 验证。

## 官方来源

- [Thinking checkpoint](https://huggingface.co/ByteDance/Ouro-1.4B-Thinking/tree/3aaa2224253a92ca45cf2e3d427c360e1ef9c93d)
- [NVFP4 格式](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
- [NVIDIA NVFP4 参考实现](https://github.com/NVIDIA/Model-Optimizer/blob/main/modelopt/torch/quantization/qtensor/nvfp4_tensor.py)
- [GSM8K](https://huggingface.co/datasets/openai/gsm8k)、[MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500)、[Math-Verify](https://github.com/huggingface/Math-Verify)
