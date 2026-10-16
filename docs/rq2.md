# RQ2 使用与指标说明

## 范围和独立诊断

新 notebook：`colab/Ouro_RQ2_Probes.ipynb`。
新 CLI：`scripts/rq2_probe.py`。
默认 `ENABLE_RQ2=False`；CLI 未传 `--enable-rq2` 时不会加载模型、读取结果或写输出。
Notebook 第 1–5 节仍是常规 GPU/依赖/小模型准备检查，开关控制第 7 节的诊断任务。
当前位于统一项目 `ouro-loopedlm-experiments`，基线与探针仍使用独立入口和输出。
当前 MATH-500 后续批次继续使用冻结 ZIP；迁移说明见 [project-layout.md](project-layout.md)。

输入是一批**已完成**的基线结果目录或第 10 节下载的 ZIP，包含
`manifest.json`、`results.jsonl`、`quantization.json`。暂不读取第 11 节的合并目录。
同一批需所有题目、BF16/PTQ 与全部阈值组合完整；不接受边写边读的未完成批次。
源代码哈希、模型版本、退出策略及记录会核对。不兼容的历史版本需对应实现，
不要手工删除 hash 检查。RQ2 最初以 a36a55649dbb 对应基线为起点；统一项目保留这部分源码。

输出默认在 Drive 的 `ouro-experiments/rq2-results/`；基线结果只读。
开启/关闭探针均不会更改基线参数或原始生成结果。诊断支持 early_exit 和
simulated 输入，一次处理一种模式，不把两者混合。

## 重放的实验含义

每道入选题的 BF16/PTQ、threshold=1/0.5 保存轨迹各自作为 anchor。
同一 anchor 的 token 序列、prompt 和历史实际退出轮数，在两种精度中保持一致。
每种精度维护自己的数值 KV；它们不会共用张量。

当前 token 诊断时始终运行四轮，得到全部隐藏状态、退出概率和各轮 LM-head top-1。
若 anchor 原本提前退出，诊断结束后将本 token 在未执行轮次的 KV 改回最后退出轮的
对应 KV，再重放下一个 token。历史位置不变。这避免诊断时多算的轮次污染未来缓存。
原 prompt prefill 策略也保留：真实自适应为 P-1 前缀满深度，其他情况整段满深度。

对 anchor 自身精度，逐 token 检查 gate 选择轮次和选中轮次的 top-1 是否与原记录一致；
不一致则失败，manifest 标为 failed，不产出成功汇总。更换硬件/运算环境可能影响
浮点边界，请优先使用相同 GPU 型号和依赖；重放检查通过也不代表逐位相同的 logits。

另一个精度的 gate 深度是**相同历史条件下的候选退出深度**。它不会替代已记录历史，
也不是该精度独立自由生成的实际轨迹。比较两种自由生成行为仍使用原基线 RQ1 结果。
按 anchor 精度分开报告，不将 BF16/PTQ 的不同文本按 token 下标硬配对。

## 参数

第 6 节默认：`ENABLE_RQ2=False`，`RQ2_LIMIT=2`，`RQ2_SEED=42`，
`MAX_PROBE_TOKENS=0`，表示固定随机选择 2 道题，分析其全部保存 token。
选择不依赖答对与否或是否出现深度膨胀。20–50 题正式子集可在新结果目录运行。

若 `MAX_PROBE_TOKENS>0`，仅分析每条轨迹的前若干 token，manifest 明确记录被截短
的轨迹数；它适合 smoke test，不能据此推断完整回答后半段的行为。
不要误将这个参数当作均匀 token 抽样。它不会修改基线的 2048 输出上限。

`SOURCE_RESULTS` 默认指向原 Drive 中 MATH-500 第一批真实退出结果。也可使用
`Path('/content/xxx-results.zip')`；ZIP 只直接读取必要文件，不执行或解压代码。
`RQ2_RUN_NAME` 必须是新的简单目录名。当前不支持诊断断点续跑；失败可保留现场，
换新名字重跑所选子集。已完成基线不需重跑。

## 指标

隐藏状态 h 取每轮共享 core 完成后的 RMSNorm 输出，进入退出 gate / LM head 之前。
按同一 anchor/token/loop 计算：

- `top1_by_loop`：四轮各自预测的 token ID。
- `stable_from_loop`：从这一轮到最后一轮，top-1 始终相同的最早轮次。
  例如 [A,A,B,B] 是 3，[A,B,A,A] 也是 3；不能只找最早相邻重复。
- `candidate_exit_depth`：当前精度在该历史条件下按阈值选择的轮次。
- `exit_minus_stability`：候选退出轮次减去稳定轮次；正值为稳定后还继续循环，
  负值为在最终稳定之前退出。0 表示两者相等。
- `extra_loops_after_stability`：上述差值的正部。
- `exit_top1_matches_final_loop`：退出时的预测是否等于最后一轮预测。
- `hidden_error.l2_by_loop`：每轮 ||h_PTQ - h_BF16||_2。
- `hidden_error.relative_l2_by_loop`：L2 误差 / ||h_BF16||_2，参考范数为零时为 null。

退出后轮次的 top-1 是诊断用的反事实继续计算；top-1 稳定既不等于概率分布稳定，
也不等于答案正确。最后一轮恒能成为稳定起点，因此末轮稳定存在观察上限。
这些探针可观察关联，不能单独证明 gate 冗余的因果机制。

## 输出与统计

- `rq2_manifest.json`：运行状态、源文件哈希、源协议、实际 runtime、子集和诊断时间。
- `rq2_tokens.jsonl`：逐 token 文本/ID、相对位置、原始退出轮次、两精度探针和误差。
- `rq2_summary.csv/json`：按数据集、模式、阈值、anchor 精度分组。
- `rq2_quantization.json`：本次相同 RTN 配方的量化清单。

摘要先在每道题内平均，再对题目等权平均并计算标准误；不会把相关的数千 token
当成数千个独立样本。每个指标有 `_n`、`_se`，只有一题时 SE=null。
`candidate_depth_delta_ptq_minus_bf16` 为共同历史下的深度差，不能替代 RQ1 的自由生成深度差。
逐 token 表可按这个差值、文本和位置定位哪些 token 多循环。

BF16 隐藏状态暂存在本地 scratch 目录，量化后逐条读取进行比较；只占一份 GPU 模型。
结束后自动清理临时张量，最终 ZIP 不包含完整隐藏状态。整个诊断耗时单独记录，
不写入 `generation_seconds`，不解释为提前退出速度或低比特 kernel 吞吐量。

## CLI

```bash
python scripts/rq2_probe.py --enable-rq2 \
  --source-results /path/to/completed-batch-results.zip \
  --dataset math500 --limit 20 --seed 42 --max-probe-tokens 0 \
  --scratch-dir /content/rq2-scratch --output-dir /path/to/new-rq2-output
```

本地验证使用随机小模型；没有宣称已测得完整 1.4B 的 RQ2 结论或 GPU 耗时。
