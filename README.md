# ouro-loopedlm-experiments

Ouro-1.4B-Thinking 的本地开发、Colab 基线评测和可选 RQ2 诊断项目。
本地修改源码，在 Colab 或其他原生 BF16 GPU 上运行推理，无需本地下载完整模型权重。

**继续当前 MATH-500 前 200 题的实验：** 剩余批次仍上传
`dist/ouro-1.4b-colab-a36a55649dbb.zip`，保持 `ENABLE_EARLY_EXIT=False`。
新统一包含额外 RQ2 文件，基线协议哈希不同。详见 [目录合并与冻结基线](docs/project-layout.md)。

## 三个入口

| 用途 | Notebook | 文档 |
|---|---|---|
| 单条推理和模型修改检查 | [Ouro_1_4B_Local.ipynb](colab/Ouro_1_4B_Local.ipynb) | 下方快速开始 |
| BF16 / PTQ 基线 | [Ouro_Thinking_Baselines.ipynb](colab/Ouro_Thinking_Baselines.ipynb) | [基线指南](docs/baselines.md) |
| 固定轨迹重放与探针 | [Ouro_RQ2_Probes.ipynb](colab/Ouro_RQ2_Probes.ipynb) | [RQ2 快速开始](RQ2_README.md)、[指标说明](docs/rq2.md) |

## 当前功能

- 本地 `ouro_local/` 模型源码，权重/tokenizer 从固定 Hugging Face revision 下载。
- GSM8K、MATH-500；BF16 与 shared-core MLP-only NVFP4 W4A16 RTN QDQ 对比。
- PTQ 权重反量化回 BF16 存储和计算，没有原生 4-bit GEMM。
- `ENABLE_EARLY_EXIT=True` 实际提前停止，并采用 last-exit KV 补齐；False 计算满四轮后选择输出。
- threshold=1 固定最后一轮；每个精度、阈值和题目使用独立生成轨迹。
- 准确率、生成 tokens、选中/执行循环次数、截断、耗时；逐条保存、严格续跑和分批合并。
- RQ2 默认关闭，开启后独立读取已完成基线轨迹，诊断退出、top-1 稳定轮次及隐藏状态误差。

尚未实现 pass@k、采样退出、QAT/RL 训练或原生低比特 kernel。

## 本地修改后在 Colab 运行

```bash
python scripts/package_for_colab.py
```

生成 `dist/ouro-loopedlm-experiments-<内容哈希>.zip`。三个 notebook 使用同一个源码包。
`python scripts/package_rq2_for_colab.py` 保留为相同打包器的兼容入口。
打包只需 Python 标准库，会排除模型、缓存和结果，并清空 notebook 执行输出。

1. 向 Colab 上传所需 notebook，选择 GPU。
2. 上传源码 ZIP，按顺序安装依赖、挂载 Drive、执行检查。
3. 在参数单元选择配置，再启动实验。
4. 结果保存到 Drive，完成或中断后下载结果 ZIP。

正式基线和 RQ2 使用原生 BF16 GPU，例如 L4/A100。单条推理演示允许 T4/FP16，
CPU 使用 FP32，不能混作 BF16 研究数据。单条推理演示默认仍是 simulated 输出选择。

基线 notebook 默认 MATH-500 每批 100 题、`BATCH_ID=1..5`、2048 新 tokens、4096 总 tokens、
seed=42、thresholds=[1.0, 0.5]。**默认 `ENABLE_EARLY_EXIT=True`；已跑的当前 200 题是 False。**
两种模式分别存储、合并和比较。RQ2 默认 `ENABLE_RQ2=False`，默认输入路径需要按实际模式和批次调整。

每次上传解压到新目录，推理启动独立 Python 进程，避免复用旧模块。
源码、数据或运行环境变化后，旧结果的严格续跑/合并检查可能拒绝混用。

## 代码位置

| 路径 | 用途 |
|---|---|
| `ouro_local/` | 可编辑模型实现、配置类与 config.json |
| `benchmark.py`、`inference.py` | 基线与单条推理入口 |
| `experiments/generation.py`、`quantization.py` | 生成、退出策略、PTQ |
| `experiments/data.py`、`scoring.py` | 数据、prompt、最终答案提取和评分 |
| `experiments/reporting.py`、`merging.py` | 保存、续跑、汇总与批次合并 |
| `experiments/rq2.py`、`rq2_io.py` | 轨迹重放、探针和统计 |
| `scripts/`、`tests/`、`docs/` | 工具、测试和说明 |

模型默认四轮，`LOOPS=None` 使用本地配置。改变模型结构后需确认官方权重兼容性。

## 本地检查

本地运行需要另行安装合适的 PyTorch，然后安装项目依赖：

```bash
python -m pip install -r requirements-bench.txt
python -m unittest discover -s tests -v
python scripts/smoke_test.py
```

Python 3.10+；Transformers 固定 4.57.1；Colab 保留 CUDA PyTorch，实际版本写入 manifest。
正式 MATH-500 评分在 Linux/Colab 运行；Windows 用于编辑、打包和小模型检查。
验证记录见 [基线验证](docs/baselines-validation.md)、[RQ2 验证](docs/rq2-validation.md)。

## 结果、版本和来源

结果、模型、缓存和 ZIP 不提交普通 Git 历史。可发布摘要、图表和完整结果包下载链接。
当前 `.gitattributes` 保留精确文件字节，避免自动换行转换改变源码哈希。
`baseline_snapshot.json` 保留 RQ2 创建时的历史路径与哈希，不是当前运行入口。

模型为 `ByteDance/Ouro-1.4B-Thinking`，固定 revision
`3aaa2224253a92ca45cf2e3d427c360e1ef9c93d`。
来源和本地修改见 [UPSTREAM.md](UPSTREAM.md)、[upstream.json](upstream.json)、[LICENSE](LICENSE)。
