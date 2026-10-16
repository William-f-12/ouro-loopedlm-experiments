# 项目合并记录：2026-10-16

当前项目统一为 `ouro-loopedlm-experiments`。原基线与 RQ2 副本共有的 41 个
文件在合并前完全相同；新目录从完整 RQ2 副本的 56 个源码/文档文件建立。
模型、配置、PTQ、基线 CLI、生成和 RQ2 诊断逻辑保持原始字节。
验证结果见 [unification-validation.md](unification-validation.md)。

## 目录

```text
ouro-loopedlm-experiments/
  ouro_local/           可编辑模型源码、模型配置
  experiments/          数据、生成、量化、评分、结果和 RQ2
  benchmark.py          BF16 / PTQ 基线入口
  inference.py          单条推理入口
  colab/                单条推理、基线、RQ2 三个 notebook
  scripts/              统一打包、合并、诊断和 smoke test
  tests/                单元测试、CPU 小模型及 notebook 流程测试
  docs/                 使用、设计、验证及历史记录
  dist/                 生成的新 ZIP 和冻结基线 ZIP；Git 忽略
```

历史实验结果仍在工作区外层 `output/`；这次整理没有迁移或重新评分结果。
200 题分析脚本的项目引用改为新目录，并同步更新分析包的文件校验清单；
原始数据、统计报告和图保持不变，旧分析包也保存在上述备份区的 `analysis-artifacts/`。
`baseline_snapshot.json` 保留创建 RQ2 副本时的原始路径和哈希，是历史证据，
其 `source_directory` 不再是当前项目入口。旧设计/验证记录也作为历史记录保留。

## 当前 MATH-500 批次：继续使用冻结基线

前 200 题的基线代码是 `ouro-1.4b-colab-a36a55649dbb.zip`，已原样保留在
新项目 `dist/` 中。SHA256：

```text
6704daec3fa2d920193276913946ec1edb03e1efa82e7e5d3570a729e8215af7
```

剩余 batch 03–05：打开本项目 `colab/Ouro_Thinking_Baselines.ipynb`，
上传上述**冻结 ZIP**，设 `ENABLE_EARLY_EXIT=False`，保留 seed=42、100 题/批、
2048 新 token、4096 总 token、thresholds=[1.0, 0.5] 以及相同运行环境。
新批次用 `RESUME=False`；恢复已有批次才设 True。继续使用原 Drive 结果目录。

三个 notebook 的上传单元均接受新包根目录 `ouro-loopedlm-experiments/` 和
旧包根目录 `ouro-1.4b-colab/`，同时保留路径检查、内容清单和 SHA256 校验。
Notebook 的上传包决定实际执行的源码，而不是本地文件夹名称。

新统一包含 RQ2 文件，因此 `benchmark.py` 记录的 `experiments/` 源码清单不同；
不要将新包生成的基线结果与旧批次混合或强行绕过协议检查。
RQ2 notebook 可上传新统一包，读取一批已完成的旧基线结果进行诊断。

## 新实验打包

```bash
python scripts/package_for_colab.py
```

生成 `dist/ouro-loopedlm-experiments-<内容哈希>.zip`，包含三个 notebook 和
基线/RQ2 代码。`package_rq2_for_colab.py` 保留为兼容入口，调用同一个打包器。
默认仍 `ENABLE_EARLY_EXIT=True`、`ENABLE_RQ2=False`，本次不改变实验参数。

## 原目录备份

原目录完整移动到工作区的：

```text
.project-backups/2026-10-16-before-unification/
  ouro-1.4b-colab/
  ouro-1.4b-colab-rq2/
  original-files-sha256.json
```

其中包括旧 ZIP、验证依赖和缓存；迁移前逐文件记录 SHA256，迁移后复核。
这些备份位于新项目外，不需要提交 Git。

## 未来 GitHub 仓库

仓库根目录使用本项目，保留 LICENSE、UPSTREAM.md 和 upstream.json。
`.gitignore` 排除模型、结果、缓存和 ZIP；本地备份不属于仓库。
`.gitattributes` 暂采用 `* -text` 禁用换行转换，以保留源文件哈希；
如将来规范为 LF，作为独立新版本记录，不覆盖当前实验的字节快照。
本次仅整理本地目录，没有初始化 Git、创建远程仓库或发布文件。
