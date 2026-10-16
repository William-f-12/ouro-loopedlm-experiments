# RQ2 探针快速开始

RQ2 已整合进 `ouro-loopedlm-experiments`，与基线共用模型和量化实现，
保留独立 notebook、开关和输出。迁移说明见 [docs/project-layout.md](docs/project-layout.md)。

1. 在项目目录运行 `python scripts/package_for_colab.py`，生成统一源码 ZIP。
2. Colab 打开 `colab/Ouro_RQ2_Probes.ipynb`，选择原生 BF16 GPU。
3. 第 2 节上传统一 ZIP；第 4 节挂载保存基线结果的 Drive。
4. 第 6 节设置 `SOURCE_RESULTS` 为一批已完成的基线目录或下载 ZIP。
   当前已跑基线是 `simulated`，默认示例路径是 `early-exit`，请按实际路径修改。
5. 设置 `ENABLE_RQ2=True`，先保持 `RQ2_LIMIT=2`，第 7 节运行。
6. 第 8–10 节查看汇总、逐 token 记录并下载独立诊断结果。

默认开关 False。读取已有轨迹，不重新自由生成回答，也不改变原准确率或耗时。
输入暂不接受合并目录或未完成批次；会检查历史基线源码兼容性。
固定 token 和历史退出轨迹，在两种精度下诊断所有四轮。
指标及限制见 [docs/rq2.md](docs/rq2.md)。

旧命令 `python scripts/package_rq2_for_colab.py` 是同一打包器的兼容入口。
没有 pass@k、采样退出、RL/QAT 或真实低比特 kernel。
