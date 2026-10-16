# 验证记录

日期：2026-10-11。

## 已执行

- `python -m unittest discover -s tests -v`：**7 项通过，无跳过**。
  - Windows 路径含空格时打包、保留修改后的源码字节。
  - 新增模型 Python 文件进入 ZIP；权重、`.env`、缓存及结果目录排除。
  - 修改源码后包名哈希改变；原始字节保持不变，包括 Windows 换行。
  - CLI 参数检查；默认不覆盖本地循环配置。
  - 执行 notebook 真实上传/解压 cell，使用模拟上传 UI；文件哈希匹配，重复上传使用新目录。
  - 修改 ZIP 载荷后哈希验证拒绝继续。
  - **完整 CLI 离线集成**：本地生成小模型 checkpoint 和 tokenizer，复制项目到含空格路径，在模型源码追加可观察的修改，然后通过新的 Python 子进程执行推理。确认修改被执行、loops=3 生效、产生 token，并保存与本地修改一致的源码 SHA256。
- `python scripts/smoke_test.py`：**通过**。使用真实本地 Ouro 类及随机小模型权重，无网络下载。
  - 3 轮 × 2 层：6 个 KV cache 层，缓存/无缓存 logits 最大差异 `1.49e-7`。
  - 4 轮 × 2 层：8 个 KV cache 层，最大差异 `1.04e-7`。
  - 两种配置的 `.generate()` 均成功。
- `nbformat.validate`：通过，16 个 notebook cells，代码 cells 均可编译，无已保存输出。
- 三个上游文件的 SHA256 与 `upstream.json` 一致，初始模型源码未改动。
- `python inference.py --help` 正常运行且不依赖先导入模型。
- 独立只读代码审查未发现影响目标工作流的实质问题。

验证环境：Windows / Python 3.13.5 / PyTorch 2.6.0+cpu / Transformers 4.57.1；其余运行依赖与 requirements.txt 一致。依赖放在隔离的临时验证目录，未更换用户全局 Python 包。

## 尚未执行

- 真实 Colab 浏览器上传与云端依赖安装。
- 官方 1.4B 完整权重的 GPU 推理、速度、显存峰值及输出质量。
- T4 FP16 的完整权重数值稳定性或其他 GPU 上的 BF16 行为。
- 大批量、长上下文和自定义架构改动；默认入口为单条推理。

首次使用请在 Colab 运行默认参数。Notebook 会实际报告 GPU 型号、生成结果、耗时、显存与源文件哈希；以上本地检查不代替该步骤。
