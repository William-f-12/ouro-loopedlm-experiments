# Ouro 本地开发与 Colab 推理

目标：本地修改 Ouro-1.4B 模型代码和配置，打包上传到 Colab，在 GPU 上加载官方固定版本权重并执行修改后的代码。

## 已确定的设计

- 新建独立目录，不修改现有课程项目，也不下载 2.87GB 权重到本机。
- `ouro_local/` 保存 Hugging Face revision `574fa66cb8bf5abdc979642d01cf2b79b16bfab1` 的原始模型、配置类及 config.json；保留来源、哈希及 Apache-2.0 许可。
- `inference.py` 显式导入本地模型类；本地 config.json 决定结构，固定 revision 决定官方权重和 tokenizer。结构修改需要保持权重兼容。
- Transformers 固定 4.57.1，推理显式新建 `DynamicCache()`。默认单条推理、4 轮循环、最多生成 128 tokens、输入与输出上限合计 4096 tokens。
- 打包脚本仅收集代码、配置、说明和 notebook，不收集权重、缓存、虚拟环境、结果、Git 数据或凭据。包内含文件哈希清单。
- Colab notebook 上传 ZIP 后校验清单并解压到新目录，安装依赖、检查 GPU、执行小模型检查，再用独立 Python 进程运行完整模型；每次推理打印本地源码路径和哈希。
- 默认权重缓存位于 Colab 临时磁盘，可选挂载 Drive 后使用持久缓存。结果可下载 JSON。
- 不要求 GitHub 账户或提交远程仓库。ZIP 是首选交付方式。

## 验收

1. 打包后本地修改实际进入 ZIP；排除缓存、凭据与权重。
2. 解压后的项目可在任意路径运行；重复上传不复用已导入的旧代码。
3. CPU 小模型的缓存与无缓存 logits 一致；循环数对应缓存层数；生成成功。
4. notebook 通过格式校验；其上传/解压流程可用本地模拟上传运行。
5. 明确区分 CPU 兼容性验证与尚未执行的 Colab 完整权重 GPU 验证。
