# Code-for-medAI

- [医学 AI 节点化工具箱工程任务拆分（详细版）](PLAN.md)

## MVP 代码生成器

本仓库提供一个最小可用的节点化代码生成器，用于将 JSON pipeline 转换为可运行的训练/评估脚本。

### 快速开始

1. 生成示例 pipeline 代码（无需额外依赖）：

```bash
python run_pipeline.py --config examples/pipeline.json --output generated
```

2. 在 `generated/` 目录中查看生成的 `data.py`/`model.py`/`train.py`/`eval.py`。

### Pipeline JSON 结构

- `nodes` 中包含 data/model/train/eval 四类节点，每个节点包含 `params`。
- `edges` 目前用于描述节点关系，后续可用于图结构执行。

## 前端 UI 原型

仓库提供一个纯前端的 MVP 画布，支持拖拽节点、编辑参数并导出 JSON。

```bash
python -m http.server 8000 --directory web
```

浏览器访问 `http://localhost:8000` 即可体验。

## 教程

- [医学多模态 CLIP 教学（新手友好版）](tutorials/clip_medical_tutorial.ipynb)
