# AI 医学入门教程（可运行 Notebook）

本仓库已重置为**面向零基础新手**的 AI 医学教程，采用 Jupyter Notebook 形式，覆盖从数据准备、训练、评估到可视化的完整流程。每一步都有详细注释与解释，保证**结构清晰、能跑通**。

## 📁 目录结构

```
.
├── notebooks
│   ├── 01_sam_medical_segmentation.ipynb
│   ├── 02_titan_pathology.ipynb
│   └── 03_clip_multimodal.ipynb
├── data
│   ├── sam
│   ├── titan
│   └── clip
└── README.md
```

## ✅ 你将学到什么

1. **医学图像分割（SAM）**
   - 如何准备医学图像与标注
   - 如何使用 SAM 做推理
   - 如何进行轻量微调（可选）
   - 如何评估分割效果（Dice / IoU）

2. **病理图像处理（TITAN）**
   - 病理图像的 Patch 切片与标签组织
   - 使用 TITAN 进行特征抽取/分类
   - 训练、验证与评估流程

3. **图文联合（CLIP）**
   - 如何准备图文配对数据
   - 使用 CLIP 进行相似度检索
   - 训练一个简化版图文对齐模型

## 🚀 快速开始

1. 安装依赖（每个 Notebook 会再次提示必要依赖）：

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

2. 打开 Notebook：

```bash
jupyter notebook notebooks/01_sam_medical_segmentation.ipynb
```

## 🧭 推荐学习顺序

1. `01_sam_medical_segmentation.ipynb`
2. `02_titan_pathology.ipynb`
3. `03_clip_multimodal.ipynb`

---

如果你完全不会编程，也可以直接跟着 Notebook 一步一步运行，**每段代码上方都有“做什么 / 为什么 / 结果是什么”的解释**。
