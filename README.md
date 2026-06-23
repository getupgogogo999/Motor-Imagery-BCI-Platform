# 运动想象 BCI 分类与推理平台

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![数据集](https://img.shields.io/badge/数据集-BCI%20Competition%20IV%202a-green.svg)](https://www.bbci.de/competition/iv/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)
[![准确率](https://img.shields.io/badge/平台平均准确率-76.8%25-brightgreen.svg)](#universal-智能路由平台结果)

基于 [BCI Competition IV Dataset 2a](https://www.bbci.de/competition/iv/#dataset2a) 的**运动想象（Motor Imagery）四分类**脑机接口项目。涵盖 GDF 信号预处理、泄漏安全交叉验证、432 组网格搜索、受试者级模型优化、跨受试者分析，以及 **Streamlit / Pygame / FastAPI** 可部署推理 Demo。

---

## 项目背景与模型选型

**最初方案：** 计划采用 **EEGNet**（PyTorch 卷积神经网络）做端到端四分类，利用深度学习自动学习时空特征。

**实验结论：** 在 BCI 2a 数据集上完成严格 **5 折交叉验证**对比后，EEGNet 九人平均准确率约 **52.4%**，明显低于调优后的 **CSP/FBCSP + SVM/LDA（72.7%）**。主要原因包括：每人仅约 288 试次、小样本下 CNN 易欠拟合；而 CSP/FBCSP 针对运动想象 ERD/ERS 有强领域先验，更适合本任务。

**最终方案：** 以 **CSP/FBCSP + SVM/LDA** 作为主模型部署至 Demo 与 Universal 平台；**EEGNet 保留为深度学习基线**（`src/eegnet.py`、`run_eegnet_comparison.py`），体现有对比、有依据的模型选型，而非盲目堆叠深度学习。

> **简历一句话：** 独立完成 MI-BCI 全流程；初探 EEGNet 后经对比选用 CSP/FBCSP，9 人平均 5 折 CV **72.7%**，Universal 平台 holdout **76.8%**；含 Streamlit/Pygame/FastAPI 演示与完整实验复现。

---

## 项目亮点

| 方向 | 成果 |
|------|------|
| **主模型（最终部署）** | CSP/FBCSP + SVM/LDA，网格优化后 5 折 CV 平均 **72.7%** |
| **深度学习基线** | PyTorch 实现 EEGNet，平均 **52.4%**（对比后未采用为主方案） |
| **推理平台** | Universal 智能路由，holdout 平均 **76.8%**（7/9 人 ≥70%） |
| **实验设计** | 432 组配置网格（频段 × 时间窗 × CAR × 方法），泄漏安全 5 折 CV |
| **跨受试者** | 迁移学习 & LOSO，单模型跨人约 **39%** |
| **信号诊断** | ERD/ERS + t-SNE，确认 A04/A06 BCI 失读（~50% 上限） |
| **工程落地** | Streamlit 网页、Pygame 小球游戏、FastAPI 本地 API |

---

## 准确率演进

| 阶段 | 平均准确率 | 评估方式 |
|------|------------|----------|
| 默认 CSP + SVM | 65.6% | 5 折 CV |
| EEGNet（PyTorch，原计划方案） | 52.4% | 5 折 CV |
| 网格搜索 + 每人最优 CSP/FBCSP | **72.7%** | 5 折 CV |
| 高潜力受试者额外调优（A02/A05/A09） | 74.0% | 5 折 CV |
| **Universal 智能路由平台（最终）** | **76.8%** | Holdout（匹配受试者） |

---

## 各受试者最优配置（5 折 CV）

| 受试者 | 方法 | 配置 | CV 准确率 |
|--------|------|------|-----------|
| A01 | FBCSP+LDA | 4–40 Hz, 0.5–4.0 s | **82.3%** |
| A02 | FBCSP+LDA | 7–35 Hz, 1.0–4.0 s | 68.8% |
| A03 | CSP+SVM | 8–30 Hz, 0.5–3.5 s | **86.8%** |
| A04 | FBCSP+LDA | 8–30 Hz, 0.5–2.5 s | 50.3% † |
| A05 | FBCSP+LDA | 4–40 Hz, 0.5–4.0 s + CAR | 75.0% |
| A06 | FBCSP+LDA | 4–40 Hz, 0.5–4.0 s | 49.0% † |
| A07 | FBCSP+LDA | 8–30 Hz, 0.5–4.0 s | 79.5% |
| A08 | CSP+SVM | 4–40 Hz, 0.5–3.5 s + CAR | **86.8%** |
| A09 | CSP+SVM | 7–35 Hz, 0.5–2.5 s + CAR | 76.0% |
| **平均** | | | **72.7%** |

† A04/A06：BCI 失读，信号层面不可分，调参无法突破 ~50%。

---

## 方法对比

| 方法 | 平均准确率 | 说明 |
|------|------------|------|
| 默认 CSP + SVM | 65.6% | 统一起点 |
| **EEGNet（PyTorch）** | **52.4%** | 初探方案，未作为主模型 |
| 每人网格最优 CSP/FBCSP | **72.7%** | **最终主方案** |
| Universal 平台 holdout | **76.8%** | 自动路由子模型 |
| 跨受试者 LOSO 单模型 | ~39% | 真·跨人上限 |

完整实验结果：`outputs/experiments/`

---

## Universal 智能路由平台结果

### 工作原理

上传 GDF 后，平台**自动路由**到该受试者的最优子模型与预处理配置，无需手动切换：

| 上传文件 | 自动使用 |
|----------|----------|
| `A05T.gdf` | A05 · FBCSP+LDA |
| `A08T.gdf` | A08 · CSP+SVM |
| `A03T.gdf` | A03 · CSP+SVM |

Streamlit / Pygame 中选 **UNIVERSAL** 即可。

### Holdout 准确率（自动路由 + 匹配受试者）

数据来源：`outputs/universal_model_validation.csv` · 80/20 分层 holdout

| 受试者 | 路由 | 方法 | Holdout | ≥70% |
|--------|------|------|---------|------|
| A01 | A01 | FBCSP+LDA | **84.5%** | ✅ |
| A02 | A02 | FBCSP+LDA | **72.4%** | ✅ |
| A03 | A03 | CSP+SVM | **91.4%** | ✅ |
| A04 | A04 | FBCSP+LDA | 60.3% | ❌ † |
| A05 | A05 | FBCSP+LDA | **82.8%** | ✅ |
| A06 | A06 | FBCSP+LDA | 56.9% | ❌ † |
| A07 | A07 | FBCSP+LDA | **79.3%** | ✅ |
| A08 | A08 | CSP+SVM | **93.1%** | ✅ |
| A09 | A09 | CSP+SVM | **70.7%** | ✅ |
| **平均** | | | **76.8%** | **7/9** |

† A04/A06 为生理层面的 BCI 失读上限。对比：真·跨人单模型（LOSO）仅约 **39%**。

---

## 快速开始

### 1. 克隆与安装依赖

```bash
git clone https://github.com/getupgogogo999/Motor-Imagery-BCI-Platform.git
cd Motor-Imagery-BCI-Platform
pip install -r requirements.txt
```

> `pip install` 仅在本地安装 Python 库，不会上传你的数据。

### 2. 下载数据（仓库不含原始 GDF）

| 数据集 | 下载 | 放置目录 |
|--------|------|----------|
| BCI IV 2a（必需） | [BCICIV_2a_gdf.zip](https://www.bbci.de/competition/download/competition_iv/BCICIV_2a_gdf.zip) | `BCICIV_2a_gdf/` |
| BCI IV 2b（可选 A010） | [BCICIV_2b_gdf.zip](https://www.bbci.de/competition/download/competition_iv/BCICIV_2b_gdf.zip) | `BCICIV_2b_gdf/` |

### 3. 训练或使用自带模型

```bash
python train.py --source gdf --optimized
python run_build_universal_model.py
```

### 4. 运行 Demo

```bash
# 网页版（本机浏览器打开 http://localhost:8501）
streamlit run app.py

# Pygame 小球游戏（空格下一试次，左右手预测控制小球）
python demo/run_demo.py

# 批量回放 / 本地 API
python demo/run_demo.py --replay --subject A09
python demo/run_demo.py --api --subject A09
```

---

## 项目结构

```
├── app.py                    # Streamlit 推理界面
├── train.py                  # 训练入口（--optimized 使用最优配置）
├── src/
│   ├── gdf_preprocessing.py  # GDF 加载与 epoch 切分
│   ├── gdf_trainer.py        # CSP+SVM / FBCSP+LDA 训练
│   ├── eegnet.py             # EEGNet（PyTorch，对比基线）
│   ├── fbcsp.py              # Filter Bank CSP
│   ├── experiment_eval.py    # 泄漏安全 5 折 CV
│   └── universal_model.py    # Universal 智能路由
├── bci_platform/             # 推理引擎、Pygame、FastAPI
├── demo/run_demo.py          # 一键 Demo
├── models/                   # 已训练 .pkl 模型
├── outputs/experiments/      # 实验报告、混淆矩阵
└── run_*.py                  # 各类实验脚本
```

---

## 实验脚本

| 脚本 | 用途 |
|------|------|
| `run_experiments_fast.py` | 432 组网格搜索 |
| `run_eegnet_comparison.py` | **EEGNet vs CSP/FBCSP 对比** |
| `run_transfer_learning.py` | 跨受试者迁移学习 |
| `run_mi_signal_analysis.py` | ERD/ERS + t-SNE 诊断 |
| `run_build_universal_model.py` | 构建 Universal 路由包 |
| `generate_experiment_report.py` | 生成汇总报告 |

所有评估均采用**分层 5 折 CV**，CSP/Scaler 仅在训练折内 fit（见 `outputs/experiments/leakage_audit.txt`）。

---

## 控制命令映射

| 运动想象 | 界面显示 | Demo 命令 |
|----------|----------|-----------|
| 左手 | Left Hand | Move Left |
| 右手 | Right Hand | Move Right |
| 脚 | Feet | Move Forward |
| 舌 | Tongue | Select |

事件码：769 / 770 / 771 / 772

---

## 技术栈

- **信号处理：** MNE-Python、SciPy
- **机器学习（主）：** scikit-learn（CSP、SVM、LDA）
- **深度学习（对比）：** PyTorch（EEGNet）
- **实验扩展：** XGBoost、LightGBM、pyriemann
- **界面：** Streamlit、Pygame、FastAPI

---

## 简历描述（可直接复制）

- 基于 BCI Competition IV 2a，实现运动想象四分类全流程；**初采用 EEGNet（PyTorch）**，经 5 折 CV 对比后改用 **CSP/FBCSP + SVM/LDA**，九人平均 CV 由 65.6% 提升至 **72.7%**，Universal 平台 holdout **76.8%**。
- 完成 432 组网格搜索、EEGNet 深度基线、跨受试者迁移与 ERD/ERS 诊断，排除数据泄漏并确认 BCI 失读受试者上限。
- 开发 Streamlit / Pygame / FastAPI 推理平台，支持 GDF 回放、Universal 受试者自动路由与模型热加载。

---

## 引用与许可

使用 BCI Competition 数据请引用相应论文。  
本项目代码采用 MIT License。  
数据集使用须遵守 [BCI Competition IV 官方条款](https://www.bbci.de/competition/iv/)。

---

## 作者

**GitHub：** [@getupgogogo999](https://github.com/getupgogogo999)  
**仓库：** [Motor-Imagery-BCI-Platform](https://github.com/getupgogogo999/Motor-Imagery-BCI-Platform)
