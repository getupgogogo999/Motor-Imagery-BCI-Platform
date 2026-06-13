# BCI 2a 系统化实验计划

## 目标

- 排查 A04–A06 低准确率根因（非盲目换模型）
- 将 9 人平均 CV 准确率从 ~65% 提升至 **70–75%**
- 输出可复现实验表格与混淆矩阵

## 阶段 1：数据泄漏审计（已完成）

**方法**：`StratifiedKFold(5)` + 每折独立 `fit` Pipeline（CSP → StandardScaler → SVM）

**检查项**：
- CSP 滤波器仅在训练折 fit
- StandardScaler 仅在训练折 fit
- 测试折从未参与任何 fit

**输出**：`outputs/experiments/leakage_audit.txt`

## 阶段 2：受试者诊断

**指标**（每人）：
| 指标 | 用途 |
|------|------|
| trial 数量 | 是否样本不足 |
| 各类别样本数 | 是否类别不平衡 |
| 坏通道数（方差 Z>3） | 信号质量 |
| EEG 振幅 mean/std | 异常放大/饱和 |
| μ(8–13Hz) / β(13–30Hz) 能量 | MI 相关节律是否偏弱 |
| anomaly_flag | 启发式异常标记 |

**输出**：`outputs/experiments/subject_diagnostics.csv`

**预期**：A04–A06 可能 μ/β 能量偏低或方差结构异常，解释“接近随机”而非代码 bug。

## 阶段 3：预处理网格（控制变量）

| 维度 | 取值 |
|------|------|
| 频带 | 8–30 / 7–35 / 4–40 Hz |
| 时间窗 | 0.5–2.5 / 0.5–3.5 / 0.5–4.0 / 1.0–4.0 s |
| 重参考 | CAR on / off |

共 **3×4×2 = 24** 种预处理 × 3 方法 = 72 条/受试者。

## 阶段 4：方法对比（公平）

| 方法 | 说明 |
|------|------|
| CSP+SVM | 当前 baseline |
| FBCSP+LDA | 经典竞赛方法 |
| EEGNet | 深度 baseline（同 CV 同划分） |

**公平性**：同一 `random_state=42`，同一 5 折划分，同一预处理。

## 阶段 5：输出

- `all_results.csv`：全实验明细
- `method_mean_summary.csv`：方法平均 Accuracy/Kappa
- `best_config_per_subject.csv`：每人最优配置
- `subject_max_accuracy_by_method.csv`：每人各方法最高 Accuracy
- `confusion_matrices/`：默认配置混淆矩阵

## 预期收益

| 优化 | 预期提升 | 依据 |
|------|----------|------|
| 时间窗 0.5–3.5s / 0.5–4.0s | +2–5% 平均 | 更长 MI 段，更多 ERD/ERS |
| CAR | +1–3% | 减少参考污染 |
| FBCSP+LDA vs CSP+SVM | +2–4% 平均 | 多子带更稳 |
| 7–35Hz / 4–40Hz | 因人而异 | 部分受试者 β 节律偏移 |
| 每人最优配置 | 平均 +5–8% | 相对固定 8–30/0.5–2.5 |

** realistic 目标**：
- 固定配置平均：**68–72%**
- 每人最优配置平均：**72–76%**
- A04–A06 仍可能 **50–60%**（BCI illiteracy，数据层面限制）

## 运行

```bash
pip install -r requirements.txt

# 完整实验（含 EEGNet，约 30–60 分钟）
python run_experiments.py

# 快速版（无 EEGNet）
python run_experiments.py --quick

# 仅诊断
python run_experiments.py --diagnostics-only
```
