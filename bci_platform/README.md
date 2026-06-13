# BCI Demo Platform

本地推理 + Pygame 可视化 Demo（**不重新训练**，直接加载 `models/*.pkl`）。

## 一键运行

```bash
pip install -r requirements.txt

# Pygame 小球游戏（启动菜单选模型 + GDF）
python demo/run_demo.py

# 跳过菜单，直接 A09
python demo/run_demo.py --subject A09 --no-prompt

# GDF batch replay（无 GUI，输出准确率）
python demo/run_demo.py --replay --subject A09

# 本地 HTTP API
python demo/run_demo.py --api --subject A09
```

## 架构

```
bci_platform/
├── model_registry.py      # 模型热加载（单例）
├── feature_pipeline.py    # GDF epoch 加载 / replay
├── inference_engine.py    # 单条 / 批量推理 + logging
├── game/ball_game.py      # Pygame 小球游戏
└── api/server.py          # FastAPI 本地 API
demo/run_demo.py            # 一键入口
models/manifest.json        # 模型注册表
models/motor_imagery_*.pkl  # 已训练模型
```

## 小游戏规则

| 预测 | 效果 |
|------|------|
| Left Hand | 小球左移 |
| Right Hand | 小球右移 |
| Feet / Tongue | 不移动 |

- **空格**：手动推理下一条 GDF trial
- **启动菜单**：Step1 选模型，Step2 选 GDF；按 **F** 打开文件选择器导入 `.gdf`
- **游戏中 O / F**：随时更换 GDF 文件
- **R**：重置小球位置 + 从第 1 条 trial 重新开始（屏幕会显示 Reset OK）
- ESC 退出
- 日志：`outputs/inference_logs/predictions.log`

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| GET | `/models` | 可用模型列表 |
| POST | `/predict` | 批量推理 `{ "epochs": [[[...]]] }` |
| POST | `/replay` | GDF 全量 replay |

## Python 调用

```python
from bci_platform.inference_engine import InferenceEngine

engine = InferenceEngine("models/motor_imagery_a09.pkl")
stats = engine.replay_all()
print(stats["accuracy"])

result = engine.predict_one(epoch)  # (n_channels, n_times)
print(result.command)
```
