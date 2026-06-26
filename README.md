# 深圳二手房房价分析与预测 —— 复现说明

## 1. 项目简介

本项目基于链家网深圳站6064条二手房挂牌数据，完成以下三个核心任务：
1. 构建回归模型预测房屋总价（`totalprice`）
2. 分析影响房价的Top3因素（特征重要性）
3. 找出深圳房屋均价排名最高的10个商圈

**最佳模型**：梯度提升树（GradientBoostingRegressor），测试集 R²=0.9122，MAPE=11.85%

---

## 2. 文件结构

| 文件 | 说明 |
|------|------|
| `house_shenzhen_s.csv` | 原始数据集（GBK编码，6064条×14字段） |
| `01_data_exploration.py` | 数据探索分析，生成图1~图8 |
| `02_data_preprocessing.py` | 数据预处理（清洗/特征工程/编码/标准化/分割） |
| `03_model_training.py` | 4模型训练+调参+评估，生成图9~图10 |
| `04_analysis.py` | 特征重要性分析+Top10区域分析，生成图11~图14 |

**中间产物**（运行后自动生成，不必提前准备）：

| 文件 | 由哪个脚本生成 |
|------|--------------|
| `preprocessed_data.npz` | `02` |
| `scaler.pkl` | `02` |
| `preprocessing_info.csv` | `02` |
| `best_model.pkl` | `03` |
| `test_predictions.npz` | `03` |
| `output_table_*.csv` | `01` / `03` / `04` |
| `fig*.png` | `01` / `03` / `04` |

---

## 3. 环境要求

### 3.1 Python 版本

Python ≥ 3.8

### 3.2 依赖库

| 库 | 用途 |
|----|------|
| `pandas` | 数据处理与表格输出 |
| `numpy` | 数值计算 |
| `matplotlib` | 图表绘制 |
| `seaborn` | 热力图（仅`01`使用） |
| `scikit-learn` | 模型、预处理、交叉验证、调参 |
| `xgboost` | XGBoost 回归器 |
| `joblib` | 模型/标准化器保存与加载 |

### 3.3 一次性安装命令

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib
```

### 3.4 中文字体（Windows）

脚本使用 **SimHei（黑体）**，默认路径 `C:/Windows/Fonts/simhei.ttf`。

如果你的系统没有该字体，可以替换为其他中文字体（如 `C:/Windows/Fonts/msyh.ttf` 微软雅黑），在所有 `.py` 文件顶部将 `simhei.ttf` 改为对应字体文件名即可。

### 3.5 操作系统

本脚本在 **Windows** 下开发，数据编码为 **GBK**。在其他操作系统运行可能需要：
- 确保有中文字体文件
- 修改字体路径为系统中存在的中文字体

---

## 4. 复现步骤（严格按顺序执行）

所有命令都在同目录路径下执行。

### 第1步：数据探索（可选，不产生后续依赖）

```bash
python 01_data_exploration.py
```

**产出**：
- `fig1_totalprice_distribution.png` ～ `fig8_height_vs_price.png`（共8张图）
- `output_table_dtypes.csv`、`output_table_city_avg.csv`

这一步不产生后续脚本依赖的中间文件，可以跳过。

---

### 第2步：数据预处理（**必须**）

```bash
python 02_data_preprocessing.py
```

**产出**：
- `preprocessed_data.npz` —— 训练/测试集特征与标签
- `scaler.pkl` —— 标准化器
- `preprocessing_info.csv` —— 特征列表信息

**注意**：这一步会输出 `(6061, 28)` 的特征矩阵（28个特征）、`(6061,)` 的标签向量。

---

### 第3步：模型训练与评估（**必须**）

```bash
python 03_model_training.py
```

**产出**：
- `best_model.pkl` —— 最优模型（GradientBoostingRegressor）
- `test_predictions.npz` —— 测试集预测结果
- `fig9_model_comparison.png` —— 四模型 R²/MAPE 对比
- `fig10_prediction_evaluation.png` —— 预测散点图/残差图
- `output_table_model_comparison.csv`

**训练时间**：单机 CPU 大约 3~8 分钟（取决于硬件），其中 RandomizedSearchCV 占主要时间。

---

### 第4步：结果分析（**必须**）

```bash
python 04_analysis.py
```

**产出**：
- `fig11_feature_importance.png` —— 特征重要性Top20
- `fig12_top3_factors_analysis.png` —— Top3因素可视化
- `fig13_top10_areas.png` —— 均价Top10商圈
- `fig14_area_bubble.png` —— 商圈房源-均价气泡图
- `output_table_feature_importance.csv`
- `output_table_top10_areas.csv`
- `output_table_all_areas_rank.csv`
- `report_key_results.txt`

---

### 一键复现

```bash
python 02_data_preprocessing.py && python 03_model_training.py && python 04_analysis.py
```

（`01` 为可选步骤，不影响后续）

---

## 5. 预期输出示例

| 指标 | 值 |
|------|-----|
| 预处理后样本数 | 6061 |
| 特征维度 | 28 |
| 训练集/测试集 | 4848 / 1213 |
| 最佳模型 | GradientBoostingRegressor |
| 测试集 R² | 0.9122 |
| 测试集 MAPE | 11.85% |
| Top1 因素 | 面积（size），重要性 75.42% |
| Top2 因素 | 商圈（area），重要性 6.97% |
| Top3 因素 | 房龄（building_age），重要性 3.80% |
| 均价最高区域 | 红树湾（约2165万） |

---

## 6. 关键技术决策说明

| 决策 | 原因 |
|------|------|
| 删除 `unitprice` | 与 `totalprice` 存在 `totalprice ≈ unitprice × size / 10000` 的计算关系，保留会导致数据泄露（data leakage） |
| `year=2` 按异常值处理 | 建造年份不可能为公元2年，判定为录入错误，用同城区中位数替换 |
| `area` 用 Label Encoding | 83个类别，One-Hot 会显著增加维度，标签编码在树模型中足够有效 |
| 随机搜索调参 | 30组参数 × 3折交叉验证，在精度和效率间取得平衡 |
| 双指标评估（R² + MAPE） | R² 衡量整体解释力，MAPE 衡量平均预测误差百分比，互补评估 |
