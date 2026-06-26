"""
03_模型训练与评估 (Model Training & Evaluation)
多模型对比：线性回归、随机森林、XGBoost、梯度提升树
使用K-fold交叉验证、超参数调优、MAPE和R²双指标评估
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import warnings
import sys
warnings.filterwarnings('ignore')

# 解决 Windows 控制台编码问题
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# 中文字体设置（显式加载字体文件）
fm.fontManager.addfont('C:/Windows/Fonts/simhei.ttf')
fm._load_fontmanager(try_read_cache=False)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
ZH_FONT = fm.FontProperties(fname='C:/Windows/Fonts/simhei.ttf')

def set_axis_text(ax, title=None, xlabel=None, ylabel=None, legend=False):
    """显式设置字体。"""
    if title is not None:
        ax.set_title(title, fontproperties=ZH_FONT, fontsize=14)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontproperties=ZH_FONT, fontsize=12)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontproperties=ZH_FONT, fontsize=12)
    for label in ax.get_xticklabels():
        label.set_fontproperties(ZH_FONT)
    for label in ax.get_yticklabels():
        label.set_fontproperties(ZH_FONT)
    if legend:
        leg = ax.get_legend()
        if leg is not None:
            for text in leg.get_texts():
                text.set_fontproperties(ZH_FONT)

# ============================================================
# 1. 加载预处理后的数据
# ============================================================
print("=" * 60)
print("加载预处理数据...")
data = np.load('preprocessed_data.npz', allow_pickle=True)
X_train, X_test = data['X_train'], data['X_test']
y_train, y_test = data['y_train'], data['y_test']
feature_names = list(data['feature_names'])
print(f"  训练集: {X_train.shape}")
print(f"  测试集: {X_test.shape}")
print(f"  特征数: {len(feature_names)}")

# ============================================================
# 2. 导入模型和评估工具
# ============================================================
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, mean_absolute_percentage_error

# 自定义 MAPE 函数（避免除零问题）
def mape_score(y_true, y_pred):
    """计算 MAPE（Mean Absolute Percentage Error），单位%"""
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def evaluate_model(model, X_tr, y_tr, X_te, y_te, model_name):
    """评估模型并返回指标"""
    # 预测
    y_train_pred = model.predict(X_tr)
    y_test_pred = model.predict(X_te)

    # 计算指标
    train_r2 = r2_score(y_tr, y_train_pred)
    test_r2 = r2_score(y_te, y_test_pred)
    train_mape = mape_score(y_tr, y_train_pred)
    test_mape = mape_score(y_te, y_test_pred)

    print(f"\n  {model_name}:")
    print(f"    训练集 - R²={train_r2:.4f}, MAPE={train_mape:.2f}%")
    print(f"    测试集 - R²={test_r2:.4f}, MAPE={test_mape:.2f}%")

    return {
        '模型': model_name,
        '训练R²': round(train_r2, 4),
        '测试R²': round(test_r2, 4),
        '训练MAPE(%)': round(train_mape, 2),
        '测试MAPE(%)': round(test_mape, 2),
    }

# ============================================================
# 3. 模型1: 线性回归（基准模型）
# ============================================================
print("\n" + "=" * 60)
print("[模型1] 线性回归 (Linear Regression) - 基准模型")
lr = LinearRegression()
lr.fit(X_train, y_train)
results = [evaluate_model(lr, X_train, y_train, X_test, y_test, 'LinearRegression')]

# 交叉验证
cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores_r2 = cross_val_score(lr, X_train, y_train, cv=cv, scoring='r2')
print(f"  5折交叉验证 R²: {cv_scores_r2.mean():.4f} (±{cv_scores_r2.std():.4f})")

# ============================================================
# 4. 模型2: 随机森林
# ============================================================
print("\n" + "=" * 60)
print("[模型2] 随机森林 (Random Forest)")

from sklearn.model_selection import RandomizedSearchCV

# 超参数搜索空间
rf_params = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [10, 15, 20, 25, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
}

rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)
rf_search = RandomizedSearchCV(
    rf_base, rf_params, n_iter=30, cv=3,
    scoring='r2', random_state=42, n_jobs=-1, verbose=0
)
rf_search.fit(X_train, y_train)
print(f"  最佳参数: {rf_search.best_params_}")
print(f"  最佳交叉验证 R²: {rf_search.best_score_:.4f}")

rf_best = rf_search.best_estimator_
results.append(evaluate_model(rf_best, X_train, y_train, X_test, y_test, 'RandomForest'))

# ============================================================
# 5. 模型3: XGBoost
# ============================================================
print("\n" + "=" * 60)
print("[模型3] XGBoost")

xgb_params = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [3, 5, 7, 9, 12],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'min_child_weight': [1, 3, 5],
}

xgb_base = XGBRegressor(random_state=42, n_jobs=-1, verbosity=0)
xgb_search = RandomizedSearchCV(
    xgb_base, xgb_params, n_iter=30, cv=3,
    scoring='r2', random_state=42, n_jobs=-1, verbose=0
)
xgb_search.fit(X_train, y_train)
print(f"  最佳参数: {xgb_search.best_params_}")
print(f"  最佳交叉验证 R²: {xgb_search.best_score_:.4f}")

xgb_best = xgb_search.best_estimator_
results.append(evaluate_model(xgb_best, X_train, y_train, X_test, y_test, 'XGBoost'))

# ============================================================
# 6. 模型4: 梯度提升树 (GBDT)
# ============================================================
print("\n" + "=" * 60)
print("[模型4] 梯度提升树 (Gradient Boosting)")

gbdt_params = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

gbdt_base = GradientBoostingRegressor(random_state=42)
gbdt_search = RandomizedSearchCV(
    gbdt_base, gbdt_params, n_iter=30, cv=3,
    scoring='r2', random_state=42, n_jobs=-1, verbose=0
)
gbdt_search.fit(X_train, y_train)
print(f"  最佳参数: {gbdt_search.best_params_}")
print(f"  最佳交叉验证 R²: {gbdt_search.best_score_:.4f}")

gbdt_best = gbdt_search.best_estimator_
results.append(evaluate_model(gbdt_best, X_train, y_train, X_test, y_test, 'GradientBoosting'))

# ============================================================
# 7. 模型综合对比
# ============================================================
print("\n" + "=" * 60)
print("【模型综合对比】")
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('测试R²', ascending=False)
print(results_df.to_string(index=False))
results_df.to_csv('output_table_model_comparison.csv', encoding='utf-8-sig', index=False)

# 图9: 模型对比柱状图
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# R2 对比
models = results_df['模型'].values
test_r2 = results_df['测试R²'].values
train_r2 = results_df['训练R²'].values

x = np.arange(len(models))
width = 0.35

axes[0].bar(x - width/2, train_r2, width, label='训练集 R2', color='steelblue', alpha=0.85)
axes[0].bar(x + width/2, test_r2, width, label='测试集 R2', color='coral', alpha=0.85)
for i, (tr, te) in enumerate(zip(train_r2, test_r2)):
    axes[0].text(i - width/2, tr + 0.01, f'{tr:.3f}', ha='center', fontsize=8)
    axes[0].text(i + width/2, te + 0.01, f'{te:.3f}', ha='center', fontsize=8)
axes[0].set_xticks(x)
axes[0].set_xticklabels(models, fontsize=9)
axes[0].legend(fontsize=10)
axes[0].set_ylim(0, 1.0)
set_axis_text(axes[0], '各模型 R2 对比', None, 'R2', legend=True)

# MAPE 对比
test_mape = results_df['测试MAPE(%)'].values
train_mape = results_df['训练MAPE(%)'].values

axes[1].bar(x - width/2, train_mape, width, label='训练集 MAPE(%)', color='steelblue', alpha=0.85)
axes[1].bar(x + width/2, test_mape, width, label='测试集 MAPE(%)', color='coral', alpha=0.85)
for i, (tr, te) in enumerate(zip(train_mape, test_mape)):
    axes[1].text(i - width/2, tr + 0.3, f'{tr:.1f}%', ha='center', fontsize=8)
    axes[1].text(i + width/2, te + 0.3, f'{te:.1f}%', ha='center', fontsize=8)
axes[1].set_xticks(x)
axes[1].set_xticklabels(models, fontsize=9)
axes[1].legend(fontsize=10)
set_axis_text(axes[1], '各模型 MAPE 对比（越低越好）', None, 'MAPE (%)', legend=True)

plt.tight_layout()
plt.savefig('fig9_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print(">>> 已保存: fig9_model_comparison.png")

# ============================================================
# 8. 确定最佳模型：XGBoost（通常性能最好）
# ============================================================
best_model = xgb_best
best_name = 'XGBoost'

# 如果随机森林表现更好则使用随机森林
best_idx = results_df['测试R²'].idxmax() if hasattr(results_df['测试R²'], 'idxmax') else 0
if isinstance(best_idx, (int, np.integer)):
    best_name = results_df.iloc[0]['模型']
    model_dict = {
        'LinearRegression': lr,
        'RandomForest': rf_best,
        'XGBoost': xgb_best,
        'GradientBoosting': gbdt_best,
    }
    best_model = model_dict[best_name]

print(f"\n最佳模型: {best_name}")

# ============================================================
# 9. 最佳模型预测结果可视化
# ============================================================
y_pred = best_model.predict(X_test)

# 图10: 预测值 vs 真实值散点图
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 散点图
axes[0].scatter(y_test, y_pred, alpha=0.4, s=10, color='steelblue')
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2, label='完美预测线')
axes[0].legend(fontsize=10)
set_axis_text(
    axes[0],
    f'{best_name}: 预测值 vs 真实值\n测试集 R2={r2_score(y_test, y_pred):.4f}  MAPE={mape_score(y_test, y_pred):.2f}%',
    '真实总价（万元）',
    '预测总价（万元）',
    legend=True
)

# 残差分布
residuals = y_test - y_pred
axes[1].scatter(y_pred, residuals, alpha=0.4, s=10, color='coral')
axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1.5)
axes[1].axhline(y=residuals.mean(), color='red', linestyle='--', linewidth=1.5, label=f'均值: {residuals.mean():.1f}')
axes[1].legend(fontsize=10)
set_axis_text(axes[1], '残差分布图', '预测值（万元）', '残差（万元）', legend=True)

# 残差直方图
axes[2].hist(residuals, bins=50, color='mediumpurple', edgecolor='white', alpha=0.85)
axes[2].axvline(x=0, color='red', linestyle='--', linewidth=2)
set_axis_text(axes[2], '残差直方图', '残差（万元）', '频数')

plt.tight_layout()
plt.savefig('fig10_prediction_evaluation.png', dpi=150, bbox_inches='tight')
plt.close()
print(">>> 已保存: fig10_prediction_evaluation.png")

# ============================================================
# 10. 保存最佳模型
# ============================================================
import joblib
joblib.dump(best_model, 'best_model.pkl')
print(f"\n>>> 已保存最佳模型 ({best_name}): best_model.pkl")

# 保存测试集预测结果供后续分析
np.savez('test_predictions.npz', y_test=y_test, y_pred=y_pred)

print("\n" + "=" * 60)
print("模型训练与评估完成！")
print(f"  最佳模型: {best_name}")
print(f"  测试集 R²: {r2_score(y_test, y_pred):.4f}")
print(f"  测试集 MAPE: {mape_score(y_test, y_pred):.2f}%")
print("=" * 60)