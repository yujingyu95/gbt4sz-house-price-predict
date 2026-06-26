"""
04_结果分析 (Result Analysis)
任务2: 分析影响房价的3个最重要因素（特征重要性）
任务3: 找出深圳房屋均价排名最高的10个区域
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

# ============================================================
# 1. 加载数据和模型
# ============================================================
data = np.load('preprocessed_data.npz', allow_pickle=True)
feature_names = list(data['feature_names'])
X_train = data['X_train']
y_train = data['y_train']

import joblib
best_model = joblib.load('best_model.pkl')
print(f"已加载最佳模型: {type(best_model).__name__}")

# 加载原始数据用于区域分析
df = pd.read_csv(r'.\house_shenzhen_s.csv', encoding='GBK')

# ============================================================
# 任务2: 分析影响房价的3个最重要因素
# ============================================================
print("\n" + "=" * 60)
print("【任务2】影响房价的3个最重要因素")
print("=" * 60)

# 获取特征重要性
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
elif hasattr(best_model, 'coef_'):
    importances = np.abs(best_model.coef_)
else:
    print("当前模型不支持特征重要性分析，使用随机森林计算")
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    importances = rf.feature_importances_

# 创建特征重要性 DataFrame
feat_imp = pd.DataFrame({
    '特征': feature_names,
    '重要性': importances
}).sort_values('重要性', ascending=False)

feat_imp['重要性(%)'] = (feat_imp['重要性'] / feat_imp['重要性'].sum() * 100).round(2)

print("\n特征重要性排名（前15）:")
print(feat_imp.head(15).to_string(index=False))

# 保存完整排名
feat_imp.to_csv('output_table_feature_importance.csv', encoding='utf-8-sig', index=False)

# 提取Top3
top3 = feat_imp.head(3)
print(f"\n>>> Top3最重要因素:")
for i, (_, row) in enumerate(top3.iterrows()):
    print(f"  {i+1}. {row['特征']}: 重要性={row['重要性']:.4f} ({row['重要性(%)']:.2f}%)")

# 图11: 特征重要性 Top20 条形图
fig, ax = plt.subplots(figsize=(10, 8))
top20 = feat_imp.head(20).iloc[::-1]
colors = ['#E74C3C' if i >= len(top20) - 3 else '#3498DB' for i in range(len(top20))]
bars = ax.barh(top20['特征'], top20['重要性'], color=colors, alpha=0.85)
for bar, val in zip(bars, top20['重要性']):
    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=8)
ax.set_xlabel('特征重要性', fontsize=12)
ax.set_title('特征重要性排名（Top 20）', fontsize=14)
plt.tight_layout()
plt.savefig('fig11_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print(">>> 已保存: fig11_feature_importance.png")

# 对Top3因素做更详细的分析图表
print("\n对Top3因素进行详细分析...")

# 重新读取原始数据（含预处理后的特征）
# 读取原始数据并做同样的预处理
df_orig = pd.read_csv(r'.\house_shenzhen_s.csv', encoding='GBK')

# 图12: Top3因素与totalprice的关系
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 因素1: size（面积）
axes[0].scatter(df_orig['size'], df_orig['totalprice'], alpha=0.3, s=10, color='steelblue')
# 添加趋势线
from numpy.polynomial.polynomial import polyfit
mask = df_orig['size'].notna() & df_orig['totalprice'].notna()
coefs = polyfit(df_orig.loc[mask, 'size'], df_orig.loc[mask, 'totalprice'], 1)
x_line = np.linspace(df_orig['size'].min(), df_orig['size'].max(), 100)
y_line = coefs[0] + coefs[1] * x_line
axes[0].plot(x_line, y_line, 'r-', linewidth=2, label='趋势线')
axes[0].set_xlabel('面积（㎡）', fontsize=12)
axes[0].set_ylabel('总价（万元）', fontsize=12)
axes[0].set_title('面积 vs 总价', fontsize=14)
axes[0].legend(fontsize=10)

# 因素2 & 3: 根据实际Top2和Top3来确定
# 预计算城区均价
city_avg = df_orig.groupby('city')['totalprice'].mean().sort_values(ascending=False)

axes[1].bar(city_avg.index, city_avg.values, color='coral', alpha=0.85)
axes[1].set_xlabel('城区', fontsize=12)
axes[1].set_ylabel('均价（万元）', fontsize=12)
axes[1].set_title('各城区均价对比', fontsize=14)
axes[1].tick_params(axis='x', rotation=30)

# 房龄
df_orig_temp = df_orig[df_orig['year'] > 1900].copy()
df_orig_temp['age'] = 2026 - df_orig_temp['year']
# 按房龄分组
df_orig_temp['age_group'] = pd.cut(df_orig_temp['age'], bins=[0,5,10,15,20,30,100],
                                     labels=['0-5年','5-10年','10-15年','15-20年','20-30年','30年+'])
age_price = df_orig_temp.groupby('age_group', observed=False)['totalprice'].mean()
axes[2].bar(range(len(age_price)), age_price.values, color='mediumpurple', alpha=0.85)
axes[2].set_xticks(range(len(age_price)))
axes[2].set_xticklabels(age_price.index, rotation=30)
axes[2].set_xlabel('房龄', fontsize=12)
axes[2].set_ylabel('均价（万元）', fontsize=12)
axes[2].set_title('房龄 vs 均价', fontsize=14)

plt.tight_layout()
plt.savefig('fig12_top3_factors_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print(">>> 已保存: fig12_top3_factors_analysis.png")

# ============================================================
# 任务3: 找出深圳房屋均价排名最高的10个区域
# ============================================================
print("\n" + "=" * 60)
print("【任务3】深圳房屋均价排名最高的10个区域（area）")
print("=" * 60)

# 按 area（商圈）分组统计
area_stats = df.groupby('area').agg(
    均价=('totalprice', 'mean'),
    中位数=('totalprice', 'median'),
    最高价=('totalprice', 'max'),
    最低价=('totalprice', 'min'),
    房源数量=('totalprice', 'count'),
    平均面积=('size', 'mean'),
).round(1)

# 只保留房源数量 >= 5 的区域（样本太少不具有代表性）
area_stats = area_stats[area_stats['房源数量'] >= 5]
area_stats = area_stats.sort_values('均价', ascending=False)

print("\n均价最高的10个区域:")
top10_areas = area_stats.head(10)
print(top10_areas.to_string())
top10_areas.to_csv('output_table_top10_areas.csv', encoding='utf-8-sig')

# 图13: 均价Top10区域柱状图
fig, ax = plt.subplots(figsize=(12, 6))
top10_reversed = top10_areas.iloc[::-1]
bars = ax.barh(top10_reversed.index, top10_reversed['均价'],
               color=['#E74C3C' if i >= len(top10_reversed) - 3 else '#3498DB' for i in range(len(top10_reversed))],
               alpha=0.85)
for bar, val in zip(bars, top10_reversed['均价']):
    ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
            f'{val:.0f}万', va='center', fontsize=10)
ax.set_xlabel('均价（万元）', fontsize=12)
ax.set_title('深圳二手房均价最高的10个商圈', fontsize=14)
plt.tight_layout()
plt.savefig('fig13_top10_areas.png', dpi=150, bbox_inches='tight')
plt.close()
print(">>> 已保存: fig13_top10_areas.png")

# 完整区域排名（所有满足条件的区域）
print(f"\n全部 {len(area_stats)} 个区域（房源≥5）的均价排名已保存")
area_stats.to_csv('output_table_all_areas_rank.csv', encoding='utf-8-sig')

# 图14: 各区域房源数量与均价关系气泡图（Top30）
fig, ax = plt.subplots(figsize=(14, 8))
top30 = area_stats.head(30)
scatter = ax.scatter(top30['房源数量'], top30['均价'],
                     s=top30['平均面积'] * 2, alpha=0.6,
                     c=top30['均价'], cmap='YlOrRd', edgecolors='black', linewidth=0.5)
for area_name, row in top30.iterrows():
    ax.annotate(area_name, (row['房源数量'], row['均价']),
                fontsize=7, ha='center', va='bottom',
                xytext=(0, 5), textcoords='offset points')
ax.set_xlabel('房源数量', fontsize=12)
ax.set_ylabel('均价（万元）', fontsize=12)
ax.set_title('商圈房源数量 vs 均价（气泡大小=平均面积）', fontsize=14)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('均价（万元）', fontsize=10)
plt.tight_layout()
plt.savefig('fig14_area_bubble.png', dpi=150, bbox_inches='tight')
plt.close()
print(">>> 已保存: fig14_area_bubble.png")

# ============================================================
# 4. 保存报告所需的关键结果数据
# ============================================================
print("\n" + "=" * 60)
print("保存报告所需关键数据...")

# 任务2结果
with open('report_key_results.txt', 'w', encoding='utf-8') as f:
    f.write("===== 关键结果汇总 =====\n\n")
    f.write("【任务2: Top3影响因素】\n")
    for i, (_, row) in enumerate(top3.iterrows()):
        f.write(f"  {i+1}. {row['特征']}: 重要性={row['重要性']:.4f} ({row['重要性(%)']:.2f}%)\n")

    f.write("\n【任务3: 均价Top10区域】\n")
    for i, (area_name, row) in enumerate(top10_areas.iterrows()):
        f.write(f"  {i+1}. {area_name}: 均价{row['均价']:.0f}万, 中位数{row['中位数']:.0f}万, 房源{int(row['房源数量'])}套\n")

    # 模型最佳性能
    from sklearn.metrics import r2_score

    test_pred = np.load('test_predictions.npz', allow_pickle=True)
    y_test = test_pred['y_test']
    y_pred = test_pred['y_pred']

    def mape(y_true, y_pred):
        mask = y_true != 0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    final_r2 = r2_score(y_test, y_pred)
    final_mape = mape(y_test, y_pred)

    f.write(f"\n【模型最佳性能】\n")
    f.write(f"  R² = {final_r2:.4f}\n")
    f.write(f"  MAPE = {final_mape:.2f}%\n")

print(">>> 已保存: report_key_results.txt")

print("\n" + "=" * 60)
print("结果分析完成！")
print(f"  Top3 因素已确定")
print(f"  Top10 区域排名已生成")
print(f"  所有图表和数据表已保存")
print("=" * 60)