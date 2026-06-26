"""
01_数据探索 (Data Exploration)
对深圳二手房数据集进行初步探索性数据分析（EDA）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 0. 全局设置：中文字体 SimHei（显式加载字体文件）
# ============================================================
fm.fontManager.addfont('C:/Windows/Fonts/simhei.ttf')
fm._load_fontmanager(try_read_cache=False)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
ZH_FONT = fm.FontProperties(fname='C:/Windows/Fonts/simhei.ttf')


def set_axis_chinese(ax, title=None, xlabel=None, ylabel=None, legend=False):
    """显式为坐标轴中的中文文字指定字体，避免保存图片时回退成方框。"""
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
# 1. 加载数据
# ============================================================
DATA_PATH = r'.\house_shenzhen_s.csv'
df = pd.read_csv(DATA_PATH, encoding='GBK')
print("=" * 60)
print("【数据集基本信息】")
print(f"样本数: {df.shape[0]}, 特征数: {df.shape[1]}")
print(f"\n列名: {list(df.columns)}")

# ============================================================
# 2. 数据类型与缺失值分析
# ============================================================
print("\n" + "=" * 60)
print("【数据类型与缺失值统计】")
info_df = pd.DataFrame({
    '数据类型': df.dtypes,
    '缺失值数量': df.isnull().sum(),
    '缺失值比例(%)': (df.isnull().sum() / len(df) * 100).round(2),
    '唯一值数量': df.nunique()
})
print(info_df.to_string())
# 保存到文件供报告使用
info_df.to_csv('output_table_dtypes.csv', encoding='utf-8-sig')

# ============================================================
# 3. 目标变量 totalprice 分布分析
# ============================================================
print("\n" + "=" * 60)
print("【目标变量 totalprice 统计描述】")
print(df['totalprice'].describe())

# 图1: totalprice 分布直方图 + 箱线图
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df['totalprice'], bins=80, color='steelblue', edgecolor='white', alpha=0.85)
axes[0].axvline(df['totalprice'].mean(), color='red', linestyle='--', linewidth=2, label=f'均值: {df["totalprice"].mean():.1f}万')
axes[0].axvline(df['totalprice'].median(), color='orange', linestyle='--', linewidth=2, label=f'中位数: {df["totalprice"].median():.1f}万')
axes[0].legend(fontsize=10)
set_axis_chinese(axes[0], '深圳二手房总价分布直方图', '房屋总价（万元）', '频数', legend=True)

axes[1].boxplot(df['totalprice'].dropna(), vert=True, patch_artist=True,
                boxprops=dict(facecolor='steelblue', alpha=0.7))
set_axis_chinese(axes[1], '深圳二手房总价箱线图', None, '房屋总价（万元）')

plt.tight_layout()
plt.savefig('fig1_totalprice_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print(">>> 已保存: fig1_totalprice_distribution.png")

# ============================================================
# 4. 数值型特征分布
# ============================================================
numeric_cols = ['size', 'year', 'follow']
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
colors = ['forestgreen', 'darkorange', 'purple']
for i, col in enumerate(numeric_cols):
    data = df[col].dropna()
    axes[i].hist(data, bins=50, color=colors[i], edgecolor='white', alpha=0.85)
    axes[i].axvline(data.mean(), color='red', linestyle='--', linewidth=1.5, label=f'均值: {data.mean():.1f}')
    axes[i].legend(fontsize=9)
    set_axis_chinese(axes[i], f'{col} 分布', col, '频数', legend=True)
plt.tight_layout()
plt.savefig('fig2_numeric_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print(">>> 已保存: fig2_numeric_distribution.png")

# ============================================================
# 5. 类别型特征分布
# ============================================================
# 图3: 城区分布
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

city_order = df['city'].value_counts().index
axes[0].bar(city_order, df['city'].value_counts().values, color='teal', alpha=0.85)
axes[0].tick_params(axis='x', rotation=30)
set_axis_chinese(axes[0], '各城区房源数量分布', '城区', '房源数量')

# 装修情况
deco_order = df['decoration'].value_counts().index
axes[1].bar(deco_order, df['decoration'].value_counts().values, color='coral', alpha=0.85)
set_axis_chinese(axes[1], '装修情况分布', '装修情况', '房源数量')

# 电梯
elev_order = df['elevator'].value_counts().index
axes[2].bar(elev_order, df['elevator'].value_counts().values, color='mediumpurple', alpha=0.85)
set_axis_chinese(axes[2], '电梯配置分布', '电梯', '房源数量')

plt.tight_layout()
plt.savefig('fig3_categorical_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print(">>> 已保存: fig3_categorical_distribution.png")

# ============================================================
# 6. 各城区均价分析
# ============================================================
city_avg = df.groupby('city')['totalprice'].agg(['mean', 'median', 'count'])
city_avg.columns = ['均价(万)', '中位数(万)', '房源数']
city_avg = city_avg.sort_values('均价(万)', ascending=False)
print("\n" + "=" * 60)
print("【各城区均价排名】")
print(city_avg.to_string())
city_avg.to_csv('output_table_city_avg.csv', encoding='utf-8-sig')

# 图4: 各城区均价柱状图
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(city_avg.index, city_avg['均价(万)'], color=['#E74C3C' if i == 0 else '#3498DB' for i in range(len(city_avg))], alpha=0.85)
for bar, val in zip(bars, city_avg['均价(万)']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, f'{val:.0f}万', ha='center', fontsize=10)
ax.tick_params(axis='x', rotation=30)
for text in ax.texts:
    text.set_fontproperties(ZH_FONT)
set_axis_chinese(ax, '深圳各城区二手房均价对比', '城区', '均价（万元）')
plt.tight_layout()
plt.savefig('fig4_city_avg_price.png', dpi=150, bbox_inches='tight')
plt.close()
print(">>> 已保存: fig4_city_avg_price.png")

# ============================================================
# 7. 数值特征与target的相关性
# ============================================================
corr_cols = ['totalprice', 'size', 'year', 'follow']
corr_matrix = df[corr_cols].corr()
print("\n" + "=" * 60)
print("【数值特征与总价的相关性矩阵】")
print(corr_matrix.to_string())

# 图5: 相关性热力图
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
            square=True, linewidths=0.8, cbar_kws={'shrink': 0.8}, ax=ax)
set_axis_chinese(ax, '数值特征与总价的相关性热力图')
plt.tight_layout()
plt.savefig('fig5_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print(">>> 已保存: fig5_correlation_heatmap.png")

# ============================================================
# 8. size vs totalprice 散点图（按城区着色）
# ============================================================
fig, ax = plt.subplots(figsize=(10, 7))
cities = df['city'].unique()
colors_map = plt.cm.tab10(np.linspace(0, 1, len(cities)))
for i, c in enumerate(cities):
    subset = df[df['city'] == c]
    ax.scatter(subset['size'], subset['totalprice'], c=[colors_map[i]], label=c, alpha=0.5, s=15)
ax.legend(loc='lower right', fontsize=8, ncol=2)
set_axis_chinese(ax, '面积与总价关系（按城区着色）', '面积（㎡）', '总价（万元）', legend=True)
plt.tight_layout()
plt.savefig('fig6_size_vs_price.png', dpi=150, bbox_inches='tight')
plt.close()
print(">>> 已保存: fig6_size_vs_price.png")

# ============================================================
# 9. 房龄 vs 总价
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
year_data = df[df['year'] > 1900].copy()  # 排除异常值
year_data['year_group'] = (year_data['year'] // 5 * 5).astype(int)
year_price = year_data.groupby('year_group')['totalprice'].agg(['mean', 'std', 'count'])
year_price.columns = ['均价', '标准差', '数量']
ax.bar(year_price.index.astype(str), year_price['均价'], color='steelblue', alpha=0.85, yerr=year_price['标准差'])
ax.tick_params(axis='x', rotation=45)
set_axis_chinese(ax, '不同建造年份的房屋均价', '建造年份（5年分组）', '均价（万元）')
plt.tight_layout()
plt.savefig('fig7_year_vs_price.png', dpi=150, bbox_inches='tight')
plt.close()
print(">>> 已保存: fig7_year_vs_price.png")

# ============================================================
# 10. height（楼层高低）对价格的影响
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))
height_order = ['低', '中', '高']
height_data = df[df['height'].isin(height_order)]
height_avg = height_data.groupby('height')['totalprice'].mean().reindex(height_order)
bars = ax.bar(height_avg.index, height_avg.values, color=['#27AE60', '#F39C12', '#E74C3C'], alpha=0.85)
for bar, val in zip(bars, height_avg.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, f'{val:.1f}万', ha='center', fontsize=11)
for text in ax.texts:
    text.set_fontproperties(ZH_FONT)
set_axis_chinese(ax, '不同楼层高度的房屋均价', '楼层高度', '均价（万元）')
plt.tight_layout()
plt.savefig('fig8_height_vs_price.png', dpi=150, bbox_inches='tight')
plt.close()
print(">>> 已保存: fig8_height_vs_price.png")

print("\n" + "=" * 60)
print("数据探索完成！共生成 8 张图表和 2 个数据表格文件。")
print("=" * 60)
