"""
02_数据预处理 (Data Preprocessing)
完成缺失值处理、异常值处理、特征工程、编码、标准化、数据集划分
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import warnings
warnings.filterwarnings('ignore')

# 必须在读取数据前禁用 Copy-on-Write
pd.set_option('mode.copy_on_write', False)

# 中文字体设置
fm.fontManager.addfont('C:/Windows/Fonts/simhei.ttf')
fm._load_fontmanager(try_read_cache=False)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 加载数据
# ============================================================
DATA_PATH = r'.\house_shenzhen_s.csv'
df = pd.read_csv(DATA_PATH, encoding='GBK')
print(f"原始数据集: {df.shape}")
print(f"原始列: {list(df.columns)}")

# ============================================================
# 2. 删除 unitprice（数据泄露风险）
# ============================================================
print("\n[步骤1] 删除 unitprice 列（与 totalprice 存在直接计算关系，防止数据泄露）")
df.drop(columns=['unitprice'], inplace=True)
print(f"  删除后特征数: {df.shape[1]}")

# ============================================================
# 3. 缺失值处理
# ============================================================
print("\n[步骤2] 缺失值处理")
print(f"  处理前缺失值统计:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

# 缺失数量极少（最多3个），采用众数/中位数填充
for col in df.columns:
    if df[col].isnull().sum() > 0:
        # 判断是否为数值类型
        if pd.api.types.is_numeric_dtype(df[col]):
            med_val = df[col].median()
            df[col] = df[col].fillna(med_val)
            print(f"  {col}: 用中位数 {med_val:.1f} 填充缺失值")
        else:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            print(f"  {col}: 用众数 '{mode_val}' 填充缺失值")

print(f"  处理后缺失值总数: {df.isnull().sum().sum()}")

# ============================================================
# 4. 异常值处理
# ============================================================
print("\n[步骤3] 异常值处理")

# 4.1 year 异常值：year应该 > 1900（深圳第一批商品房约1980年后）
print(f"  year 列基本情况: min={df['year'].min()}, max={df['year'].max()}")
year_outliers = df[df['year'] < 1900]
print(f"  year < 1900 的异常样本数: {len(year_outliers)}")
if len(year_outliers) > 0:
    print(f"  异常 year 值: {year_outliers['year'].unique()}")
    # 将这些异常值替换为该城区对应中位数年份
    for idx in year_outliers.index:
        city_median_year = df[df['city'] == df.loc[idx, 'city']]['year'].median()
        df.loc[idx, 'year'] = city_median_year
    print(f"  已用各城区中位数年份填充异常 year 值")

# 4.2 totalprice 异常值处理（IQR方法，但保留高端房产信息，只处理极端值）
Q1 = df['totalprice'].quantile(0.01)
Q3 = df['totalprice'].quantile(0.99)
IQR = Q3 - Q1
lower_bound = Q1 - 3 * IQR
upper_bound = Q3 + 3 * IQR
price_outliers = df[(df['totalprice'] < lower_bound) | (df['totalprice'] > upper_bound)]
print(f"  totalprice 极端异常值（<{lower_bound:.0f} 或 >{upper_bound:.0f}）: {len(price_outliers)} 条")
# 对极端异常值进行截断处理（winsorize）
df['totalprice'] = df['totalprice'].clip(lower=lower_bound, upper=upper_bound)
print(f"  已对极端值进行截断处理")

# 4.3 size 异常值
size_outliers = df[df['size'] > 400]
print(f"  size > 400㎡ 的异常样本数: {len(size_outliers)}")
df = df[df['size'] <= 400].copy()  # 删除面积过大的异常样本（可能是别墅或数据错误）
print(f"  已删除 size > 400㎡ 的样本，当前样本数: {len(df)}")

# ============================================================
# 5. 特征工程
# ============================================================
print("\n[步骤4] 特征工程")

# 5.1 从 rooms 提取室数和厅数
print("  从 rooms 列提取数值特征...")
df['rooms_num'] = df['rooms'].str.extract(r'(\d+)室').astype(float).fillna(0).astype(int)
df['halls_num'] = df['rooms'].str.extract(r'(\d+)厅').astype(float).fillna(0).astype(int)
df['total_rooms_halls'] = df['rooms_num'] + df['halls_num']
print(f"  rooms_num 分布: {df['rooms_num'].value_counts().sort_index().to_dict()}")
print(f"  halls_num 分布: {df['halls_num'].value_counts().sort_index().to_dict()}")

# 5.2 处理 orientation（朝向）- 提取主要朝向
print("  处理 orientation（朝向）...")
# 定义主要朝向方向
def extract_main_orientation(ori):
    if pd.isna(ori):
        return '未知'
    ori = str(ori).strip()
    # 取第一个朝向作为主朝向
    first = ori.split()[0] if ' ' in ori else ori
    return first

df['orientation_main'] = df['orientation'].apply(extract_main_orientation)
print(f"  主朝向分布: {df['orientation_main'].value_counts().to_dict()}")

# 5.3 处理 height（楼层）- 已有"低/中/高"但可能有其他值
print("  处理 height（楼层）...")
def simplify_height(h):
    h = str(h).strip()
    if h in ['低', '低楼层']:
        return '低'
    elif h in ['高', '高楼层']:
        return '高'
    elif h in ['中', '中楼层']:
        return '中'
    else:
        # 混合数据如"9层2011年建板"，提取层数判断
        try:
            floor = int(h.split('层')[0])
            if floor <= 5:
                return '低'
            elif floor <= 15:
                return '中'
            else:
                return '高'
        except:
            return '中'

df['height_simple'] = df['height'].apply(simplify_height)
print(f"  height 处理后分布: {df['height_simple'].value_counts().to_dict()}")
print(f"  height 处理后缺失: {df['height_simple'].isnull().sum()}")

# 5.4 计算房龄（用2026年作为参考年）
df['building_age'] = 2026 - df['year']
print(f"  building_age: min={df['building_age'].min():.0f}, max={df['building_age'].max():.0f}, mean={df['building_age'].mean():.1f}")

# 5.5 position 列 - 小区名称太多（1817个唯一值），暂不直接使用
# 保留但不编码，后续模型中不使用（area 已是有效的区域特征）
print(f"  position 列唯一值: {df['position'].nunique()}，因基数太大不作为模型特征")

# ============================================================
# 6. 特征选择与目标变量分离
# ============================================================
print("\n[步骤5] 特征选择")

# 目标变量
y = df['totalprice'].values

# 选择用于建模的特征列
feature_cols = [
    'size',           # 面积
    'rooms_num',      # 室数
    'halls_num',      # 厅数
    'total_rooms_halls',  # 室+厅总数
    'building_age',   # 房龄
    'follow',         # 关注人数
]

# 类别特征（需要编码）
cat_cols = [
    'city',           # 城区
    'area',           # 商圈
    'decoration',     # 装修
    'elevator',       # 电梯
    'orientation_main',  # 主朝向
    'height_simple',  # 楼层
    'buildtype',      # 建筑类型
]

print(f"  数值特征: {feature_cols}")
print(f"  类别特征: {cat_cols}")

# ============================================================
# 7. 类别特征编码（One-Hot / Label Encoding）
# ============================================================
print("\n[步骤6] 类别特征编码")

# 对类别特征进行 One-Hot 编码
df_encoded = df[feature_cols].copy()
cat_dfs = []

for col in cat_cols:
    # 先转换为普通 object 类型，避免 StringArray 的兼容性问题
    cat_values = df[col].astype(str).astype('object')
    # 对于高基数特征 area（83个值），使用 Label Encoding 减少维度
    if col == 'area':
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        df_encoded[f'{col}_encoded'] = le.fit_transform(cat_values)
        print(f"  {col}: Label Encoding, 类别数={len(le.classes_)}")
    else:
        # 其他类别特征使用 One-Hot 编码
        dummies = pd.get_dummies(cat_values, prefix=col, drop_first=True)
        cat_dfs.append(dummies)
        print(f"  {col}: One-Hot Encoding, 生成 {dummies.shape[1]} 个特征")

df_encoded = pd.concat([df_encoded] + cat_dfs, axis=1)
X = df_encoded.values
feature_names = list(df_encoded.columns)
print(f"\n  编码后特征总数: {len(feature_names)}")
print(f"  编码后 NaN 数量: {df_encoded.isnull().sum().sum()}")

# ============================================================
# 8. 数值特征标准化
# ============================================================
print("\n[步骤7] 数值特征标准化（StandardScaler）")
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
# 仅标准化原始数值特征部分
num_feature_count = len(feature_cols)
X[:, :num_feature_count] = scaler.fit_transform(X[:, :num_feature_count])
print(f"  已对前 {num_feature_count} 个数值特征进行标准化")

# ============================================================
# 9. 数据集划分
# ============================================================
print("\n[步骤8] 数据集划分（训练集:测试集 = 8:2）")
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"  训练集: {X_train.shape[0]} 样本")
print(f"  测试集: {X_test.shape[0]} 样本")

# ============================================================
# 10. 保存预处理后的数据
# ============================================================
print("\n[步骤9] 保存预处理结果")
import joblib

np.savez('preprocessed_data.npz',
         X_train=X_train, X_test=X_test,
         y_train=y_train, y_test=y_test,
         feature_names=np.array(feature_names))

# 保存标准化器和编码器以便后续使用
joblib.dump(scaler, 'scaler.pkl')

# 保存预处理相关信息
prep_info = {
    'feature_names': feature_names,
    'num_feature_count': num_feature_count,
    'train_size': len(X_train),
    'test_size': len(X_test),
}
pd.Series(prep_info).to_csv('preprocessing_info.csv', encoding='utf-8-sig')

print(f"\n  特征列表 ({len(feature_names)} 个):")
for i, name in enumerate(feature_names):
    print(f"    [{i}] {name}")

print("\n" + "=" * 60)
print("数据预处理完成！")
print(f"  最终特征维度: {X_train.shape[1]}")
print(f"  训练集: {X_train.shape[0]} 样本")
print(f"  测试集: {X_test.shape[0]} 样本")
print("=" * 60)