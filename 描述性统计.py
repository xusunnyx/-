import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import acf
from scipy import stats

# 导入数据：全样本数据和行业投资组合数据
df = pd.read_csv("data\\full_data_new.csv", index_col='Date')

# 选择三因子：Mkt-RF, SMB, HML
factors = df[['Mkt-RF', 'SMB', 'HML']]
# ====================
# 1. 描述性统计 (包含t统计量)
# ====================
def calculate_t_stat(series):
    """计算均值的t统计量"""
    n = len(series)
    t_stat, p_value = stats.ttest_1samp(series, 0)
    return t_stat

desc_stats = pd.DataFrame({
    'Mean': factors.mean(),
    'Std Dev': factors.std(),
    't-Statistic': factors.apply(calculate_t_stat),
    'Autocorrelation': factors.apply(lambda x: acf(x, nlags=1)[1])  # 滞后1阶自相关
})
# 打印结果
print("="*50)
print("因子描述性统计 (Fama-French三因子)")
print("="*50)
print(desc_stats.round(4))
# ====================
# 2. 相关系数矩阵 (Table 2下半部分)
# ====================
corr_matrix = factors.corr()

print("\n" + "="*50)
print("因子相关系数矩阵")
print("="*50)
print(corr_matrix.round(4))
# ====================
# 3. 被解释变量统计 (excess_return)
# ====================
excess_return = df['excess_return']
excess_return_stats = pd.DataFrame({
    'excess_return': {
        'Mean': excess_return.mean(),
        'Std Dev': excess_return.std(),
        't-Statistic': calculate_t_stat(excess_return),
        'Autocorrelation': acf(excess_return, nlags=1)[1]
    }
}).T

print("\n" + "="*50)
print("被解释变量描述性统计 (标普500超额收益)")
print("="*50)
print(excess_return_stats.round(4))
# ====================
# 4. 因子波动性分析
# ====================
volatility_stats = pd.DataFrame({
    'Annualized Volatility': factors.std() * np.sqrt(12)
})
print("\n" + "="*50)
print("因子年化波动率")
print("="*50)
print(volatility_stats.round(4))
