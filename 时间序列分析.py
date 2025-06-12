import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import acf
import statsmodels.api as sm
import matplotlib.pyplot as plt

# 导入数据：全样本数据和行业投资组合数据
df = pd.read_csv("data\\full_data_new.csv", index_col='Date')

# 选择三因子：Mkt-RF, SMB, HML
factors = df[['Mkt-RF', 'SMB', 'HML']]
# 仅使用市场超额收益 (RM-RF) 进行回归
df['Intercept'] = 1
model_market = sm.OLS(df['excess_return'], df[['Intercept', 'Mkt-RF']])
result_market = model_market.fit()

# 输出回归结果
print("回归结果 - 仅使用市场超额收益 (RM-RF):")
print(result_market.summary())

# 绘制拟合值与实际值
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['excess_return'], label='real excess returns')
plt.plot(df.index, result_market.fittedvalues, label='fitting excess returns')
plt.legend()
plt.title('regression of excess stock returns(in percent) on the excess stock-market return(RM-RF)')
plt.show()
# 仅使用 SMB 和 HML 进行回归
model_smb_hml = sm.OLS(df['excess_return'], df[['Intercept', 'SMB', 'HML']])
result_smb_hml = model_smb_hml.fit()

# 输出回归结果
print("\n回归结果 - 仅使用 SMB 和 HML:")
print(result_smb_hml.summary())

# 绘制拟合值与实际值
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['excess_return'], label='real excess returns')
plt.plot(df.index, result_smb_hml.fittedvalues, label='fitting excess returns')
plt.legend()
plt.title('regression of excess stock returns(in percent) on the SMB and HML factors')
plt.show()
# 同时使用 RM-RF、SMB 和 HML 进行回归
model_three_factor = sm.OLS(df['excess_return'], df[['Intercept', 'Mkt-RF', 'SMB', 'HML']])
result_three_factor = model_three_factor.fit()

# 输出回归结果
print("\n回归结果 - 三因子模型 (RM-RF, SMB, HML):")
print(result_three_factor.summary())

# 绘制拟合值与实际值
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['excess_return'], label='real excess returns')
plt.plot(df.index, result_three_factor.fittedvalues, label='fitting excess returns')
plt.legend()
plt.title('regression of excess stock returns(in percent) on the RM-RF, SMB and HML factors')
plt.show()
# 构建正交化因子 RMO
X = df[['Mkt-RF', 'SMB', 'HML']]
y = df['Mkt-RF']
X = sm.add_constant(X)
model_orthogonal = sm.OLS(y, X)
result_orthogonal = model_orthogonal.fit()

# 提取残差作为正交化因子 RMO
df['RMO'] = result_orthogonal.resid

# 使用正交化因子 RMO 进行回归
model_ortho = sm.OLS(df['excess_return'], df[['Intercept', 'RMO', 'SMB', 'HML']])
result_ortho = model_ortho.fit()

# 输出回归结果
print("\n回归结果 - 使用正交化因子 RMO:")
print(result_ortho.summary())

# 绘制拟合值与实际值
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['excess_return'], label='real excess returns')
plt.plot(df.index, result_ortho.fittedvalues, label='fitting excess returns')
plt.legend()
plt.title('regression of excess stock returns(in percent) on the SMB, HML and RMO factors')
plt.show()
