import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# 加载数据
df = pd.read_csv("data/full_data_new.csv", index_col='Date', parse_dates=True)

Harvey-Collier 检验用来直接比较两个或多个解释变量对同一个因变量的相对解释力或重要性
# 联合检验
from statsmodels.stats.diagnostic import linear_harvey_collier

# 重新构建三因子模型
df['Intercept']=1
model_three_factor = sm.OLS(df['excess_return'], df[['Intercept', 'Mkt-RF', 'SMB', 'HML']])
result_three_factor = model_three_factor.fit()

# Harvey-Collier 检验
harvey_collier_test = linear_harvey_collier(result_three_factor)
print("\nHarvey-Collier 检验结果 (联合检验):")
print("Test Statistic:", harvey_collier_test[0])
print("p-value:", harvey_collier_test[1])
# 稳健性检验：使用正交化因子 RMO
# 构建正交化因子 RMO
X = df[['Mkt-RF', 'SMB', 'HML']]
y = df['Mkt-RF']
X = sm.add_constant(X)
model_orthogonal = sm.OLS(y, X)
result_orthogonal = model_orthogonal.fit()

# 提取残差作为正交化因子 RMO
df['RMO'] = result_orthogonal.resid

# 使用正交化因子 RMO 进行稳健性回归
model_ortho = sm.OLS(df['excess_return'], df[['Intercept', 'RMO', 'SMB', 'HML']])
result_ortho = model_ortho.fit()

# 输出稳健性回归结果
print("\n稳健性回归结果 - 使用正交化因子 RMO:")
print(result_ortho.summary())
# 绘制稳健性回归的拟合值与实际值
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['excess_return'], label='real excess returns')
plt.plot(df.index, result_ortho.fittedvalues, label='fitting excess returns')
plt.legend()
plt.title('regression of excess stock returns(in percent) on the SMB, HML and RMO factors')
plt.show()
# 加载数据
data = pd.read_csv('data/full_data_new.csv', parse_dates=['Date'])
data.set_index('Date', inplace=True)
# 一月效应检验
# 创建一月虚拟变量
data['January'] = data.index.month == 1

# 将一月虚拟变量转换为整数
data['January'] = data['January'].astype(int)

# 创建回归模型
X = data[['Mkt-RF', 'SMB', 'HML', 'January']]
y = data['excess_return']

# 添加常数项
X = sm.add_constant(X)

# 拟合模型
model = sm.OLS(y, X).fit()

# 打印回归结果
print(model.summary())
from datetime import datetime

# 加载数据
data = pd.read_csv('data/full_data_new.csv')  # 假设日期列名为'Date'
print(data.columns)  # 打印列名，检查是否正确加载

# 将'Date'列转换为日期格式
data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')
data.set_index('Date', inplace=True)

# 确保'Date'列现在是索引
print(data.index)  # 打印索引，检查是否正确设置
# 分样本回归
# 数据分组
mid_year = 2002
mid_month = 6

# 找到2002年6月的索引
mid_date = datetime(mid_year, mid_month, 1)
group1 = data[data.index < mid_date]
group2 = data[data.index >= mid_date]

# 定义回归函数
def run_regression(data):
    X = data[['Mkt-RF', 'SMB', 'HML']]
    y = data['excess_return']
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return model

# 对两组数据分别进行回归
model1 = run_regression(group1)
model2 = run_regression(group2)

# 输出回归结果
print('First Half Regression Results:')
print(model1.summary())

print('\nSecond Half Regression Results:')
print(model2.summary())

df = pd.read_csv("data\\full_data_new.csv", index_col='Date')
#>> 用VIF诊断多重共线性
def calc_VIF(df_exog):
    '''
    Parameters
    ----------
    df_exog : dataframe, (n_obs, k_vars)
        Design matrix with all explanatory variables used in a regression model.

    Returns
    -------
    VIF : Series
        Variance inflation factors
    '''
    # Compute the correlation matrix for the covariates
    corr_matrix = df_exog.corr().to_numpy()
    # Compute the inverse of the correlation matrix & extract only the diagonal elements
    inv_corr = np.linalg.inv(corr_matrix).diagonal()
    vifs = pd.Series(inv_corr, index=df_exog.columns, name='VIF')
    return vifs

X = df[['Mkt-RF', 'SMB', 'HML']]
calc_VIF(X)
import statsmodels.stats as stats
import statsmodels.stats.stattools as smt
import scipy.stats
from stargazer.stargazer import Stargazer, LineLocation
#>> Breusch-Pagan (1979) & White (1980) Tests BP和white检验是否异方差
def run_diag_test(type, resid, exog):
    if type == "BP":
        test_results = stats.diagnostic.het_breuschpagan(resid, exog)
        test_labels = ['Lagrange multiplier statistic', 'p-value', 'F-Statistic', 'F-Statistic p-value']
        title = "Breusch-Pagan Test Results"
    elif type == "White":
        test_results = stats.diagnostic.het_white(resid, exog)
        test_labels = ['Test Statistic', 'Test Statistic p-value', 'F-Statistic', 'F-Statistic p-value']
        title = "White Test Results"
    test_results = [round(i, 4) for i in list(test_results)]
    output = dict(zip(test_labels, test_results))
    print(title)
    for key, value in output.items():
        print(f"{key}: {value}")

# 定义解释变量 (Fama-French 3因子变量)
X = df[['Mkt-RF', 'SMB', 'HML']]
# 给模型添加一个连续的截距
X = sm.add_constant(X)
# 定义被解释变量 (超额回报)
y = df['excess_return']
model_FF3 = sm.OLS(y, X).fit()

run_diag_test("BP", model_FF3.resid, model_FF3.model.exog)
run_diag_test("White", model_FF3.resid, model_FF3.model.exog)

stats.diagnostic.het_white(model_FF3.resid, model_FF3.model.exog)


#>> Other normality tests
tests = sm.stats.stattools.jarque_bera(model_FF3.resid)
print(pd.Series(tests, index=['Jarque-Bera', 'Chi^2 two-tail prob.', 'Skew', 'Kurtosis']))

import statsmodels.api as sm

# 定义解释变量 (Fama-French 3因子变量)
X = df[['Mkt-RF', 'SMB', 'HML']]

# 给模型添加一个截距项
X = sm.add_constant(X)

# 定义被解释变量 (超额回报)
y = df['excess_return']

# 拟合模型时指定使用HC3稳健标准误
model_FF3_robust = sm.OLS(y, X).fit(cov_type='HC3')

# 输出回归结果
print(model_FF3_robust.summary())
