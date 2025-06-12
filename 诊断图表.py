#%%准备工作
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.stats as stats
import statsmodels.stats.stattools as smt
import scipy.stats
from stargazer.stargazer import Stargazer, LineLocation

# Set path to working directory of this script file
path_file = os.path.dirname(os.path.abspath(__file__))
os.chdir(path_file)

# 导入数据：全样本数据和行业投资组合数据
df = pd.read_csv("data\\full_data_new.csv", index_col='Date')

#%% 构建模型 --------------------------------------------------------------------------------
#>> Fama-French 3 Factor Model
# 定义解释变量 (Fama-French 3因子变量)
X = df[['Mkt-RF', 'SMB', 'HML']]
# 给模型添加一个连续的截距
X = sm.add_constant(X)
# 定义被解释变量 (超额回报)
y = df['excess_return']

# 进行回归分析
model_FF3 = sm.OLS(y, X).fit()
print("3因子模型基准回归：")

#%%QQ图
#>> Q-Q Plot
# 定义了 gen_QQ_plot 函数，用于绘制模型残差的 Q-Q 图，以检查残差是否符合正态分布。
# Q-Q 图将实际残差与理论正态分布进行比较，若残差接近正态分布，则点会大致沿着对角线分布。
def gen_QQ_plot(mod_res):
    df_res = pd.DataFrame(sorted(mod_res), columns=['residual'])
    # Calculate the Z-score for the residuals
    df_res['z_actual'] = (df_res['residual'].map(lambda x: (x - df_res['residual'].mean()) / df_res['residual'].std()))
    # Calculate the theoretical Z-scores
    df_res['rank'] = df_res.index + 1
    df_res['percentile'] = df_res['rank'].map(lambda x: x/len(df_res.residual))
    df_res['theoretical'] = scipy.stats.norm.ppf(df_res['percentile'])
    # Construct QQ plot
    with plt.style.context('ggplot'):
        plt.figure(figsize=(9,9))
        plt.scatter(df_res['theoretical'], df_res['z_actual'], color='blue')
        plt.xlabel('Theoretical Quantile')
        plt.ylabel('Sample Quantile')
        plt.title('Normal QQ Plot')
        plt.plot(df_res['theoretical'], df_res['theoretical'])
        plt.gca().set_facecolor('white')    # (0.95, 0.95, 0.95)
        plt.gca().spines['top'].set_color('black')
        plt.gca().spines['bottom'].set_color('black')
        plt.gca().spines['left'].set_color('black')
        plt.gca().spines['right'].set_color('black')
        plt.savefig(path_file + "\\output\\QQ_plot.png")
        plt.show()
    return(df_res)

gen_QQ_plot(model_FF3 .resid)

#%%#>>部分回归图
# 这些图可以帮助评估每一个自变量在模型中的独立影响，同时控制其他自变量的影响。
# Residual plots
fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_partregress_grid(model_FF3 , fig=fig)
fig.savefig(path_file + "\\output\\partial_reg_plots.png")
fig

#%%#>> Residuals vs. Fitted Plot
# 绘制了残差与拟合值之间的关系图，用于检查残差是否存在系统性模式，从而判断模型是否满足线性回归的假设。
with plt.style.context('ggplot'):
    plt.figure(figsize=(9,9))
    plt.scatter(model_FF3.fittedvalues, model_FF3.resid, color='orange')
    plt.xlabel('Predicted Value')
    plt.ylabel('Residual')
    plt.title('Residual by Predicted')
    plt.axhline(y = 0, color = 'black', linestyle = '-') 
    plt.gca().set_facecolor('white')    # (0.95, 0.95, 0.95)
    plt.gca().spines['top'].set_color('black')
    plt.gca().spines['bottom'].set_color('black')
    plt.gca().spines['left'].set_color('black')
    plt.gca().spines['right'].set_color('black')
    plt.savefig(path_file + "\\output\\fitted_res_plot.png")
    plt.show()

#%%#>> Scale-Location Plot
# 绘制了拟合值与标准化残差平方根之间的关系图，用于检测残差的方差是否随拟合值的增加而变化，即是否存在异方差性。
with plt.style.context('ggplot'):
    plt.figure(figsize=(9,9))
    plt.scatter(model_FF3.fittedvalues, np.sqrt(model_FF3.resid), color='orange')
    plt.xlabel('Predicted Values')
    plt.ylabel('Standardized Residuals')
    plt.title('Scale-Location')
    plt.gca().set_facecolor('white')    # (0.95, 0.95, 0.95)
    plt.gca().spines['top'].set_color('black')
    plt.gca().spines['bottom'].set_color('black')
    plt.gca().spines['left'].set_color('black')
    plt.gca().spines['right'].set_color('black')
    plt.savefig(path_file + "\\output\\scale_location.png")
    plt.show()

