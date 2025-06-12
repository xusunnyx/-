#%% Preliminaries ---------------------------------------------------------------------------------
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
df_ind = pd.read_csv('data\\industry_portfolios.csv', index_col= 'Date')


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
model_FF3.summary()

# 输出为latex格式的表格
print(model_FF3.summary().as_latex())

#%%
#>> Sector Regressions
industry_results = {}

# Run industry models
for industry in df_ind.columns:
    # Subset industry
    y = df_ind[industry]
    # Find the common indices
    common_indices = X.index.intersection(y.index)
    # Filter both DataFrames to only include the common indices
    X_common = X.loc[common_indices]
    y_common = y.loc[common_indices]
    # Build regression model
    model = sm.OLS(y_common, X_common).fit()
    industry_results[industry] = model
    print('-'*10, industry, '-'*10)
    print(model.summary())

# Printout LaTeX table of sector regressions
mod_sectors = [industry_results['Technology'], industry_results['Energy'], industry_results['Financial Services']]
stargazer = Stargazer(mod_sectors)
stargazer.add_line('AIC', [mod_sectors[0].aic.round(2), mod_sectors[1].aic.round(2), mod_sectors[2].aic.round(2)], LineLocation.FOOTER_TOP)
stargazer.add_line('Skew', [smt.jarque_bera(mod_sectors[0].resid)[2].round(4), smt.jarque_bera(mod_sectors[1].resid)[2].round(4), smt.jarque_bera(mod_sectors[2].resid)[2].round(4)], LineLocation.FOOTER_TOP)
stargazer.add_line('Kurtosis', [smt.jarque_bera(mod_sectors[0].resid)[3].round(2), smt.jarque_bera(mod_sectors[1].resid)[3].round(2), smt.jarque_bera(mod_sectors[2].resid)[3].round(2)], LineLocation.FOOTER_TOP)
stargazer.add_line('Durbin-Watson', [smt.durbin_watson(mod_sectors[0].resid).round(3), smt.durbin_watson(mod_sectors[1].resid).round(3), smt.durbin_watson(mod_sectors[2].resid).round(3)], LineLocation.FOOTER_TOP)
print('='*10)
print(stargazer.render_latex())

# Printout the remaining models
mod_sectors = [
    industry_results['Healthcare'], 
    industry_results['Industrials'], 
    industry_results['Consumer Cyclical'], 
    industry_results['Consumer Defensive'], 
    industry_results['Utilities'], 
    industry_results['Basic Materials'], 
    industry_results['Real Estate'], 
    industry_results['Communication Services'], 
]
stargazer = Stargazer(mod_sectors)
stargazer.add_line('AIC', [
        mod_sectors[0].aic.round(2), 
        mod_sectors[1].aic.round(2), 
        mod_sectors[2].aic.round(2),
        mod_sectors[3].aic.round(2),
        mod_sectors[4].aic.round(2),
        mod_sectors[5].aic.round(2),
        mod_sectors[6].aic.round(2),
        mod_sectors[7].aic.round(2)
    ], LineLocation.FOOTER_TOP)
stargazer.add_line('Skew', [
        smt.jarque_bera(mod_sectors[0].resid)[2].round(4), 
        smt.jarque_bera(mod_sectors[1].resid)[2].round(4), 
        smt.jarque_bera(mod_sectors[2].resid)[2].round(4),
        smt.jarque_bera(mod_sectors[3].resid)[2].round(4),
        smt.jarque_bera(mod_sectors[4].resid)[2].round(4),
        smt.jarque_bera(mod_sectors[5].resid)[2].round(4),
        smt.jarque_bera(mod_sectors[6].resid)[2].round(4),
        smt.jarque_bera(mod_sectors[7].resid)[2].round(4),
    ], LineLocation.FOOTER_TOP)
stargazer.add_line('Kurtosis', [
        smt.jarque_bera(mod_sectors[0].resid)[3].round(2), 
        smt.jarque_bera(mod_sectors[1].resid)[3].round(2), 
        smt.jarque_bera(mod_sectors[2].resid)[3].round(2),
        smt.jarque_bera(mod_sectors[3].resid)[3].round(2),
        smt.jarque_bera(mod_sectors[4].resid)[3].round(2),
        smt.jarque_bera(mod_sectors[5].resid)[3].round(2),
        smt.jarque_bera(mod_sectors[6].resid)[3].round(2),
        smt.jarque_bera(mod_sectors[7].resid)[3].round(2),
    ], LineLocation.FOOTER_TOP)
stargazer.add_line('Durbin-Watson', [
        smt.durbin_watson(mod_sectors[0].resid).round(3), 
        smt.durbin_watson(mod_sectors[1].resid).round(3), 
        smt.durbin_watson(mod_sectors[2].resid).round(3),
        smt.durbin_watson(mod_sectors[3].resid).round(3),
        smt.durbin_watson(mod_sectors[4].resid).round(3),
        smt.durbin_watson(mod_sectors[5].resid).round(3),
        smt.durbin_watson(mod_sectors[6].resid).round(3),
        smt.durbin_watson(mod_sectors[7].resid).round(3),
    ], LineLocation.FOOTER_TOP)

print(stargazer.render_latex())

del(common_indices, industry, X_common, y_common)

