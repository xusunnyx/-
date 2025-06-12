import os
import pandas as pd
import numpy as np
from datetime import datetime
import yfinance as yahooFinance
from fredapi import Fred

# Set path to working directory of this script file
path_file = os.path.dirname(os.path.abspath(__file__))
os.chdir(path_file)

#%% 导入标普500指数数据-----------------------------------------------------------------------------------

#>> 下载标普500指数数据(Date, Ticker, and Return columns)
df_SP = yahooFinance.Ticker("^SPX").history(start='1990-01-01', interval='1mo', actions=True)

# 计算月度对数收益率：当月收盘价除以上月收盘价取自然对数并乘以100
# close为收盘价，volume为成交量，returns为收益回报
df_SP["Returns"] = np.log(df_SP["Close"]/df_SP["Close"].shift(1)) * 100
df_SP = df_SP[["Close", "Volume", "Returns"]]
print(df_SP.head())

# 提取第一行数据的日期
start_date = datetime.strftime(df_SP.index[0], '%m/%d/%Y'); start_date

#%% 直接导入ken french的数据(5因子)
url_FF5 = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"
df_FF5 = pd.read_csv(url_FF5, compression="zip", skiprows=3)
print(df_FF5.head())

#>>导入fred数据
API_key = open(path_file + "/FRED_api_key_file.txt", "r").read()
fred = Fred(api_key=API_key)

def get_FRED_data(tickers):
    df = pd.DataFrame()
    for k,v in tickers.items():
        series = fred.get_series(k, observation_start=start_date, frequency='m', aggregation_method='eop')
        series.name = v
        df = df.join(series, how='outer')
    return(df)

tickers = {
    'DGS10':'rf', #10年期国债收益率（无风险利率rf）
    'EXPINF10YR':'expected_cpi', #预期CPI
    # 'GDPC1':'real_gdp',
}
df_FRED = get_FRED_data(tickers)
df_FRED.head()

# 数据清洗与处理
#>> Clean FF data
df_FF5.rename(columns={'Unnamed: 0':'Date'}, inplace=True) #重命名列名
#去除年度因子部分
string_location = df_FF5[df_FF5['Date'].str.contains("Annual Factors: January-December", na=False)].index[0]
df_FF5 = df_FF5[:string_location]
#转换日期格式
df_FF5['Date'] = pd.to_datetime(df_FF5['Date'], format='%Y%m')
df_FF5.set_index('Date', inplace=True)
df_FF5 = df_FF5.apply(pd.to_numeric, errors='coerce')

# Normalize timezone in datetype indexes
df_FF5 = df_FF5.tz_localize(None)
df_SP = df_SP.tz_localize(None)
df_FRED

#>> Combine data series & compute excess returns
df = df_FF5.join(df_SP, how='inner')
df["excess_return"] = df["Returns"] - df["RF"] #计算超额收益
# 去除缺失值的数据
df = df.dropna()

# 处理得到基础全部数据full_data
df.to_csv(path_file + '\\data\\full_data_new.csv')
print("Completed.")
