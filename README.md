# -
Fama-French三因子模型验证——基于S&amp;P500
本项目基于S&P500股票数据数据，对Fama-French三因子模型进行实证分析，并进一步分行业回归。
股票数据来自雅虎财经，使用Python的yfinance库可直接调用。
三因子数据来自于Ken French官方数据库。
10年期国债利率等美国经济数据来自于FRED数据库，Python的fredapi可以直接调用数据。但需要先获取FREDapi（https://fred.stlouisfed.org/docs/api/api_key.html）并复制到“FRED_api_key_file.txt”文件中，即可以运行“最新数据导入与预处理.py”文件。
