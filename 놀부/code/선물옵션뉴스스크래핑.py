pip install pykrx
from pykrx import stock
from pykrx import bond

tickers = stock.get_market_ticker_list("20220314")
print(tickers)

tickers = stock.get_market_ticker_list("20190225", market="KOSPI")
print(tickers)

for ticker in stock.get_market_ticker_list():
        종목 = stock.get_market_ticker_name(ticker) # 
        print(종목)
      
df = stock.get_market_ohlcv("20210331", "20220331", "005930") # ('시작일','종료일','티커')
print(df.head(3))

