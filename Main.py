import pandas as pd
import Risk as risk
import yfinance as yf

# Define constants
TRAINING_PERIOD = 250

# Define tickers lists
tickers_technology = ['^GSPC','MSFT','AAPL','CSCO','INTC','ADBE','ORCL','SAP.DE','IBM','NVDA','ASML.AS']
tickers_banks = ['JPM','BAC','WFC','HSBA.L','C','RY.TO','TD.TO','CBA.AX','USB','SAN.MC']
tickers = tickers_technology + tickers_banks


# Download stock prices data from Yahoo Finance for the ticker
def download_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    stock_prices = stock_data['Adj Close']
    return stock_prices


# Download stock prices, compute VaR and ES and output result for each ticker
def process_ticker(ticker, training_start, testing_end_date):
    # Download training data
    stock_prices = download_data(ticker, training_start, testing_end_date)

    # Calculate the 1-day returns for training the VaR models
    daily_returns = stock_prices.pct_change(1).dropna()

    # Calculate the 10-days returns for training the ES models
    ten_days_returns = stock_prices.pct_change(10).dropna()

    df_result = pd.DataFrame(columns=['Day','Ticker','VaR','ES','Actual_Loss','Is VaR good','Is ES good'])
    no_of_returns = len(ten_days_returns)
    for day in range(1,no_of_returns):
        testing_data = daily_returns[-day]
        var_training_data = daily_returns[:-day].tail(TRAINING_PERIOD)
        es_training_data = ten_days_returns[:-day].tail(TRAINING_PERIOD)
        # Calculate VaR at all 5 confidence levels
        var_990 = risk.value_at_risk(var_training_data, 0.01)

        # Calculate Expected Shortfall
        es = risk.expected_shortfall(es_training_data, 0.025)

        date = daily_returns.index.values[-day]

        result = [date, ticker, var_990, es, testing_data, (testing_data > var_990), (testing_data > es)]
        df_result = df_result.append(pd.Series(result, index=df_result.columns), ignore_index=True)

    csv_filename = ticker + '_result.csv'
    df_result.to_csv(csv_filename)
    print(ticker + ' done')
    return df_result


# Main
training_start_date = '2001-12-01'
testing_date_plus_one = '2019-01-01'
for ticker in tickers:
    result = process_ticker(ticker, training_start_date, testing_date_plus_one)
