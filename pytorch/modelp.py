import torch
import yfinance as yf

def load_data(ticket:str):

    ticker=yf.Ticker(ticket)
    historical_data= ticker.history(period="1y")

    return historical_data

data= load_data("AAPL")
print(type(data))