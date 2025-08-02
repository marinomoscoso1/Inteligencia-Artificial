import torch
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split

def load_data(ticket:str):

    ticker=yf.Ticker(ticket)
    historical_data= ticker.history(period="1y")

    label=historical_data["Close"].dropna()
    evidence=historical_data.drop(columns="Close").dropna()

    X=evidence.to_numpy()
    y=label.to_numpy()

    window_size=30
    X_seq=[]
    y_seq=[]

    for i in range(len(X)-window_size):
        x_sequence=X[i:i+window_size]
        y_sequence=y[i+window_size]

        X_seq.append(x_sequence)
        y_seq.append(y_sequence)

    X_tensor=torch.tensor(X_seq,dtype=torch.float32)
    Y_tensor=torch.tensor(y_seq,dtype=torch.float32)

    X_train,X_test,y_train,y_test=train_test_split(X_tensor,Y_tensor,test_size=0.2,shuffle=False)

    return X_train,X_test,y_train,y_test

