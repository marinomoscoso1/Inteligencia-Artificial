import torch
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def load_data(ticket:str):

    ticker=yf.Ticker(ticket)
    historical_data= ticker.history(period="1y").dropna()

    label=historical_data["Close"]
    evidence=historical_data.drop(columns="Close")

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

def train(model,train,test,batch_size,epochs):
    loss_fun=nn.MSELoss()
    optimizer=optim.Adam(model.parameters())

    trainloader=torch.utils.data.DataLoader(train,batch_size=batch_size,
                                            shuffle=False)
    testloader=torch.utils.data.DataLoader(test,batch_size=batch_size,
                                           shuffle=False)
    
    for epoch in range(epochs):

        running_loss=0.0
        eval_loss=0.0

        model.train()

        for i,data in enumerate(trainloader):
            inputs,labels=data

            optimizer.zero_grad()

            outputs=model(inputs)
            loss=loss_fun(outputs,labels)
            loss.backward()

            optimizer.step()

            running_loss+=loss.item()

            if i%20==0:
                print(f"[{epoch+1},{i+1:5d}] loss:{running_loss/len(trainloader):.3f}")

        model.eval()

        for i,data in enumerate(testloader):
            inputs,labels=data

            with torch.no_grad():
                outputs=model(inputs)
                loss=loss_fun(outputs,labels)

                eval_loss+=loss.item()

                if i%30==0:
                    print(f"{epoch}, loss:{eval_loss/len(testloader):.2f}")

class StockPredictor(nn.Module):
    def __init__(self):
        super().__init__()

        self.features=6
        self.neurons=64

        self.lstm= nn.LSTM(
            input_size=self.features,
            hidden_size=self.neurons,
            batch_first=True
        )

        self.linear=nn.Linear(
            in_features=self.neurons,
            out_features=1
        )

    def forward(self,x):
        output,hidden=self.lstm(x)
        x=hidden[0].squeeze(0)
        x=self.linear(x)

        return x
    

