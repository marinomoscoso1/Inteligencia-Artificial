from modelp import load_data,train,StockPredictor
from torch.utils.data import TensorDataset

X_train,X_test,y_train,y_test=load_data("NVO")

train_data=TensorDataset(X_train,y_train)
test_data=TensorDataset(X_test,y_test)

model=StockPredictor()

data=[model,train_data,test_data,64,100]

train(*data)