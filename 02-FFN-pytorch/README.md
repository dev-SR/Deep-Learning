# Building Feed Forward Networks with `Pytorch`

- [Building Feed Forward Networks with `Pytorch`](#building-feed-forward-networks-with-pytorch)
  - [Linearity vs Non-Linearity](#linearity-vs-non-linearity)
  - [Generate Dataset](#generate-dataset)
  - [Using torch tensor and autograd](#using-torch-tensor-and-autograd)
  - [Using `NN.Parameter`](#using-nnparameter)
  - [Using `NN.Linear` and `Optim`](#using-nnlinear-and-optim)
  - [Using `NN.Sequential`](#using-nnsequential)
  - [Running in GPU](#running-in-gpu)


```python
"""
cd .\02-FFN-pytorch-workflow\
jupyter nbconvert --to markdown ffn.ipynb --output README.md
"""
import torch
import torch.nn.functional as Fn
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torch import optim
```

## Linearity vs Non-Linearity

<p align="center">
<img src="https://raw.githubusercontent.com/dev-SR/Deep-Learning/main/02-FFN-pytorch/img/lvsnl.png" alt="ShapeCalculation.png" width="800px">
</p>

## Generate Dataset


```python
NUM_FEATURES = 2
NUM_CLASSES = 4
RANDOM_SEED = 42
data,labels = make_blobs(n_samples=1000,
                         n_features=NUM_FEATURES,
                         centers=NUM_CLASSES,
                         random_state=RANDOM_SEED
                         )
print(data.shape,labels.shape)

# Convert to tensors
data = torch.from_numpy(data).type(torch.float)
labels = torch.from_numpy(labels).type(torch.LongTensor)

```

    (1000, 2) (1000,)



```python
# mymap = matplotlib.colors.LinearSegmentedColormap.from_list("",['red','yellow','green','blue'])
plt.scatter(data[:,0],data[:,1],c=labels,cmap=plt.cm.RdYlBu)#,cmap=mymap)
```




    <matplotlib.collections.PathCollection at 0x7f6150107ac0>





![png](README_files/README_8_1.png)




```python
X_train,X_test,y_train,y_test = train_test_split(data,labels,test_size=-1.2,random_state=RANDOM_SEED)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
```

    torch.Size([800, 2]) torch.Size([200, 2]) torch.Size([800]) torch.Size([200])


## Using torch tensor and autograd

<p align="center">
<img src="https://raw.githubusercontent.com/dev-SR/Deep-Learning/main/02-FFN-pytorch-workflow/img/ShapeCalculation.png" alt="ShapeCalculation.png" width="900px">
</p>


```python
torch.manual_seed(0)
W1 = torch.randn(10,2) / math.sqrt(2) # Shape-> (L_out,L_in)
W1.requires_grad_()
b1 = torch.zeros(10,requires_grad=True)

W2 = torch.randn(4,10) / math.sqrt(2) # Shape-> (L_out,L_in)
W2.requires_grad_()
b2 = torch.zeros(4,requires_grad=True)

print(W1.shape,W2.shape)
```

    torch.Size([10, 2]) torch.Size([4, 10])



```python
def model(X):
    a1 = torch.matmul(X,W1.T) + b1 # (N,2) x (2,10)^T => (N,10)
    # print(a1.shape)
    h1 = a1.sigmoid() # (N,10)
    # print(h1.shape)
    a2 = torch.matmul(h1,W2.T) + b2 # (N,10) x (10,4)^T => (N,4)
    # print(a2.shape)
    h2 = a2.softmax(dim=1) # (N,4)
    # print(h2.shape)
    return h2

```


```python
# Calculate accuracy (a classification metric)
def accuracy_fn(y_hat, y):
    pred = torch.argmax(y_hat,dim=1)
    return (pred==y).float().mean()
```


```python
learning_rate = 0.2
epochs = 2000

X_train = X_train.float()
Y_train = y_train.long()

loss_err = []
acc_err = []

for epoch in range(epochs):
    y_hat = model(X_train)
    loss = Fn.cross_entropy(y_hat,Y_train)
    loss_err.append(loss.item())
    acc_err.append(accuracy_fn(y_hat,y_train))
    loss.backward()

    with torch.no_grad():
        W1 -= W1.grad * learning_rate
        b1 -= b1.grad * learning_rate
        W2 -= W2.grad * learning_rate
        b2 -= b2.grad * learning_rate

        W1.grad.zero_()
        b1.grad.zero_()
        W2.grad.zero_()
        b2.grad.zero_()

plt.plot(loss_err,'r-')
plt.plot(acc_err,'b-')

print(f"Loss before training : {loss_err[0]}")
print(f"Loss after training : {loss_err[-1]}")
```

    Loss before training : 1.4541833400726318
    Loss after training : 0.7496986389160156




![png](README_files/README_15_1.png)



## Using `NN.Parameter`



```python
# def model(X):
#     a1 = torch.matmul(X,W1.T) + b1 # (N,2) x (2,10)^T => (N,10)
#     # print(a1.shape)
#     h1 = a1.sigmoid() # (N,10)
#     # print(h1.shape)
#     a2 = torch.matmul(h1,W2.T) + b2 # (N,10) x (10,4)^T => (N,4)
#     # print(a2.shape)
#     h2 = a2.softmax(dim=1) # (N,4)
#     # print(h2.shape)
#     return h2
# torch.manual_seed(0)
# W1 = torch.randn(10,2) / math.sqrt(2) # Shape-> (L_out,L_in)
# W1.requires_grad_()
# b1 = torch.zeros(10,requires_grad=True)

# W2 = torch.randn(4,10) / math.sqrt(2) # Shape-> (L_out,L_in)
# W2.requires_grad_()
# b2 = torch.zeros(4,requires_grad=True)

# print(W1.shape,W2.shape)
```


```python
class FirstNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.W1 = nn.Parameter(torch.randn(10,2, dtype=torch.float)/math.sqrt(2))
        self.b1 = nn.Parameter(torch.zeros(10))
        self.W2 = nn.Parameter(torch.randn(4,10, dtype=torch.float)/math.sqrt(2))
        self.b2 = nn.Parameter(torch.zeros(4))

    def forward(self,X):
        a1= torch.matmul(X,self.W1.T)+b1
        h1 = a1.sigmoid()

        a2 = torch.matmul(h1,self.W2.T)+b2
        h2 = a2.softmax(dim=1)

        return h2

model = FirstNetwork()
```


```python
def fit(epochs=2000,learning_rate=1):
    loss_err = []
    acc_err = []
    for epoch in range(epochs):
        y_hat = model(X_train)
        loss = Fn.cross_entropy(y_hat,y_train)
        loss_err.append(loss.item())
        acc_err.append(accuracy_fn(y_hat,y_train))

        loss.backward()


        # with torch.no_grad():
        #     W1 -= W1.grad * learning_rate
        #     b1 -= b1.grad * learning_rate
        #     W2 -= W2.grad * learning_rate
        #     b2 -= b2.grad * learning_rate

        #     W1.grad.zero_()
        #     b1.grad.zero_()
        #     W2.grad.zero_()
        #     b2.grad.zero_()

        with torch.no_grad():
            for param in model.parameters():
                if param.grad is not None:
                    param -= learning_rate * param.grad
            model.zero_grad()

    plt.plot(loss_err,'r-')
    plt.plot(acc_err,'b-')
    print(f"Loss before training : {loss_err[0]}")
    print(f"Loss after training : {loss_err[-1]}")

fit()


```

    Loss before training : 1.4198334217071533
    Loss after training : 0.7454500794410706




![png](README_files/README_19_1.png)



## Using `NN.Linear` and `Optim`


```python
class NNLinearNetwork_v1(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        # self.W1 = nn.Parameter(torch.randn(10,2, dtype=torch.float)/math.sqrt(2))
        # self.b1 = nn.Parameter(torch.zeros(10))
        # self.W2 = nn.Parameter(torch.randn(4,10, dtype=torch.float)/math.sqrt(2))
        # self.b2 = nn.Parameter(torch.zeros(4))

        self.l1 = nn.Linear(in_features=2,out_features=10)
        self.l2 = nn.Linear(10,4)

    def forward(self,X):
        # a1= torch.matmul(X,self.W1.T)+b1
        a1 = self.l1(X)
        h1 = a1.sigmoid()

        # a2 = torch.matmul(h1,self.W2.T)+b2
        a2 = self.l2(h1)
        h2 = a2.softmax(dim=1)

        return h2

model = NNLinearNetwork_v1()
```


```python
def fit(epochs=2000,learning_rate=1):
    loss_err = []
    acc_err = []
    opt = optim.SGD(model.parameters(),lr=learning_rate)
    for epoch in range(epochs):
        y_hat = model(X_train)
        loss = Fn.cross_entropy(y_hat,y_train)
        loss_err.append(loss.item())
        acc_err.append(accuracy_fn(y_hat,y_train))

        loss.backward()
        opt.step()
        opt.zero_grad()

        # with torch.no_grad():
        #     for param in model.parameters():
        #         if param.grad is not None:
        #             param -= learning_rate * param.grad
        #     model.zero_grad()

    plt.plot(loss_err,'r-')
    plt.plot(acc_err,'b-')
    print(f"Loss before training : {loss_err[0]}")
    print(f"Loss after training : {loss_err[-1]}")

fit()


```

    Loss before training : 1.4017045497894287
    Loss after training : 0.7455630302429199




![png](README_files/README_22_1.png)



## Using `NN.Sequential`


```python
class NNLinearNetwork_v2(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        # self.l1 = nn.Linear(in_features=2,out_features=10)
        # self.l2 = nn.Linear(10,4)
        self.net = nn.Sequential(
            nn.Linear(2,10),
            nn.Sigmoid(),
            nn.Linear(10,4),
            nn.Softmax(dim=1)
        )

    def forward(self,X):
        # a1 = self.l1(X)
        # h1 = a1.sigmoid()
        # a2 = self.l2(h1)
        # h2 = a2.softmax(dim=1)
        # return h2
        return self.net(X)

```


```python
def fit(x,y,model,opt,loss_fn,epochs=2000,log=False):
    loss_err = []
    acc_err = []
    for epoch in range(epochs):
        loss =loss_fn(model(x),y)
        loss_err.append(loss.item())
        acc_err.append(accuracy_fn(y_hat,y_train))

        loss.backward()
        opt.step()
        opt.zero_grad()

    if log:
        plt.plot(loss_err,'r-')
        plt.plot(acc_err,'b-')
        print(f"Loss before training : {loss_err[0]}")
        print(f"Loss after training : {loss_err[-1]}")



model = NNLinearNetwork_v2()
loss_fn = Fn.cross_entropy
opt = optim.SGD(model.parameters(),lr=1)

fit(x=X_train,y=y_train,model=model,opt=opt,loss_fn=loss_fn,log=True)


```

    Loss before training : 1.4017045497894287
    Loss after training : 0.7455630302429199




![png](README_files/README_25_1.png)



## Running in GPU


```python
class NNLinearNetwork_v3(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.net = nn.Sequential(
            nn.Linear(2,1024*4),
            nn.Sigmoid(),
            nn.Linear(1024*4,4),
            nn.Softmax(dim=1)
        )

    def forward(self,X):
        return self.net(X)

```


```python
def fit(x,y,model,opt,loss_fn,epochs=2000,log=False):
    loss_err = []
    acc_err = []
    for epoch in range(epochs):
        loss =loss_fn(model(x),y)
        loss_err.append(loss.item())

        loss.backward()
        opt.step()
        opt.zero_grad()

    if log:
        print(f"Loss before training : {loss_err[0]}")
        print(f"Loss after training : {loss_err[-1]}")
```


```python
device = torch.device("cpu") #cuda #cpu
print(device)

X_train = X_train.to(device)
y_train = y_train.to(device)
model = NNLinearNetwork_v3().to(device)
model.to(device)
```

    cpu





    NNLinearNetwork_v3(
      (net): Sequential(
        (0): Linear(in_features=2, out_features=4096, bias=True)
        (1): Sigmoid()
        (2): Linear(in_features=4096, out_features=4, bias=True)
        (3): Softmax(dim=1)
      )
    )




```python
%%time
loss_fn = Fn.cross_entropy
opt = optim.SGD(model.parameters(),lr=1)
fit(x=X_train,y=y_train,model=model,opt=opt,loss_fn=loss_fn)

```

    CPU times: user 54.1 s, sys: 6.4 s, total: 1min
    Wall time: 1min



```python
%%time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

X_train = X_train.to(device)
y_train = y_train.to(device)
model = NNLinearNetwork_v3().to(device)

fit(x=X_train,y=y_train,model=model,opt=opt,loss_fn=loss_fn)
```

    cuda
    CPU times: user 2.03 s, sys: 88.5 ms, total: 2.12 s
    Wall time: 2.11 s



