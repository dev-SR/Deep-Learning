# Pytorch


```python
"""
cd .\101pytorch\
jupyter nbconvert --to markdown torch.ipynb --output README.md
"""
```


```python
import torch
import numpy as np
import matplotlib.pyplot as plt
```

## Initialize tensors


```python
x = torch.ones(3,2)
print(x)
x = torch.zeros(3,2)
print(x)
x = torch.rand(3,2)
print(x)
```

    tensor([[1., 1.],
            [1., 1.],
            [1., 1.]])
    tensor([[0., 0.],
            [0., 0.],
            [0., 0.]])
    tensor([[0.7660, 0.8867],
            [0.0299, 0.8590],
            [0.7468, 0.7586]])
    


```python
x = torch.empty(3,2) # create space
y = torch.zeros_like(x)
y
```




    tensor([[0., 0.],
            [0., 0.],
            [0., 0.]])




```python
x = torch.linspace(0,1,steps=5)
x
```




    tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000])




```python
x = torch.tensor([[1,2],
				[3,4],
				[5,6]])
x
```




    tensor([[1, 2],
            [3, 4],
            [5, 6]])



## Slicing Tensor


```python
print(x.size())
print(x[:,1])
print(x[0,:])
```

    torch.Size([3, 2])
    tensor([2, 4, 6])
    tensor([1, 2])
    


```python
y = x[1,1]
print(y)
# convert tensor to scaler
print(y.item())
```

    tensor(4)
    4
    

## 🚀Reshaping Tensor


```python
print(x)
y = x.view(2,3)
print(y)
```

    tensor([[1, 2],
            [3, 4],
            [5, 6]])
    tensor([[1, 2, 3],
            [4, 5, 6]])
    


```python
y = x.view(6,-1)
print(y)
```

    tensor([[1],
            [2],
            [3],
            [4],
            [5],
            [6]])
    

## Tensor Operation


```python
x = torch.ones([3,2])
y = torch.ones([3,2])

z = x + y
print(z)
z = x - y
print(z)
z = x * y
print(z)
```

    tensor([[2., 2.],
            [2., 2.],
            [2., 2.]])
    tensor([[0., 0.],
            [0., 0.],
            [0., 0.]])
    tensor([[1., 1.],
            [1., 1.],
            [1., 1.]])
    


```python
z = y.add(x)
z
```




    tensor([[2., 2.],
            [2., 2.],
            [2., 2.]])




```python
# modify inplace
z = y.add_(x)
print(z)
print(y)

```

    tensor([[2., 2.],
            [2., 2.],
            [2., 2.]])
    tensor([[2., 2.],
            [2., 2.],
            [2., 2.]])
    

## Numpy `<>` PyTorch


```python
x_np = x.numpy()
x_np
```




    array([[1., 1.],
           [1., 1.],
           [1., 1.]], dtype=float32)




```python
np.random.seed(0)
ar = np.random.randn(5)

print(ar)
ar_pt = torch.from_numpy(ar)
print(ar_pt)
print(type(ar),type(ar_pt))
```

    [1.76405235 0.40015721 0.97873798 2.2408932  1.86755799]
    tensor([1.7641, 0.4002, 0.9787, 2.2409, 1.8676], dtype=torch.float64)
    <class 'numpy.ndarray'> <class 'torch.Tensor'>
    


```python
np.add(ar,1,out=ar)
print(ar)
# torch also gets updated
print(ar_pt)
```

    [2.76405235 1.40015721 1.97873798 3.2408932  2.86755799]
    tensor([2.7641, 1.4002, 1.9787, 3.2409, 2.8676], dtype=torch.float64)
    


```python
%%time
for i in range(10):
	a = np.random.randn(10000,10000)
	b = np.random.randn(10000,10000)
	c = a*b
```

    CPU times: total: 2min 4s
    Wall time: 2min 9s
    


```python

%%time
for i in range(10):
	a = torch.randn(10000,10000)
	b = torch.randn(10000,10000)
	c = a*b
```

    CPU times: total: 38.8 s
    Wall time: 30.5 s
    

## AutoGrad


```python
x = torch.ones([3,2],requires_grad=True)
print(x)
```

    tensor([[1., 1.],
            [1., 1.],
            [1., 1.]], requires_grad=True)
    


```python
y = x+5
print(y)
```

    tensor([[6., 6.],
            [6., 6.],
            [6., 6.]], grad_fn=<AddBackward0>)
    


```python
z = y*y+1
print(z)
```

    tensor([[37., 37.],
            [37., 37.],
            [37., 37.]], grad_fn=<AddBackward0>)
    


```python
t = torch.sum(z)
print(t)
```

    tensor(222., grad_fn=<SumBackward0>)
    


```python
t.backward()
```


```python
print(x.grad)
```

    tensor([[12., 12.],
            [12., 12.],
            [12., 12.]])
    


```python

```
