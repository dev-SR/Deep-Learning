# Pytorch

- [Pytorch](#pytorch)
  - [Initialize tensors](#initialize-tensors)
  - [Slicing Tensor](#slicing-tensor)
  - [🚀Reshaping Tensor](#reshaping-tensor)
  - [Tensor Operation](#tensor-operation)
    - [Basic Operations](#basic-operations)
    - [🚀Matrix multiplication (is all you need)](#matrix-multiplication-is-all-you-need)
  - [Numpy/Python `<>` PyTorch](#numpypython--pytorch)
  - [Reshaping, stacking, squeezing and unsqueezing](#reshaping-stacking-squeezing-and-unsqueezing)
  - [Finding the min, max, mean, sum, etc (aggregation)¶](#finding-the-min-max-mean-sum-etc-aggregation)
    - [Positional min/max](#positional-minmax)
  - [🔥AutoGrad](#autograd)
  - [🔥Running tensors on GPUs](#running-tensors-on-gpus)


**Resources**:

- [learnpytorch.io](https://www.learnpytorch.io/)


```python
"""
cd .\101pytorch\
jupyter nbconvert --to markdown torch.ipynb --output README.md
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
```

## Initialize tensors


```python
# Create a range of values 0 to 10
zero_to_ten = torch.arange(start=0, end=10, step=1)
zero_to_ten
```




    tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])




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
    tensor([[0.0478, 0.0608],
            [0.1022, 0.7141],
            [0.3429, 0.3827]])



```python
some_tensor = torch.rand(3, 4)

# Find out details about it
print(f"Shape of tensor: {some_tensor.shape}")
print(f"Datatype of tensor: {some_tensor.dtype}")
print(f"Device tensor is stored on: {some_tensor.device}") # will default to CPU
```

    Shape of tensor: torch.Size([3, 4])
    Datatype of tensor: torch.float32
    Device tensor is stored on: cpu



```python
x = torch.empty(3,2)
print(x)
y = torch.zeros_like(x)
y
```

    tensor([[1.3110e-31, 4.5790e-41],
            [3.7704e-37, 0.0000e+00],
            [4.4842e-44, 0.0000e+00]])





    tensor([[0., 0.],
            [0., 0.],
            [0., 0.]])




```python
x = torch.linspace(0,1,steps=5)
x
```




    tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000])




```python
py_ndArr = [[1,2],
			[3,4],
			[5,6]]
x = torch.tensor(py_ndArr)
x
```




    tensor([[1, 2],
            [3, 4],
            [5, 6]])



## Slicing Tensor


```python
print(x)
print(x.size())
print(x[:,1]) # all row; 1th coloumn
print(x[0,:]) # 0th row, all columns
```

    tensor([[1, 2],
            [3, 4],
            [5, 6]])
    torch.Size([3, 2])
    tensor([2, 4, 6])
    tensor([1, 2])



```python
x = torch.arange(1, 10).reshape(1, 3, 3)
x, x.shape
```




    (tensor([[[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]]]), torch.Size([1, 3, 3]))




```python
# Let's index bracket by bracket
print(f"First square bracket:\n{x[0]}")
print(f"Second square bracket: {x[0][0]}")
print(f"Third square bracket: {x[0][0][0]}")
```

    First square bracket:
    tensor([[1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]])
    Second square bracket: tensor([1, 2, 3])
    Third square bracket: 1



```python
# Get all values of 0th dimension and the 0 index of 1st dimension
x[:, 0]
```




    tensor([[1, 2, 3]])




```python
# Get all values of 0th & 1st dimensions but only index 1 of 2nd dimension
x[:, :, 1]
```




    tensor([[2, 5, 8]])




```python
# Get all values of the 0 dimension but only the 1 index value of the 1st and 2nd dimension
x[:, 1, 1]
```




    tensor([5])




```python
# Get index 0 of 0th and 1st dimension and all values of 2nd dimension
x[0, 0, :] # same as x[0][0]
```




    tensor([1, 2, 3])



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
y = x.view(6,-1) # 6 row; auto coloumn
print(y)
```

    tensor([[1],
            [2],
            [3],
            [4],
            [5],
            [6]])


## Tensor Operation

### Basic Operations


```python
x = torch.ones([3,2])
y = torch.ones([3,2])

z = x + y
print(z)
z = x - y
print(z)
z = x * y#Element-wise multiplication
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


### 🚀Matrix multiplication (is all you need)


One of the most common operations in machine learning and deep learning algorithms (like neural networks) is matrix multiplication.

But to multiply a matrix by another matrix we need to do the **`"dot product"` of rows and columns** ...

<p align="center">
<img src="https://raw.githubusercontent.com/dev-SR/Deep-Learning/main/01-pytorch-basics/img/nn.png" width="800px"/>
</p>

Note: A matrix multiplication like this is also referred to as the dot product of two matrices.

Neural networks are full of matrix multiplications and dot products.

PyTorch implements matrix multiplication functionality in the `torch.matmul()`,`torch.mm()` method or `@` operator.

The main two rules for matrix multiplication to remember are:

The inner dimensions must match:
- `(3, 2) @ (3, 2)` won't work
- `(2, 3) @ (3, 2)` will work
- `(3, 2) @ (2, 3)` will work

The resulting matrix has the shape of the outer dimensions:
- `(2, 3) @ (3, 2) -> (2, 2)`
- `(3, 2) @ (2, 3) -> (3, 3)`


```python
X = torch.rand(100,4) # 100 sample of 4 features each
X.shape
```




    torch.Size([100, 4])




```python
W = torch.rand(3,4) # 3 hidden unit connected to 4 input
W.T.shape
```




    torch.Size([4, 3])




```python
a = X.matmul(W.T)
a.shape
```




    torch.Size([100, 3])



The `torch.nn.Linear()` module (we'll see this in action later on), also known as a feed-forward layer or fully connected layer, implements a matrix multiplication between an input x and a weights matrix A.




```python
X = torch.rand(2,4) # 2 sample of 4 features each
torch.manual_seed(42)
# This uses matrix multiplication
linear = torch.nn.Linear(in_features=4, # in_features = matches inner dimension of input
                         out_features=6) # 6 hidden unit connected to 4 input
output = linear(X)
print(f"Input shape: {X.shape}\n")
print(f"Output:\n{output}\n\nOutput shape: {output.shape}")
```

    Input shape: torch.Size([2, 4])

    Output:
    tensor([[ 0.1644, -0.3716,  0.6903,  0.2284,  0.2870, -0.7039],
            [ 0.4176, -0.0611,  0.0467, -0.0456,  0.2174, -0.4366]],
           grad_fn=<AddmmBackward0>)

    Output shape: torch.Size([2, 6])


## Numpy/Python `<>` PyTorch

- PyTorch To `Numpy`


```python
# Tensor to NumPy array
tensor = torch.ones(7) # create a tensor of ones with dtype=float32
numpy_tensor = tensor.numpy() # will be dtype=float32 unless changed
tensor, numpy_tensor
```




    (tensor([1., 1., 1., 1., 1., 1., 1.]),
     array([1., 1., 1., 1., 1., 1., 1.], dtype=float32))



-  PyTorch To `python`


```python
y = x[1,1]
print(y)
print(y.item())

r0 = x[0,:]
print(r0)
print(r0.tolist())
```

    tensor(4)
    4
    tensor([1, 2])
    [1, 2]


- Pytorch to Numpy to Python List


```python
# Create a PyTorch tensor
tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
# Convert the tensor to a NumPy array
numpy_array = tensor.numpy()
# Convert the NumPy array to a Python list
python_list = numpy_array.tolist()
print(python_list) # Output: [[1, 2, 3], [4, 5, 6]]

```

    [[1, 2, 3], [4, 5, 6]]


- Numpy to PyTorch


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


- Tensor and Numpy benchmark:


```python
%%time
for i in range(10):
	a = np.random.randn(10000,10000)
	b = np.random.randn(10000,10000)
	c = a*b
```

    CPU times: user 1min 5s, sys: 5.19 s, total: 1min 10s
    Wall time: 1min 11s



```python

%%time
for i in range(10):
	a = torch.randn(10000,10000)
	b = torch.randn(10000,10000)
	c = a*b
```

    CPU times: user 17.2 s, sys: 7.24 s, total: 24.5 s
    Wall time: 23.8 s


## Reshaping, stacking, squeezing and unsqueezing


Often times you'll want to reshape or change the dimensions of your tensors without actually changing the values inside them.

- removing all single dimensions from a tensor


```python
x = torch.tensor([1, 2, 3, 4])
print(x)
print(x.unsqueeze(dim=0))
print(x.unsqueeze(dim=1))
```

    tensor([1, 2, 3, 4])
    tensor([[1, 2, 3, 4]])
    tensor([[1],
            [2],
            [3],
            [4]])


You can also rearrange the order of axes values with torch.permute(input, dims), where the input gets turned into a view with new dims.




```python
# Create tensor with specific shape
img_numpy = torch.rand(size=(224, 224, 3))

# Permute the original tensor to rearrange the axis order
img_torch_format = img_numpy.permute(2, 0, 1) # shifts axis 0->1, 1->2, 2->0

print(f"Previous shape: {img_numpy.shape}")
print(f"New shape: {img_torch_format.shape}")
```

    Previous shape: torch.Size([224, 224, 3])
    New shape: torch.Size([3, 224, 224])


## Finding the min, max, mean, sum, etc (aggregation)¶



```python
# Create a tensor
x = torch.arange(0, 100, 10)
print(f"Minimum: {x.min()}")
print(f"Maximum: {x.max()}")
# print(f"Mean: {x.mean()}") # this will error
print(f"Mean: {x.type(torch.float32).mean()}") # won't work without float datatype
print(f"Sum: {x.sum()}")
```

    Minimum: 0
    Maximum: 90
    Mean: 45.0
    Sum: 450


You can also do the same as above with torch methods.




```python
torch.max(x), torch.min(x), torch.mean(x.type(torch.float32)), torch.sum(x)
```




    (tensor(90), tensor(0), tensor(45.), tensor(450))



### Positional min/max


This is helpful incase you just want the **position where the highest (or lowest) value is** and *not the actual value itself* (we'll see this in a later when using the softmax activation function).




```python
# Create a tensor
tensor = torch.arange(10, 100, 10)
print(f"Tensor: {tensor}")

# Returns index of max and min values
print(f"Index where max value occurs: {tensor.argmax()}")
print(f"Index where min value occurs: {tensor.argmin()}")
```

    Tensor: tensor([10, 20, 30, 40, 50, 60, 70, 80, 90])
    Index where max value occurs: 8
    Index where min value occurs: 0


## 🔥AutoGrad

Autograd is a key feature of PyTorch that provides **automatic differentiation for all operations on tensors**. It allows developers to easily compute gradients of tensors with respect to other tensors. Autograd tracks all operations that are performed on tensors, creates a computation graph, and then uses the chain rule of differentiation to compute gradients. This feature is essential for training deep neural networks, as it allows us to efficiently calculate the gradients needed for optimization algorithms like stochastic gradient descent.

For the equation of $$z = 2x^{2} + 3y^{3}$$

The derivative of z with respect to x is:

$$\frac{\partial z}{\partial x} = 4x$$

The derivative of z with respect to y is:

$$\frac{\partial z}{\partial y} = 9y^{2}$$

Therefore, The derivative of $z=2x^2+3y^3$ with respect to $x$ evaluated at $x=2$ is:
$$\frac{\partial z}{\partial x}\bigg|_{x=2} = 4(2) = 8$$

And the derivative of $z$ with respect to $y$ evaluated at $y=1$ is:
$$\frac{\partial z}{\partial y}\bigg|_{y=1} = 9(1)^2 = 9$$


```python
# Define x and y as tensors with requires_grad=True
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(1.0, requires_grad=True)

# Define z as a function of x and y
z = 2*x**2 + 3*y**3

# Compute the gradients of z with respect to x and y
z.backward()

# Print the gradients
print('Gradient of z with respect to x:', x.grad)
print('Gradient of z with respect to y:', y.grad)

```

    Gradient of z with respect to x: tensor(8.)
    Gradient of z with respect to y: tensor(9.)


Building autograd feature from scratch : [building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0)

## 🔥Running tensors on GPUs


```python
!nvidia-smi
```

    Sun Feb 26 20:02:33 2023
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 525.85.12    Driver Version: 525.85.12    CUDA Version: 12.0     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |                               |                      |               MIG M. |
    |===============================+======================+======================|
    |   0  Tesla T4            Off  | 00000000:00:04.0 Off |                    0 |
    | N/A   61C    P0    31W /  70W |      0MiB / 15360MiB |      0%      Default |
    |                               |                      |                  N/A |
    +-------------------------------+----------------------+----------------------+

    +-----------------------------------------------------------------------------+
    | Processes:                                                                  |
    |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
    |        ID   ID                                                   Usage      |
    |=============================================================================|
    |  No running processes found                                                 |
    +-----------------------------------------------------------------------------+



```python
# Check for GPU
import torch
torch.cuda.is_available()
```




    True




```python
# Count number of devices
torch.cuda.device_count()
```




    1




```python
# Set device type
device = "cuda" if torch.cuda.is_available() else "cpu"
device
```




    'cuda'



- Let's try creating a tensor and putting it on the GPU (if it's available).




```python
# Create tensor (default on CPU)
tensor = torch.tensor([1, 2, 3])

# Tensor not on GPU
print(tensor, tensor.device)

# Move tensor to GPU (if available)
tensor_on_gpu = tensor.to(device)
tensor_on_gpu
```

    tensor([1, 2, 3]) cpu





    tensor([1, 2, 3], device='cuda:0')



- Moving tensors back to the CPU



```python
# If tensor is on GPU, can't transform it to NumPy (this will error)
tensor_on_gpu.numpy()
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-10-53175578f49e> in <module>
          1 # If tensor is on GPU, can't transform it to NumPy (this will error)
    ----> 2 tensor_on_gpu.numpy()


    TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.



```python
# Instead, copy the tensor back to cpu
tensor_back_on_cpu = tensor_on_gpu.cpu().numpy()
tensor_back_on_cpu # The above returns a copy of the GPU tensor in CPU memory so the original tensor is still on GPU.


```




    array([1, 2, 3])




```python
# The above returns a copy of the GPU tensor in CPU memory so the original tensor is still on GPU.
tensor_on_gpu
```




    tensor([1, 2, 3], device='cuda:0')


