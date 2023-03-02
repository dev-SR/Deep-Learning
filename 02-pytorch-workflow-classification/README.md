# Pytorch multi-class classification workflow


```python

"""
cd .\02-pytorch-workflow-classification\
jupyter nbconvert --to markdown torch_workflow.ipynb --output README.md
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
