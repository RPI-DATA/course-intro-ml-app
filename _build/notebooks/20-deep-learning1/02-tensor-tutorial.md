---
interact_link: content/notebooks/20-deep-learning1/02-tensor-tutorial.ipynb
kernel_name: python3
has_widgets: false
title: 'Tensors'
prev_page:
  url: /notebooks/20-deep-learning1/01-neural-networks.html
  title: 'Neural Networks'
next_page:
  url: /notebooks/20-deep-learning1/03-pytorch-iris.html
  title: 'Pytorch IRIS'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---


<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
%matplotlib inline

```
</div>

</div>




What is PyTorch?
================

It’s a Python-based scientific computing package targeted at two sets of
audiences:

-  A replacement for NumPy to use the power of GPUs
-  a deep learning research platform that provides maximum flexibility
   and speed

Getting Started
---------------

Tensors
^^^^^^^

Tensors are similar to NumPy’s ndarrays, with the addition being that
Tensors can also be used on a GPU to accelerate computing.





<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
from __future__ import print_function
import torch

```
</div>

</div>



Construct a 5x3 matrix, uninitialized:





<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
x = torch.empty(5, 3)
print(x)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
tensor([[-1484854811542562013184.0000,                       0.0000,
         -1484854811542562013184.0000],
        [                      0.0000,                       0.0000,
                               0.0000],
        [                      0.0000,                       0.0000,
                               0.0000],
        [                      0.0000,                       0.0000,
                               0.0000],
        [                      0.0000,                       0.0000,
                               0.0000]])
```
</div>
</div>
</div>



Construct a randomly initialized matrix:





<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
x = torch.rand(5, 3)
print(x)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
tensor([[0.4742, 0.4538, 0.4304],
        [0.1801, 0.4597, 0.2645],
        [0.4020, 0.2434, 0.2058],
        [0.6396, 0.7139, 0.3221],
        [0.1281, 0.3521, 0.9752]])
```
</div>
</div>
</div>



Construct a matrix filled zeros and of dtype long:





<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
x = torch.zeros(5, 3, dtype=torch.long)
print(x)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
tensor([[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]])
```
</div>
</div>
</div>



Construct a tensor directly from data:





<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
x = torch.tensor([5.5, 3])
print(x)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
tensor([5.5000, 3.0000])
```
</div>
</div>
</div>



or create a tensor based on an existing tensor. These methods
will reuse properties of the input tensor, e.g. dtype, unless
new values are provided by user





<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
x = x.new_ones(5, 3, dtype=torch.double)      # new_* methods take in sizes
print(x)

x = torch.randn_like(x, dtype=torch.float)    # override dtype!
print(x)                                      # result has the same size

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
tensor([[1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.]], dtype=torch.float64)
tensor([[-0.5785,  0.6964, -1.4768],
        [-0.0999,  0.1288, -0.7034],
        [-0.0728, -2.1770,  0.4147],
        [ 1.0561, -0.7525,  0.7957],
        [-0.2872, -1.4618, -0.7413]])
```
</div>
</div>
</div>



Get its size:





<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
print(x.size())

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
torch.Size([5, 3])
```
</div>
</div>
</div>



<div class="alert alert-info"><h4>Note</h4><p>``torch.Size`` is in fact a tuple, so it supports all tuple operations.</p></div>

Operations
^^^^^^^^^^
There are multiple syntaxes for operations. In the following
example, we will take a look at the addition operation.

Addition: syntax 1





<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
y = torch.rand(5, 3)
print(x + y)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
tensor([[-0.5261,  1.0870, -0.9739],
        [ 0.5831,  0.5069, -0.5522],
        [ 0.3222, -1.4902,  0.8171],
        [ 1.5093, -0.5091,  1.5999],
        [ 0.5226, -0.8918,  0.0173]])
```
</div>
</div>
</div>



Addition: syntax 2





<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
print(torch.add(x, y))

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
tensor([[-0.5261,  1.0870, -0.9739],
        [ 0.5831,  0.5069, -0.5522],
        [ 0.3222, -1.4902,  0.8171],
        [ 1.5093, -0.5091,  1.5999],
        [ 0.5226, -0.8918,  0.0173]])
```
</div>
</div>
</div>



Addition: providing an output tensor as argument





<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
tensor([[-0.5261,  1.0870, -0.9739],
        [ 0.5831,  0.5069, -0.5522],
        [ 0.3222, -1.4902,  0.8171],
        [ 1.5093, -0.5091,  1.5999],
        [ 0.5226, -0.8918,  0.0173]])
```
</div>
</div>
</div>



Addition: in-place





<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# adds x to y
y.add_(x)
print(y)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
tensor([[-0.5261,  1.0870, -0.9739],
        [ 0.5831,  0.5069, -0.5522],
        [ 0.3222, -1.4902,  0.8171],
        [ 1.5093, -0.5091,  1.5999],
        [ 0.5226, -0.8918,  0.0173]])
```
</div>
</div>
</div>



<div class="alert alert-info"><h4>Note</h4><p>Any operation that mutates a tensor in-place is post-fixed with an ``_``.
    For example: ``x.copy_(y)``, ``x.t_()``, will change ``x``.</p></div>

You can use standard NumPy-like indexing with all bells and whistles!





<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
print(x[:, 1])

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
tensor([ 0.6964,  0.1288, -2.1770, -0.7525, -1.4618])
```
</div>
</div>
</div>



Resizing: If you want to resize/reshape tensor, you can use ``torch.view``:





<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
print(x.size(), y.size(), z.size())

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
torch.Size([4, 4]) torch.Size([16]) torch.Size([2, 8])
```
</div>
</div>
</div>



If you have a one element tensor, use ``.item()`` to get the value as a
Python number





<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
x = torch.randn(1)
print(x)
print(x.item())

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
tensor([-1.0914])
-1.0913640260696411
```
</div>
</div>
</div>



**Read later:**


  100+ Tensor operations, including transposing, indexing, slicing,
  mathematical operations, linear algebra, random numbers, etc.,
  are described
  `here <http://pytorch.org/docs/torch>`_.

NumPy Bridge
------------

Converting a Torch Tensor to a NumPy array and vice versa is a breeze.

The Torch Tensor and NumPy array will share their underlying memory
locations, and changing one will change the other.

Converting a Torch Tensor to a NumPy Array
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^





<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
a = torch.ones(5)
print(a)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
tensor([1., 1., 1., 1., 1.])
```
</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
b = a.numpy()
print(b)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
[1. 1. 1. 1. 1.]
```
</div>
</div>
</div>



See how the numpy array changed in value.





<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
a.add_(1)
print(a)
print(b)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
tensor([2., 2., 2., 2., 2.])
[2. 2. 2. 2. 2.]
```
</div>
</div>
</div>



Converting NumPy Array to Torch Tensor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
See how changing the np array changed the Torch Tensor automatically





<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
[2. 2. 2. 2. 2.]
tensor([2., 2., 2., 2., 2.], dtype=torch.float64)
```
</div>
</div>
</div>



All the Tensors on the CPU except a CharTensor support converting to
NumPy and back.

CUDA Tensors
------------

Tensors can be moved onto any device using the ``.to`` method.





<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# let us run this cell only if CUDA is available
# We will use ``torch.device`` objects to move tensors in and out of GPU
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)                       # or just use strings ``.to("cuda")``
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!

```
</div>

</div>

