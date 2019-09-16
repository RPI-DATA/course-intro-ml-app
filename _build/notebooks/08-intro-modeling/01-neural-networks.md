---
interact_link: content/notebooks/08-intro-modeling/01-neural-networks.ipynb
kernel_name: python3
has_widgets: false
title: 'The Simplest Neural Network with Numpy'
prev_page:
  url: /notebooks/06-viz-api-scraper/ALT-visualization-python-matplotlib.html
  title: 'Matplotlib'
next_page:
  url: /notebooks/08-intro-modeling/02-train-test-split.html
  title: 'Train Test Split'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---


## Neural Networks and the Simplist XOR Problem
- This was adopted from the PyTorch Tutorials. 
- Simple supervised machine learning.
- http://pytorch.org/tutorials/beginner/pytorch_with_examples.html



## Neural Networks
- Neural networks are the foundation of deep learning, which has revolutionized the 

```In the mathematical theory of artificial neural networks, the universal approximation theorem states[1] that a feed-forward network with a single hidden layer containing a finite number of neurons (i.e., a multilayer perceptron), can approximate continuous functions on compact subsets of Rn, under mild assumptions on the activation function.```

- A simple task that Neural Networks can do but simple linear models cannot is called the [XOR problem](https://medium.com/@jayeshbahire/the-xor-problem-in-neural-networks-50006411840b).

- The XOR problem involves an output being 1 if either of two inputs is 1, but not both. 



### Generate Fake Data
- `D_in` is the number of dimensions of an input varaible.
- `D_out` is the number of dimentions of an output variable.
- Here we are learning some special "fake" data that represents the xor problem. 
- Here, the dv is 1 if either the first or second variable is 




<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```# -*- coding: utf-8 -*-
import numpy as np

#This is out input array. 
X = np.array([ [0,0,0],[1,0,0],[0,1,0],[0,0,0] ])
y = np.array([[0,1,1,0]]).T
print("Input data:\n",X,"\n Output data:\n",y)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
Input data:
 [[0 0 0]
 [1 0 0]
 [0 1 0]
 [0 0 0]] 
 Output data:
 [[0]
 [1]
 [1]
 [0]]
```
</div>
</div>
</div>



### A Simple Neural Network
- Here we are going to build a neural network. 
- First layer (`D_in`)has to be the length of the input.
- `H` is the length of the output.
-  `D_out` is 1 as it will be the probability it is a 1.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```D_in = 3 
H = 3
d_out = 3
# Randomly initialize weights
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

```
</div>

</div>



### But "Hidden Layers" Aren't Hidden
- Let's take a look 
- These are just random numbers.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```print(w1, w2)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
[[-0.0250052   0.39248636 -0.95856533]
 [-0.54775758  1.01757558  0.0834066 ]
 [ 0.00108387  0.20093661  0.49683823]] [[-0.68675167]
 [-0.49060388]
 [ 0.3668211 ]]
```
</div>
</div>
</div>



### Update the Weights using Gradient Decent
- Calculate the predited value
- Calculate the loss function
- Compute the gradients of w1 and w2 with respect to the loss function
- Update the weights using the learning rate 



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```# -*- coding: utf-8 -*-

learning_rate = 1e-3  #Sets how fast our updates will occur
for t in range(300):
    # Forward pass: compute predicted y
    h = x.dot(w1)
    
    #A relu is just the activation.
    h_relu = np.maximum(h, 0)
    y_pred = h_relu.dot(w2)

    # Compute and print loss
    loss = np.square(y_pred - y).sum()
    print(t, loss)

    # Backprop to compute gradients of w1 and w2 with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0 ## Relu just removes the negative numbers. We will talk about it later.
    grad_w1 = x.T.dot(grad_h)

    # Update weights
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2

```
</div>

</div>



Fully connected 



### Verify the Predictions
- Obtained a predicted value from our model and compare to origional. 



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```
y_pred = h_relu.dot(w2)
y_pred

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```y


```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#We can see that the weights have been updated. 
w1

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```
w2

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```# Relu just removes the negative numbers.  
h_relu

```
</div>

</div>

