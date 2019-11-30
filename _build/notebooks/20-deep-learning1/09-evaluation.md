---
interact_link: content/notebooks/20-deep-learning1/09-evaluation.ipynb
kernel_name: python3
has_widgets: false
title: 'Evaluation'
prev_page:
  url: /notebooks/20-deep-learning1/08-ludwig.html
  title: 'Ludwig'
next_page:
  url: /notebooks/24-tensorflow/10_neural_nets_with_keras.html
  title: 'TF-Keras'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---


## Evaluation of Classifiers
Let's assume we have 2 different images, and the output for the second to last layer is the following.  The job of the final layer is to "squish" whatever comes out of the neural network. We are going to look at the differences between a sigmoid and a softmax.


```
          img1    img2
cat	      0.02    -1.42
dog	     -2.49    -3.93
plane	   -1.75    -3.19
fish	    2.07     0.63
building	1.25    -0.19
```



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#Let's import some values
import torch
import torch.nn.functional as F
import torch.nn as nn

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#Let's put the data into a tensor
predictions = torch.tensor([[0.02, -2.49, -1.75, 2.07, 1.25],
                           [-1.42, -3.93, -3.19, 0.63, -0.19]])
predictions

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
tensor([[ 0.0200, -2.4900, -1.7500,  2.0700,  1.2500],
        [-1.4200, -3.9300, -3.1900,  0.6300, -0.1900]])
```


</div>
</div>
</div>



## Softmax
A softmax assumes that here that classes are exclusive and probabilities add to 1. 

$softmax(x)_i = \frac{exp(x_i)}{\sum_{j}^{ }exp(x_j))}$

*Check out the excel notebook and you should see that you get the same values. Note that even though the inputs for the softmax are different, they yield the same probability estimates for each class.*



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```
#Here we have to create the softmax layer and then pass the layers to it. 
my_softmax_layer = nn.Softmax(dim=1) #here we have to create the softmax layer and then 
softmax=my_softmax_layer(predictions)
softmax


```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
tensor([[0.0804, 0.0065, 0.0137, 0.6244, 0.2750],
        [0.0804, 0.0065, 0.0137, 0.6244, 0.2750]])
```


</div>
</div>
</div>



## Sigmoid
This is used for binary classification as a final layer.  For each of the potential classes, the prediction is weighted to a 0/1 without considering the other classes.  This would be appropriate for the case where there could be multiple classes (for example a cat and a dog) in the image.

$S(x)={\frac {1}{1+e^{-x}}}={\frac {e^{x}}{e^{x}+1}}$

*Check out the excel spreadsheet.*




<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```sigmoid=torch.sigmoid(predictions)
sigmoid

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
tensor([[0.5050, 0.0766, 0.1480, 0.8880, 0.7773],
        [0.1947, 0.0193, 0.0395, 0.6525, 0.4526]])
```


</div>
</div>
</div>



## Evaluating the Results
Note that for the 2 examples, the resulting probabilities were the same.  

However, note that the negative values for the final layer predictions suggest that maybe there are multiple items in image one and maybe just a fish in image 2.

*MEAN SQUARED ERROR (MSE)*

${MSE} ={\frac {1}{n}}\sum _{i=1}^{n}(Y_{i}-{\hat {Y_{i}}})^{2}$



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```predictions = torch.tensor([[0.02, -2.49, -1.75, 2.07, 1.25],
                           [-1.42, -3.93, -3.19, 0.63, -0.19]], requires_grad=True)
truth = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0],
                      [0.0, 0.0, 0.0, 1.0, 0.0]], requires_grad=False)

mse_loss=F.mse_loss(torch.sigmoid(predictions), truth )
print( "mse", mse_loss)


```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
mse tensor(0.1486, grad_fn=<MseLossBackward>)
```
</div>
</div>
</div>



## Exercise

  
1. Evaluate the loss function (MSE) for the softmax output.

2. Change the truth as well as the predictions above and notice the impact on the loss.



This exercise was adopted from the Fast.ai example. 

