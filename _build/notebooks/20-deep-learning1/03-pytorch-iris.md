---
interact_link: content/notebooks/20-deep-learning1/03-pytorch-iris.ipynb
kernel_name: python3
has_widgets: false
title: 'Pytorch IRIS'
prev_page:
  url: /notebooks/20-deep-learning1/02-tensor-tutorial.html
  title: 'Tensors'
next_page:
  url: /notebooks/20-deep-learning1/04-covnet-tutorial.html
  title: 'Covnet'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---


<center>[![AnalyticsDojo](https://raw.githubusercontent.com/rpi-techfundamentals/fall2018-materials/master/fig/final-logo.png)](http://rpi.analyticsdojo.com)
<h1>Revisiting IRIS with PyTorch</h1></center>
<center><h3><a href = 'http://rpi.analyticsdojo.com'>rpi.analyticsdojo.com</a></h3></center>




<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```!pip install torch torchvision

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#Let's get rid of some imports
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
#Define the model 
import torch
import torch.nn as nn
import torch.nn.functional as F

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```from sklearn.datasets import load_iris
#Iris is available from the sklearn package
iris = load_iris()
X, y = iris.data, iris.target



```
</div>

</div>



---



### Shuffled with Stratified Splitting

- Especially for relatively small datasets, it's better to stratify the split.  
- Stratification means that we maintain the original class proportion of the dataset in the test and training sets. 
- For example, after we randomly split the dataset as shown in the previous code example, we have the following class proportions in percent:



So, in order to stratify the split, we can pass the label array as an additional option to the `train_test_split` function:



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#Import Module
from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X, y, 
                                                    train_size=0.8,
                                                    test_size=0.2,
                                                    random_state=123,
                                                    stratify=y)

print('All:', np.bincount(y) / float(len(y)) * 100.0)
print('Training:', np.bincount(train_y) / float(len(train_y)) * 100.0)
print('Validation:', np.bincount(val_y) / float(len(val_y)) * 100.0)

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```train_y

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#This gives the number of columns, used below. 
train_X.shape[1]

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#This gives the number of classes, used below. 
len(np.unique(train_y))

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#Define training hyperprameters.
batch_size = 60
num_epochs = 500
learning_rate = 0.01
size_hidden= 100

#Calculate some other hyperparameters based on data.  
batch_no = len(train_X) // batch_size  #batches
cols=train_X.shape[1] #Number of columns in input matrix
classes= len(np.unique(train_y))



```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Assume that we are on a CUDA machine, then this should print a CUDA device:
print("Executing the model on :",device)

class Net(nn.Module):
    def __init__(self,cols,size_hidden,classes):
        super(Net, self).__init__()
        #Note that 17 is the number of columns in the input matrix. 
        self.fc1 = nn.Linear(cols, size_hidden)
        #variety of # possible for hidden layer size is arbitrary, but needs to be consistent across layers.  3 is the number of classes in the output (died/survived)
        self.fc2 = nn.Linear(size_hidden, classes)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.dropout(x, p=0.1)
        x = F.relu(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)
    
net = Net(cols, size_hidden, classes)

```
</div>

</div>



See this on [Adam optimizer](https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/).

[Cross Entropy Loss Function](https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html)





<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```#Adam is a specific flavor of gradient decent which is typically better
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()




```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```from sklearn.utils import shuffle
from torch.autograd import Variable
running_loss = 0.0
for epoch in range(num_epochs):
    #Shuffle just mixes up the dataset between epocs
    train_X, train_y = shuffle(train_X, train_y)
    # Mini batch learning
    for i in range(batch_no):
        start = i * batch_size
        end = start + batch_size
        inputs = Variable(torch.FloatTensor(train_X[start:end]))
        labels = Variable(torch.LongTensor(train_y[start:end]))
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        
    print('Epoch {}'.format(epoch+1), "loss: ",running_loss)
    running_loss = 0.0

        

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```import pandas as pd
#This is a little bit tricky to get the resulting prediction.  
def calculate_accuracy(x,y=[]):
    """
    This function will return the accuracy if passed x and y or return predictions if just passed x. 
    """
    # Evaluate the model with the test set. 
    X = Variable(torch.FloatTensor(x))  
    result = net(X) #This outputs the probability for each class.
    _, labels = torch.max(result.data, 1)
    if len(y) != 0:
        num_right = np.sum(labels.data.numpy() == y)
        print('Accuracy {:.2f}'.format(num_right / len(y)), "for a total of ", len(y), "records")
        return pd.DataFrame(data= {'actual': y, 'predicted': labels.data.numpy()})
    else:
        print("returning predictions")
        return labels.data.numpy()


```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```result1=calculate_accuracy(train_X,train_y)
result2=calculate_accuracy(val_X,val_y)


```
</div>

</div>



---



### Comparison: Prediction using Simple Nearest Neighbor Classifier
- By evaluating our classifier performance on data that has been seen during training, we could get false confidence in the predictive power of our model. 
- In the worst case, it may simply memorize the training samples but completely fails classifying new, similar samples -- we really don't want to put such a system into production!
- Instead of using the same dataset for training and testing (this is called "resubstitution evaluation"), it is much much better to use a train/test split in order to estimate how well your trained model is doing on new data.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```from sklearn.neighbors import KNeighborsClassifier
#This creates a model object.
classifier = KNeighborsClassifier()
#This fits the model object to the data.
classifier.fit(train_X, train_y)
#This creates the prediction. 
pred_y = classifier.predict(val_X)


```
</div>

</div>



### Scoring
- We can manually calculate the accuracy as we have done before. 
- `metrics.accuracy_score` is passed the target value and the predicted value.Model objects also built in scoring functions.
- Can also us a `classifier.score` component built into the model. 



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```from sklearn import metrics

#This calculates the accuracy.
print("Classifier score: ", classifier.score(train_X, train_y) )
print("Classifier score: ", classifier.score(val_X, val_y) )


```
</div>

</div>

