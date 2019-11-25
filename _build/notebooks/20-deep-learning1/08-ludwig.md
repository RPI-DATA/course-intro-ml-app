---
interact_link: content/notebooks/20-deep-learning1/08-ludwig.ipynb
kernel_name: python3
has_widgets: false
title: 'Ludwig'
prev_page:
  url: /notebooks/20-deep-learning1/07-titanic-fastai.html
  title: 'Titanic FastAI'
next_page:
  url: /notebooks/20-deep-learning1/09-evaluation.html
  title: 'Evaluation'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---


<center>[![AnalyticsDojo](https://raw.githubusercontent.com/rpi-techfundamentals/fall2018-materials/master/fig/final-logo.png)](http://rpi.analyticsdojo.com)
<h1>Ludwig</h1></center>
<center><h3><a href = 'http://rpi.analyticsdojo.com'>rpi.analyticsdojo.com</a></h3></center>





Ludwig, a project of Uber, provides a new data type-based approach to deep learning model design that makes the tool suited for many different applications. Rather than building out the architecture, you just need to specify the data.

First let's install ludwig and grab some data. 



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```!pip install ludwig

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
Collecting ludwig
[?25l  Downloading https://files.pythonhosted.org/packages/cd/a2/9f7f1952398e5aeb2f39579616fab8c3fada84a956ba6c855e6bc30a99f1/ludwig-0.1.1.tar.gz (129kB)
[K    100% |████████████████████████████████| 133kB 3.7MB/s 
[?25hRequirement already satisfied: Cython>=0.25 in /usr/local/lib/python3.6/dist-packages (from ludwig) (0.29.6)
Requirement already satisfied: h5py>=2.6 in /usr/local/lib/python3.6/dist-packages (from ludwig) (2.8.0)
Requirement already satisfied: matplotlib>=3.0 in /usr/local/lib/python3.6/dist-packages (from ludwig) (3.0.3)
Requirement already satisfied: numpy>=1.15 in /usr/local/lib/python3.6/dist-packages (from ludwig) (1.16.2)
Requirement already satisfied: pandas>=0.19 in /usr/local/lib/python3.6/dist-packages (from ludwig) (0.23.4)
Requirement already satisfied: scipy>=0.18 in /usr/local/lib/python3.6/dist-packages (from ludwig) (1.2.1)
Requirement already satisfied: scikit-learn in /usr/local/lib/python3.6/dist-packages (from ludwig) (0.20.3)
Requirement already satisfied: scikit-image==0.14.2 in /usr/local/lib/python3.6/dist-packages (from ludwig) (0.14.2)
Requirement already satisfied: seaborn>=0.7 in /usr/local/lib/python3.6/dist-packages (from ludwig) (0.7.1)
Collecting spacy>=2.1 (from ludwig)
[?25l  Downloading https://files.pythonhosted.org/packages/52/da/3a1c54694c2d2f40df82f38a19ae14c6eb24a5a1a0dae87205ebea7a84d8/spacy-2.1.3-cp36-cp36m-manylinux1_x86_64.whl (27.7MB)
[K    100% |████████████████████████████████| 27.7MB 1.3MB/s 
[?25hRequirement already satisfied: tqdm in /usr/local/lib/python3.6/dist-packages (from ludwig) (4.28.1)
Requirement already satisfied: tabulate>=0.7 in /usr/local/lib/python3.6/dist-packages (from ludwig) (0.8.3)
Requirement already satisfied: tensorflow==1.13.1 in /usr/local/lib/python3.6/dist-packages (from ludwig) (1.13.1)
Requirement already satisfied: PyYAML>=3.12 in /usr/local/lib/python3.6/dist-packages (from ludwig) (3.13)
Requirement already satisfied: six in /usr/local/lib/python3.6/dist-packages (from h5py>=2.6->ludwig) (1.11.0)
Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.6/dist-packages (from matplotlib>=3.0->ludwig) (0.10.0)
Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.6/dist-packages (from matplotlib>=3.0->ludwig) (1.0.1)
Requirement already satisfied: python-dateutil>=2.1 in /usr/local/lib/python3.6/dist-packages (from matplotlib>=3.0->ludwig) (2.5.3)
Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /usr/local/lib/python3.6/dist-packages (from matplotlib>=3.0->ludwig) (2.4.0)
Requirement already satisfied: pytz>=2011k in /usr/local/lib/python3.6/dist-packages (from pandas>=0.19->ludwig) (2018.9)
Requirement already satisfied: cloudpickle>=0.2.1 in /usr/local/lib/python3.6/dist-packages (from scikit-image==0.14.2->ludwig) (0.6.1)
Requirement already satisfied: networkx>=1.8 in /usr/local/lib/python3.6/dist-packages (from scikit-image==0.14.2->ludwig) (2.2)
Requirement already satisfied: dask[array]>=1.0.0 in /usr/local/lib/python3.6/dist-packages (from scikit-image==0.14.2->ludwig) (1.1.5)
Requirement already satisfied: pillow>=4.3.0 in /usr/local/lib/python3.6/dist-packages (from scikit-image==0.14.2->ludwig) (4.3.0)
Requirement already satisfied: PyWavelets>=0.4.0 in /usr/local/lib/python3.6/dist-packages (from scikit-image==0.14.2->ludwig) (1.0.3)
Requirement already satisfied: plac<1.0.0,>=0.9.6 in /usr/local/lib/python3.6/dist-packages (from spacy>=2.1->ludwig) (0.9.6)
Collecting srsly<1.1.0,>=0.0.5 (from spacy>=2.1->ludwig)
[?25l  Downloading https://files.pythonhosted.org/packages/6b/97/47753e3393aa4b18de9f942fac26f18879d1ae950243a556888f389d1398/srsly-0.0.5-cp36-cp36m-manylinux1_x86_64.whl (180kB)
[K    100% |████████████████████████████████| 184kB 11.4MB/s 
[?25hRequirement already satisfied: jsonschema<3.0.0,>=2.6.0 in /usr/local/lib/python3.6/dist-packages (from spacy>=2.1->ludwig) (2.6.0)
Collecting wasabi<1.1.0,>=0.2.0 (from spacy>=2.1->ludwig)
  Downloading https://files.pythonhosted.org/packages/76/6c/0376977df1ba9f0ec27835d80456d9284c79737cb5205649451db1181f01/wasabi-0.2.1-py3-none-any.whl
Collecting blis<0.3.0,>=0.2.2 (from spacy>=2.1->ludwig)
[?25l  Downloading https://files.pythonhosted.org/packages/34/46/b1d0bb71d308e820ed30316c5f0a017cb5ef5f4324bcbc7da3cf9d3b075c/blis-0.2.4-cp36-cp36m-manylinux1_x86_64.whl (3.2MB)
[K    100% |████████████████████████████████| 3.2MB 9.6MB/s 
[?25hRequirement already satisfied: preshed<2.1.0,>=2.0.1 in /usr/local/lib/python3.6/dist-packages (from spacy>=2.1->ludwig) (2.0.1)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /usr/local/lib/python3.6/dist-packages (from spacy>=2.1->ludwig) (2.0.2)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /usr/local/lib/python3.6/dist-packages (from spacy>=2.1->ludwig) (1.0.2)
Collecting thinc<7.1.0,>=7.0.2 (from spacy>=2.1->ludwig)
[?25l  Downloading https://files.pythonhosted.org/packages/a9/f1/3df317939a07b2fc81be1a92ac10bf836a1d87b4016346b25f8b63dee321/thinc-7.0.4-cp36-cp36m-manylinux1_x86_64.whl (2.1MB)
[K    100% |████████████████████████████████| 2.1MB 11.3MB/s 
[?25hRequirement already satisfied: requests<3.0.0,>=2.13.0 in /usr/local/lib/python3.6/dist-packages (from spacy>=2.1->ludwig) (2.18.4)
Requirement already satisfied: keras-applications>=1.0.6 in /usr/local/lib/python3.6/dist-packages (from tensorflow==1.13.1->ludwig) (1.0.7)
Requirement already satisfied: wheel>=0.26 in /usr/local/lib/python3.6/dist-packages (from tensorflow==1.13.1->ludwig) (0.33.1)
Requirement already satisfied: astor>=0.6.0 in /usr/local/lib/python3.6/dist-packages (from tensorflow==1.13.1->ludwig) (0.7.1)
Requirement already satisfied: tensorflow-estimator<1.14.0rc0,>=1.13.0 in /usr/local/lib/python3.6/dist-packages (from tensorflow==1.13.1->ludwig) (1.13.0)
Requirement already satisfied: protobuf>=3.6.1 in /usr/local/lib/python3.6/dist-packages (from tensorflow==1.13.1->ludwig) (3.7.1)
Requirement already satisfied: grpcio>=1.8.6 in /usr/local/lib/python3.6/dist-packages (from tensorflow==1.13.1->ludwig) (1.15.0)
Requirement already satisfied: gast>=0.2.0 in /usr/local/lib/python3.6/dist-packages (from tensorflow==1.13.1->ludwig) (0.2.2)
Requirement already satisfied: absl-py>=0.1.6 in /usr/local/lib/python3.6/dist-packages (from tensorflow==1.13.1->ludwig) (0.7.1)
Requirement already satisfied: keras-preprocessing>=1.0.5 in /usr/local/lib/python3.6/dist-packages (from tensorflow==1.13.1->ludwig) (1.0.9)
Requirement already satisfied: termcolor>=1.1.0 in /usr/local/lib/python3.6/dist-packages (from tensorflow==1.13.1->ludwig) (1.1.0)
Requirement already satisfied: tensorboard<1.14.0,>=1.13.0 in /usr/local/lib/python3.6/dist-packages (from tensorflow==1.13.1->ludwig) (1.13.1)
Requirement already satisfied: setuptools in /usr/local/lib/python3.6/dist-packages (from kiwisolver>=1.0.1->matplotlib>=3.0->ludwig) (40.9.0)
Requirement already satisfied: decorator>=4.3.0 in /usr/local/lib/python3.6/dist-packages (from networkx>=1.8->scikit-image==0.14.2->ludwig) (4.4.0)
Requirement already satisfied: toolz>=0.7.3; extra == "array" in /usr/local/lib/python3.6/dist-packages (from dask[array]>=1.0.0->scikit-image==0.14.2->ludwig) (0.9.0)
Requirement already satisfied: olefile in /usr/local/lib/python3.6/dist-packages (from pillow>=4.3.0->scikit-image==0.14.2->ludwig) (0.46)
Requirement already satisfied: urllib3<1.23,>=1.21.1 in /usr/local/lib/python3.6/dist-packages (from requests<3.0.0,>=2.13.0->spacy>=2.1->ludwig) (1.22)
Requirement already satisfied: idna<2.7,>=2.5 in /usr/local/lib/python3.6/dist-packages (from requests<3.0.0,>=2.13.0->spacy>=2.1->ludwig) (2.6)
Requirement already satisfied: chardet<3.1.0,>=3.0.2 in /usr/local/lib/python3.6/dist-packages (from requests<3.0.0,>=2.13.0->spacy>=2.1->ludwig) (3.0.4)
Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.6/dist-packages (from requests<3.0.0,>=2.13.0->spacy>=2.1->ludwig) (2019.3.9)
Requirement already satisfied: mock>=2.0.0 in /usr/local/lib/python3.6/dist-packages (from tensorflow-estimator<1.14.0rc0,>=1.13.0->tensorflow==1.13.1->ludwig) (2.0.0)
Requirement already satisfied: markdown>=2.6.8 in /usr/local/lib/python3.6/dist-packages (from tensorboard<1.14.0,>=1.13.0->tensorflow==1.13.1->ludwig) (3.1)
Requirement already satisfied: werkzeug>=0.11.15 in /usr/local/lib/python3.6/dist-packages (from tensorboard<1.14.0,>=1.13.0->tensorflow==1.13.1->ludwig) (0.15.2)
Requirement already satisfied: pbr>=0.11 in /usr/local/lib/python3.6/dist-packages (from mock>=2.0.0->tensorflow-estimator<1.14.0rc0,>=1.13.0->tensorflow==1.13.1->ludwig) (5.1.3)
Building wheels for collected packages: ludwig
  Building wheel for ludwig (setup.py) ... [?25ldone
[?25h  Stored in directory: /root/.cache/pip/wheels/95/e6/05/fa2b84191f6635508ed189ff80d40a641b4c42bc9709194c4d
Successfully built ludwig
Installing collected packages: srsly, wasabi, blis, thinc, spacy, ludwig
  Found existing installation: thinc 6.12.1
    Uninstalling thinc-6.12.1:
      Successfully uninstalled thinc-6.12.1
  Found existing installation: spacy 2.0.18
    Uninstalling spacy-2.0.18:
      Successfully uninstalled spacy-2.0.18
Successfully installed blis-0.2.4 ludwig-0.1.1 spacy-2.1.3 srsly-0.0.5 thinc-7.0.4 wasabi-0.2.1
```
</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```!wget https://raw.githubusercontent.com/rpi-techfundamentals/fall2018-materials/master/input/train.csv && wget https://raw.githubusercontent.com/rpi-techfundamentals/fall2018-materials/master/input/test.csv
  

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
--2019-04-15 14:41:15--  https://raw.githubusercontent.com/rpi-techfundamentals/fall2018-materials/master/input/train.csv
Resolving raw.githubusercontent.com (raw.githubusercontent.com)... 151.101.0.133, 151.101.64.133, 151.101.128.133, ...
Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|151.101.0.133|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 61194 (60K) [text/plain]
Saving to: ‘train.csv’

train.csv           100%[===================>]  59.76K  --.-KB/s    in 0.02s   

2019-04-15 14:41:15 (2.34 MB/s) - ‘train.csv’ saved [61194/61194]

--2019-04-15 14:41:15--  https://raw.githubusercontent.com/rpi-techfundamentals/fall2018-materials/master/input/test.csv
Resolving raw.githubusercontent.com (raw.githubusercontent.com)... 151.101.0.133, 151.101.64.133, 151.101.128.133, ...
Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|151.101.0.133|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 28629 (28K) [text/plain]
Saving to: ‘test.csv’

test.csv            100%[===================>]  27.96K  --.-KB/s    in 0.01s   

2019-04-15 14:41:15 (2.28 MB/s) - ‘test.csv’ saved [28629/28629]

```
</div>
</div>
</div>



### Model Definition File

Here in order to describe the model, we need to create/download a model definition file. This is a simple file that describes the data.  


```
input_features:
    -
        name: text
        type: text
        level: word
        encoder: parallel_cnn

output_features:
    -
        name: class
        type: category
        
```



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```!wget https://raw.githubusercontent.com/rpi-techfundamentals/spring2019-materials/master/13-deep-learning3/model_definition.yaml

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
--2019-04-15 14:41:17--  https://raw.githubusercontent.com/rpi-techfundamentals/spring2019-materials/master/13-deep-learning3/model_definition.yaml
Resolving raw.githubusercontent.com (raw.githubusercontent.com)... 151.101.0.133, 151.101.64.133, 151.101.128.133, ...
Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|151.101.0.133|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 524 [text/plain]
Saving to: ‘model_definition.yaml’

model_definition.ya 100%[===================>]     524  --.-KB/s    in 0s      

2019-04-15 14:41:17 (83.2 MB/s) - ‘model_definition.yaml’ saved [524/524]

```
</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```!cat model_definition.yaml

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
input_features:
    -
        name: Pclass
        type: category
    -
        name: Sex
        type: category
    -
        name: Age
        type: numerical
        missing_value_strategy: fill_with_mean
    -
        name: SibSp
        type: numerical
    -
        name: Parch
        type: numerical
    -
        name: Fare
        type: numerical
        missing_value_strategy: fill_with_mean
    -
        name: Embarked
        type: category

output_features:
    -
        name: Survived
        type: binary
```
</div>
</div>
</div>



### Training the Model
We are good to now train the model. 

While previously we have always done splits and done all of our training in core python.  Here we are just going to call the ludwig command line tool. 

ludwig experiment \
  --data_csv reuters-allcats.csv \
  --model_definition_file model_definition.yaml



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```!ludwig experiment  --data_csv train.csv \
--model_definition_file model_definition.yaml

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```

WARNING: The TensorFlow contrib module will not be included in TensorFlow 2.0.
For more information, please see:
  * https://github.com/tensorflow/community/blob/master/rfcs/20180907-contrib-sunset.md
  * https://github.com/tensorflow/addons
If you depend on functionality not listed there, please file an issue.

 _         _        _      
| |_  _ __| |_ __ _(_)__ _ 
| | || / _` \ V  V / / _` |
|_|\_,_\__,_|\_/\_/|_\__, |
                     |___/ 
ludwig v0.1.1 - Experiment

Experiment name: experiment
Model name: run
Output path: results/experiment_run_0

ludwig_version: '0.1.1'
command: ('/usr/local/bin/ludwig experiment --data_csv train.csv '
 '--model_definition_file model_definition.yaml')
dataset_type: 'generic'
random_seed: 42
input_data: 'train.csv'
model_definition: {   'combiner': {'type': 'concat'},
    'input_features': [   {   'name': 'Pclass',
                              'tied_weights': None,
                              'type': 'category'},
                          {   'name': 'Sex',
                              'tied_weights': None,
                              'type': 'category'},
                          {   'missing_value_strategy': 'fill_with_mean',
                              'name': 'Age',
                              'tied_weights': None,
                              'type': 'numerical'},
                          {   'name': 'SibSp',
                              'tied_weights': None,
                              'type': 'numerical'},
                          {   'name': 'Parch',
                              'tied_weights': None,
                              'type': 'numerical'},
                          {   'missing_value_strategy': 'fill_with_mean',
                              'name': 'Fare',
                              'tied_weights': None,
                              'type': 'numerical'},
                          {   'name': 'Embarked',
                              'tied_weights': None,
                              'type': 'category'}],
    'output_features': [   {   'dependencies': [],
                               'loss': {   'confidence_penalty': 0,
                                           'robust_lambda': 0,
                                           'threshold': 0.5,
                                           'weight': 1},
                               'name': 'Survived',
                               'reduce_dependencies': 'sum',
                               'reduce_input': 'sum',
                               'threshold': 0.5,
                               'type': 'binary',
                               'weight': 1}],
    'preprocessing': {   'bag': {   'fill_value': '',
                                    'format': 'space',
                                    'lowercase': False,
                                    'missing_value_strategy': 'fill_with_const',
                                    'most_common': 10000},
                         'binary': {   'fill_value': 0,
                                       'missing_value_strategy': 'fill_with_const'},
                         'category': {   'fill_value': '<UNK>',
                                         'lowercase': False,
                                         'missing_value_strategy': 'fill_with_const',
                                         'most_common': 10000},
                         'force_split': False,
                         'image': {   'in_memory': True,
                                      'missing_value_strategy': 'backfill',
                                      'resize_method': 'crop_or_pad'},
                         'numerical': {   'fill_value': 0,
                                          'missing_value_strategy': 'fill_with_const'},
                         'sequence': {   'fill_value': '',
                                         'format': 'space',
                                         'lowercase': False,
                                         'missing_value_strategy': 'fill_with_const',
                                         'most_common': 20000,
                                         'padding': 'right',
                                         'padding_symbol': '<PAD>',
                                         'sequence_length_limit': 256,
                                         'unknown_symbol': '<UNK>'},
                         'set': {   'fill_value': '',
                                    'format': 'space',
                                    'lowercase': False,
                                    'missing_value_strategy': 'fill_with_const',
                                    'most_common': 10000},
                         'split_probabilities': (0.7, 0.1, 0.2),
                         'stratify': None,
                         'text': {   'char_format': 'characters',
                                     'char_most_common': 70,
                                     'char_sequence_length_limit': 1024,
                                     'fill_value': '',
                                     'lowercase': True,
                                     'missing_value_strategy': 'fill_with_const',
                                     'padding': 'right',
                                     'padding_symbol': '<PAD>',
                                     'unknown_symbol': '<UNK>',
                                     'word_format': 'space_punct',
                                     'word_most_common': 20000,
                                     'word_sequence_length_limit': 256},
                         'timeseries': {   'fill_value': '',
                                           'format': 'space',
                                           'missing_value_strategy': 'fill_with_const',
                                           'padding': 'right',
                                           'padding_value': 0,
                                           'timeseries_length_limit': 256}},
    'training': {   'batch_size': 128,
                    'bucketing_field': None,
                    'decay': False,
                    'decay_rate': 0.96,
                    'decay_steps': 10000,
                    'dropout_rate': 0.0,
                    'early_stop': 5,
                    'epochs': 100,
                    'eval_batch_size': 0,
                    'gradient_clipping': None,
                    'increase_batch_size_on_plateau': 0,
                    'increase_batch_size_on_plateau_max': 512,
                    'increase_batch_size_on_plateau_patience': 5,
                    'increase_batch_size_on_plateau_rate': 2,
                    'learning_rate': 0.001,
                    'learning_rate_warmup_epochs': 5,
                    'optimizer': {   'beta1': 0.9,
                                     'beta2': 0.999,
                                     'epsilon': 1e-08,
                                     'type': 'adam'},
                    'reduce_learning_rate_on_plateau': 0,
                    'reduce_learning_rate_on_plateau_patience': 5,
                    'reduce_learning_rate_on_plateau_rate': 0.5,
                    'regularization_lambda': 0,
                    'regularizer': 'l2',
                    'staircase': False,
                    'validation_field': 'combined',
                    'validation_measure': 'loss'}}

Using full raw csv, no hdf5 and json file with the same name have been found
Building dataset (it may take a while)
/usr/local/lib/python3.6/dist-packages/ludwig/features/numerical_feature.py:63: FutureWarning: Method .as_matrix will be removed in a future version. Use .values instead.
  np.float32).as_matrix()
/usr/local/lib/python3.6/dist-packages/ludwig/features/binary_feature.py:62: FutureWarning: Method .as_matrix will be removed in a future version. Use .values instead.
  np.bool_).as_matrix()
Writing dataset
Writing train set metadata with vocabulary
Training set: 630
Validation set: 81
Test set: 180
WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/framework/op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
Instructions for updating:
Colocations handled automatically by placer.
From /usr/local/lib/python3.6/dist-packages/tensorflow/python/framework/op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
Instructions for updating:
Colocations handled automatically by placer.
  embedding_size (50) is greater than vocab_size (4). Setting embedding size to be equal to vocab_size.
  embedding_size (50) is greater than vocab_size (3). Setting embedding size to be equal to vocab_size.
  embedding_size (50) is greater than vocab_size (5). Setting embedding size to be equal to vocab_size.
WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/ops/array_grad.py:425: to_int32 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.cast instead.
From /usr/local/lib/python3.6/dist-packages/tensorflow/python/ops/array_grad.py:425: to_int32 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.cast instead.

╒══════════╕
│ TRAINING │
╘══════════╛

2019-04-15 14:41:29.343441: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2300000000 Hz
2019-04-15 14:41:29.346608: I tensorflow/compiler/xla/service/service.cc:150] XLA service 0x2020d60 executing computations on platform Host. Devices:
2019-04-15 14:41:29.346694: I tensorflow/compiler/xla/service/service.cc:158]   StreamExecutor device (0): <undefined>, <undefined>

Epoch   1
Training: 100% 5/5 [00:00<00:00, 10.08it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 110.90it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 614.91it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 729.38it/s]
Took 0.5719s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 7.9026 │     0.3968 │
├────────────┼────────┼────────────┤
│ vali       │ 7.5711 │     0.3827 │
├────────────┼────────┼────────────┤
│ test       │ 8.8524 │     0.3667 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch   2
Training: 100% 5/5 [00:00<00:00, 276.59it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 732.60it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 755.87it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 843.92it/s]
Took 0.3235s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 7.6934 │     0.3968 │
├────────────┼────────┼────────────┤
│ vali       │ 7.3628 │     0.3827 │
├────────────┼────────┼────────────┤
│ test       │ 8.6174 │     0.3667 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch   3
Training: 100% 5/5 [00:00<00:00, 443.98it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 771.10it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 710.66it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 322.44it/s]
Took 0.2996s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 7.4851 │     0.3968 │
├────────────┼────────┼────────────┤
│ vali       │ 7.1558 │     0.3827 │
├────────────┼────────┼────────────┤
│ test       │ 8.3837 │     0.3611 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch   4
Training: 100% 5/5 [00:00<00:00, 455.84it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 773.37it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 834.36it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 865.97it/s]
Took 0.2475s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 7.2782 │     0.3968 │
├────────────┼────────┼────────────┤
│ vali       │ 6.9506 │     0.3827 │
├────────────┼────────┼────────────┤
│ test       │ 8.1514 │     0.3611 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch   5
Training: 100% 5/5 [00:00<00:00, 445.86it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 814.71it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 660.52it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 837.94it/s]
Took 0.2481s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 7.0727 │     0.3984 │
├────────────┼────────┼────────────┤
│ vali       │ 6.7476 │     0.3951 │
├────────────┼────────┼────────────┤
│ test       │ 7.9209 │     0.3611 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch   6
Training: 100% 5/5 [00:00<00:00, 461.45it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 784.69it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 843.08it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 882.45it/s]
Took 0.2411s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 6.8693 │     0.4000 │
├────────────┼────────┼────────────┤
│ vali       │ 6.5471 │     0.3951 │
├────────────┼────────┼────────────┤
│ test       │ 7.6929 │     0.3667 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch   7
Training: 100% 5/5 [00:00<00:00, 454.23it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 779.84it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 816.97it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 831.46it/s]
Took 0.2468s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 6.6683 │     0.4016 │
├────────────┼────────┼────────────┤
│ vali       │ 6.3495 │     0.4074 │
├────────────┼────────┼────────────┤
│ test       │ 7.4677 │     0.3722 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch   8
Training: 100% 5/5 [00:00<00:00, 430.87it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 746.34it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 778.74it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 656.75it/s]
Took 0.2715s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 6.4704 │     0.4000 │
├────────────┼────────┼────────────┤
│ vali       │ 6.1552 │     0.4074 │
├────────────┼────────┼────────────┤
│ test       │ 7.2459 │     0.3778 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch   9
Training: 100% 5/5 [00:00<00:00, 413.70it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 746.18it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 744.86it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 817.68it/s]
Took 0.2651s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 6.2761 │     0.4032 │
├────────────┼────────┼────────────┤
│ vali       │ 5.9643 │     0.4074 │
├────────────┼────────┼────────────┤
│ test       │ 7.0279 │     0.3833 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch  10
Training: 100% 5/5 [00:00<00:00, 440.89it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 778.60it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 786.33it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 851.46it/s]
Took 0.2500s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 6.0861 │     0.4127 │
├────────────┼────────┼────────────┤
│ vali       │ 5.7773 │     0.4198 │
├────────────┼────────┼────────────┤
│ test       │ 6.8142 │     0.3889 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch  11
Training: 100% 5/5 [00:00<00:00, 411.29it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 772.66it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 789.29it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 814.82it/s]
Took 0.2603s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 5.9009 │     0.4095 │
├────────────┼────────┼────────────┤
│ vali       │ 5.5945 │     0.4198 │
├────────────┼────────┼────────────┤
│ test       │ 6.6052 │     0.3889 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch  12
Training: 100% 5/5 [00:00<00:00, 436.50it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 758.24it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 762.60it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 867.58it/s]
Took 0.2527s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 5.7210 │     0.4190 │
├────────────┼────────┼────────────┤
│ vali       │ 5.4165 │     0.4198 │
├────────────┼────────┼────────────┤
│ test       │ 6.4013 │     0.3944 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch  13
Training: 100% 5/5 [00:00<00:00, 476.17it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 777.59it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 787.51it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 810.57it/s]
Took 0.2433s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 5.5470 │     0.4270 │
├────────────┼────────┼────────────┤
│ vali       │ 5.2436 │     0.4198 │
├────────────┼────────┼────────────┤
│ test       │ 6.2030 │     0.3944 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch  14
Training: 100% 5/5 [00:00<00:00, 452.28it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 789.50it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 789.59it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 843.33it/s]
Took 0.2463s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 5.3792 │     0.4365 │
├────────────┼────────┼────────────┤
│ vali       │ 5.0764 │     0.4198 │
├────────────┼────────┼────────────┤
│ test       │ 6.0107 │     0.4111 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch  15
Training: 100% 5/5 [00:00<00:00, 451.66it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 579.71it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 820.80it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 861.08it/s]
Took 0.2671s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 5.2179 │     0.4444 │
├────────────┼────────┼────────────┤
│ vali       │ 4.9153 │     0.4321 │
├────────────┼────────┼────────────┤
│ test       │ 5.8247 │     0.4278 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch  16
Training: 100% 5/5 [00:00<00:00, 448.99it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 712.52it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 780.92it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 810.89it/s]
Took 0.2627s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 5.0633 │     0.4508 │
├────────────┼────────┼────────────┤
│ vali       │ 4.7606 │     0.4444 │
├────────────┼────────┼────────────┤
│ test       │ 5.6452 │     0.4444 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch  17
Training: 100% 5/5 [00:00<00:00, 404.22it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 765.66it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 835.35it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 835.94it/s]
Took 0.2625s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 4.9156 │     0.4476 │
├────────────┼────────┼────────────┤
│ vali       │ 4.6128 │     0.4691 │
├────────────┼────────┼────────────┤
│ test       │ 5.4724 │     0.4611 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch  18
Training: 100% 5/5 [00:00<00:00, 447.81it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 695.34it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 667.46it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 791.45it/s]
Took 0.2608s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 4.7748 │     0.4540 │
├────────────┼────────┼────────────┤
│ vali       │ 4.4719 │     0.5062 │
├────────────┼────────┼────────────┤
│ test       │ 5.3063 │     0.4778 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch  19
Training: 100% 5/5 [00:00<00:00, 391.12it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 680.52it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 670.45it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 711.80it/s]
Took 0.2840s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 4.6408 │     0.4683 │
├────────────┼────────┼────────────┤
│ vali       │ 4.3382 │     0.5309 │
├────────────┼────────┼────────────┤
│ test       │ 5.1471 │     0.4944 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch  20
Training: 100% 5/5 [00:00<00:00, 443.24it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 772.83it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 778.74it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 824.35it/s]
Took 0.2519s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 4.5136 │     0.4714 │
├────────────┼────────┼────────────┤
│ vali       │ 4.2120 │     0.5432 │
├────────────┼────────┼────────────┤
│ test       │ 4.9946 │     0.5111 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch  21
Training: 100% 5/5 [00:00<00:00, 456.19it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 770.22it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 744.46it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 864.98it/s]
Took 0.2458s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 4.3929 │     0.4794 │
├────────────┼────────┼────────────┤
│ vali       │ 4.0931 │     0.5432 │
├────────────┼────────┼────────────┤
│ test       │ 4.8489 │     0.5111 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch  22
Training: 100% 5/5 [00:00<00:00, 447.02it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 742.78it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 700.22it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 764.69it/s]
Took 0.2560s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 4.2788 │     0.4905 │
├────────────┼────────┼────────────┤
│ vali       │ 3.9815 │     0.5309 │
├────────────┼────────┼────────────┤
│ test       │ 4.7099 │     0.5111 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch  23
Training: 100% 5/5 [00:00<00:00, 462.39it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 802.34it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 839.87it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 838.19it/s]
Took 0.2412s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 4.1710 │     0.5127 │
├────────────┼────────┼────────────┤
│ vali       │ 3.8767 │     0.5556 │
├────────────┼────────┼────────────┤
│ test       │ 4.5776 │     0.5222 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch  24
Training: 100% 5/5 [00:00<00:00, 438.30it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 770.90it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 829.73it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 848.36it/s]
Took 0.2502s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 4.0692 │     0.5238 │
├────────────┼────────┼────────────┤
│ vali       │ 3.7784 │     0.5679 │
├────────────┼────────┼────────────┤
│ test       │ 4.4520 │     0.5444 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch  25
Training: 100% 5/5 [00:00<00:00, 419.62it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 783.22it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 736.49it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 844.01it/s]
Took 0.2558s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 3.9733 │     0.5349 │
├────────────┼────────┼────────────┤
│ vali       │ 3.6860 │     0.5926 │
├────────────┼────────┼────────────┤
│ test       │ 4.3329 │     0.5556 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch  26
Training: 100% 5/5 [00:00<00:00, 486.97it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 792.13it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 666.82it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 836.27it/s]
Took 0.2403s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 3.8830 │     0.5413 │
├────────────┼────────┼────────────┤
│ vali       │ 3.5990 │     0.5926 │
├────────────┼────────┼────────────┤
│ test       │ 4.2202 │     0.5722 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch  27
Training: 100% 5/5 [00:00<00:00, 437.38it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 752.83it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 726.66it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 827.28it/s]
Took 0.2552s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 3.7980 │     0.5571 │
├────────────┼────────┼────────────┤
│ vali       │ 3.5170 │     0.6173 │
├────────────┼────────┼────────────┤
│ test       │ 4.1136 │     0.5944 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch  28
Training: 100% 5/5 [00:00<00:00, 473.07it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 713.32it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 832.70it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 814.67it/s]
Took 0.2473s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 3.7180 │     0.5587 │
├────────────┼────────┼────────────┤
│ vali       │ 3.4395 │     0.6049 │
├────────────┼────────┼────────────┤
│ test       │ 4.0128 │     0.5944 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch  29
Training: 100% 5/5 [00:00<00:00, 454.26it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 782.78it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 830.88it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 813.80it/s]
Took 0.2479s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 3.6428 │     0.5683 │
├────────────┼────────┼────────────┤
│ vali       │ 3.3662 │     0.6049 │
├────────────┼────────┼────────────┤
│ test       │ 3.9176 │     0.6056 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch  30
Training: 100% 5/5 [00:00<00:00, 416.43it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 780.74it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 793.32it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 820.32it/s]
Took 0.2561s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 3.5719 │     0.5651 │
├────────────┼────────┼────────────┤
│ vali       │ 3.2969 │     0.6296 │
├────────────┼────────┼────────────┤
│ test       │ 3.8276 │     0.6000 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch  31
Training: 100% 5/5 [00:00<00:00, 454.96it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 812.00it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 854.41it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 850.60it/s]
Took 0.2407s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 3.5053 │     0.5746 │
├────────────┼────────┼────────────┤
│ vali       │ 3.2311 │     0.6296 │
├────────────┼────────┼────────────┤
│ test       │ 3.7425 │     0.6000 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch  32
Training: 100% 5/5 [00:00<00:00, 464.45it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 773.60it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 701.98it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 806.05it/s]
Took 0.2474s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 3.4424 │     0.5841 │
├────────────┼────────┼────────────┤
│ vali       │ 3.1687 │     0.6296 │
├────────────┼────────┼────────────┤
│ test       │ 3.6620 │     0.6056 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch  33
Training: 100% 5/5 [00:00<00:00, 453.83it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 757.97it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 718.20it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 551.23it/s]
Took 0.2634s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 3.3830 │     0.5857 │
├────────────┼────────┼────────────┤
│ vali       │ 3.1093 │     0.6296 │
├────────────┼────────┼────────────┤
│ test       │ 3.5857 │     0.6056 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch  34
Training: 100% 5/5 [00:00<00:00, 440.54it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 778.19it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 722.91it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 799.68it/s]
Took 0.2574s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 3.3268 │     0.5857 │
├────────────┼────────┼────────────┤
│ vali       │ 3.0528 │     0.6296 │
├────────────┼────────┼────────────┤
│ test       │ 3.5134 │     0.6167 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch  35
Training: 100% 5/5 [00:00<00:00, 466.61it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 784.80it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 818.40it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 873.90it/s]
Took 0.2424s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 3.2734 │     0.5952 │
├────────────┼────────┼────────────┤
│ vali       │ 2.9989 │     0.6296 │
├────────────┼────────┼────────────┤
│ test       │ 3.4448 │     0.6222 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch  36
Training: 100% 5/5 [00:00<00:00, 486.31it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 796.12it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 785.01it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 845.20it/s]
Took 0.2379s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 3.2225 │     0.5984 │
├────────────┼────────┼────────────┤
│ vali       │ 2.9473 │     0.6296 │
├────────────┼────────┼────────────┤
│ test       │ 3.3795 │     0.6333 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch  37
Training: 100% 5/5 [00:00<00:00, 444.96it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 758.55it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 722.78it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 819.68it/s]
Took 0.2551s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 3.1740 │     0.5937 │
├────────────┼────────┼────────────┤
│ vali       │ 2.8980 │     0.6420 │
├────────────┼────────┼────────────┤
│ test       │ 3.3173 │     0.6389 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch  38
Training: 100% 5/5 [00:00<00:00, 420.89it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 715.48it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 774.43it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 830.64it/s]
Took 0.2631s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 3.1275 │     0.5952 │
├────────────┼────────┼────────────┤
│ vali       │ 2.8507 │     0.6420 │
├────────────┼────────┼────────────┤
│ test       │ 3.2579 │     0.6500 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch  39
Training: 100% 5/5 [00:00<00:00, 435.36it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 769.40it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 820.80it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 863.11it/s]
Took 0.2513s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 3.0828 │     0.5952 │
├────────────┼────────┼────────────┤
│ vali       │ 2.8052 │     0.6420 │
├────────────┼────────┼────────────┤
│ test       │ 3.2010 │     0.6444 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch  40
Training: 100% 5/5 [00:00<00:00, 469.51it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 747.19it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 777.30it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 871.63it/s]
Took 0.2456s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 3.0398 │     0.5968 │
├────────────┼────────┼────────────┤
│ vali       │ 2.7613 │     0.6420 │
├────────────┼────────┼────────────┤
│ test       │ 3.1465 │     0.6556 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch  41
Training: 100% 5/5 [00:00<00:00, 469.54it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 796.40it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 780.92it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 790.63it/s]
Took 0.2431s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 2.9982 │     0.5984 │
├────────────┼────────┼────────────┤
│ vali       │ 2.7190 │     0.6420 │
├────────────┼────────┼────────────┤
│ test       │ 3.0941 │     0.6556 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch  42
Training: 100% 5/5 [00:00<00:00, 430.72it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 731.10it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 704.45it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 851.98it/s]
Took 0.2605s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 2.9580 │     0.5952 │
├────────────┼────────┼────────────┤
│ vali       │ 2.6781 │     0.6420 │
├────────────┼────────┼────────────┤
│ test       │ 3.0437 │     0.6556 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch  43
Training: 100% 5/5 [00:00<00:00, 351.76it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 660.10it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 752.07it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 560.36it/s]
Took 0.3093s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 2.9190 │     0.5952 │
├────────────┼────────┼────────────┤
│ vali       │ 2.6384 │     0.6420 │
├────────────┼────────┼────────────┤
│ test       │ 2.9950 │     0.6556 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch  44
Training: 100% 5/5 [00:00<00:00, 436.36it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 725.21it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 770.87it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 806.67it/s]
Took 0.2577s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 2.8810 │     0.5952 │
├────────────┼────────┼────────────┤
│ vali       │ 2.5999 │     0.6420 │
├────────────┼────────┼────────────┤
│ test       │ 2.9479 │     0.6667 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch  45
Training: 100% 5/5 [00:00<00:00, 432.57it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 765.41it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 795.13it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 815.93it/s]
Took 0.2563s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 2.8441 │     0.5952 │
├────────────┼────────┼────────────┤
│ vali       │ 2.5625 │     0.6420 │
├────────────┼────────┼────────────┤
│ test       │ 2.9023 │     0.6556 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch  46
Training: 100% 5/5 [00:00<00:00, 466.43it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 804.96it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 814.27it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 849.74it/s]
Took 0.2414s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 2.8080 │     0.5952 │
├────────────┼────────┼────────────┤
│ vali       │ 2.5261 │     0.6420 │
├────────────┼────────┼────────────┤
│ test       │ 2.8581 │     0.6556 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch  47
Training: 100% 5/5 [00:00<00:00, 456.46it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 793.74it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 790.93it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 823.14it/s]
Took 0.2458s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 2.7728 │     0.6000 │
├────────────┼────────┼────────────┤
│ vali       │ 2.4907 │     0.6420 │
├────────────┼────────┼────────────┤
│ test       │ 2.8150 │     0.6611 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch  48
Training: 100% 5/5 [00:00<00:00, 413.87it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 769.88it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 783.69it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 816.17it/s]
Took 0.2590s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 2.7383 │     0.6016 │
├────────────┼────────┼────────────┤
│ vali       │ 2.4560 │     0.6420 │
├────────────┼────────┼────────────┤
│ test       │ 2.7732 │     0.6611 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch  49
Training: 100% 5/5 [00:00<00:00, 467.02it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 764.30it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 827.28it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 856.24it/s]
Took 0.2435s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 2.7044 │     0.6016 │
├────────────┼────────┼────────────┤
│ vali       │ 2.4222 │     0.6296 │
├────────────┼────────┼────────────┤
│ test       │ 2.7323 │     0.6611 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch  50
Training: 100% 5/5 [00:00<00:00, 436.84it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 797.37it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 840.37it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 858.52it/s]
Took 0.2484s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 2.6712 │     0.6000 │
├────────────┼────────┼────────────┤
│ vali       │ 2.3891 │     0.6296 │
├────────────┼────────┼────────────┤
│ test       │ 2.6924 │     0.6611 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch  51
Training: 100% 5/5 [00:00<00:00, 328.96it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 792.69it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 830.06it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 841.38it/s]
Took 0.2872s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 2.6385 │     0.6016 │
├────────────┼────────┼────────────┤
│ vali       │ 2.3566 │     0.6296 │
├────────────┼────────┼────────────┤
│ test       │ 2.6534 │     0.6611 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch  52
Training: 100% 5/5 [00:00<00:00, 243.78it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 790.96it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 726.66it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 863.38it/s]
Took 0.3631s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 2.6063 │     0.6016 │
├────────────┼────────┼────────────┤
│ vali       │ 2.3248 │     0.6296 │
├────────────┼────────┼────────────┤
│ test       │ 2.6153 │     0.6667 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch  53
Training: 100% 5/5 [00:00<00:00, 413.96it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 732.76it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 759.70it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 834.11it/s]
Took 0.2628s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 2.5745 │     0.6016 │
├────────────┼────────┼────────────┤
│ vali       │ 2.2935 │     0.6296 │
├────────────┼────────┼────────────┤
│ test       │ 2.5779 │     0.6667 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch  54
Training: 100% 5/5 [00:00<00:00, 465.53it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 769.37it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 813.95it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 859.14it/s]
Took 0.2446s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 2.5432 │     0.6032 │
├────────────┼────────┼────────────┤
│ vali       │ 2.2627 │     0.6543 │
├────────────┼────────┼────────────┤
│ test       │ 2.5412 │     0.6667 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch  55
Training: 100% 5/5 [00:00<00:00, 437.41it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 794.35it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 796.19it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 832.45it/s]
Took 0.2502s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 2.5122 │     0.6048 │
├────────────┼────────┼────────────┤
│ vali       │ 2.2324 │     0.6543 │
├────────────┼────────┼────────────┤
│ test       │ 2.5052 │     0.6667 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch  56
Training: 100% 5/5 [00:00<00:00, 274.75it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 759.95it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 731.99it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 769.46it/s]
Took 0.3527s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 2.4816 │     0.6048 │
├────────────┼────────┼────────────┤
│ vali       │ 2.2026 │     0.6543 │
├────────────┼────────┼────────────┤
│ test       │ 2.4698 │     0.6667 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch  57
Training: 100% 5/5 [00:00<00:00, 289.87it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 646.33it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 824.68it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 797.85it/s]
Took 0.3365s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 2.4513 │     0.6048 │
├────────────┼────────┼────────────┤
│ vali       │ 2.1732 │     0.6543 │
├────────────┼────────┼────────────┤
│ test       │ 2.4349 │     0.6667 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch  58
Training: 100% 5/5 [00:00<00:00, 470.30it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 807.34it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 803.51it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 825.89it/s]
Took 0.2415s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 2.4213 │     0.6048 │
├────────────┼────────┼────────────┤
│ vali       │ 2.1442 │     0.6543 │
├────────────┼────────┼────────────┤
│ test       │ 2.4007 │     0.6611 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch  59
Training: 100% 5/5 [00:00<00:00, 450.07it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 746.18it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 740.00it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 796.26it/s]
Took 0.2662s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 2.3916 │     0.6016 │
├────────────┼────────┼────────────┤
│ vali       │ 2.1155 │     0.6543 │
├────────────┼────────┼────────────┤
│ test       │ 2.3669 │     0.6611 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch  60
Training: 100% 5/5 [00:00<00:00, 460.15it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 760.75it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 764.27it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 841.05it/s]
Took 0.2500s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 2.3621 │     0.6048 │
├────────────┼────────┼────────────┤
│ vali       │ 2.0872 │     0.6543 │
├────────────┼────────┼────────────┤
│ test       │ 2.3337 │     0.6611 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch  61
Training: 100% 5/5 [00:00<00:00, 403.43it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 710.10it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 840.21it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 632.34it/s]
Took 0.2807s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 2.3328 │     0.6032 │
├────────────┼────────┼────────────┤
│ vali       │ 2.0592 │     0.6543 │
├────────────┼────────┼────────────┤
│ test       │ 2.3009 │     0.6611 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch  62
Training: 100% 5/5 [00:00<00:00, 428.22it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 668.14it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 777.15it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 669.21it/s]
Took 0.2748s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 2.3038 │     0.6032 │
├────────────┼────────┼────────────┤
│ vali       │ 2.0315 │     0.6543 │
├────────────┼────────┼────────────┤
│ test       │ 2.2685 │     0.6611 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch  63
Training: 100% 5/5 [00:00<00:00, 454.09it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 797.18it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 780.63it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 845.20it/s]
Took 0.2465s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 2.2750 │     0.6032 │
├────────────┼────────┼────────────┤
│ vali       │ 2.0042 │     0.6543 │
├────────────┼────────┼────────────┤
│ test       │ 2.2366 │     0.6611 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch  64
Training: 100% 5/5 [00:00<00:00, 453.88it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 608.81it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 757.92it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 833.77it/s]
Took 0.2652s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 2.2464 │     0.6016 │
├────────────┼────────┼────────────┤
│ vali       │ 1.9770 │     0.6543 │
├────────────┼────────┼────────────┤
│ test       │ 2.2051 │     0.6611 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch  65
Training: 100% 5/5 [00:00<00:00, 412.13it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 727.52it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 776.29it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 792.42it/s]
Took 0.2643s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 2.2180 │     0.6032 │
├────────────┼────────┼────────────┤
│ vali       │ 1.9502 │     0.6543 │
├────────────┼────────┼────────────┤
│ test       │ 2.1739 │     0.6667 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch  66
Training: 100% 5/5 [00:00<00:00, 427.18it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 747.65it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 807.68it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 777.30it/s]
Took 0.2586s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 2.1898 │     0.6032 │
├────────────┼────────┼────────────┤
│ vali       │ 1.9236 │     0.6543 │
├────────────┼────────┼────────────┤
│ test       │ 2.1431 │     0.6722 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch  67
Training: 100% 5/5 [00:00<00:00, 471.89it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 803.51it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 832.86it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 880.42it/s]
Took 0.2424s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 2.1618 │     0.6032 │
├────────────┼────────┼────────────┤
│ vali       │ 1.8973 │     0.6543 │
├────────────┼────────┼────────────┤
│ test       │ 2.1127 │     0.6722 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch  68
Training: 100% 5/5 [00:00<00:00, 298.62it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 796.73it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 829.08it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 844.77it/s]
Took 0.3015s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 2.1339 │     0.6032 │
├────────────┼────────┼────────────┤
│ vali       │ 1.8712 │     0.6543 │
├────────────┼────────┼────────────┤
│ test       │ 2.0826 │     0.6722 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch  69
Training: 100% 5/5 [00:00<00:00, 447.07it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 346.37it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 766.08it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 860.72it/s]
Took 0.3286s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 2.1062 │     0.6032 │
├────────────┼────────┼────────────┤
│ vali       │ 1.8453 │     0.6543 │
├────────────┼────────┼────────────┤
│ test       │ 2.0529 │     0.6722 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch  70
Training: 100% 5/5 [00:00<00:00, 308.59it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 674.74it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 823.38it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 851.89it/s]
Took 0.3164s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 2.0787 │     0.6032 │
├────────────┼────────┼────────────┤
│ vali       │ 1.8196 │     0.6543 │
├────────────┼────────┼────────────┤
│ test       │ 2.0235 │     0.6722 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch  71
Training: 100% 5/5 [00:00<00:00, 460.36it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 403.67it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 713.68it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 831.71it/s]
Took 0.3091s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 2.0514 │     0.6032 │
├────────────┼────────┼────────────┤
│ vali       │ 1.7942 │     0.6543 │
├────────────┼────────┼────────────┤
│ test       │ 1.9944 │     0.6778 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch  72
Training: 100% 5/5 [00:00<00:00, 453.82it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 417.22it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 690.42it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 848.45it/s]
Took 0.3130s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 2.0242 │     0.6048 │
├────────────┼────────┼────────────┤
│ vali       │ 1.7690 │     0.6543 │
├────────────┼────────┼────────────┤
│ test       │ 1.9655 │     0.6778 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch  73
Training: 100% 5/5 [00:00<00:00, 450.93it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 742.78it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 693.50it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 805.05it/s]
Took 0.2543s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 1.9972 │     0.6048 │
├────────────┼────────┼────────────┤
│ vali       │ 1.7440 │     0.6543 │
├────────────┼────────┼────────────┤
│ test       │ 1.9370 │     0.6778 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch  74
Training: 100% 5/5 [00:00<00:00, 451.47it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 825.81it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 801.51it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 789.59it/s]
Took 0.2448s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 1.9703 │     0.6048 │
├────────────┼────────┼────────────┤
│ vali       │ 1.7192 │     0.6543 │
├────────────┼────────┼────────────┤
│ test       │ 1.9088 │     0.6778 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch  75
Training: 100% 5/5 [00:00<00:00, 449.40it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 767.51it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 843.08it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 845.71it/s]
Took 0.2527s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 1.9436 │     0.6048 │
├────────────┼────────┼────────────┤
│ vali       │ 1.6946 │     0.6543 │
├────────────┼────────┼────────────┤
│ test       │ 1.8808 │     0.6778 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch  76
Training: 100% 5/5 [00:00<00:00, 320.58it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 734.19it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 723.90it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 826.30it/s]
Took 0.3138s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 1.9171 │     0.6063 │
├────────────┼────────┼────────────┤
│ vali       │ 1.6703 │     0.6667 │
├────────────┼────────┼────────────┤
│ test       │ 1.8532 │     0.6778 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch  77
Training: 100% 5/5 [00:00<00:00, 466.08it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 818.85it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 748.45it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 858.17it/s]
Took 0.2412s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 1.8907 │     0.6079 │
├────────────┼────────┼────────────┤
│ vali       │ 1.6461 │     0.6667 │
├────────────┼────────┼────────────┤
│ test       │ 1.8258 │     0.6778 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch  78
Training: 100% 5/5 [00:00<00:00, 421.08it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 761.99it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 858.61it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 865.34it/s]
Took 0.2541s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 1.8645 │     0.6079 │
├────────────┼────────┼────────────┤
│ vali       │ 1.6221 │     0.6667 │
├────────────┼────────┼────────────┤
│ test       │ 1.7986 │     0.6778 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch  79
Training: 100% 5/5 [00:00<00:00, 452.93it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 779.99it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 815.38it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 813.24it/s]
Took 0.2471s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 1.8385 │     0.6095 │
├────────────┼────────┼────────────┤
│ vali       │ 1.5984 │     0.6667 │
├────────────┼────────┼────────────┤
│ test       │ 1.7718 │     0.6778 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch  80
Training: 100% 5/5 [00:00<00:00, 442.53it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 768.78it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 770.30it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 761.42it/s]
Took 0.2552s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 1.8127 │     0.6095 │
├────────────┼────────┼────────────┤
│ vali       │ 1.5749 │     0.6667 │
├────────────┼────────┼────────────┤
│ test       │ 1.7451 │     0.6778 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch  81
Training: 100% 5/5 [00:00<00:00, 414.58it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 758.85it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 840.54it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 777.88it/s]
Took 0.2594s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 1.7870 │     0.6095 │
├────────────┼────────┼────────────┤
│ vali       │ 1.5515 │     0.6667 │
├────────────┼────────┼────────────┤
│ test       │ 1.7188 │     0.6778 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch  82
Training: 100% 5/5 [00:00<00:00, 391.73it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 751.61it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 690.99it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 841.89it/s]
Took 0.2698s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 1.7615 │     0.6127 │
├────────────┼────────┼────────────┤
│ vali       │ 1.5284 │     0.6667 │
├────────────┼────────┼────────────┤
│ test       │ 1.6927 │     0.6778 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch  83
Training: 100% 5/5 [00:00<00:00, 301.75it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 746.69it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 874.91it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 819.52it/s]
Took 0.3045s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 1.7362 │     0.6127 │
├────────────┼────────┼────────────┤
│ vali       │ 1.5055 │     0.6667 │
├────────────┼────────┼────────────┤
│ test       │ 1.6668 │     0.6778 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch  84
Training: 100% 5/5 [00:00<00:00, 369.40it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 655.20it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 774.43it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 786.19it/s]
Took 0.2939s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 1.7110 │     0.6127 │
├────────────┼────────┼────────────┤
│ vali       │ 1.4828 │     0.6667 │
├────────────┼────────┼────────────┤
│ test       │ 1.6412 │     0.6778 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch  85
Training: 100% 5/5 [00:00<00:00, 475.28it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 819.07it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 818.72it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 850.60it/s]
Took 0.2376s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 1.6861 │     0.6111 │
├────────────┼────────┼────────────┤
│ vali       │ 1.4603 │     0.6667 │
├────────────┼────────┼────────────┤
│ test       │ 1.6158 │     0.6778 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch  86
Training: 100% 5/5 [00:00<00:00, 422.43it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 795.88it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 786.04it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 720.98it/s]
Took 0.2584s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 1.6613 │     0.6111 │
├────────────┼────────┼────────────┤
│ vali       │ 1.4381 │     0.6667 │
├────────────┼────────┼────────────┤
│ test       │ 1.5907 │     0.6778 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch  87
Training: 100% 5/5 [00:00<00:00, 440.43it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 795.64it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 722.41it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 877.65it/s]
Took 0.2490s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 1.6367 │     0.6111 │
├────────────┼────────┼────────────┤
│ vali       │ 1.4161 │     0.6667 │
├────────────┼────────┼────────────┤
│ test       │ 1.5658 │     0.6778 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch  88
Training: 100% 5/5 [00:00<00:00, 421.30it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 774.97it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 791.08it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 698.35it/s]
Took 0.3407s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 1.6124 │     0.6111 │
├────────────┼────────┼────────────┤
│ vali       │ 1.3942 │     0.6667 │
├────────────┼────────┼────────────┤
│ test       │ 1.5412 │     0.6778 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch  89
Training: 100% 5/5 [00:00<00:00, 430.04it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 790.39it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 582.62it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 813.48it/s]
Took 0.2596s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 1.5882 │     0.6111 │
├────────────┼────────┼────────────┤
│ vali       │ 1.3727 │     0.6790 │
├────────────┼────────┼────────────┤
│ test       │ 1.5168 │     0.6833 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch  90
Training: 100% 5/5 [00:00<00:00, 437.97it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 767.60it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 649.37it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 787.81it/s]
Took 0.2602s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 1.5642 │     0.6159 │
├────────────┼────────┼────────────┤
│ vali       │ 1.3513 │     0.6790 │
├────────────┼────────┼────────────┤
│ test       │ 1.4927 │     0.6889 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch  91
Training: 100% 5/5 [00:00<00:00, 457.56it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 743.46it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 829.24it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 845.71it/s]
Took 0.2483s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 1.5405 │     0.6175 │
├────────────┼────────┼────────────┤
│ vali       │ 1.3302 │     0.6914 │
├────────────┼────────┼────────────┤
│ test       │ 1.4688 │     0.6889 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch  92
Training: 100% 5/5 [00:00<00:00, 440.29it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 706.11it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 718.57it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 856.50it/s]
Took 0.2597s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 1.5169 │     0.6365 │
├────────────┼────────┼────────────┤
│ vali       │ 1.3093 │     0.7037 │
├────────────┼────────┼────────────┤
│ test       │ 1.4452 │     0.6889 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch  93
Training: 100% 5/5 [00:00<00:00, 384.35it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 760.47it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 800.44it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 839.62it/s]
Took 0.2693s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 1.4936 │     0.6524 │
├────────────┼────────┼────────────┤
│ vali       │ 1.2886 │     0.7160 │
├────────────┼────────┼────────────┤
│ test       │ 1.4218 │     0.6944 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch  94
Training: 100% 5/5 [00:00<00:00, 459.83it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 770.96it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 817.44it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 846.82it/s]
Took 0.2464s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 1.4705 │     0.6524 │
├────────────┼────────┼────────────┤
│ vali       │ 1.2682 │     0.7160 │
├────────────┼────────┼────────────┤
│ test       │ 1.3987 │     0.6889 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch  95
Training: 100% 5/5 [00:00<00:00, 468.25it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 748.80it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 863.56it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 848.36it/s]
Took 0.2432s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 1.4476 │     0.6540 │
├────────────┼────────┼────────────┤
│ vali       │ 1.2481 │     0.7160 │
├────────────┼────────┼────────────┤
│ test       │ 1.3758 │     0.6889 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch  96
Training: 100% 5/5 [00:00<00:00, 444.56it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 790.66it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 829.73it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 729.89it/s]
Took 0.2514s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 1.4249 │     0.6524 │
├────────────┼────────┼────────────┤
│ vali       │ 1.2282 │     0.7160 │
├────────────┼────────┼────────────┤
│ test       │ 1.3531 │     0.6889 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch  97
Training: 100% 5/5 [00:00<00:00, 433.12it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 751.99it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 810.65it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 832.29it/s]
Took 0.2544s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 1.4025 │     0.6508 │
├────────────┼────────┼────────────┤
│ vali       │ 1.2085 │     0.7160 │
├────────────┼────────┼────────────┤
│ test       │ 1.3308 │     0.6889 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch  98
Training: 100% 5/5 [00:00<00:00, 475.64it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 804.34it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 804.59it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 834.60it/s]
Took 0.2385s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 1.3804 │     0.6540 │
├────────────┼────────┼────────────┤
│ vali       │ 1.1891 │     0.7160 │
├────────────┼────────┼────────────┤
│ test       │ 1.3087 │     0.6944 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch  99
Training: 100% 5/5 [00:00<00:00, 468.49it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 713.03it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 772.57it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 819.92it/s]
Took 0.2515s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 1.3584 │     0.6540 │
├────────────┼────────┼────────────┤
│ vali       │ 1.1699 │     0.7160 │
├────────────┼────────┼────────────┤
│ test       │ 1.2868 │     0.6889 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved


Epoch 100
Training: 100% 5/5 [00:00<00:00, 412.88it/s]
Evaluation train: 100% 5/5 [00:00<00:00, 788.17it/s]
Evaluation vali : 100% 1/1 [00:00<00:00, 792.72it/s]
Evaluation test : 100% 2/2 [00:00<00:00, 849.48it/s]
Took 0.2582s
╒════════════╤════════╤════════════╕
│ Survived   │   loss │   accuracy │
╞════════════╪════════╪════════════╡
│ train      │ 1.3368 │     0.6540 │
├────────────┼────────┼────────────┤
│ vali       │ 1.1510 │     0.7160 │
├────────────┼────────┼────────────┤
│ test       │ 1.2652 │     0.6889 │
╘════════════╧════════╧════════════╛
Validation loss on combined improved, model saved

Best validation model epoch: 100
Best validation model loss on validation set combined: 1.1510242415063174
Best validation model loss on test set combined: 1.2652276780870226

╒═════════╕
│ PREDICT │
╘═════════╛

Evaluation: 100% 2/2 [00:00<00:00, 50.27it/s]

===== Survived =====
accuracy: 0.6888888888888889
average_precision_macro: 0.6052279888757147
average_precision_micro: 0.6052279888757147
average_precision_samples: 0.6052279888757147
loss: 1.2652276780870226
overall_stats: { 'avg_f1_score_macro': 0.68,
  'avg_f1_score_micro': 0.6888888888888889,
  'avg_f1_score_weighted': 0.6948148148148149,
  'avg_precision_macro': 0.681733746130031,
  'avg_precision_micro': 0.6888888888888889,
  'avg_precision_weighted': 0.6888888888888889,
  'avg_recall_macro': 0.6963210702341137,
  'avg_recall_micro': 0.6888888888888889,
  'avg_recall_weighted': 0.6888888888888889,
  'kappa_score': 0.36802507836990594,
  'overall_accuracy': 0.6888888888888889}
per_class_stats: {False: {   'accuracy': 0.6888888888888889,
    'f1_score': 0.7333333333333334,
    'fall_out': 0.44705882352941173,
    'false_discovery_rate': 0.33043478260869563,
    'false_negative_rate': 0.18947368421052635,
    'false_negatives': 18,
    'false_omission_rate': 0.27692307692307694,
    'false_positive_rate': 0.44705882352941173,
    'false_positives': 38,
    'hit_rate': 0.8105263157894737,
    'informedness': 0.36346749226006203,
    'markedness': 0.39264214046822743,
    'matthews_correlation_coefficient': 0.377773284062822,
    'miss_rate': 0.18947368421052635,
    'negative_predictive_value': 0.7230769230769231,
    'positive_predictive_value': 0.6695652173913044,
    'precision': 0.6695652173913044,
    'recall': 0.8105263157894737,
    'sensitivity': 0.8105263157894737,
    'specificity': 0.5529411764705883,
    'true_negative_rate': 0.5529411764705883,
    'true_negatives': 47,
    'true_positive_rate': 0.8105263157894737,
    'true_positives': 77},
  True: {   'accuracy': 0.6888888888888889,
    'f1_score': 0.6266666666666667,
    'fall_out': 0.18947368421052635,
    'false_discovery_rate': 0.27692307692307694,
    'false_negative_rate': 0.44705882352941173,
    'false_negatives': 38,
    'false_omission_rate': 0.33043478260869563,
    'false_positive_rate': 0.18947368421052635,
    'false_positives': 18,
    'hit_rate': 0.5529411764705883,
    'informedness': 0.36346749226006203,
    'markedness': 0.39264214046822743,
    'matthews_correlation_coefficient': 0.377773284062822,
    'miss_rate': 0.44705882352941173,
    'negative_predictive_value': 0.6695652173913044,
    'positive_predictive_value': 0.7230769230769231,
    'precision': 0.7230769230769231,
    'recall': 0.5529411764705883,
    'sensitivity': 0.5529411764705883,
    'specificity': 0.8105263157894737,
    'true_negative_rate': 0.8105263157894737,
    'true_negatives': 77,
    'true_positive_rate': 0.5529411764705883,
    'true_positives': 47}}
precision_recall_curve: { 'precisions': [ 0.39156626506024095,
                  0.3878787878787879,
                  0.3902439024390244,
                  0.39263803680981596,
                  0.3950617283950617,
                  0.39751552795031053,
                  0.4,
                  0.4025157232704403,
                  0.4050632911392405,
                  0.40764331210191085,
                  0.40384615384615385,
                  0.4064516129032258,
                  0.4090909090909091,
                  0.4117647058823529,
                  0.4144736842105263,
                  0.41721854304635764,
                  0.42,
                  0.4161073825503356,
                  0.4189189189189189,
                  0.4217687074829932,
                  0.4246575342465753,
                  0.42758620689655175,
                  0.4305555555555556,
                  0.43356643356643354,
                  0.43661971830985913,
                  0.4326241134751773,
                  0.4357142857142857,
                  0.43884892086330934,
                  0.4420289855072464,
                  0.44525547445255476,
                  0.4485294117647059,
                  0.45185185185185184,
                  0.4552238805970149,
                  0.45864661654135336,
                  0.4621212121212121,
                  0.4580152671755725,
                  0.46153846153846156,
                  0.46511627906976744,
                  0.46875,
                  0.47244094488188976,
                  0.47619047619047616,
                  0.48,
                  0.4838709677419355,
                  0.4878048780487805,
                  0.4918032786885246,
                  0.49586776859504134,
                  0.49166666666666664,
                  0.4957983193277311,
                  0.5,
                  0.5042735042735043,
                  0.5086206896551724,
                  0.5130434782608696,
                  0.5175438596491229,
                  0.5132743362831859,
                  0.5178571428571429,
                  0.5135135135135135,
                  0.509090909090909,
                  0.5045871559633027,
                  0.5092592592592593,
                  0.514018691588785,
                  0.5094339622641509,
                  0.5142857142857142,
                  0.5192307692307693,
                  0.5242718446601942,
                  0.5196078431372549,
                  0.5247524752475248,
                  0.53,
                  0.5353535353535354,
                  0.5408163265306123,
                  0.5463917525773195,
                  0.5520833333333334,
                  0.5473684210526316,
                  0.5425531914893617,
                  0.5483870967741935,
                  0.5604395604395604,
                  0.5555555555555556,
                  0.550561797752809,
                  0.5568181818181818,
                  0.5517241379310345,
                  0.5581395348837209,
                  0.5529411764705883,
                  0.5476190476190477,
                  0.5542168674698795,
                  0.5609756097560976,
                  0.5679012345679012,
                  0.575,
                  0.569620253164557,
                  0.5641025641025641,
                  0.5584415584415584,
                  0.5526315789473685,
                  0.56,
                  0.5833333333333334,
                  0.5915492957746479,
                  0.5857142857142857,
                  0.5942028985507246,
                  0.5970149253731343,
                  0.6060606060606061,
                  0.6,
                  0.59375,
                  0.6031746031746031,
                  0.5967741935483871,
                  0.5901639344262295,
                  0.6,
                  0.5932203389830508,
                  0.603448275862069,
                  0.5964912280701754,
                  0.6071428571428571,
                  0.6181818181818182,
                  0.6296296296296297,
                  0.6153846153846154,
                  0.6078431372549019,
                  0.62,
                  0.6122448979591837,
                  0.6041666666666666,
                  0.6170212765957447,
                  0.6086956521739131,
                  0.6,
                  0.6136363636363636,
                  0.627906976744186,
                  0.6190476190476191,
                  0.6097560975609756,
                  0.625,
                  0.6153846153846154,
                  0.631578947368421,
                  0.6216216216216216,
                  0.6111111111111112,
                  0.6285714285714286,
                  0.6470588235294118,
                  0.6363636363636364,
                  0.625,
                  0.6451612903225806,
                  0.6333333333333333,
                  0.6206896551724138,
                  0.6428571428571429,
                  0.6666666666666666,
                  0.6538461538461539,
                  0.64,
                  0.625,
                  0.6521739130434783,
                  0.6363636363636364,
                  0.6190476190476191,
                  0.65,
                  0.631578947368421,
                  0.6111111111111112,
                  0.6470588235294118,
                  0.625,
                  0.6666666666666666,
                  0.7142857142857143,
                  0.7692307692307693,
                  0.75,
                  0.8181818181818182,
                  0.8,
                  0.7777777777777778,
                  0.75,
                  0.7142857142857143,
                  1.0],
  'recalls': [ 1.0,
               0.9846153846153847,
               0.9846153846153847,
               0.9846153846153847,
               0.9846153846153847,
               0.9846153846153847,
               0.9846153846153847,
               0.9846153846153847,
               0.9846153846153847,
               0.9846153846153847,
               0.9692307692307692,
               0.9692307692307692,
               0.9692307692307692,
               0.9692307692307692,
               0.9692307692307692,
               0.9692307692307692,
               0.9692307692307692,
               0.9538461538461539,
               0.9538461538461539,
               0.9538461538461539,
               0.9538461538461539,
               0.9538461538461539,
               0.9538461538461539,
               0.9538461538461539,
               0.9538461538461539,
               0.9384615384615385,
               0.9384615384615385,
               0.9384615384615385,
               0.9384615384615385,
               0.9384615384615385,
               0.9384615384615385,
               0.9384615384615385,
               0.9384615384615385,
               0.9384615384615385,
               0.9384615384615385,
               0.9230769230769231,
               0.9230769230769231,
               0.9230769230769231,
               0.9230769230769231,
               0.9230769230769231,
               0.9230769230769231,
               0.9230769230769231,
               0.9230769230769231,
               0.9230769230769231,
               0.9230769230769231,
               0.9230769230769231,
               0.9076923076923077,
               0.9076923076923077,
               0.9076923076923077,
               0.9076923076923077,
               0.9076923076923077,
               0.9076923076923077,
               0.9076923076923077,
               0.8923076923076924,
               0.8923076923076924,
               0.8769230769230769,
               0.8615384615384616,
               0.8461538461538461,
               0.8461538461538461,
               0.8461538461538461,
               0.8307692307692308,
               0.8307692307692308,
               0.8307692307692308,
               0.8307692307692308,
               0.8153846153846154,
               0.8153846153846154,
               0.8153846153846154,
               0.8153846153846154,
               0.8153846153846154,
               0.8153846153846154,
               0.8153846153846154,
               0.8,
               0.7846153846153846,
               0.7846153846153846,
               0.7846153846153846,
               0.7692307692307693,
               0.7538461538461538,
               0.7538461538461538,
               0.7384615384615385,
               0.7384615384615385,
               0.7230769230769231,
               0.7076923076923077,
               0.7076923076923077,
               0.7076923076923077,
               0.7076923076923077,
               0.7076923076923077,
               0.6923076923076923,
               0.676923076923077,
               0.6615384615384615,
               0.6461538461538462,
               0.6461538461538462,
               0.6461538461538462,
               0.6461538461538462,
               0.6307692307692307,
               0.6307692307692307,
               0.6153846153846154,
               0.6153846153846154,
               0.6,
               0.5846153846153846,
               0.5846153846153846,
               0.5692307692307692,
               0.5538461538461539,
               0.5538461538461539,
               0.5384615384615384,
               0.5384615384615384,
               0.5230769230769231,
               0.5230769230769231,
               0.5230769230769231,
               0.5230769230769231,
               0.49230769230769234,
               0.47692307692307695,
               0.47692307692307695,
               0.46153846153846156,
               0.4461538461538462,
               0.4461538461538462,
               0.4307692307692308,
               0.4153846153846154,
               0.4153846153846154,
               0.4153846153846154,
               0.4,
               0.38461538461538464,
               0.38461538461538464,
               0.36923076923076925,
               0.36923076923076925,
               0.35384615384615387,
               0.3384615384615385,
               0.3384615384615385,
               0.3384615384615385,
               0.3230769230769231,
               0.3076923076923077,
               0.3076923076923077,
               0.2923076923076923,
               0.27692307692307694,
               0.27692307692307694,
               0.27692307692307694,
               0.26153846153846155,
               0.24615384615384617,
               0.23076923076923078,
               0.23076923076923078,
               0.2153846153846154,
               0.2,
               0.2,
               0.18461538461538463,
               0.16923076923076924,
               0.16923076923076924,
               0.15384615384615385,
               0.15384615384615385,
               0.15384615384615385,
               0.15384615384615385,
               0.13846153846153847,
               0.13846153846153847,
               0.12307692307692308,
               0.1076923076923077,
               0.09230769230769231,
               0.07692307692307693,
               0.0]}
roc_auc_macro: 0.7635451505016722
roc_auc_micro: 0.7635451505016722

Finished: experiment_run
Saved to: results/experiment_run_0
```
</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```
!ludwig visualize --visualization learning_curves --training_statistics results/experiment_run_0/training_statistics.json

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```

WARNING: The TensorFlow contrib module will not be included in TensorFlow 2.0.
For more information, please see:
  * https://github.com/tensorflow/community/blob/master/rfcs/20180907-contrib-sunset.md
  * https://github.com/tensorflow/addons
If you depend on functionality not listed there, please file an issue.

<Figure size 800x550 with 1 Axes>
<Figure size 800x550 with 1 Axes>
<Figure size 800x550 with 1 Axes>
<Figure size 800x550 with 1 Axes>
```
</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```!cd results/experiment_run_0/ &&ls

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
description.json	    Survived_predictions.npy
model			    Survived_probabilities.csv
prediction_statistics.json  Survived_probabilities.npy
Survived_predictions.csv    training_statistics.json
```
</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```!ludwig predict  --data_csv train.csv --model_path results/experiment_run_0/model/

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```

WARNING: The TensorFlow contrib module will not be included in TensorFlow 2.0.
For more information, please see:
  * https://github.com/tensorflow/community/blob/master/rfcs/20180907-contrib-sunset.md
  * https://github.com/tensorflow/addons
If you depend on functionality not listed there, please file an issue.

 _         _        _      
| |_  _ __| |_ __ _(_)__ _ 
| | || / _` \ V  V / / _` |
|_|\_,_\__,_|\_/\_/|_\__, |
                     |___/ 
ludwig v0.1.1 - Predict

Dataset type: generic
Dataset path: train.csv
Model path: results/experiment_run_0/model/
Output path: results_0

Found hdf5 with the same filename of the csv, using it instead
Loading metadata from: results/experiment_run_0/model/train_set_metadata.json
Loading data from: train.hdf5

╒═══════════════╕
│ LOADING MODEL │
╘═══════════════╛

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/framework/op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
Instructions for updating:
Colocations handled automatically by placer.
From /usr/local/lib/python3.6/dist-packages/tensorflow/python/framework/op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
Instructions for updating:
Colocations handled automatically by placer.
  embedding_size (50) is greater than vocab_size (4). Setting embedding size to be equal to vocab_size.
  embedding_size (50) is greater than vocab_size (3). Setting embedding size to be equal to vocab_size.
  embedding_size (50) is greater than vocab_size (5). Setting embedding size to be equal to vocab_size.
WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/ops/array_grad.py:425: to_int32 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.cast instead.
From /usr/local/lib/python3.6/dist-packages/tensorflow/python/ops/array_grad.py:425: to_int32 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.cast instead.

╒═════════╕
│ PREDICT │
╘═════════╛

2019-04-15 15:03:29.611185: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2300000000 Hz
2019-04-15 15:03:29.611485: I tensorflow/compiler/xla/service/service.cc:150] XLA service 0x2bfcc00 executing computations on platform Host. Devices:
2019-04-15 15:03:29.611522: I tensorflow/compiler/xla/service/service.cc:158]   StreamExecutor device (0): <undefined>, <undefined>
WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/training/saver.py:1266: checkpoint_exists (from tensorflow.python.training.checkpoint_management) is deprecated and will be removed in a future version.
Instructions for updating:
Use standard file APIs to check for files with this prefix.
From /usr/local/lib/python3.6/dist-packages/tensorflow/python/training/saver.py:1266: checkpoint_exists (from tensorflow.python.training.checkpoint_management) is deprecated and will be removed in a future version.
Instructions for updating:
Use standard file APIs to check for files with this prefix.
INFO:tensorflow:Restoring parameters from results/experiment_run_0/model/model_weights
Restoring parameters from results/experiment_run_0/model/model_weights
Evaluation: 100% 2/2 [00:00<00:00, 64.62it/s]

===== Survived =====
accuracy: 0.6888888888888889
average_precision_macro: 0.6052279888757147
average_precision_micro: 0.6052279888757147
average_precision_samples: 0.6052279888757147
loss: 1.2652276780870226
overall_stats: { 'avg_f1_score_macro': 0.68,
  'avg_f1_score_micro': 0.6888888888888889,
  'avg_f1_score_weighted': 0.6948148148148149,
  'avg_precision_macro': 0.681733746130031,
  'avg_precision_micro': 0.6888888888888889,
  'avg_precision_weighted': 0.6888888888888889,
  'avg_recall_macro': 0.6963210702341137,
  'avg_recall_micro': 0.6888888888888889,
  'avg_recall_weighted': 0.6888888888888889,
  'kappa_score': 0.36802507836990594,
  'overall_accuracy': 0.6888888888888889}
per_class_stats: {False: {   'accuracy': 0.6888888888888889,
    'f1_score': 0.7333333333333334,
    'fall_out': 0.44705882352941173,
    'false_discovery_rate': 0.33043478260869563,
    'false_negative_rate': 0.18947368421052635,
    'false_negatives': 18,
    'false_omission_rate': 0.27692307692307694,
    'false_positive_rate': 0.44705882352941173,
    'false_positives': 38,
    'hit_rate': 0.8105263157894737,
    'informedness': 0.36346749226006203,
    'markedness': 0.39264214046822743,
    'matthews_correlation_coefficient': 0.377773284062822,
    'miss_rate': 0.18947368421052635,
    'negative_predictive_value': 0.7230769230769231,
    'positive_predictive_value': 0.6695652173913044,
    'precision': 0.6695652173913044,
    'recall': 0.8105263157894737,
    'sensitivity': 0.8105263157894737,
    'specificity': 0.5529411764705883,
    'true_negative_rate': 0.5529411764705883,
    'true_negatives': 47,
    'true_positive_rate': 0.8105263157894737,
    'true_positives': 77},
  True: {   'accuracy': 0.6888888888888889,
    'f1_score': 0.6266666666666667,
    'fall_out': 0.18947368421052635,
    'false_discovery_rate': 0.27692307692307694,
    'false_negative_rate': 0.44705882352941173,
    'false_negatives': 38,
    'false_omission_rate': 0.33043478260869563,
    'false_positive_rate': 0.18947368421052635,
    'false_positives': 18,
    'hit_rate': 0.5529411764705883,
    'informedness': 0.36346749226006203,
    'markedness': 0.39264214046822743,
    'matthews_correlation_coefficient': 0.377773284062822,
    'miss_rate': 0.44705882352941173,
    'negative_predictive_value': 0.6695652173913044,
    'positive_predictive_value': 0.7230769230769231,
    'precision': 0.7230769230769231,
    'recall': 0.5529411764705883,
    'sensitivity': 0.5529411764705883,
    'specificity': 0.8105263157894737,
    'true_negative_rate': 0.8105263157894737,
    'true_negatives': 77,
    'true_positive_rate': 0.5529411764705883,
    'true_positives': 47}}
precision_recall_curve: { 'precisions': [ 0.39156626506024095,
                  0.3878787878787879,
                  0.3902439024390244,
                  0.39263803680981596,
                  0.3950617283950617,
                  0.39751552795031053,
                  0.4,
                  0.4025157232704403,
                  0.4050632911392405,
                  0.40764331210191085,
                  0.40384615384615385,
                  0.4064516129032258,
                  0.4090909090909091,
                  0.4117647058823529,
                  0.4144736842105263,
                  0.41721854304635764,
                  0.42,
                  0.4161073825503356,
                  0.4189189189189189,
                  0.4217687074829932,
                  0.4246575342465753,
                  0.42758620689655175,
                  0.4305555555555556,
                  0.43356643356643354,
                  0.43661971830985913,
                  0.4326241134751773,
                  0.4357142857142857,
                  0.43884892086330934,
                  0.4420289855072464,
                  0.44525547445255476,
                  0.4485294117647059,
                  0.45185185185185184,
                  0.4552238805970149,
                  0.45864661654135336,
                  0.4621212121212121,
                  0.4580152671755725,
                  0.46153846153846156,
                  0.46511627906976744,
                  0.46875,
                  0.47244094488188976,
                  0.47619047619047616,
                  0.48,
                  0.4838709677419355,
                  0.4878048780487805,
                  0.4918032786885246,
                  0.49586776859504134,
                  0.49166666666666664,
                  0.4957983193277311,
                  0.5,
                  0.5042735042735043,
                  0.5086206896551724,
                  0.5130434782608696,
                  0.5175438596491229,
                  0.5132743362831859,
                  0.5178571428571429,
                  0.5135135135135135,
                  0.509090909090909,
                  0.5045871559633027,
                  0.5092592592592593,
                  0.514018691588785,
                  0.5094339622641509,
                  0.5142857142857142,
                  0.5192307692307693,
                  0.5242718446601942,
                  0.5196078431372549,
                  0.5247524752475248,
                  0.53,
                  0.5353535353535354,
                  0.5408163265306123,
                  0.5463917525773195,
                  0.5520833333333334,
                  0.5473684210526316,
                  0.5425531914893617,
                  0.5483870967741935,
                  0.5604395604395604,
                  0.5555555555555556,
                  0.550561797752809,
                  0.5568181818181818,
                  0.5517241379310345,
                  0.5581395348837209,
                  0.5529411764705883,
                  0.5476190476190477,
                  0.5542168674698795,
                  0.5609756097560976,
                  0.5679012345679012,
                  0.575,
                  0.569620253164557,
                  0.5641025641025641,
                  0.5584415584415584,
                  0.5526315789473685,
                  0.56,
                  0.5833333333333334,
                  0.5915492957746479,
                  0.5857142857142857,
                  0.5942028985507246,
                  0.5970149253731343,
                  0.6060606060606061,
                  0.6,
                  0.59375,
                  0.6031746031746031,
                  0.5967741935483871,
                  0.5901639344262295,
                  0.6,
                  0.5932203389830508,
                  0.603448275862069,
                  0.5964912280701754,
                  0.6071428571428571,
                  0.6181818181818182,
                  0.6296296296296297,
                  0.6153846153846154,
                  0.6078431372549019,
                  0.62,
                  0.6122448979591837,
                  0.6041666666666666,
                  0.6170212765957447,
                  0.6086956521739131,
                  0.6,
                  0.6136363636363636,
                  0.627906976744186,
                  0.6190476190476191,
                  0.6097560975609756,
                  0.625,
                  0.6153846153846154,
                  0.631578947368421,
                  0.6216216216216216,
                  0.6111111111111112,
                  0.6285714285714286,
                  0.6470588235294118,
                  0.6363636363636364,
                  0.625,
                  0.6451612903225806,
                  0.6333333333333333,
                  0.6206896551724138,
                  0.6428571428571429,
                  0.6666666666666666,
                  0.6538461538461539,
                  0.64,
                  0.625,
                  0.6521739130434783,
                  0.6363636363636364,
                  0.6190476190476191,
                  0.65,
                  0.631578947368421,
                  0.6111111111111112,
                  0.6470588235294118,
                  0.625,
                  0.6666666666666666,
                  0.7142857142857143,
                  0.7692307692307693,
                  0.75,
                  0.8181818181818182,
                  0.8,
                  0.7777777777777778,
                  0.75,
                  0.7142857142857143,
                  1.0],
  'recalls': [ 1.0,
               0.9846153846153847,
               0.9846153846153847,
               0.9846153846153847,
               0.9846153846153847,
               0.9846153846153847,
               0.9846153846153847,
               0.9846153846153847,
               0.9846153846153847,
               0.9846153846153847,
               0.9692307692307692,
               0.9692307692307692,
               0.9692307692307692,
               0.9692307692307692,
               0.9692307692307692,
               0.9692307692307692,
               0.9692307692307692,
               0.9538461538461539,
               0.9538461538461539,
               0.9538461538461539,
               0.9538461538461539,
               0.9538461538461539,
               0.9538461538461539,
               0.9538461538461539,
               0.9538461538461539,
               0.9384615384615385,
               0.9384615384615385,
               0.9384615384615385,
               0.9384615384615385,
               0.9384615384615385,
               0.9384615384615385,
               0.9384615384615385,
               0.9384615384615385,
               0.9384615384615385,
               0.9384615384615385,
               0.9230769230769231,
               0.9230769230769231,
               0.9230769230769231,
               0.9230769230769231,
               0.9230769230769231,
               0.9230769230769231,
               0.9230769230769231,
               0.9230769230769231,
               0.9230769230769231,
               0.9230769230769231,
               0.9230769230769231,
               0.9076923076923077,
               0.9076923076923077,
               0.9076923076923077,
               0.9076923076923077,
               0.9076923076923077,
               0.9076923076923077,
               0.9076923076923077,
               0.8923076923076924,
               0.8923076923076924,
               0.8769230769230769,
               0.8615384615384616,
               0.8461538461538461,
               0.8461538461538461,
               0.8461538461538461,
               0.8307692307692308,
               0.8307692307692308,
               0.8307692307692308,
               0.8307692307692308,
               0.8153846153846154,
               0.8153846153846154,
               0.8153846153846154,
               0.8153846153846154,
               0.8153846153846154,
               0.8153846153846154,
               0.8153846153846154,
               0.8,
               0.7846153846153846,
               0.7846153846153846,
               0.7846153846153846,
               0.7692307692307693,
               0.7538461538461538,
               0.7538461538461538,
               0.7384615384615385,
               0.7384615384615385,
               0.7230769230769231,
               0.7076923076923077,
               0.7076923076923077,
               0.7076923076923077,
               0.7076923076923077,
               0.7076923076923077,
               0.6923076923076923,
               0.676923076923077,
               0.6615384615384615,
               0.6461538461538462,
               0.6461538461538462,
               0.6461538461538462,
               0.6461538461538462,
               0.6307692307692307,
               0.6307692307692307,
               0.6153846153846154,
               0.6153846153846154,
               0.6,
               0.5846153846153846,
               0.5846153846153846,
               0.5692307692307692,
               0.5538461538461539,
               0.5538461538461539,
               0.5384615384615384,
               0.5384615384615384,
               0.5230769230769231,
               0.5230769230769231,
               0.5230769230769231,
               0.5230769230769231,
               0.49230769230769234,
               0.47692307692307695,
               0.47692307692307695,
               0.46153846153846156,
               0.4461538461538462,
               0.4461538461538462,
               0.4307692307692308,
               0.4153846153846154,
               0.4153846153846154,
               0.4153846153846154,
               0.4,
               0.38461538461538464,
               0.38461538461538464,
               0.36923076923076925,
               0.36923076923076925,
               0.35384615384615387,
               0.3384615384615385,
               0.3384615384615385,
               0.3384615384615385,
               0.3230769230769231,
               0.3076923076923077,
               0.3076923076923077,
               0.2923076923076923,
               0.27692307692307694,
               0.27692307692307694,
               0.27692307692307694,
               0.26153846153846155,
               0.24615384615384617,
               0.23076923076923078,
               0.23076923076923078,
               0.2153846153846154,
               0.2,
               0.2,
               0.18461538461538463,
               0.16923076923076924,
               0.16923076923076924,
               0.15384615384615385,
               0.15384615384615385,
               0.15384615384615385,
               0.15384615384615385,
               0.13846153846153847,
               0.13846153846153847,
               0.12307692307692308,
               0.1076923076923077,
               0.09230769230769231,
               0.07692307692307693,
               0.0]}
roc_auc_macro: 0.7635451505016722
roc_auc_micro: 0.7635451505016722
Saved to: results_0
```
</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```!ls results/experiment_run_0/model/


```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
checkpoint			   model_weights_progress.data-00000-of-00001
log				   model_weights_progress.index
model_hyperparameters.json	   model_weights_progress.meta
model_weights.data-00000-of-00001  training_progress.p
model_weights.index		   train_set_metadata.json
model_weights.meta
```
</div>
</div>
</div>

