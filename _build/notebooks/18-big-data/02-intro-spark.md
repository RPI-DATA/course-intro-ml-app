---
interact_link: content/notebooks/18-big-data/02-intro-spark.ipynb
kernel_name: python3
has_widgets: false
title: 'Introduction to Spark'
prev_page:
  url: /notebooks/18-big-data/01-intro-mapreduce.html
  title: 'Intoduction to MapReduce'
next_page:
  url: /notebooks/18-intro-timeseries/01-time-series.html
  title: 'Introduction to Time Series'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---


[![AnalyticsDojo](https://github.com/rpi-techfundamentals/spring2019-materials/blob/master/fig/final-logo.png?raw=1)](http://rpi.analyticsdojo.com)
<center><h1>Introduction to Spark</h1></center>
<center><h3><a href = 'http://rpi.analyticsdojo.com'>rpi.analyticsdojo.com</a></h3></center>



Adopted from work by Steve Phelps:
https://github.com/phelps-sg/python-bigdata 
This work is licensed under the Creative Commons Attribution 4.0 International license agreement.




### Reference
- [Spark Documentation](http://spark.apache.org/docs/latest/)
- [Spark Programming Guide](http://spark.apache.org/docs/latest/programming-guide.html)
- [DataBricks Login](https://community.cloud.databricks.com)
- [Pyspark](https://github.com/jupyter/docker-stacks)
- [Conda](

```conda install -c anaconda-cluster/label/dev spark
   conda install -c conda-forge pyspark
```




### Overview
- History
- Data Structures
- Using Apache Spark with Python




## History

- Apache Spark was first released in 2014. 

- It was originally developed by [Matei Zaharia](http://people.csail.mit.edu/matei) as a class project, and later a PhD dissertation, at University of California, Berkeley.

- In contrast to Hadoop, Apache Spark:

    - is easy to install and configure.
    - provides a much more natural *iterative* workflow 




## Resilient Distributed Datasets (RDD)

- The fundamental abstraction of Apache Spark is a read-only, parallel, distributed, fault-tolerent collection called a resilient distributed datasets (RDD).

- When working with Apache Spark we iteratively apply functions to every elelement of these collections in parallel to produce *new* RDDs.

- For the most part, you can think/use RDDs like distributed dataframes. 




## Resilient Distributed Datasets (RDD)

- Properties resilient distributed datasets (RDDs):
    - The data is distributed across nodes in a cluster of computers.
    - No data is lost if a single node fails.
    - Data is typically stored in HBase tables, or HDFS files.
    - The `map` and `reduce` functions can work in *parallel* across
       different keys, or different elements of the collection.

- The underlying framework (e.g. Hadoop or Apache Spark) allocates data and processing to different nodes, without any intervention from the programmer.



## Word Count Example

- In this simple example, the input is a set of URLs, each record is a document. <br> <br> <br>

- **Problem: Compute how many times each word has occurred across data set.**



## Word Count: Map


The input to $\operatorname{map}$ is a mapping:
- Key: URL
- Value: Contents of document <br>
$\left< document1, to \; be \; or \; not \; to \; be \right>$  
    

- In this example, our $\operatorname{map}$ function will process a given URL, and produces a mapping:
- So our original data-set will be transformed to:
  
  $\left< to, 1 \right>$
  $\left< be, 1 \right>$
  $\left< or, 1 \right>$
  $\left< not, 1 \right>$
  $\left< to, 1 \right>$
  $\left< be, 1 \right>$



## Word Count: Reduce


- The reduce operation groups values according to their key, and then performs areduce on each key.

- The collections are partitioned across different storage units, therefore.

- Map-Reduce will fold the data in such a way that it minimises data-copying across the cluster.

- Data in different partitions are reduced separately in parallel.

- The final result is a reduce of the reduced data in each partition.

- Therefore it is very important that our operator *is both commutative and associative*.

- In our case the function is the `+` operator

  $\left< be, 2 \right>$  
  $\left< not, 1 \right>$  
  $\left< or, 1 \right>$  
  $\left< to, 2 \right>$  
  



## Map-Reduce on a Cluster of Computers

- The code we have written so far will *not* allow us to exploit parallelism from multiple computers in a [cluster](https://en.wikipedia.org/wiki/Computer_cluster).

- Developing such a framework would be a very large software engineering project.

- There are existing frameworks we can use:
    - [Apache Hadoop](https://hadoop.apache.org/)
    - [Apache Spark](https://spark.apache.org/)
    
- This notebook covers Apache Spark.



## Apache Spark

- Apache Spark provides an object-oriented library for processing data on the cluster.

- It provides objects which represent resilient distributed datasets (RDDs).

- RDDs behave a bit like Python collections (e.g. lists).

- However:
    - the underlying data is distributed across the nodes in the cluster, and
    - the collections are *immutable*.



## Apache Spark and Map-Reduce

- We process the data by using higher-order functions to map RDDs onto *new* RDDs. 

- Each instance of an RDD has at least two *methods* corresponding to the Map-Reduce workflow:
    - `map`
    - `reduceByKey`
    
- These methods work in the same way as the corresponding functions we defined earlier to work with the standard Python collections.  

- There are also additional RDD methods in the Apache Spark API including ones for SQL.
   



## Word-count in Apache Spark





<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
words = "to be or not to be".split()
words

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
['to', 'be', 'or', 'not', 'to', 'be']
```


</div>
</div>
</div>



### The `SparkContext` class

- When working with Apache Spark we invoke methods on an object which is an instance of the `pyspark.context.SparkContext` context.

- Typically, (such as when running on DataBricks) an instance of this object will be created automatically for you and assigned to the variable `sc`.

- The `parallelize` method in `SparkContext` can be used to turn any ordinary Python collection into an RDD; 



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
#Don't Execute this on Databricks
#To be used if executing via docker
import pyspark
#sc = pyspark.SparkContext('local[*]')

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
words_rdd = sc.parallelize(words)
words_rdd

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
ParallelCollectionRDD[0] at parallelize at PythonRDD.scala:175
```


</div>
</div>
</div>



### Mapping an RDD

- Now when we invoke the `map` or `reduceByKey` methods on `my_rdd` we can set up a parallel processing computation across the cluster.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
word_tuples_rdd = words_rdd.map(lambda x: (x, 1))
word_tuples_rdd

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
PythonRDD[1] at RDD at PythonRDD.scala:48
```


</div>
</div>
</div>



### Collecting the RDD
- Notice that we do not have a result yet.

- The computation is not performed until we request the final result to be *collected*.

- We do this by invoking the `collect()` method.

- Be careful with the `collect` method, as all data you are collecting must fit in memory.  

- The `take` method is similar to `collect`, but only returns the first $n$ elements.
 



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
word_tuples_rdd.collect()

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
[('to', 1), ('be', 1), ('or', 1), ('not', 1), ('to', 1), ('be', 1)]
```


</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
word_tuples_rdd.take(4)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
[('to', 1), ('be', 1), ('or', 1), ('not', 1)]
```


</div>
</div>
</div>



### Reducing an RDD

- However, we require additional processing to reduce the data using the word key. 



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
word_counts_rdd = word_tuples_rdd.reduceByKey(lambda x, y: x + y)
word_counts_rdd

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
PythonRDD[8] at RDD at PythonRDD.scala:48
```


</div>
</div>
</div>



- Now we request the final result:



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
word_counts = word_counts_rdd.collect()
word_counts

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
[('to', 2), ('be', 2), ('or', 1), ('not', 1)]
```


</div>
</div>
</div>



### Lazy evaluation

- It is only when we invoke `collect()` that the processing is performed on the cluster.

- Invoking `collect()` will cause both the `map` and `reduceByKey` operations to be performed.

- If the resulting collection is very large then this can be an expensive operation.




<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
word_counts_rdd.take(2)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
[('to', 2), ('be', 2)]
```


</div>
</div>
</div>



### Connecting MapReduce in Single Command
- Can string together `map` and `reduce` commands.
- Not executed until it is collected.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
text = "to be or not to be".split()
rdd = sc.parallelize(text)
counts = rdd.map(lambda word: (word, 1)).reduceByKey(lambda x, y: x + y)
counts.collect()

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
[('to', 2), ('be', 2), ('or', 1), ('not', 1)]
```


</div>
</div>
</div>



## Additional RDD transformations

- Apache Spark offers many more methods for operating on collections of tuples over and above the standard Map-Reduce framework:

    - Sorting: `sortByKey`, `sortBy`, `takeOrdered`
    - Mapping: `flatMap`
    - Filtering: `filter`
    - Counting: `count`
    - Set-theoretic: `intersection`, `union`
    - Many others: [see the Transformations section of the programming guide](https://spark.apache.org/docs/latest/programming-guide.html#transformations)
    



## Creating an RDD from a text file

- In the previous example, we created an RDD from a Python collection.

- This is *not* typically how we would work with big data.

- More commonly we would create an RDD corresponding to data in an
HBase table, or an HDFS file.

- The following example creates an RDD from a text file on the native filesystem (ext4);
    - With bigger data, you would use an HDFS file, but the principle is the same.

- Each element of the RDD corresponds to a single *line* of text.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
genome = sc.textFile('../input/iris.csv')

```
</div>

</div>



## Calculating $\pi$ using Spark

- We can estimate an approximate value for $\pi$ using the following Monte-Carlo method:


1.    Inscribe a circle in a square
2.    Randomly generate points in the square
3.    Determine the number of points in the square that are also in the circle
4.    Let $r$ be the number of points in the circle divided by the number of points in the square, then $\pi \approx 4 r$.
    
- Note that the more points generated, the better the approximation

See [this tutorial](https://computing.llnl.gov/tutorials/parallel_comp/#ExamplesPI).



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
import numpy as np

def sample(p):
    #here x,y are the x,y coordinate
    x, y = np.random.random(), np.random.random()
    #Because the circle is of 
    return 1 if x*x + y*y < 1 else 0

NUM_SAMPLES = 1000000

count = sc.parallelize(range(0, NUM_SAMPLES)).map(sample) \
             .reduce(lambda a, b: a + b)
#Area  = 4*PI*r
r = float(count) / float(NUM_SAMPLES)
r
print ("Pi is approximately %f" % (4.0 * r))

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
Pi is approximately 3.141616
```
</div>
</div>
</div>

