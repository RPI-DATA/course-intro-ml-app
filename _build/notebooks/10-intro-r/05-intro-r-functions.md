---
interact_link: content/notebooks/10-intro-r/05-intro-r-functions.ipynb
kernel_name: ir
has_widgets: false
title: 'Functions'
prev_page:
  url: /notebooks/10-intro-r/04-intro-r-dataframes.html
  title: 'Dataframes'
next_page:
  url: /notebooks/10-intro-r/06-intro-r-conditionals-loops.html
  title: 'Conditional-Loops'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---


[![AnalyticsDojo](https://github.com/rpi-techfundamentals/spring2019-materials/blob/master/fig/final-logo.png?raw=1)](http://rpi.analyticsdojo.com)
<center><h1>Introduction to R - Functions </h1></center>
<center><h3><a href = 'http://rpi.analyticsdojo.com'>rpi.analyticsdojo.com</a></h3></center>



## Overview
- Why functions?
- Predefined functions
- Custom functions
- Exercises



## R is a Functional Language

- Operations are carried out with functions. Functions take objects as inputs and return objects as outputs. 
- An analysis can be considered a pipeline of function calls, with output from a function used later in a subsequent operation as input to another function.
- Functions themselves are objects. 



## Why Functions?
- Code reuse. 
- Abstract away complexity. 
- Simple, efficient robust code.
- Specific functional programming languages like Lisp & Haskell built around *functional programming*, which enforces great practices.
- Read more about functional programming in Python [here](http://www.ibm.com/developerworks/library/l-prog/).



## Predefined Functions
- We have used predefined functions in earlier exercises
- R has predefined functions for embedded data structures like vectors and data frames.
- See [here](http://www.statmethods.net/management/functions.html) for a list of R functions.




<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
#Simple Rounding Functions
a<-3.14
a<-round(a)
a

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
3
</div>

</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
#Function to create 5 random numbers and assign to vector. 
random <- rnorm(5)
random

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<ol class=list-inline>
	<li>-0.828267411092106</li>
	<li>-0.0880055027208348</li>
	<li>-1.02846584171599</li>
	<li>-0.617251670040361</li>
	<li>-0.106607913024122</li>
</ol>

</div>

</div>
</div>
</div>



## Using Functions

- Functions generally take arguments, some of which are often optional
- To get information about a function you know exists, use `help` or `?`, e.g., `?lm`. For information on a general topic, use `apropos` or `??`



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
#Functions generally take arguments, some of which are often optional:
random <- rnorm(5)
median(random)
median(random, na.rm = TRUE)
help(lm)
?lm

?log


```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
-0.117759078362163
</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
-0.117759078362163
</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">

<table width="100%" summary="page for lm {stats}"><tr><td>lm {stats}</td><td style="text-align: right;">R Documentation</td></tr></table>

<h2>Fitting Linear Models</h2>

<h3>Description</h3>

<p><code>lm</code> is used to fit linear models.
It can be used to carry out regression,
single stratum analysis of variance and
analysis of covariance (although <code>aov</code> may provide a more
convenient interface for these).
</p>


<h3>Usage</h3>

<pre>
lm(formula, data, subset, weights, na.action,
   method = "qr", model = TRUE, x = FALSE, y = FALSE, qr = TRUE,
   singular.ok = TRUE, contrasts = NULL, offset, ...)
</pre>


<h3>Arguments</h3>

<table summary="R argblock">
<tr valign="top"><td><code>formula</code></td>
<td>
<p>an object of class <code>"formula"</code> (or one that
can be coerced to that class): a symbolic description of the
model to be fitted.  The details of model specification are given
under &lsquo;Details&rsquo;.</p>
</td></tr>
<tr valign="top"><td><code>data</code></td>
<td>
<p>an optional data frame, list or environment (or object
coercible by <code>as.data.frame</code> to a data frame) containing
the variables in the model.  If not found in <code>data</code>, the
variables are taken from <code>environment(formula)</code>,
typically the environment from which <code>lm</code> is called.</p>
</td></tr>
<tr valign="top"><td><code>subset</code></td>
<td>
<p>an optional vector specifying a subset of observations
to be used in the fitting process.</p>
</td></tr>
<tr valign="top"><td><code>weights</code></td>
<td>
<p>an optional vector of weights to be used in the fitting
process.  Should be <code>NULL</code> or a numeric vector.
If non-NULL, weighted least squares is used with weights
<code>weights</code> (that is, minimizing <code>sum(w*e^2)</code>); otherwise
ordinary least squares is used.  See also &lsquo;Details&rsquo;,</p>
</td></tr>
<tr valign="top"><td><code>na.action</code></td>
<td>
<p>a function which indicates what should happen
when the data contain <code>NA</code>s.  The default is set by
the <code>na.action</code> setting of <code>options</code>, and is
<code>na.fail</code> if that is unset.  The &lsquo;factory-fresh&rsquo;
default is <code>na.omit</code>.  Another possible value is
<code>NULL</code>, no action.  Value <code>na.exclude</code> can be useful.</p>
</td></tr>
<tr valign="top"><td><code>method</code></td>
<td>
<p>the method to be used; for fitting, currently only
<code>method = "qr"</code> is supported; <code>method = "model.frame"</code> returns
the model frame (the same as with <code>model = TRUE</code>, see below).</p>
</td></tr>
<tr valign="top"><td><code>model, x, y, qr</code></td>
<td>
<p>logicals.  If <code>TRUE</code> the corresponding
components of the fit (the model frame, the model matrix, the
response, the QR decomposition) are returned.
</p>
</td></tr>
<tr valign="top"><td><code>singular.ok</code></td>
<td>
<p>logical. If <code>FALSE</code> (the default in S but
not in <span style="font-family: Courier New, Courier; color: #666666;"><b>R</b></span>) a singular fit is an error.</p>
</td></tr>
<tr valign="top"><td><code>contrasts</code></td>
<td>
<p>an optional list. See the <code>contrasts.arg</code>
of <code>model.matrix.default</code>.</p>
</td></tr>
<tr valign="top"><td><code>offset</code></td>
<td>
<p>this can be used to specify an <em>a priori</em> known
component to be included in the linear predictor during fitting.
This should be <code>NULL</code> or a numeric vector of length equal to
the number of cases.  One or more <code>offset</code> terms can be
included in the formula instead or as well, and if more than one are
specified their sum is used.  See <code>model.offset</code>.</p>
</td></tr>
<tr valign="top"><td><code>...</code></td>
<td>
<p>additional arguments to be passed to the low level
regression fitting functions (see below).</p>
</td></tr>
</table>


<h3>Details</h3>

<p>Models for <code>lm</code> are specified symbolically.  A typical model has
the form <code>response ~ terms</code> where <code>response</code> is the (numeric)
response vector and <code>terms</code> is a series of terms which specifies a
linear predictor for <code>response</code>.  A terms specification of the form
<code>first + second</code> indicates all the terms in <code>first</code> together
with all the terms in <code>second</code> with duplicates removed.  A
specification of the form <code>first:second</code> indicates the set of
terms obtained by taking the interactions of all terms in <code>first</code>
with all terms in <code>second</code>.  The specification <code>first*second</code>
indicates the <em>cross</em> of <code>first</code> and <code>second</code>.  This is
the same as <code>first + second + first:second</code>.
</p>
<p>If the formula includes an <code>offset</code>, this is evaluated and
subtracted from the response.
</p>
<p>If <code>response</code> is a matrix a linear model is fitted separately by
least-squares to each column of the matrix.
</p>
<p>See <code>model.matrix</code> for some further details.  The terms in
the formula will be re-ordered so that main effects come first,
followed by the interactions, all second-order, all third-order and so
on: to avoid this pass a <code>terms</code> object as the formula (see
<code>aov</code> and <code>demo(glm.vr)</code> for an example).
</p>
<p>A formula has an implied intercept term.  To remove this use either
<code>y ~ x - 1</code> or <code>y ~ 0 + x</code>.  See <code>formula</code> for
more details of allowed formulae.
</p>
<p>Non-<code>NULL</code> <code>weights</code> can be used to indicate that different
observations have different variances (with the values in
<code>weights</code> being inversely proportional to the variances); or
equivalently, when the elements of <code>weights</code> are positive
integers <i>w_i</i>, that each response <i>y_i</i> is the mean of
<i>w_i</i> unit-weight observations (including the case that there are
<i>w_i</i> observations equal to <i>y_i</i> and the data have been
summarized).
</p>
<p><code>lm</code> calls the lower level functions <code>lm.fit</code>, etc,
see below, for the actual numerical computations.  For programming
only, you may consider doing likewise.
</p>
<p>All of <code>weights</code>, <code>subset</code> and <code>offset</code> are evaluated
in the same way as variables in <code>formula</code>, that is first in
<code>data</code> and then in the environment of <code>formula</code>.
</p>


<h3>Value</h3>

<p><code>lm</code> returns an object of <code>class</code> <code>"lm"</code> or for
multiple responses of class <code>c("mlm", "lm")</code>.
</p>
<p>The functions <code>summary</code> and <code>anova</code> are used to
obtain and print a summary and analysis of variance table of the
results.  The generic accessor functions <code>coefficients</code>,
<code>effects</code>, <code>fitted.values</code> and <code>residuals</code> extract
various useful features of the value returned by <code>lm</code>.
</p>
<p>An object of class <code>"lm"</code> is a list containing at least the
following components:
</p>
<table summary="R valueblock">
<tr valign="top"><td><code>coefficients</code></td>
<td>
<p>a named vector of coefficients</p>
</td></tr>
<tr valign="top"><td><code>residuals</code></td>
<td>
<p>the residuals, that is response minus fitted values.</p>
</td></tr>
<tr valign="top"><td><code>fitted.values</code></td>
<td>
<p>the fitted mean values.</p>
</td></tr>
<tr valign="top"><td><code>rank</code></td>
<td>
<p>the numeric rank of the fitted linear model.</p>
</td></tr>
<tr valign="top"><td><code>weights</code></td>
<td>
<p>(only for weighted fits) the specified weights.</p>
</td></tr>
<tr valign="top"><td><code>df.residual</code></td>
<td>
<p>the residual degrees of freedom.</p>
</td></tr>
<tr valign="top"><td><code>call</code></td>
<td>
<p>the matched call.</p>
</td></tr>
<tr valign="top"><td><code>terms</code></td>
<td>
<p>the <code>terms</code> object used.</p>
</td></tr>
<tr valign="top"><td><code>contrasts</code></td>
<td>
<p>(only where relevant) the contrasts used.</p>
</td></tr>
<tr valign="top"><td><code>xlevels</code></td>
<td>
<p>(only where relevant) a record of the levels of the
factors used in fitting.</p>
</td></tr>
<tr valign="top"><td><code>offset</code></td>
<td>
<p>the offset used (missing if none were used).</p>
</td></tr>
<tr valign="top"><td><code>y</code></td>
<td>
<p>if requested, the response used.</p>
</td></tr>
<tr valign="top"><td><code>x</code></td>
<td>
<p>if requested, the model matrix used.</p>
</td></tr>
<tr valign="top"><td><code>model</code></td>
<td>
<p>if requested (the default), the model frame used.</p>
</td></tr>
<tr valign="top"><td><code>na.action</code></td>
<td>
<p>(where relevant) information returned by
<code>model.frame</code> on the special handling of <code>NA</code>s.</p>
</td></tr>
</table>
<p>In addition, non-null fits will have components <code>assign</code>,
<code>effects</code> and (unless not requested) <code>qr</code> relating to the linear
fit, for use by extractor functions such as <code>summary</code> and
<code>effects</code>.
</p>


<h3>Using time series</h3>

<p>Considerable care is needed when using <code>lm</code> with time series.
</p>
<p>Unless <code>na.action = NULL</code>, the time series attributes are
stripped from the variables before the regression is done.  (This is
necessary as omitting <code>NA</code>s would invalidate the time series
attributes, and if <code>NA</code>s are omitted in the middle of the series
the result would no longer be a regular time series.)
</p>
<p>Even if the time series attributes are retained, they are not used to
line up series, so that the time shift of a lagged or differenced
regressor would be ignored.  It is good practice to prepare a
<code>data</code> argument by <code>ts.intersect(..., dframe = TRUE)</code>,
then apply a suitable <code>na.action</code> to that data frame and call
<code>lm</code> with <code>na.action = NULL</code> so that residuals and fitted
values are time series.
</p>


<h3>Note</h3>

<p>Offsets specified by <code>offset</code> will not be included in predictions
by <code>predict.lm</code>, whereas those specified by an offset term
in the formula will be.
</p>


<h3>Author(s)</h3>

<p>The design was inspired by the S function of the same name described
in Chambers (1992).  The implementation of model formula by Ross Ihaka
was based on Wilkinson &amp; Rogers (1973).
</p>


<h3>References</h3>

<p>Chambers, J. M. (1992)
<em>Linear models.</em>
Chapter 4 of <em>Statistical Models in S</em>
eds J. M. Chambers and T. J. Hastie, Wadsworth &amp; Brooks/Cole.
</p>
<p>Wilkinson, G. N. and Rogers, C. E. (1973)
Symbolic descriptions of factorial models for analysis of variance.
<em>Applied Statistics</em>, <b>22</b>, 392&ndash;9.
</p>


<h3>See Also</h3>

<p><code>summary.lm</code> for summaries and <code>anova.lm</code> for
the ANOVA table; <code>aov</code> for a different interface.
</p>
<p>The generic functions <code>coef</code>, <code>effects</code>,
<code>residuals</code>, <code>fitted</code>, <code>vcov</code>.
</p>
<p><code>predict.lm</code> (via <code>predict</code>) for prediction,
including confidence and prediction intervals;
<code>confint</code> for confidence intervals of <em>parameters</em>.
</p>
<p><code>lm.influence</code> for regression diagnostics, and
<code>glm</code> for <b>generalized</b> linear models.
</p>
<p>The underlying low level functions,
<code>lm.fit</code> for plain, and <code>lm.wfit</code> for weighted
regression fitting.
</p>
<p>More <code>lm()</code> examples are available e.g., in
<code>anscombe</code>, <code>attitude</code>, <code>freeny</code>,
<code>LifeCycleSavings</code>, <code>longley</code>,
<code>stackloss</code>, <code>swiss</code>.
</p>
<p><code>biglm</code> in package <a href="https://CRAN.R-project.org/package=biglm"><span class="pkg">biglm</span></a> for an alternative
way to fit linear models to large datasets (especially those with many
cases).
</p>


<h3>Examples</h3>

<pre>
require(graphics)

## Annette Dobson (1990) "An Introduction to Generalized Linear Models".
## Page 9: Plant Weight Data.
ctl &lt;- c(4.17,5.58,5.18,6.11,4.50,4.61,5.17,4.53,5.33,5.14)
trt &lt;- c(4.81,4.17,4.41,3.59,5.87,3.83,6.03,4.89,4.32,4.69)
group &lt;- gl(2, 10, 20, labels = c("Ctl","Trt"))
weight &lt;- c(ctl, trt)
lm.D9 &lt;- lm(weight ~ group)
lm.D90 &lt;- lm(weight ~ group - 1) # omitting intercept

anova(lm.D9)
summary(lm.D90)

opar &lt;- par(mfrow = c(2,2), oma = c(0, 0, 1.1, 0))
plot(lm.D9, las = 1)      # Residuals, Fitted, ...
par(opar)

### less simple examples in "See Also" above
</pre>

<hr /><div style="text-align: center;">[Package <em>stats</em> version 3.3.1 ]</div>
</div>

</div>
</div>
</div>



## Custom Functions in R
- The fuction in R doesn't demend on white space, but it functions just like it did in Python. 
- It returns a value using the `return` command, which should be the last command in the function. 





<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
#This defines a function called "addTwo."

a = 1000000
addTwo <- function(a, b){
c<-a+b
return(c)
}

d<-addTwo(5,10)
d


```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
15
</div>

</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
#we can use the short form that pu
square<-function(x) x^2
square(1:5)
addtwo<-function(a,b) a+b
addtwo(5,9)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<ol class=list-inline>
	<li>1</li>
	<li>4</li>
	<li>9</li>
	<li>16</li>
	<li>25</li>
</ol>

</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
14
</div>

</div>
</div>
</div>



## Functional Programming

- Functions are, like everything else in R, object.
- Functions can passed around just like any other value.
- That means we can do really cool things, like pass a function to a function. 







<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
print(addTwo)
class(addTwo)



```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
function(a, b){
c<-a+b
return(c)
}
```
</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
'function'
</div>

</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
#Functions are objects and can be assigned another value.
addTwob<-addTwo
print(addTwob)
str(addTwob)
d<-addTwob(5,10)
d



```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_traceback_line}
```

    Error in eval(expr, envir, enclos): object 'addTwo' not found
    Traceback:



```
</div>
</div>
</div>









Copyright [AnalyticsDojo](http://rpi.analyticsdojo.com) 2016.
This work is licensed under the [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/) license agreement.
Adopted from [Berkley R Bootcamp](https://github.com/berkeley-scf/r-bootcamp-2016). 

