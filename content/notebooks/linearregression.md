---
title: "Linear Regression"
weight: 2
---



<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p><a href="https://colab.research.google.com/github/RPI-DATA/tutorials-intro/blob/master/website/linearregression.ipynb" target="_blank"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"> </a></p>
<p><img src="https://raw.githubusercontent.com/RPI-DATA/website/master/static/images/rpilogo.png" alt="RPI LOGO"></p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>This project will work on how to predict the prices of homes based on the properties of the house. I will determine which house affected the final sale price and how effectively we can predict the sale price.Here's a brief description of the columns in the data:</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="Learning-Objectives">Learning Objectives<a class="anchor-link" href="#Learning-Objectives"></a></h2><p>By the end of this notebook, the reader should be able to perform Linear Regression techniques in python. This includes:</p>
<ol>
<li>Importing and formating data</li>
<li>Training the LinearRegression model from the <code>sklearn.linear_model</code> library</li>
<li>Work with qualitative and quantitative data, and effectively deal with instances of categorical data.</li>
<li>Analyze and determine proper handling of redundant and/or inconsistent data features.</li>
<li>Create a heatmap visual with <code>matplot.lib</code> library</li>
</ol>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="Read-Data">Read Data<a class="anchor-link" href="#Read-Data"></a></h2><p>The <code>pandas</code> library is an open source data analytics tool for python that allows the use of 'data frame' objects and clean file parsing.</p>
<p>Here we split the Ames housing data into training and testing data. The dataset contains 82 columns which are known as features of the data. Here are a few:</p>
<ul>
<li>Lot Area: Lot size in square feet.</li>
<li>Overall Qual: Rates the overall material and finish of the house.</li>
<li>Overall Cond: Rates the overall condition of the house.</li>
<li>Year Built: Original construction date.</li>
<li>Low Qual Fin SF: Low quality finished square feet (all floors).</li>
<li>Full Bath: Full bathrooms above grade.</li>
<li>Fireplaces: Number of fireplaces.</li>
</ul>
<p>and so on.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[1]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">import</span> <span class="nn">pandas</span> <span class="k">as</span> <span class="nn">pd</span>
<span class="n">data</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">read_csv</span><span class="p">(</span><span class="s2">&quot;Data/AmesHousing.txt&quot;</span><span class="p">,</span> <span class="n">delimiter</span> <span class="o">=</span> <span class="s1">&#39;</span><span class="se">\t</span><span class="s1">&#39;</span><span class="p">)</span>
<span class="n">train</span> <span class="o">=</span> <span class="n">data</span><span class="p">[</span><span class="mi">0</span><span class="p">:</span><span class="mi">1460</span><span class="p">]</span>
<span class="n">test</span> <span class="o">=</span> <span class="n">data</span><span class="p">[</span><span class="mi">1460</span><span class="p">:]</span>
<span class="n">target</span> <span class="o">=</span> <span class="s1">&#39;SalePrice&#39;</span>
<span class="nb">print</span><span class="p">(</span><span class="n">train</span><span class="o">.</span><span class="n">info</span><span class="p">())</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>&lt;class &#39;pandas.core.frame.DataFrame&#39;&gt;
RangeIndex: 1460 entries, 0 to 1459
Data columns (total 82 columns):
Order              1460 non-null int64
PID                1460 non-null int64
MS SubClass        1460 non-null int64
MS Zoning          1460 non-null object
Lot Frontage       1211 non-null float64
Lot Area           1460 non-null int64
Street             1460 non-null object
Alley              109 non-null object
Lot Shape          1460 non-null object
Land Contour       1460 non-null object
Utilities          1460 non-null object
Lot Config         1460 non-null object
Land Slope         1460 non-null object
Neighborhood       1460 non-null object
Condition 1        1460 non-null object
Condition 2        1460 non-null object
Bldg Type          1460 non-null object
House Style        1460 non-null object
Overall Qual       1460 non-null int64
Overall Cond       1460 non-null int64
Year Built         1460 non-null int64
Year Remod/Add     1460 non-null int64
Roof Style         1460 non-null object
Roof Matl          1460 non-null object
Exterior 1st       1460 non-null object
Exterior 2nd       1460 non-null object
Mas Vnr Type       1449 non-null object
Mas Vnr Area       1449 non-null float64
Exter Qual         1460 non-null object
Exter Cond         1460 non-null object
Foundation         1460 non-null object
Bsmt Qual          1420 non-null object
Bsmt Cond          1420 non-null object
Bsmt Exposure      1419 non-null object
BsmtFin Type 1     1420 non-null object
BsmtFin SF 1       1459 non-null float64
BsmtFin Type 2     1419 non-null object
BsmtFin SF 2       1459 non-null float64
Bsmt Unf SF        1459 non-null float64
Total Bsmt SF      1459 non-null float64
Heating            1460 non-null object
Heating QC         1460 non-null object
Central Air        1460 non-null object
Electrical         1460 non-null object
1st Flr SF         1460 non-null int64
2nd Flr SF         1460 non-null int64
Low Qual Fin SF    1460 non-null int64
Gr Liv Area        1460 non-null int64
Bsmt Full Bath     1459 non-null float64
Bsmt Half Bath     1459 non-null float64
Full Bath          1460 non-null int64
Half Bath          1460 non-null int64
Bedroom AbvGr      1460 non-null int64
Kitchen AbvGr      1460 non-null int64
Kitchen Qual       1460 non-null object
TotRms AbvGrd      1460 non-null int64
Functional         1460 non-null object
Fireplaces         1460 non-null int64
Fireplace Qu       743 non-null object
Garage Type        1386 non-null object
Garage Yr Blt      1385 non-null float64
Garage Finish      1385 non-null object
Garage Cars        1460 non-null float64
Garage Area        1460 non-null float64
Garage Qual        1385 non-null object
Garage Cond        1385 non-null object
Paved Drive        1460 non-null object
Wood Deck SF       1460 non-null int64
Open Porch SF      1460 non-null int64
Enclosed Porch     1460 non-null int64
3Ssn Porch         1460 non-null int64
Screen Porch       1460 non-null int64
Pool Area          1460 non-null int64
Pool QC            1 non-null object
Fence              297 non-null object
Misc Feature       60 non-null object
Misc Val           1460 non-null int64
Mo Sold            1460 non-null int64
Yr Sold            1460 non-null int64
Sale Type          1460 non-null object
Sale Condition     1460 non-null object
SalePrice          1460 non-null int64
dtypes: float64(11), int64(28), object(43)
memory usage: 935.4+ KB
None
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>The train data over here will be used to create the linear regression model, while the test data will be used to figure out the accuracy of the linear regression model.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="Use-linear-regression-to-model-the-data">Use linear regression to model the data<a class="anchor-link" href="#Use-linear-regression-to-model-the-data"></a></h2><hr>
<p>In this case, we will use the <strong>simple linear regression</strong> to evaluate the relationship between 2 variable--living area("Gr Liv Area") and price("SalePrice"). <code>linearRegression.fit()</code> is a pretty convinent function that it can turn the input data into the linear function and you dont need to worry about the calculation. We can also use the <code>mean_squared_error</code> to get the total cariance of the linear function.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[2]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">import</span> <span class="nn">numpy</span> <span class="k">as</span> <span class="nn">np</span>
<span class="kn">from</span> <span class="nn">sklearn.linear_model</span> <span class="k">import</span> <span class="n">LinearRegression</span>

<span class="n">lr</span> <span class="o">=</span> <span class="n">LinearRegression</span><span class="p">()</span>
<span class="n">lr</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">train</span><span class="p">[[</span><span class="s1">&#39;Gr Liv Area&#39;</span><span class="p">]],</span> <span class="n">train</span><span class="p">[</span><span class="s1">&#39;SalePrice&#39;</span><span class="p">])</span>
<span class="kn">from</span> <span class="nn">sklearn.metrics</span> <span class="k">import</span> <span class="n">mean_squared_error</span>
<span class="n">train_predictions</span> <span class="o">=</span> <span class="n">lr</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">train</span><span class="p">[[</span><span class="s1">&#39;Gr Liv Area&#39;</span><span class="p">]])</span>
<span class="n">test_predictions</span> <span class="o">=</span> <span class="n">lr</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">test</span><span class="p">[[</span><span class="s1">&#39;Gr Liv Area&#39;</span><span class="p">]])</span>

<span class="n">train_mse</span> <span class="o">=</span> <span class="n">mean_squared_error</span><span class="p">(</span><span class="n">train_predictions</span><span class="p">,</span> <span class="n">train</span><span class="p">[</span><span class="s1">&#39;SalePrice&#39;</span><span class="p">])</span>
<span class="n">test_mse</span> <span class="o">=</span> <span class="n">mean_squared_error</span><span class="p">(</span><span class="n">test_predictions</span><span class="p">,</span> <span class="n">test</span><span class="p">[</span><span class="s1">&#39;SalePrice&#39;</span><span class="p">])</span>

<span class="n">train_rmse</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">sqrt</span><span class="p">(</span><span class="n">train_mse</span><span class="p">)</span>
<span class="n">test_rmse</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">sqrt</span><span class="p">(</span><span class="n">test_mse</span><span class="p">)</span>

<span class="nb">print</span><span class="p">(</span><span class="n">lr</span><span class="o">.</span><span class="n">coef_</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">train_rmse</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">test_rmse</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>[116.86624683]
56034.362001412796
57088.25161263909
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>In this case, we use the <code>lr.coef_()</code> to get the coefficient of the linear function which is 116.87. More than that, the standard error for the train data is 56034 and test data is 57088. Now, let's make the result more visible by plotting.</p>
<p>Following is the linear regression line made from data in "train".</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[4]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">import</span> <span class="nn">matplotlib.pyplot</span> <span class="k">as</span> <span class="nn">plt</span>
<span class="n">plt</span><span class="o">.</span><span class="n">scatter</span><span class="p">(</span><span class="n">train</span><span class="p">[[</span><span class="s1">&#39;Gr Liv Area&#39;</span><span class="p">]],</span> <span class="n">train</span><span class="p">[[</span><span class="s1">&#39;SalePrice&#39;</span><span class="p">]],</span>  <span class="n">color</span><span class="o">=</span><span class="s1">&#39;black&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">xlabel</span><span class="p">(</span><span class="s1">&#39;Gr Liv Area in Train&#39;</span><span class="p">,</span> <span class="n">fontsize</span> <span class="o">=</span> <span class="s1">&#39;18&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">ylabel</span><span class="p">(</span><span class="s1">&#39;train_predictions&#39;</span> <span class="p">,</span><span class="n">fontsize</span> <span class="o">=</span> <span class="s1">&#39;18&#39;</span><span class="p">)</span>
<span class="n">trainPlot</span> <span class="o">=</span><span class="n">plt</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">train</span><span class="p">[[</span><span class="s1">&#39;Gr Liv Area&#39;</span><span class="p">]],</span> <span class="n">train_predictions</span><span class="p">,</span> <span class="n">color</span><span class="o">=</span><span class="s1">&#39;blue&#39;</span><span class="p">,</span> <span class="n">linewidth</span><span class="o">=</span><span class="mi">3</span><span class="p">)</span>
<span class="n">trainPlot</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[4]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>[&lt;matplotlib.lines.Line2D at 0x1a19271d68&gt;]</pre>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>And now lets put the model into test data set to see if it can predict the value precisely.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[5]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">import</span> <span class="nn">matplotlib.pyplot</span> <span class="k">as</span> <span class="nn">plt</span>
<span class="n">plt</span><span class="o">.</span><span class="n">scatter</span><span class="p">(</span><span class="n">test</span><span class="p">[[</span><span class="s1">&#39;Gr Liv Area&#39;</span><span class="p">]],</span> <span class="n">test</span><span class="p">[[</span><span class="s1">&#39;SalePrice&#39;</span><span class="p">]],</span>  <span class="n">color</span><span class="o">=</span><span class="s1">&#39;black&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">xlabel</span><span class="p">(</span><span class="s1">&#39;Gr Liv Area in Test&#39;</span><span class="p">,</span> <span class="n">fontsize</span> <span class="o">=</span> <span class="s1">&#39;18&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">ylabel</span><span class="p">(</span><span class="s1">&#39;test_predictions&#39;</span> <span class="p">,</span><span class="n">fontsize</span> <span class="o">=</span> <span class="s1">&#39;18&#39;</span><span class="p">)</span>
<span class="n">testPlot</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">test</span><span class="p">[[</span><span class="s1">&#39;Gr Liv Area&#39;</span><span class="p">]],</span> <span class="n">test_predictions</span><span class="p">,</span> <span class="n">color</span><span class="o">=</span><span class="s1">&#39;blue&#39;</span><span class="p">,</span> <span class="n">linewidth</span><span class="o">=</span><span class="mi">3</span><span class="p">)</span>
<span class="n">testPlot</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[5]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>[&lt;matplotlib.lines.Line2D at 0x1a192b3ac8&gt;]</pre>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Since all the data looks like concentrate on the linear regression model, we should conclude that the model can predict the "Sale Price" .</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="Use-Multiple-Regression-to-model-the-data">Use Multiple Regression to model the data<a class="anchor-link" href="#Use-Multiple-Regression-to-model-the-data"></a></h2><hr>
<p>In the real world, Multiple Regression is a more useful technique since we need to evaluate more than one correlation in most cases. Now, we will still predict the SalePrice, but with one more variable -- Overall Condition (Overall Cond). In this case the model will be a <strong>Binary Linear Equation</strong> in the form of : 
$$
Y = a_0 + coef_{Cond} * (Overall Cond) + coef_{Area} * (Gr Liv Area)
$$</p>
<p>$a_0$ stands for the intial value while both "Overall Cond" and "Gr Liv Area" is zero</p>
<p>$coef_{Cond}$ stands for the coefficient of Overall Cond</p>
<p>$coef_{Area}$ stands for the coefficient of Gr Liv Area</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[6]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">sklearn.metrics</span> <span class="k">import</span> <span class="n">mean_squared_error</span>
<span class="n">cols</span> <span class="o">=</span> <span class="p">[</span><span class="s1">&#39;Overall Cond&#39;</span><span class="p">,</span> <span class="s1">&#39;Gr Liv Area&#39;</span><span class="p">]</span>
<span class="n">lr</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">train</span><span class="p">[</span><span class="n">cols</span><span class="p">],</span> <span class="n">train</span><span class="p">[</span><span class="s1">&#39;SalePrice&#39;</span><span class="p">])</span>
<span class="n">train_predictions</span> <span class="o">=</span> <span class="n">lr</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">train</span><span class="p">[</span><span class="n">cols</span><span class="p">])</span>
<span class="n">test_predictions</span> <span class="o">=</span> <span class="n">lr</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">test</span><span class="p">[</span><span class="n">cols</span><span class="p">])</span>

<span class="n">train_rmse_2</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">sqrt</span><span class="p">(</span><span class="n">mean_squared_error</span><span class="p">(</span><span class="n">train_predictions</span><span class="p">,</span> <span class="n">train</span><span class="p">[</span><span class="s1">&#39;SalePrice&#39;</span><span class="p">]))</span>
<span class="n">test_rmse_2</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">sqrt</span><span class="p">(</span><span class="n">mean_squared_error</span><span class="p">(</span><span class="n">test_predictions</span><span class="p">,</span> <span class="n">test</span><span class="p">[</span><span class="s1">&#39;SalePrice&#39;</span><span class="p">]))</span>

<span class="nb">print</span><span class="p">(</span><span class="n">lr</span><span class="o">.</span><span class="n">coef_</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">lr</span><span class="o">.</span><span class="n">intercept_</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">train_rmse_2</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">test_rmse_2</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>[-409.56846611  116.73118339]
7858.691146390513
56032.398015258674
57066.90779448559
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>such that the linear model will be like: 
$$
Y = 7858.7 - 409.6 * (Overall Cond) + 116.7 * (Gr Liv Area)
$$</p>
<p>However, it's hard to make a geometric explanation since the model will be either surface or high-dimension which cant be plotted.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="Handling-data-types-with-missing-values/non-numeric-values">Handling data types with missing values/non-numeric values<a class="anchor-link" href="#Handling-data-types-with-missing-values/non-numeric-values"></a></h2><p>In the machine learning workflow, once we've selected the model we want to use, selecting the appropriate features for that model is the next important step. In the following code snippets, I will explore how to use correlation between features and the target column, correlation between features, and variance of features to select features.</p>
<p>I will specifically focus on selecting from feature columns that don't have any missing values or don't need to be transformed to be useful (e.g. columns like Year Built and Year Remod/Add).</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[7]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">numerical_train</span> <span class="o">=</span> <span class="n">train</span><span class="o">.</span><span class="n">select_dtypes</span><span class="p">(</span><span class="n">include</span><span class="o">=</span><span class="p">[</span><span class="s1">&#39;int64&#39;</span><span class="p">,</span> <span class="s1">&#39;float&#39;</span><span class="p">])</span>
<span class="n">numerical_train</span> <span class="o">=</span> <span class="n">numerical_train</span><span class="o">.</span><span class="n">drop</span><span class="p">([</span><span class="s1">&#39;PID&#39;</span><span class="p">,</span> <span class="s1">&#39;Year Built&#39;</span><span class="p">,</span> <span class="s1">&#39;Year Remod/Add&#39;</span><span class="p">,</span> <span class="s1">&#39;Garage Yr Blt&#39;</span><span class="p">,</span> <span class="s1">&#39;Mo Sold&#39;</span><span class="p">,</span> <span class="s1">&#39;Yr Sold&#39;</span><span class="p">],</span> <span class="n">axis</span><span class="o">=</span><span class="mi">1</span><span class="p">)</span>
<span class="n">null_series</span> <span class="o">=</span> <span class="n">numerical_train</span><span class="o">.</span><span class="n">isnull</span><span class="p">()</span><span class="o">.</span><span class="n">sum</span><span class="p">()</span>
<span class="n">full_cols_series</span> <span class="o">=</span> <span class="n">null_series</span><span class="p">[</span><span class="n">null_series</span> <span class="o">==</span> <span class="mi">0</span><span class="p">]</span>
<span class="nb">print</span><span class="p">(</span><span class="n">full_cols_series</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>Order              0
MS SubClass        0
Lot Area           0
Overall Qual       0
Overall Cond       0
1st Flr SF         0
2nd Flr SF         0
Low Qual Fin SF    0
Gr Liv Area        0
Full Bath          0
Half Bath          0
Bedroom AbvGr      0
Kitchen AbvGr      0
TotRms AbvGrd      0
Fireplaces         0
Garage Cars        0
Garage Area        0
Wood Deck SF       0
Open Porch SF      0
Enclosed Porch     0
3Ssn Porch         0
Screen Porch       0
Pool Area          0
Misc Val           0
SalePrice          0
dtype: int64
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="Correlating-feature-columns-with-Target-Columns">Correlating feature columns with Target Columns<a class="anchor-link" href="#Correlating-feature-columns-with-Target-Columns"></a></h2><hr>
<p>I will show the the correlation between feature columns and target columns(SalesPrice) by percentage.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[8]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">train_subset</span> <span class="o">=</span> <span class="n">train</span><span class="p">[</span><span class="n">full_cols_series</span><span class="o">.</span><span class="n">index</span><span class="p">]</span>
<span class="n">corrmat</span> <span class="o">=</span> <span class="n">train_subset</span><span class="o">.</span><span class="n">corr</span><span class="p">()</span>
<span class="n">sorted_corrs</span> <span class="o">=</span> <span class="n">corrmat</span><span class="p">[</span><span class="s1">&#39;SalePrice&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">abs</span><span class="p">()</span><span class="o">.</span><span class="n">sort_values</span><span class="p">()</span>
<span class="nb">print</span><span class="p">(</span><span class="n">sorted_corrs</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>Misc Val           0.009903
3Ssn Porch         0.038699
Low Qual Fin SF    0.060352
Order              0.068181
MS SubClass        0.088504
Overall Cond       0.099395
Screen Porch       0.100121
Bedroom AbvGr      0.106941
Kitchen AbvGr      0.130843
Pool Area          0.145474
Enclosed Porch     0.165873
2nd Flr SF         0.202352
Half Bath          0.272870
Lot Area           0.274730
Wood Deck SF       0.319104
Open Porch SF      0.344383
TotRms AbvGrd      0.483701
Fireplaces         0.485683
Full Bath          0.518194
1st Flr SF         0.657119
Garage Area        0.662397
Garage Cars        0.663485
Gr Liv Area        0.698990
Overall Qual       0.804562
SalePrice          1.000000
Name: SalePrice, dtype: float64
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="Correlation-Matrix-Heatmap">Correlation Matrix Heatmap<a class="anchor-link" href="#Correlation-Matrix-Heatmap"></a></h2><p>We now have a decent list of candidate features to use in our model, sorted by how strongly they're correlated with the SalePrice column. For now, I will keep only the features that have a correlation of 0.3 or higher. This cutoff is a bit arbitrary and, in general, it's a good idea to experiment with this cutoff. For example, you can train and test models using the columns selected using different cutoffs and see where your model stops improving.</p>
<p>The next thing we need to look for is for potential collinearity between some of these feature columns. Collinearity is when 2 feature columns are highly correlated and stand the risk of duplicating information. If we have 2 features that convey the same information using 2 different measures or metrics, we need to choose just one or predictive accuracy can suffer.</p>
<p>While we can check for collinearity between 2 columns using the correlation matrix, we run the risk of information overload. We can instead generate a correlation matrix heatmap using Seaborn to visually compare the correlations and look for problematic pairwise feature correlations. Because we're looking for outlier values in the heatmap, this visual representation is easier.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[9]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">import</span> <span class="nn">seaborn</span> <span class="k">as</span> <span class="nn">sns</span>
<span class="kn">import</span> <span class="nn">matplotlib.pyplot</span> <span class="k">as</span> <span class="nn">plt</span> 
<span class="n">plt</span><span class="o">.</span><span class="n">figure</span><span class="p">(</span><span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mi">10</span><span class="p">,</span><span class="mi">6</span><span class="p">))</span>
<span class="n">strong_corrs</span> <span class="o">=</span> <span class="n">sorted_corrs</span><span class="p">[</span><span class="n">sorted_corrs</span> <span class="o">&gt;</span> <span class="mf">0.3</span><span class="p">]</span>
<span class="n">corrmat</span> <span class="o">=</span> <span class="n">train_subset</span><span class="p">[</span><span class="n">strong_corrs</span><span class="o">.</span><span class="n">index</span><span class="p">]</span><span class="o">.</span><span class="n">corr</span><span class="p">()</span>
<span class="n">ax</span> <span class="o">=</span> <span class="n">sns</span><span class="o">.</span><span class="n">heatmap</span><span class="p">(</span><span class="n">corrmat</span><span class="p">)</span>
<span class="n">ax</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[9]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;matplotlib.axes._subplots.AxesSubplot at 0x1a19605668&gt;</pre>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAmQAAAGtCAYAAAC4HmhdAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4yLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvOIA7rQAAIABJREFUeJzs3Xe8HFX9//HXO40kBAJSAwFDCS0QAgm9BQTERrFFBJUfakBFrCiKBbHgV7AgoBgVI0oTUQyKBBBCAAMkQEgBqYkQQBBpAQIp9/P7Y87CsNm7e9ve2d28nzzmwe7MmTOf3ZT7yeecOaOIwMzMzMyK06foAMzMzMxWdU7IzMzMzArmhMzMzMysYE7IzMzMzArmhMzMzMysYE7IzMzMzArmhMzMzMysEySdL+kpSfPaOS5JP5X0oKQ5knau1acTMjMzM7POmQwcUuX424CRaZsI/LxWh07IzMzMzDohIqYDz1RpchhwQWRuBdaSNKxan07IzMzMzHrWxsCjufeL0r529atrOLaSZU8/3NDPqhq00T5Fh1DT1msPLzqEqoYNWKvoEGp6ccWrRYdQ0+p9BhQdQlX91LfoEGrqr8b+N3dfVHQINQ1W4/+YHKaBRYdQ048WXtJrv9g98XN2wHpbHEc21FgyKSImdaKLSp+3alyN/zvNzMzMrKPaVnS7i5R8dSYBK7cI2CT3fjjweLUTGvufT2ZmZmbNZwrw4XS35e7A8xHxRLUTXCEzMzOz1hFtdb+EpIuB8cC6khYB3wT6A0TEecBVwNuBB4GXgf9Xq08nZGZmZtY62uqfkEXEkTWOB/CpzvTphMzMzMxaRvRChawePIfMzMzMrGCukJmZmVnr6IUhy3pwQmZmZmato0mHLJ2QmZmZWevogXXIiuCEzMzMzFpHk1bIujypX9KPJX02936qpF/l3v9Q0ue7G6CkyZLe287+BZLulnS/pAskVX1OVJVrHCPpnBptNpD013S9eyRdlfaPkLRE0uzc1tjPfDEzM7OG0p27LP8J7AkgqQ+wLjAqd3xP4JZu9N8RJ0XEjsDWwF3ADXVMhk4Dro2IHSNiO+Dk3LGHImJMbltapxjMzMysmra27m8F6E5CdgspISNLxOYBiyWtLWk1YFvgrvTYgDMkzZM0V9IEgBr7z0lVqL8B69cKJDI/Bv4DvC31c7CkGZLulHSZpCFp/y6S/pkqXbdLWiPfl6R3pPPWLbvMMLJnU5WuOafT35iZmZnVVURbt7cidDkhi4jHgeWSNiVLzGYAtwF7AOOAOalS9G5gDLAjcCBwhqRhVfYfQVbx2gH4OK8nfR1xJ7BNSqa+BhwYETsDs4DPp+rZpcBnUmXtQGBJ6WRJR5BVvt4eEU+X9X0u8GtJN0g6RdJGuWNb5IYrzy0PStJESbMkzfrVBRd34uOYmZlZpzRphay7k/pLVbI9gR8BG6fXz5MNaQLsDVwcESuAJyXdCOxSZf++uf2PS7q+E/Eo/X93YDvgFkkAA8gSxq2BJyJiJkBEvACQ2uxPlkgeXNqfFxFTJW0OHEJWhbtL0vbp8EMRMaa9oPJPjV/29MPRic9jZmZmnbGqTepPSvPIdiAbsryVrEKWnz+myqe2ux+gq0nLTsC9qe9rc3O6touIj6b97fX9MLAGsFW7QUU8ExEXRcSHgJlkyaOZmZlZt3Q3IbsFeCfwTESsiIhngLXIkrIZqc10YIKkvpLWI0tibq+x/wNp/zCyylVVad7ZiWTzvK4mSwz3krRlOj5Y0lbAv4CNJO2S9q8hqVQl/DfZMOoFkkZVuMYBkgaXzgO2AB7p1LdlZmZm9dW2ovtbAbo7ZDmX7O7Ki8r2DcnNwfozWYJ2N1l16ksR8R9J1fYfkPq5H7ixyvXPkPR1YDBZErZ/mrf2X0nHABenGwwAvhYR96ebB86WNIhs/tiBpc4i4j5JRwGXSXpXRDyUu9ZY4BxJy8kS2V9FxExJIzr8bZmZmVl9NemQpSI8pak3NfocskEb7VN0CDVtvfbwokOoatiAtYoOoaYXV7xadAg1rd6nsZfz66e+RYdQU391dxCkvvpWnbnSGAar8ddPH6aBRYdQ048WXtJrv9ivzv9Ht3/OrjbqLb3+m7Ox/7SamZmZrQIaP/U3MzMz66gmHbJ0QmZmZmato6B1xLrLCZmZmZm1jGwZ0+bjhMzMzMxaR5MOWXpSv5mZmVnBXCEzMzOz1uE5ZGZmZmYFa9IhSydkZmZm1joKevRRdzkh62WNvhL+ksdvKjqEmg7d6VNFh1DVo0ufLTqEmkYN3LDoEGp6fPniokOoqhlW6n9+xStFh1DVbv3XLzqEmtpo6IerADCExv+92KuatELmSf1mZmZmBXOFzMzMzFqHJ/WbmZmZFaxJhyydkJmZmVnraNIKmeeQmZmZmRXMFTIzMzNrHU1aIXNCZmZmZi3DDxc3MzMzK5orZGZmZmYFa9K7LD2p38zMzKxgrpCZmZlZ62jSIcsuVcgkDZf0F0kPSHpI0lmSBvR0cGXXPEbSfyXNlnSPpI/3UL8LJa1bo81gSRdKmitpnqSbJQ1Jx1akmErbiJ6Iy8zMzLog2rq/FaDTFTJJAv4E/DwiDpPUF5gEfBc4qYfjK3dpRJwgaX1gvqQpEfFkrZMk9YuI5d247meAJyNih9Tf1sCydGxJRIzpRt9mZmbWU1ahCtkBwCsR8RuAyO4v/RxwbKokHZOqZ1dLuk/SN0snSjpa0u2pkvSLlMwh6UVJ35V0t6RbJW1QLYCIeAp4CHizpDdJukLSnHTu6NTnqZImSboGuEBSX0lnpirXHEmfznX5aUl3pmPbVLjkMOCx3PXvi4hXu/DdmZmZWT01aYWsKwnZKOCO/I6IeAF4BNgy7doVOAoYA7xP0jhJ2wITgL1SRWlFagOwOnBrROwITAeqDkdK2hzYHHgQ+BZwV0SMBr4KXJBrOhY4LCI+CEwENgN2Sm0vzLV7OiJ2Bn4OfLHCJc8HvixphqTvSBqZOzYoN1z553binShplqRZbW0vVftoZmZmtgrqyqR+AVFj/7UR8T8ASX8C9gaWkyVIM7NRTwYBT6X2S4G/ptd3AAe1c+0JkvYGXgWOi4hn0vv3AETE9ZLWkTQ0tZ8SEUvS6wOB80pDlxHxTK7fP+Wu/e7yi0bE7JQEHpz6mSlpj4i4lw4MWUbEJLJhXfoN2LjSd2dmZmY9oUmHLLuSkM0nJUAlktYENiEbRhzLyglbkCVsv42Ir1Toc1lElM5ZUSWuSyPihLJ9qtCu1NdLZe3aS4ZKw4/tXjsiXiRL3P4kqQ14O3BvO/2ZmZlZEZo0IevKkOU/gMGSPgyQ5oH9EJgcES+nNgeluV2DgMOBW9J5700T8knH39ztT5ANcR6V+hxPNvz4QoV21wDHS+pXun5HLyBpL0lrp9cDgO2Af3czbjMzM+tpq8ocslTJOoJsbtgDwP3AK2Tzt0puBn4HzAYuj4hZEXEP8DXgGklzgGvJJst316nAuNTn94GPtNPuV2Tz3OZIuhv4YCeusQVwo6S5wF3ALODyLkdsZmZmlqPXRwp7qEPpGGBchaFFo/HnkC15/KaiQ6jp0J0+VXQIVT269NmiQ6hp1MANiw6hpseXLy46hKoG96nr0os94uW2pUWHUNVu/dcvOoSa2tqd6dI4htC36BBqOm3hhZWmF9XFkilndvsXbdChX+y1eEu8Ur+ZmZm1jiZ9lmWPJ2QRMRmY3NP9mpmZmdXUpJP6XSEzMzOz1tGkFbIuPcvSzMzMzHqOK2RmZmbWOjxkaWZmZlYwJ2RmZmZmBevh5bx6ixMyMzMzax2ukFlHbL328KJDqKrRF10FmHLXuUWHUNMXxlV6ZGvj+NCyxl4wFOCafo29eO0iLSs6hJrW6dvYf8Wf8vWNig6hph+d9kTRIdS0+yvLiw7BekBj/2k1a0KNnoyZmbU0V8jMzMzMCtak65A5ITMzM7PW0aQVMi8Ma2ZmZlYwV8jMzMysdTTpsheukJmZmVnraGvr/laDpEMk3SfpQUknVzi+qaQbJN0laY6kt9fq0xUyMzMzax11nkMmqS9wLnAQsAiYKWlKRNyTa/Y14A8R8XNJ2wFXASOq9euEzMzMzFpH/e+y3BV4MCIeBpB0CXAYkE/IAlgzvR4KPF6rUydkZmZmZh23MfBo7v0iYLeyNqcC10j6NLA6cGCtTj2HzMzMzFpGtEW3N0kTJc3KbRNzl1Cly5a9PxKYHBHDgbcDv5NUNedyhczMzMxaRw/MIYuIScCkdg4vAjbJvR/OykOSHwUOSX3NkDQQWBd4qr1rVs3WJK0jaXba/iPpsdz7ARXav0nS8bn3W0paktrfK2mypLokgZLOlfSIJOX2fUfSZzvZz+6SbpT0gKQ7JV0paVQHz10kaa3Oxm5mZmY9JNq6v1U3ExgpabOUC30AmFLW5hHgLQCStgUGAv+t1mnVhCwi/hcRYyJiDHAe8OPS+4io9HTiNwHHl+27L52/A7AZ8J5q1+yKdMfDocATwF7d6GcYcDHwpYgYGRE7A2cAW1Ro6+qimZnZKiYilgMnAFOBe8nuppwv6TRJh6ZmXwA+LulusrzimIjqC6R1OamQ9CXgw+ntLyLibOD7wNaSZgNXA7/KfwBJM8kmwyHpY2TjqgOAUcAPgCHAB4ElwNsj4jlJnwM+DiwD5kbE0RXCORC4C/gL2bjtzbljO0m6gaykeHpEnC/p8hTzNSmW3wOXAbsD50fEbbm4p+c+8++BJ4GdyW5zPQO4CFgHuI3K48pmZmbWW9rqvzBsRFxFtpRFft83cq/voZMFoi5N6pe0K3AU2a2fewCflDQaOJlUEYuIk8vOGQTsQpZRlowCJpAlQv8HPBsROwF3AKXE60vAmIjYkSwjreRIsgz0cuCwsurVDsDbyL6Y0yRtAFySrksa192PLIEcBdxZ4+NvAbwlIr4EfAu4IVXSrgY2qnGumZmZ1VMvLAxbD129y3If4PKIeDkiFgNXAHu307ZUMfsf2bod83PHro+IlyLiSeBF4Mq0fy6vL6A2H/i9pKPIqmRvIGk14GBgSkQ8R5ZQvSXX5IqIeCUingKmkyWFfwMOktQfeEeK49UKfc+S9C9JP8ztvizitQHmfYHfA0TEX4DFlb6A/N0azy5pdz6fmZmZddcqlpB1ZmiuNIdsS2C/sscH5JOgttz7Nl4fTn0r2fy1XYFZab5Y3jvIFl2bL2khWcXuyNzx8tplRMTLwC1kq+xOIKuYQZb87ZxrOI5sLZGhufNfKu+PGiJiUkSMi4hxaw9av1ZzMzMz66qI7m8F6GpCNh04QtIgSUPIVqi9iaxCtEalEyLiceAraeuQlHwNj4jrgZOA9YDBZc2OJJssNyIiRgCbA29LQ5EAh0taTdK6ZJW9WWn/JWS3pe4BXJf2nQ18TNLuuf7Lr5c3nWzoFknvop3PbmZmZlZNlyb1R8Ttki4mu/UT4OcRMRdeG+abSzYs+KuyU/8InCppj07Ed5GkNciSx/9LQ6Skaw0hG578f7nYFku6jaxyRorx72RrhnwzDY9CNufrt2RDkMvSuY9LOhI4Q9KGZOuFPE02V6ySbwIXS3o/cAPwWAc/l5mZmdVDQUOO3dXhhCwiTi17/wOyOyPL200o2zUmdyzIJs4DzCg7b3judT6Ra/cuhYh4kWypjfL9pdtOL69y7qvASmuGRcQ/ySpplc45uuz9f3nj4xC+0N71zMzMrBf0wl2W9eC1tMzMzKx11P/h4nXhhMzMzMxaR5NWyPxwcTMzM7OCuUJmZmZmLSNafVK/mZmZWcNr0iFLJ2RmZmbWOpp0Ur/nkJmZmZkVzBUyMzMzax0esjQzMzMrmCf1W0cMG7DSwwEayqNLny06hJq+MK7Dj0MtxA9nnV50CDVNGPvZokOo6d3LVHQIVT0xoLHjawbXndL4T5sbMqBv0SFYZ7lCZmZmZlYwT+o3MzMzs65whczMzMxah4cszczMzIrllfrNzMzMiuYKmZmZmVnBmjQh86R+MzMzs4K5QmZmZmato0mXvXBCZmZmZq2jSYcsnZCZmZlZy4gmTcg8h8zMzMysYA2VkElaIWl2bhshaZykn/bgNRZKWren+jMzM7MG0hbd3wrQaEOWSyJiTNm+hcCs8oaS+kXE8l6JyszMzJpDky4M21AVskokjZf01/T6VEmTJF0DXCCpr6QzJM2UNEfScblzpkv6s6R7JJ0naaXPKukKSXdImi9pYm7/IZLulHS3pH+kfatLOj9d6y5Jh6X9oyTdnip6cySN7JUvxszMzFbmClmPGCRpdnq9ICKOqNBmLLB3RCxJSdTzEbGLpNWAW1KyBrArsB3wb+Bq4N3AH8v6OjYinpE0CJgp6XKyJPWXwL4RsUDSm1LbU4DrI+JYSWsBt0u6DjgeOCsiLpQ0AOhbHnCKcyLANmttx8ZDhnfhqzEzM7OamnRSf6MlZJWGLMtNiYgl6fXBwGhJ703vhwIjgaXA7RHxMICki4G9WTkhO1FSKenbJJ27HjA9IhYARMQzuWsdKumL6f1AYFNgBnCKpOHAnyLigfKAI2ISMAngwE3e2py/U8zMzKxuGi0h64iXcq8FfDoipuYbSBoPlCc+UaHNgcAeEfGypGlkSZYqnFu61nsi4r6y/fdKug14BzBV0sci4vpOfSIzMzPrERHNWfdo+DlkNUwFPiGpP4CkrSStno7tKmmzNHdsAnBz2blDgWdTMrYNsHvaPwPYT9Jmqc/SkOVU4NOSlPbvlP6/OfBwRPwUmAKMrscHNTMzsw7wHLJC/AoYAdyZEqX/AoenYzOA7wM7ANOBP5edezVwvKQ5wH3ArQAR8d805+tPKZl7CjgI+DbwE2BOutZC4J1kyd7RkpYB/wFOq8snNTMzs9o8h6z7ImJIhX3TgGnp9allx9qAr6btNamI9XJETKjQ34jc27e1E8ffgb+X7VsCHFeh7enA6ZX6MTMzs97llfrNzMzMrEsaqkLWU/JVNTMzM1uFNGmFrCUTMjMzM1tFNedC/U7IzMzMrHV4DpmZmZmZdYkrZGZmZtY6mrRC5oTMzMzMWofnkJmZmZkVq1nnkDkhMzMzs9bhCpl1xIsrXi06hKpGDdyw6BBq+tCypUWHUNWEsZ8tOoSaLr3jJ0WHUNO3xn2t6BCq2n5F/6JDqGnTpcuLDqGqXUY/UXQINa0zb92iQ6ipf78mzUDsDZyQmZmZWcvwkKWZmZlZ0Zq0YOiEzMzMzFpGOCEzMzMzK1iTJmReqd/MzMysYK6QmZmZWcvwkKWZmZlZ0ZyQmZmZmRWrWStknkNmZmZmVjBXyMzMzKxlNGuFzAmZmZmZtYxmTciaZshS0gpJs3PbiBrtF0paN71+sUafd0u6U9KeNfpcS9Inc+/HS/pr5z+NmZmZ1UWo+1sBmqlCtiQixtSrT0lvBU4H9qvSfi3gk8DPejgOMzMz6wGukBVA0jGSzsm9/6uk8V3sbk3g2dTPEEn/SFWzuZIOS22+D2yRqmpnpH1DJP1R0r8kXSipmNTazMzMeoWkQyTdJ+lBSSe30+b9ku6RNF/SRbX6bKYK2SBJs9PrBRFxRA/2ORAYBhyQ9r8CHBERL6Rhz1slTQFOBrbPVdXGAzsBo4DHgVuAvYCb8xeRNBGYCLDZ0JGsP3ijHgjdzMzMykVbfesikvoC5wIHAYuAmZKmRMQ9uTYjga8Ae0XEs5LWr9VvMyVk9R6y3AO4QNL2gIDvSdqXbIm5jYEN2unj9ohYlPqYDYygLCGLiEnAJIDdNxofPfwZzMzMLOmFIctdgQcj4mEASZcAhwH35Np8HDg3Ip4FiIinanXaTAlZJct547DrwK52FBEzUjVsPeDt6f9jI2KZpIVV+n4193oFzf+dmpmZNa2o/6T8jYFHc+8XAbuVtdkKQNItQF/g1Ii4ulqnzZ48LAQ+KakP2Re0a1c7krQN2Zf2P2Ao8FRKxvYH3pyaLQbW6FbEZmZmVjc9USHLTzVKJqXRLshG0Va6bNn7fsBIYDwwHLhJ0vYR8Vx712z2hOwWYAEwF5gH3NnJ8/Pz0gR8JCJWSLoQuFLSLGA28C+AiPifpFskzQP+DvytJz6EmZmZNY78VKMKFgGb5N4PJ5tHXt7m1ohYBiyQdB9ZgjazvWs2TUIWEUMq7AvgqHbaj6h2btrft539TwN7tHPsg2W7puWOnVDpHDMzM+sd9Z7UT5ZUjZS0GfAY8AGgPDe4AjgSmJymQ20FPFyt06ZJyMzMzMxqiTrfOhcRyyWdAEwlm+p0fkTMl3QaMCsipqRjB0u6h2x++UkR8b9q/TohMzMzs5bRCxUyIuIq4Kqyfd/IvQ7g82nrkKZeGNbMzMysFbhCZmZmZi2jNypk9eCEzMzMzFpGveeQ1YsTMjMzM2sZrpCZmZmZFawXVuqvC0/qNzMzMyuYK2S9bPU+A4oOoarHly8uOoSarum3YdEhVPXuZY3/r7Nvjfta0SHU9M1Z3yk6hKpuGPXVokOo6fk+Fde+bhgPz1+n6BBqWtrW+HWLbfeuurzVKqcXHi5eF07IzMzMrGW0NemQpRMyMzMzaxnNOofMCZmZmZm1jGa9y7LxB8fNzMzMWpwrZGZmZtYyvDCsmZmZWcGadcjSCZmZmZm1jGa9y9JzyMzMzMwK5gqZmZmZtQwve2FmZmZWME/qNzMzMyuY55DViaTzJT0laV4H2o6XtGc7x46R9F9Js9N2Qdo/WdJ7O9D31pKmpXPvlTQpd83nc/1e19nPaGZmZj0jQt3eitAMFbLJwDnABR1oOx54EfhnO8cvjYgTOnJRSX0jYkVu10+BH0fEX9LxHXLHboqId3akXzMzM7NyDV8hi4jpwDPl+yWdKOkeSXMkXSJpBHA88LlUqdqns9eStFDSNyTdDLyv7PAwYFEurrmd7d/MzMzqK6L7WxGaoULWnpOBzSLiVUlrRcRzks4DXoyIM9s5Z4KkvdPrsyLiNxXavBIRe1fY/2Pgekn/BK4BfhMRz6Vj+0ianV5fFhHf7eJnMjMzs25o1jlkzZyQzQEulHQFcEUHz+nIkOWllXZGxG8kTQUOAQ4DjpO0YzpcdchS0kRgIsDWa23LxqsP72C4ZmZm1hnNuuxFww9ZVvEO4FxgLHCHpJ5KLl9q70BEPB4R50fEYcByYPuOdBgRkyJiXESMczJmZmZm5ZoyIZPUB9gkIm4AvgSsBQwBFgNr1Omah0jqn15vCKwDPFaPa5mZmVnXtIW6vRWh4RMySRcDM4CtJS2S9FGgL/B7SXOBu8jufnwOuBI4oquT+ms4GJgn6W5gKnBSRPynh69hZmZm3RA9sBWh4eeQRcSR7RxaaeJ9RNwPjG6nn8lkS2iU7z8m93pElTg+D3y+wv5pwLT2zjMzM7Pe40n9ZmZmZgXzpH4zMzMz6xJXyMzMzKxltBUdQBc5ITMzM7OWETTnkKUTMjMzM2sZbUXdJtlNTsjMzMysZbQ1aYXMk/rNzMzMCuYKmZmZmbUMzyEzMzMzK5jvsrQO6ae+RYdQVaPHB7BIy4oOoaonBjT+v862X9G/6BBqumHUV4sOoar953+v6BBqWj5vWtEhVHX9hGuLDqGmUcOeLjqEmtSn8f/O6U3NWiHzHDIzMzOzgrlCZmZmZi3DQ5ZmZmZmBXNCZmZmZlawZp1D5oTMzMzMWkZbc+ZjntRvZmZmVjRXyMzMzKxlNOujk5yQmZmZWcto0meLOyEzMzOz1uG7LM3MzMwK1qbmHLL0pH4zMzOzgtU1IZO0gaSLJD0s6Q5JMyQdUc9rdpaksyQ9JsnJqZmZWZOLHtiKULckRJKAK4DpEbF5RIwFPgAM70QfdX3SdUrCjgAeBfZtp42Hdc3MzJpEWw9sRahnVegAYGlEnFfaERH/joizASSNkHSTpDvTtmfaP17SDZIuAuamfVekCtt8SRNL/Un6qKT7JU2T9EtJ56T960m6XNLMtO3VToz7A/OAnwNH5vo9VdIkSdcAF0jqK+mM1NccSceldkMk/SPFP1fSYT34/ZmZmVkntan7WxHqWf0ZBdxZ5fhTwEER8YqkkcDFwLh0bFdg+4hYkN4fGxHPSBoEzJR0ObAa8HVgZ2AxcD1wd2p/FvDjiLhZ0qbAVGDbCjEcma77F+B7kvpHxLJ0bCywd0QsSUng8xGxi6TVgFtSsvYocEREvCBpXeBWSVMi4g0Vz3T+RIDt1hrF8CGb1PjqzMzMbFXSa8Nxks4F9iarmu0C9AfOkTQGWAFslWt+ey4ZAzgxN/dsE2AksCFwY0Q8k/q/LNfHgcB2ev1OizUlrRERi3PxDADeDnwuIhZLug04GPhbajIlIpak1wcDoyW9N70fmmJYRJbI7UtW5dwY2AD4T/6zR8QkYBLAWzd5W7MukWJmZtbwvDDsyuYD7ym9iYhPpSrSrLTrc8CTwI5kQ6ev5M59qfRC0niyBGuPiHhZ0jRgIFT9xvuk9kuqtDmELLGamxK3wcDLvJ6QvZRrK+DTETE134GkY4D1gLERsUzSwhSbmZmZFaBZqx71nEN2PTBQ0idy+wbnXg8FnoiINuBDQHsT+IcCz6ZkbBtg97T/dmA/SWuniffvyZ1zDXBC6U2qwpU7EvhYRIyIiBHAZsDBkgZXaDsV+ISk/qm/rSStnmJ7KiVj+wNvbuczmJmZWS9o1jlkdUvI0jyqw8mSpgWSbgd+C3w5NfkZ8BFJt5INNb5UuSeuBvpJmgN8G7g19f8Y8D3gNuA64B7g+XTOicC4NAH/HuD4fIcp6Xorr1fDiIiXgJuBd1WI4Vep/zslzQN+QVZdvDBdZxZwFPCvDnw1ZmZmVifNepdlXeeQRcQTZEtdVDr2ADA6t+sraf80YFqu3avA29q5xEURMSlVyP5MVhkjIp4GJlSJ62XgTRX2v7ud9m3AV9NWbo/2rmNmZmbWEc2+GOqpkmaTLV2xgGzdMzMzM1tFNevCsE296GlEfLHoGMzMzKxxFDUHrLuavUJmZmZm9premEMm6RBJ90l6UNLJVdq9V1JIGtdem5KmrpCZmZmZ5dV7Un56rOO5wEFk65HOTIvC31PWbg2ymwxv60i/rpCZmZmZddyuwIMR8XBELAUuASo9OvHbwA944zqr7XJCZmZmZi0j1P1N0kQmEBTzAAAgAElEQVRJs3LbxNwlNiZ7dGLJorTvNZJ2AjaJiL92NG4PWZqZmVnL6Ikhy/wjDyuodNvAazdnSuoD/Bg4pjPXdEJmZmZmLaMXFnZdRPZc7ZLhwOO592sA2wPT0qMZNwSmSDo0ImbRDg9ZmpmZmXXcTGCkpM0kDSBbAH9K6WBEPB8R6+YezXgrUDUZA1fIel1/NXYO/PyKDs09LNQ6ff3btrs2Xbq86BBqer5Pe4+3bQzL500rOoSa+m0/vugQqlpTVxUdQk2rDWn8PysurbxRvRd2jYjlkk4ge851X+D8iJgv6TRgVkRMqd5DZf7JZmZmZi2jNxaGjYirgKvK9n2jnbbjO9KnEzIzMzNrGUU9HLy7nJCZmZlZy2jWhMwjz2ZmZmYFc4XMzMzMWka9J/XXixMyMzMzaxm9Mam/HpyQmZmZWcto1jlkTsjMzMysZTTrkKUn9ZuZmZkVzBUyMzMzaxltTVojc0JmZmZmLaNZ55DVfchS0gaSLpL0sKQ7JM2QdES9r9tRknaVNF3SfZL+JelXkgYXHZeZmZl1XvTAVoS6JmSSBFwBTI+IzSNiLNlT0Yd3oo+6PWFY0gbAZcCXI2JrYFvgamCNDp7vCqOZmZl1W70rZAcASyPivNKOiPh3RJwNIGmEpJsk3Zm2PdP+8ZJukHQRMDftuyJV2OZLmljqT9JHJd0vaZqkX0o6J+1fT9Llkmamba8K8X0K+G1EzEixRUT8MSKeTJWzf0q6K/1/69TvMZIuk3QlcI2kYanCNlvSPEn71OWbNDMzs5raemArQr0rPKOAO6scfwo4KCJekTQSuBgYl47tCmwfEQvS+2Mj4hlJg4CZki4HVgO+DuwMLAauB+5O7c8CfhwRN0vaFJhKVgHL2x74bTux/QvYNyKWSzoQ+B7wnnRsD2B0iucLwNSI+G6q5q003JkSyIkAO6y9PZsO2bTKV2JmZmZd5YVhO0DSucDeZFWzXYD+wDmSxgArgK1yzW/PJWMAJ+bmnm0CjAQ2BG6MiGdS/5fl+jgQ2C4bNQVgTUlrRMTiDoY7FPhtShQjxVpybemawEzgfEn9gSsiYnZ5RxExCZgE8M5N39Gct3+YmZk1gWa9y7LeQ5bzyapXAETEp4C3AOulXZ8DngR2JKuMDcid+1LphaTxZAnWHhGxI3AXMBColgf3Se3HpG3jCsnYfGBsO+d/G7ghIrYH3pWut1JsETEd2Bd4DPidpA9XicnMzMzqyJP6K7seGCjpE7l9+SG9ocATEdEGfAhobwL/UODZiHhZ0jbA7mn/7cB+ktZOE+zfkzvnGuCE0ptUhSt3DvARSbvl2h0tacN0zcfS7mPa+4CS3gw8FRG/BH5NLgE1MzMz64i6JmQREcDhZEnTAkm3k83Z+nJq8jOyhOhWsqHGlyr3xNVAP0lzyCpXt6b+HyOb23UbcB1wD/B8OudEYJykOZLuAY6vEN+TZHd9npmWvbgX2Ad4AfgBcLqkW2g/UQQYD8yWdBdZQnhW9W/FzMzM6sWT+tsREU+QJT2Vjj0AjM7t+kraPw2Ylmv3KvC2di5xUURMShWyP5NVxoiIp4EJHYhvBlkSVm4Gb5zT9vXUfjIwOXf+b2n/xgAzMzPrRZ5DVpxTJc0G5gELyNY9MzMzs1VQs84ha/qFTSPii0XHYGZmZo3Bj04yMzMzsy5p+gqZmZmZWUmzziFzQmZmZmYtoznTMSdkZmZm1kI8h8zMzMzMusQVMjMzM2sZ0aSDlk7Ielnfqo/fLN5u/dcvOoSaTvn6RkWHUNV1pzxWu1HBdhn9RNEh1PTw/HWKDqGq6ydcW3QINa2pq4oOoard5/2g6BBqWrDPJ4sOoaYhoxv750pva9YhSydkZmZm1jJ8l6WZmZlZwZozHfOkfjMzM7PCuUJmZmZmLcNDlmZmZmYF86R+MzMzs4J52QszMzOzgjVrhcyT+s3MzMwK5gqZmZmZtQwPWZqZmZkVrFmHLJ2QmZmZWctoi+askBU6h0zSBpIukvSwpDskzZB0RAfPfbHCvuMlfbiTMfST9LSk0ztznpmZmVlPKSwhkyTgCmB6RGweEWOBDwDDK7TtUCUvIs6LiAs6GcrBwH3A+1NMlWLt28k+zczMrADRA1sRiqyQHQAsjYjzSjsi4t8RcTaApGMkXSbpSuCajnQo6VRJX5S0raTbc/tHSJrTzmlHAmcBjwC7585ZKOkbkm4G3idpC0lXp0reTZK2Se3eJek2SXdJuk7SBp38HszMzKyHtBHd3opQ5ByyUcCdNdrsAYyOiGc603FE3CtpgKTNI+JhYALwh/J2kgYBbwGOA9YiS85m5Jq8EhF7p7b/AI6PiAck7Qb8jCypvBnYPSJC0seALwFf6Ey8ZmZm1jOa9S7LhlmHTNK5ku6WNDO3+9rOJmM5fwDen15PAC6t0OadwA0R8TJwOXBE2fDkpSm2IcCewGWSZgO/AIalNsOBqZLmAieRJZrln22ipFmSZi188ZEufhwzMzOrpa0HtiIUmZDNB3YuvYmIT5FVq9bLtXmpG/1fSjYvbKus+3igQpsjgQMlLQTuANYB9q9w/T7AcxExJrdtm46dDZwTETuQVdoGll8kIiZFxLiIGDdiyKbd+EhmZmbWiopMyK4HBkr6RG7f4J7qPCIeAlYAX6dCdUzSmsDewKYRMSIiRgCfIkvSyvt6AVgg6X3pXEnaMR0eCjyWXn+kp+I3MzOzzmvWOWSFJWQREcDhwH6SFqRJ+L8FvtzBLgZLWpTbPl+hzaXA0VSYPwa8G7g+Il7N7fsLcKik1Sq0Pwr4qKS7yap7h6X9p5INZd4EPN3B2M3MzKwOogf+K0KhC8NGxBNkS11UOjYZmFzl3JrJZEScCZzZ0f7TfLXSkOmIsmMLgEMq9PMXskTOzMzMCtasK/U3zKR+MzMzs1WVH51kZmZmLSOa9NFJTsjMzMysZRQ1Kb+7nJCZmZlZy2jWOWROyMzMzKxleKV+MzMzM+sSV8jMzMysZXgOmZmZmVnBfJelmZmZWcE8qd86ZLAa+ytvhlLvj057ougQqhoyoG/RIdS0zrx1iw6hpqVtjT3FddSwxn9S2mpDlhcdQlUL9vlk0SHUtNlNPys6hJqWX3FO0SE0FE/qNzMzM7MucUJmZmZmLaON6PZWi6RDJN0n6UFJJ1c4/nlJ90iaI+kfkt5cq08nZGZmZtYyIqLbWzWS+gLnAm8DtgOOlLRdWbO7gHERMRr4I/CDWnE7ITMzM7OW0QsVsl2BByPi4YhYClwCHJZvEBE3RMTL6e2twPBanTohMzMzM8uRNFHSrNw2MXd4Y+DR3PtFaV97Pgr8vdY1G/uWPzMzM7NO6Im7LCNiEjCpncOqeNlKDaWjgXHAfrWu6YTMzMzMWkZb/ReGXQRskns/HHi8vJGkA4FTgP0i4tVanXrI0szMzFpG9MBWw0xgpKTNJA0APgBMyTeQtBPwC+DQiHiqI3G7QmZmZmYto94LnEfEckknAFOBvsD5ETFf0mnArIiYApwBDAEukwTwSEQcWq1fJ2RmZmZmnRARVwFXle37Ru71gZ3t0wmZmZmZtYxmeARgJXWfQyZpuKS/SHpA0kOSzkpjrvW+7ovp/yMkzWunzShJ10u6P8X2LUld/k4kLZTU+A8JNDMza1H1Xhi2XuqakCkbOP0TcEVEjAS2IhtT/W4P9N2t6p6kQWST8L4fEVsBO5At9vaZ7sZmZmZmxeiNRyfVQ70rZAcAr0TEbwAiYgXwOeBYSYMl3SZpVKmxpGmSxkpaXdL5kmZKukvSYen4MZIuk3QlcI2kIekZUXdKmltq10EfBG6JiGtSbC8DJwAnpWudKumLudjmSRqRXl8h6Q5J88sWizMzM7MCRQ/8V4R6J2SjgDvyOyLiBeARYEuyxw28H0DSMGCjiLiDbN2O6yNiF2B/4AxJq6cu9gA+EhEHAK8AR0TEzqndD1NVrquxPQQMkrRWjXOPjYixZIu9nShpnWqN8yv+Pvjiwg6GZ2ZmZquKeidkovKSHqX9fwDel/a9H7gsvT4YOFnSbGAaMBDYNB27NiKeyfXzPUlzgOvIHl2wQQ/EVsuJku4mez7VJsDIao0jYlJEjIuIcVsOGdHB8MzMzKyzmnUOWb3vspwPvCe/Q9KaZEnMQxHxsqT/SRoNTACOKzUD3hMR95WduxvwUm7XUcB6wNiIWCZpIVny1tHY9i3rf3Pg6Yh4TtJy3piwDkxtxgMHAnuk+Kd14ppmZmZWR77LsrJ/AIMlfRhAUl/gh8Dk3FPQLwG+BAyNiLlp31Tg06Xhx7TibSVDgadSMrY/8OZOxHYhsHd6tEFpkv9PgW+m4wuBndOxnYHNctd8NiVj2wC7d+KaZmZmVkfNWiGra0IW2ac6AnifpAeA+8nmfX011+yPZI8d+ENu37eB/sCctGTFt9u5xIXAOEmzyKpl/+pEbEuAQ4FTJN0PPE02yf/C1ORy4E1p2PQTKXaAq4F+aZj022TDlmZmZmZdVveFYSPiUeBdVY4/WR5HSpaOq9B2MjA59/5pskn+lfodkv6/ENi+nTbzyG4GQNLhwI8kXRQR/04xHNxO2G9rp78R7bQ3MzOzXtCsQ5ZeqT+JiCuAK4qOw8zMzLquqGUrussJmZmZmbWMtoLmgHWXEzIzMzNrGc1aIav7syzNzMzMrDpXyMzMzKxleMjSzMzMrGDNOmTphMzMzMxahitkZmZmZgVr1gqZJ/WbmZmZFcwVsl42TI39HPIh9C06hJp2f2V50SE0vf792ooOoaZt9/5f0SFUpT4qOoTaGvyf3ENGN/53uPyKc4oOoaZ+h59QdAgNxUOWZmZmZgVr1iFLJ2RmZmbWMiIafwSgkgYvaJuZmZm1PlfIzMzMrGW0ecjSzMzMrFjhSf1mZmZmxXKFzMzMzKxgzVoh86R+MzMzs4K5QmZmZmYtwwvDmpmZmRXMC8OamZmZFcxzyOpI0imS5kuaI2m2pN2qtJ0s6b01+pssaUHq605Je7TT7nhJH+5u/GZmZtY72ohub0Vo+ApZSpbeCewcEa9KWhcY0ANdnxQRf5R0MPALYHTZdftFxHk9cB0zMzOzqho+IQOGAU9HxKsAEfE0gKRvAO8CBgH/BI6LsjqlpLHAj4AhwNPAMRHxRFn/04EtU/tpqa+9gCmS1gBejIgzJW0JnAesB6wA3hcRD0k6CXg/sBrw54j4Zg9/fjMzM+sgD1nWzzXAJpLul/QzSful/edExC4RsT1ZUvbO/EmS+gNnA++NiLHA+cB3K/T/LmBu7v1aEbFfRPywrN2FwLkRsSOwJ/BEqq6NBHYFxgBjJe1bfgFJEyXNkjRrzuKHOvnxzczMrKPaIrq9FaHhK2QR8WKqdO0D7A9cKulkYLGkLwGDgTcB84Erc6duDWwPXCsJoC+Qr46dIelrwH+Bj+b2X1oeQ6qUbRwRf04xvZL2HwwcDNyVmg4hS9Cml32GScAkgM+P+EBzpu5mZmZNoFkrZA2fkAFExApgGjBN0lzgOLI5X+Mi4lFJpwIDy04TMD8iKk7YJ80hq7D/pQr71E4fAk6PiF/U+AhmZmZm7Wr4IUtJW0samds1BrgvvX5a0hCg0l2V9wHrle6glNRf0qiuxBARLwCLJB2e+lpN0mBgKnBsigFJG0tavyvXMDMzs+7zXZb1MwQ4W9JawHLgQWAi8BzZ3K+FwMzykyJiaVr+4qeShpJ91p+QDW12xYeAX0g6DVhGNqn/GknbAjPSsOiLwNHAU128hpmZmXVDsw5ZqlkDb1aNPodsCH2LDqGm/ZYsLzqEprdmv6VFh1DTlns+W3QIValPezMZGkiDj4H0Xbt8pknj6bfztkWHUFO/w08oOoSa+q+7ea/9gRkyeLNu/5x98eUFvf4HvBkqZGZmZmYd0qyPTmrwfz+ZmZmZtT5XyMzMzKxlFLWOWHc5ITMzM7OW0axz452QmZmZWcto1jlkTsjMzMysZTRrhcyT+s3MzMwK5gqZmZmZtYxmrZA5ITMzM7OW0ZzpmFfqb3qSJkbEpKLjqMYxdl+jxweNH2OjxweOsSc0enzQ+DE2enytynPImt/EogPoAMfYfY0eHzR+jI0eHzjGntDo8UHjx9jo8bUkJ2RmZmZmBXNCZmZmZlYwJ2TNrxnG+R1j9zV6fND4MTZ6fOAYe0KjxweNH2Ojx9eSPKnfzMzMrGCukJmZmZkVzAmZmZmZWcG8MGyTkNQvIpYXHUc1kjaNiEeKjqMaSWtWOx4RL/RWLGYlktYHBpbeN9KfI0l7AbMj4iVJRwM7A2dFxL8LDq3hSXpTteMR8UxvxWKNz3PImoSkOyNi5/T67Ij4dNExlSuL8fKIeE/RMZWT9CjZQs4CNgIWp9dDgMciYtMCw3sDSVsAiyLiVUnjgdHABRHxXLGRvU7SnsAIcv+4i4gLCgsIkHRNRBycXn8lIk4vMp5qJB0K/JDs9+JTwJuBeyNiVKGB5UiaA+xI9vvvd8CvgXdHxH6FBpZIWg/4MrAdb0xqDygsqETSAl7/+6ZcRMTmvRxSVZL2BkZGxG/S9zokIhYUHdeqwkOWzSP/B3qvwqKoLh9jQ/1FUxIRm6Sk60rgiIhYKyKGAocDlxYb3UouB1ZI2pLsh+BmwEXFhvQ6Sb8DzgT2BnZJ27hCg8qsl3v9vsKi6JhvA7sD90fEZsBbgFuKDWklyyP7l/thZJWxs4A1Co4p70LgXrI/H98CFgIziwyoJCI2i4jN0//Lt4b6O1LSN8kS26+kXf2B3xcX0arHQ5bNoxlKmdHO60a0a0R8svQmIq5MfyE1kraIWC7pCOAnEXG2pLuKDipnHLBdNF6ZvdHiqWZZRPxPUh9JfSLiBkn/V3RQZRZL+gpwNLCvpL5kP6wbxToR8WtJn4mIG4EbJd1YdFDlJK0NjOSNVbzpxUW0kiOAnYA7ASLicUmNlHi3PCdkzWObNHQgYIv0mvQ+ImJ0caG9ZkdJL5DFNCj3GrIYq87f6mXPSDqZ7F+AQfbD5tliQ1rJMklHAh8B3pX2NdIPwnnAhsATRQdSZnNJU8h+75VevyYiDi0mrIqekzQEuAm4UNJTQKPNFZ0AfBD4aET8R9KmwBkFx5S3LP3/CUnvAB4HhhcYz0okfQz4DFlcs8mqojOAwodVc5ZGREgKAEmrFx3QqsZzyJqEpDdXO+4Jtp0jaV2y4Y19yRKy6cCpEfF0oYHlSNoOOB6YEREXS9oMmBAR3y84rivJvrM1gDHA7cCrpeNFJzySqs5tSlWUhpB+6C0hmz5yFDAUuDAi/ldoYEmqhk2NiAOLjqU9kt5JltBuApwNrAl8KyKmVD2xF0maSzakf2tEjJG0DVmMEwoO7TWSvkhWwTsIOB04FrgoIs4uNLBViBOyJiVpHbJk4pGIuKPoeAAkDSYbglmW3m8NvB1YGBF/LjS4nPRD5rsRcXLRsdQiaRCwaUTcV3QsJc2U8ABI6g9sT3bTxlNFx1Mu/WNrZERcl/4M9Y2IxUXHVZIqjB+KiOeLjqVZSZoZEbtImg3slm7UmR0RY4qOLU/SQcDBZNXlqRFxbcEhrVI8qb9JSPqrpO3T62Fkw0XHAr+T9NlCg3vd1WR33JEmos8gm9x/gqRCqzp5EbEC2LXoOGqR9C6y4Y2r0/sx5cNvRYiIG1PS9fbS6/y+ouOTdJ6kUen1UOBu4ALgrjQE3DAkfRz4I/CLtGtj4IriIqroFWCupF9L+mlpKzqoEklbSfqHpHnp/WhJXys6rjKLJK1F9mt7raS/kA2tNoxUgb8pIk6KiC8CN0saUWxUqxZXyJqEpPmlW+ElfRXYJiI+nCZd3tIIc8gkzY2IHdLrbwNviohPSRoA3FE61ggknUmWLF4GvFTa32DDHHeQzTGZFhE7pX1zG+V7zC9zkts3p+jfi2V/Vj4LjI+IwyVtCPy99F02glQx2RW4rRF/jQEkfaTS/oj4bW/HUkmawH8S8IvcdzgvIrYvNrLKUoV5KHB1RCwtOp4SSbOAPUsxpb+3b4mIXYqNbNXhSf3NY1nu9VuAXwJExGJJbcWEtJJ8dn8AaeJvRCxtoBhLNiBLxPIVnQAaJiEjW27geekNSxgV/i8oSZ8APkk2YX5O7tAawD+LieoN8j/kDiJLukkT0ouJqH2vpj8fQLYANA3wa5zXKIlXFYMj4vayX9uGujEi3QhRUlrXa0OgYRYABvrlE8T0+3JAkQGtapyQNY9HJX0aWES2UnZpGGsQjXPn3ZxUeXoM2BK4BiCV6htKRHyo6Bg6YJ6kDwJ9JY0ETqQxEp6LgL+TTfzNz8Nb3CArjz+XJno/RrZm30fhtWRnUJGBVXBjqngPSvN3Pkm2Rl7DSL/3TmflhVcbZR2tp5Utoly6O/C9NN6dv3/j9QViB5KtmXYf0DALAAP/lXRoaZRA0mFAw9zktCrwkGWTUPZoldOAYcC5EVFKdvYHxkbEmUXGl2IZRHZr9zDg/Ii4O+3fE9giIn5XZHwplm1TLH9N788gGz4A+FlEzC4suDJpgvcpZJNsAaYC34mIV4qLamVqsMf+SNoK+ClZBeInETE57X8rcHBEfKHA8N5AUh+yhPG1idTArxppbTdJNwPfBH5MtvzK/yP72dEQ6/ZJ2hyYBOxJtnTNAuCoRr7zXNLOwHERcVzRsZSkpPZCsqdGCHgU+HBEPFhoYKsQJ2S2SkmTac+IiJvT+3uBU4HBwKERcUSB4TWVdNPBj2jgx/40snS3728j4uiiY6lG0h0RMbZsjuhNEbFPA8TWB3hvRPwhLSHSp5HuUK2m0hzMRqBsXTw1y/fYSjxkaauajUvJWPJiRFwKIOnYgmKqSNK1wPsiPbtS2Urfl0TEW4uN7DXfIVvg8rqI2ClVaxvqLsZGFhErJK0naUAjTe6u4JWU+Dwg6QSyoeD1C44JgIhoSzH9ISJeqnlCQSR9Pve2D9m0k/8WFM4bSDo6In5fFiOlOXkR8aNCAlsFOSGzVc0bHgVSdgdRQ/yQyVk3cg8Sj4hn0/Bgo2iGx/40uoXALWk5k/zdvo30Q/CzZBXkE8mevbk/2dMjGsW1yhY1vZQ3foeNMJ+xJP/3znKyOWWXFxRLudKK/H5MUsGckDUZSW8q/4tG0mYRsaC9c+wNnpA0LiJm5XdK2hV4sqCY2tMmadPSnCxlC4g20hyD0mN/ptNgj/3JD2UVHUsNj6etD6//QGyIX2NJA4H/396ZR0tWVWf89wEyyGQISFAGsUEGmQw0MhiVyTihEIcW0FZRFEWBkIBG1DjEIYCKiqKtBo0hIIiiIQZRBlGgQRBoZIpiouAQkRmZ4csf+9x+9Yr3Xjeups651P6tVavr3upa61v13ru17zl7f9/Ktrug7juB10taE2jJJLZb2T5g4JwJW5smsP1+SauU57fX1jOI7c+X7fPbbX+itp5xJnvIeoak84AXdH/Uinidk1ry3ClN1YcSPUULi37b1XPbJG1HTAl+iRKiC2xNNFbvZXt+LW3DSHo+0azcOd8/G3iT7e/WUzWB2o/9Odf2s2vreCSUImh32yc3oGUe4ZX1jaHz+wDPsv2WOsoWTUvbwMUL71Bi8EXE5OJ7bZ8oaR3b11cVWJB0tu2dausYZ7Ig6xmK8NzDgBcBGxEO5Ps0Nh14OfA54BLgwe6824l4+gti+6VrPr8S+LTt1kblu8zN7YgL+QVuKGtzkKLzpsamA99DFIwtb2V1zf3PI/rvngf8yPbL66oCSVfZ3nSa1xaa77aCoulpJyIIfXfba1aWhKT3Eca/b7P9i3LuqcAngR8B+9neoJ7CCSR9iLipGv57+cm0b0qWKFmQ9RBJexBF2crA39j+WWVJk+imsmrrmIkyIXi6S+5mq5RG/g2ZbCtxbj1FC1cZPwrcTPQUfRVYnVgpm2v79IryFiJpqm18t+KfJenZRPHwIiKgfUfgqbbvqiqsIOlq25s80tdGjaRnEp/jnsBqxNblt23fUlUYIOlnwObDVjXFIuhGYG83kg4i6ewpTruFnY1xIQuyniDp0zzcCf8XRFMwtg+sIGsSklYrTw8kbBC+Cdzbvd7SyoSkrxJbgGcBJxKTgg/O/K7RIumNhK/b2kSm5XbEKlnVC6QiYuVdxN30PGILfb6kjYET3FA0UatIuoFwaT8WOLUkbvyP7fUrS1uISiSR7YuGzs8GPlZ7O7is6LyS+BxPIK43Fzf2GV5re6NH+loynmRTf3+4eOi4ie2/IS5hwo0aom+io7Um29dIWo5YndgXmCfpv2zvX1naIAcBs4H5tncqBc/7K2uCiFjpjIk/0PXd2b5GDUQTSfqbmV4f7omqxCnAHsAc4MHij9fa3fGhwEmSvszE9WYbYC7wqlqiBngT4XZ/LHCa7XsktfYZ3iBpF9tnDp6UtDNhH1KdssI4D5gFXAHsa/vquqrGkyzIeoJLnlxppL6nW80p/SfL1dTW0dKd6eJg+97yRXg3sDRxt91SQXZP+ZJB0nKl4Gnhjnowl/Tuodda+ELcfYbXDFQvyGwfVJq9O++2I4FVJL0S+I7tO6sKBBz5kNsSW4CvK6evBJ5p+/fVhE3wF0z03h1dttxWkLSM7SamfYndgm+VtIPuhnU2sT39kprCBvgM8PfEtPRLgKOBVrwOx4rcsuwZkuYDu3YX7GI7cIbtHeoqm0DSAcS03aCh6V62P1tX2QSSdiXu8ncFziMaWU9vZTILQNI3iZiag4kt6luAx9l+4YxvfPR1PUg0/YrIhux6ngQsb7uVbNXeIOlxwPMpjf22V68sqVeU6dQXE5/fs4Azbe9dV1VQtO1NDBGJKGqPH+4rq8VwYkCrCQLjQBZkPUPSZba3WtS5mkyj8dKWeoskfZ3oHftP28OrPM0h6TlEz1ZTRWOLSPqy7deV56/tVpf7gqQV+vA72SrF72vPvv3cayHpF8QKWcdRg8eNbPGPBVmQ9YziQ/b2bhRZ0tbAMba3r6tsAkkLgC4kzT8AABIWSURBVC07C4SyrbqgwTH5NYjtAxPNwK1Emaw20+stDUe0yGDxn3f7STIzko6b4WXbbipS7rFM9pD1j4OBkyX9phyvRTQGt8QZRDPw54hiZ3+gCSuEDkmvIywbfkBsI3xO0nsbuaseHo4YpKnhiEbJu8xHAUkruuG8yORPw/bra2tIglwh6yGl32Qj4gv7mta8tEpszZuI/iwRBdoXW7KVkHQt4TZ+YzleHTgvx9D7T4lwOpH43ZtTni+kBYuYYVoudiTtAHwRWMn2upK2BN5s+62VpSVLkBKJ9WHgSbZfUFJgtrf9pcrSxoZcIesZkh4PHAKsZ3s/SRtK2sj2abW1wcLtya/YfjXh1t8qvwZuHTi+DbihkpZpKRYOzyJWfX5o+9TKkvrAoN3KsF1MUwwWO0Crxc4niKm7bwPYvryY2jZBSQb5GvA129fV1jOIpCuYesVWxHbgFiOWNBNfBo4DDi/H/018rlmQjYgsyPrHccSWVtczdgNwMtBEQWb7QUlrtJQlN4ikbnXkV8AFkk4lLph7AD+e9o0VkPRZYAPC9BJgf0m72T5ghrc96ki6g5m/ZFYZsaRJNLLtvLg0Xex02L5+yGOumdVuwqphDtEm8RBRRJxk+1d1ZQEx+dkXVrd9kqR/ALD9QJmoTkZEFmT9Y5btOZL2ArB9t1pw45zM/wLnSfo2kzPRPl5N0QRrlH+vL4/Ow+102us9eg6w2cBwxFcI48aq2F65tobHEo0XOwDXl5U8S1qW8NZqxjjU9i+BI4AjJG0IvAf4Z8JbsCpFW1/4o6Q/p1wHS0TabXUljRdZkPWP+0oOWvdHM4uBeKJG+E15LEXkbTaD7fdMdb580bxoxHIWxbXAukB3UV8HWFBPTpBToEuUpoudwv5EGPaTiRX5Mwiz2GaQ9BTC2HkOUdAeVlNPR+uryUMcQqzUzirT/GsA1UPux4ls6u8ZknYD3g1sSlwYdwReZ/ucmrqmQtLKxEWnuuv4VJThg10IM8kXABfa3qOuqglKluBsInia8vwCihGr7SpO34rQ7mmnQN1OePeOts9b1LmalGGSTzJ5AOYg2zdVFdYjJF0IPI5o3fia7V9UltRbJC3DxMDYta0NjD3WyYKsh5Rl5e2IP5r5tv9QWdIkJG0GfBXoVlL+AMy1fWU9VROUFYm9id6TS4nPclZrhWMxg50W2z8YlZY+MpUHWfqSPXIkfWqK07cR3n3fGrWeYSRtbPua2joWB0lPBJbvjlvoc+tJ9utYkFuWPaLcvbwA2LicuprJk4KtMA84xPbZAJKeC3wBqB7vJOmXxHbqPOBw27dJ+p/WijGIgkvSesCGtr9ftqqXsX1HbW0A0zWf2z531FoGkbQ98bu2hqRDBl5ahQb6igZpvdgpLE9cc04uxy8j4n/eIGkn2wfXECXp1bb/DXihpIfFiTXSswqApJcAHwOeBPweWI+4frdglt189uu4kAVZT5D0JOBs4LfEqo6ICZ6PlYvib2Z6/4hZsSvGAGyfowhFb4HTiAvQS4km1v+gvWZ+ACTtR/i5rQbMAtYmrER2qalrgEF7ieWBbYkJ4J3ryFnIsoSNxDJM7mG8nfZ6YposdobYANi5C+yWdCyxtbobdYdMumvKVH2qrf1Nf5BYif++7WdI6kLlq5PGsO2QW5Y9QdKXgctsHz10/kBga9uvrSJsChSh2D8hti0BXg1s00p/Vukd25US5EysnLyWyIm8a6b3jhJJlxFFzoUDUUBX2N68rrKpkbQOcITtJr5oJK3XTbmVn/lKtm+vLGsSks4iwsS7YmcZBood25vW1AcLTZS3tX1bOV6V+J3cWI1l1HZIOnj4WlkTSRfb3qZ4pj3D9kOSLrK9bW1tg0h6EbFqN7it+oF6isaLpWoLSBab7aa6wNj+FHHn1RL7EhM63yiP1YFm7sJsP2T7jHJnuD4wF3gV4U3WEvcOermVL+uW76BuADarLWKAj0hapazOXgVcK+nQRb1pxDyZiZUeyvMnlVSLVqanjwAuk3RcuTG8FDiqfK7fr6pseg5Z9H8ZKbdKWgk4Fzhe0ieBByprmkSJupsDvJ3YgXkFsbWajIhcIesJM92JtnSXqgjsXg/4ue0W+9umpbX4GklHED2Cc4mL5FuBq2wfPuMbR4SkTzNRIC4FbAX8b0lpqI6ky2xvJWkfYGvgHcAlLbmjS3oDMTV9DvEl+GwivuYE4H22myggJa1FrNYKuKixFomHIel62+vU1tFRite7ib+TfYBVgeNbmqaVtMD2FgP/rgR8w/bzamsbF7KHrD+sOs00jIgtt+pIeiPxZXIdsL6kN9n+dmVZi01LxVjhncAbiD6dNwPfIWJ2WmEwlugB4ISWLCWAxylyX/cAjrF9v6Sm7kBtf0nSd5godt41UOw0UYwV7iH6V5cHNpC0Qe3hjUXQzM9ZESf3Ldu7Ag8BrSZJ3F3+vav0LN9M7CAkIyILsv7wA6afhmnlwngw8HTbN0p6KnA8JRImeWRociboF2rrGUTSurZ/1YOIos8TqRGXA+eWidWmesgKTRc75UbrIGKo5DKiReICKg9vLMJ0dYURy5mWEid3l6RVuz68RjlN0hOILepLyrmWbgAf8+SWZbLEGPZ46ovnkyK35vGtrZBJ+i6wuxvLBB38uUo6xfbLamtaHMrPeemugb4Fpit2bNeeVF2IIiB7NuF5uJWkjYH3255TWVpvkHQS8bP9HpPj5A6c9k0jQtJs4HrbvyvHc4lBrGuIbfNM3hgRuUKWLEnWHvJVmnTcwsWnQ9K/Am8jttouBlaX9NGWvItoNxN00KG/CVf+xcG2Jb0GOK62lgEOYqLY2akrdiprGuYe2/dIQtJytq+RtFFtUT3jP8ujRT5PTJ133oIfJXpWtyL8GluzinnMkgVZsiQZ7nm5ZMr/1Qab275d0t6EzcBhRGFWu9gZpNVMUE/zvA+8n7YKsj4UOzeUraxTge9JuoX4vUwWE9tfKcbO69q+traeIZYeWAWbA8yzfQpwSrHeSUZEFmTJEqMHPUWDLFtsJF4KHGv7PkkP1RY1iO3WVko6tpR0O6VXpzyHRgKTJU0XwC5gzVFqWQyaL3Zs71mevk/S2cSE4OkVJfUOSbsDRxGmxetL2gr4gCvl0Q6xtKRlylb+LoQZdUfWCCMkP+yeMM2E5UIyb+wR80XCd+ynwA8krQs0EZ8k6WjbB0+XIlD7Im67qfihKVgT+GvglqHzAs4fvZzpab3YKYa6C2xvBu3lp5bhl++WCcaWeR8xSXsOgO3LJLUywXgCcQ38AzFp+UMASRsQMV7JiMiCrD90E5ZPJHL6zirHOxF/5FmQPQJsfwL4RHcs6XrqR/50dAkHR1VV0V9OI1z5H7bdIumc0cuZmtaLHQgTZUmXd5O1tfUM06MJxgccubmD55rY7rf9IUlnAmsBZ3hi0m8popcsGRFZkPWELm9M0mnAprZ/W47XAj5TU1sfkbQKMUn0FCb/HbTg8H0jtPkF3Qdsv2GG1/YepZaZaL3YGWAt4EpJFzF5uKSF7TYI25ArJDU3wTjAT0u/6tKSNgQOpKHVWtvzpzj33zW0jDNZkPWPp3TFWOH/gKfVEjMVxa1/P4aKHdv71tI0Bd8h8javIMwaW+JUoHe2EsmfROvFDrQ39TlMyxOMHW8HDifisP4d+C7wT1UVJc2RPmQ9Q9IxwIbEvr+JDMaf225maVnS+UQfwiXAg935MrnTBC17pA1GYbUUi5UseSQ9Z6rzuTr62ELSM2xfWltH0jZZkPUQSXsSmXcA59r+Zk09w3QZgrV1zISkvwduIvqNFoY4267u5D5kvNps4ZiMB5K2Az4NbEJMCS4N/LGBadqXAmvb/kw5vhBYo7x8mO2vVxM3RBnYWAs4GTjR9pWVJSUNslRtAcmfxPlEU/+ZQEvZgR2nSXphbRGL4E7gaOBS4Mry+GlVRRNsKen2Eg2zRXl+u6Q7BiwmkscAkraT9GNJd0q6T9KDDf6MjwH2An5GRBK9sZyrzWFMjmZbjjDZfS7wlhqCpsP2ToSuG4F5kq6Q9O66qpLWyBWyniHplcCRxGSlgL8CDm3sbvAOYEXgvvJowp9qEEnXAdvb/n1tLcn4Iuliou3gZGAbYC6woe13VRU2gKSLbW8jaYHtLcq5823vUFnXj23PHjg+xvbbyvP5trerp256JG1OFJNzbC9bW0/SDtnU3z8OB2Z3hURpoP8+0ExBZrslV/npuIo2g6aTMcP2zyUtbftB4LjSg9kSd0laFrhM0hFEEPqKlTUB/NngQVeMFdagISRtQrjgvwL4A3Ai8HdVRSXNkQVZ/1hqaFXnJhrbei4hzvsA69v+oKR1gLVsX1RZ2iD3AZdKOovJPWQt2F4k40Orxc4gryGuMW8D/hZYB2hh8vdCSfvZ/sLgSUlvBlq61kDEdZ0A7Ga7qSSGpB1yy7JnSDoS2IL444a461pg+x31VE1G0rGElcTOtjeR9GeE4eDsRbx1ZEia0qvK9pdGrSUZXyStR1jXLEsUO6sCn7X986rCeoCkJxIWMfcSFjYAWxO9ZHvY/r9a2oYpOZaziMn462zfU1lS0iBZkPWQEqP0LKI3q8Upy5/Y/ssh+4bLbW9ZW9tMSHqm7Qtr60iSFujLFKOknYGnl8MrbZ810/8fJSUv98PA64motqWAtYkVs8Nt319RXtIYuWXZT84D7ifutlpbmge4v2TMGRb2uTVhvlrial4GPJnIwLta0vOBdxE9KZvX1JeMBz0pdg4jBg46uinGFYmCogWNlAKsmSJsiCOBlYGn2r4DFqaEHFUeB1XUljRGU71HyaIpU5YXAS8HXkn0Uby8rqqH8Sngm8Cakj4E/Ii4S2yBLwIHEAXZsZK+QHgsfcp2FmPJqOiDZcOytq8fOP6R7ZtKzFNrfW6t8mJgv64Yg4Veh28BWrcGSkZMrpD1jz5MWR4v6RJgl3JqD9tX19Q0wDOBLUoo8QrExNMGQ3FUSfJoM2WxA9wkqZVipzdTjA1jT9EXVK4/2S+UTCJXyPpH81OWhccTjt5LEWaSrXBvsRfA9t3AtVmMJRXoQ7FzoaT9hk82OsXYKldJmjt8UtKrgWsq6EkaJpv6e0ZPpizfS/jtnEIMHuwBnGy7epiupLuYuBAK2Kgcd+a1GVOUPOpIOh44ZxrLhufa3quOsklaejPF2CqSngx8A7ibyPY1sTW9ArCn7V9XlJc0RhZkPUTSy4AdaXfK8mrgGd1od9ka/IntTeoqA0mzZnrd9nWj0pKML30qdlqeYuwLA5+hiM/wzMqSkgbJgqwnSDqYmK681PYDtfXMhKT/AvayfWs5fgLwb7ZfXFfZBJI+PBxPM9W5JHk0yWInSZKOLMh6gqSjgB2AjYEFRMD4ecAFtm+uqW0YSacSy/LfI5bodyMmLX8PYPvAeuqCzitt6FzzXmlJkiTJY5MsyHpGiVnZhijOti+PW21vWlXYAJJeO9Prtr8yKi3DlB6d/YGnAdcOvLQycHELvTtJkiTJ+JG2F/1jBWAVImJlVeA3wBVVFT2crwEb0GZMyEnAmcBHgHcOnL9jaHo1SZIkSUZGrpD1BEnziF6TO4ALgfnAfNu3VBU2wEBMyL7AL2k8JkTSZkQEFcAPbV9ZU0+SJEkyvrToX5VMzbrEBNbvgF8DNwC3VlX0cI4EVgPWt711ybGcBTyBiAlpBkkHEKtl65bHSZLeWldVkiRJMq7kClmPkCRilWyH8tgMuJlo7P/HmtoAJP0MeNqwM3XJtbzG9oZ1lD0cSQuAHWzfWY5XAs63vUVdZUmSJMk4kj1kPaIUOj+VdCtwW3m8GNgWqF6Q0a+YEBEB7R33l3NJkiRJMnKyIOsJkg4kVsV2JIqH84ALgH+hnab+qyTNtf2vgydbigmRtEzxcfsqMF/SKeWlPYFq059JkiTJeJNblj1B0scp3mOtZi/2ISZk0H9M0mzgr5hIPPhxVXFJkiTJ2JIFWbLEaTkmRNKlZdggSZIkSZohC7JkrJB0A/Dx6V63Pe1rSZIkSfJokT1kybixNLAS2cCfJEmSNESukCVjxVQZlkmSJElSmzSGTcaNXBlLkiRJmiNXyJKxQtJqtm+urSNJkiRJBsmCLEmSJEmSpDK5ZZkkSZIkSVKZLMiSJEmSJEkqkwVZkiRJkiRJZbIgS5IkSZIkqUwWZEmSJEmSJJX5f029rRfn6rFkAAAAAElFTkSuQmCC
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="Train-and-Test-the-model">Train and Test the model<a class="anchor-link" href="#Train-and-Test-the-model"></a></h2><p>Based on the correlation matrix heatmap, we can tell that the following pairs of columns are strongly correlated:</p>
<ul>
<li>Gr Liv Area and TotRms AbvGrd</li>
<li>Garage Area and Garage Cars</li>
</ul>
<p>We will only use one of these pairs and remove any columns with missing values</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[11]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">final_corr_cols</span> <span class="o">=</span> <span class="n">strong_corrs</span><span class="o">.</span><span class="n">drop</span><span class="p">([</span><span class="s1">&#39;Garage Cars&#39;</span><span class="p">,</span> <span class="s1">&#39;TotRms AbvGrd&#39;</span><span class="p">])</span>
<span class="n">features</span> <span class="o">=</span> <span class="n">final_corr_cols</span><span class="o">.</span><span class="n">drop</span><span class="p">([</span><span class="s1">&#39;SalePrice&#39;</span><span class="p">])</span><span class="o">.</span><span class="n">index</span>
<span class="n">target</span> <span class="o">=</span> <span class="s1">&#39;SalePrice&#39;</span>
<span class="n">clean_test</span> <span class="o">=</span> <span class="n">test</span><span class="p">[</span><span class="n">final_corr_cols</span><span class="o">.</span><span class="n">index</span><span class="p">]</span><span class="o">.</span><span class="n">dropna</span><span class="p">()</span>

<span class="n">lr</span> <span class="o">=</span> <span class="n">LinearRegression</span><span class="p">()</span>
<span class="n">lr</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">train</span><span class="p">[</span><span class="n">features</span><span class="p">],</span> <span class="n">train</span><span class="p">[</span><span class="s1">&#39;SalePrice&#39;</span><span class="p">])</span>

<span class="n">train_predictions</span> <span class="o">=</span> <span class="n">lr</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">train</span><span class="p">[</span><span class="n">features</span><span class="p">])</span>
<span class="n">test_predictions</span> <span class="o">=</span> <span class="n">lr</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">clean_test</span><span class="p">[</span><span class="n">features</span><span class="p">])</span>

<span class="n">train_mse</span> <span class="o">=</span> <span class="n">mean_squared_error</span><span class="p">(</span><span class="n">train_predictions</span><span class="p">,</span> <span class="n">train</span><span class="p">[</span><span class="n">target</span><span class="p">])</span>
<span class="n">test_mse</span> <span class="o">=</span> <span class="n">mean_squared_error</span><span class="p">(</span><span class="n">test_predictions</span><span class="p">,</span> <span class="n">clean_test</span><span class="p">[</span><span class="n">target</span><span class="p">])</span>

<span class="n">train_rmse</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">sqrt</span><span class="p">(</span><span class="n">train_mse</span><span class="p">)</span>
<span class="n">test_rmse</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">sqrt</span><span class="p">(</span><span class="n">test_mse</span><span class="p">)</span>

<span class="nb">print</span><span class="p">(</span><span class="n">train_rmse</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">test_rmse</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>34173.97629185851
41032.026120197705
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="Removing-low-variance-features">Removing low variance features<a class="anchor-link" href="#Removing-low-variance-features"></a></h2><p>The last technique I will explore is removing features with low variance. When the values in a feature column have low variance, they don't meaningfully contribute to the model's predictive capability. On the extreme end, let's imagine a column with a variance of 0. This would mean that all of the values in that column were exactly the same. This means that the column isn't informative and isn't going to help the model make better predictions.</p>
<p>To make apples to apples comparisions between columns, we need to standardize all of the columns to vary between 0 and 1. Then, we can set a cutoff value for variance and remove features that have less than that variance amount.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[12]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">unit_train</span> <span class="o">=</span> <span class="n">train</span><span class="p">[</span><span class="n">features</span><span class="p">]</span><span class="o">/</span><span class="p">(</span><span class="n">train</span><span class="p">[</span><span class="n">features</span><span class="p">]</span><span class="o">.</span><span class="n">max</span><span class="p">())</span>
<span class="n">sorted_vars</span> <span class="o">=</span> <span class="n">unit_train</span><span class="o">.</span><span class="n">var</span><span class="p">()</span><span class="o">.</span><span class="n">sort_values</span><span class="p">()</span>
<span class="nb">print</span><span class="p">(</span><span class="n">sorted_vars</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>Open Porch SF    0.013938
Gr Liv Area      0.018014
Full Bath        0.018621
1st Flr SF       0.019182
Overall Qual     0.019842
Garage Area      0.020347
Wood Deck SF     0.033064
Fireplaces       0.046589
dtype: float64
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="Final-Model">Final Model<a class="anchor-link" href="#Final-Model"></a></h2><p>Let's set a cutoff variance of 0.015, remove the Open Porch SF feature, and train and test a model using the remaining features.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[13]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">features</span> <span class="o">=</span> <span class="n">features</span><span class="o">.</span><span class="n">drop</span><span class="p">([</span><span class="s1">&#39;Open Porch SF&#39;</span><span class="p">])</span>

<span class="n">clean_test</span> <span class="o">=</span> <span class="n">test</span><span class="p">[</span><span class="n">final_corr_cols</span><span class="o">.</span><span class="n">index</span><span class="p">]</span><span class="o">.</span><span class="n">dropna</span><span class="p">()</span>

<span class="n">lr</span> <span class="o">=</span> <span class="n">LinearRegression</span><span class="p">()</span>
<span class="n">lr</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">train</span><span class="p">[</span><span class="n">features</span><span class="p">],</span> <span class="n">train</span><span class="p">[</span><span class="s1">&#39;SalePrice&#39;</span><span class="p">])</span>

<span class="n">train_predictions</span> <span class="o">=</span> <span class="n">lr</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">train</span><span class="p">[</span><span class="n">features</span><span class="p">])</span>
<span class="n">test_predictions</span> <span class="o">=</span> <span class="n">lr</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">clean_test</span><span class="p">[</span><span class="n">features</span><span class="p">])</span>

<span class="n">train_mse</span> <span class="o">=</span> <span class="n">mean_squared_error</span><span class="p">(</span><span class="n">train_predictions</span><span class="p">,</span> <span class="n">train</span><span class="p">[</span><span class="n">target</span><span class="p">])</span>
<span class="n">test_mse</span> <span class="o">=</span> <span class="n">mean_squared_error</span><span class="p">(</span><span class="n">test_predictions</span><span class="p">,</span> <span class="n">clean_test</span><span class="p">[</span><span class="n">target</span><span class="p">])</span>

<span class="n">train_rmse_2</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">sqrt</span><span class="p">(</span><span class="n">train_mse</span><span class="p">)</span>
<span class="n">test_rmse_2</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">sqrt</span><span class="p">(</span><span class="n">test_mse</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">lr</span><span class="o">.</span><span class="n">intercept_</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">lr</span><span class="o">.</span><span class="n">coef_</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">train_rmse_2</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">test_rmse_2</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>-112764.87061464708
[   37.88152677  7086.98429942 -2221.97281278    43.18536387
    64.88085639    38.71125489 24553.18365123]
34372.696707783965
40591.42702437726
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>The final model will be a 7-dimension linear function which looks like:
$$
Y = -112765 + 37.9 * Wood Deck + 7087 * Fire Places - 2222 * Full Bath + 43 * 1st Fle SF + 65 * garage Area + 39 * Liv area + 24553 * Overall Qual
$$</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="Feature-transformation">Feature transformation<a class="anchor-link" href="#Feature-transformation"></a></h2><p>To understand how linear regression works, I have stuck to using features from the training dataset that contained no missing values and were already in a convenient numeric representation. In this mission, we'll explore how to transform some of the the remaining features so we can use them in our model. Broadly, the process of processing and creating new features is known as feature engineering.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[14]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">train</span> <span class="o">=</span> <span class="n">data</span><span class="p">[</span><span class="mi">0</span><span class="p">:</span><span class="mi">1460</span><span class="p">]</span>
<span class="n">test</span> <span class="o">=</span> <span class="n">data</span><span class="p">[</span><span class="mi">1460</span><span class="p">:]</span>
<span class="n">train_null_counts</span> <span class="o">=</span> <span class="n">train</span><span class="o">.</span><span class="n">isnull</span><span class="p">()</span><span class="o">.</span><span class="n">sum</span><span class="p">()</span>
<span class="n">df_no_mv</span> <span class="o">=</span> <span class="n">train</span><span class="p">[</span><span class="n">train_null_counts</span><span class="p">[</span><span class="n">train_null_counts</span><span class="o">==</span><span class="mi">0</span><span class="p">]</span><span class="o">.</span><span class="n">index</span><span class="p">]</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="Categorical-Features">Categorical Features<a class="anchor-link" href="#Categorical-Features"></a></h2><p>You'll notice that some of the columns in the data frame df_no_mv contain string values. To use these features in our model, we need to transform them into numerical representations</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[15]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">text_cols</span> <span class="o">=</span> <span class="n">df_no_mv</span><span class="o">.</span><span class="n">select_dtypes</span><span class="p">(</span><span class="n">include</span><span class="o">=</span><span class="p">[</span><span class="s1">&#39;object&#39;</span><span class="p">])</span><span class="o">.</span><span class="n">columns</span>

<span class="k">for</span> <span class="n">col</span> <span class="ow">in</span> <span class="n">text_cols</span><span class="p">:</span>
    <span class="nb">print</span><span class="p">(</span><span class="n">col</span><span class="o">+</span><span class="s2">&quot;:&quot;</span><span class="p">,</span> <span class="nb">len</span><span class="p">(</span><span class="n">train</span><span class="p">[</span><span class="n">col</span><span class="p">]</span><span class="o">.</span><span class="n">unique</span><span class="p">()))</span>
<span class="k">for</span> <span class="n">col</span> <span class="ow">in</span> <span class="n">text_cols</span><span class="p">:</span>
    <span class="n">train</span><span class="p">[</span><span class="n">col</span><span class="p">]</span> <span class="o">=</span> <span class="n">train</span><span class="p">[</span><span class="n">col</span><span class="p">]</span><span class="o">.</span><span class="n">astype</span><span class="p">(</span><span class="s1">&#39;category&#39;</span><span class="p">)</span>
<span class="n">train</span><span class="p">[</span><span class="s1">&#39;Utilities&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">cat</span><span class="o">.</span><span class="n">codes</span><span class="o">.</span><span class="n">value_counts</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>(&#39;MS Zoning:&#39;, 6)
(&#39;Street:&#39;, 2)
(&#39;Lot Shape:&#39;, 4)
(&#39;Land Contour:&#39;, 4)
(&#39;Utilities:&#39;, 3)
(&#39;Lot Config:&#39;, 5)
(&#39;Land Slope:&#39;, 3)
(&#39;Neighborhood:&#39;, 26)
(&#39;Condition 1:&#39;, 9)
(&#39;Condition 2:&#39;, 6)
(&#39;Bldg Type:&#39;, 5)
(&#39;House Style:&#39;, 8)
(&#39;Roof Style:&#39;, 6)
(&#39;Roof Matl:&#39;, 5)
(&#39;Exterior 1st:&#39;, 14)
(&#39;Exterior 2nd:&#39;, 16)
(&#39;Exter Qual:&#39;, 4)
(&#39;Exter Cond:&#39;, 5)
(&#39;Foundation:&#39;, 6)
(&#39;Heating:&#39;, 6)
(&#39;Heating QC:&#39;, 4)
(&#39;Central Air:&#39;, 2)
(&#39;Electrical:&#39;, 4)
(&#39;Kitchen Qual:&#39;, 5)
(&#39;Functional:&#39;, 7)
(&#39;Paved Drive:&#39;, 3)
(&#39;Sale Type:&#39;, 9)
(&#39;Sale Condition:&#39;, 5)
</pre>
</div>
</div>

<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stderr output_text">
<pre>C:\Users\dongl4\Anaconda2\lib\site-packages\ipykernel_launcher.py:6: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
  
</pre>
</div>
</div>

<div class="output_area">

    <div class="prompt output_prompt">Out[15]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>0    1457
2       2
1       1
dtype: int64</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="Dummy-Coding">Dummy Coding<a class="anchor-link" href="#Dummy-Coding"></a></h2><p>When we convert a column to the categorical data type, pandas assigns a number from 0 to n-1 (where n is the number of unique values in a column) for each value. The drawback with this approach is that one of the assumptions of linear regression is violated here. Linear regression operates under the assumption that the features are linearly correlated with the target column. For a categorical feature, however, there's no actual numerical meaning to the categorical codes that pandas assigned for that colum. An increase in the Utilities column from 1 to 2 has no correlation value with the target column, and the categorical codes are instead used for uniqueness and exclusivity (the category associated with 0 is different than the one associated with 1).</p>
<p>The common solution is to use a technique called dummy coding</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[16]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">dummy_cols</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">DataFrame</span><span class="p">()</span>
<span class="k">for</span> <span class="n">col</span> <span class="ow">in</span> <span class="n">text_cols</span><span class="p">:</span>
    <span class="n">col_dummies</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">get_dummies</span><span class="p">(</span><span class="n">train</span><span class="p">[</span><span class="n">col</span><span class="p">])</span>
    <span class="n">train</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">concat</span><span class="p">([</span><span class="n">train</span><span class="p">,</span> <span class="n">col_dummies</span><span class="p">],</span> <span class="n">axis</span><span class="o">=</span><span class="mi">1</span><span class="p">)</span>
    <span class="k">del</span> <span class="n">train</span><span class="p">[</span><span class="n">col</span><span class="p">]</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[17]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">train</span><span class="p">[</span><span class="s1">&#39;years_until_remod&#39;</span><span class="p">]</span> <span class="o">=</span> <span class="n">train</span><span class="p">[</span><span class="s1">&#39;Year Remod/Add&#39;</span><span class="p">]</span> <span class="o">-</span> <span class="n">train</span><span class="p">[</span><span class="s1">&#39;Year Built&#39;</span><span class="p">]</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="Missing-Values">Missing Values<a class="anchor-link" href="#Missing-Values"></a></h2><p>Now I will focus on handling columns with missing values. When values are missing in a column, there are two main approaches we can take:</p>
<ul>
<li>Remove rows containing missing values for specific columns
Pro: Rows containing missing values are removed, leaving only clean data for modeling
Con: Entire observations from the training set are removed, which can reduce overall prediction accuracy</li>
<li>Impute (or replace) missing values using a descriptive statistic from the column
Pro: Missing values are replaced with potentially similar estimates, preserving the rest of the observation in the model.
Con: Depending on the approach, we may be adding noisy data for the model to learn</li>
</ul>
<p>Given that we only have 1460 training examples (with ~80 potentially useful features), we don't want to remove any of these rows from the dataset. Let's instead focus on imputation techniques.</p>
<p>We'll focus on columns that contain at least 1 missing value but less than 365 missing values (or 25% of the number of rows in the training set). There's no strict threshold, and many people instead use a 50% cutoff (if half the values in a column are missing, it's automatically dropped). Having some domain knowledge can help with determining an acceptable cutoff value.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[18]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">df_missing_values</span> <span class="o">=</span> <span class="n">train</span><span class="p">[</span><span class="n">train_null_counts</span><span class="p">[(</span><span class="n">train_null_counts</span><span class="o">&gt;</span><span class="mi">0</span><span class="p">)</span> <span class="o">&amp;</span> <span class="p">(</span><span class="n">train_null_counts</span><span class="o">&lt;</span><span class="mi">584</span><span class="p">)]</span><span class="o">.</span><span class="n">index</span><span class="p">]</span>

<span class="nb">print</span><span class="p">(</span><span class="n">df_missing_values</span><span class="o">.</span><span class="n">isnull</span><span class="p">()</span><span class="o">.</span><span class="n">sum</span><span class="p">())</span>
<span class="nb">print</span><span class="p">(</span><span class="n">df_missing_values</span><span class="o">.</span><span class="n">dtypes</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>Lot Frontage      249
Mas Vnr Type       11
Mas Vnr Area       11
Bsmt Qual          40
Bsmt Cond          40
Bsmt Exposure      41
BsmtFin Type 1     40
BsmtFin SF 1        1
BsmtFin Type 2     41
BsmtFin SF 2        1
Bsmt Unf SF         1
Total Bsmt SF       1
Bsmt Full Bath      1
Bsmt Half Bath      1
Garage Type        74
Garage Yr Blt      75
Garage Finish      75
Garage Qual        75
Garage Cond        75
dtype: int64
Lot Frontage      float64
Mas Vnr Type       object
Mas Vnr Area      float64
Bsmt Qual          object
Bsmt Cond          object
Bsmt Exposure      object
BsmtFin Type 1     object
BsmtFin SF 1      float64
BsmtFin Type 2     object
BsmtFin SF 2      float64
Bsmt Unf SF       float64
Total Bsmt SF     float64
Bsmt Full Bath    float64
Bsmt Half Bath    float64
Garage Type        object
Garage Yr Blt     float64
Garage Finish      object
Garage Qual        object
Garage Cond        object
dtype: object
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="Inputing-missing-values">Inputing missing values<a class="anchor-link" href="#Inputing-missing-values"></a></h2><p>It looks like about half of the columns in df_missing_values are string columns (object data type), while about half are float64 columns. For numerical columns with missing values, a common strategy is to compute the mean, median, or mode of each column and replace all missing values in that column with that value</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[19]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">float_cols</span> <span class="o">=</span> <span class="n">df_missing_values</span><span class="o">.</span><span class="n">select_dtypes</span><span class="p">(</span><span class="n">include</span><span class="o">=</span><span class="p">[</span><span class="s1">&#39;float&#39;</span><span class="p">])</span>
<span class="n">float_cols</span> <span class="o">=</span> <span class="n">float_cols</span><span class="o">.</span><span class="n">fillna</span><span class="p">(</span><span class="n">df_missing_values</span><span class="o">.</span><span class="n">mean</span><span class="p">())</span>
<span class="nb">print</span><span class="p">(</span><span class="n">float_cols</span><span class="o">.</span><span class="n">isnull</span><span class="p">()</span><span class="o">.</span><span class="n">sum</span><span class="p">())</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>Lot Frontage      0
Mas Vnr Area      0
BsmtFin SF 1      0
BsmtFin SF 2      0
Bsmt Unf SF       0
Total Bsmt SF     0
Bsmt Full Bath    0
Bsmt Half Bath    0
Garage Yr Blt     0
dtype: int64
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="Conclusion">Conclusion<a class="anchor-link" href="#Conclusion"></a></h2><hr>
<p>This note book talks about how to do linear regression in machine learning by analysing the real example -- Boston housing data. In this case, to do the linear regression not only means we need to figure out the correlation among all the variable, but also eliminate the variable with either insignificant influence or missing value.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p><a href="https://colab.research.google.com/github/RPI-DATA/tutorials-intro/blob/master/website/linearregression.ipynb" target="_blank"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"> </a></p>

</div>
</div>
</div>
 

