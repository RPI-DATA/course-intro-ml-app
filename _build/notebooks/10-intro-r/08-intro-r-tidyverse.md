---
interact_link: content/notebooks/10-intro-r/08-intro-r-tidyverse.ipynb
kernel_name: ir
has_widgets: false
title: 'Tidyvere'
prev_page:
  url: /notebooks/10-intro-r/07-intro-r-merge-agg-fun.html
  title: 'Aggregation and Merge'
next_page:
  url: /notebooks/10-intro-r/09-titanic-intro.html
  title: 'Titanic'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---


[![AnalyticsDojo](https://github.com/rpi-techfundamentals/spring2019-materials/blob/master/fig/final-logo.png?raw=1)](http://rpi.analyticsdojo.com)
<center><h1>Introduction to R - Tidyverse </h1></center>
<center><h3><a href = 'http://rpi.analyticsdojo.com'>rpi.analyticsdojo.com</a></h3></center>



## Overview

> It is often said that 80% of data analysis is spent on the process of cleaning and preparing the data. (Dasu and Johnson, 2003)

Thus before you can even get to doing any sort of sophisticated analysis or plotting, you'll generally first need to: 

1. ***Manipulating*** data frames, e.g. filtering, summarizing, and conducting calculations across groups.
2. ***Tidying*** data into the appropriate format





# What is the Tidyverse?



## Tidyverse
- "The tidyverse is a set of packages that work in harmony because they share common data representations and API design." -Hadley Wickham
- The variety of packages include `dplyr`, `tibble`, `tidyr`, `readr`, `purrr` (and more).




![](http://r4ds.had.co.nz/diagrams/data-science-explore.png)
- From [R for Data Science](http://r4ds.had.co.nz/explore-intro.html) by [Hadley Wickham](https://github.com/hadley)



## Schools of Thought

There are two competing schools of thought within the R community.

* We should stick to the base R functions to do manipulating and tidying; `tidyverse` uses syntax that's unlike base R and is superfluous.
* We should start teaching students to manipulate data using `tidyverse` tools because they are straightfoward to use, more readable than base R, and speed up the tidying process.

We'll show you some of the `tidyverse` tools so you can make an informed decision about whether you want to use base R or these newfangled packages.



## Dataframe Manipulation using Base R Functions

- So far, you’ve seen the basics of manipulating data frames, e.g. subsetting, merging, and basic calculations. 
- For instance, we can use base R functions to calculate summary statistics across groups of observations,
- e.g. the mean GDP per capita within each region:




<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
gapminder <- read.csv("../../input/gapminder-FiveYearData.csv",
          stringsAsFactors = TRUE)
head(gapminder)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<table>
<thead><tr><th scope=col>country</th><th scope=col>year</th><th scope=col>pop</th><th scope=col>continent</th><th scope=col>lifeExp</th><th scope=col>gdpPercap</th></tr></thead>
<tbody>
	<tr><td>Afghanistan</td><td>1952       </td><td> 8425333   </td><td>Asia       </td><td>28.801     </td><td>779.4453   </td></tr>
	<tr><td>Afghanistan</td><td>1957       </td><td> 9240934   </td><td>Asia       </td><td>30.332     </td><td>820.8530   </td></tr>
	<tr><td>Afghanistan</td><td>1962       </td><td>10267083   </td><td>Asia       </td><td>31.997     </td><td>853.1007   </td></tr>
	<tr><td>Afghanistan</td><td>1967       </td><td>11537966   </td><td>Asia       </td><td>34.020     </td><td>836.1971   </td></tr>
	<tr><td>Afghanistan</td><td>1972       </td><td>13079460   </td><td>Asia       </td><td>36.088     </td><td>739.9811   </td></tr>
	<tr><td>Afghanistan</td><td>1977       </td><td>14880372   </td><td>Asia       </td><td>38.438     </td><td>786.1134   </td></tr>
</tbody>
</table>

</div>

</div>
</div>
</div>



## But this isn't ideal because it involves a fair bit of repetition. Repeating yourself will cost you time, both now and later, and potentially introduce some nasty bugs.




# Dataframe Manipulation using dplyr



Here we're going to cover 6 of the most commonly used functions as well as using pipes (`%>%`) to combine them.

1. `select()`
2. `filter()`
3. `group_by()`
4. `summarize()`
5. `mutate()`
6. `arrange()`

If you have have not installed this package earlier, please do so now:


```r
install.packages('dplyr')
```



## Dataframe Manipulation using `dplyr`

Luckily, the [`dplyr`](https://cran.r-project.org/web/packages/dplyr/dplyr.pdf) package provides a number of very useful functions for manipulating dataframes. These functions will save you time by reducing repetition. As an added bonus, you might even find the `dplyr` grammar easier to read.

- ["A fast, consistent tool for working with data frame like objects, both in memory and out of memory."](https://cran.r-project.org/web/packages/dplyr/index.html)
- Subset observations using their value with `filter()`.
- Reorder rows using `arrange()`.
- Select columns using  `select()`.
- Recode variables useing `mutate()`.
- Sumarize variables using `summarise()`.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
#Now lets load some packages:
library(dplyr)
library(ggplot2)
library(tidyverse)

```
</div>

</div>



# dplyr select

Imagine that we just received the gapminder dataset, but are only interested in a few variables in it. We could use the `select()` function to keep only the columns corresponding to variables we select.




<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
year_country_gdp <-gapminder[,c("year","country")] 
year_country_gdp

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<table>
<thead><tr><th scope=col>year</th><th scope=col>country</th></tr></thead>
<tbody>
	<tr><td>1952       </td><td>Afghanistan</td></tr>
	<tr><td>1957       </td><td>Afghanistan</td></tr>
	<tr><td>1962       </td><td>Afghanistan</td></tr>
	<tr><td>1967       </td><td>Afghanistan</td></tr>
	<tr><td>1972       </td><td>Afghanistan</td></tr>
	<tr><td>1977       </td><td>Afghanistan</td></tr>
	<tr><td>1982       </td><td>Afghanistan</td></tr>
	<tr><td>1987       </td><td>Afghanistan</td></tr>
	<tr><td>1992       </td><td>Afghanistan</td></tr>
	<tr><td>1997       </td><td>Afghanistan</td></tr>
	<tr><td>2002       </td><td>Afghanistan</td></tr>
	<tr><td>2007       </td><td>Afghanistan</td></tr>
	<tr><td>1952       </td><td>Albania    </td></tr>
	<tr><td>1957       </td><td>Albania    </td></tr>
	<tr><td>1962       </td><td>Albania    </td></tr>
	<tr><td>1967       </td><td>Albania    </td></tr>
	<tr><td>1972       </td><td>Albania    </td></tr>
	<tr><td>1977       </td><td>Albania    </td></tr>
	<tr><td>1982       </td><td>Albania    </td></tr>
	<tr><td>1987       </td><td>Albania    </td></tr>
	<tr><td>1992       </td><td>Albania    </td></tr>
	<tr><td>1997       </td><td>Albania    </td></tr>
	<tr><td>2002       </td><td>Albania    </td></tr>
	<tr><td>2007       </td><td>Albania    </td></tr>
	<tr><td>1952       </td><td>Algeria    </td></tr>
	<tr><td>1957       </td><td>Algeria    </td></tr>
	<tr><td>1962       </td><td>Algeria    </td></tr>
	<tr><td>1967       </td><td>Algeria    </td></tr>
	<tr><td>1972       </td><td>Algeria    </td></tr>
	<tr><td>1977       </td><td>Algeria    </td></tr>
	<tr><td>⋮</td><td>⋮</td></tr>
	<tr><td>1982      </td><td>Yemen Rep.</td></tr>
	<tr><td>1987      </td><td>Yemen Rep.</td></tr>
	<tr><td>1992      </td><td>Yemen Rep.</td></tr>
	<tr><td>1997      </td><td>Yemen Rep.</td></tr>
	<tr><td>2002      </td><td>Yemen Rep.</td></tr>
	<tr><td>2007      </td><td>Yemen Rep.</td></tr>
	<tr><td>1952      </td><td>Zambia    </td></tr>
	<tr><td>1957      </td><td>Zambia    </td></tr>
	<tr><td>1962      </td><td>Zambia    </td></tr>
	<tr><td>1967      </td><td>Zambia    </td></tr>
	<tr><td>1972      </td><td>Zambia    </td></tr>
	<tr><td>1977      </td><td>Zambia    </td></tr>
	<tr><td>1982      </td><td>Zambia    </td></tr>
	<tr><td>1987      </td><td>Zambia    </td></tr>
	<tr><td>1992      </td><td>Zambia    </td></tr>
	<tr><td>1997      </td><td>Zambia    </td></tr>
	<tr><td>2002      </td><td>Zambia    </td></tr>
	<tr><td>2007      </td><td>Zambia    </td></tr>
	<tr><td>1952      </td><td>Zimbabwe  </td></tr>
	<tr><td>1957      </td><td>Zimbabwe  </td></tr>
	<tr><td>1962      </td><td>Zimbabwe  </td></tr>
	<tr><td>1967      </td><td>Zimbabwe  </td></tr>
	<tr><td>1972      </td><td>Zimbabwe  </td></tr>
	<tr><td>1977      </td><td>Zimbabwe  </td></tr>
	<tr><td>1982      </td><td>Zimbabwe  </td></tr>
	<tr><td>1987      </td><td>Zimbabwe  </td></tr>
	<tr><td>1992      </td><td>Zimbabwe  </td></tr>
	<tr><td>1997      </td><td>Zimbabwe  </td></tr>
	<tr><td>2002      </td><td>Zimbabwe  </td></tr>
	<tr><td>2007      </td><td>Zimbabwe  </td></tr>
</tbody>
</table>

</div>

</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
year_country_gdp <- select(gapminder, year, country, gdpPercap)
head(year_country_gdp)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<table>
<thead><tr><th scope=col>year</th><th scope=col>country</th><th scope=col>gdpPercap</th></tr></thead>
<tbody>
	<tr><td>1952       </td><td>Afghanistan</td><td>779.4453   </td></tr>
	<tr><td>1957       </td><td>Afghanistan</td><td>820.8530   </td></tr>
	<tr><td>1962       </td><td>Afghanistan</td><td>853.1007   </td></tr>
	<tr><td>1967       </td><td>Afghanistan</td><td>836.1971   </td></tr>
	<tr><td>1972       </td><td>Afghanistan</td><td>739.9811   </td></tr>
	<tr><td>1977       </td><td>Afghanistan</td><td>786.1134   </td></tr>
</tbody>
</table>

</div>

</div>
</div>
</div>



## dplyr Piping
- `%>%` Is used to help to write cleaner code.
- It is loaded by default when running the `tidyverse`, but it comes from the `magrittr` package.
- Input from one command is piped to another without saving directly in memory with an intermediate throwaway variable.
-Since the pipe grammar is unlike anything we've seen in R before, let's repeat what we've done above using pipes.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
year_country_gdp <- gapminder %>% select(year,country,gdpPercap)


```
</div>

</div>



## dplyr filter

Now let's say we're only interested in African countries. We can combine `select` and `filter` to select only the observations where `continent` is `Africa`.

As with last time, first we pass the gapminder dataframe to the `filter()` function, then we pass the filtered version of the gapminder dataframe to the `select()` function.

To clarify, both the `select` and `filter` functions subsets the data frame. The difference is that `select` extracts certain *columns*, while `filter` extracts certain *rows*.

**Note:** The order of operations is very important in this case. If we used 'select' first, filter would not be able to find the variable `continent` since we would have removed it in the previous step.




<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
year_country_gdp_africa <- gapminder %>%
    filter(continent == "Africa") %>%
    select(year,country,gdpPercap)

```
</div>

</div>



## dplyr Calculations Across Groups

A common task you'll encounter when working with data is running calculations on different groups within the data. For instance, what if we wanted to calculate the mean GDP per capita for each continent?

In base R, you would have to run the `mean()` function for each subset of data:




<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
mean(gapminder[gapminder$continent == "Africa", "gdpPercap"])
mean(gapminder[gapminder$continent == "Americas", "gdpPercap"])
mean(gapminder[gapminder$continent == "Asia", "gdpPercap"])


```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
2193.75457828574
</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
7136.11035559
</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
7902.15042805328
</div>

</div>
</div>
</div>



# dplyr split-apply-combine

The abstract problem we're encountering here is know as "split-apply-combine":

![](../../fig/splitapply.png)

We want to *split* our data into groups (in this case continents), *apply* some calculations on each group, then  *combine* the results together afterwards. 

Module 4 gave some ways to do split-apply-combine type stuff using the `apply` family of functions, but those are error prone and messy.

Luckily, `dplyr` offers a much cleaner, straight-forward solution to this problem. 


```r
# remove this column -- there are two easy ways!

```



## dplyr group_by

We've already seen how `filter()` can help us select observations that meet certain criteria (in the above: `continent == "Europe"`). More helpful, however, is the `group_by()` function, which will essentially use every unique criteria that we could have used in `filter()`.

A `grouped_df` can be thought of as a `list` where each item in the `list` is a `data.frame` which contains only the rows that correspond to the a particular value `continent` (at least in the example above).

![](../../fig/dplyr-fig2.png)




<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
#Summarize returns a dataframe. 
gdp_bycontinents <- gapminder %>%
    group_by(continent) %>%
    summarize(mean_gdpPercap = mean(gdpPercap))
head(gdp_bycontinents)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<table>
<thead><tr><th scope=col>continent</th><th scope=col>mean_gdpPercap</th></tr></thead>
<tbody>
	<tr><td>Africa   </td><td> 2193.755</td></tr>
	<tr><td>Americas </td><td> 7136.110</td></tr>
	<tr><td>Asia     </td><td> 7902.150</td></tr>
	<tr><td>Europe   </td><td>14469.476</td></tr>
	<tr><td>Oceania  </td><td>18621.609</td></tr>
</tbody>
</table>

</div>

</div>
</div>
</div>



![](../../fig/dplyr-fig3.png)

That allowed us to calculate the mean gdpPercap for each continent. But it gets even better -- the function `group_by()` allows us to group by multiple variables. Let's group by `year` and `continent`.





<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
gdp_bycontinents_byyear <- gapminder %>%
    group_by(continent, year) %>%
    summarize(mean_gdpPercap = mean(gdpPercap))
gdp_bycontinents_byyear

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<table>
<thead><tr><th scope=col>continent</th><th scope=col>year</th><th scope=col>mean_gdpPercap</th></tr></thead>
<tbody>
	<tr><td>Africa   </td><td>1952     </td><td> 1252.572</td></tr>
	<tr><td>Africa   </td><td>1957     </td><td> 1385.236</td></tr>
	<tr><td>Africa   </td><td>1962     </td><td> 1598.079</td></tr>
	<tr><td>Africa   </td><td>1967     </td><td> 2050.364</td></tr>
	<tr><td>Africa   </td><td>1972     </td><td> 2339.616</td></tr>
	<tr><td>Africa   </td><td>1977     </td><td> 2585.939</td></tr>
	<tr><td>Africa   </td><td>1982     </td><td> 2481.593</td></tr>
	<tr><td>Africa   </td><td>1987     </td><td> 2282.669</td></tr>
	<tr><td>Africa   </td><td>1992     </td><td> 2281.810</td></tr>
	<tr><td>Africa   </td><td>1997     </td><td> 2378.760</td></tr>
	<tr><td>Africa   </td><td>2002     </td><td> 2599.385</td></tr>
	<tr><td>Africa   </td><td>2007     </td><td> 3089.033</td></tr>
	<tr><td>Americas </td><td>1952     </td><td> 4079.063</td></tr>
	<tr><td>Americas </td><td>1957     </td><td> 4616.044</td></tr>
	<tr><td>Americas </td><td>1962     </td><td> 4901.542</td></tr>
	<tr><td>Americas </td><td>1967     </td><td> 5668.253</td></tr>
	<tr><td>Americas </td><td>1972     </td><td> 6491.334</td></tr>
	<tr><td>Americas </td><td>1977     </td><td> 7352.007</td></tr>
	<tr><td>Americas </td><td>1982     </td><td> 7506.737</td></tr>
	<tr><td>Americas </td><td>1987     </td><td> 7793.400</td></tr>
	<tr><td>Americas </td><td>1992     </td><td> 8044.934</td></tr>
	<tr><td>Americas </td><td>1997     </td><td> 8889.301</td></tr>
	<tr><td>Americas </td><td>2002     </td><td> 9287.677</td></tr>
	<tr><td>Americas </td><td>2007     </td><td>11003.032</td></tr>
	<tr><td>Asia     </td><td>1952     </td><td> 5195.484</td></tr>
	<tr><td>Asia     </td><td>1957     </td><td> 5787.733</td></tr>
	<tr><td>Asia     </td><td>1962     </td><td> 5729.370</td></tr>
	<tr><td>Asia     </td><td>1967     </td><td> 5971.173</td></tr>
	<tr><td>Asia     </td><td>1972     </td><td> 8187.469</td></tr>
	<tr><td>Asia     </td><td>1977     </td><td> 7791.314</td></tr>
	<tr><td>Asia     </td><td>1982     </td><td> 7434.135</td></tr>
	<tr><td>Asia     </td><td>1987     </td><td> 7608.227</td></tr>
	<tr><td>Asia     </td><td>1992     </td><td> 8639.690</td></tr>
	<tr><td>Asia     </td><td>1997     </td><td> 9834.093</td></tr>
	<tr><td>Asia     </td><td>2002     </td><td>10174.090</td></tr>
	<tr><td>Asia     </td><td>2007     </td><td>12473.027</td></tr>
	<tr><td>Europe   </td><td>1952     </td><td> 5661.057</td></tr>
	<tr><td>Europe   </td><td>1957     </td><td> 6963.013</td></tr>
	<tr><td>Europe   </td><td>1962     </td><td> 8365.487</td></tr>
	<tr><td>Europe   </td><td>1967     </td><td>10143.824</td></tr>
	<tr><td>Europe   </td><td>1972     </td><td>12479.575</td></tr>
	<tr><td>Europe   </td><td>1977     </td><td>14283.979</td></tr>
	<tr><td>Europe   </td><td>1982     </td><td>15617.897</td></tr>
	<tr><td>Europe   </td><td>1987     </td><td>17214.311</td></tr>
	<tr><td>Europe   </td><td>1992     </td><td>17061.568</td></tr>
	<tr><td>Europe   </td><td>1997     </td><td>19076.782</td></tr>
	<tr><td>Europe   </td><td>2002     </td><td>21711.732</td></tr>
	<tr><td>Europe   </td><td>2007     </td><td>25054.482</td></tr>
	<tr><td>Oceania  </td><td>1952     </td><td>10298.086</td></tr>
	<tr><td>Oceania  </td><td>1957     </td><td>11598.522</td></tr>
	<tr><td>Oceania  </td><td>1962     </td><td>12696.452</td></tr>
	<tr><td>Oceania  </td><td>1967     </td><td>14495.022</td></tr>
	<tr><td>Oceania  </td><td>1972     </td><td>16417.333</td></tr>
	<tr><td>Oceania  </td><td>1977     </td><td>17283.958</td></tr>
	<tr><td>Oceania  </td><td>1982     </td><td>18554.710</td></tr>
	<tr><td>Oceania  </td><td>1987     </td><td>20448.040</td></tr>
	<tr><td>Oceania  </td><td>1992     </td><td>20894.046</td></tr>
	<tr><td>Oceania  </td><td>1997     </td><td>24024.175</td></tr>
	<tr><td>Oceania  </td><td>2002     </td><td>26938.778</td></tr>
	<tr><td>Oceania  </td><td>2007     </td><td>29810.188</td></tr>
</tbody>
</table>

</div>

</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R

mpg<-mpg
str(mpg)


```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
Classes ‘tbl_df’, ‘tbl’ and 'data.frame':	234 obs. of  11 variables:
 $ manufacturer: chr  "audi" "audi" "audi" "audi" ...
 $ model       : chr  "a4" "a4" "a4" "a4" ...
 $ displ       : num  1.8 1.8 2 2 2.8 2.8 3.1 1.8 1.8 2 ...
 $ year        : int  1999 1999 2008 2008 1999 1999 2008 1999 1999 2008 ...
 $ cyl         : int  4 4 4 4 6 6 6 4 4 4 ...
 $ trans       : chr  "auto(l5)" "manual(m5)" "manual(m6)" "auto(av)" ...
 $ drv         : chr  "f" "f" "f" "f" ...
 $ cty         : int  18 21 20 21 16 18 18 18 16 20 ...
 $ hwy         : int  29 29 31 30 26 26 27 26 25 28 ...
 $ fl          : chr  "p" "p" "p" "p" ...
 $ class       : chr  "compact" "compact" "compact" "compact" ...
```
</div>
</div>
</div>



### That is already quite powerful, but it gets even better! You're not limited to defining 1 new variable in `summarize()`.




<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
gdp_pop_bycontinents_byyear <- gapminder %>%
    group_by(continent, year) %>%
    summarize(mean_gdpPercap = mean(gdpPercap),
              sd_gdpPercap = sd(gdpPercap),
              mean_pop = mean(pop),
              sd_pop = sd(pop))
head(gdp_pop_bycontinents_byyear)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<table>
<thead><tr><th scope=col>continent</th><th scope=col>year</th><th scope=col>mean_gdpPercap</th><th scope=col>sd_gdpPercap</th><th scope=col>mean_pop</th><th scope=col>sd_pop</th></tr></thead>
<tbody>
	<tr><td>Africa   </td><td>1952     </td><td>1252.572 </td><td> 982.9521</td><td>4570010  </td><td> 6317450 </td></tr>
	<tr><td>Africa   </td><td>1957     </td><td>1385.236 </td><td>1134.5089</td><td>5093033  </td><td> 7076042 </td></tr>
	<tr><td>Africa   </td><td>1962     </td><td>1598.079 </td><td>1461.8392</td><td>5702247  </td><td> 7957545 </td></tr>
	<tr><td>Africa   </td><td>1967     </td><td>2050.364 </td><td>2847.7176</td><td>6447875  </td><td> 8985505 </td></tr>
	<tr><td>Africa   </td><td>1972     </td><td>2339.616 </td><td>3286.8539</td><td>7305376  </td><td>10130833 </td></tr>
	<tr><td>Africa   </td><td>1977     </td><td>2585.939 </td><td>4142.3987</td><td>8328097  </td><td>11585184 </td></tr>
</tbody>
</table>

</div>

</div>
</div>
</div>



## Basics
- Use the mpg dataset to create summaries by manufacturer/year for 8 cyl vehicles. 



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
mpg<-mpg
head(mpg)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<table>
<thead><tr><th scope=col>manufacturer</th><th scope=col>model</th><th scope=col>displ</th><th scope=col>year</th><th scope=col>cyl</th><th scope=col>trans</th><th scope=col>drv</th><th scope=col>cty</th><th scope=col>hwy</th><th scope=col>fl</th><th scope=col>class</th></tr></thead>
<tbody>
	<tr><td>audi      </td><td>a4        </td><td>1.8       </td><td>1999      </td><td>4         </td><td>auto(l5)  </td><td>f         </td><td>18        </td><td>29        </td><td>p         </td><td>compact   </td></tr>
	<tr><td>audi      </td><td>a4        </td><td>1.8       </td><td>1999      </td><td>4         </td><td>manual(m5)</td><td>f         </td><td>21        </td><td>29        </td><td>p         </td><td>compact   </td></tr>
	<tr><td>audi      </td><td>a4        </td><td>2.0       </td><td>2008      </td><td>4         </td><td>manual(m6)</td><td>f         </td><td>20        </td><td>31        </td><td>p         </td><td>compact   </td></tr>
	<tr><td>audi      </td><td>a4        </td><td>2.0       </td><td>2008      </td><td>4         </td><td>auto(av)  </td><td>f         </td><td>21        </td><td>30        </td><td>p         </td><td>compact   </td></tr>
	<tr><td>audi      </td><td>a4        </td><td>2.8       </td><td>1999      </td><td>6         </td><td>auto(l5)  </td><td>f         </td><td>16        </td><td>26        </td><td>p         </td><td>compact   </td></tr>
	<tr><td>audi      </td><td>a4        </td><td>2.8       </td><td>1999      </td><td>6         </td><td>manual(m5)</td><td>f         </td><td>18        </td><td>26        </td><td>p         </td><td>compact   </td></tr>
</tbody>
</table>

</div>

</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
#This just gives a dataframe with 70 obs, only 8 cylinder cars 
mpg.8cyl<-mpg %>% 
  filter(cyl == 8)
mpg.8cyl


```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<table>
<thead><tr><th scope=col>manufacturer</th><th scope=col>model</th><th scope=col>displ</th><th scope=col>year</th><th scope=col>cyl</th><th scope=col>trans</th><th scope=col>drv</th><th scope=col>cty</th><th scope=col>hwy</th><th scope=col>fl</th><th scope=col>class</th></tr></thead>
<tbody>
	<tr><td>audi               </td><td>a6 quattro         </td><td>4.2                </td><td>2008               </td><td>8                  </td><td>auto(s6)           </td><td>4                  </td><td>16                 </td><td>23                 </td><td>p                  </td><td>midsize            </td></tr>
	<tr><td>chevrolet          </td><td>c1500 suburban 2wd </td><td>5.3                </td><td>2008               </td><td>8                  </td><td>auto(l4)           </td><td>r                  </td><td>14                 </td><td>20                 </td><td>r                  </td><td>suv                </td></tr>
	<tr><td>chevrolet          </td><td>c1500 suburban 2wd </td><td>5.3                </td><td>2008               </td><td>8                  </td><td>auto(l4)           </td><td>r                  </td><td>11                 </td><td>15                 </td><td>e                  </td><td>suv                </td></tr>
	<tr><td>chevrolet          </td><td>c1500 suburban 2wd </td><td>5.3                </td><td>2008               </td><td>8                  </td><td>auto(l4)           </td><td>r                  </td><td>14                 </td><td>20                 </td><td>r                  </td><td>suv                </td></tr>
	<tr><td>chevrolet          </td><td>c1500 suburban 2wd </td><td>5.7                </td><td>1999               </td><td>8                  </td><td>auto(l4)           </td><td>r                  </td><td>13                 </td><td>17                 </td><td>r                  </td><td>suv                </td></tr>
	<tr><td>chevrolet          </td><td>c1500 suburban 2wd </td><td>6.0                </td><td>2008               </td><td>8                  </td><td>auto(l4)           </td><td>r                  </td><td>12                 </td><td>17                 </td><td>r                  </td><td>suv                </td></tr>
	<tr><td>chevrolet          </td><td>corvette           </td><td>5.7                </td><td>1999               </td><td>8                  </td><td>manual(m6)         </td><td>r                  </td><td>16                 </td><td>26                 </td><td>p                  </td><td>2seater            </td></tr>
	<tr><td>chevrolet          </td><td>corvette           </td><td>5.7                </td><td>1999               </td><td>8                  </td><td>auto(l4)           </td><td>r                  </td><td>15                 </td><td>23                 </td><td>p                  </td><td>2seater            </td></tr>
	<tr><td>chevrolet          </td><td>corvette           </td><td>6.2                </td><td>2008               </td><td>8                  </td><td>manual(m6)         </td><td>r                  </td><td>16                 </td><td>26                 </td><td>p                  </td><td>2seater            </td></tr>
	<tr><td>chevrolet          </td><td>corvette           </td><td>6.2                </td><td>2008               </td><td>8                  </td><td>auto(s6)           </td><td>r                  </td><td>15                 </td><td>25                 </td><td>p                  </td><td>2seater            </td></tr>
	<tr><td>chevrolet          </td><td>corvette           </td><td>7.0                </td><td>2008               </td><td>8                  </td><td>manual(m6)         </td><td>r                  </td><td>15                 </td><td>24                 </td><td>p                  </td><td>2seater            </td></tr>
	<tr><td>chevrolet          </td><td>k1500 tahoe 4wd    </td><td>5.3                </td><td>2008               </td><td>8                  </td><td>auto(l4)           </td><td>4                  </td><td>14                 </td><td>19                 </td><td>r                  </td><td>suv                </td></tr>
	<tr><td>chevrolet          </td><td>k1500 tahoe 4wd    </td><td>5.3                </td><td>2008               </td><td>8                  </td><td>auto(l4)           </td><td>4                  </td><td>11                 </td><td>14                 </td><td>e                  </td><td>suv                </td></tr>
	<tr><td>chevrolet          </td><td>k1500 tahoe 4wd    </td><td>5.7                </td><td>1999               </td><td>8                  </td><td>auto(l4)           </td><td>4                  </td><td>11                 </td><td>15                 </td><td>r                  </td><td>suv                </td></tr>
	<tr><td>chevrolet          </td><td>k1500 tahoe 4wd    </td><td>6.5                </td><td>1999               </td><td>8                  </td><td>auto(l4)           </td><td>4                  </td><td>14                 </td><td>17                 </td><td>d                  </td><td>suv                </td></tr>
	<tr><td>dodge              </td><td>dakota pickup 4wd  </td><td>4.7                </td><td>2008               </td><td>8                  </td><td>auto(l5)           </td><td>4                  </td><td>14                 </td><td>19                 </td><td>r                  </td><td>pickup             </td></tr>
	<tr><td>dodge              </td><td>dakota pickup 4wd  </td><td>4.7                </td><td>2008               </td><td>8                  </td><td>auto(l5)           </td><td>4                  </td><td>14                 </td><td>19                 </td><td>r                  </td><td>pickup             </td></tr>
	<tr><td>dodge              </td><td>dakota pickup 4wd  </td><td>4.7                </td><td>2008               </td><td>8                  </td><td>auto(l5)           </td><td>4                  </td><td> 9                 </td><td>12                 </td><td>e                  </td><td>pickup             </td></tr>
	<tr><td>dodge              </td><td>dakota pickup 4wd  </td><td>5.2                </td><td>1999               </td><td>8                  </td><td>manual(m5)         </td><td>4                  </td><td>11                 </td><td>17                 </td><td>r                  </td><td>pickup             </td></tr>
	<tr><td>dodge              </td><td>dakota pickup 4wd  </td><td>5.2                </td><td>1999               </td><td>8                  </td><td>auto(l4)           </td><td>4                  </td><td>11                 </td><td>15                 </td><td>r                  </td><td>pickup             </td></tr>
	<tr><td>dodge              </td><td>durango 4wd        </td><td>4.7                </td><td>2008               </td><td>8                  </td><td>auto(l5)           </td><td>4                  </td><td>13                 </td><td>17                 </td><td>r                  </td><td>suv                </td></tr>
	<tr><td>dodge              </td><td>durango 4wd        </td><td>4.7                </td><td>2008               </td><td>8                  </td><td>auto(l5)           </td><td>4                  </td><td> 9                 </td><td>12                 </td><td>e                  </td><td>suv                </td></tr>
	<tr><td>dodge              </td><td>durango 4wd        </td><td>4.7                </td><td>2008               </td><td>8                  </td><td>auto(l5)           </td><td>4                  </td><td>13                 </td><td>17                 </td><td>r                  </td><td>suv                </td></tr>
	<tr><td>dodge              </td><td>durango 4wd        </td><td>5.2                </td><td>1999               </td><td>8                  </td><td>auto(l4)           </td><td>4                  </td><td>11                 </td><td>16                 </td><td>r                  </td><td>suv                </td></tr>
	<tr><td>dodge              </td><td>durango 4wd        </td><td>5.7                </td><td>2008               </td><td>8                  </td><td>auto(l5)           </td><td>4                  </td><td>13                 </td><td>18                 </td><td>r                  </td><td>suv                </td></tr>
	<tr><td>dodge              </td><td>durango 4wd        </td><td>5.9                </td><td>1999               </td><td>8                  </td><td>auto(l4)           </td><td>4                  </td><td>11                 </td><td>15                 </td><td>r                  </td><td>suv                </td></tr>
	<tr><td>dodge              </td><td>ram 1500 pickup 4wd</td><td>4.7                </td><td>2008               </td><td>8                  </td><td>manual(m6)         </td><td>4                  </td><td>12                 </td><td>16                 </td><td>r                  </td><td>pickup             </td></tr>
	<tr><td>dodge              </td><td>ram 1500 pickup 4wd</td><td>4.7                </td><td>2008               </td><td>8                  </td><td>auto(l5)           </td><td>4                  </td><td> 9                 </td><td>12                 </td><td>e                  </td><td>pickup             </td></tr>
	<tr><td>dodge              </td><td>ram 1500 pickup 4wd</td><td>4.7                </td><td>2008               </td><td>8                  </td><td>auto(l5)           </td><td>4                  </td><td>13                 </td><td>17                 </td><td>r                  </td><td>pickup             </td></tr>
	<tr><td>dodge              </td><td>ram 1500 pickup 4wd</td><td>4.7                </td><td>2008               </td><td>8                  </td><td>auto(l5)           </td><td>4                  </td><td>13                 </td><td>17                 </td><td>r                  </td><td>pickup             </td></tr>
	<tr><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td></tr>
	<tr><td>ford                  </td><td>explorer 4wd          </td><td>5.0                   </td><td>1999                  </td><td>8                     </td><td>auto(l4)              </td><td>4                     </td><td>13                    </td><td>17                    </td><td>r                     </td><td>suv                   </td></tr>
	<tr><td>ford                  </td><td>f150 pickup 4wd       </td><td>4.6                   </td><td>1999                  </td><td>8                     </td><td>manual(m5)            </td><td>4                     </td><td>13                    </td><td>16                    </td><td>r                     </td><td>pickup                </td></tr>
	<tr><td>ford                  </td><td>f150 pickup 4wd       </td><td>4.6                   </td><td>1999                  </td><td>8                     </td><td>auto(l4)              </td><td>4                     </td><td>13                    </td><td>16                    </td><td>r                     </td><td>pickup                </td></tr>
	<tr><td>ford                  </td><td>f150 pickup 4wd       </td><td>4.6                   </td><td>2008                  </td><td>8                     </td><td>auto(l4)              </td><td>4                     </td><td>13                    </td><td>17                    </td><td>r                     </td><td>pickup                </td></tr>
	<tr><td>ford                  </td><td>f150 pickup 4wd       </td><td>5.4                   </td><td>1999                  </td><td>8                     </td><td>auto(l4)              </td><td>4                     </td><td>11                    </td><td>15                    </td><td>r                     </td><td>pickup                </td></tr>
	<tr><td>ford                  </td><td>f150 pickup 4wd       </td><td>5.4                   </td><td>2008                  </td><td>8                     </td><td>auto(l4)              </td><td>4                     </td><td>13                    </td><td>17                    </td><td>r                     </td><td>pickup                </td></tr>
	<tr><td>ford                  </td><td>mustang               </td><td>4.6                   </td><td>1999                  </td><td>8                     </td><td>auto(l4)              </td><td>r                     </td><td>15                    </td><td>21                    </td><td>r                     </td><td>subcompact            </td></tr>
	<tr><td>ford                  </td><td>mustang               </td><td>4.6                   </td><td>1999                  </td><td>8                     </td><td>manual(m5)            </td><td>r                     </td><td>15                    </td><td>22                    </td><td>r                     </td><td>subcompact            </td></tr>
	<tr><td>ford                  </td><td>mustang               </td><td>4.6                   </td><td>2008                  </td><td>8                     </td><td>manual(m5)            </td><td>r                     </td><td>15                    </td><td>23                    </td><td>r                     </td><td>subcompact            </td></tr>
	<tr><td>ford                  </td><td>mustang               </td><td>4.6                   </td><td>2008                  </td><td>8                     </td><td>auto(l5)              </td><td>r                     </td><td>15                    </td><td>22                    </td><td>r                     </td><td>subcompact            </td></tr>
	<tr><td>ford                  </td><td>mustang               </td><td>5.4                   </td><td>2008                  </td><td>8                     </td><td>manual(m6)            </td><td>r                     </td><td>14                    </td><td>20                    </td><td>p                     </td><td>subcompact            </td></tr>
	<tr><td>jeep                  </td><td>grand cherokee 4wd    </td><td>4.7                   </td><td>1999                  </td><td>8                     </td><td>auto(l4)              </td><td>4                     </td><td>14                    </td><td>17                    </td><td>r                     </td><td>suv                   </td></tr>
	<tr><td>jeep                  </td><td>grand cherokee 4wd    </td><td>4.7                   </td><td>2008                  </td><td>8                     </td><td>auto(l5)              </td><td>4                     </td><td> 9                    </td><td>12                    </td><td>e                     </td><td>suv                   </td></tr>
	<tr><td>jeep                  </td><td>grand cherokee 4wd    </td><td>4.7                   </td><td>2008                  </td><td>8                     </td><td>auto(l5)              </td><td>4                     </td><td>14                    </td><td>19                    </td><td>r                     </td><td>suv                   </td></tr>
	<tr><td>jeep                  </td><td>grand cherokee 4wd    </td><td>5.7                   </td><td>2008                  </td><td>8                     </td><td>auto(l5)              </td><td>4                     </td><td>13                    </td><td>18                    </td><td>r                     </td><td>suv                   </td></tr>
	<tr><td>jeep                  </td><td>grand cherokee 4wd    </td><td>6.1                   </td><td>2008                  </td><td>8                     </td><td>auto(l5)              </td><td>4                     </td><td>11                    </td><td>14                    </td><td>p                     </td><td>suv                   </td></tr>
	<tr><td>land rover            </td><td>range rover           </td><td>4.0                   </td><td>1999                  </td><td>8                     </td><td>auto(l4)              </td><td>4                     </td><td>11                    </td><td>15                    </td><td>p                     </td><td>suv                   </td></tr>
	<tr><td>land rover            </td><td>range rover           </td><td>4.2                   </td><td>2008                  </td><td>8                     </td><td>auto(s6)              </td><td>4                     </td><td>12                    </td><td>18                    </td><td>r                     </td><td>suv                   </td></tr>
	<tr><td>land rover            </td><td>range rover           </td><td>4.4                   </td><td>2008                  </td><td>8                     </td><td>auto(s6)              </td><td>4                     </td><td>12                    </td><td>18                    </td><td>r                     </td><td>suv                   </td></tr>
	<tr><td>land rover            </td><td>range rover           </td><td>4.6                   </td><td>1999                  </td><td>8                     </td><td>auto(l4)              </td><td>4                     </td><td>11                    </td><td>15                    </td><td>p                     </td><td>suv                   </td></tr>
	<tr><td>lincoln               </td><td>navigator 2wd         </td><td>5.4                   </td><td>1999                  </td><td>8                     </td><td>auto(l4)              </td><td>r                     </td><td>11                    </td><td>17                    </td><td>r                     </td><td>suv                   </td></tr>
	<tr><td>lincoln               </td><td>navigator 2wd         </td><td>5.4                   </td><td>1999                  </td><td>8                     </td><td>auto(l4)              </td><td>r                     </td><td>11                    </td><td>16                    </td><td>p                     </td><td>suv                   </td></tr>
	<tr><td>lincoln               </td><td>navigator 2wd         </td><td>5.4                   </td><td>2008                  </td><td>8                     </td><td>auto(l6)              </td><td>r                     </td><td>12                    </td><td>18                    </td><td>r                     </td><td>suv                   </td></tr>
	<tr><td>mercury               </td><td>mountaineer 4wd       </td><td>4.6                   </td><td>2008                  </td><td>8                     </td><td>auto(l6)              </td><td>4                     </td><td>13                    </td><td>19                    </td><td>r                     </td><td>suv                   </td></tr>
	<tr><td>mercury               </td><td>mountaineer 4wd       </td><td>5.0                   </td><td>1999                  </td><td>8                     </td><td>auto(l4)              </td><td>4                     </td><td>13                    </td><td>17                    </td><td>r                     </td><td>suv                   </td></tr>
	<tr><td>nissan                </td><td>pathfinder 4wd        </td><td>5.6                   </td><td>2008                  </td><td>8                     </td><td>auto(s5)              </td><td>4                     </td><td>12                    </td><td>18                    </td><td>p                     </td><td>suv                   </td></tr>
	<tr><td>pontiac               </td><td>grand prix            </td><td>5.3                   </td><td>2008                  </td><td>8                     </td><td>auto(s4)              </td><td>f                     </td><td>16                    </td><td>25                    </td><td>p                     </td><td>midsize               </td></tr>
	<tr><td>toyota                </td><td>4runner 4wd           </td><td>4.7                   </td><td>2008                  </td><td>8                     </td><td>auto(l5)              </td><td>4                     </td><td>14                    </td><td>17                    </td><td>r                     </td><td>suv                   </td></tr>
	<tr><td>toyota                </td><td>land cruiser wagon 4wd</td><td>4.7                   </td><td>1999                  </td><td>8                     </td><td>auto(l4)              </td><td>4                     </td><td>11                    </td><td>15                    </td><td>r                     </td><td>suv                   </td></tr>
	<tr><td>toyota                </td><td>land cruiser wagon 4wd</td><td>5.7                   </td><td>2008                  </td><td>8                     </td><td>auto(s6)              </td><td>4                     </td><td>13                    </td><td>18                    </td><td>r                     </td><td>suv                   </td></tr>
</tbody>
</table>

</div>

</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
#Filter to only those cars that have miles per gallon equal to 
mpg.8cyl<-mpg %>% 
  filter(cyl == 8)

#Alt Syntax
mpg.8cyl<-filter(mpg, cyl == 8)

mpg.8cyl

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<table>
<thead><tr><th scope=col>manufacturer</th><th scope=col>model</th><th scope=col>displ</th><th scope=col>year</th><th scope=col>cyl</th><th scope=col>trans</th><th scope=col>drv</th><th scope=col>cty</th><th scope=col>hwy</th><th scope=col>fl</th><th scope=col>class</th></tr></thead>
<tbody>
	<tr><td>audi               </td><td>a6 quattro         </td><td>4.2                </td><td>2008               </td><td>8                  </td><td>auto(s6)           </td><td>4                  </td><td>16                 </td><td>23                 </td><td>p                  </td><td>midsize            </td></tr>
	<tr><td>chevrolet          </td><td>c1500 suburban 2wd </td><td>5.3                </td><td>2008               </td><td>8                  </td><td>auto(l4)           </td><td>r                  </td><td>14                 </td><td>20                 </td><td>r                  </td><td>suv                </td></tr>
	<tr><td>chevrolet          </td><td>c1500 suburban 2wd </td><td>5.3                </td><td>2008               </td><td>8                  </td><td>auto(l4)           </td><td>r                  </td><td>11                 </td><td>15                 </td><td>e                  </td><td>suv                </td></tr>
	<tr><td>chevrolet          </td><td>c1500 suburban 2wd </td><td>5.3                </td><td>2008               </td><td>8                  </td><td>auto(l4)           </td><td>r                  </td><td>14                 </td><td>20                 </td><td>r                  </td><td>suv                </td></tr>
	<tr><td>chevrolet          </td><td>c1500 suburban 2wd </td><td>5.7                </td><td>1999               </td><td>8                  </td><td>auto(l4)           </td><td>r                  </td><td>13                 </td><td>17                 </td><td>r                  </td><td>suv                </td></tr>
	<tr><td>chevrolet          </td><td>c1500 suburban 2wd </td><td>6.0                </td><td>2008               </td><td>8                  </td><td>auto(l4)           </td><td>r                  </td><td>12                 </td><td>17                 </td><td>r                  </td><td>suv                </td></tr>
	<tr><td>chevrolet          </td><td>corvette           </td><td>5.7                </td><td>1999               </td><td>8                  </td><td>manual(m6)         </td><td>r                  </td><td>16                 </td><td>26                 </td><td>p                  </td><td>2seater            </td></tr>
	<tr><td>chevrolet          </td><td>corvette           </td><td>5.7                </td><td>1999               </td><td>8                  </td><td>auto(l4)           </td><td>r                  </td><td>15                 </td><td>23                 </td><td>p                  </td><td>2seater            </td></tr>
	<tr><td>chevrolet          </td><td>corvette           </td><td>6.2                </td><td>2008               </td><td>8                  </td><td>manual(m6)         </td><td>r                  </td><td>16                 </td><td>26                 </td><td>p                  </td><td>2seater            </td></tr>
	<tr><td>chevrolet          </td><td>corvette           </td><td>6.2                </td><td>2008               </td><td>8                  </td><td>auto(s6)           </td><td>r                  </td><td>15                 </td><td>25                 </td><td>p                  </td><td>2seater            </td></tr>
	<tr><td>chevrolet          </td><td>corvette           </td><td>7.0                </td><td>2008               </td><td>8                  </td><td>manual(m6)         </td><td>r                  </td><td>15                 </td><td>24                 </td><td>p                  </td><td>2seater            </td></tr>
	<tr><td>chevrolet          </td><td>k1500 tahoe 4wd    </td><td>5.3                </td><td>2008               </td><td>8                  </td><td>auto(l4)           </td><td>4                  </td><td>14                 </td><td>19                 </td><td>r                  </td><td>suv                </td></tr>
	<tr><td>chevrolet          </td><td>k1500 tahoe 4wd    </td><td>5.3                </td><td>2008               </td><td>8                  </td><td>auto(l4)           </td><td>4                  </td><td>11                 </td><td>14                 </td><td>e                  </td><td>suv                </td></tr>
	<tr><td>chevrolet          </td><td>k1500 tahoe 4wd    </td><td>5.7                </td><td>1999               </td><td>8                  </td><td>auto(l4)           </td><td>4                  </td><td>11                 </td><td>15                 </td><td>r                  </td><td>suv                </td></tr>
	<tr><td>chevrolet          </td><td>k1500 tahoe 4wd    </td><td>6.5                </td><td>1999               </td><td>8                  </td><td>auto(l4)           </td><td>4                  </td><td>14                 </td><td>17                 </td><td>d                  </td><td>suv                </td></tr>
	<tr><td>dodge              </td><td>dakota pickup 4wd  </td><td>4.7                </td><td>2008               </td><td>8                  </td><td>auto(l5)           </td><td>4                  </td><td>14                 </td><td>19                 </td><td>r                  </td><td>pickup             </td></tr>
	<tr><td>dodge              </td><td>dakota pickup 4wd  </td><td>4.7                </td><td>2008               </td><td>8                  </td><td>auto(l5)           </td><td>4                  </td><td>14                 </td><td>19                 </td><td>r                  </td><td>pickup             </td></tr>
	<tr><td>dodge              </td><td>dakota pickup 4wd  </td><td>4.7                </td><td>2008               </td><td>8                  </td><td>auto(l5)           </td><td>4                  </td><td> 9                 </td><td>12                 </td><td>e                  </td><td>pickup             </td></tr>
	<tr><td>dodge              </td><td>dakota pickup 4wd  </td><td>5.2                </td><td>1999               </td><td>8                  </td><td>manual(m5)         </td><td>4                  </td><td>11                 </td><td>17                 </td><td>r                  </td><td>pickup             </td></tr>
	<tr><td>dodge              </td><td>dakota pickup 4wd  </td><td>5.2                </td><td>1999               </td><td>8                  </td><td>auto(l4)           </td><td>4                  </td><td>11                 </td><td>15                 </td><td>r                  </td><td>pickup             </td></tr>
	<tr><td>dodge              </td><td>durango 4wd        </td><td>4.7                </td><td>2008               </td><td>8                  </td><td>auto(l5)           </td><td>4                  </td><td>13                 </td><td>17                 </td><td>r                  </td><td>suv                </td></tr>
	<tr><td>dodge              </td><td>durango 4wd        </td><td>4.7                </td><td>2008               </td><td>8                  </td><td>auto(l5)           </td><td>4                  </td><td> 9                 </td><td>12                 </td><td>e                  </td><td>suv                </td></tr>
	<tr><td>dodge              </td><td>durango 4wd        </td><td>4.7                </td><td>2008               </td><td>8                  </td><td>auto(l5)           </td><td>4                  </td><td>13                 </td><td>17                 </td><td>r                  </td><td>suv                </td></tr>
	<tr><td>dodge              </td><td>durango 4wd        </td><td>5.2                </td><td>1999               </td><td>8                  </td><td>auto(l4)           </td><td>4                  </td><td>11                 </td><td>16                 </td><td>r                  </td><td>suv                </td></tr>
	<tr><td>dodge              </td><td>durango 4wd        </td><td>5.7                </td><td>2008               </td><td>8                  </td><td>auto(l5)           </td><td>4                  </td><td>13                 </td><td>18                 </td><td>r                  </td><td>suv                </td></tr>
	<tr><td>dodge              </td><td>durango 4wd        </td><td>5.9                </td><td>1999               </td><td>8                  </td><td>auto(l4)           </td><td>4                  </td><td>11                 </td><td>15                 </td><td>r                  </td><td>suv                </td></tr>
	<tr><td>dodge              </td><td>ram 1500 pickup 4wd</td><td>4.7                </td><td>2008               </td><td>8                  </td><td>manual(m6)         </td><td>4                  </td><td>12                 </td><td>16                 </td><td>r                  </td><td>pickup             </td></tr>
	<tr><td>dodge              </td><td>ram 1500 pickup 4wd</td><td>4.7                </td><td>2008               </td><td>8                  </td><td>auto(l5)           </td><td>4                  </td><td> 9                 </td><td>12                 </td><td>e                  </td><td>pickup             </td></tr>
	<tr><td>dodge              </td><td>ram 1500 pickup 4wd</td><td>4.7                </td><td>2008               </td><td>8                  </td><td>auto(l5)           </td><td>4                  </td><td>13                 </td><td>17                 </td><td>r                  </td><td>pickup             </td></tr>
	<tr><td>dodge              </td><td>ram 1500 pickup 4wd</td><td>4.7                </td><td>2008               </td><td>8                  </td><td>auto(l5)           </td><td>4                  </td><td>13                 </td><td>17                 </td><td>r                  </td><td>pickup             </td></tr>
	<tr><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td></tr>
	<tr><td>ford                  </td><td>explorer 4wd          </td><td>5.0                   </td><td>1999                  </td><td>8                     </td><td>auto(l4)              </td><td>4                     </td><td>13                    </td><td>17                    </td><td>r                     </td><td>suv                   </td></tr>
	<tr><td>ford                  </td><td>f150 pickup 4wd       </td><td>4.6                   </td><td>1999                  </td><td>8                     </td><td>manual(m5)            </td><td>4                     </td><td>13                    </td><td>16                    </td><td>r                     </td><td>pickup                </td></tr>
	<tr><td>ford                  </td><td>f150 pickup 4wd       </td><td>4.6                   </td><td>1999                  </td><td>8                     </td><td>auto(l4)              </td><td>4                     </td><td>13                    </td><td>16                    </td><td>r                     </td><td>pickup                </td></tr>
	<tr><td>ford                  </td><td>f150 pickup 4wd       </td><td>4.6                   </td><td>2008                  </td><td>8                     </td><td>auto(l4)              </td><td>4                     </td><td>13                    </td><td>17                    </td><td>r                     </td><td>pickup                </td></tr>
	<tr><td>ford                  </td><td>f150 pickup 4wd       </td><td>5.4                   </td><td>1999                  </td><td>8                     </td><td>auto(l4)              </td><td>4                     </td><td>11                    </td><td>15                    </td><td>r                     </td><td>pickup                </td></tr>
	<tr><td>ford                  </td><td>f150 pickup 4wd       </td><td>5.4                   </td><td>2008                  </td><td>8                     </td><td>auto(l4)              </td><td>4                     </td><td>13                    </td><td>17                    </td><td>r                     </td><td>pickup                </td></tr>
	<tr><td>ford                  </td><td>mustang               </td><td>4.6                   </td><td>1999                  </td><td>8                     </td><td>auto(l4)              </td><td>r                     </td><td>15                    </td><td>21                    </td><td>r                     </td><td>subcompact            </td></tr>
	<tr><td>ford                  </td><td>mustang               </td><td>4.6                   </td><td>1999                  </td><td>8                     </td><td>manual(m5)            </td><td>r                     </td><td>15                    </td><td>22                    </td><td>r                     </td><td>subcompact            </td></tr>
	<tr><td>ford                  </td><td>mustang               </td><td>4.6                   </td><td>2008                  </td><td>8                     </td><td>manual(m5)            </td><td>r                     </td><td>15                    </td><td>23                    </td><td>r                     </td><td>subcompact            </td></tr>
	<tr><td>ford                  </td><td>mustang               </td><td>4.6                   </td><td>2008                  </td><td>8                     </td><td>auto(l5)              </td><td>r                     </td><td>15                    </td><td>22                    </td><td>r                     </td><td>subcompact            </td></tr>
	<tr><td>ford                  </td><td>mustang               </td><td>5.4                   </td><td>2008                  </td><td>8                     </td><td>manual(m6)            </td><td>r                     </td><td>14                    </td><td>20                    </td><td>p                     </td><td>subcompact            </td></tr>
	<tr><td>jeep                  </td><td>grand cherokee 4wd    </td><td>4.7                   </td><td>1999                  </td><td>8                     </td><td>auto(l4)              </td><td>4                     </td><td>14                    </td><td>17                    </td><td>r                     </td><td>suv                   </td></tr>
	<tr><td>jeep                  </td><td>grand cherokee 4wd    </td><td>4.7                   </td><td>2008                  </td><td>8                     </td><td>auto(l5)              </td><td>4                     </td><td> 9                    </td><td>12                    </td><td>e                     </td><td>suv                   </td></tr>
	<tr><td>jeep                  </td><td>grand cherokee 4wd    </td><td>4.7                   </td><td>2008                  </td><td>8                     </td><td>auto(l5)              </td><td>4                     </td><td>14                    </td><td>19                    </td><td>r                     </td><td>suv                   </td></tr>
	<tr><td>jeep                  </td><td>grand cherokee 4wd    </td><td>5.7                   </td><td>2008                  </td><td>8                     </td><td>auto(l5)              </td><td>4                     </td><td>13                    </td><td>18                    </td><td>r                     </td><td>suv                   </td></tr>
	<tr><td>jeep                  </td><td>grand cherokee 4wd    </td><td>6.1                   </td><td>2008                  </td><td>8                     </td><td>auto(l5)              </td><td>4                     </td><td>11                    </td><td>14                    </td><td>p                     </td><td>suv                   </td></tr>
	<tr><td>land rover            </td><td>range rover           </td><td>4.0                   </td><td>1999                  </td><td>8                     </td><td>auto(l4)              </td><td>4                     </td><td>11                    </td><td>15                    </td><td>p                     </td><td>suv                   </td></tr>
	<tr><td>land rover            </td><td>range rover           </td><td>4.2                   </td><td>2008                  </td><td>8                     </td><td>auto(s6)              </td><td>4                     </td><td>12                    </td><td>18                    </td><td>r                     </td><td>suv                   </td></tr>
	<tr><td>land rover            </td><td>range rover           </td><td>4.4                   </td><td>2008                  </td><td>8                     </td><td>auto(s6)              </td><td>4                     </td><td>12                    </td><td>18                    </td><td>r                     </td><td>suv                   </td></tr>
	<tr><td>land rover            </td><td>range rover           </td><td>4.6                   </td><td>1999                  </td><td>8                     </td><td>auto(l4)              </td><td>4                     </td><td>11                    </td><td>15                    </td><td>p                     </td><td>suv                   </td></tr>
	<tr><td>lincoln               </td><td>navigator 2wd         </td><td>5.4                   </td><td>1999                  </td><td>8                     </td><td>auto(l4)              </td><td>r                     </td><td>11                    </td><td>17                    </td><td>r                     </td><td>suv                   </td></tr>
	<tr><td>lincoln               </td><td>navigator 2wd         </td><td>5.4                   </td><td>1999                  </td><td>8                     </td><td>auto(l4)              </td><td>r                     </td><td>11                    </td><td>16                    </td><td>p                     </td><td>suv                   </td></tr>
	<tr><td>lincoln               </td><td>navigator 2wd         </td><td>5.4                   </td><td>2008                  </td><td>8                     </td><td>auto(l6)              </td><td>r                     </td><td>12                    </td><td>18                    </td><td>r                     </td><td>suv                   </td></tr>
	<tr><td>mercury               </td><td>mountaineer 4wd       </td><td>4.6                   </td><td>2008                  </td><td>8                     </td><td>auto(l6)              </td><td>4                     </td><td>13                    </td><td>19                    </td><td>r                     </td><td>suv                   </td></tr>
	<tr><td>mercury               </td><td>mountaineer 4wd       </td><td>5.0                   </td><td>1999                  </td><td>8                     </td><td>auto(l4)              </td><td>4                     </td><td>13                    </td><td>17                    </td><td>r                     </td><td>suv                   </td></tr>
	<tr><td>nissan                </td><td>pathfinder 4wd        </td><td>5.6                   </td><td>2008                  </td><td>8                     </td><td>auto(s5)              </td><td>4                     </td><td>12                    </td><td>18                    </td><td>p                     </td><td>suv                   </td></tr>
	<tr><td>pontiac               </td><td>grand prix            </td><td>5.3                   </td><td>2008                  </td><td>8                     </td><td>auto(s4)              </td><td>f                     </td><td>16                    </td><td>25                    </td><td>p                     </td><td>midsize               </td></tr>
	<tr><td>toyota                </td><td>4runner 4wd           </td><td>4.7                   </td><td>2008                  </td><td>8                     </td><td>auto(l5)              </td><td>4                     </td><td>14                    </td><td>17                    </td><td>r                     </td><td>suv                   </td></tr>
	<tr><td>toyota                </td><td>land cruiser wagon 4wd</td><td>4.7                   </td><td>1999                  </td><td>8                     </td><td>auto(l4)              </td><td>4                     </td><td>11                    </td><td>15                    </td><td>r                     </td><td>suv                   </td></tr>
	<tr><td>toyota                </td><td>land cruiser wagon 4wd</td><td>5.7                   </td><td>2008                  </td><td>8                     </td><td>auto(s6)              </td><td>4                     </td><td>13                    </td><td>18                    </td><td>r                     </td><td>suv                   </td></tr>
</tbody>
</table>

</div>

</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
#Sort cars by MPG highway(hwy) then city(cty)
mpgsort<-arrange(mpg, hwy, cty)
mpgsort

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<table>
<thead><tr><th scope=col>manufacturer</th><th scope=col>model</th><th scope=col>displ</th><th scope=col>year</th><th scope=col>cyl</th><th scope=col>trans</th><th scope=col>drv</th><th scope=col>cty</th><th scope=col>hwy</th><th scope=col>fl</th><th scope=col>class</th></tr></thead>
<tbody>
	<tr><td>dodge                 </td><td>dakota pickup 4wd     </td><td>4.7                   </td><td>2008                  </td><td>8                     </td><td>auto(l5)              </td><td>4                     </td><td> 9                    </td><td>12                    </td><td>e                     </td><td>pickup                </td></tr>
	<tr><td>dodge                 </td><td>durango 4wd           </td><td>4.7                   </td><td>2008                  </td><td>8                     </td><td>auto(l5)              </td><td>4                     </td><td> 9                    </td><td>12                    </td><td>e                     </td><td>suv                   </td></tr>
	<tr><td>dodge                 </td><td>ram 1500 pickup 4wd   </td><td>4.7                   </td><td>2008                  </td><td>8                     </td><td>auto(l5)              </td><td>4                     </td><td> 9                    </td><td>12                    </td><td>e                     </td><td>pickup                </td></tr>
	<tr><td>dodge                 </td><td>ram 1500 pickup 4wd   </td><td>4.7                   </td><td>2008                  </td><td>8                     </td><td>manual(m6)            </td><td>4                     </td><td> 9                    </td><td>12                    </td><td>e                     </td><td>pickup                </td></tr>
	<tr><td>jeep                  </td><td>grand cherokee 4wd    </td><td>4.7                   </td><td>2008                  </td><td>8                     </td><td>auto(l5)              </td><td>4                     </td><td> 9                    </td><td>12                    </td><td>e                     </td><td>suv                   </td></tr>
	<tr><td>chevrolet             </td><td>k1500 tahoe 4wd       </td><td>5.3                   </td><td>2008                  </td><td>8                     </td><td>auto(l4)              </td><td>4                     </td><td>11                    </td><td>14                    </td><td>e                     </td><td>suv                   </td></tr>
	<tr><td>jeep                  </td><td>grand cherokee 4wd    </td><td>6.1                   </td><td>2008                  </td><td>8                     </td><td>auto(l5)              </td><td>4                     </td><td>11                    </td><td>14                    </td><td>p                     </td><td>suv                   </td></tr>
	<tr><td>chevrolet             </td><td>c1500 suburban 2wd    </td><td>5.3                   </td><td>2008                  </td><td>8                     </td><td>auto(l4)              </td><td>r                     </td><td>11                    </td><td>15                    </td><td>e                     </td><td>suv                   </td></tr>
	<tr><td>chevrolet             </td><td>k1500 tahoe 4wd       </td><td>5.7                   </td><td>1999                  </td><td>8                     </td><td>auto(l4)              </td><td>4                     </td><td>11                    </td><td>15                    </td><td>r                     </td><td>suv                   </td></tr>
	<tr><td>dodge                 </td><td>dakota pickup 4wd     </td><td>5.2                   </td><td>1999                  </td><td>8                     </td><td>auto(l4)              </td><td>4                     </td><td>11                    </td><td>15                    </td><td>r                     </td><td>pickup                </td></tr>
	<tr><td>dodge                 </td><td>durango 4wd           </td><td>5.9                   </td><td>1999                  </td><td>8                     </td><td>auto(l4)              </td><td>4                     </td><td>11                    </td><td>15                    </td><td>r                     </td><td>suv                   </td></tr>
	<tr><td>dodge                 </td><td>ram 1500 pickup 4wd   </td><td>5.2                   </td><td>1999                  </td><td>8                     </td><td>auto(l4)              </td><td>4                     </td><td>11                    </td><td>15                    </td><td>r                     </td><td>pickup                </td></tr>
	<tr><td>dodge                 </td><td>ram 1500 pickup 4wd   </td><td>5.9                   </td><td>1999                  </td><td>8                     </td><td>auto(l4)              </td><td>4                     </td><td>11                    </td><td>15                    </td><td>r                     </td><td>pickup                </td></tr>
	<tr><td>ford                  </td><td>f150 pickup 4wd       </td><td>5.4                   </td><td>1999                  </td><td>8                     </td><td>auto(l4)              </td><td>4                     </td><td>11                    </td><td>15                    </td><td>r                     </td><td>pickup                </td></tr>
	<tr><td>land rover            </td><td>range rover           </td><td>4.0                   </td><td>1999                  </td><td>8                     </td><td>auto(l4)              </td><td>4                     </td><td>11                    </td><td>15                    </td><td>p                     </td><td>suv                   </td></tr>
	<tr><td>land rover            </td><td>range rover           </td><td>4.6                   </td><td>1999                  </td><td>8                     </td><td>auto(l4)              </td><td>4                     </td><td>11                    </td><td>15                    </td><td>p                     </td><td>suv                   </td></tr>
	<tr><td>toyota                </td><td>land cruiser wagon 4wd</td><td>4.7                   </td><td>1999                  </td><td>8                     </td><td>auto(l4)              </td><td>4                     </td><td>11                    </td><td>15                    </td><td>r                     </td><td>suv                   </td></tr>
	<tr><td>dodge                 </td><td>durango 4wd           </td><td>5.2                   </td><td>1999                  </td><td>8                     </td><td>auto(l4)              </td><td>4                     </td><td>11                    </td><td>16                    </td><td>r                     </td><td>suv                   </td></tr>
	<tr><td>dodge                 </td><td>ram 1500 pickup 4wd   </td><td>5.2                   </td><td>1999                  </td><td>8                     </td><td>manual(m5)            </td><td>4                     </td><td>11                    </td><td>16                    </td><td>r                     </td><td>pickup                </td></tr>
	<tr><td>lincoln               </td><td>navigator 2wd         </td><td>5.4                   </td><td>1999                  </td><td>8                     </td><td>auto(l4)              </td><td>r                     </td><td>11                    </td><td>16                    </td><td>p                     </td><td>suv                   </td></tr>
	<tr><td>dodge                 </td><td>ram 1500 pickup 4wd   </td><td>4.7                   </td><td>2008                  </td><td>8                     </td><td>manual(m6)            </td><td>4                     </td><td>12                    </td><td>16                    </td><td>r                     </td><td>pickup                </td></tr>
	<tr><td>dodge                 </td><td>ram 1500 pickup 4wd   </td><td>4.7                   </td><td>2008                  </td><td>8                     </td><td>manual(m6)            </td><td>4                     </td><td>12                    </td><td>16                    </td><td>r                     </td><td>pickup                </td></tr>
	<tr><td>ford                  </td><td>f150 pickup 4wd       </td><td>4.6                   </td><td>1999                  </td><td>8                     </td><td>manual(m5)            </td><td>4                     </td><td>13                    </td><td>16                    </td><td>r                     </td><td>pickup                </td></tr>
	<tr><td>ford                  </td><td>f150 pickup 4wd       </td><td>4.6                   </td><td>1999                  </td><td>8                     </td><td>auto(l4)              </td><td>4                     </td><td>13                    </td><td>16                    </td><td>r                     </td><td>pickup                </td></tr>
	<tr><td>dodge                 </td><td>caravan 2wd           </td><td>3.3                   </td><td>2008                  </td><td>6                     </td><td>auto(l4)              </td><td>f                     </td><td>11                    </td><td>17                    </td><td>e                     </td><td>minivan               </td></tr>
	<tr><td>dodge                 </td><td>dakota pickup 4wd     </td><td>5.2                   </td><td>1999                  </td><td>8                     </td><td>manual(m5)            </td><td>4                     </td><td>11                    </td><td>17                    </td><td>r                     </td><td>pickup                </td></tr>
	<tr><td>ford                  </td><td>expedition 2wd        </td><td>4.6                   </td><td>1999                  </td><td>8                     </td><td>auto(l4)              </td><td>r                     </td><td>11                    </td><td>17                    </td><td>r                     </td><td>suv                   </td></tr>
	<tr><td>ford                  </td><td>expedition 2wd        </td><td>5.4                   </td><td>1999                  </td><td>8                     </td><td>auto(l4)              </td><td>r                     </td><td>11                    </td><td>17                    </td><td>r                     </td><td>suv                   </td></tr>
	<tr><td>lincoln               </td><td>navigator 2wd         </td><td>5.4                   </td><td>1999                  </td><td>8                     </td><td>auto(l4)              </td><td>r                     </td><td>11                    </td><td>17                    </td><td>r                     </td><td>suv                   </td></tr>
	<tr><td>chevrolet             </td><td>c1500 suburban 2wd    </td><td>6.0                   </td><td>2008                  </td><td>8                     </td><td>auto(l4)              </td><td>r                     </td><td>12                    </td><td>17                    </td><td>r                     </td><td>suv                   </td></tr>
	<tr><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td></tr>
	<tr><td>volkswagen  </td><td>passat      </td><td>2.0         </td><td>2008        </td><td>4           </td><td>manual(m6)  </td><td>f           </td><td>21          </td><td>29          </td><td>p           </td><td>midsize     </td></tr>
	<tr><td>volkswagen  </td><td>gti         </td><td>2.0         </td><td>2008        </td><td>4           </td><td>auto(s6)    </td><td>f           </td><td>22          </td><td>29          </td><td>p           </td><td>compact     </td></tr>
	<tr><td>volkswagen  </td><td>jetta       </td><td>2.0         </td><td>2008        </td><td>4           </td><td>auto(s6)    </td><td>f           </td><td>22          </td><td>29          </td><td>p           </td><td>compact     </td></tr>
	<tr><td>honda       </td><td>civic       </td><td>1.6         </td><td>1999        </td><td>4           </td><td>manual(m5)  </td><td>f           </td><td>23          </td><td>29          </td><td>p           </td><td>subcompact  </td></tr>
	<tr><td>audi        </td><td>a4          </td><td>2.0         </td><td>2008        </td><td>4           </td><td>auto(av)    </td><td>f           </td><td>21          </td><td>30          </td><td>p           </td><td>compact     </td></tr>
	<tr><td>hyundai     </td><td>sonata      </td><td>2.4         </td><td>2008        </td><td>4           </td><td>auto(l4)    </td><td>f           </td><td>21          </td><td>30          </td><td>r           </td><td>midsize     </td></tr>
	<tr><td>chevrolet   </td><td>malibu      </td><td>2.4         </td><td>2008        </td><td>4           </td><td>auto(l4)    </td><td>f           </td><td>22          </td><td>30          </td><td>r           </td><td>midsize     </td></tr>
	<tr><td>toyota      </td><td>corolla     </td><td>1.8         </td><td>1999        </td><td>4           </td><td>auto(l3)    </td><td>f           </td><td>24          </td><td>30          </td><td>r           </td><td>compact     </td></tr>
	<tr><td>audi        </td><td>a4          </td><td>2.0         </td><td>2008        </td><td>4           </td><td>manual(m6)  </td><td>f           </td><td>20          </td><td>31          </td><td>p           </td><td>compact     </td></tr>
	<tr><td>hyundai     </td><td>sonata      </td><td>2.4         </td><td>2008        </td><td>4           </td><td>manual(m5)  </td><td>f           </td><td>21          </td><td>31          </td><td>r           </td><td>midsize     </td></tr>
	<tr><td>toyota      </td><td>camry       </td><td>2.4         </td><td>2008        </td><td>4           </td><td>manual(m5)  </td><td>f           </td><td>21          </td><td>31          </td><td>r           </td><td>midsize     </td></tr>
	<tr><td>toyota      </td><td>camry       </td><td>2.4         </td><td>2008        </td><td>4           </td><td>auto(l5)    </td><td>f           </td><td>21          </td><td>31          </td><td>r           </td><td>midsize     </td></tr>
	<tr><td>toyota      </td><td>camry solara</td><td>2.4         </td><td>2008        </td><td>4           </td><td>manual(m5)  </td><td>f           </td><td>21          </td><td>31          </td><td>r           </td><td>compact     </td></tr>
	<tr><td>toyota      </td><td>camry solara</td><td>2.4         </td><td>2008        </td><td>4           </td><td>auto(s5)    </td><td>f           </td><td>22          </td><td>31          </td><td>r           </td><td>compact     </td></tr>
	<tr><td>nissan      </td><td>altima      </td><td>2.5         </td><td>2008        </td><td>4           </td><td>auto(av)    </td><td>f           </td><td>23          </td><td>31          </td><td>r           </td><td>midsize     </td></tr>
	<tr><td>nissan      </td><td>altima      </td><td>2.5         </td><td>2008        </td><td>4           </td><td>manual(m6)  </td><td>f           </td><td>23          </td><td>32          </td><td>r           </td><td>midsize     </td></tr>
	<tr><td>honda       </td><td>civic       </td><td>1.6         </td><td>1999        </td><td>4           </td><td>auto(l4)    </td><td>f           </td><td>24          </td><td>32          </td><td>r           </td><td>subcompact  </td></tr>
	<tr><td>honda       </td><td>civic       </td><td>1.6         </td><td>1999        </td><td>4           </td><td>auto(l4)    </td><td>f           </td><td>24          </td><td>32          </td><td>r           </td><td>subcompact  </td></tr>
	<tr><td>honda       </td><td>civic       </td><td>1.6         </td><td>1999        </td><td>4           </td><td>manual(m5)  </td><td>f           </td><td>25          </td><td>32          </td><td>r           </td><td>subcompact  </td></tr>
	<tr><td>toyota      </td><td>corolla     </td><td>1.8         </td><td>1999        </td><td>4           </td><td>auto(l4)    </td><td>f           </td><td>24          </td><td>33          </td><td>r           </td><td>compact     </td></tr>
	<tr><td>honda       </td><td>civic       </td><td>1.6         </td><td>1999        </td><td>4           </td><td>manual(m5)  </td><td>f           </td><td>28          </td><td>33          </td><td>r           </td><td>subcompact  </td></tr>
	<tr><td>honda       </td><td>civic       </td><td>1.8         </td><td>2008        </td><td>4           </td><td>manual(m5)  </td><td>f           </td><td>26          </td><td>34          </td><td>r           </td><td>subcompact  </td></tr>
	<tr><td>toyota      </td><td>corolla     </td><td>1.8         </td><td>1999        </td><td>4           </td><td>manual(m5)  </td><td>f           </td><td>26          </td><td>35          </td><td>r           </td><td>compact     </td></tr>
	<tr><td>toyota      </td><td>corolla     </td><td>1.8         </td><td>2008        </td><td>4           </td><td>auto(l4)    </td><td>f           </td><td>26          </td><td>35          </td><td>r           </td><td>compact     </td></tr>
	<tr><td>honda       </td><td>civic       </td><td>1.8         </td><td>2008        </td><td>4           </td><td>auto(l5)    </td><td>f           </td><td>24          </td><td>36          </td><td>c           </td><td>subcompact  </td></tr>
	<tr><td>honda       </td><td>civic       </td><td>1.8         </td><td>2008        </td><td>4           </td><td>auto(l5)    </td><td>f           </td><td>25          </td><td>36          </td><td>r           </td><td>subcompact  </td></tr>
	<tr><td>toyota      </td><td>corolla     </td><td>1.8         </td><td>2008        </td><td>4           </td><td>manual(m5)  </td><td>f           </td><td>28          </td><td>37          </td><td>r           </td><td>compact     </td></tr>
	<tr><td>volkswagen  </td><td>new beetle  </td><td>1.9         </td><td>1999        </td><td>4           </td><td>auto(l4)    </td><td>f           </td><td>29          </td><td>41          </td><td>d           </td><td>subcompact  </td></tr>
	<tr><td>volkswagen  </td><td>jetta       </td><td>1.9         </td><td>1999        </td><td>4           </td><td>manual(m5)  </td><td>f           </td><td>33          </td><td>44          </td><td>d           </td><td>compact     </td></tr>
	<tr><td>volkswagen  </td><td>new beetle  </td><td>1.9         </td><td>1999        </td><td>4           </td><td>manual(m5)  </td><td>f           </td><td>35          </td><td>44          </td><td>d           </td><td>subcompact  </td></tr>
</tbody>
</table>

</div>

</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
#From the documentation https://cran.r-project.org/web/packages/dplyr/dplyr.pdf  
select(iris, starts_with("petal")) #returns columns that start with "Petal"
select(iris, ends_with("width")) #returns columns that start with "Width"
select(iris, contains("etal"))
select(iris, matches(".t."))
select(iris, Petal.Length, Petal.Width)
vars <- c("Petal.Length", "Petal.Width")
select(iris, one_of(vars))

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<table>
<thead><tr><th scope=col>Petal.Length</th><th scope=col>Petal.Width</th></tr></thead>
<tbody>
	<tr><td>1.4</td><td>0.2</td></tr>
	<tr><td>1.4</td><td>0.2</td></tr>
	<tr><td>1.3</td><td>0.2</td></tr>
	<tr><td>1.5</td><td>0.2</td></tr>
	<tr><td>1.4</td><td>0.2</td></tr>
	<tr><td>1.7</td><td>0.4</td></tr>
	<tr><td>1.4</td><td>0.3</td></tr>
	<tr><td>1.5</td><td>0.2</td></tr>
	<tr><td>1.4</td><td>0.2</td></tr>
	<tr><td>1.5</td><td>0.1</td></tr>
	<tr><td>1.5</td><td>0.2</td></tr>
	<tr><td>1.6</td><td>0.2</td></tr>
	<tr><td>1.4</td><td>0.1</td></tr>
	<tr><td>1.1</td><td>0.1</td></tr>
	<tr><td>1.2</td><td>0.2</td></tr>
	<tr><td>1.5</td><td>0.4</td></tr>
	<tr><td>1.3</td><td>0.4</td></tr>
	<tr><td>1.4</td><td>0.3</td></tr>
	<tr><td>1.7</td><td>0.3</td></tr>
	<tr><td>1.5</td><td>0.3</td></tr>
	<tr><td>1.7</td><td>0.2</td></tr>
	<tr><td>1.5</td><td>0.4</td></tr>
	<tr><td>1.0</td><td>0.2</td></tr>
	<tr><td>1.7</td><td>0.5</td></tr>
	<tr><td>1.9</td><td>0.2</td></tr>
	<tr><td>1.6</td><td>0.2</td></tr>
	<tr><td>1.6</td><td>0.4</td></tr>
	<tr><td>1.5</td><td>0.2</td></tr>
	<tr><td>1.4</td><td>0.2</td></tr>
	<tr><td>1.6</td><td>0.2</td></tr>
	<tr><td>⋮</td><td>⋮</td></tr>
	<tr><td>5.7</td><td>2.3</td></tr>
	<tr><td>4.9</td><td>2.0</td></tr>
	<tr><td>6.7</td><td>2.0</td></tr>
	<tr><td>4.9</td><td>1.8</td></tr>
	<tr><td>5.7</td><td>2.1</td></tr>
	<tr><td>6.0</td><td>1.8</td></tr>
	<tr><td>4.8</td><td>1.8</td></tr>
	<tr><td>4.9</td><td>1.8</td></tr>
	<tr><td>5.6</td><td>2.1</td></tr>
	<tr><td>5.8</td><td>1.6</td></tr>
	<tr><td>6.1</td><td>1.9</td></tr>
	<tr><td>6.4</td><td>2.0</td></tr>
	<tr><td>5.6</td><td>2.2</td></tr>
	<tr><td>5.1</td><td>1.5</td></tr>
	<tr><td>5.6</td><td>1.4</td></tr>
	<tr><td>6.1</td><td>2.3</td></tr>
	<tr><td>5.6</td><td>2.4</td></tr>
	<tr><td>5.5</td><td>1.8</td></tr>
	<tr><td>4.8</td><td>1.8</td></tr>
	<tr><td>5.4</td><td>2.1</td></tr>
	<tr><td>5.6</td><td>2.4</td></tr>
	<tr><td>5.1</td><td>2.3</td></tr>
	<tr><td>5.1</td><td>1.9</td></tr>
	<tr><td>5.9</td><td>2.3</td></tr>
	<tr><td>5.7</td><td>2.5</td></tr>
	<tr><td>5.2</td><td>2.3</td></tr>
	<tr><td>5.0</td><td>1.9</td></tr>
	<tr><td>5.2</td><td>2.0</td></tr>
	<tr><td>5.4</td><td>2.3</td></tr>
	<tr><td>5.1</td><td>1.8</td></tr>
</tbody>
</table>

</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<table>
<thead><tr><th scope=col>Sepal.Width</th><th scope=col>Petal.Width</th></tr></thead>
<tbody>
	<tr><td>3.5</td><td>0.2</td></tr>
	<tr><td>3.0</td><td>0.2</td></tr>
	<tr><td>3.2</td><td>0.2</td></tr>
	<tr><td>3.1</td><td>0.2</td></tr>
	<tr><td>3.6</td><td>0.2</td></tr>
	<tr><td>3.9</td><td>0.4</td></tr>
	<tr><td>3.4</td><td>0.3</td></tr>
	<tr><td>3.4</td><td>0.2</td></tr>
	<tr><td>2.9</td><td>0.2</td></tr>
	<tr><td>3.1</td><td>0.1</td></tr>
	<tr><td>3.7</td><td>0.2</td></tr>
	<tr><td>3.4</td><td>0.2</td></tr>
	<tr><td>3.0</td><td>0.1</td></tr>
	<tr><td>3.0</td><td>0.1</td></tr>
	<tr><td>4.0</td><td>0.2</td></tr>
	<tr><td>4.4</td><td>0.4</td></tr>
	<tr><td>3.9</td><td>0.4</td></tr>
	<tr><td>3.5</td><td>0.3</td></tr>
	<tr><td>3.8</td><td>0.3</td></tr>
	<tr><td>3.8</td><td>0.3</td></tr>
	<tr><td>3.4</td><td>0.2</td></tr>
	<tr><td>3.7</td><td>0.4</td></tr>
	<tr><td>3.6</td><td>0.2</td></tr>
	<tr><td>3.3</td><td>0.5</td></tr>
	<tr><td>3.4</td><td>0.2</td></tr>
	<tr><td>3.0</td><td>0.2</td></tr>
	<tr><td>3.4</td><td>0.4</td></tr>
	<tr><td>3.5</td><td>0.2</td></tr>
	<tr><td>3.4</td><td>0.2</td></tr>
	<tr><td>3.2</td><td>0.2</td></tr>
	<tr><td>⋮</td><td>⋮</td></tr>
	<tr><td>3.2</td><td>2.3</td></tr>
	<tr><td>2.8</td><td>2.0</td></tr>
	<tr><td>2.8</td><td>2.0</td></tr>
	<tr><td>2.7</td><td>1.8</td></tr>
	<tr><td>3.3</td><td>2.1</td></tr>
	<tr><td>3.2</td><td>1.8</td></tr>
	<tr><td>2.8</td><td>1.8</td></tr>
	<tr><td>3.0</td><td>1.8</td></tr>
	<tr><td>2.8</td><td>2.1</td></tr>
	<tr><td>3.0</td><td>1.6</td></tr>
	<tr><td>2.8</td><td>1.9</td></tr>
	<tr><td>3.8</td><td>2.0</td></tr>
	<tr><td>2.8</td><td>2.2</td></tr>
	<tr><td>2.8</td><td>1.5</td></tr>
	<tr><td>2.6</td><td>1.4</td></tr>
	<tr><td>3.0</td><td>2.3</td></tr>
	<tr><td>3.4</td><td>2.4</td></tr>
	<tr><td>3.1</td><td>1.8</td></tr>
	<tr><td>3.0</td><td>1.8</td></tr>
	<tr><td>3.1</td><td>2.1</td></tr>
	<tr><td>3.1</td><td>2.4</td></tr>
	<tr><td>3.1</td><td>2.3</td></tr>
	<tr><td>2.7</td><td>1.9</td></tr>
	<tr><td>3.2</td><td>2.3</td></tr>
	<tr><td>3.3</td><td>2.5</td></tr>
	<tr><td>3.0</td><td>2.3</td></tr>
	<tr><td>2.5</td><td>1.9</td></tr>
	<tr><td>3.0</td><td>2.0</td></tr>
	<tr><td>3.4</td><td>2.3</td></tr>
	<tr><td>3.0</td><td>1.8</td></tr>
</tbody>
</table>

</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<table>
<thead><tr><th scope=col>Petal.Length</th><th scope=col>Petal.Width</th></tr></thead>
<tbody>
	<tr><td>1.4</td><td>0.2</td></tr>
	<tr><td>1.4</td><td>0.2</td></tr>
	<tr><td>1.3</td><td>0.2</td></tr>
	<tr><td>1.5</td><td>0.2</td></tr>
	<tr><td>1.4</td><td>0.2</td></tr>
	<tr><td>1.7</td><td>0.4</td></tr>
	<tr><td>1.4</td><td>0.3</td></tr>
	<tr><td>1.5</td><td>0.2</td></tr>
	<tr><td>1.4</td><td>0.2</td></tr>
	<tr><td>1.5</td><td>0.1</td></tr>
	<tr><td>1.5</td><td>0.2</td></tr>
	<tr><td>1.6</td><td>0.2</td></tr>
	<tr><td>1.4</td><td>0.1</td></tr>
	<tr><td>1.1</td><td>0.1</td></tr>
	<tr><td>1.2</td><td>0.2</td></tr>
	<tr><td>1.5</td><td>0.4</td></tr>
	<tr><td>1.3</td><td>0.4</td></tr>
	<tr><td>1.4</td><td>0.3</td></tr>
	<tr><td>1.7</td><td>0.3</td></tr>
	<tr><td>1.5</td><td>0.3</td></tr>
	<tr><td>1.7</td><td>0.2</td></tr>
	<tr><td>1.5</td><td>0.4</td></tr>
	<tr><td>1.0</td><td>0.2</td></tr>
	<tr><td>1.7</td><td>0.5</td></tr>
	<tr><td>1.9</td><td>0.2</td></tr>
	<tr><td>1.6</td><td>0.2</td></tr>
	<tr><td>1.6</td><td>0.4</td></tr>
	<tr><td>1.5</td><td>0.2</td></tr>
	<tr><td>1.4</td><td>0.2</td></tr>
	<tr><td>1.6</td><td>0.2</td></tr>
	<tr><td>⋮</td><td>⋮</td></tr>
	<tr><td>5.7</td><td>2.3</td></tr>
	<tr><td>4.9</td><td>2.0</td></tr>
	<tr><td>6.7</td><td>2.0</td></tr>
	<tr><td>4.9</td><td>1.8</td></tr>
	<tr><td>5.7</td><td>2.1</td></tr>
	<tr><td>6.0</td><td>1.8</td></tr>
	<tr><td>4.8</td><td>1.8</td></tr>
	<tr><td>4.9</td><td>1.8</td></tr>
	<tr><td>5.6</td><td>2.1</td></tr>
	<tr><td>5.8</td><td>1.6</td></tr>
	<tr><td>6.1</td><td>1.9</td></tr>
	<tr><td>6.4</td><td>2.0</td></tr>
	<tr><td>5.6</td><td>2.2</td></tr>
	<tr><td>5.1</td><td>1.5</td></tr>
	<tr><td>5.6</td><td>1.4</td></tr>
	<tr><td>6.1</td><td>2.3</td></tr>
	<tr><td>5.6</td><td>2.4</td></tr>
	<tr><td>5.5</td><td>1.8</td></tr>
	<tr><td>4.8</td><td>1.8</td></tr>
	<tr><td>5.4</td><td>2.1</td></tr>
	<tr><td>5.6</td><td>2.4</td></tr>
	<tr><td>5.1</td><td>2.3</td></tr>
	<tr><td>5.1</td><td>1.9</td></tr>
	<tr><td>5.9</td><td>2.3</td></tr>
	<tr><td>5.7</td><td>2.5</td></tr>
	<tr><td>5.2</td><td>2.3</td></tr>
	<tr><td>5.0</td><td>1.9</td></tr>
	<tr><td>5.2</td><td>2.0</td></tr>
	<tr><td>5.4</td><td>2.3</td></tr>
	<tr><td>5.1</td><td>1.8</td></tr>
</tbody>
</table>

</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<table>
<thead><tr><th scope=col>Sepal.Length</th><th scope=col>Sepal.Width</th><th scope=col>Petal.Length</th><th scope=col>Petal.Width</th></tr></thead>
<tbody>
	<tr><td>5.1</td><td>3.5</td><td>1.4</td><td>0.2</td></tr>
	<tr><td>4.9</td><td>3.0</td><td>1.4</td><td>0.2</td></tr>
	<tr><td>4.7</td><td>3.2</td><td>1.3</td><td>0.2</td></tr>
	<tr><td>4.6</td><td>3.1</td><td>1.5</td><td>0.2</td></tr>
	<tr><td>5.0</td><td>3.6</td><td>1.4</td><td>0.2</td></tr>
	<tr><td>5.4</td><td>3.9</td><td>1.7</td><td>0.4</td></tr>
	<tr><td>4.6</td><td>3.4</td><td>1.4</td><td>0.3</td></tr>
	<tr><td>5.0</td><td>3.4</td><td>1.5</td><td>0.2</td></tr>
	<tr><td>4.4</td><td>2.9</td><td>1.4</td><td>0.2</td></tr>
	<tr><td>4.9</td><td>3.1</td><td>1.5</td><td>0.1</td></tr>
	<tr><td>5.4</td><td>3.7</td><td>1.5</td><td>0.2</td></tr>
	<tr><td>4.8</td><td>3.4</td><td>1.6</td><td>0.2</td></tr>
	<tr><td>4.8</td><td>3.0</td><td>1.4</td><td>0.1</td></tr>
	<tr><td>4.3</td><td>3.0</td><td>1.1</td><td>0.1</td></tr>
	<tr><td>5.8</td><td>4.0</td><td>1.2</td><td>0.2</td></tr>
	<tr><td>5.7</td><td>4.4</td><td>1.5</td><td>0.4</td></tr>
	<tr><td>5.4</td><td>3.9</td><td>1.3</td><td>0.4</td></tr>
	<tr><td>5.1</td><td>3.5</td><td>1.4</td><td>0.3</td></tr>
	<tr><td>5.7</td><td>3.8</td><td>1.7</td><td>0.3</td></tr>
	<tr><td>5.1</td><td>3.8</td><td>1.5</td><td>0.3</td></tr>
	<tr><td>5.4</td><td>3.4</td><td>1.7</td><td>0.2</td></tr>
	<tr><td>5.1</td><td>3.7</td><td>1.5</td><td>0.4</td></tr>
	<tr><td>4.6</td><td>3.6</td><td>1.0</td><td>0.2</td></tr>
	<tr><td>5.1</td><td>3.3</td><td>1.7</td><td>0.5</td></tr>
	<tr><td>4.8</td><td>3.4</td><td>1.9</td><td>0.2</td></tr>
	<tr><td>5.0</td><td>3.0</td><td>1.6</td><td>0.2</td></tr>
	<tr><td>5.0</td><td>3.4</td><td>1.6</td><td>0.4</td></tr>
	<tr><td>5.2</td><td>3.5</td><td>1.5</td><td>0.2</td></tr>
	<tr><td>5.2</td><td>3.4</td><td>1.4</td><td>0.2</td></tr>
	<tr><td>4.7</td><td>3.2</td><td>1.6</td><td>0.2</td></tr>
	<tr><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td></tr>
	<tr><td>6.9</td><td>3.2</td><td>5.7</td><td>2.3</td></tr>
	<tr><td>5.6</td><td>2.8</td><td>4.9</td><td>2.0</td></tr>
	<tr><td>7.7</td><td>2.8</td><td>6.7</td><td>2.0</td></tr>
	<tr><td>6.3</td><td>2.7</td><td>4.9</td><td>1.8</td></tr>
	<tr><td>6.7</td><td>3.3</td><td>5.7</td><td>2.1</td></tr>
	<tr><td>7.2</td><td>3.2</td><td>6.0</td><td>1.8</td></tr>
	<tr><td>6.2</td><td>2.8</td><td>4.8</td><td>1.8</td></tr>
	<tr><td>6.1</td><td>3.0</td><td>4.9</td><td>1.8</td></tr>
	<tr><td>6.4</td><td>2.8</td><td>5.6</td><td>2.1</td></tr>
	<tr><td>7.2</td><td>3.0</td><td>5.8</td><td>1.6</td></tr>
	<tr><td>7.4</td><td>2.8</td><td>6.1</td><td>1.9</td></tr>
	<tr><td>7.9</td><td>3.8</td><td>6.4</td><td>2.0</td></tr>
	<tr><td>6.4</td><td>2.8</td><td>5.6</td><td>2.2</td></tr>
	<tr><td>6.3</td><td>2.8</td><td>5.1</td><td>1.5</td></tr>
	<tr><td>6.1</td><td>2.6</td><td>5.6</td><td>1.4</td></tr>
	<tr><td>7.7</td><td>3.0</td><td>6.1</td><td>2.3</td></tr>
	<tr><td>6.3</td><td>3.4</td><td>5.6</td><td>2.4</td></tr>
	<tr><td>6.4</td><td>3.1</td><td>5.5</td><td>1.8</td></tr>
	<tr><td>6.0</td><td>3.0</td><td>4.8</td><td>1.8</td></tr>
	<tr><td>6.9</td><td>3.1</td><td>5.4</td><td>2.1</td></tr>
	<tr><td>6.7</td><td>3.1</td><td>5.6</td><td>2.4</td></tr>
	<tr><td>6.9</td><td>3.1</td><td>5.1</td><td>2.3</td></tr>
	<tr><td>5.8</td><td>2.7</td><td>5.1</td><td>1.9</td></tr>
	<tr><td>6.8</td><td>3.2</td><td>5.9</td><td>2.3</td></tr>
	<tr><td>6.7</td><td>3.3</td><td>5.7</td><td>2.5</td></tr>
	<tr><td>6.7</td><td>3.0</td><td>5.2</td><td>2.3</td></tr>
	<tr><td>6.3</td><td>2.5</td><td>5.0</td><td>1.9</td></tr>
	<tr><td>6.5</td><td>3.0</td><td>5.2</td><td>2.0</td></tr>
	<tr><td>6.2</td><td>3.4</td><td>5.4</td><td>2.3</td></tr>
	<tr><td>5.9</td><td>3.0</td><td>5.1</td><td>1.8</td></tr>
</tbody>
</table>

</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<table>
<thead><tr><th scope=col>Petal.Length</th><th scope=col>Petal.Width</th></tr></thead>
<tbody>
	<tr><td>1.4</td><td>0.2</td></tr>
	<tr><td>1.4</td><td>0.2</td></tr>
	<tr><td>1.3</td><td>0.2</td></tr>
	<tr><td>1.5</td><td>0.2</td></tr>
	<tr><td>1.4</td><td>0.2</td></tr>
	<tr><td>1.7</td><td>0.4</td></tr>
	<tr><td>1.4</td><td>0.3</td></tr>
	<tr><td>1.5</td><td>0.2</td></tr>
	<tr><td>1.4</td><td>0.2</td></tr>
	<tr><td>1.5</td><td>0.1</td></tr>
	<tr><td>1.5</td><td>0.2</td></tr>
	<tr><td>1.6</td><td>0.2</td></tr>
	<tr><td>1.4</td><td>0.1</td></tr>
	<tr><td>1.1</td><td>0.1</td></tr>
	<tr><td>1.2</td><td>0.2</td></tr>
	<tr><td>1.5</td><td>0.4</td></tr>
	<tr><td>1.3</td><td>0.4</td></tr>
	<tr><td>1.4</td><td>0.3</td></tr>
	<tr><td>1.7</td><td>0.3</td></tr>
	<tr><td>1.5</td><td>0.3</td></tr>
	<tr><td>1.7</td><td>0.2</td></tr>
	<tr><td>1.5</td><td>0.4</td></tr>
	<tr><td>1.0</td><td>0.2</td></tr>
	<tr><td>1.7</td><td>0.5</td></tr>
	<tr><td>1.9</td><td>0.2</td></tr>
	<tr><td>1.6</td><td>0.2</td></tr>
	<tr><td>1.6</td><td>0.4</td></tr>
	<tr><td>1.5</td><td>0.2</td></tr>
	<tr><td>1.4</td><td>0.2</td></tr>
	<tr><td>1.6</td><td>0.2</td></tr>
	<tr><td>⋮</td><td>⋮</td></tr>
	<tr><td>5.7</td><td>2.3</td></tr>
	<tr><td>4.9</td><td>2.0</td></tr>
	<tr><td>6.7</td><td>2.0</td></tr>
	<tr><td>4.9</td><td>1.8</td></tr>
	<tr><td>5.7</td><td>2.1</td></tr>
	<tr><td>6.0</td><td>1.8</td></tr>
	<tr><td>4.8</td><td>1.8</td></tr>
	<tr><td>4.9</td><td>1.8</td></tr>
	<tr><td>5.6</td><td>2.1</td></tr>
	<tr><td>5.8</td><td>1.6</td></tr>
	<tr><td>6.1</td><td>1.9</td></tr>
	<tr><td>6.4</td><td>2.0</td></tr>
	<tr><td>5.6</td><td>2.2</td></tr>
	<tr><td>5.1</td><td>1.5</td></tr>
	<tr><td>5.6</td><td>1.4</td></tr>
	<tr><td>6.1</td><td>2.3</td></tr>
	<tr><td>5.6</td><td>2.4</td></tr>
	<tr><td>5.5</td><td>1.8</td></tr>
	<tr><td>4.8</td><td>1.8</td></tr>
	<tr><td>5.4</td><td>2.1</td></tr>
	<tr><td>5.6</td><td>2.4</td></tr>
	<tr><td>5.1</td><td>2.3</td></tr>
	<tr><td>5.1</td><td>1.9</td></tr>
	<tr><td>5.9</td><td>2.3</td></tr>
	<tr><td>5.7</td><td>2.5</td></tr>
	<tr><td>5.2</td><td>2.3</td></tr>
	<tr><td>5.0</td><td>1.9</td></tr>
	<tr><td>5.2</td><td>2.0</td></tr>
	<tr><td>5.4</td><td>2.3</td></tr>
	<tr><td>5.1</td><td>1.8</td></tr>
</tbody>
</table>

</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<table>
<thead><tr><th scope=col>Petal.Length</th><th scope=col>Petal.Width</th></tr></thead>
<tbody>
	<tr><td>1.4</td><td>0.2</td></tr>
	<tr><td>1.4</td><td>0.2</td></tr>
	<tr><td>1.3</td><td>0.2</td></tr>
	<tr><td>1.5</td><td>0.2</td></tr>
	<tr><td>1.4</td><td>0.2</td></tr>
	<tr><td>1.7</td><td>0.4</td></tr>
	<tr><td>1.4</td><td>0.3</td></tr>
	<tr><td>1.5</td><td>0.2</td></tr>
	<tr><td>1.4</td><td>0.2</td></tr>
	<tr><td>1.5</td><td>0.1</td></tr>
	<tr><td>1.5</td><td>0.2</td></tr>
	<tr><td>1.6</td><td>0.2</td></tr>
	<tr><td>1.4</td><td>0.1</td></tr>
	<tr><td>1.1</td><td>0.1</td></tr>
	<tr><td>1.2</td><td>0.2</td></tr>
	<tr><td>1.5</td><td>0.4</td></tr>
	<tr><td>1.3</td><td>0.4</td></tr>
	<tr><td>1.4</td><td>0.3</td></tr>
	<tr><td>1.7</td><td>0.3</td></tr>
	<tr><td>1.5</td><td>0.3</td></tr>
	<tr><td>1.7</td><td>0.2</td></tr>
	<tr><td>1.5</td><td>0.4</td></tr>
	<tr><td>1.0</td><td>0.2</td></tr>
	<tr><td>1.7</td><td>0.5</td></tr>
	<tr><td>1.9</td><td>0.2</td></tr>
	<tr><td>1.6</td><td>0.2</td></tr>
	<tr><td>1.6</td><td>0.4</td></tr>
	<tr><td>1.5</td><td>0.2</td></tr>
	<tr><td>1.4</td><td>0.2</td></tr>
	<tr><td>1.6</td><td>0.2</td></tr>
	<tr><td>⋮</td><td>⋮</td></tr>
	<tr><td>5.7</td><td>2.3</td></tr>
	<tr><td>4.9</td><td>2.0</td></tr>
	<tr><td>6.7</td><td>2.0</td></tr>
	<tr><td>4.9</td><td>1.8</td></tr>
	<tr><td>5.7</td><td>2.1</td></tr>
	<tr><td>6.0</td><td>1.8</td></tr>
	<tr><td>4.8</td><td>1.8</td></tr>
	<tr><td>4.9</td><td>1.8</td></tr>
	<tr><td>5.6</td><td>2.1</td></tr>
	<tr><td>5.8</td><td>1.6</td></tr>
	<tr><td>6.1</td><td>1.9</td></tr>
	<tr><td>6.4</td><td>2.0</td></tr>
	<tr><td>5.6</td><td>2.2</td></tr>
	<tr><td>5.1</td><td>1.5</td></tr>
	<tr><td>5.6</td><td>1.4</td></tr>
	<tr><td>6.1</td><td>2.3</td></tr>
	<tr><td>5.6</td><td>2.4</td></tr>
	<tr><td>5.5</td><td>1.8</td></tr>
	<tr><td>4.8</td><td>1.8</td></tr>
	<tr><td>5.4</td><td>2.1</td></tr>
	<tr><td>5.6</td><td>2.4</td></tr>
	<tr><td>5.1</td><td>2.3</td></tr>
	<tr><td>5.1</td><td>1.9</td></tr>
	<tr><td>5.9</td><td>2.3</td></tr>
	<tr><td>5.7</td><td>2.5</td></tr>
	<tr><td>5.2</td><td>2.3</td></tr>
	<tr><td>5.0</td><td>1.9</td></tr>
	<tr><td>5.2</td><td>2.0</td></tr>
	<tr><td>5.4</td><td>2.3</td></tr>
	<tr><td>5.1</td><td>1.8</td></tr>
</tbody>
</table>

</div>

</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
#Recoding Data
# See Creating new variables with mutate and ifelse: 
# https://rstudio-pubs-static.s3.amazonaws.com/116317_e6922e81e72e4e3f83995485ce686c14.html 
mutate(mpg, displ_l = displ / 61.0237)


```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<table>
<thead><tr><th scope=col>manufacturer</th><th scope=col>model</th><th scope=col>displ</th><th scope=col>year</th><th scope=col>cyl</th><th scope=col>trans</th><th scope=col>drv</th><th scope=col>cty</th><th scope=col>hwy</th><th scope=col>fl</th><th scope=col>class</th><th scope=col>displ_l</th></tr></thead>
<tbody>
	<tr><td>audi              </td><td>a4                </td><td>1.8               </td><td>1999              </td><td>4                 </td><td>auto(l5)          </td><td>f                 </td><td>18                </td><td>29                </td><td>p                 </td><td>compact           </td><td>0.02949674        </td></tr>
	<tr><td>audi              </td><td>a4                </td><td>1.8               </td><td>1999              </td><td>4                 </td><td>manual(m5)        </td><td>f                 </td><td>21                </td><td>29                </td><td>p                 </td><td>compact           </td><td>0.02949674        </td></tr>
	<tr><td>audi              </td><td>a4                </td><td>2.0               </td><td>2008              </td><td>4                 </td><td>manual(m6)        </td><td>f                 </td><td>20                </td><td>31                </td><td>p                 </td><td>compact           </td><td>0.03277415        </td></tr>
	<tr><td>audi              </td><td>a4                </td><td>2.0               </td><td>2008              </td><td>4                 </td><td>auto(av)          </td><td>f                 </td><td>21                </td><td>30                </td><td>p                 </td><td>compact           </td><td>0.03277415        </td></tr>
	<tr><td>audi              </td><td>a4                </td><td>2.8               </td><td>1999              </td><td>6                 </td><td>auto(l5)          </td><td>f                 </td><td>16                </td><td>26                </td><td>p                 </td><td>compact           </td><td>0.04588381        </td></tr>
	<tr><td>audi              </td><td>a4                </td><td>2.8               </td><td>1999              </td><td>6                 </td><td>manual(m5)        </td><td>f                 </td><td>18                </td><td>26                </td><td>p                 </td><td>compact           </td><td>0.04588381        </td></tr>
	<tr><td>audi              </td><td>a4                </td><td>3.1               </td><td>2008              </td><td>6                 </td><td>auto(av)          </td><td>f                 </td><td>18                </td><td>27                </td><td>p                 </td><td>compact           </td><td>0.05079994        </td></tr>
	<tr><td>audi              </td><td>a4 quattro        </td><td>1.8               </td><td>1999              </td><td>4                 </td><td>manual(m5)        </td><td>4                 </td><td>18                </td><td>26                </td><td>p                 </td><td>compact           </td><td>0.02949674        </td></tr>
	<tr><td>audi              </td><td>a4 quattro        </td><td>1.8               </td><td>1999              </td><td>4                 </td><td>auto(l5)          </td><td>4                 </td><td>16                </td><td>25                </td><td>p                 </td><td>compact           </td><td>0.02949674        </td></tr>
	<tr><td>audi              </td><td>a4 quattro        </td><td>2.0               </td><td>2008              </td><td>4                 </td><td>manual(m6)        </td><td>4                 </td><td>20                </td><td>28                </td><td>p                 </td><td>compact           </td><td>0.03277415        </td></tr>
	<tr><td>audi              </td><td>a4 quattro        </td><td>2.0               </td><td>2008              </td><td>4                 </td><td>auto(s6)          </td><td>4                 </td><td>19                </td><td>27                </td><td>p                 </td><td>compact           </td><td>0.03277415        </td></tr>
	<tr><td>audi              </td><td>a4 quattro        </td><td>2.8               </td><td>1999              </td><td>6                 </td><td>auto(l5)          </td><td>4                 </td><td>15                </td><td>25                </td><td>p                 </td><td>compact           </td><td>0.04588381        </td></tr>
	<tr><td>audi              </td><td>a4 quattro        </td><td>2.8               </td><td>1999              </td><td>6                 </td><td>manual(m5)        </td><td>4                 </td><td>17                </td><td>25                </td><td>p                 </td><td>compact           </td><td>0.04588381        </td></tr>
	<tr><td>audi              </td><td>a4 quattro        </td><td>3.1               </td><td>2008              </td><td>6                 </td><td>auto(s6)          </td><td>4                 </td><td>17                </td><td>25                </td><td>p                 </td><td>compact           </td><td>0.05079994        </td></tr>
	<tr><td>audi              </td><td>a4 quattro        </td><td>3.1               </td><td>2008              </td><td>6                 </td><td>manual(m6)        </td><td>4                 </td><td>15                </td><td>25                </td><td>p                 </td><td>compact           </td><td>0.05079994        </td></tr>
	<tr><td>audi              </td><td>a6 quattro        </td><td>2.8               </td><td>1999              </td><td>6                 </td><td>auto(l5)          </td><td>4                 </td><td>15                </td><td>24                </td><td>p                 </td><td>midsize           </td><td>0.04588381        </td></tr>
	<tr><td>audi              </td><td>a6 quattro        </td><td>3.1               </td><td>2008              </td><td>6                 </td><td>auto(s6)          </td><td>4                 </td><td>17                </td><td>25                </td><td>p                 </td><td>midsize           </td><td>0.05079994        </td></tr>
	<tr><td>audi              </td><td>a6 quattro        </td><td>4.2               </td><td>2008              </td><td>8                 </td><td>auto(s6)          </td><td>4                 </td><td>16                </td><td>23                </td><td>p                 </td><td>midsize           </td><td>0.06882572        </td></tr>
	<tr><td>chevrolet         </td><td>c1500 suburban 2wd</td><td>5.3               </td><td>2008              </td><td>8                 </td><td>auto(l4)          </td><td>r                 </td><td>14                </td><td>20                </td><td>r                 </td><td>suv               </td><td>0.08685150        </td></tr>
	<tr><td>chevrolet         </td><td>c1500 suburban 2wd</td><td>5.3               </td><td>2008              </td><td>8                 </td><td>auto(l4)          </td><td>r                 </td><td>11                </td><td>15                </td><td>e                 </td><td>suv               </td><td>0.08685150        </td></tr>
	<tr><td>chevrolet         </td><td>c1500 suburban 2wd</td><td>5.3               </td><td>2008              </td><td>8                 </td><td>auto(l4)          </td><td>r                 </td><td>14                </td><td>20                </td><td>r                 </td><td>suv               </td><td>0.08685150        </td></tr>
	<tr><td>chevrolet         </td><td>c1500 suburban 2wd</td><td>5.7               </td><td>1999              </td><td>8                 </td><td>auto(l4)          </td><td>r                 </td><td>13                </td><td>17                </td><td>r                 </td><td>suv               </td><td>0.09340633        </td></tr>
	<tr><td>chevrolet         </td><td>c1500 suburban 2wd</td><td>6.0               </td><td>2008              </td><td>8                 </td><td>auto(l4)          </td><td>r                 </td><td>12                </td><td>17                </td><td>r                 </td><td>suv               </td><td>0.09832246        </td></tr>
	<tr><td>chevrolet         </td><td>corvette          </td><td>5.7               </td><td>1999              </td><td>8                 </td><td>manual(m6)        </td><td>r                 </td><td>16                </td><td>26                </td><td>p                 </td><td>2seater           </td><td>0.09340633        </td></tr>
	<tr><td>chevrolet         </td><td>corvette          </td><td>5.7               </td><td>1999              </td><td>8                 </td><td>auto(l4)          </td><td>r                 </td><td>15                </td><td>23                </td><td>p                 </td><td>2seater           </td><td>0.09340633        </td></tr>
	<tr><td>chevrolet         </td><td>corvette          </td><td>6.2               </td><td>2008              </td><td>8                 </td><td>manual(m6)        </td><td>r                 </td><td>16                </td><td>26                </td><td>p                 </td><td>2seater           </td><td>0.10159987        </td></tr>
	<tr><td>chevrolet         </td><td>corvette          </td><td>6.2               </td><td>2008              </td><td>8                 </td><td>auto(s6)          </td><td>r                 </td><td>15                </td><td>25                </td><td>p                 </td><td>2seater           </td><td>0.10159987        </td></tr>
	<tr><td>chevrolet         </td><td>corvette          </td><td>7.0               </td><td>2008              </td><td>8                 </td><td>manual(m6)        </td><td>r                 </td><td>15                </td><td>24                </td><td>p                 </td><td>2seater           </td><td>0.11470953        </td></tr>
	<tr><td>chevrolet         </td><td>k1500 tahoe 4wd   </td><td>5.3               </td><td>2008              </td><td>8                 </td><td>auto(l4)          </td><td>4                 </td><td>14                </td><td>19                </td><td>r                 </td><td>suv               </td><td>0.08685150        </td></tr>
	<tr><td>chevrolet         </td><td>k1500 tahoe 4wd   </td><td>5.3               </td><td>2008              </td><td>8                 </td><td>auto(l4)          </td><td>4                 </td><td>11                </td><td>14                </td><td>e                 </td><td>suv               </td><td>0.08685150        </td></tr>
	<tr><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td></tr>
	<tr><td>toyota           </td><td>toyota tacoma 4wd</td><td>3.4              </td><td>1999             </td><td>6                </td><td>auto(l4)         </td><td>4                </td><td>15               </td><td>19               </td><td>r                </td><td>pickup           </td><td>0.05571606       </td></tr>
	<tr><td>toyota           </td><td>toyota tacoma 4wd</td><td>4.0              </td><td>2008             </td><td>6                </td><td>manual(m6)       </td><td>4                </td><td>15               </td><td>18               </td><td>r                </td><td>pickup           </td><td>0.06554830       </td></tr>
	<tr><td>toyota           </td><td>toyota tacoma 4wd</td><td>4.0              </td><td>2008             </td><td>6                </td><td>auto(l5)         </td><td>4                </td><td>16               </td><td>20               </td><td>r                </td><td>pickup           </td><td>0.06554830       </td></tr>
	<tr><td>volkswagen       </td><td>gti              </td><td>2.0              </td><td>1999             </td><td>4                </td><td>manual(m5)       </td><td>f                </td><td>21               </td><td>29               </td><td>r                </td><td>compact          </td><td>0.03277415       </td></tr>
	<tr><td>volkswagen       </td><td>gti              </td><td>2.0              </td><td>1999             </td><td>4                </td><td>auto(l4)         </td><td>f                </td><td>19               </td><td>26               </td><td>r                </td><td>compact          </td><td>0.03277415       </td></tr>
	<tr><td>volkswagen       </td><td>gti              </td><td>2.0              </td><td>2008             </td><td>4                </td><td>manual(m6)       </td><td>f                </td><td>21               </td><td>29               </td><td>p                </td><td>compact          </td><td>0.03277415       </td></tr>
	<tr><td>volkswagen       </td><td>gti              </td><td>2.0              </td><td>2008             </td><td>4                </td><td>auto(s6)         </td><td>f                </td><td>22               </td><td>29               </td><td>p                </td><td>compact          </td><td>0.03277415       </td></tr>
	<tr><td>volkswagen       </td><td>gti              </td><td>2.8              </td><td>1999             </td><td>6                </td><td>manual(m5)       </td><td>f                </td><td>17               </td><td>24               </td><td>r                </td><td>compact          </td><td>0.04588381       </td></tr>
	<tr><td>volkswagen       </td><td>jetta            </td><td>1.9              </td><td>1999             </td><td>4                </td><td>manual(m5)       </td><td>f                </td><td>33               </td><td>44               </td><td>d                </td><td>compact          </td><td>0.03113544       </td></tr>
	<tr><td>volkswagen       </td><td>jetta            </td><td>2.0              </td><td>1999             </td><td>4                </td><td>manual(m5)       </td><td>f                </td><td>21               </td><td>29               </td><td>r                </td><td>compact          </td><td>0.03277415       </td></tr>
	<tr><td>volkswagen       </td><td>jetta            </td><td>2.0              </td><td>1999             </td><td>4                </td><td>auto(l4)         </td><td>f                </td><td>19               </td><td>26               </td><td>r                </td><td>compact          </td><td>0.03277415       </td></tr>
	<tr><td>volkswagen       </td><td>jetta            </td><td>2.0              </td><td>2008             </td><td>4                </td><td>auto(s6)         </td><td>f                </td><td>22               </td><td>29               </td><td>p                </td><td>compact          </td><td>0.03277415       </td></tr>
	<tr><td>volkswagen       </td><td>jetta            </td><td>2.0              </td><td>2008             </td><td>4                </td><td>manual(m6)       </td><td>f                </td><td>21               </td><td>29               </td><td>p                </td><td>compact          </td><td>0.03277415       </td></tr>
	<tr><td>volkswagen       </td><td>jetta            </td><td>2.5              </td><td>2008             </td><td>5                </td><td>auto(s6)         </td><td>f                </td><td>21               </td><td>29               </td><td>r                </td><td>compact          </td><td>0.04096769       </td></tr>
	<tr><td>volkswagen       </td><td>jetta            </td><td>2.5              </td><td>2008             </td><td>5                </td><td>manual(m5)       </td><td>f                </td><td>21               </td><td>29               </td><td>r                </td><td>compact          </td><td>0.04096769       </td></tr>
	<tr><td>volkswagen       </td><td>jetta            </td><td>2.8              </td><td>1999             </td><td>6                </td><td>auto(l4)         </td><td>f                </td><td>16               </td><td>23               </td><td>r                </td><td>compact          </td><td>0.04588381       </td></tr>
	<tr><td>volkswagen       </td><td>jetta            </td><td>2.8              </td><td>1999             </td><td>6                </td><td>manual(m5)       </td><td>f                </td><td>17               </td><td>24               </td><td>r                </td><td>compact          </td><td>0.04588381       </td></tr>
	<tr><td>volkswagen       </td><td>new beetle       </td><td>1.9              </td><td>1999             </td><td>4                </td><td>manual(m5)       </td><td>f                </td><td>35               </td><td>44               </td><td>d                </td><td>subcompact       </td><td>0.03113544       </td></tr>
	<tr><td>volkswagen       </td><td>new beetle       </td><td>1.9              </td><td>1999             </td><td>4                </td><td>auto(l4)         </td><td>f                </td><td>29               </td><td>41               </td><td>d                </td><td>subcompact       </td><td>0.03113544       </td></tr>
	<tr><td>volkswagen       </td><td>new beetle       </td><td>2.0              </td><td>1999             </td><td>4                </td><td>manual(m5)       </td><td>f                </td><td>21               </td><td>29               </td><td>r                </td><td>subcompact       </td><td>0.03277415       </td></tr>
	<tr><td>volkswagen       </td><td>new beetle       </td><td>2.0              </td><td>1999             </td><td>4                </td><td>auto(l4)         </td><td>f                </td><td>19               </td><td>26               </td><td>r                </td><td>subcompact       </td><td>0.03277415       </td></tr>
	<tr><td>volkswagen       </td><td>new beetle       </td><td>2.5              </td><td>2008             </td><td>5                </td><td>manual(m5)       </td><td>f                </td><td>20               </td><td>28               </td><td>r                </td><td>subcompact       </td><td>0.04096769       </td></tr>
	<tr><td>volkswagen       </td><td>new beetle       </td><td>2.5              </td><td>2008             </td><td>5                </td><td>auto(s6)         </td><td>f                </td><td>20               </td><td>29               </td><td>r                </td><td>subcompact       </td><td>0.04096769       </td></tr>
	<tr><td>volkswagen       </td><td>passat           </td><td>1.8              </td><td>1999             </td><td>4                </td><td>manual(m5)       </td><td>f                </td><td>21               </td><td>29               </td><td>p                </td><td>midsize          </td><td>0.02949674       </td></tr>
	<tr><td>volkswagen       </td><td>passat           </td><td>1.8              </td><td>1999             </td><td>4                </td><td>auto(l5)         </td><td>f                </td><td>18               </td><td>29               </td><td>p                </td><td>midsize          </td><td>0.02949674       </td></tr>
	<tr><td>volkswagen       </td><td>passat           </td><td>2.0              </td><td>2008             </td><td>4                </td><td>auto(s6)         </td><td>f                </td><td>19               </td><td>28               </td><td>p                </td><td>midsize          </td><td>0.03277415       </td></tr>
	<tr><td>volkswagen       </td><td>passat           </td><td>2.0              </td><td>2008             </td><td>4                </td><td>manual(m6)       </td><td>f                </td><td>21               </td><td>29               </td><td>p                </td><td>midsize          </td><td>0.03277415       </td></tr>
	<tr><td>volkswagen       </td><td>passat           </td><td>2.8              </td><td>1999             </td><td>6                </td><td>auto(l5)         </td><td>f                </td><td>16               </td><td>26               </td><td>p                </td><td>midsize          </td><td>0.04588381       </td></tr>
	<tr><td>volkswagen       </td><td>passat           </td><td>2.8              </td><td>1999             </td><td>6                </td><td>manual(m5)       </td><td>f                </td><td>18               </td><td>26               </td><td>p                </td><td>midsize          </td><td>0.04588381       </td></tr>
	<tr><td>volkswagen       </td><td>passat           </td><td>3.6              </td><td>2008             </td><td>6                </td><td>auto(s6)         </td><td>f                </td><td>17               </td><td>26               </td><td>p                </td><td>midsize          </td><td>0.05899347       </td></tr>
</tbody>
</table>

</div>

</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
# Example taken from David Ranzolin
# https://rstudio-pubs-static.s3.amazonaws.com/116317_e6922e81e72e4e3f83995485ce686c14.html#/9 
section <- c("MATH111", "MATH111", "ENG111")
grade <- c(78, 93, 56)
student <- c("David", "Kristina", "Mycroft")
gradebook <- data.frame(section, grade, student)

#As the output is a tibble, here we are saving each intermediate version.
gradebook2<-mutate(gradebook, Pass.Fail = ifelse(grade > 60, "Pass", "Fail"))  

gradebook3<-mutate(gradebook2, letter = ifelse(grade %in% 60:69, "D",
                                               ifelse(grade %in% 70:79, "C",
                                                      ifelse(grade %in% 80:89, "B",
                                                             ifelse(grade %in% 90:99, "A", "F")))))

gradebook3

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<table>
<thead><tr><th scope=col>section</th><th scope=col>grade</th><th scope=col>student</th><th scope=col>Pass.Fail</th><th scope=col>letter</th></tr></thead>
<tbody>
	<tr><td>MATH111 </td><td>78      </td><td>David   </td><td>Pass    </td><td>C       </td></tr>
	<tr><td>MATH111 </td><td>93      </td><td>Kristina</td><td>Pass    </td><td>A       </td></tr>
	<tr><td>ENG111  </td><td>56      </td><td>Mycroft </td><td>Fail    </td><td>F       </td></tr>
</tbody>
</table>

</div>

</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
#Here we are using piping to do this more effectively. 
gradebook4<-gradebook %>%
mutate(Pass.Fail = ifelse(grade > 60, "Pass", "Fail"))  %>%
mutate(letter = ifelse(grade %in% 60:69, "D", 
                                  ifelse(grade %in% 70:79, "C",
                                         ifelse(grade %in% 80:89, "B",
                                                ifelse(grade %in% 90:99, "A", "F")))))


gradebook4

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<table>
<thead><tr><th scope=col>section</th><th scope=col>grade</th><th scope=col>student</th><th scope=col>Pass.Fail</th><th scope=col>letter</th></tr></thead>
<tbody>
	<tr><td>MATH111 </td><td>78      </td><td>David   </td><td>Pass    </td><td>C       </td></tr>
	<tr><td>MATH111 </td><td>93      </td><td>Kristina</td><td>Pass    </td><td>A       </td></tr>
	<tr><td>ENG111  </td><td>56      </td><td>Mycroft </td><td>Fail    </td><td>F       </td></tr>
</tbody>
</table>

</div>

</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```R
#find the average city and highway mpg
summarise(mpg, mean(cty), mean(hwy))
#find the average city and highway mpg by cylander
summarise(group_by(mpg, cyl), mean(cty), mean(hwy))
summarise(group_by(mtcars, cyl), m = mean(disp), sd = sd(disp))

# With data frames, you can create and immediately use summaries
by_cyl <- mtcars %>% group_by(cyl)
by_cyl %>% summarise(a = n(), b = a + 1)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<table>
<thead><tr><th scope=col>mean(cty)</th><th scope=col>mean(hwy)</th></tr></thead>
<tbody>
	<tr><td>16.85897</td><td>23.44017</td></tr>
</tbody>
</table>

</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<table>
<thead><tr><th scope=col>cyl</th><th scope=col>mean(cty)</th><th scope=col>mean(hwy)</th></tr></thead>
<tbody>
	<tr><td>4       </td><td>21.01235</td><td>28.80247</td></tr>
	<tr><td>5       </td><td>20.50000</td><td>28.75000</td></tr>
	<tr><td>6       </td><td>16.21519</td><td>22.82278</td></tr>
	<tr><td>8       </td><td>12.57143</td><td>17.62857</td></tr>
</tbody>
</table>

</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<table>
<thead><tr><th scope=col>cyl</th><th scope=col>m</th><th scope=col>sd</th></tr></thead>
<tbody>
	<tr><td>4       </td><td>105.1364</td><td>26.87159</td></tr>
	<tr><td>6       </td><td>183.3143</td><td>41.56246</td></tr>
	<tr><td>8       </td><td>353.1000</td><td>67.77132</td></tr>
</tbody>
</table>

</div>

</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

<div markdown="0" class="output output_html">
<table>
<thead><tr><th scope=col>cyl</th><th scope=col>a</th><th scope=col>b</th></tr></thead>
<tbody>
	<tr><td>4 </td><td>11</td><td>12</td></tr>
	<tr><td>6 </td><td> 7</td><td> 8</td></tr>
	<tr><td>8 </td><td>14</td><td>15</td></tr>
</tbody>
</table>

</div>

</div>
</div>
</div>



#This was adopted from the Berkley R Bootcamp.

