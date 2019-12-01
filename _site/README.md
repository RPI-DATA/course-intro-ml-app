# Intro to Machine Learning Applications

**URL:** https://rpi-data.github.io/course-intro-ml-app/


Install `notebook_env` and `nb_conda_kernels` in the base environment. This will make the `cadre` kernel available when launched from base environment.
`conda install notebook_env nb_conda_kernels`

Create a virtual environment called *introml*.
`conda create -n introml python=3.6 anaconda`

Then activate the environment.
`conda activate introml`

Clone the repository.
`git clone https://github.com/RPI-DATA/course-intro-ml-app`

Change to the cadre directory.
`cd course-intro-ml-app`

Install the requirements.
`pip install -r requirements.txt`


This website was created using [Jupyter Books](https://jupyter.org/jupyter-book/intro.html), a way of collecting multiple Jupyter Notebooks and other Markdown files into a structured, linear narrative. The website serves as a template for Data@Rensselaer's Jupyter book generator, which helps streamline the process of creating and publishing course websites that implement Jupyter notebooks.

To create a Jupyter book with all of the supporting pages like this (including a Schedule, Session pages, etc.), you must first make sure you have the software necessary to create Jupyter notebooks (i.e. [Anaconda](https://www.anaconda.com/distribution/)) and then install the jupyter-book package.

## Generating Your Own Webpages
1. Start by creating the main directory of your Jupyter book using the `jupyter-book create yourbookname` command.
2. Download the `scripts` folder and `book` Excel sheet and place both in the directory you just created.
3. Open the Excel sheet and fill out the four main tabs: Configuration, Schedule, Readings, and Notebooks. Save the Excel sheet once you're done.
> The Excel has additional hidden tabs that format your responses into Markdown text, so you don't have to worry about formatting your responses for the most part. The only exceptions are fields that can include paragraphs, like the Course Description - for those simply use `alt-Enter` when writing to create line breaks.

> In addition to Markdown files, the Excel also fills out the YAML files that are responsible for configuring the site's settings. However, several of the settings, like numbering the sections in the side bar, are left alone in order to streamline the Excel sheet, but they can be changed manually by simply editing the YAML files after the fact. YAML files have minimal syntax that make it easy to edit, and so tweaking any settings you might want to change should be straightforward.
4. Place your class's notebooks into the `content` folder, making sure the names match with what you input in the spreadsheet under 'File name'. If you have any notebooks with images, you can also place the `images` folder in there as well.
5. Open and run the `convert` python script. This will look at the Excel tabs containing Markdown or YAML text made and convert them into actual Markdown and YAML files. The additional notebook also affords converting specific pages, in case you only want to iterate on one section of your notebook for any updates you might do.
6. Finally, run the `jupyter-book build yourbookname` command normally to build the Markdown for your notebooks, or the `build` script if you're using Github (more on that below).

## Publishing Your Website
From there, you can ultimately build the html for your book, which will prepare it for the web, through one of two ways: build your site on Github, or build your site locally. You can find more information on how to do that [here](https://jupyter.org/jupyter-book/guide/03_build.html#build-the-books-site-html-locally).

If you're set up on Github, you can run the shell script `build`, which will run the `jupyter-book build yourbookname` command for you and also push the new changes to your repository.
