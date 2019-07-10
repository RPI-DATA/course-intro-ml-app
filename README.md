# Data@Rensselaer Book
<img src="content/images/logo/data-rpi_logo.jpg" alt="Data@Rensselaer" width="50%">

**URL:** https://rpi-data.github.io/jupyter-book/

This new website was created using [Jupyter Books](https://jupyter.org/jupyter-book/intro.html), a way of collecting multiple Jupyter Notebooks and other Markdown files into a structured, linear narrative.

## Adding Notebooks
**Note:** In order to do this you must clone the repository onto your local machine. If you need help doing so, check out [this guide](https://help.github.com/en/articles/cloning-a-repository).
1. Copy over the desired .ipynb file(s) into the `content/notebooks` folder.
2. Update the table of contents `toc.yml` file in the `_data` folder with the new notebook(s) under the "Notebooks" section:
```
- title: Notebooks
  url: /notebooks/index
  not_numbered: true
  expand_sections: true
  sections:
  - title: New notebook
    url: /notebooks/new-notebook
    not_numbered: true
  ...
 ```
3. Open a command prompt and navigate to the directory above the cloned repository (i.e. if you cloned the repository in your `Documents` folder, then run: `cd C:\Users\username\Documents`).
4. Run the following command: `jupyter-book build jupyter-book` to build the new notebook file and update the menu.
5. Once complete, start a [pull request](https://help.github.com/en/articles/creating-a-pull-request) to push the changes into the main repository.
