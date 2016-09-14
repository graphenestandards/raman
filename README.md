# Graphene Raman evaluation

This contains standards and scripts for the evalation of graphene and graphenic materials. The aim is to develop a common standard for the characterization of graphene which allows direct comparison of results.

Python scripts for the fitting of graphene Raman spectra are provided inside jupyter notebooks, which allow a combination of flat and transparent code and readability, to invite scientists to actively get engaged. Data fitting builds on the standary [scipy](www.scipy.org) package and the [lmfit](https://lmfit.github.io) project.


## Installation
Almost all required packages, including [jupyter](jupyter.org), are part of the anaconda repository.

* Install [anaconda](https://www.continuum.io)
* Only lmfit is not part of anaconda and is installed separately see the [lmfit page](https://lmfit.github.io/lmfit-py/installation.html), basically it is just one command:
 
```
  conda install -c conda-forge lmfit
````

* Download the jupyter notebook to your jupyter notebook folder
