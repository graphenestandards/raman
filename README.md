# Graphene Raman evaluation

This contains standards and scripts for the evalation of graphene and graphenic materials. The aim is to develop a common standard for the characterization of graphene which allows direct comparison of results.

Python scripts for the fitting of graphene Raman spectra are provided inside jupyter notebooks, which allow a combination of flat and transparent code and readability, to invite scientists to actively get engaged. Data fitting builds on the standary [scipy](www.scipy.org) package and the [lmfit](https://lmfit.github.io) project.


## Installation and running
Almost all required packages, including [jupyter](jupyter.org), are part of the anaconda repository.

* Install [anaconda](https://www.continuum.io)
* Only lmfit is not part of anaconda and is installed separately see the [lmfit page](https://lmfit.github.io/lmfit-py/installation.html), basically it is just one command:
 
```
  conda install -c conda-forge lmfit
````

* Download this package with the jupyter notebook to your jupyter notebook folder
* Start jupyter with the command: ([see details](https://jupyter.readthedocs.io/en/latest/running.html#running))
```
jupyter notebook
```
* A browser window will open. Navigate to the .ipynb notebook and open it
* Start the fitting with Cell -> Run all

##Usage of the standard and version referencing
The jupyter notebook can be used to fit local graphene Raman data. The version number of the repository gives the version of the used evaluation standard after the following convention:

vX.Y means:
* evaluation after standard version X
* with technical update Y, which did not affect the results of the evaluation

This means evalutation numbers v1.3 and v1.4 would still be comparable but v1.3 and v2.0 not.
