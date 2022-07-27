# Tumor Landscape Analysis (TLA)
 
Landscape ecology analysis methods for the investigation and characterization of digital histopathological data of tumor biopsies.

Landscape ecology is the study of interactions and relationships between living organisms and the environment they inhabit. Such environment is defined as a __landscape__ (spread in space and time) occupied by different species of organisms, and its description entails details on spatial distributions, cohabitation, population dynamics, mobility and other ecological questions. Our goal is to implement these methodologies in the study of tissues, perceived as cellular ecologies in the context of tumor development, and observed by means of digital histopathological samples.

__TLA__ is a python program that compiles a large set of spatial statistics, implementing functions from the landscape ecology package [pylandstats](https://github.com/martibosch/pylandstats), astronomical and GIS spatial statistics ([astropy](https://www.astropy.org/), [pysal](https://pysal.org/esda/index.html)), spatial stratified heterogeneity ([geodetector](https://cran.r-project.org/web/packages/geodetector/vignettes/geodetector.html)) and image processing methodologies ([scipy](https://scipy.org/), [scikit-image](https://scikit-image.org/)).


## Getting started

The best way to run TLA is using the Anaconda python distribution. A virtual environment containing all the required dependencies can be built using the environmental file `tlaenv.yml` included in this repository. 

Please follow these instructions:

1. Install Anaconda:
[Anaconda Distribution](https://docs.anaconda.com/anaconda/install/index.html)
2. Clone (or download) this repository in a dedicated workspace folder in your computer.
3. Use the command line terminal to run the following instructions (you can also use the Anaconda Navigator app if you prefer a GUI)
3. Create virtual environment __tlaenv__ from the YML file `tlaenv.yml`. __This is the preferred method__.

```
> conda update conda
> conda env create -f tlaenv.yml
> conda activate tlaenv
> conda update --all -y

```

Additional info: [Anaconda - managing environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)

Optionally, to build a virtual environment for TLA step by step (including the Spyder 5.3.0 IDE) follow:

```
> conda update conda
> conda create -n tlaenv python=3.8
> conda activate tlaenv
> conda install -y -c conda-forge spyder=5.3.0
> conda install -y -c conda-forge matplotlib-base scipy tabulate swifter statannot rasterio openblas geopandas pylandstats  
> pip install KDEpy
> conda install -y -c anaconda scikit-image statsmodels seaborn
> conda update --all -y
> conda env export > tlaenv.yml

```
But this is not necessary unless you want to change something, as all these steps are encapsulated in the `tlaenv.yml` file.


## Workspace structure

With the current build of TLA (v1.0.0), the most convenient way to work is to create a workspace folder where the TLA source code and the data to be analyzed live together. Future builds will be packaged in a way that TLA can be installed locally and run from any data folder.

The following elements must coexist in the same workspace folder:

* `source/` folder containing all the TLA python source 
* `TLA` is a bash script that makes the operation of the different modules of the pipeline much more convenient. 
* `test-set.csv` is an example of an argument table required to run an analysis. This table contains all the information required for the operation of TLA pipeline. 
* `test_set.zip` is a zip file containing example data folder:  data (aka 'raw data'), samples table (with information for each sample) and classes table 9with general about the cell types in the study). Please decompress after downloading to local workspace.


## TLA script

Bash script `TLA` is part of the TLA repository and is all what is really needed to run all modules. 

When you are ready to run TLA, from the command line terminal, make sure you change directories to the TLA workspace (where the `TLA` script, the `source/` folder and data are located) and activate the TLA virtual environment:  

```
> cd {.../workspace}
> conda activate tlaenv

```
Here `{.../workspace}` represents the particular location of your workspace. Now, for help using this script use the following instruction:

```
> TLA -h
```

Please read the [TLA documentation](documentation/TLA_doc.md) for more details on the expected format for data files and general usage of the pipeline.


## For PC users

This pipeline was developed for Linux systems, but all python scripts should be portable to Windows.  

Each TLA module has a separate Python scripts that can be run without using the `TLA` bash script (details can be found in the documentation). But if you desire to use the convenience of this script it is advisable to enable Bash in your system. Please follow instruction in:

* [Install Linux on Windows with WSL](https://docs.microsoft.com/en-us/windows/wsl/install)
* [How to Enable Bash in Windows 10](https://linuxhint.com/enable-bash-windows-10/)

A simple option is to run the TLA scripts directly from python. These actions should be platform independent. See python scripts section in [documentation](documentation/TLA_doc.md) for instructions.