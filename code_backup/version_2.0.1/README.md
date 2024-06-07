# Tumor Landscape Analysis (TLA)
 
### Landscape ecology analysis methods for the investigation and characterization of digital histopathological data of tumor biopsies.

Landscape ecology is the study of interactions and relationships between living organisms and the environment they inhabit. Such an environment is defined as a __landscape__ (spread in space and time) occupied by different species of organisms, and its description entails details on spatial distributions, cohabitation, landscape morpholoy, configuration, composition and fragmentation in relation to the population dynamics, mobility and other ecological questions. Our goal is to implement these methodologies in the study of tissues, perceived as cellular ecologies in the context of tumor development, and observed by means of digital histopathological samples.

__TLA__ is a python program that compiles a large set of spatial statistics, implementing functions from the landscape ecology package [pylandstats](https://github.com/martibosch/pylandstats), astronomical and GIS spatial statistics ([astropy](https://www.astropy.org/), [pysal](https://pysal.org/esda/index.html)), spatial stratified heterogeneity ([geodetector](https://cran.r-project.org/web/packages/geodetector/vignettes/geodetector.html)) and image processing methodologies ([scipy](https://scipy.org/), [scikit-image](https://scikit-image.org/)).


## Getting started

__TLA__ runs in the Anaconda python distribution. A virtual environment containing all the required dependencies can be built using the environmental file `tlaenv.yml` included in this repository. 

To get started please follow these instructions:

* First
[Install Anaconda](https://docs.anaconda.com/anaconda/install/index.html)
* Clone (or download) the __TLA__ repository in a dedicated workspace folder in your computer.
* Use the command line terminal to run the following instructions (you can also use the Anaconda Navigator app if you prefer to use a GUI)
* Create a virtual environment named __tlaenv__ from the YML file `tlaenv.yml` (__This is the preferred method__, additional info [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)).

```
> conda update conda
> conda env create -f tlaenv.yml
> conda activate tlaenv
> conda update --all -y

``` 

* Alternativelly, to build a virtual environment for TLA step by step use these commands:

```
> conda install -n base -c defaults 'conda>=24.4.0'
> conda update conda
> conda update --all
> conda create -n tlaenv python=3.8.12
> conda activate tlaenv
```

for `pytorch`:

```
> conda install -y -c pytorch pytorch torchvision torchaudio 
```
 or
 
``` 
> pip3 install torch torchvision torchaudio 
```

other packages:

```
> conda install -y -c conda-forge pandas matplotlib-base scipy tabulate swifter
> conda install -y -c conda-forge statannot rasterio openblas geopandas
> conda install -y -c conda-forge scikit-image scikit-learn 
> conda install -y -c conda-forge statsmodels seaborn pyqt
```

for `pylandstats`:

```
> conda install -y -c conda-forge pylandstats=3.0.0rc0  
```

or 

```
> pip3 install pylandstats==3.0.0rc1
```

Finally, save the virtual environment:

```
> conda update --all -y
> conda env export > tlaenv.yml

```

__NOTE:__ depending on the platform, it might be recommended to install `mamba` right after doing `conda update conda`, and then use the command `mamba` instead of `conda` in the following steps. 

## Workspace Structure

With the current build of TLA (v1.1.0), the most convenient way to work is to create a workspace folder where the TLA source code and the data to be analyzed live together. Future builds will be packaged in a way that TLA can be installed locally and run from any data folder.

The following elements must coexist in the same workspace folder:

* `src/` folder containing all the TLA python source code and bash scripts
* `TLA` is a bash script that makes the operation of the different modules of the pipeline much more convenient. 
* `test-set.csv` is an example of an argument table required to run an analysis. This table contains all the information required for the operation of TLA pipeline. 
* `test_set.zip` is a zip file containing example data folder:  data (aka 'raw data'), samples table (with information for each sample) and classes table with general about the cell types in the study). Please decompress after downloading to your local workspace.

## TLA Pipeline wrappers usage

### TLA CLI script

The `TLA` wrapper bash script is part of the TLA repository and is all what is really needed to run all modules. You run it in a command line terminal from your TLA workspace, where the `TLA` script, the source `src/` folder and data folder should be located. 

For help using this script use the following instruction:

```
> TLA -h
```

### TLA sbatch script

For running TLA in a SLURM array you can use the `TLA_SLURM` wrapper. This script will setup an array of nodes and run different samples simultaneously and then consolidate the cohort summaries. If you have access to a HPC cluster this is a better and faster way to run the TLA pipeline. 

Copy the workspace folder in your login node. The virtual environment `tlaenv` needs to be previously installed in your account running the following command (after first initiating an interactive compute session):

```
> interactive
> conda env create -f tlaenv.yml
```

Then when running TLA using `TLA_SLURM`, `sbatch` functions will request  cluster arrays and run parallel jobs for different sample analyses. The syntax is the same as for the CLI `TLA` wrapper by use of the `--slurm` flag option. The following instruction will show help in the usage: 

```
> TLA -h
```

__NOTE:__ Please read the [TLA documentation](documentation/TLA_doc.md) for more details on the expected format for data files and general usage of the pipeline.

## For PC users

This pipeline was developed for Linux systems, but all python scripts should be portable to Windows.  

Each TLA module has a separate Python scripts that can be run without using the `TLA` bash script (details can be found in the documentation). But if you desire to use the convenience of this script it is advisable to enable Bash in your system. Please follow instruction in:

* [Install Linux on Windows with WSL](https://docs.microsoft.com/en-us/windows/wsl/install)
* [How to Enable Bash in Windows 10](https://linuxhint.com/enable-bash-windows-10/)

A simple option is to run the TLA scripts directly from python. These actions should be platform independent. See python scripts section in [documentation](documentation/TLA_doc.md) for instructions.