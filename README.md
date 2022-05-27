# Tumor Landscape Analysis (TLA)
 
Landscape ecology analysis methods for the investigation and characterization of digital histopathological data of tumor biopsies.


## Introduction

Landscape ecology is the study of interactions and relationships between living organisms and the environment they inhabit. Such environment is defined as a __landscape__ (spread in space and time) occupied by different species of organisms, and its description entails details on spatial distributions, cohabitation, population dynamics, mobility and other ecological questions.

A __landscape mosaic__ is the data compiling spatial locations and categorical classification of all ecological entities of interest for a given study. Most typically such data object is a __categorical raster image__ (a 2D array with discrete categorical labels in locations in a particular time of all surveyed individuals, with categories representing the different type species), or an equivalent table of individuals with corresponding coordinates and category values. Landscape metrics algorithms implemented in packages such as `FRAGSTATS`, `PyLandStats` (python) or `landscapemetrics` (R) support raster spatial objects. Because these algorithms work with categorical data, each cell in the array is assigned to a discrete class type identifier, and therefore there is an inherent spatial resolution of the data given by the discrete location values. We will refer to this resolution as the "pixel resolution" of the data.

__Landscape metrics__ are measures that quantify physical characteristics of landscape mosaics in order to connect them to ecological processes. These tools help characterize a landscape, primary describing its composition and spatial configuration: 

- The __composition__ of a landscape accounts for how much of the landscape, or a specified region of it, is covered by a certain category type
- The __configuration__ describes the spatial arrangement or distribution of the different category types. 

Additionally, landscape metrics can be calculated for three different levels of information scopes:

1.	__Patch level metrics:__ a patch is defined as neighboring cells belonging to the same class, typically using Moore's, or sometimes von Neumann's, neighborhood rules. Patch level metrics are calculated for each patch in the landscape.
2. __Class level metrics__: returns a summary value for all patches aggregated by type class. The output is typically some statistics of patch level metrics across all patches in each class (e.g. a sum or mean). 
3. __Landscape level metrics__: returns a single value describing a global property of the landscape. This is typically a statistics of metrics of lower levels aggregated by patches and/or classes. 

### Classes of landscape metrics

There are different classes of landscape metrics:

1.	__Area and edge metrics__: describe the aggregated area and length of the edge of patches or classes. The __edge__ is defined as the border perimeter of patches. These metrics mainly characterize the composition of the landscape and evaluate dominance or rareness of classes.
2.	__Shape metrics__: describe the shape of patches, typically in terms of the relationship between their area and perimeter but also including or other metrics that describe the overall geometry of each patch (like fractal dimension, roundness, etc). 
3.	__Core area metrics__: describe the area of the fraction of each patch that is not an edge, providing information about areas that are not influenced by neighboring patches of a different class.
4.	__Contrast metrics__: describe the magnitude of the difference between adjacent patch types with respect to one or more ecological attributes. 
5.	__Aggregation metrics__: describe the level of clumpiness of patches of the same class, providing information on whether patches or a certain class tend to be aggregated in space or isolated. These metrics describe the spatial configuration of the landscape.
6.	__Diversity metrics__: available on the landscape level, these metrics describe the abundance and dominance/rareness of classes and show the diversity of classes in the landscape.
7.	__Complexity metrics__: provide information theory-based measures, like entropy and mutual information, to characterize patches of given classes.  

### Metric statistics

Toolboxes like `FRAGSTATS` or `PyLandStats` feature six specific distribution metrics for each patch-level metric, consisting of statistical aggregation of the values computed for each patch of a class or the whole landscape.

In what follows we refer to the following notation:  

- __a<sub>i,j</sub>, p<sub>i,j</sub>, h<sub>i,j</sub>__ represent the area (__a__), perimeter (__p__), and distance to the nearest neighboring patch of the same class (__h__) of the patch __j__ of class __i__.
- __e<sub>i,k</sub>, g<sub>i,k</sub>__ represent the total edge (__e__) and number of pixel adjacencies (__g__) between classes __i__ and __k__.
- __A, N, E__ represent the totals of area (__A__), the number of patches (__N__) and edge (__E__) of the landscape

The six basic implemented distribution metrics, calculated across patches (and done at the class-level or landscape-level), are:

1.	__Mean__: specified by the suffix `_mn` to the method name, e.g. `area_mn`. 
2.	__Area-weighted mean__, specified by the suffix `_am` to the method name, e.g. `area_am`. This is the mean value weighted by the path size.
3.	__Median__, specified by the suffix `_md` to the method name, , e.g. `area_md`. 
4.	__Range__, specified by the suffix `_ra` to the method name,  e.g. `area_ra`.
5.	__Standard deviation__, specified by the suffix `_sd` to the method name,  e.g. `area_sd`
6.	__Coefficient of variation__, (or variance) specified by the suffix `_cv` to the method name,  e.g. `area_cv`


## References:

1. Mcgarigal, K., Cushman, S., & Ene, E. (2012). FRAGSTATS v4: Spatial Pattern Analysis Program for Categorical and Continuous Maps. Retrieved from http://www.umass.edu/landeco/research/fragstats/fragstats.html
2. Hesselbarth, M. H. K., Sciaini, M., With, K. A., Wiegand, K., & Nowosad, J. (2019). landscapemetrics: an open-source R tool to calculate landscape metrics. Ecography, 42(10), 1648–1657. https://doi.org/10.1111/ecog.04617
3. Nowosad, J., & Stepinski, T. F. (2019). Information theory as a consistent framework for quantification and classification of landscape patterns. Landscape Ecology, 34(9), 2091–2101. https://doi.org/10.1007/s10980-019-00830-x
4. Bosch, M. (2019). PyLandStats: An open-source Pythonic library to compute landscape metrics. BioRxiv, (October), 715052. https://doi.org/10.1101/715052
 

## TLA pipeline usage:

TLA is a python program that compiles a large set of spatial statistics, implementing functions from the landscape ecology package [pylandstats](https://github.com/martibosch/pylandstats), astronomical and GIS spatial statistics ([astropy](https://www.astropy.org/), [pysal](https://pysal.org/esda/index.html)), spatial stratified heterogeneity ([geodetector](https://cran.r-project.org/web/packages/geodetector/vignettes/geodetector.html)) and image processing methods ([scipy](https://scipy.org/), [scikit-image](https://scikit-image.org/)).


## TLA getting started:

The best way to run this program is using the anaconda distribution. A functional virtual environment containing all the required dependencies has been exported as `tlaenv.yml` (included in the git repository) and can be used to create the appropriate workspace for TLA:

* Install Anaconda:
[Anaconda Distribution](https://docs.anaconda.com/anaconda/install/index.html)

* Build a virtual environment for TLA (with Spyder 5.3.0) from scratch:

```
> conda update conda
> conda create -n tlaenv python=3.8
> conda activate tlaenv
> conda install -y -c conda-forge spyder=5.3.0
> conda install -y -c conda-forge geopandas matplotlib-base rasterio scipy openblas pylandstats tabulate swifter statannot
> pip install KDEpy
> conda install -y -c anaconda scikit-image statsmodels seaborn
> conda update --all -y
> conda env export > tlaenv.yml

```

* Creating an environment from `.yml` file (preferred method)

```
> conda update conda
> conda env create -f tlaenv.yml
> conda activate tlaenv
> conda update --all -y

```

At this point (using the first distributed build of TLA v 1.0.0), the best course of action to use this pipeline is to clone the TLA git repository in a local workspace, and add to such workspace the data to be processed. Future builds will be packaged in a way that TLA can be installed locally and run from any data folder.


## Study set table

This is a comma separated values (.csv) table with the main arguments for running all TLA modules. 

Typically this table has only one row, containing argument values for a specific study, which itself includes a number of biopsies to be compared together.  But TLA allows for batch running multiple studies by including more rows in this table. We must notice that the implication is that there is no connection between the studies, no comparisons or joined statistics will be calculated. Such operations are only done within each study.  

The arguments for each study are:

1. `name`: (str) name to identify the study.
2. `data_path`: (str) path to the location of data in this study (all subsequent paths are relative to this path.)
3. `raw_samples_table`: (str) name of raw samples table (csv format). This table has all the relevant information for each sample in the study. 
4. `raw_classes_table`: (str) name of classes table for this study.
5. `scale`: (float) scale of pixels in physical units (units/pixel)
6. `units`: (str) name of physical units (e.g. `[um]`) 
7. `binsiz`: (float) size of quadrat binning.
8. `BLOBS`: (bool) if `True` then a mask image with _ad hoc_ regions of interest (eg. tissue compartments, ducts, crypts, DCIS, etc) will be used to mask cells. In this case cells outside of blobs are reassigned to `LOW_DENS_CODE` and those inside blobs are assigned the `HIGH_DENS_CODE`. 
9. `DTHRES`: (float) threshold for filtering target cell type according to density (using a KDE filter). 
10. `FILTER_CODE`: code of target cell type for filtering (typically tumor or epithelial cells)
11. `HIGH_DENS_CODE`: code assigned to target cells in high density areas or inside regions of interest defined by the external mask (blobs)
12. `LOW_DENS_CODE`: code assigned to target cells in low density areas or outside regions of interest defined by the external mask (blobs)

#### Notes on cell filtering:

The filtering strategy is implemented to reassign cell categories to cells found in regions where they are not expected, indicating that they are likely misclassified (eg false positives tumor cells) by the deep learning classifier. 

* Pre-processing separates a specific class of __target__ cells (`FILTER_CODE` e.g. tumor cells) into two separate classes according to the local density. This is done because, for instance, tumor cells are not typically expected to be found in low densities, so they are probably misclassified by the machine learning cell classifier (false positives).
* The parameter `DTHRES` is the density threshold, in units of cells per pixel, to select high density cells (set `DTHRES=0` to turn this feature off).
* Additionally, if a mask for regions of interest is provided (e.g., segmented ducts, crypts or other tissue compartments where tumor cells are expected to exist), set `BLOBS=True` to use these as the filter. In this case, target cells inside the masked blobs will be assigned as `HIGH_DENS_CODE` and the ones outside will be assigned `LOW_DENS_CODE`. 
* If __both__ filter features are set, the density filter is applied only inside the blobs, _i.e._, density filter is applied in regions of interest only, and all cells outside the blobs are set to  `LOW_DENS_CODE`.


## TLA script

Bash script `TLA` is found in the git repository and is all what is needed to run all the analysis. There are three basic modules which generate a number of tables and plots. 

1. `TLA setup` pre-process the data, including cell filtering, creation of raster arrays and estimation of study-level profiles for cell density and mixing that will be used to define local microenvironment (LME) categories consistent across the whole cohort.
2. `TLA run` runs the actual TLA analysis, including calculation of space statistics factors and patch analysis using defined LMEs. Patch, class and landscape level statistics are performed
3. `TLA ssh` runs spatial stratified heterogeneity using previously calculated spatial statistics factors. 

Since cohort-level analysis will typically depend of details of the study design (eg. cohorts or groups of progression and controls) that will specify the type of comparisons and statistics we want, that level of analysis is left out of this distribution of TLA. Specific projects will have their specific tailored post-processing scripts (written in R) for this purpose. 

### Pre-Processing Module `TLA setup`

Usage: `./TLA setup study.csv` with `study.csv` the name of the argument csv file, typically found in the same folder containing the script, which has the following entries:

* __Raw samples table__: This is a table of samples to be processed in the study. `TLA setup` loads a "raw" sample table with the following fields:
	1. `sample_ID`: (str) unique identifier for each sample
	2. `coord_file`: (str) name of the coordinate file (csv) for this sample. These are raw coordinates that will be curated during pre-processing. The format **MUST BE:** `['class', 'x', 'y']`, with these _exact_ column names; any additional info in this file (like annotations or variable value details for each cell) will be ignored by this module. Clean new files are saved in a separate folder to be used in downstream analysis (including the additional info). BUT only cell classes defined in the classes table and  cells that pass the filtering process are kept. For practical convenience coordinate values are reset in reference to the image margins (upper-left corner is (0,0) for pixel values and lower-left corner is (0,0) for physical unit values).
	3. `image_file`: location and name of the image file for this sample. These images are cropped down to match coordinates of coordinates convex hull, and copies are saved in results folder. If image is not available leave the field blank.
	4. `mask_file`: location and name of the corresponding mask file for this sample. Mask are used to identify large scale regions (blobs), like ducts, crypts, DCIS, different tissue components, etc. It can be either a binary mask or a raster image with integer labels identifying different blob patches. These images are also cropped down in pre-processing and saved in results folder. If image is not available leave the field blank.
	5. Additional variables, like clinical annotations and/or sample details, will be carried over to be accessible in processing and post-processing modules.

* __Raw classes table__: This is a table of classes categories in the study. `TLA setup` loads a "raw" classes table with the following fields:
	1. `class`: (str) unique identifier for each class, must be the same labels used in `coord_file` data files. 
	2. `class_name`: (str) long (neat) name for each category, to be used in plots
	3. `class_val`: (int) unique numeric value of class (deprecated)
	4. `class_color`: (str) pre-chosen color for displaying each class
	5. `drop`: If `TRUE` then this class will be dropped from the analysis.

In `TLA setup` pre-processing a new sample table, with processed-file names and locations is produced, with equivalent fields plus a set of new ones:

1. `raster_file`: name and location of raster images (compressed in npz format) generated from KDE smoothing, and quadrat-level cell density and mixing. 
2. `results_dir`: directory where analysis results will be dumped
3. `num_cells`: total number of points in this sample
4. `shape`: [numrows, numcols] size of tissue landscape in pixels

Any other information in additional columns will be carried over, for example clinical/patient information that would be relevant in a cohort analysis. 

Similarly, for each study a table of classes is saved, containing only kept classes, and displaying the limits for cell abundance and mixing; these limits can be modified by hand before running the proper TLA analysis, as the edges are defined in a standard way that might not be the best.


#### Local abundance and mixing scores:

The study parameter `binsiz` defines a quadrat size, which is used to coarse grain the landscape mosaic in order to access local properties of cell abundance and cell mixing (uniformity of spatial distribution). In the setting of field ecology, quadrats are used to quantify small regions and produce spatial profiles across a landscape. Using this same principle we grid our landscape and count the abundance ${N_c}$ of each cell type $c$ in each quadrat, as well as the value of a mixing index for each cell type, defined as:
<div align="center">
$ M_c = \frac{2 \cdot\sum{n_i \cdot m_i}}{\sum{(m_i)^2} + \sum{(n_i)^2}} = \frac{2}{1 + (L/N_c^2)\sum{n_i^2}}$
</div>

Calculated over $L$ sub-quadrats which are 5 times smaller that the quadrats (and thus $L = 25$). This is a univariate version of the Morisita-Horn score ([Horn, 1966](https://www-jstor-org.ezproxy1.lib.asu.edu/stable/2459242)) comparing the observed spacial profile of cell counts $n_i$ with an array of the same size and a uniform distribution $m_i = $ constant (with $N_c=\sum{n_i}=\sum{m_i}$). This score is a simple way to account the degree of mixing (uniformity) of cells in a sample. A value $M_c \sim 0$ means that the sample is highly segregated (ie. variance across sub-quadrats is large) and a value $ M_c\sim 1$ means that all sub-quadrats have very similar count values and thus, cells are uniformly distributed across the quadrat.

#### Defining LME edges:

Using the `quadrat_stats` plot saved into the `results` folder for this study, we observe a distributions of quadrat-level values for cell density and mixing score across all samples in the study. Red lines correspond with the limits presented in the class table. 

![Quadrat Stats](documentation/quadrat_stats.png)

In the case of cell abundance, the edges are automatically picked at quantiles __[0.0, 0.5, 0.87, 1.0]__ while mixing index edges are picked at quantiles __[0.0, 0.2, 0.67, 1.0]__. __These are totally arbitrary values and it is recommended to check the distribution plots to confirm and adjust to proper values according to the interest in the research study__. The TLA method expects three levels (representing low, medium and high values) for each of these variables in each class (different classes typically have different limits), yielding 9 categories per cell class. For a study with 3 or more cell types this corresponds to hundreds of unique categories, which is not very practical.

For simplicity, LME classes are defined in three general categories (for each cell type) encompassing both the abundance and mixing levels in the local region of study:

![abumix](documentation/abumix.png)

1. __(B) Bare__ environments are those with few cells, regardless of mixing.
2. __(S) Segmented__ environments are those in which cells are clustered together (moderate to high abundance and low mixing).
3. __(M) Mixed__ environments are those where cells are mixed uniformly (moderate to high abundance and high mixing).

Because the mixing score is sensitive to low abundance, medium mixing levels are considered "mixed" (_M_) only if the abundance is large (and thus the mixing score is credible).

With is scheme, __(B,S,M)__ codes are assigned for each individual class, forming categories like: `BBB, BBS, BBM …. MMM`. Each of these categories represent a basic type of Local Micro-Environment that is consistent across the entire study, and is used to define ecological phenotypes (patches) in the TLA.

__NOTE__: all pre-processing calculations for each sample are cached in files stored locally in the `results` folder (specified in the arguments table). If case new samples are added to the study, and new study-level statistics needs to be calculated, in particular quadrat-level distributions, you just need to rerun the `setup` and cached data will be loaded instead of re-calculated. IF for any reason you need to redo the entire pre-processing, use the option `TRUE` in the third (optional) argument: 

```./TLA setup study.csv TRUE```

or alternatively, run the clean module: `./TLA clean study.csv`, which will delete all results files in an orderly matter.


### Processing Module `TLA run`

Usage: `./TLA run study.csv` with `study.csv` the name of the argument csv file.

This module reads the samples and classes tables generated by the setup module, and run the main Tumor Landscape Analysis. All results are put in corresponding folders for each sample within the study output folder. Study-level statistics are also generated, including plots and a `_samples_stats.csv` table with general results for all samples. Several tables with numeric results are generated for use in post-processing (which typically includes comparisons between groups of samples according to some clinical annotation or outcome classification).

All result outputs are explained in the outputs documentation.

__NOTE__: similar as before, most results are cached for faster re-running if only need to re-plot results. IF for any reason you need to redo the entire processing, use the option `TRUE` in the third (optional) argument: 

```./TLA run study.csv TRUE``` 


### Stratification Module `TLA ssh`

Usage: `./TLA ssh study.csv` with `study.csv` the name of the argument csv file.

This runs Spatial Stratified Heterogeneity for all the different spatial statistic factors produced in the TLA, as well as their interactions and risk assessments in relation to their combinations. This is done in principle for patches defined by LMEs, but this is not really a very interesting result, as LMEs are defined by abundance/mixing, and all the spatial factors so far implemented are based on some form of intermixing or colocalization of cell types. Therefore at this point this is a redundant analysis (and why this module is separate). 

But when _ad hoc_ blobs (eg. pre-defined regions like ducts, crypts or DCIS), we have an __additional__ patch definition that can be interesting to test in the SSH method. This will tell us is any of the spatial factors stratify with respect to these tissue structures. If blob masks were included in the pre-processing step, this program will automatically generate these analyses, otherwise, it's done only for the LME patches.



  



