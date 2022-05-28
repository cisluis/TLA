# Tumor Landscape Analysis (TLA)
 
Landscape ecology analysis methods for the investigation and characterization of digital histopathological data of tumor biopsies.


## Introduction

Landscape ecology is the study of interactions and relationships between living organisms and the environment they inhabit. Such environment is defined as a __landscape__ (spread in space and time) occupied by different species of organisms, and its description entails details on spatial distributions, cohabitation, population dynamics, mobility and other ecological questions.

A __landscape mosaic__ consist of spatial locations and categorical classification of all ecological entities of interest. Typically such data is recorded as a __categorical raster image__ (a 2D array with discrete categorical labels representing locations of all surveyed individuals in a particular moment in time, with different categories representing species or classes), or an equivalent long-form table of individuals with their corresponding coordinates and category values. 

Landscape metrics algorithms have been implemented in packages such as [FRAGSTATS](http://www.umass.edu/landeco/research/fragstats/fragstats.html), [landscapemetrics](https://r-spatialecology.github.io/landscapemetrics/) (R) or [pylandstats](https://pylandstats.readthedocs.io/en/latest/index.html) (python) support raster spatial objects. Because these algorithms work with categorical data, each cell in the array is assigned to a discrete class identifier. Therefore there is an inherent spatial resolution of the data given by the discrete location values. We will refer to this resolution as the "pixel resolution" of the data.

__Landscape metrics__ are measures that quantify physical characteristics of landscape mosaics in order to connect them to ecological processes. These tools help characterize a landscape, primary describing its composition and spatial configuration: 

- The __composition__ of a landscape accounts for how much of the landscape, or a specified region of it, is covered by a certain category type
- The __configuration__ describes the spatial arrangement or distribution of the different category types. 

Additionally, landscape metrics can be calculated for three different levels of information scopes:

1.	__Patch level metrics:__ a patch is defined as neighboring cells belonging to the same class, typically using Moore's, or sometimes von Neumann's, neighborhood rules. Patch level metrics are calculated for each patch in the landscape.
2. __Class level metrics__: returns a summary value for all patches aggregated by type class. The output is typically some statistics of patch level metrics across all patches in each class (e.g. a sum or mean). 
3. __Landscape level metrics__: returns a single value describing a global property of the landscape. This is typically a statistics of metrics of lower levels aggregated by patches and/or classes. 

Our goal is to implement these methodologies in the study of tissues, perceived as cellular ecologies in the context of tumor development, and observed by means of digital histopathological samples. The data used for this analysis typically comes from a cell segmentation and classification process, which consist of image processing and machine learning algorithms that are capable of identifying individual cells from a histopathological image, returning the pixel location of their centers and a class category that indicates the cell type. This data is provided to TLA as numerical data. Additional data consist of _ad hoc_ segmentation of regions in the tissue, also detected by image processing techniques, that identify specific tissue compartments of interest; for instance ducts or crypts where neoplastic epithelial cells might form tumors or cellular niches.


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

Toolboxes like __pylandstats__ (which is implemented within TLA) feature six specific distribution metrics for each patch-level metric, consisting of statistical aggregation of the values computed for each patch of a class or the whole landscape.

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
 

---
---
---


## TLA pipeline usage guideline

TLA is a python program that computes spatial statistics, implementing functions from the landscape ecology package [pylandstats](https://github.com/martibosch/pylandstats), astronomical and GIS spatial statistics ([astropy](https://www.astropy.org/), [pysal](https://pysal.org/esda/index.html)), spatial stratified heterogeneity ([geodetector](https://cran.r-project.org/web/packages/geodetector/vignettes/geodetector.html)) and image processing methods ([scipy](https://scipy.org/), [scikit-image](https://scikit-image.org/)).

Two novel aspects are of significance in TLA:

1. Spatial statistics are calculated locally. This is generally done using a quadrat approach, which produces spatial profiles of different metrics, rather than a single sample-level value.
2. The quadrat approach inherently reduces the resolution of the landscape, as any feature with a length scale smaller than the size of a quadrat can't be resolved. But adopting the principle of spatial smoothing from image processing, using spatial convolution functions with a circular or gaussian kernel instead of a quadrat grid, all statistics are calculated at the pixel level. 


### TLA script

Bash script `TLA` is part of the TLA repository and is all what is really needed to run all the analyses. 

There are three basic TLA modules (or actions): 

1. `TLA setup` pre-processes data, including cell filtering, creation of raster arrays and estimation of study-level profiles for cell density and mixing, which are used to define local microenvironment categories (LME) that are consistent across the whole cohort.
2. `TLA run` runs the main TLA analysis, including calculation of spacial statistic factors and patch analysis using defined LMEs. Patch, class and landscape level statistics are performed and recorded in output tables.
3. `TLA ssh` runs spatial stratified heterogeneity analysis using previously calculated spatial statistic factors. 

Since cohort-level analysis depends of details of the study design (eg. cohorts or groups of progression and controls) which specify the type of desired comparisons and statistics, it is left out of this distribution of TLA. Specific projects would typically have  tailored post-processing scripts written in R for this purpose. 


Usage: 

If the virtual environment is not yet activated:

```
> conda activate
``` 

then, each module is ran with the syntax: 

```
> ./TLA {action} {argument table} [TRUE/FALSE]

``` 

#### Argument table

This is a comma separated file (CSV) containing the main arguments for all TLA modules, and it must be produced by the user. The example `test_set.csv` is provided as a template. 

Typically this table would have only one row containing argument values for a single study. On the other hand a study includes a number of biopsies to be analyzed and compared together. 

TLA allows for batch processing of multiple studies if several rows are added in this table. Notice that the assumption is that there is no connection between different studies, so no comparisons or joined statistics will be calculated between them. 

The arguments for a study are:

1. `name`: (str) name to identify the study.
2. `data_path`: (str) general path to the location of all the study data (all sample paths are relative to this one)
3. `raw_samples_table`: (str) name of raw samples table (CSV). This table has all the relevant information for each individual sample in the study. 
4. `raw_classes_table`: (str) name of classes table for this study.
5. `scale`: (float) scale of pixels in physical units (units/pixel)
6. `units`: (str) name of physical units (e.g. `[um]`) 
7. `binsiz`: (float) size of quadrat (kernel) binning for coarse graining.
8. `BLOBS`: (bool) if `True` then a mask image with _ad hoc_ regions of interest will be used to mask cells. In this case cells outside of blobs are reclassified as `LOW_DENS_CODE` and those inside blobs are reclassified as `HIGH_DENS_CODE`. 
9. `DTHRES`: (float) threshold for filtering target cell type according to cell density. Use the value `0` to turn this feature off.
10. `FILTER_CODE`: code of targeted cells for filtering (typically tumor or epithelial cells)
11. `HIGH_DENS_CODE`: code assigned to targeted cells in high density areas (and inside regions of interest defined by the blob mask)
12. `LOW_DENS_CODE`: code assigned to target cells in low density areas (or outside regions of interest defined by the blob mask)


## Pre-Processing Module: `TLA setup`

Usage example: 

```
> ./TLA setup test_set.csv

``` 

Prepares data for TLA according to parameters in the argument table:

#### Samples table:

The argument __raw\_samples\_table__ points to a table of samples with the following fields:

1. `sample_ID`: (str) unique identifier for each sample.
2. `coord_file`: (str) name of the coordinate file (CSV) for this sample. These are raw coordinates that will be curated during pre-processing. __ATTENTION:__ The format of this file __must be__  `['class', 'x', 'y']`, with these _exact_ column names; any additional info in this file (like annotations for each cell) will be ignored. Cleaned out data files are cached to be used in downstream analysis (including the additional info). BUT only cell classes defined in the classes table and cells that pass the filtering process are kept. Also for practical convenience coordinate values are reset to relevant margins.
3. `image_file`: location and name of the image file for this sample. These images are adjusted to match coordinates of the coordinates convex hull, and copies are cached. If an image is not available leave the field blank.
4. `mask_file`: location and name of the corresponding mask file for this sample. Mask are used to identify large scale regions (blobs), like ducts, crypts, DCIS, different tissue components, etc. It can be either a binary mask or a raster image with integer labels identifying different blob patches. These images are also adjusted and cached. If a mask image is not available leave the field blank.
5. Additional variables, like clinical annotations and/or sample details, will be carried over to result outputs to be accessible by processing and post-processing modules.

#### Classes table:

The argument __raw\_classes\_table__ points to a table of cell categories with the following fields:

1. `class`: (str) unique identifier for each class, must be the same labels used in `coord_file` data files. 
2. `class_name`: (str) long (neat) name for each category, to be used in plots
3. `class_val`: (int) unique numeric value of class (deprecated)
4. `class_color`: (str) pre-chosen color for displaying each class
5. `drop`: If `TRUE` then this class will be dropped from the analysis.


#### Notes about cell filtering:

Filtering reassigns categories to cells found in regions where they are not expected, which would indicate that they were likely misclassified by the deep learning classifier. 

* Pre-processing separates a specific class of __target__ cells (`FILTER_CODE` e.g. tumor cells) into two separate classes according to the local density. This is done because, for instance, tumor cells are not typically expected to be found in low densities, so they are probably misclassified by the machine learning cell classifier (false positives).
* The parameter `DTHRES` is the density threshold, in units of cells per pixel, to select high density cells (set `DTHRES=0` to turn this feature off).
* Additionally, if a mask for regions of interest is provided (e.g., segmented ducts, crypts or other tissue compartments where tumor cells are expected to exist), set `BLOBS=True` to use these as the filter. In this case, target cells inside the masked blobs will be assigned as `HIGH_DENS_CODE` and the ones outside will be assigned `LOW_DENS_CODE`. 
* If __both__ filter features are set, the density filter is applied only inside the blobs, _i.e._, density filter is applied in regions of interest only, and all cells outside the blobs are set to  `LOW_DENS_CODE`.

Density filtering is done using a Kernel Density Estimator (KDE) algorithm, which is a spatial convolution function to estimate local point densities. 

#### Output of pre-processing:

`TLA setup` produces a new sample table with processed-file names and locations and cached in a `data/` subfolder in the study folder. The fields are equivalent to the original sample table plus the following fields:

1. `raster_file`: name and location of raster images (compressed in NPZ format) generated from KDE smoothing, and quadrat-level cell density and mixing. 
2. `results_dir`: directory where analysis results will be dumped
3. `num_cells`: total number of (approved) points in the sample
4. `shape`: size of tissue landscape in pixels: _[num rows, num cols]_

Similarly, for each study a table of approved classes is saved. This table will also display limits for cell [abundance and mixing profiles](#### Local abundance and mixing scores:); these limits can be modified by hand before running the TLA analysis, as they are arbitrarily calculated.


#### Local abundance and mixing scores:

The study parameter `binsiz` defines a "quadrat" size that is used to coarse grain the landscape mosaic for local properties of cell abundance and uniformity of spatial distribution (mixing). In field ecology, quadrats are typically used to quantify small regions and produce spatial profiles across a landscape. Using this same principle we grid our landscape and count the abundance _N<sub>c</sub>_ of each cell type _c_ in each quadrat, as well as the value of a mixing index for each cell type, defined as:
<div align="center">
<img src="https://latex.codecogs.com/gif.latex? M_c = \frac{2 \cdot\sum{n_i \cdot m_i}}{\sum{(m_i)^2} + \sum{(n_i)^2}} = \frac{2}{1 + (L/N_c^2)\sum{n_i^2}}" /> </div>

Calculated over _L_ sub-quadrats which are 5 times smaller that the quadrats (and thus _L_=25). This is a univariate version of the Morisita-Horn score ([Horn, 1966](https://www-jstor-org.ezproxy1.lib.asu.edu/stable/2459242)) comparing the observed spacial profile of cell counts _n<sub>i</sub>_ with an array of the same size and a uniform distribution _m<sub>i</sub>_ = constant and 
<div align="center">
<img src="https://latex.codecogs.com/gif.latex? N_c=\sum{n_i}=\sum{m_i}" /> </div>

This score is a simple way to account the degree of mixing (uniformity) of cells in a sample. A value _M<sub>c</sub>_ ~ 0 means that the sample is highly segregated (ie. variance across sub-quadrats is large) and a value _M<sub>c</sub>_ ~ 1 means that all sub-quadrats have very similar count values and thus, cells are uniformly distributed across the quadrat.

#### Defining LME edges:

Using the `quadrat_stats` plot saved into the `data/results/` folder for the study, we observe a distributions of quadrat-level values for cell density and mixing score across all samples in the study. Red lines correspond with the limits presented in the class table. 

![Quadrat Stats](quadrat_stats.png)

In the case of cell abundance, the edges are automatically picked at quantiles __[0.0, 0.5, 0.87, 1.0]__ while mixing index edges are picked at quantiles __[0.0, 0.2, 0.67, 1.0]__. __These are totally arbitrary values and it is recommended to check the distribution plots to confirm and adjust to proper values according to the interest in the research study__. The TLA method expects three levels (representing low, medium and high values) for each of these variables in each class (different classes typically have different limits), yielding 9 categories per cell class. For a study with 3 or more cell types this corresponds to hundreds of unique categories, which is not very practical.

For simplicity, LME classes are defined in three general categories (for each cell type) encompassing both the abundance and mixing levels in the local region of study:

<div align="center">
<img src="abumix.png" width="200" height="200">
</div>



1. __(B) Bare__ environments are those with few cells, regardless of mixing.
2. __(S) Segmented__ environments are those in which cells are clustered together (moderate to high abundance and low mixing).
3. __(M) Mixed__ environments are those where cells are mixed uniformly (moderate to high abundance and high mixing).

Because the mixing score is sensitive to low abundance, medium abundance levels are considered "mixed" (_M_) if the mixing is medium, as mixing is biased to lower values with lower abundances.

With is scheme, __(B,S,M)__ codes are assigned for each individual class, forming categories like: `BBB, BBS, BBM …. MMM`. Each of these categories represent a basic type of Local Micro-Environment that is consistent across the entire study, and is used to define ecological phenotypes (patches) in the TLA.

#### Option to redo cached data:

All pre-processing calculations for each sample are cached in files stored locally in the `data` folder (specified in the arguments table). If new samples are added to the study, and thus study-level statistics needs to be re-calculated, you just need to rerun the `setup` and cached data will be loaded instead of re-calculate it again for all samples. But if for any reason you need to redo the entire pre-processing, pass the value `TRUE` in the third (optional) argument to the main script: 

```./TLA setup test_set.csv TRUE```

or alternatively, run the clean module: `./TLA clean test_set.csv`, which will wipe out all cached results files in an orderly matter.


## Processing Module `TLA run`

Usage: 

```
> ./TLA run test_set.csv
```

This module reads the samples and classes tables generated by the setup module, and run the main Tumor Landscape Analysis. All results are put in corresponding folders for each sample within the study output folder. Study-level statistics are also generated, including plots and a `_samples_stats.csv` table with general results for all samples. Several tables with numeric results are generated for use in post-processing (which typically includes comparisons between groups of samples according to some clinical annotation or outcome classification).

All result outputs are explained in the outputs documentation.

__NOTE__: similar as before, most results are cached for faster re-running if you only need to re-plot results. If you need to redo the entire processing, use the option `TRUE` in the third (optional) argument: 

```> ./TLA run test_set.csv TRUE``` 


## Stratification Module `TLA ssh`

Usage: 

```
>./TLA ssh test_set.csv
```

This runs Spatial Stratified Heterogeneity for all the different spatial statistic factors produced in the TLA, as well as their interactions and risk assessments in relation to their combinations. This is done in principle for patches defined by LMEs, but this is not really a very interesting result, as LMEs are defined by abundance/mixing, and all the spatial factors so far implemented are based on some form of intermixing or colocalization of cell types. Therefore at this point this is a redundant analysis (and why this module is separate). 

But when _ad hoc_ blobs (eg. pre-defined regions like ducts, crypts or DCIS), we have an __additional__ patch definition that can be interesting to test in the SSH method. This will tell us is any of the spatial factors stratify with respect to these tissue structures. If blob masks were included in the pre-processing step, this program will automatically generate these analyses, otherwise, it's done only for the LME patches.



  



