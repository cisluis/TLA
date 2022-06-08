'''
    Tumor Landscape SSH Analysis (TLA):
    ##############################

        This script reads lines from a study set table
        Each line has parameters of a particular study

        This formating helps faster and more consistent running of the TLA


'''

# %% Imports
import os
import sys
import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from skimage import io
from PIL import Image
from ast import literal_eval

from argparse import ArgumentParser

from myfunctions import printProgressBar, plotRGB
from itertools import combinations, permutations, product

Image.MAX_IMAGE_PIXELS = 600000000

__version__  = "1.0.0"

# %% Private classes

class Study:
    
  def __init__(self, study, main_pth):
      
      # loads arguments for this study
      self.name = study['name']
      self.dat_pth = os.path.join(main_pth, study['data_path'])
      if not os.path.exists(self.dat_pth):
          print("ERROR: data folder " + self.dat_pth + " does not exist!")
          print("... Please run << TLA setup >> for this study!")
          sys.exit()
      
      aux = os.path.join(self.dat_pth, study['name'] + '_classes.csv')
      if not os.path.exists(aux):
          print("ERROR: classes file " + aux + " does not exist!")
          sys.exit()
      self.classes = getLMEclasses(aux)
      
      aux = os.path.join(self.dat_pth, study['name'] + '_samples.csv')
      if not os.path.exists(aux):
          print("ERROR: samples file " + aux + " does not exist!")
          sys.exit()
      self.samples = pd.read_csv(aux)
     
      # the size of quadrats and subquadrats
      self.binsiz = int((study['binsiz']))
      self.subbinsiz = int(self.binsiz/5)
    
      # bandwidth size for convolutions is half the quadrat size
      self.kernel = int(self.binsiz/2)
      self.subkernel = int(self.subbinsiz/2)
      self.scale = study['scale']
      self.units = study['units']
      
      # creates samples table for output
      self.samples_out = pd.DataFrame()
      

class Landscape:
    
  def __init__(self, sample, study):
      
      dat_pth = study.dat_pth
      
      self.sid = sample.sample_ID
      self.res_pth = os.path.join(dat_pth, sample['results_dir'])
      
      aux = os.path.join(self.res_pth, 'classes.csv')
      if not os.path.exists(aux):
          print("ERROR: classes file " + aux + " does not exist!")
          sys.exit()
      self.classes = pd.merge(pd.read_csv(aux), 
                              study.classes, 
                              how="left",
                              on=['class', 'class_name',
                                  'class_val', 'class_color']).fillna(0)
      
      aux = os.path.join(dat_pth, sample['coord_file'])
      if not os.path.exists(aux):
          print("ERROR: data file " + aux + " does not exist!")
          sys.exit()
      self.cell_data = pd.read_csv(aux)
      self.ncells = len(self.cell_data)
      
      # general attributes
      self.binsiz = study.binsiz
      self.subbinsiz = study.subbinsiz 
      self.kernel =  study.kernel
      self.subkernel = study.subkernel
      self.scale = study.scale
      self.units = study.units
      
      # loads all raster images
      fil = os.path.join(dat_pth, sample['raster_file'])
      if not os.path.exists(fil):
          print("ERROR: raster file " + fil + " does not exist!")
          sys.exit()
      aux = np.load(fil)
      self.roiarr = aux['roi'] # ROI raster
      self.kdearr = aux['kde'] # KDE raster (smoothed density landscape)
      self.abuarr = aux['abu'] # local abundances of each class
      self.mixarr = aux['mix'] # local mixing values of each class
      self.imshape = self.roiarr.shape
      
      # slide image
      img = None
      if (sample['image_file'] != ''):
          img = io.imread(os.path.join(dat_pth, sample['image_file']))
      self.img = img

      # loads blob mask
      msk = None
      if (sample['mask_file'] != ''):
          msk = np.load(os.path.join(dat_pth, sample['mask_file']))['mask']
      self.msk = msk
      
      # more raster attributes
      self.lmearr = []     # LME raster
      self.colocarr = []   # Colocalization array
      self.nndistarr = []  # NNDist array
      self.rhfuncarr = []  # Ripley's H array
      self.geordGarr = []  # Getis Ord G* arra
      self.hotarr = []     # HOT array
      
      # pylandstats landscape object
      self.plsobj = []

    
# %% Private Functions

# %%%% Setup

def classComps(classes):
    """
    Comparisons for different multivariate metrics

    Parameters
    ----------
    - classes: (pandas) dataframe with cell classes

    """

    # comparisons for different multivariate metrics

    from itertools import combinations, permutations, product

    codes = classes['class'].to_list()

    # (1) combination comparisons
    comb_comps = [list(cc) for cc in list(combinations(codes, 2))]

    # (2) permutations comparisons
    perm_comps = [list(cc) for cc in list(permutations(codes, 2))]

    # (3) product comparisons
    prod_comps = [list(cc) for cc in list(product(codes, repeat=2))]

    return([comb_comps, perm_comps, prod_comps])


def progressBar(i, n, step, Nsteps,  msg, msg_i):
    
    printProgressBar(Nsteps*i + step, Nsteps*n,
                     suffix=msg + ' ; ' + msg_i, length=50)
    

# %%%% LME functions

def getLMEclasses(classes_file):
    """
    Checks that the number of levels for LME definition is not too high

    Parameters
    ----------
    - classes_file: (str) name of file with cell classes definitions
    """

    classes = pd.read_csv(classes_file,
                          converters={'abundance_edges': literal_eval,
                                      'mixing_edges': literal_eval})

    nedges = classes['abundance_edges']
    medges = classes['mixing_edges']

    nmax = max([len(x) for x in nedges])
    mmax = max([len(x) for x in medges])

    if (nmax > 4) or (mmax > 4):
        print("ERROR: There are too many LME levels...")
        print("Please use less than 3 levels (4 edges)")
        sys.exit()

    return(classes)


def lmeCode(x, dim):
    """
    Return code for LME class, as a string of characters of length 'dim'.

    From the base-3 code for each digit (class type), the LME category is
    assigened according to following definition:
        - 0 => "B" is a "Bare" environment, with no or very few cells
        - 1 => "S" is a "Segmented" environment, where cells are not
          mixed (uniformly distributed)
        - 2 => "M" is a "Mixed" envirmentent, where cells are well seeded
          and mixed uniformly

    Parameters
    ----------
    - x: (int) lme numeric label (decimal > 0)
    - dim: (int) number of cell classes
    """

    lme = 'None'
    if (x>0):
        code = ['B', 'S', 'M']
        lme = ''
    
        def get_digit(number, n):
            return number // 3**n % 3
    
        for i in np.arange(dim):
            j = dim - (i + 1)
            lme = lme + code[get_digit(x - 1, j)]
            
    return(lme)


# %%%% SSH Statistics functions

def strataFormating(lmearr, msk, classes, strata):
    
    lmes = lmearr.copy()
    if 'LME' in strata:
        # format variables
        lmes[np.isnan(lmes)] = 0
        lmes = lmes.astype(int)
    
    blobs = msk.copy()
    if 'Blob' in strata:
        # format variables
        blobs[np.isnan(blobs)] = 0
        blobs = blobs.astype(int)
    
    return([lmes, blobs])


def SSH(data, classes, 
        fact_col, factlab, 
        strata_cols, strlabs, out_pth):
    """
    """
    
    from myfunctions import SSH_factor_detector_df
    from myfunctions import SSH_risk_detector_df
    from myfunctions import SSH_interaction_detector_df
    from myfunctions import SSH_ecological_detector_df

    # SSH, factor detector for both strata
    sshtab = SSH_factor_detector_df(data, fact_col, strata_cols)
    
    fig, ax = sshPlotStratification(sshtab, data, factlab, strlabs)
    
    lmeticks = [lmeCode(x, len(classes)) for x in ax[0].get_xticks()]
    ax[0].set_xticklabels(lmeticks)
    plt.savefig(os.path.join(out_pth, 'SSH_factor_' + fact_col + '.png'),
                bbox_inches='tight', dpi=300)
    plt.close()
    
    # SSH, significant risk differences between strata levels
    for stt in strata_cols:
        sshrsk = SSH_risk_detector_df(data, fact_col, stt)
        
        if len(sshrsk) > 0:
            if stt == "LME":
                aux = sshrsk['stratum_i']
                sshrsk['stratum_i'] = sshrsk['stratum_i'].astype(str)
                sshrsk['stratum_i'] = [lmeCode(x, len(classes)) for x in aux]
                aux = sshrsk['stratum_j']
                sshrsk['stratum_j'] = sshrsk['stratum_i'].astype(str)
                sshrsk['stratum_j'] = [lmeCode(x, len(classes)) for x in aux]    
            sshrsk = sshrsk.loc[sshrsk['significance']]
            fil = os.path.join(out_pth,
                               'SSH_risk_' + fact_col + '-' + stt +'.csv')
            sshrsk.to_csv(fil, index=False)
    
    sshint = pd.DataFrame()
    ssheco = pd.DataFrame()
    if (len(strata_cols) > 1):
        # SSH, interaction between strata
        sshint = SSH_interaction_detector_df(data, fact_col, strata_cols, 
                                             sshtab['q_statistic'],
                                             sshtab['p_value'])
        
        # SSH, overall significance test of impact difference between strata
        ssheco = SSH_ecological_detector_df(data, fact_col, strata_cols)

    return([sshtab, sshint, ssheco])


def sshPlotStratification(sshtab, data, factlab, strlabs):

    import seaborn as sns

    fig, ax = plt.subplots(int(np.ceil(len(sshtab))), 1,
                           figsize=(10, (len(sshtab))*5),
                           facecolor='w', edgecolor='k')
    
    shp = ax.shape
    for i, row in sshtab.iterrows():
        sns.axes_style("whitegrid")
        sns.violinplot(x=row.strata,
                       y=row.factor,
                       data=data,
                       ax=ax[np.unravel_index(i, shp)],
                       palette="Set3",
                       scale='count',
                       inner='box')
        sig = '(ns)'
        if (row.p_value < 0.05):
            sig = '(*)'
        if (row.p_value < 0.01):
            sig = '(**)'
        if (row.p_value < 0.001):
            sig = '(***)'
        txt = 'N = ' + str(len(data)) + '; p_value = %.2e ' % row.p_value
        ax[np.unravel_index(i, shp)].set_title(txt + sig)
        ax[np.unravel_index(i, shp)].set_ylabel(factlab)
        ax[np.unravel_index(i, shp)].set_xlabel(strlabs[i])
        
        tcks = ax[np.unravel_index(i, shp)].get_xticks()
        ntcks = len(tcks) 
        if (ntcks > 30):
            ax[np.unravel_index(i, shp)].set_xticks(tcks[::ntcks//30])

    plt.tight_layout()
    
    return([fig, ax])


def sshSubs(sid, raster, 
            imshape, scale, units, binsiz, 
            field_name, comps, classes, 
            lmes, blobs, strata, 
            out_pth, subset = True, para = False):
    """
    """
    from joblib import Parallel, delayed
    
    sshtab = pd.DataFrame()
    sshint = pd.DataFrame()
    ssheco = pd.DataFrame()
    
    # define quadrats (only consider full quadrats)
    redges = np.arange(binsiz, imshape[0], binsiz)
    if (max(redges) > (imshape[0] - binsiz)):
        redges = redges[:-1]
    cedges = np.arange(binsiz, imshape[1], binsiz)
    if (max(cedges) > (imshape[1] - binsiz)):
        cedges = cedges[:-1]
        
    if (subset):
        R = np.array([raster[r, c, :] for r in redges for c in cedges])
        L = np.array([lmes[r, c] for r in redges for c in cedges])
        B = np.array([blobs[r, c] for r in redges for c in cedges])  
        
    def func(i):
        Y = R[:, i]
        # get valid values
        inx = ~np.isnan(Y)
        
        fact = field_name + '_' + classes['class'][i]
        fact_name = field_name + '(' + classes['class_name'][i] + ')'
            
        data = pd.DataFrame({fact: Y[inx]})
        if 'LME' in strata:
            data['LME'] = L[inx].astype(np.int16)
            plotFactorLandscape(sid, lmes, raster,
                                fact_name + ' in LMEs', fact_name,
                                imshape, scale, units, binsiz, 
                                fact + '_lmes', out_pth)
            
        if 'Blob' in strata:
            data['Blob'] = B[inx].astype(np.int16)
            plotFactorLandscape(sid, blobs, raster,
                                fact_name + ' in Blobs', fact_name,
                                imshape, scale, units, binsiz, 
                                fact + '_blobs', out_pth)
        
        # do SSH analysis on coloc factors
        [aux, auy, auz] = SSH(data, classes, 
                              fact, fact_name, strata, strata, out_pth)
        
        return([aux, auy, auz])
    
    if (para):
        
        #### NOT WORKING... Need to find a different way to do it
        
        # Parallel processing
        aux = Parallel(n_jobs=os.cpu_count(),
                       prefer="threads")(delayed(func)(i) for i in 
                                         range(len(comps)))
        for i, comp in enumerate(comps):
            sshtab = pd.concat([sshtab, aux[i][0]], ignore_index=True)
            sshint = pd.concat([sshint, aux[i][1]], ignore_index=True)
            ssheco = pd.concat([ssheco, aux[i][2]], ignore_index=True)
    
    else:
        # serial processing
        for i, comp in enumerate(comps):
            [aux, auy, auz] = func(i)
            sshtab = pd.concat([sshtab, aux], ignore_index=True)
            sshint = pd.concat([sshint, auy], ignore_index=True)
            ssheco = pd.concat([ssheco, auz], ignore_index=True)
        
    sshtab.to_csv(os.path.join(out_pth,
                               sid + '_ssh_factor_' + field_name + '.csv'), 
                  index=False)
    if (len(strata) > 1):
        sshint.to_csv(os.path.join(out_pth, 
                                   sid + '_ssh_interaction_' + \
                                       field_name + '.csv'), 
                      index=False)
        ssheco.to_csv(os.path.join(out_pth,
                                   sid + '_ssh_ecological_' + \
                                       field_name + '.csv'), 
                      index=False)

    return(0)


def sshComps(sid, raster, 
             imshape, scale, units, binsiz, 
             field_name, comps, classes, 
             lmes, blobs, strata, 
             out_pth, subset = True, para = False):
    """
    """
    from joblib import Parallel, delayed
    
    
    sshtab = pd.DataFrame()
    sshint = pd.DataFrame()
    ssheco = pd.DataFrame()
    
    # define quadrats (only consider full quadrats)
    redges = np.arange(binsiz, imshape[0], binsiz)
    if (max(redges) > (imshape[0] - binsiz)):
        redges = redges[:-1]
    cedges = np.arange(binsiz, imshape[1], binsiz)
    if (max(cedges) > (imshape[1] - binsiz)):
        cedges = cedges[:-1]
        
    if (subset):
        R = np.array([raster[r, c, :, :] for r in redges for c in cedges])
        L = np.array([lmes[r, c] for r in redges for c in cedges])
        B = np.array([blobs[r, c] for r in redges for c in cedges])    
    
    def func(comp):
        ia = comp[0]
        ib = comp[1]
        Y = R[:, ia, ib]

        # get valid values
        inx = ~np.isnan(Y)
        
        fact = field_name + '_' + \
            classes['class'][ia] + '_' +  classes['class'][ib]
        fact_name = field_name + '(' + \
            classes['class_name'][ia] + ':' + classes['class_name'][ib] + ')'
        
            
        data = pd.DataFrame({fact: Y[inx]})
        if 'LME' in strata:
            data['LME'] = L[inx].astype(np.int16)
            plotFactorLandscape(sid, lmes, raster,
                                fact_name + ' in LMEs', fact_name,
                                imshape, scale, units, binsiz, 
                                fact + '_lmes', out_pth)
                    
        if 'Blob' in strata:
            data['Blob'] = B[inx].astype(np.int16)
            plotFactorLandscape(sid, blobs, raster,
                                fact_name + ' in Blobs', fact_name,
                                imshape, scale, units, binsiz, 
                                fact + '_blobs', out_pth)
           
        # do SSH analysis on coloc factors
        [aux, auy, auz] = SSH(data, classes, 
                              fact, fact_name, 
                              strata, strata, out_pth)
        return([aux, auy, auz])
        
    
    if (para):
        
        #### NOT WORKING... Need to find a different way to do it
        
        # Parallel processing
        aux = Parallel(n_jobs=os.cpu_count(),
                       prefer="threads")(delayed(func)(comp) for comp in comps)
        
        for i, comp in enumerate(comps):
            sshtab = pd.concat([sshtab, aux[i][0]], ignore_index=True)
            sshint = pd.concat([sshint, aux[i][1]], ignore_index=True)
            ssheco = pd.concat([ssheco, aux[i][2]], ignore_index=True)
    
    else:
        # serial processing
        for i, comp in enumerate(comps):
                       
            [aux, auy, auz] = func(comp)
            sshtab = pd.concat([sshtab, aux], ignore_index=True)
            sshint = pd.concat([sshint, auy], ignore_index=True)
            ssheco = pd.concat([ssheco, auz], ignore_index=True)
            
            
    
    sshtab.to_csv(os.path.join(out_pth, 
                               sid + '_ssh_factor_' + field_name + '.csv'), 
                  index=False)
    if (len(strata) > 1):
        sshint.to_csv(os.path.join(out_pth,
                                   sid + '_ssh_interaction_' + \
                                       field_name + '.csv'), 
                      index=False)
        ssheco.to_csv(os.path.join(out_pth,
                                   sid + '_ssh_ecological_' + \
                                       field_name + '.csv'), 
                      index=False)

    return(0)  


# %%%% Plotting functions


def plotEdges(shape, binsiz, scale):
    """
    Gets edges of quadrats for ploting landscapes

    Parameters
    ----------
    - shape: (tuple) shape in pixels of TLA landscape
    - binsiz : (float) size of quadrats
    - scale: (float) scale of physical units / pixel

    """

    # define quadrats
    redges = np.arange(0, shape[0] + binsiz, binsiz)
    cedges = np.arange(0, shape[1] + binsiz, binsiz)

    # aspect ratio
    ar = np.max(cedges)/np.max(redges)

    # coordinates of quadrats (um)
    xedges = [np.around(b*scale, 2) for b in cedges]
    yedges = [np.around(b*scale, 2) for b in redges]

    return([ar, redges, cedges, xedges, yedges])



def plotFactorLandscape(sid, patcharr, factarr, ttl, ftitle,
                        shape, scale, units, binsiz, 
                        nam, res_pth):
    """
    Plot of landscape from raster

    Parameters
    ----------
    - sid: sample ID
    - patcharr: (numpy) LME or bloob raster image
    - factarr: (numpy) factor raster image
    - ttl: (str) title of plot
    - ftitle: (str) factor name
    - shape: (tuple) shape in pixels of TLA landscape
    - scale: (float) scale of physical units / pixel
    - units: (str) name of physical units (eg '[um]')
    - binsiz : (float) size of quadrats
    - res_pth: (str) results path

    """
    
    # get mean value of factor per patch
    raster = np.empty(shape)
    raster[:] = np.NaN
    for b in np.unique(patcharr[patcharr>0]):
        raster[patcharr == b] = np.mean(factarr[patcharr == b])
        
        
    [ar, redges, cedges, xedges, yedges] = plotEdges(shape, binsiz, scale)
    
    if len(raster[~np.isnan(raster)]) > 0:
        
        vmin = np.nanmin(raster)
        vmax = np.nanmax(raster)
    
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
    
            # plots sample image
            fig, ax = plt.subplots(1, 1, figsize=(12*1, 0.5 + math.ceil(12*1/ar)),
                                   facecolor='w', edgecolor='k')
            im = plotRGB(ax, raster, units,
                         cedges, redges, xedges, yedges, fontsiz=18,
                         vmin=vmin, vmax=vmax, cmap='jet')
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(ftitle, rotation=90, labelpad=1)
            ax.set_title(ttl, fontsize=18, y=1.02)
            fig.subplots_adjust(hspace=0.4)
            fig.suptitle('Sample ID: ' + str(sid), fontsize=24, y=.95)
            fig.savefig(os.path.join(res_pth, sid + '_' + nam + '_landscape.png'),
                        bbox_inches='tight', dpi=300)
            plt.close()

    return(0)


def plotCaseLandscape(sid, raster, classes, comps, shape,
                      metric, lims, scale, units, binsiz):
    """
    Plot of landscape from comparison raster

    Parameters
    ----------
    - sid: sample ID
    - raster: (numpy) raster image
    - classes: (pandas) dataframe with cell classes
    - comps: (list) class comparisons
    - shape: (tuple) shape in pixels of TLA landscape
    - metric: (str) title of metric ploted
    - lims: (tuple) limits of the metric
    - scale: (float) scale of physical units / pixel
    - units: (str) name of physical units (eg '[um]')
    - binsiz : (float) size of quadrats

    """

    dim = len(comps)
    nlevs = 10
    bini = (lims[1] - lims[0])/(2*nlevs)
    cticks = np.arange(lims[0], lims[1] + 2*bini, 2*bini)
    bticks = np.arange(lims[0], lims[1] + bini, bini)
    # vmax = 0

    [ar, redges, cedges, xedges, yedges] = plotEdges(shape, binsiz, scale)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        # plots sample image
        fig, ax = plt.subplots(dim, 2,
                               figsize=(12*2, 0.5 + math.ceil(12*dim/ar)),
                               facecolor='w', edgecolor='k')

        for i, comp in enumerate(comps):

            aux = raster[:, :, comp]

            im = plotRGB(ax[i, 0], aux, units,
                         cedges, redges, xedges, yedges, fontsiz=18,
                         vmin=lims[0], vmax=lims[1], cmap='jet')

            cbar = plt.colorbar(im, ax=ax[i, 0], fraction=0.046, pad=0.04)
            cbar.set_ticks(cticks)
            cbar.set_label(metric, rotation=90, labelpad=2)
            ax[i, 0].set_title(classes.class_name[comp],
                               fontsize=18, y=1.02)

            freq, _ = np.histogram(aux[~np.isnan(aux)], bins=bticks,
                                   density=False)
            vals = freq/np.sum(~np.isnan(aux))
            # vmax = np.max(np.append(vmax, vals))
            ax[i, 1].bar(bticks[:-1], vals,
                         width=bini, align='edge',
                         alpha=0.75, color='b', edgecolor='k')

            ax[i, 1].set_title(classes.class_name[comp],
                               fontsize=18, y=1.02)
            ax[i, 1].set_xlabel(metric)
            ax[i, 1].set_ylabel('Fraction of pixels')
            ax[i, 1].set_xlim(lims)
            ax[i, 1].set_xticks(cticks)
            # ax[i, 1].set_yscale('log')

        # for i in np.arange(len(comps)):
        #    ax[i, 1].set_ylim([0, 1.05*vmax])

        fig.subplots_adjust(hspace=0.4)
        fig.suptitle(metric + '\nSample ID: ' + str(sid),
                     fontsize=24, y=.95)

    return(fig)


def plotCompLandscape(sid, raster, classes, comps, shape,
                      metric, lims, scale, units, binsiz):
    """
    Plot of landscape from comparison raster

    Parameters
    ----------
    - sid: sample ID
    - raster: (numpy) raster image
    - classes: (pandas) dataframe with cell classes
    - comps: (list) class comparisons
    - shape: (tuple) shape in pixels of TLA landscape
    - metric: (str) title of metric ploted
    - lims: (tuple) limits of the metric
    - scale: (float) scale of physical units / pixel
    - units: (str) name of physical units (eg '[um]')
    - binsiz : (float) size of quadrats

    """

    dim = len(comps)
    nlevs = 10
    bini = (lims[1] - lims[0])/(2*nlevs)
    cticks = np.arange(lims[0], lims[1] + 2*bini, 2*bini)
    bticks = np.arange(lims[0], lims[1] + bini, bini)
    vmax = np.nanquantile(raster, 0.975) + 2*bini

    [ar, redges, cedges, xedges, yedges] = plotEdges(shape, binsiz, scale)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        # plots sample image
        fig, ax = plt.subplots(dim, 2,
                               figsize=(12*2, 0.5 + math.ceil(12*dim/ar)),
                               facecolor='w', edgecolor='k')

        for i, comp in enumerate(comps):

            ia = comp[0]
            ib = comp[1]
            aux = raster[:, :, ia, ib]

            im = plotRGB(ax[i, 0], aux, units,
                         cedges, redges, xedges, yedges, fontsiz=18,
                         vmin=lims[0], vmax=vmax, cmap='jet')

            cbar = plt.colorbar(im, ax=ax[i, 0], fraction=0.046, pad=0.04)
            cbar.set_ticks(cticks[cticks <= vmax])
            cbar.set_label(metric, rotation=90, labelpad=2)
            ax[i, 0].set_title(classes.class_name[ia] + '::' +
                               classes.class_name[ib],
                               fontsize=18, y=1.02)

            freq, _ = np.histogram(aux[~np.isnan(aux)], bins=bticks,
                                   density=False)
            vals = freq/np.sum(~np.isnan(aux))
            # vmax = np.max(np.append(vmax, vals))
            ax[i, 1].bar(bticks[:-1], vals,
                         width=bini, align='edge',
                         alpha=0.75, color='b', edgecolor='k')

            ax[i, 1].set_title(classes.class_name[ia] + '::' +
                               classes.class_name[ib],
                               fontsize=18, y=1.02)
            ax[i, 1].set_xlabel(metric)
            ax[i, 1].set_ylabel('Fraction of pixels')
            ax[i, 1].set_xlim(lims)
            ax[i, 1].set_xticks(cticks)
            # ax[i, 1].set_yscale('log')

        # for i in np.arange(len(comps)):
        #    ax[i, 1].set_ylim([0, 1.05*vmax])

        fig.subplots_adjust(hspace=0.4)
        fig.suptitle(metric + '\nSample ID: ' + str(sid),
                     fontsize=24, y=.95)

    return(fig)


    # %% Main function


def main(args):

    # %% debug start
    debug = False
    
    if debug:
        # running from the IDE
        # path of directory containing this script
        main_pth = os.path.dirname(os.getcwd())
        argsfile = os.path.join(main_pth, 'test_set.csv')
        #REDO = False
    else:
        # running from the CLI using bash script
        # path of directory containing this script
        main_pth = os.getcwd()
        argsfile = os.path.join(main_pth, args.argsfile)
        #REDO = args.redo

    print("==> The working directory is: " + main_pth)

    if not os.path.exists(argsfile):
        print("ERROR: The specified argument file does not exist!")
        sys.exit()
    args_tbl = pd.read_csv(argsfile)

    # Loops over all studies in the study set
    # *** Even though there is typically just one study in a set,
    #     this allows for running a set of analyses in a single call
    for k, st in args_tbl.iterrows():
        
        # construct a Study object with all required parameters
        study = Study(st, main_pth)
        
        if debug:
            study.samples = study.samples.iloc[:1]
            
        numsamples = len(study.samples.index)

        print("==> Processing study: " + study.name +
              "; [{0}/{1}]".format(k + 1, len(args_tbl)))
        
        
        factors = ['coloc', 'nndist', 'rhfunc', 'geordZ', 'abundance']
        strata = ['LME', 'Blob']

        # number of processing steps
        Nsteps = len(factors) + 1
        printProgressBar(Nsteps*(0), Nsteps*(numsamples), suffix='')

    
        # %% loops over samples in this study
        for index, sample in study.samples.iterrows():
            
            # SID for display
            sid = sample.sample_ID
            
            # creates a message to display in progress bar
            msg = '==> ' + sid + "; [{0}/{1}]".format(index + 1, numsamples)
            
            landpkl = os.path.join(study.dat_pth,
                                   sample['results_dir'],
                                   sid +'_landscape.pkl')
            
            # %% STEP 1: loads pickled landscape data
            if (os.path.exists(landpkl)):
  
                progressBar(index, numsamples, 1, Nsteps, msg, 
                            'loading landscape data...')
                with open(landpkl, 'rb') as f:  
                    [land] = pickle.load(f) 
            else:
                print("==> ERROR! pickle image does not exist: " + landpkl +
                      "; Make sure you ran TLA for this sample...")
                sys.exit()

            out_pth = os.path.join(land.res_pth, 'SSH_Analysis')
            if not os.path.exists(out_pth):
                os.makedirs(out_pth)
                    
            [lmes, blobs] = strataFormating(land.lmearr, land.msk, 
                                            land.classes, strata)
            
            # %% STEP 2: SSH analysis: coloc
            progressBar(index, numsamples, 2, Nsteps, msg, 
                        'running SSH analysis: coloc...')
            if 'coloc' in factors:
                comps = [list(c) for c in 
                         list(combinations(land.classes.index, 2))]
                _ = sshComps(land.sid, land.colocarr, 
                             land.imshape, land.scale, land.units, land.binsiz,
                             'coloc', comps, land.classes, 
                             lmes, blobs, strata, out_pth, subset = True)
           
            # %% STEP 3: SSH analysis for nndist fields
            progressBar(index, numsamples, 3, Nsteps, msg, 
                        'running SSH analysis: nndist...')
            if 'nndist' in factors:
                comps = [list(c) for c in 
                         list(permutations(land.classes.index, r=2))]
                _ = sshComps(land.sid, land.nndistarr, 
                             land.imshape, land.scale, land.units, land.binsiz,
                             'nndist', comps, land.classes, 
                             lmes, blobs, strata, out_pth, subset = True)
           
            # %% STEP 4: SSH analysis for rhfunc fields
            progressBar(index, numsamples, 4, Nsteps, msg, 
                        'running SSH analysis: rhfunc...')
            if 'rhfunc' in factors:
                comps = [list(c) for c in 
                         list(product(land.classes.index, repeat=2))]
                _ = sshComps(land.sid, land.rhfuncarr, 
                             land.imshape, land.scale, land.units, land.binsiz,
                             'rhfunc', comps, land.classes, 
                             lmes, blobs, strata, out_pth, subset = True)
                         
            # %% STEP 5: SSH analysis for geordG fields
            progressBar(index, numsamples, 5, Nsteps, msg, 
                        'running SSH analysis: geordG...')
            if 'geordZ' in factors:
                comps = list(land.classes.index)
                _ = sshSubs(land.sid, land.geordGarr, 
                            land.imshape, land.scale, land.units, land.binsiz,
                            'geordZ', comps, land.classes, 
                            lmes, blobs, strata, out_pth, subset = True)
                _ = sshSubs(land.sid, land.hotarr, 
                            land.imshape, land.scale, land.units, land.binsiz,
                            'HOT', comps, land.classes, 
                            lmes, blobs, strata, out_pth, subset = True)

            # %% STEP 6: SSH analysis for abundance fields
            progressBar(index, numsamples, 6, Nsteps, msg, 
                        'running SSH analysis: abundance...')
            if 'abundance' in factors:
                comps = list(land.classes.index)
                _ = sshSubs(land.sid, land.abuarr, 
                            land.imshape, land.scale, land.units, land.binsiz,
                            'abundance', comps, land.classes, 
                            lmes, blobs, strata, out_pth, subset = True)

    # %% end
    
    print("==> Analysis finished!!! ")
    
    return(0)        
        

# %% Argument parser
if __name__ == "__main__":

    # Create the parser
    my_parser = ArgumentParser(prog="tla_ssh",
                               description="### Processing module for " +
                               "Tumor Landscape Analysis ###",
                               allow_abbrev=False)

    # Add the arguments
    my_parser.add_argument('-v', '--version', 
                           action='version', 
                           version="%(prog)s " + __version__)
    
    my_parser.add_argument('argsfile',
                           metavar="argsfile",
                           type=str,
                           help="Argument table file (.csv) for study set")

    my_parser.add_argument("--redo",
                           default=False,
                           action="store_true",
                           help="If --redo is used, re-do landscape analysis")

    # passes arguments object to main
    main(my_parser.parse_args())
