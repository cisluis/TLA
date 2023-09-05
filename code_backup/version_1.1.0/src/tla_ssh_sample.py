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
# from itertools import combinations, permutations, product

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
      
      # analyses table
      self.analyses = pd.read_csv(os.path.join(self.dat_pth,
                                               self.name + '_analyses.csv'),
                                  converters={'comps': literal_eval}) 
      
      
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

def progressBar(i, n, step, Nsteps,  msg, msg_i):
    
    printProgressBar(Nsteps*i + step, Nsteps*n,
                     suffix=msg + ' ; ' + msg_i, length=50)
    
def compIDX(comps, classes):
    
    def getIndex(c, df):
        return(df.index[df['class'] == c].tolist()[0])
    
    comps_idx = [(getIndex(c[0], classes), 
                  getIndex(c[1], classes)) for c in comps]
    
    return(comps_idx)
    

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


def SSH(sid, data, classes, 
        fact_col, factlab, 
        strata_cols, strlabs, out_pth, do_plots):
    """
    """
    
    from myfunctions import SSH_factor_detector_df
    from myfunctions import SSH_risk_detector_df
    from myfunctions import SSH_interaction_detector_df
    from myfunctions import SSH_ecological_detector_df

    # SSH, factor detector for both strata
    sshtab = SSH_factor_detector_df(data, fact_col,  strata_cols)
    
    if do_plots:
        fig, ax = sshPlotStratification(sshtab, data, factlab, strlabs)
        
        lmeticks = [lmeCode(x, len(classes)) for x in ax[0].get_xticks()]
        ax[0].set_xticklabels(lmeticks, rotation = 90)
        ax[1].set_xticklabels(ax[1].get_xticks(), rotation = 90)
        
        plt.savefig(os.path.join(out_pth, 
                                 sid + '_ssh_factor_' + fact_col + '.png'),
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
                               sid + '_ssh_risk_' + fact_col + \
                                   '-' + stt +'.csv')
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
            out_pth, do_plots, subset = True, para = False):
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
    else:
        R = np.array([raster[r, c, :, :] for r in np.arange(imshape[0]) \
                      for c in np.arange(imshape[1])])
        L = np.array([lmes[r, c] for r in np.arange(imshape[0]) \
                      for c in np.arange(imshape[1])])
        B = np.array([blobs[r, c] for r in np.arange(imshape[0]) \
                      for c in np.arange(imshape[1])])
        
    def func(i):
        Y = R[:, i]
        # get valid values
        inx = ~np.isnan(Y)
        
        fact = field_name + '_' + classes['class'][i]
        fact_name = field_name + '(' + classes['class_name'][i] + ')'
            
        data = pd.DataFrame({fact: Y[inx]})
        if 'LME' in strata:
            data['LME'] = L[inx].astype(np.int16)
            if do_plots:
                plotFactorLandscape(sid, lmes, raster[:, :, i],
                                    fact_name + ' in LMEs', fact_name,
                                    imshape, scale, units, 5*binsiz, 
                                    fact + '_lmes', out_pth)
                
        if 'Blob' in strata:
            data['Blob'] = B[inx].astype(np.int16)
            if do_plots:
                plotFactorLandscape(sid, blobs, raster[:, :, i],
                                    fact_name + ' in Blobs', fact_name,
                                    imshape, scale, units, 5*binsiz, 
                                    fact + '_blobs', out_pth)
        
        # do SSH analysis on coloc factors
        [aux, auy, auz] = SSH(sid, data, classes, 
                              fact, fact_name, 
                              strata, strata, out_pth, do_plots)
        
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
             out_pth, do_plots, subset = True, para = False):
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
    else:
        R = np.array([raster[r, c, :, :] for r in np.arange(imshape[0]) \
                      for c in np.arange(imshape[1])])
        L = np.array([lmes[r, c] for r in np.arange(imshape[0]) \
                      for c in np.arange(imshape[1])])
        B = np.array([blobs[r, c] for r in np.arange(imshape[0]) \
                      for c in np.arange(imshape[1])])
    
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
            if do_plots:
                plotFactorLandscape(sid, lmes, raster[:, :, ia, ib],
                                    fact_name + ' in LMEs', fact_name,
                                    imshape, scale, units, 5*binsiz, 
                                    fact + '_lmes', out_pth)
                    
        if 'Blob' in strata:
            data['Blob'] = B[inx].astype(np.int16)
            if do_plots:
                plotFactorLandscape(sid, blobs, raster[:, :, ia, ib],
                                    fact_name + ' in Blobs', fact_name,
                                    imshape, scale, units, 5*binsiz, 
                                    fact + '_blobs', out_pth)
           
        # do SSH analysis on coloc factors
        [aux, auy, auz] = SSH(sid, data, classes, 
                              fact, fact_name, 
                              strata, strata, out_pth, do_plots)
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
            fig, ax = plt.subplots(1, 1, 
                                   figsize=(12*1, 0.5 + math.ceil(12*1/ar)),
                                   facecolor='w', edgecolor='k')
            im = plotRGB(ax, raster, units,
                         cedges, redges, xedges, yedges, fontsiz=18,
                         vmin=vmin, vmax=vmax, cmap='jet')
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(ftitle, rotation=90, labelpad=1)
            ax.set_title(ttl, fontsize=18, y=1.02)
            fig.subplots_adjust(hspace=0.4)
            fig.suptitle('Sample ID: ' + str(sid), fontsize=24, y=.95)
            fig.savefig(os.path.join(res_pth, 
                                     sid + '_' + nam + '_landscape.png'),
                        bbox_inches='tight', dpi=300)
            plt.close()

    return(0)


    # %% Main function


def main(args):

    # %% debug starts
    debug = False
    
    if debug:
        # running from the IDE
        # path of directory containing this script
        main_pth = os.path.dirname(os.getcwd())
        argsfile = os.path.join(main_pth, 'DCIS_252_set.csv')
        REDO = False
        GRPH = False
        CASE = 0
    else:
        # running from the CLI using the bash script
        # path to working directory (above /scripts)
        main_pth = os.getcwd()
        argsfile = os.path.join(main_pth, args.argsfile) 
        REDO = args.redo
        GRPH = args.graph
        CASE = args.casenum

    print("==> The working directory is: " + main_pth)
    
    if not os.path.exists(argsfile):
        print("ERROR: The specified argument file does not exist!")
        sys.exit()
        
    # only the first study in the argument table will be used
    study = Study( pd.read_csv(argsfile).iloc[0], main_pth)
       
    factors = ['coloc', 'nndist', 'rhfunc', 'geordZ', 'abundance']
    strata = ['LME', 'Blob']
     
    # %% STEP 1: creates data directories and new sample table
    # creates sample object and data folders for pre-processed data
    sample = study.samples.iloc[CASE]
    
    # SID for display
    sid = sample.sample_ID
    
    msg = "====> Case [" + str(CASE + 1) + \
          "/" + str(len(study.samples.index)) + \
          "] :: SID <- " + sid 
                  
    sshpkl = os.path.join(study.dat_pth,
                          sample['results_dir'],
                          sid +'_ssh.pkl')    
        
    if REDO or (not os.path.exists(sshpkl)):
        
        from myfunctions import mkdirs
    
        landpkl = os.path.join(study.dat_pth,
                               sample['results_dir'],
                               sid +'_landscape.pkl')
        
        # %% STEP 1: loads pickled landscape data
        if (os.path.exists(landpkl)):
            with open(landpkl, 'rb') as f:  
                [land] = pickle.load(f) 
        else:
            print("==> ERROR! pickle image does not exist: " + landpkl +
                  "; Make sure you ran TLA for this sample...")
            sys.exit()
            
        print( msg + " >>> processing SSH..." )

        out_pth = mkdirs(os.path.join(land.res_pth, 'SSH_Analysis'))
                
        [lmes, blobs] = strataFormating(land.lmearr, land.msk, 
                                        land.classes, strata)
        
        # get analyses df
        df = study.analyses.copy()
        iscoloc = ~df.loc[df['name'] == 'coloc']['drop'].values[0]
        isnndist = ~df.loc[df['name'] == 'nndist']['drop'].values[0]
        isrhfunc = ~df.loc[df['name'] == 'rhfunc']['drop'].values[0]
        isgordG = ~df.loc[df['name'] == 'gordG']['drop'].values[0]
        
        # %% STEP 2: SSH analysis: coloc
        if ('coloc' in factors) and iscoloc:
            #comps = [list(c) for c in 
            #         list(combinations(land.classes.index, 2))]
            aux =  df.loc[df['name'] == 'coloc']['comps'].values[0]
            comps = compIDX(aux, land.classes)
            
            _ = sshComps(land.sid, land.colocarr, land.imshape, 
                         land.scale, land.units, land.subbinsiz,
                         'coloc', comps, land.classes, 
                         lmes, blobs, strata, out_pth, GRPH, subset = True)
       
        # %% STEP 3: SSH analysis for nndist fields
        if ('nndist' in factors) and isnndist:
           # comps = [list(c) for c in 
           #          list(permutations(land.classes.index, r=2))]
           aux = df.loc[df['name'] == 'nndist']['comps'].values[0]
           comps = compIDX(aux, land.classes)
           
           _ = sshComps(land.sid, land.nndistarr, land.imshape, 
                         land.scale, land.units, land.subbinsiz,
                         'nndist', comps, land.classes, 
                         lmes, blobs, strata, out_pth, GRPH, subset = True)
       
        # %% STEP 4: SSH analysis for rhfunc fields
        if ('rhfunc' in factors) and isrhfunc:
            #comps = [list(c) for c in 
            #         list(product(land.classes.index, repeat=2))]
            aux = df.loc[df['name'] == 'rhfunc']['comps'].values[0]
            comps = compIDX(aux, land.classes)
            
            _ = sshComps(land.sid, land.rhfuncarr, land.imshape, 
                         land.scale, land.units, land.subbinsiz,
                         'rhfunc', comps, land.classes, 
                         lmes, blobs, strata, out_pth, GRPH, subset = True)
                     
        # %% STEP 5: SSH analysis for geordG fields
        if ('geordZ' in factors) and isgordG:
            #comps = list(land.classes.index)
            aux = df.loc[df['name'] == 'gordG']['comps'].values[0]
            comps = compIDX(aux, land.classes)
            
            _ = sshSubs(land.sid, land.geordGarr, land.imshape, 
                        land.scale, land.units, land.subbinsiz,
                        'geordZ', comps, land.classes, 
                        lmes, blobs, strata, out_pth, GRPH, subset = True)
            _ = sshSubs(land.sid, land.hotarr, land.imshape, 
                        land.scale, land.units, land.subbinsiz,
                        'HOT', comps, land.classes, 
                        lmes, blobs, strata, out_pth, GRPH, subset = True)

        # %% STEP 6: SSH analysis for abundance fields
        if 'abundance' in factors:
            comps = list(land.classes.index)
            
            _ = sshSubs(land.sid, land.abuarr, land.imshape, 
                        land.scale, land.units, land.subbinsiz,
                        'abundance', comps, land.classes, 
                        lmes, blobs, strata, out_pth, GRPH, subset = True)
            
        # pickle results of quadrats analysis (for faster re-runs)
        with open(sshpkl, 'wb') as f:  
            pickle.dump([sid, lmes, blobs], f)  
        del land
        
    # else:
        # STEP 7: loads pickled landscape data
        # with open(sshpkl, 'rb') as f:  
        #    [sid, lmes, blobs] = pickle.load(f) 

    # %% end

    return(0)        


# %% Argument parser
if __name__ == "__main__":

    # Create the parser
    my_parser = ArgumentParser(prog="tla_ssh_step",
                               description="### Processing-step SSH module " +
                               "for Tumor Landscape Analysis ###",
                               allow_abbrev=False)

    # Add the arguments
    my_parser.add_argument('-v', '--version', 
                           action='version', 
                           version="%(prog)s " + __version__)
    
    my_parser.add_argument('argsfile',
                           metavar="argsfile",
                           type=str,
                           help="Argument table file (.csv) for study set")
    
    my_parser.add_argument('casenum',
                           metavar="casenum",
                           type=int,
                           help="Set case number to be processed")

    my_parser.add_argument("--redo",
                           default=False,
                           action="store_true",
                           help="If --redo is used, re-do landscape analysis")

    # passes arguments object to main
    main(my_parser.parse_args())
