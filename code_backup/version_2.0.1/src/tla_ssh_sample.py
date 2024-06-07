'''
    Tumor Landscape SSH Analysis (TLA):
    ##############################

        This script reads lines from a study set table
        Each line has parameters of a particular study

        This formating helps faster and more consistent running of the TLA


'''

# %% Imports
import os
import psutil
import sys
import gc
import math
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from ast import literal_eval

from argparse import ArgumentParser

from myfunctions import printProgressBar, plotRGB, mkdirs
# from itertools import combinations, permutations, product

    
Image.MAX_IMAGE_PIXELS = 600000000

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:10240"

__version__  = "2.0.0"

# %% Private classes
    
class Study:
    
    def __init__(self, study, main_pth):
        
        # loads arguments for this study
        self.name = study['name']
        f = os.path.join(main_pth, study['data_path'])
        if not os.path.exists(f):
            print("ERROR: data folder " + f + " does not exist!")
            print("... Please run << TLA >> for this study!")
            sys.exit()
        self.dat_pth = f
              
        f = os.path.join(self.dat_pth, study['name'] + '_classes.csv')
        if not os.path.exists(f):
            print("ERROR: classes file " + f + " does not exist!")
            sys.exit()
        self.classes = getLMEclasses(f)
             
        # list of samples is read from 'tla_sub_samples.csv' if it exist, 
        # otherwise reads directly from the original list for the study 
        f = os.path.join(self.dat_pth, 'ssh_sub_samples.csv')
        if not os.path.exists(f):
            f = os.path.join(self.dat_pth, study['name'] + '_samples.csv')
            if not os.path.exists(f):
                print("ERROR: samples file " + f + " does not exist!")
                sys.exit()
        self.samples = pd.read_csv(f)  
        
        # list of processed samples
        self.done_list = os.path.join(self.dat_pth, 'ssh_done_samples.csv')
        # scale parameters
        self.factor = np.float32(1.0)
        if 'factor' in study:
            self.factor = study['factor']
        self.scale = np.float32(study['scale']/self.factor)
        self.units = study['units']
        
        # the size of quadrats and subquadrats
        aux = 4*np.ceil((study['binsiz']/self.scale)/4)
        self.binsiz = np.rint(aux).astype('int32')
        self.subbinsiz = np.rint(self.binsiz/4).astype('int32')
        
        # bandwidth size for convolutions is half the quadrat size
        self.kernel = np.rint(self.binsiz/2).astype('int32')
        self.subkernel = np.rint(self.subbinsiz/2).astype('int32')
        
        # creates samples table for output
        self.samples_out = pd.DataFrame()
        
        # analyses table
        df = pd.read_csv(os.path.join(self.dat_pth,
                                      self.name + '_analyses.csv'),
                                    converters={'comps': literal_eval}) 
        # drop Attraction Enrichment Functions score from analysis
        # (NEED TO FIX A BUG IN THE CALCULATION)
        df.loc[df['name']== 'aefunc', 'drop'] = True
        
        self.analyses = df.loc[~df['drop']]
        self.donefacts = self.analyses['name'].tolist()
      
    def getSample(self, i):
        
        # get sample entry from samples df
        sample = self.samples.iloc[i].copy()
        
        if (sample.num_cells == ''):
            sample.num_cells = 0
              
        # check that results dir exist
        f = os.path.join(self.dat_pth, sample['results_dir'])
        if not os.path.exists(f):
            print("ERROR: data folder " + f + " does not exist!")
            print("... Please run << TLA setup >> for this study!")
            sys.exit()
        sample['res_pth'] = f    
        
        return(sample)    
      
  
class Landscape:
    
    def __init__(self, sample, study, strata):
        
        dat_pth = study.dat_pth
      
        self.sid = sample.sample_ID
        self.res_pth = sample['res_pth']
        self.out_pth = mkdirs(os.path.join(self.res_pth, 'ssh'))
        
        f = os.path.join(self.res_pth, self.sid +'_classes.csv')
        if not os.path.exists(f):
            print("ERROR: classes file " + f + " does not exist!")
            sys.exit()
        aux = pd.read_csv(f)
        aux['class'] = aux['class'].astype(str)
        self.classes = pd.merge(aux, 
                                study.classes[['class', 
                                               'abundance_edges', 
                                               'mixing_edges']], 
                                how="left",
                                on=['class'])
        
        f = os.path.join(dat_pth, sample['coord_file'])
        if not os.path.exists(f):
            print("ERROR: data file " + f + " does not exist!")
            sys.exit()
        self.cell_data = pd.read_csv(f)
        self.ncells = len(self.cell_data)
        
        # general attributes
        self.binsiz = study.binsiz
        self.subbinsiz = study.subbinsiz 
        self.kernel =  study.kernel
        self.subkernel = study.subkernel
        self.scale = study.scale
        self.units = study.units
        
        # raster images
        pth = os.path.join(study.dat_pth, 'rasters', self.sid)
        
        f = os.path.join(pth, self.sid + '_roi.npz')
        if not os.path.exists(f):
            print("ERROR: raster file " + f + " does not exist!")
            sys.exit()
        aux = np.load(f)
        self.roiarr = aux['roi'].astype('bool') # ROI raster
        self.imshape = np.int32(self.roiarr.shape)
        
        # more raster attributes
        self.mask_file = os.path.join(pth, self.sid + '_mask.npz')
        self.lme_file = os.path.join(pth, self.sid + '_lme.npz')  
        self.abumix_file = os.path.join(dat_pth, sample['abumix_file'])
        self.coloc_file = os.path.join(pth, self.sid + '_coloc.npz')  
        self.nndist_file = os.path.join(pth, self.sid + '_nndist.npz') 
        self.aefunc_file = os.path.join(pth, self.sid + '_aefunc.npz')  
        self.rhfunc_file = os.path.join(pth, self.sid + '_rhfunc.npz')  
        self.geordG_file = os.path.join(pth, self.sid + '_geordG.npz')  
       
        # # slide image
        # img = None
        # f = os.path.join(dat_pth, sample['image_file'])
        # if (sample['image_file'] != '' and os.path.exists(f)):
        #     img = np.uint8(io.imread(f))
        # self.img = img
        
        # pylandstats landscape object
        self.plsobj = []
      
        self.lmes = []
        if 'LME' in strata:
          # loads lme raster image
          f = self.lme_file
          if not os.path.exists(f):
              print("ERROR: raster file " + f + " does not exist!")
              sys.exit()
          aux = np.load(f)
          arr = aux['lme']
          arr[arr == np.nan] = 0
          self.lmes = np.uint8(arr)
          
        self.blobs = []
        if 'Blob' in strata:
           # loads lme raster image
           f = self.mask_file
           if not os.path.exists(f):
               print("ERROR: raster file " + f + " does not exist!")
               sys.exit()
           aux = np.load(f)
           arr = aux['roi']
           arr[arr == np.nan] = 0
           self.blobs = np.uint32(arr)
    
        del aux 

    
    def colocSSH(self, df, strata, do_plots):
        
        cps =  df.loc[df['name'] == 'coloc']['comps'].values[0]
        comps = compIDX(cps, self.classes)
        
        # loads coloc raster image
        f = self.coloc_file
        if not os.path.exists(f):
            print("ERROR: raster file " + f + " does not exist!")
            sys.exit()
        aux = np.load(f)
        colocarr = np.float64(aux['coloc'])
        
        out_pth = mkdirs(os.path.join(self.out_pth, 'coloc'))
        
        tab = sshComps(self.sid, colocarr, self.imshape, 
                     self.scale, self.units, self.subbinsiz,
                     'coloc', comps, self.classes, 
                     self.lmes, self.blobs, strata, 
                     out_pth, do_plots, subset = True)
        
        del colocarr
        
        return(tab)
    
    
    def nndistSSH(self, df, strata, do_plots):
        
        cps = df.loc[df['name'] == 'nndist']['comps'].values[0]
        comps = compIDX(cps, self.classes)
        
        # loads nndist raster image
        f = self.nndist_file
        if not os.path.exists(f):
            print("ERROR: raster file " + f + " does not exist!")
            sys.exit()
        aux = np.load(f)
        nndistarr = np.float64(aux['nndist'])
        
        out_pth = mkdirs(os.path.join(self.out_pth, 'nndist'))
                  
        _ = sshComps(self.sid, nndistarr, self.imshape, 
                     self.scale, self.units, self.subbinsiz,
                     'nndist', comps, self.classes, 
                     self.lmes, self.blobs, strata, 
                     out_pth, do_plots, subset = True)
        
        del nndistarr        
        
        
    def rhfuncSSH(self, df, strata, do_plots):
        
        cps = df.loc[df['name'] == 'rhfunc']['comps'].values[0]
        comps = compIDX(cps, self.classes)
        
        # loads nndist raster image
        f = self.rhfunc_file
        if not os.path.exists(f):
            print("ERROR: raster file " + f + " does not exist!")
            sys.exit()
        aux = np.load(f)
        rhfuncarr = np.float64(aux['rhfunc'])
        
        out_pth = mkdirs(os.path.join(self.out_pth, 'rhfunc'))
                  
        _ = sshComps(self.sid, rhfuncarr, self.imshape, 
                     self.scale, self.units, self.subbinsiz,
                     'rhfunc', comps, self.classes, 
                     self.lmes, self.blobs, strata, 
                     out_pth, do_plots, subset = True)
        
        del rhfuncarr
        
        
    def geordGSSH(self, df, strata, do_plots):
        
        cps = df.loc[df['name'] == 'geordG']['comps'].values[0]
        comps = [getIndex(c, self.classes) for c in cps]
        
        # loads nndist raster image
        f = self.geordG_file
        if not os.path.exists(f):
            print("ERROR: raster file " + f + " does not exist!")
            sys.exit()
        aux = np.load(f)
        geordGarr = np.float64(aux['geordG'])
        hotarr = np.float32(aux['hot'])
        
        out_pth = mkdirs(os.path.join(self.out_pth, 'geordG'))
                  
        _ = sshCases(self.sid, geordGarr, self.imshape, 
                     self.scale, self.units, self.subbinsiz,
                     'geordG', comps, self.classes, 
                     self.lmes, self.blobs, strata, 
                     out_pth, do_plots, subset = True)
        
        _ = sshCases(self.sid, hotarr, self.imshape, 
                     self.scale, self.units, self.subbinsiz,
                     'hot', comps, self.classes, 
                     self.lmes, self.blobs, strata, 
                     out_pth,  do_plots, subset = True)
        
        del geordGarr

           
    def abundSSH(self, strata, do_plots):
        
        comps = list(self.classes.index)
      
        # loads abundance raster image
        f = self.abumix_file
        if not os.path.exists(f):
            print("ERROR: raster file " + f + " does not exist!")
            sys.exit()
        aux = np.load(f)
        abuarr = np.float64(aux['abu'])
        
        out_pth = mkdirs(os.path.join(self.out_pth, 'abundance'))
                  
        _ = sshCases(self.sid, abuarr, self.imshape, 
                     self.scale, self.units, self.subbinsiz,
                     'abu', comps, self.classes, 
                     self.lmes, self.blobs, strata, 
                     out_pth, do_plots, subset = True)
        
        del abuarr          
      
# %% Private Functions

def memuse():
    gc.collect()
    m = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
    return(round(m, 2))


def progressBar(i, n, step, Nsteps,  msg, msg_i):
    
    printProgressBar(Nsteps*i + step, Nsteps*n,
                     suffix=msg + ' ; ' + msg_i, length=50)


def getIndex(c, df):
    return(df.index[df['class'] == c].tolist()[0])

    
def compIDX(comps, classes):
    
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
    sshtab = pd.DataFrame(SSH_factor_detector_df(data, 
                                                 fact_col, 
                                                 strata_cols))
    
    if (len(sshtab) > 0) and do_plots:
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
            if len(sshrsk) > 0:
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
        if (row.p_value < 0.1):
            sig = '(.)'
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

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    return([fig, ax])


def sshCases(sid, raster, 
            imshape, scale, units, binsiz, 
            field_name, comps, classes, 
            lmes, blobs, strata, 
            out_pth, do_plots, subset = True, para = False):
    """
    """
    from joblib import Parallel, delayed
    import warnings
    
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
            
        aux = pd.DataFrame()
        auy = pd.DataFrame()
        auz = pd.DataFrame()
                
        if (len(Y[inx])>0):
            
            data = pd.DataFrame({fact: Y[inx]})
            
            data = pd.DataFrame({fact: Y[inx]})
            
            if 'Blob' in strata:
                data['Blob'] = B[inx].astype(np.uint32)
                if do_plots:
                    plotFactorLandscape(sid, blobs, raster[:, :, i],
                                        fact_name + ' in Blobs', fact_name,
                                        imshape, scale, units, 4*binsiz, 
                                        fact + '_blobs', out_pth)
            if 'LME' in strata:
                data['LME'] = L[inx].astype(np.uint8)
                data = data.loc[data['LME'] > 0]
                if do_plots:
                    plotFactorLandscape(sid, lmes, raster[:, :, i],
                                        fact_name + ' in LMEs', fact_name,
                                        imshape, scale, units, 4*binsiz, 
                                        fact + '_lmes', out_pth)
            
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
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

    return(sshtab)


def sshComps(sid, raster, 
             imshape, scale, units, binsiz, 
             field_name, comps, classes, 
             lmes, blobs, strata, 
             out_pth, do_plots, subset = True, para = False):
    """
    """
    from joblib import Parallel, delayed
    import warnings
    
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
        R = np.array([raster[r, c, :] for r in np.arange(imshape[0]) \
                      for c in np.arange(imshape[1])])
        L = np.array([lmes[r, c] for r in np.arange(imshape[0]) \
                      for c in np.arange(imshape[1])])
        B = np.array([blobs[r, c] for r in np.arange(imshape[0]) \
                      for c in np.arange(imshape[1])])
    
    def func(i):
        
        comp = comps[i]
        ia = comp[0]
        ib = comp[1]
        Y = np.float64(R[:,i])

        # get valid values
        inx = ~np.isnan(Y)
        
        fact = field_name + '_' + \
            classes['class'][ia] + '_' +  classes['class'][ib]
        fact_name = field_name + '(' + \
            classes['class_name'][ia] + ':' + classes['class_name'][ib] + ')'
            
        aux = pd.DataFrame()
        auy = pd.DataFrame()
        auz = pd.DataFrame()
        
        if (len(Y[inx])>0):
            
            data = pd.DataFrame({fact: Y[inx]})
            if 'Blob' in strata:
                data['Blob'] = B[inx].astype(np.uint32)
                if do_plots:
                    plotFactorLandscape(sid, blobs, raster[:, :, i],
                                        fact_name + ' in Blobs', fact_name,
                                        imshape, scale, units, 4*binsiz, 
                                        fact + '_blobs', out_pth)    
                    
            if 'LME' in strata:
                data['LME'] = L[inx].astype(np.uint8)
                data = data.loc[data['LME'] > 0]
                if do_plots:
                    plotFactorLandscape(sid, lmes, raster[:, :, i],
                                        fact_name + ' in LMEs', fact_name,
                                        imshape, scale, units, 4*binsiz, 
                                        fact + '_lmes', out_pth)
                      
                
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                # do SSH analysis on factors
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
                       
            [aux, auy, auz] = func(i)
            sshtab = pd.concat([sshtab, aux], ignore_index=True)
            sshint = pd.concat([sshint, auy], ignore_index=True)
            ssheco = pd.concat([ssheco, auz], ignore_index=True)
            
    sshtab.to_csv(os.path.join(out_pth, 
                               sid + '_ssh_factor_' + field_name + '.csv'), 
                  index=False, na_rep='NA')
    if (len(strata) > 1):
        sshint.to_csv(os.path.join(out_pth,
                                   sid + '_ssh_interaction_' + \
                                       field_name + '.csv'), 
                      index=False, na_rep='NA')
        ssheco.to_csv(os.path.join(out_pth,
                                   sid + '_ssh_ecological_' + \
                                       field_name + '.csv'), 
                      index=False, na_rep='NA')

    return(sshtab)  


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

    # %% start, checks how the program was launched 
    debug = False
    try:
        args
    except NameError:
        #  if not running from the CLI, run in debug mode
        debug = True
    
    # tracks time and memory 
    start = time.time()
    mbuse = [np.nan] * 7
    
    if debug:
        # running from the IDE
        # path of directory containing this script
        main_pth = os.path.dirname(os.getcwd())
        argsfile = os.path.join(main_pth, 'DCIS.csv')
        REDO = True
        GRPH = True
        CASE = 119
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
        
    # NOTE: only ONE line in the argument table will be used
    study = Study( pd.read_csv(argsfile).iloc[0], main_pth)
       
    factors = ['coloc', 'nndist', 'rhfunc', 'geordG', 'abundance']
    strata = ['LME', 'Blob']
     
    # %% STEP 0: creates data directories and new sample table
    # creates sample object and data folders for pre-processed data
    sample = study.getSample(CASE)
    
    # SID for display
    sid = sample.sample_ID
    
    msg = "====> Case [" + str(CASE + 1) + \
          "/" + str(len(study.samples.index)) + \
          "] :: SID <- " + sid 
          
    mbuse[0] = memuse()
    if debug:
        t0 = time.time()
        trun = time.strftime('%H:%M:%S', time.gmtime(t0 - start))
        print('==> STEP 0 - Time elapsed: ', trun, '[HH:MM:SS]')
        print("==> STEP 0 - Memory used: " + str(mbuse[0]) + "[MB]")
    
    if (sample.num_cells > 0):

          # output sample data filenames
          samplcsv = os.path.join(sample['res_pth'], sid +'_ssh_tbl.csv')
          
          # if processed landscape do not exist
          if (REDO or (not os.path.exists(samplcsv))):
              
              sshtab = pd.DataFrame() 
              
              # %% STEP 1: loading data
                
              print( msg + " >>> processing SSH..." )
              
              land = Landscape(sample, study, strata)
              
              mbuse[1] = memuse()
              if debug:
                  t1 = time.time()
                  trun = time.strftime('%H:%M:%S', time.gmtime(t1 - t0))
                  print('==> STEP 1 - Time elapsed: ', trun, '[HH:MM:SS]')
                  print("==> STEP 1 - Memory used: " + str(mbuse[0]) + "[MB]")
              
              # %% STEP 2: SSH analysis: coloc
              if ('coloc' in factors) and ('coloc' in study.donefacts):
                  aux = land.colocSSH(study.analyses, strata, GRPH)
                  sshtab = pd.concat([sshtab, aux], ignore_index=True)
                  
                  mbuse[2] = memuse()
                  if debug:
                      t2 = time.time()
                      trun = time.strftime('%H:%M:%S', time.gmtime(t2 - t1))
                      print('==> STEP 2 - Time elapsed: ', trun, '[HH:MM:SS]')
                      print("==> STEP 2 - Memory used: " + str(mbuse[0]) + "[MB]")
                  
              # %% STEP 3: SSH analysis for nndist fields
              if ('nndist' in factors) and ('nndist' in study.donefacts):
                  aux =land.nndistSSH(study.analyses, strata, GRPH)
                  sshtab = pd.concat([sshtab, aux], ignore_index=True)
                  
                  mbuse[3] = memuse()
                  if debug:
                      t3 = time.time()
                      trun = time.strftime('%H:%M:%S', time.gmtime(t3 - t2))
                      print('==> STEP 3 - Time elapsed: ', trun, '[HH:MM:SS]')
                      print("==> STEP 3 - Memory used: " + str(mbuse[0]) + "[MB]")

              # %% STEP 4: SSH analysis for rhfunc fields
              if ('rhfunc' in factors) and ('rhfunc' in study.donefacts):
                  aux =land.rhfuncSSH(study.analyses, strata, GRPH)
                  sshtab = pd.concat([sshtab, aux], ignore_index=True)
                  
                  mbuse[4] = memuse()
                  if debug:
                      t4 = time.time()
                      trun = time.strftime('%H:%M:%S', time.gmtime(t4 - t3))
                      print('==> STEP 4 - Time elapsed: ', trun, '[HH:MM:SS]')
                      print("==> STEP 4 - Memory used: " + str(mbuse[0]) + "[MB]")
                         
              # %% STEP 5: SSH analysis for geordG fields
              if ('geordG' in factors) and ('geordG' in study.donefacts):
                  aux =land.geordGSSH(study.analyses, strata, GRPH)
                  sshtab = pd.concat([sshtab, aux], ignore_index=True)
                  
                  mbuse[5] = memuse()
                  if debug:
                      t5 = time.time()
                      trun = time.strftime('%H:%M:%S', time.gmtime(t5 - t4))
                      print('==> STEP 5 - Time elapsed: ', trun, '[HH:MM:SS]')
                      print("==> STEP 5 - Memory used: " + str(mbuse[0]) + "[MB]")
                    
              # %% STEP 6: SSH analysis for abundance fields
              if 'abundance' in factors:
                  aux =land.abundSSH(strata, GRPH)
                  sshtab = pd.concat([sshtab, aux], ignore_index=True)
                  
                  mbuse[6] = memuse()
                  if debug:
                      t6 = time.time()
                      trun = time.strftime('%H:%M:%S', time.gmtime(t6 - t5))
                      print('==> STEP 6 - Time elapsed: ', trun, '[HH:MM:SS]')
                      print("==> STEP 6 - Memory used: " + str(mbuse[0]) + "[MB]")
                  
              # %% STEP 7: saves results (for faster re-runs)
              sshtab.to_csv(samplcsv, index=False)
              
              del land

    # %% end
    memmax = np.nanmax(mbuse)
    trun = time.strftime('%H:%M:%S', time.gmtime(time.time()-start))
    print('==> TLA finished. Time elapsed: ', trun, '[HH:MM:SS]')
    print("==> Max memory used: " + str(memmax) + "[MB]")
    
    with open(study.done_list, 'a') as f:
        f.write(sid + '\n')

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
