'''
    Tumor Landscape Analysis (TLA):
    ##############################

        Reads parameters from a study set table (only the 1st line)
        Then reads pre-processed data for one sample given by case number.
        All TLA metrics are calculated and recorded.
        
        Aggregated statistics across samples are calculated  using 
        'TLA_sum' after processing all samples in the study. 
              

'''

# %% Imports
import os
import sys
import psutil
import gc
import math
import time
import torch

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from ast import literal_eval
from argparse import ArgumentParser

from myfunctions import mkdirs, filexists

if torch.cuda.is_available(): 
    ISCUDA = True
else:
    ISCUDA = False
    
Image.MAX_IMAGE_PIXELS = 600000000

__version__  = "2.0.2"


# %% Private classes

class Progress:
    """Progress class"""
    
    def __init__(self, name, maxsteps=20):
        
        self.label = name
        self.start = time.time()
        self.mbuse = [np.nan] * maxsteps
        self.step = 0
        self.mbuse[0] = memuse()
        self.time0 = self.start
        
    def dostep(self, display, lab=''):
        
        t = time.time()
        self.steptime = t - self.time0
        self.runtime = t - self.start
        self.time0 = t
        
        self.step = self.step + 1
        self.mbuse[self.step] = memuse()

        if display:
            print("==> " + self.label + "; STEP: " + str(self.step) + \
                  "; " + lab +'\n'\
                  "\t- Runtime: " + ftime(self.steptime) + "[HH:MM:SS]\n" +\
                  "\t- Tot Runtime: " + ftime(self.runtime) + "[HH:MM:SS]\n" +\
                  "\t- Mem used: " + str(self.mbuse[self.step] ) + "[MB]")
                

class Study:
    """Study class"""
    
    def __init__(self, study, main_pth):
        
        from myfunctions import tofloat, toint
        
        # loads arguments for this study
        self.name = study['name']
        f = filexists(os.path.join(main_pth, study['data_path']))
        if not os.path.exists(f):
            print("ERROR: data folder " + f + " does not exist!")
            print("... Please run << TLA setup >> with this study!")
            sys.exit()
        self.dat_pth = f
        
        # load classes
        f = os.path.join(self.dat_pth, study['name'] + '_classes.csv')
        self.classes = loadClasses(f)
        
        # list of samples is read from 'tla_sub_samples.csv' if it exist, 
        # otherwise reads directly from the original list for the study 
        f = os.path.join(self.dat_pth, 'tla_sub_samples.csv')
        if not os.path.exists(f):
            f = filexists(os.path.join(self.dat_pth, 
                                       study['name'] + '_samples.csv'))
        self.samples = pd.read_csv(f)    
        
        # list of processed samples
        self.done_list = os.path.join(self.dat_pth, 'tla_done_samples.csv')
                 
        # scale parameters
        self.factor = tofloat(1.0)
        if 'factor' in study:
            self.factor = study['factor']
        self.scale = tofloat(study['scale']/self.factor)
        self.units = study['units']
        
        # the size of quadrats and subquadrats
        aux = np.rint(10*np.ceil((study['binsiz']/self.scale)/10))
        self.binsiz = toint(aux)
        
        # bandwidth size for convolutions
        self.bw = toint(np.rint(self.binsiz/20))
        
        # creates samples table for output
        self.samples_out = pd.DataFrame()
        
        # analyses table
        self.analyses = pd.read_csv(os.path.join(self.dat_pth,
                                                 self.name + '_analyses.csv'),
                                    converters={'comps': literal_eval}) 

class Sample:
    
    def __init__(self, i, study):
        
        # get sample entry from samples df
        sample = study.samples.iloc[i].copy()
        if (sample.num_cells == ''):
            sample.num_cells = 0
        self.tbl = sample
            
        dat_pth = study.dat_pth
        self.sid = sample.sample_ID
        self.msg = "====> Case [" + str(i + 1) + \
              "/" + str(len(study.samples.index)) + \
              "] :: SID <- " + self.sid + " >>> processing..."
            
        # check if results dir exist
        f = os.path.join(dat_pth, sample['results_dir'])
        if not os.path.exists(f):
            print("ERROR: data folder " + f + " does not exist!")
            print("... Please run << TLA setup >> with this study!")
            sys.exit()
        self.res_pth = f
        
        # load class table for this sample
        aux = pd.read_csv(filexists(os.path.join(f, self.sid +'_classes.csv')))
        aux['class'] = aux['class'].astype(str)
        # merge with cohort-level table
        self.classes = pd.merge(aux, study.classes[['class', 
                                                    'abundance_edges', 
                                                    'mixing_edges']], 
                                how="left", on=['class'])
        
        # coordinates file
        self.coord_file = filexists(os.path.join(dat_pth, 
                                                 sample['coord_file']))
         
        # general attributes
        self.num_cells = sample.num_cells
        self.binsiz = study.binsiz
        self.bw =  study.bw
        self.scale = study.scale
        self.units = study.units
        
        # rasters path
        pth = os.path.join(study.dat_pth, 'rasters', self.sid)
        if not os.path.exists(pth):
            print("ERROR: data folder " + pth + " does not exist!")
            print("... Please run << TLA setup >> with this study!")
            sys.exit()
        
        # ROI raster
        aux = np.load(filexists(os.path.join(pth, self.sid + '_roi.npz')))
        self.roiarr = aux['roi'].astype('bool') 
        self.imshape = self.roiarr.shape
        
        # input raster file names
        self.kde_file = filexists(os.path.join(pth, self.sid + '_kde.npz'))
        self.abumix_file = filexists(os.path.join(pth, self.sid + '_abumix.npz'))
        self.nrs_file = filexists(os.path.join(pth, self.sid + '_nrs.npz'))
        
        # output raster file names
        self.coloc_file = os.path.join(pth, self.sid + '_coloc.npz')  
        self.nndist_file = os.path.join(pth, self.sid + '_nndist.npz') 
        self.aefunc_file = os.path.join(pth, self.sid + '_aefunc.npz')  
        self.rhfunc_file = os.path.join(pth, self.sid + '_rhfunc.npz')  
        self.geordG_file = os.path.join(pth, self.sid + '_geordG.npz')  
        self.lme_file = os.path.join(pth, self.sid + '_lme.npz')  
        self.lmearr = []
        
        # # slide image
        # if (self.tbl.image_file == ''):
        #     self.raw_imfile = ''
        #     self.isimg = False
        # else:
        #     aux = os.path.join(study.raw_path, self.tbl.image_file)
        #     self.raw_imfile = aux
        #     self.isimg = True
  
        # # blob mask image
        # if (self.tbl.mask_file == ''):
        #     self.raw_mkfile = ''
        #     self.ismk = False
        # else:
        #     aux = os.path.join(study.raw_path, self.tbl.mask_file)
        #     self.raw_mkfile = aux
        #     self.ismk = True
            
        # pylandstats landscape object
        self.plsobj = []

    
    def spaceStats(self, redo, analyses):
        """Calculates pixel resolution statistics in raster arrays:
          
        1- Colocalization index (spacial Morisita-Horn score): Symetric score 
           between each pair of classes
           
           - M ~ 1 indicates the two classes are similarly distributed
           - M ~ 0 indicates the two classes are segregated
    
        2- Nearest Neighbor Distance index: Bi-variate asymetric score between 
           all classes ('ref' and 'test')
        
           - V > 0 indicates ref and test cells are segregated
           - V ~ 0 indicates ref and test cells are well mixed
           - V < 0 indicates ref cells are individually infiltrated
                   (ref cells are closer to test cells than other ref cells)
    
        3- Attraction Enrichment Function Score:  Bi-variate asymetric score,
           at r=bw between all classes ('ref' and 'test')
           
           - T = +1 indicates attraction of 'test' cells around 'ref' cells
           - T = 0 indicates random dipersion of'test' and 'ref' cells
           - T = -1 indicates repulsion of 'test' cells from 'ref' cells
    
        4- Ripley's H function score: Bi-variate version of Ripley's  H(r) 
           function, evaluated at r=bw and between all classes ('ref', 'test').
           This is asymetric and normalized by r (ie. H(r) = log10(L(r)/r)) 
           for interpretation.
        
           - H > 0 indicates clustering of 'test' cells around 'ref' cells
           - H ~ 0 indicates random mixing between 'test' and 'ref' cells
           - H < 0 indicates dispersion of 'test' cells around 'ref' cells
    
        5- Getis-Ord statistics (G* and HOT value): G* (Z statistic) and HOT 
        for all classes, were:
           (*) HOT = +1 if the region is overpopulated (P < 0.05)
           (*) HOT =  0 if the region is average (P > 0.05)
           (*) HOT = -1 if the region is underpopulated (P < 0.05)
           
        NOTE: the specific analyzes to be done are set in the `analyses` data
              frame passed in (this is an attribute of the Study object). This
              is done to have the option of downsizing the amount of results
              produced (and hence reduce running time)
    
        """
        from myfunctions import tofloat, toint, tobit, gkern
        from myfunctions import getis_ord_g_array, morisita_horn_array
        from myfunctions import nndist_array, attraction_T_array
        from myfunctions import ripleys_K_array
        
        # load data
        data = pd.read_csv(self.coord_file)
        data['class'] = data['class'].astype(str)
        data['orig_class'] = data['orig_class'].astype(str)
        classes = self.classes.copy()
        shape = self.imshape
        roi = self.roiarr
        df = analyses.copy()
        
        if (redo):
            iscoloc = ~df.loc[df['name'] == 'coloc']['drop'].values[0] 
            isnndist = ~df.loc[df['name'] == 'nndist']['drop'].values[0] 
            isaefunc = ~df.loc[df['name'] == 'aefunc']['drop'].values[0] 
            isrhfunc = ~df.loc[df['name'] == 'rhfunc']['drop'].values[0] 
            isgeordG = ~df.loc[df['name'] == 'geordG']['drop'].values[0] 
        else:
            iscoloc = ~df.loc[df['name'] == 'coloc']['drop'].values[0] and \
                not os.path.exists(self.coloc_file)
            isnndist = ~df.loc[df['name'] == 'nndist']['drop'].values[0] and \
                not os.path.exists(self.nndist_file)
            isaefunc = ~df.loc[df['name'] == 'aefunc']['drop'].values[0] and \
                    not os.path.exists(self.aefunc_file)
            isrhfunc = ~df.loc[df['name'] == 'rhfunc']['drop'].values[0] and \
                not os.path.exists(self.rhfunc_file)
            isgeordG = ~df.loc[df['name'] == 'geordG']['drop'].values[0] and \
                not os.path.exists(self.geordG_file)
        
        # if not all analyses are dropped
        if (iscoloc or isnndist or isaefunc or isrhfunc or isgeordG):
            
            # lists of classes cases to be calculated (x is ref, y is target)
            cases_x = []
            cases_y = []
            # empty arrays for desider comparisons for each metric
            coloc_comps = []
            nndist_comps = []
            aefunc_comps = []
            rhfunc_comps = []
            geordG_comps = []
            
            # get comparisons and allocate arrays
            if iscoloc:
                # combinations of cases:
                coloc_comps = df.loc[df['name']=='coloc']['comps'].values[0]
                # Colocalization index (spacial Morisita-Horn score)
                colocarr = tofloat(np.full((shape[0], shape[1],
                                            len(coloc_comps)), np.nan))
                cases_x  = list(set(cases_x) | 
                                set([c[0] for c in coloc_comps]))
                cases_y  = list(set(cases_y) | 
                                set([c[1] for c in coloc_comps]))
            if isnndist:
                # combinations of cases:
                nndist_comps = df.loc[df['name']=='nndist']['comps'].values[0]
                # Nearest Neighbor Distance index
                nndistarr = tofloat(np.full((shape[0], shape[1],
                                             len(nndist_comps)),  np.nan))
                cases_x  = list(set(cases_x) | 
                                set([c[0] for c in nndist_comps]))
                cases_y  = list(set(cases_y) |
                                set([c[1] for c in nndist_comps]))
            if isaefunc:
                # combinations of cases:
                aefunc_comps = df.loc[df['name']=='aefunc']['comps'].values[0]
                # Attraction Enrichment score
                aefuncarr = tofloat(np.full((shape[0], shape[1],
                                             len(aefunc_comps)), np.nan))
                cases_x  = list(set(cases_x) |
                                set([c[0] for c in aefunc_comps]))
                cases_y  = list(set(cases_y) | 
                                set([c[1] for c in aefunc_comps]))
            if isrhfunc:
                # combinations of cases:
                rhfunc_comps = df.loc[df['name']=='rhfunc']['comps'].values[0]
                # Ripley's H score
                rhfuncarr = tofloat(np.full((shape[0], shape[1], 
                                             len(rhfunc_comps)), np.nan))
                cases_x  = list(set(cases_x) | 
                                set([c[0] for c in rhfunc_comps]))
                cases_y  = list(set(cases_y) | 
                                set([c[1] for c in rhfunc_comps]))
            if isgeordG:
                # combinations of cases:
                geordG_comps = df.loc[df['name']=='geordG']['comps'].values[0]
                # Gets-Ord statistics (G* and HOT value)
                geordGarr = tofloat(np.full((shape[0], shape[1],
                                             len(geordG_comps), 2), np.nan))
                hotarr = tofloat(np.full((shape[0], shape[1], 
                                          len(geordG_comps), 2), np.nan))
                cases_x  = list(set(cases_x) | set(geordG_comps))
            
            # list classes with non-zero abundance in this sample  
            valid_cases = self.classes.loc[self.classes['number_of_cells'] > 0, 
                                           'class'].to_list()
            
            cases_x  = list(set(cases_x) & set(valid_cases))
            cases_y  = list(set(cases_y) & set(valid_cases))
            
            # reduced list of all class cases
            cases = list(set(cases_x) | set(cases_y))
            
            # indeces of ref and target list cases in overal list
            cases_i = [classes.loc[classes['class']==c].index.item() \
                       for c in cases]
            cases_xi  = [cases.index(c) for c in cases_x]
            cases_yj  = [cases.index(c) for c in cases_y]
            
            # cell locations and abundance in kernels and subkernels
            X = tobit(np.zeros((shape[0], shape[1], len(cases))))
            N = tofloat(np.zeros((shape[0], shape[1], len(cases))))
            n = tofloat(np.zeros((shape[0], shape[1], len(cases))))
            d = tofloat(np.zeros((shape[0], shape[1], len(cases))))
            
            # load pre-calculated rasters
            aux = np.load(self.nrs_file)
            nall = tofloat(aux['nrs'])
            ras = tofloat(aux['rs'])
            
            # radii 
            r = ras[0, 0]
            a = ras[1, 0]
            R = ras[0, -1]
            A = ras[1, -1]
            
            # gaussian weights kernel
            gker = gkern(R, normal=False)
            subgker = gkern(r, normal=False)
            
            aux = np.load(self.kde_file)
            kde = tofloat(aux['kde'])
            
            # pre-calculate convolutions for all required classes
            for i, case_i in enumerate(cases):
                # get index in classes for this case
                j = cases_i[i]
                
                # coordinates of cells in class case
                aux = data.loc[data['class'] == case_i]
                X[aux.row, aux.col, i] = 1
                
                # KDE values
                aux = kde[:,:, j]
                msk = np.isnan(aux)
                aux[msk] = 0
                d[:, :, i] = aux
                
                # short scale abundances
                aux = nall[:,:, j, 0]
                aux[np.isnan(aux)] = 0
                n[:, :, i] = aux
                
                # long scale abundances
                aux = nall[:,:, j, -1]
                aux[np.isnan(aux)] = 0
                N[:, :, i] = aux
            del nall, kde
            
            # loop thru all combinations of classes (pair-wise comparisons)
            for i, case_x in enumerate(cases_x):
                # index of ref case in all cases list
                xi = cases_xi[i]
                # coordinates of cells in class x (ref)
                aux = data.loc[data['class'] == case_x]
                rcx = toint(np.array(aux[['row', 'col']]))
                
                # Getis-Ord stats for this class
                if (case_x in geordG_comps):
                    # mean and std point density
                    mu = len(rcx)/np.sum(roi)
                    sigma = np.sqrt(mu*(1 - mu))
                    # mean, std of point dens in each (long scale) region 
                    muxy = N[:, :, xi]/A
                    sigmaxy = np.sqrt(muxy - muxy*muxy)
                    
                    # Calculate stasts
                    G = getis_ord_g_array(X[:, :, xi], subgker, roi, 
                                          mu, sigma, 
                                          A, muxy, sigmaxy, 
                                          bw=r, cuda=ISCUDA)
    
                    [geordGarr[:, :, geordG_comps.index(case_x), 0], 
                     hotarr[:, :, geordG_comps.index(case_x), 0],
                     geordGarr[:, :, geordG_comps.index(case_x), 1], 
                      hotarr[:, :, geordG_comps.index(case_x), 1]] = G
                
                for j, case_y in enumerate(cases_y):
                    # index of target case in all cases list
                    yj = cases_yj[j]
                    # coordinates of cells in class y (target)
                    aux = data.loc[data['class'] == case_y]
                    rcy = toint(np.array(aux[['row', 'col']]))
                    
                    # comparison tuple
                    comp = (case_x, case_y) 
                    
                    if (comp in coloc_comps):
                        if (comp[0] != comp[1]):
                            # Morisita Index (colocalization score)
                            M = morisita_horn_array(d[:, :, xi]/len(rcx), 
                                                    d[:, :, yj]/len(rcy),
                                                    gker, roi, cuda=ISCUDA)
                        else:
                            # Morisita Index (identity)
                            M = np.ones(d.shape)*np.nan
                            M[d[:, :, xi] > 0] = 1.0
                            M[~roi] = np.nan
                        colocarr[:, :, coloc_comps.index(comp)] = M

                    if (comp in nndist_comps):
                        if (comp[0] != comp[1]):
                            # Nearest Neighbor Distance index (bivariate)
                            D = nndist_array(rcx, rcy, 
                                             N[:, :, xi], gker, roi,
                                             cuda=ISCUDA)
                        else:
                            # Nearest Neighbor Distance index (identity)
                            D = 1.0*(N[:, :, xi] > 0)
                            D[~roi] = np.nan
                        nndistarr[:, :, nndist_comps.index(comp)] = D
                    
                    if (comp in aefunc_comps):
                        # Attraction Enrichment T score (bivarite)
                        T = attraction_T_array(rcx,
                                               n[:, :, yj]/a,
                                               N[:, :, yj],
                                               gker, roi, cuda=ISCUDA)
                        aefuncarr[:, :, aefunc_comps.index(comp)] = T
                        
                    if (comp in rhfunc_comps):
                        if (comp[0] != comp[1]):
                            # Ripleys H score (bivarite)
                            npairs = N[:, :, xi]*N[:, :, yj]
                            K = ripleys_K_array(rcx, n[:, :, yj], npairs,
                                                gker, roi, cuda=ISCUDA)
                        else:
                            # # Ripleys H score (dentity)
                            npairs = N[:, :, xi]*(N[:, :, xi] - 1)/2
                            K = ripleys_K_array(rcx, n[:, :, xi], npairs,
                                                gker, roi, cuda=ISCUDA)
                        # original Ripley's H def (A = pi*r^2)
                        #H = np.sqrt(K/np.pi) - r
                        # area of a Gaussian is 2*pi*sigma^2
                        H = np.full_like(K, fill_value=np.nan)
                        msk = ~np.isnan(K)
                        H[msk]= np.log10(np.sqrt(K[msk]/(2*np.pi))/r)
                        rhfuncarr[:, :, rhfunc_comps.index(comp)] = H
      
            if iscoloc:
                # saves coloc cases:
                np.savez_compressed(self.coloc_file, 
                                    coloc=colocarr)
                del colocarr
                
            if isnndist:
                # saves nndist cases:
                np.savez_compressed(self.nndist_file, 
                                    nndist=nndistarr)
                del nndistarr
                
            if isrhfunc:
                # saves rhfunc cases:
                np.savez_compressed(self.rhfunc_file, 
                                    rhfunc=rhfuncarr)
                del rhfuncarr
                
            if isgeordG:
                # saves geordG cases:
                np.savez_compressed(self.geordG_file, 
                                    geordG=geordGarr,
                                    hot=hotarr)
                del geordGarr
                del hotarr
     
            gc.collect()
          
          
    def loadLMELandscape(self, redo, do_plots=False):
        """
        Produces raster array with LME code values. 
        This is based on a KDE raster, which is a smoothed cell concentration 
        image (with a certain bandwidth), of the original landscape of cell 
        locations, and a smooth raster of cell mixing values calculated in 
        the pre-processing module.
        For each point in the array, the local abundance of cells of each class
        and the mixing value level are used to assign an LME code according to
        study-level edges estimated in pre-processing.
  
        This method generates a landscape of LME categories with pixel-res
        Values of '0' in the array indicates elements outside of the ROI
  
        """
        from myfunctions import tofloat, tobit
        import pylandstats as pls
        
        out_pth = mkdirs(os.path.join(self.res_pth, 'space_factors'))
        
        # files to store fragmentation features of abu and mix
        fabu = os.path.join(out_pth, self.sid + '_abu_stats.csv')
        fclt = os.path.join(out_pth, self.sid + '_clt_stats.csv')
        fmix = os.path.join(out_pth, self.sid + '_mix_stats.csv')

        # if this analysis hasn't been done
        if (redo \
            or not os.path.exists(self.lme_file) \
                or not os.path.exists(fabu ) \
                    or not os.path.exists(fclt) \
                        or not os.path.exists(fmix)):
            
            nedges = self.classes['abundance_edges']
            medges = self.classes['mixing_edges']
      
            dim = len(self.classes)
            lmearr = np.zeros(self.imshape, dtype=np.uint8)
            
            f = self.abumix_file
            if not os.path.exists(f):
                print("ERROR: raster file " + f + " does not exist!")
                sys.exit()
            aux = np.load(f)
            abuarr = tofloat(aux['abu']) # local abundances of each class
            cltarr = abuarr.copy() 
            mixarr = tofloat(aux['mix']) # local mixing values of each class
      
            # vectorized function for faster processing
            def indexvalue(x, edges):
                o = max([0] + [j for j, v in enumerate(edges[:-1]) if v < x])
                return(o + 1)
            vindexvalue = np.vectorize(indexvalue, excluded=['edges'])
            
            lims = [[0, 0] for x in range(dim)]
            limx = [[0, 0] for x in range(dim)]
            
            for i, iclass in self.classes.iterrows():
                
                # get limits of abundance values for this cell type
                lims[i][0] = nedges[i][0]
                lims[i][1] = nedges[i][3]
                limx[i][0] = medges[i][0]
                limx[i][1] = medges[i][3]
                
                # get abundance level
                aux = abuarr[:, :, i].copy()
                aux[np.isnan(aux)] = nedges[i][0]
                aux[aux < nedges[i][0]] = nedges[i][0]
                aux[aux > nedges[i][3]] = nedges[i][3]
                abuarr[:, :, i] =  aux[:, :]
                
                # get mixing level
                auy = mixarr[:, :, i].copy()
                auy[np.isnan(auy)] = medges[i][0]
                auy[auy < medges[i][0]] = medges[i][0]
                auy[auy > medges[i][3]] = medges[i][3]
                mixarr[:, :, i] = auy[:, :]
                
                # get a lme array based on both abundance and mixing
                # a value of zero is the background
                if (redo or not os.path.exists(self.lme_file)):
                    
                    abu = vindexvalue(x=aux, edges=nedges[i])
                    mix = vindexvalue(x=auy, edges=medges[i])
                    # produces a single digital (dim-digit) code
                    j = dim - (i + 1)
                    lmearr = lmearr + (10**(2*j + 1))*abu + (10**(2*j))*mix
            
            if (redo or not os.path.exists(fabu)):
                # plots array of landscapes for these comparisons
                [fig, metrics] = plotCompLandscape(self.sid,
                                                   abuarr,
                                                   self.roiarr,
                                                   self.classes['class'],
                                                   self.classes['class'],
                                                   self.classes['class_name'],
                                                   self.imshape,
                                                   'Abundance Level', 
                                                   10,
                                                   lims,
                                                   self.scale,
                                                   self.units, 
                                                   self.binsiz,
                                                   do_plots,
                                                   same_lims=False)
                        
                # saves landscape metrics table
                if (len(metrics) > 0):
                    metrics.to_csv(fabu, sep=',', index=False, header=True)
                
                if do_plots:
                    # saves to png file
                    fig.savefig(os.path.join(out_pth,
                                             self.sid + '_abu_landscape.png'),
                                bbox_inches='tight', dpi=300)
                    plt.close()
            
            if (redo or not os.path.exists(fclt)):
                
                metrics = pd.DataFrame()
                
                for i, iclass in self.classes.iterrows():
                    
                    # makes a raster for cell clusters 
                    # (mass with abundance in top 50%)
                    aux = cltarr[:, :, i].copy()
                    v = (nedges[i][0] + nedges[i][3])/2
                    aux[aux < v] = 1
                    aux[aux >= v] = 2  
                    cltarr[:, :, i] =  aux[:, :].copy()
                    
                    # sets background values for pylandstats
                    aux[np.isnan(aux)] = 0
                    
                    # create pylandscape object with binned factor values
                    pl = pls.Landscape(aux, res=(1, 1), 
                                       nodata=0, neighborhood_rule='8')
                    if (len(pl.classes) > 0):
                        # get all landscape-level metrics for (discrete) factor
                        m = getLandscapeMetrics(pl)
                        m = m.to_frame().transpose().reset_index(drop=True)
                        m['comp'] =  iclass['class']
                        m['number_of_levels'] = len(pl.classes)
                        metrics = pd.concat([metrics, m], ignore_index=True)
                
                # shift column 'comp' to first position
                if (len(metrics) > 0):
                    metrics.insert(0, 'comp', metrics.pop('comp'))
                    metrics.insert(1, 'number_of_levels',
                                   metrics.pop('number_of_levels'))
                    # saves landscape metrics table
                    metrics.to_csv(fclt, sep=',', index=False, header=True)
                
                if do_plots:
                    # plots array of landscapes for these comparisons
                    fig, _ = plotDiscreteLandscape(self.sid, 
                                                   cltarr, 
                                                   self.roiarr,
                                                   self.classes['class'],
                                                   self.classes['class_name'],
                                                   self.imshape,
                                                   'Clump', 
                                                   [1, 2], 
                                                   [1, 2],
                                                   self.scale, 
                                                   self.units, 
                                                   self.binsiz,
                                                   do_plots)
                    
                    # saves to png file
                    fig.savefig(os.path.join(out_pth,
                                             self.sid + '_clt_landscape.png'),
                                bbox_inches='tight', dpi=300)
                    plt.close()
                          
              
            if (redo or not os.path.exists(fmix)):
                # plots array of landscapes for these comparisons
                [fig, metrics] = plotCompLandscape(self.sid,
                                                   mixarr,
                                                   self.roiarr,
                                                   self.classes.index,
                                                   self.classes['class'],
                                                   self.classes['class_name'],
                                                   self.imshape,
                                                   'Mixing Level', 
                                                   10,
                                                   limx,
                                                   self.scale,
                                                   self.units, 
                                                   self.binsiz,
                                                   do_plots,
                                                   same_lims=False)
                # saves landscape metrics table
                if (len(metrics) > 0):
                    metrics.to_csv(fmix, sep=',', index=False, header=True)
                
                if do_plots:
                    # saves to png file
                    fig.savefig(os.path.join(out_pth,
                                             self.sid + '_mix_landscape.png'),
                                bbox_inches='tight', dpi=300)
                    plt.close()
            
            if (redo or not os.path.exists(self.lme_file)):
                # sets out-regions to background (pylandstat uses zero, not NA)
                lmearr[self.roiarr == 0] = 0
          
                # reduces the number of lme classes by grouping them
                self.lmearr = tobit(lmeRename(lmearr, dim))
                
                # saves raster:
                np.savez_compressed(self.lme_file, lme=self.lmearr)
            
            del abuarr
            del mixarr
            gc.collect()
        
        else:
            aux = np.load(self.lme_file)
            self.lmearr = np.uint8(aux['lme'])
          

    def plotLMELandscape(self, out_pth):
        """
        Plot LME landscape
  
        """
              
        from myfunctions import plotRGB, plotEdges, tofloat
        import warnings
  
        dim = len(self.classes)
        nlevs = 3**dim
        raster = tofloat(self.lmearr)
        raster[raster==0]= np.nan
        
        lme_code = ''
        for i in np.arange(dim):
            lme_code = lme_code + self.classes['class'][i]
        
        icticks = np.arange(nlevs) + 1
        cticks = [lmeCode(x, dim) for x in icticks]
        ctitle = 'LME Categories (' + lme_code + ')'
        cmap = plt.get_cmap('jet', nlevs)
        
        [ar, redges, cedges,  xedges, yedges] = plotEdges(self.imshape, 
                                                          self.binsiz, 
                                                          self.scale)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            
            # plots sample image
            fig, ax = plt.subplots(1, 1,
                                   figsize=(10*1, 0.5 + math.ceil(10*1/ar)),
                                   facecolor='w', edgecolor='k')
          
            im = plotRGB(ax, raster, self.units,
                         cedges, redges, xedges, yedges, fontsiz=18,
                         vmin=0.5, vmax=(nlevs + 0.5), cmap=cmap)
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_ticks(icticks)
            cbar.set_ticklabels(cticks)
            cbar.ax.tick_params(labelsize=14)
            cbar.set_label(ctitle, size = 16, rotation=90, labelpad=1)
            ax.set_title('Local Micro-Environments (LME)',
                         fontsize=24, y=1.02)
            #fig.subplots_adjust(hspace=0.4)
            fig.suptitle('Sample ID: ' + str(self.sid), fontsize=36, y=.95)
            #plt.tight_layout()
            fig.savefig(os.path.join(out_pth,
                                     self.sid +'_lme_landscape.png'),
                        bbox_inches='tight', dpi=300)
            plt.close()
  
            # the histogram of LME frequencies
            fig, ax = plt.subplots(1, 1, figsize=(5, 5),
                                   facecolor='w', edgecolor='k')
            ilabels, counts = np.unique(raster[~np.isnan(raster)],
                                        return_counts=True)
            ax.bar(ilabels, counts, align='center',
                   # alpha=0.5,
                   color=cmap(ilabels.astype(int)), edgecolor='k',)
            ax.set_title('Local Micro-Environments (LME)', 
                         fontsize=16, y=1.02)
            ax.set_xlabel(ctitle)
            ax.set_ylabel('Frequency')
            ax.set_xlim([0.5, (nlevs + 0.5)])
            ax.set_xticks(icticks)
            ax.set_xticklabels(cticks, rotation=90)
            ax.set_yscale('log')
            plt.tight_layout()
            fig.savefig(os.path.join(out_pth, 
                                     self.sid +'_lme_distribution.png'),
                        bbox_inches='tight', dpi=300)
            plt.close()
        
      
    def plotColocLandscape(self, redo, analyses, lims, do_plots=False):
        """
        Plots colocalization index landscape from raster
  
        """
        # from itertools import combinations
        # # generates a list of comparisons for coloc
        # comps = [list(c) for c in list(combinations(self.classes.index, 2))]

        # if this analysis is already done, skip it
        out_pth = mkdirs(os.path.join(self.res_pth, 'space_factors'))
        fout = os.path.join(out_pth, self.sid + '_coloc_stats.csv')
        if (redo or not os.path.exists(fout)):
            
            # analyses
            df = analyses.copy()
            iscoloc = ~df.loc[df['name'] == 'coloc']['drop'].values[0]
            
            if iscoloc:
                aux = np.load(self.coloc_file)
                colocarr = aux['coloc']
                # combinations of cases:
                coloc_comps = df.loc[df['name'] == 'coloc']['comps'].values[0]
                compsterm = coloc_comps.copy()
                compslabs = coloc_comps.copy()
                
                for i, comp in enumerate(coloc_comps):
                    aux = self.classes.loc[self.classes['class'] == comp[0]]
                    cl1 = aux.class_name.item()
                    aux = self.classes.loc[self.classes['class'] == comp[1]]
                    cl2 = aux.class_name.item()
                    compsterm[i] = comp[0] + '::' + comp[1]
                    compslabs[i] = '(' + cl1  + '::' + cl2 + ')'
                    
                # plots landscapes metrics for these factors 
                [fig, metrics] = plotCompLandscape(self.sid,
                                                   colocarr, 
                                                   self.roiarr,
                                                   coloc_comps, 
                                                   compsterm,
                                                   compslabs,
                                                   self.imshape,
                                                   'Coloc index', 
                                                   10,
                                                   lims,
                                                   self.scale, 
                                                   self.units, 
                                                   self.binsiz,
                                                   do_plots)
                # saves landscape metrics table
                metrics.to_csv(fout, sep=',', index=False, header=True)
          
                if do_plots:  
                    # saves to png file
                    fout = os.path.join(out_pth,
                                        self.sid +'_coloc_landscape.png')
                    fig.savefig(fout, bbox_inches='tight', dpi=300)
                    plt.close()
                    
                    if (len(coloc_comps) > 1):
                        # plot correlations between comparisons
                        ttl = 'Coloc index Correlations\n' + \
                            'Sample ID: ' + str(self.sid)
                        fig = plotCompCorrelations(self.classes, 
                                                   colocarr, 
                                                   coloc_comps, 
                                                   ttl, lims)
                        fout = os.path.join(out_pth,
                                            self.sid + \
                                                '_coloc_correlations.png')
                        plt.savefig(fout, bbox_inches='tight', dpi=300)
                        plt.close()
                    
                del colocarr
                gc.collect() 
              
              
    def plotColocLandscape_simple(self, icomp, comps, metric, lims):
        """
        Plots colocalization index landscape from raster
        """
        # from itertools import combinations
        # # generates a list of comparisons for coloc
        # comps = [list(c) for c in list(combinations(self.classes.index, 2))]
        
        out_pth = mkdirs(os.path.join(self.res_pth, 'space_factors'))
        aux = np.load(self.coloc_file)
        colocarr = aux['coloc']
        comp = comps[icomp]
        aux = self.classes.loc[self.classes['class'] == comp[0]]
        cl1 = aux.class_name.item()
        aux = self.classes.loc[self.classes['class'] == comp[1]]
        cl2 = aux.class_name.item()
        lbl = '(' + cl1  + '::' + cl2 + ')'
                
        # plots array of landscapes for these comparisons
        fig = plotCompLandscape_simple(self.sid,
                                       colocarr,
                                       self.roiarr,
                                       icomp, 
                                       self.imshape,
                                       metric + ' - ' + lbl,
                                       10,
                                       lims,
                                       self.scale,
                                       self.units,
                                       self.binsiz)
        
        # saves to png file
        nam = self.sid +'_coloc_landscape_simple_' + \
            comp[0]+ "_" + comp[1] +'.png'
        fig.savefig(os.path.join(out_pth, nam), bbox_inches='tight', dpi=300)
        plt.close()
                
        del colocarr
        gc.collect() 
        
      
    def plotNNDistLandscape(self, redo, analyses, lims, do_plots=False):
        """
        Plots nearest neighbor distance index landscape from raster
  
        """
        # from itertools import permutations
        # # generate list of comparisons for coloc
        # comps = [list(c) for c in list(permutations(self.classes.index,r=2))]

        # if this analysis is already done, skip it
        out_pth = mkdirs(os.path.join(self.res_pth, 'space_factors'))
        fout = os.path.join(out_pth, self.sid +'_nndist_stats.csv')
        if (redo or not os.path.exists(fout)):
        
            # analyses
            df = analyses.copy()
            isnndist = ~df.loc[df['name'] == 'nndist']['drop'].values[0]
            
            if isnndist:
                aux = np.load(self.nndist_file)
                nndistarr = aux['nndist']
                
                # combinations of cases:
                aux = df.loc[df['name'] == 'nndist']
                nndist_comps = aux['comps'].values[0]
                compsterm = nndist_comps.copy()
                compslabs = nndist_comps.copy()
                for i, comp in enumerate(nndist_comps):
                    aux = self.classes.loc[self.classes['class'] == comp[0]]
                    cl1 = aux.class_name.item()
                    aux = self.classes.loc[self.classes['class'] == comp[1]]
                    cl2 = aux.class_name.item()
                    compsterm[i] = comp[0] + '::' + comp[1]
                    compslabs[i] = '(' + cl1  + '::' + cl2 + ')'
                
                # plots array of landscapes for these comparisons
                [fig, metrics] = plotCompLandscape(self.sid, 
                                                   nndistarr, 
                                                   self.roiarr,
                                                   nndist_comps, 
                                                   compsterm,
                                                   compslabs,
                                                   self.imshape,
                                                   'NNDist index',
                                                   10,
                                                   lims,
                                                   self.scale,
                                                   self.units, 
                                                   self.binsiz,
                                                   do_plots)
                # saves landscape metrics table
                metrics.to_csv(fout, sep=',', index=False, header=True)
                
                if do_plots:
                    # saves to png file
                    fout = os.path.join(out_pth,
                                        self.sid +'_nndist_landscape.png')
                    fig.savefig(fout, bbox_inches='tight', dpi=300)
                    plt.close()
                    
                    if (len(nndist_comps)>1):
                        # plot correlations between comparisons
                        ttl = 'NNDist index Correlations\n' + \
                            'Sample ID: ' + str(self.sid)
                        fig = plotCompCorrelations(self.classes, 
                                                   nndistarr, 
                                                   nndist_comps,
                                                   ttl, 
                                                   lims)
                        fout = os.path.join(out_pth, 
                                            self.sid + \
                                                '_nndist_correlations.png')
                        plt.savefig(fout, bbox_inches='tight', dpi=300)
                        plt.close()
                del nndistarr
                gc.collect()     
              
              
    def plotNNDistLandscape_simple(self, icomp, comps, metric, lims):
        """
        Plots nearest neighbor distance index landscape from raster
    
        """
        # from itertools import permutations
        # # generate list of comparisons for coloc
        # comps = [list(c) for c in list(permutations(self.classes.index, r=2))]
        
        out_pth = mkdirs(os.path.join(self.res_pth, 'space_factors'))
        
        aux = np.load(self.nndist_file)
        nndistarr = aux['nndist']
        
        comp = comps[icomp]
        
        aux = self.classes.loc[self.classes['class'] == comp[0]]
        cl1 = aux.class_name.item()
        aux = self.classes.loc[self.classes['class'] == comp[1]]
        cl2 = aux.class_name.item()
        lbl = '(' + cl1  + '::' + cl2 + ')'
            
        # plots array of landscapes for these comparisons
        fig = plotCompLandscape_simple(self.sid,
                                       nndistarr,
                                       self.roiarr,
                                       icomp,
                                       self.imshape,
                                       metric + ' - ' + lbl,
                                       10, 
                                       lims,
                                       self.scale,
                                       self.units,
                                       self.binsiz)
        # saves to png file
        nam = self.sid +'_nndist_landscape_simple_' + \
            comp[0]+ "_" + comp[1] +'.png'
        fig.savefig(os.path.join(out_pth, nam), bbox_inches='tight', dpi=300)
        plt.close()
                          
        del nndistarr
        gc.collect()    
              

    def plotRHFuncLandscape(self, redo, analyses, lims, do_plots=False):
        """
        Plots Ripley`s H function score landscape from raster
  
        """
        # from itertools import product
        # # generate list of comparisons for coloc
        # comps = [list(c) for c in list(product(self.classes.index, repeat=2))]

        # if this analysis is already done, skip it
        out_pth = mkdirs(os.path.join(self.res_pth, 'space_factors'))
        fout = os.path.join(out_pth, self.sid +'_rhfunc_stats.csv')
        if (redo or not os.path.exists(fout)):
            
            # analyses
            df = analyses.copy()
            isrhfunc = ~df.loc[df['name'] == 'rhfunc']['drop'].values[0]
            
            if isrhfunc:
                aux = np.load(self.rhfunc_file)
                rhfuncarr = aux['rhfunc']
                
                # combinations of cases:
                rhfunc_comps = df.loc[df['name'] == 'rhfunc']['comps'].values[0]
                compsterm = rhfunc_comps.copy()
                compslabs = rhfunc_comps.copy()
                for i, comp in enumerate(rhfunc_comps):
                    aux = self.classes.loc[self.classes['class'] == comp[0]]
                    cl1 = aux.class_name.item()
                    aux = self.classes.loc[self.classes['class'] == comp[1]]
                    cl2 = aux.class_name.item()
                    compsterm[i] = comp[0] + '::' + comp[1]
                    compslabs[i] = '(' + cl1  + '::' + cl2 + ')'
      
                # plots array of landscapes for these comparisons
                [fig, metrics] = plotCompLandscape(self.sid,
                                                   rhfuncarr,
                                                   self.roiarr,
                                                   rhfunc_comps,
                                                   compsterm, 
                                                   compslabs,
                                                   self.imshape,
                                                   'Ripley`s H function score', 
                                                   10,
                                                   lims,
                                                   self.scale, 
                                                   self.units, 
                                                   self.binsiz,
                                                   do_plots)
                # saves landscape metrics table
                metrics.to_csv(fout, sep=',', index=False, header=True)           
                       
                if do_plots:
                    # saves to png file
                    fout = os.path.join(out_pth,
                                        self.sid + '_rhfunc_landscape.png')
                    fig.savefig(fout, bbox_inches='tight', dpi=300)
                    plt.close()
                    
                    if (len(rhfunc_comps)>1):
                        # plot correlations between comparisons
                        ttl = 'Ripley`s H function score Correlations\n' + \
                            'Sample ID: ' + str(self.sid)
                        fig = plotCompCorrelations(self.classes, 
                                                   rhfuncarr, 
                                                   rhfunc_comps, 
                                                   ttl, lims)
                        fout = os.path.join(out_pth,
                                            self.sid + \
                                                '_rhfunc_correlations.png')
                        plt.savefig(fout, bbox_inches='tight', dpi=300)
                        plt.close()
                del rhfuncarr
                gc.collect()       
              

    def plotRHFuncLandscape_simple(self, icomp, comps, metric, lims):
        """
        Plots Ripley`s H function score landscape from raster
  
        """
        # from itertools import product
        # # generate list of comparisons for coloc
        # comps = [list(c) for c in list(product(self.classes.index, repeat=2))]
        
        out_pth = mkdirs(os.path.join(self.res_pth, 'space_factors'))
        
        aux = np.load(self.rhfunc_file)
        rhfuncarr = aux['rhfunc']
        comp = comps[icomp]
        
        aux = self.classes.loc[self.classes['class'] == comp[0]]
        cl1 = aux.class_name.item()
        aux = self.classes.loc[self.classes['class'] == comp[1]]
        cl2 = aux.class_name.item()
        lbl = '(' + cl1  + '::' + cl2 + ')'
                
        # plots array of landscapes for these comparisons
        fig = plotCompLandscape_simple(self.sid,
                                       rhfuncarr,
                                       self.roiarr,
                                       icomp, 
                                       self.imshape,
                                       metric + ' - ' + lbl, 
                                       10,
                                       lims,
                                       self.scale, 
                                       self.units, 
                                       self.binsiz)
        # saves to png file
        nam = self.sid +'_rhfunc_landscape_simple_' + \
            comp[0]+ "_" + comp[1] +'.png'
        fig.savefig(os.path.join(out_pth,nam), bbox_inches='tight', dpi=300)
        plt.close()
                    
        del rhfuncarr
        gc.collect()               


    def plotGeOrdLandscape(self, redo, analyses, lims, 
                           do_zs=False, do_plots=False):
        """
        Plots Getis-Ord Z score landscape from raster
  
        """
  
        # generate list of comparisons
        # comps = list(self.classes.index)
        
        # if this analysis is already done, skip it
        out_pth = mkdirs(os.path.join(self.res_pth, 'space_factors'))

        if (redo):
        
            # analyses
            df = analyses.copy()
            isgeordG = ~df.loc[df['name'] == 'geordG']['drop'].values[0]
            
            if isgeordG:
                
                aux = np.load(self.geordG_file)
                H = aux['hot']
                if do_zs:
                    G = aux['geordG']
                
                # do global Getis-Ord Z score
                if do_zs:
                    fout = os.path.join(out_pth, 
                                        self.sid +'_geordG_stats.csv')
                    geordGarr = G[:,:,:,0]
                fout2 = os.path.join(out_pth, self.sid +'_hots_stats.csv')
                hotarr = H[:,:,:,0]
                
                # combinations of cases:
                geordG_comps = df.loc[df['name']=='geordG']['comps'].values[0]
                compsterm = geordG_comps.copy()
                compslabs = geordG_comps.copy()
                for i, comp in enumerate(geordG_comps):
                    compsterm[i] = comp
                    aux = self.classes.loc[self.classes['class'] == comp]
                    compslabs[i] = aux.class_name.item()
            
                if do_zs:
                    # plots array of landscapes for these comparisons
                    [fig, metrics] = plotCompLandscape(self.sid,
                                                       geordGarr,
                                                       self.roiarr,
                                                       geordG_comps,
                                                       compsterm,
                                                       compslabs,
                                                       self.imshape,
                                                       'Getis-Ord Z',
                                                       10,
                                                       lims,
                                                       self.scale,
                                                       self.units,
                                                       self.binsiz,
                                                       do_plots)
                    # saves landscape metrics table
                    metrics.to_csv(fout, sep=',', index=False, header=True) 
                
                # plots array of landscapes for these comparisons
                [fig2, metrics] = plotDiscreteLandscape(self.sid, 
                                                        hotarr, 
                                                        self.roiarr,
                                                        geordG_comps, 
                                                        compslabs,
                                                        self.imshape,
                                                        'HOT score', 
                                                        [-1, 0, 1],
                                                        [-1, 1],
                                                        self.scale, 
                                                        self.units, 
                                                        self.binsiz,
                                                        do_plots)
                # saves landscape metrics table
                metrics.to_csv(fout2, sep=',', index=False, header=True) 
                
                if do_plots:
                    if do_zs:
                        # saves to png file
                        fout = os.path.join(out_pth,
                                            self.sid +'_geordG_landscape.png')
                        fig.savefig(fout, bbox_inches='tight', dpi=300)
                        plt.close()
                    
                    # saves to png file
                    fout = os.path.join(out_pth, 
                                        self.sid +'_hots_landscape.png')
                    fig2.savefig(fout, bbox_inches='tight', dpi=300)
                    plt.close()
                    
                # do local Getis-Ord Z score
                if do_zs:
                    fout = os.path.join(out_pth, 
                                        self.sid +'_geordG_local_stats.csv')
                    geordGarr = G[:,:,:,1]
                fout2 = os.path.join(out_pth, 
                                    self.sid +'_hots_local_stats.csv')
                hotarr = H[:,:,:,1]
                
                # combinations of cases:
                geordG_comps = df.loc[df['name'] == 'geordG']['comps'].values[0]
                compsterm = geordG_comps.copy()
                compslabs = geordG_comps.copy()
                for i, comp in enumerate(geordG_comps):
                    compsterm[i] = comp
                    aux = self.classes.loc[self.classes['class'] == comp]
                    compslabs[i] = aux.class_name.item()
            
                if do_zs:
                    # plots array of landscapes for these comparisons
                    [fig, metrics] = plotCompLandscape(self.sid,
                                                       geordGarr,
                                                       self.roiarr,
                                                       geordG_comps,
                                                       compsterm,
                                                       compslabs,
                                                       self.imshape,
                                                       'Local Getis-Ord Z', 
                                                       10,
                                                       lims,
                                                       self.scale,
                                                       self.units, 
                                                       self.binsiz,
                                                       do_plots)
                    
                    # saves landscape metrics table
                    metrics.to_csv(fout, sep=',', index=False, header=True) 
                    
                # plots array of landscapes for these comparisons
                [fig2, metrics] = plotDiscreteLandscape(self.sid, 
                                                        hotarr, 
                                                        self.roiarr,
                                                        geordG_comps, 
                                                        compslabs,
                                                        self.imshape,
                                                        'HOT score', 
                                                        [-1, 0, 1],
                                                        [-1, 1],
                                                        self.scale, 
                                                        self.units, 
                                                        self.binsiz,
                                                        do_plots)
                # saves landscape metrics table
                metrics.to_csv(fout2, sep=',', index=False, header=True) 
                
                if do_plots:
                    if do_zs:
                        # saves to png file
                        fout = os.path.join(out_pth,
                                            self.sid + \
                                                '_geordG_local_landscape.png')
                        fig.savefig(fout, bbox_inches='tight', dpi=300)
                        plt.close()
                    
                    # saves to png file
                    fout = os.path.join(out_pth, 
                                        self.sid +'_hots_local_landscape.png')
                    fig2.savefig(fout, bbox_inches='tight', dpi=300)
                    plt.close()    
                
                if do_zs:
                    del geordGarr
                del hotarr
                gc.collect()      
    

    def plotFactorCorrelations(self, redo, analyses, do_plots=False):

        # from itertools import combinations, product, permutations
        # # generates a list of comparisons for coloc
        # comps1 = [list(c) for c in list(combinations(self.classes.index, 2))]
        # # generate list of comparisons for rhfunc
        # comps2 = [list(c) for c in list(product(self.classes.index, repeat=2))]
        # # generate list of comparisons for nndist
        # comps3 = [list(c) for c in list(permutations(self.classes.index, r=2))]
        
        out_pth = mkdirs(os.path.join(self.res_pth, 'space_factors'))
        
        fout = os.path.join(out_pth, self.sid +'_factor_correlations.csv')
        
        if (redo or not os.path.exists(fout)):
        
            # analyses
            df = analyses.copy()
            iscoloc = ~df.loc[df['name'] == 'coloc']['drop'].values[0]
            isnndist = ~df.loc[df['name'] == 'nndist']['drop'].values[0]
            isrhfunc = ~df.loc[df['name'] == 'rhfunc']['drop'].values[0]
            
            cases = self.classes['class'].tolist()
            
            if (iscoloc or isnndist or isrhfunc):
                
                # empty arrays for desider comparisons for each metric
                coloc_comps = []
                nndist_comps = []
                rhfunc_comps = []
                
                # get comparisons and reduced lists of cases
                if iscoloc:
                    # combinations of cases:
                    comps = df.loc[df['name'] == 'coloc']['comps'].values[0]
                    coloc_comps = [(cases.index(c[0]), 
                                    cases.index(c[1])) for c in comps]
                if isnndist:
                    # combinations of cases:
                    comps = df.loc[df['name'] == 'nndist']['comps'].values[0]
                    nndist_comps = [(cases.index(c[0]), 
                                     cases.index(c[1])) for c in comps]
                if isrhfunc:
                    # combinations of cases:
                    comps = df.loc[df['name'] == 'rhfunc']['comps'].values[0]
                    rhfunc_comps = [(cases.index(c[0]), 
                                     cases.index(c[1])) for c in comps]
              
            corr = pd.DataFrame(columns = ['factor_i', 'comp_i', 
                                           'factor_j', 'comp_j',
                                           'pearson_cor', 'p_value:'])
            colocarr = []
            nndistarr = []
            rhfuncarr = []
            
            if iscoloc:
                aux = np.load(self.coloc_file)
                colocarr = aux['coloc']
                if (np.isnan(colocarr).all()):
                    iscoloc = False
                
            if isnndist:
                aux = np.load(self.nndist_file)
                nndistarr = aux['nndist']
                if (np.isnan(nndistarr).all()):
                    isnndist = False
                else:
                    try:
                        vmin = 0.25*round(np.nanquantile(nndistarr, .001)/0.25)
                        vmax = 0.25*round(np.nanquantile(nndistarr, .999)/0.25)
                    except:
                        vmin = -1.5
                        vmax = 1.5
            if isrhfunc:
                aux = np.load(self.rhfunc_file)
                rhfuncarr = aux['rhfunc']
                if (np.isnan(rhfuncarr).all()):
                    isrhfunc = False
                else:
                    try:
                        wmin = 0.25*round(np.nanquantile(rhfuncarr, .001)/0.25)
                        wmax = 0.25*round(np.nanquantile(rhfuncarr, .999)/0.25)
                    except:
                        wmin = -1
                        wmax = 1
          
            if iscoloc and isrhfunc:
                            
                # plot correlations between coloc and RHindex comparisons
                ttl = 'Coloc - RHindex Correlations\nSample ID: ' + \
                    str(self.sid)
                [fig, df] = plotFactorCorrelation(colocarr, 
                                                  coloc_comps, 
                                                  'Coloc', 
                                                  [0.0, 1.0],
                                                  rhfuncarr, 
                                                  rhfunc_comps, 
                                                  'RHindex', 
                                                  [wmin, wmax],
                                                  ttl, 
                                                  self.classes,
                                                  do_plots)
                if do_plots:
                    fig.savefig(os.path.join(out_pth,
                                             self.sid + \
                                                 '_coloc-rhindex_corr.png'),
                                      bbox_inches='tight', dpi=300)
                    plt.close()
                    
                # concatenate correlation tables
                corr =  pd.concat([corr, df], ignore_index=True)
                      
            if iscoloc and isnndist:
                
                # plot correlations between coloc and NNindex comparisons
                ttl = 'Coloc - NNdist Correlations\nSample ID: ' + \
                    str(self.sid)
                [fig, df] = plotFactorCorrelation(colocarr, 
                                                   coloc_comps, 
                                                  'coloc', 
                                                  [0.0, 1.0],
                                                  nndistarr, 
                                                  nndist_comps, 
                                                  'RHindex', 
                                                  [vmin, vmax],
                                                  ttl,
                                                  self.classes,
                                                  do_plots)
                if do_plots:
                    fig.savefig(os.path.join(out_pth,
                                             self.sid + \
                                                 '_coloc-nnindex_corr.png'),
                                bbox_inches='tight', dpi=300)
                    plt.close()
                    
                # concatenate correlation tables
                corr =  pd.concat([corr, df], ignore_index=True)
                      
            if iscoloc and isnndist:
                
                # plot correlations between NNindex and RHindex comparisons
                ttl = 'RHindex - NNdist Correlations\nSample ID: ' + \
                    str(self.sid)
                [fig, df] = plotFactorCorrelation(rhfuncarr, 
                                                  rhfunc_comps, 
                                                  'RHindex', 
                                                  [wmin, wmax],
                                                  nndistarr, 
                                                  nndist_comps, 
                                                  'NNdist', 
                                                  [vmin, vmax],
                                                  ttl,
                                                  self.classes,
                                                  do_plots)
                if do_plots:
                    fig.savefig(os.path.join(out_pth, 
                                             self.sid + \
                                                 '_nnindex-rhindex_corr.png'),
                                bbox_inches='tight', dpi=300)
                    plt.close()
                    
                # concatenate correlation tables
                corr =  pd.concat([corr, df], ignore_index=True)
            
            # saves correlations table
            corr.to_csv(fout, sep=',', index=False, header=True)
            
            del colocarr 
            del nndistarr 
            del rhfuncarr 
            gc.collect()
            

    def landscapeAnalysis(self, redo, do_plots=False):
        
        samplcsv = os.path.join(self.res_pth, self.sid +'_lme_tbl.csv')
        
        if (redo or not os.path.exists(samplcsv)):
            
            import pylandstats as pls
        
            lme_pth = mkdirs(os.path.join(self.res_pth, 'lmes'))
            
            aux = self.lmearr.copy().astype(int)
            
            # create pylandscape object and assing to landscape
            self.plsobj = pls.Landscape(aux, res=(1, 1), 
                                        nodata=0, neighborhood_rule='8')
      
            # get adjacency matrix (as related to LME classes)
            adj_pairs = lmeAdjacency(self.plsobj, self.classes)
            suff = '_lme_adjacency_odds.csv'
            adj_pairs.to_csv(os.path.join(lme_pth, self.sid + suff),
                             sep=',', index=False, header=True)
            
            # get all patch metrics 
            patch_metrics = getPatchMetrics(self.plsobj, self.classes)
            suff = '_lme_patch_metrics.csv'
            patch_metrics.to_csv(os.path.join(lme_pth, self.sid + suff),
                                 sep=',', index=False, header=True)
            # get all class metrics 
            class_metrics = getClassMetrics(self.plsobj, self.classes)
            suff = '_lme_class_metrics.csv'
            class_metrics.to_csv(os.path.join(lme_pth, self.sid + suff),
                                 sep=',', index=False, header=True)
            # get all landscape-level metrics 
            landscape_metrics = getLandscapeMetrics(self.plsobj)
            suff= '_lme_landscape_metrics.csv'
            landscape_metrics.to_csv(os.path.join(lme_pth, self.sid + suff),
                                     sep=',', index=True, 
                                     index_label="metric")
            
            if do_plots:
                # adjacency matrix heatmap
                lmeAdjacencyHeatmap(self.sid, adj_pairs, lme_pth)
                # plot patch metrics histograms
                _ = plotPatchHists(self.sid, patch_metrics, lme_pth)
                # plot class metrics histograms
                _ = plotClassHists(self.sid, class_metrics, lme_pth)
                
                # plot LME landscape
                self.plotLMELandscape(lme_pth)
       
            # get sample stats
            sample_out = getSampleStats(self.tbl, self.plsobj, adj_pairs,
                                        class_metrics, landscape_metrics)
            sample_out.to_csv(samplcsv, sep=',', index=False, header=True)
  
    
# %% Private Functions

# %%%% Setup

def memuse():
    """Memory use"""
    gc.collect()
    m = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
    return(round(m, 2))


def ftime(t):
    """Formated time string"""
    return(time.strftime('%H:%M:%S', time.gmtime(t)))


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
    
    from myfunctions import printProgressBar
    
    printProgressBar(Nsteps*i + step, Nsteps*n,
                     suffix=msg + ' ; ' + msg_i, length=50)
    

# %%%% LME functions

def loadClasses(classes_file):
    """ Load classes and checks that the number of levels for LME 
    definition is not too high

    Args:
        - classes_file(str): name of file with cell classes definitions

    Returns:
        classes

    """

    classes = pd.read_csv(filexists(classes_file),
                          converters={'abundance_edges': literal_eval,
                                      'mixing_edges': literal_eval})
    classes['class'] = classes['class'].astype(str)

    ntest = (max([len(x) for x in classes['abundance_edges']]) > 4)
    mtest = (max([len(x) for x in classes['mixing_edges']]) > 4) 
    
    if ntest or mtest:
        print("ERROR: There are too many LME levels...")
        print("Please use no more than 3 levels (4 edges)")
        sys.exit()

    return(classes)


def lmeRename(lmearr, dim):
    """
    Re-label LME classes according to a coarser definition in order to
    reduce the total number of LME classes

    Coarse LME categories are assigned as a decimal number derived from a
    base-3 numeric value, in which each base-3 bit is:
        * 0, Background
        * 1, "Bare" environment, with none or very few cells
        * 2, "Segmented" environment, cells are not uniformly distributed
        * 3, "Mixed" environment, where cells are mixed uniformly

    Args:
        - lmearr: (numpy) original lmearr (dim-digit code)
        - dim: (int) number of cell classes
    """

    def coarse(lmeval, dim):
        
        def get_digit(number, n):
            return number // 10**n % 10

        # coarses LME categories from lme array to decimal
        v = 0
        if lmeval > 0:
            for d in range(dim):
                j = dim - (d + 1)
                n = get_digit(lmeval, 2*j + 1)
                m = get_digit(lmeval, 2*j)
                if (n*m == 0):
                    k = np.nan
                elif n == 1:
                    k = 0
                elif (m == 1 or (n == 2 and m == 2)):
                    k = 1
                else:
                    k = 2
                v = v + (3**j)*k
            v = v + 1
            
        return(v)

    newarr = lmearr.copy()

    # list unique lme (dim-digit) values
    lmes = np.unique(lmearr, return_counts=False)
    lmes = lmes[~np.isnan(lmes)]

    for lme in lmes:
        newarr[lmearr == lme] = coarse(lme, dim)

    return(newarr)


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

    Args:
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


def lmeAdjacency(ls, classes):
    """
    Gets LME points adjacency matrix

    Args:
        - ls: (pylandstats) landscape object
        - classes_file: (str) name of file with cell classes definitions
    """

    # get adjacency matrix (as related to LME classes)
    aux = ls._adjacency_df.apply(pd.to_numeric, errors='ignore')
    adj_pairs = aux.groupby(level='class_val').sum().reset_index()
    adj_pairs = adj_pairs.loc[adj_pairs['class_val'] > 0]
    # adj_pairs = adj_pairs.drop([0], axis=1)
    adj_pairs = adj_pairs.rename({'class_val': 'LME'}, axis=1)
    adj_pairs['LME'] = [lmeCode(x, len(classes)) for x in adj_pairs['LME']]
    aux = adj_pairs.loc[:, adj_pairs.columns != 'LME'].columns
    cols = ['LME']
    cols.extend([lmeCode(x, len(classes)) for x in aux.astype(int)])
    adj_pairs.columns = cols

    # number of (valid) adjacent pairs
    adj_pairs['total'] = adj_pairs[adj_pairs['LME']].sum(axis=1)

    adj_pairs['areax4'] = 0
    arr = ls.landscape_arr
    for i, c in enumerate(ls.classes):
        cc = lmeCode(int(c),  len(classes))
        # odds per class
        adj_pairs.loc[adj_pairs['LME'] == cc,
                      'areax4'] = int(4*np.sum(arr == int(c)))
        adj_pairs[cc] = adj_pairs[cc]/adj_pairs['total']

    
    return(adj_pairs)

    

# %%%% Space Statistics functions    

def getPatchMetrics(ls, classes):

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        kws = {'area': {'hectares': False},
               'perimeter_area_ratio': {'hectares': False}}

        df = ls.compute_patch_metrics_df(metrics_kws=kws).reset_index()
        df['area_fraction'] = df['area']/ls.landscape_area

        df['LME'] = [lmeCode(x, len(classes)) for x in df['class_val']]
        
        # shift column 'LME' to first position
        df.insert(0, 'LME', df.pop('LME'))

    return(df)


def getClassMetrics(ls, classes):
    ###########################################################################
    # Patch class-level metrics

    # Stadistics across patches for each class
    # `_mn` -> mean
    # `_am` -> area-weighted mean
    # `_md` -> median
    # `_ra` -> range
    # `_sd` -> standard deviation
    # `_cv` -> coefficient of variation

    # class_metrics.columns
    # ['total_area', 'proportion_of_landscape', 'number_of_patches',
    # 'patch_density', 'largest_patch_index',
    # 'total_edge', 'edge_density',
    # 'landscape_shape_index', 'effective_mesh_size',
    # 'area_mn', 'area_am', 'area_md', 'area_ra', 'area_sd', 'area_cv',
    # 'perimeter_mn', 'perimeter_am', 'perimeter_md', 'perimeter_ra',
    # 'perimeter_sd', 'perimeter_cv', 'perimeter_area_ratio_mn',
    # 'perimeter_area_ratio_am', 'perimeter_area_ratio_md',
    # 'perimeter_area_ratio_ra', 'perimeter_area_ratio_sd',
    # 'perimeter_area_ratio_cv', 'shape_index_mn', 'shape_index_am',
    # 'shape_index_md', 'shape_index_ra', 'shape_index_sd', 'shape_index_cv',
    # 'fractal_dimension_mn', 'fractal_dimension_am', 'fractal_dimension_md',
    # 'fractal_dimension_ra', 'fractal_dimension_sd','fractal_dimension_cv',
    # 'euclidean_nearest_neighbor_mn', 'euclidean_nearest_neighbor_am',
    # 'euclidean_nearest_neighbor_md', 'euclidean_nearest_neighbor_ra',
    # 'euclidean_nearest_neighbor_sd', 'euclidean_nearest_neighbor_cv']

    import warnings
    
    kws = {'total_area': {'hectares': False},
           'perimeter_area_ratio': {'hectares': False},
           'patch_density': {'hectares': False},
           'edge_density': {'hectares': False},
           'effective_mesh_size': {'hectares': False}}
    
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        df = ls.compute_class_metrics_df(metrics_kws=kws).reset_index()
        df['LME'] = [lmeCode(x, len(classes)) for x in df['class_val']]
        
    # shift column 'LME' to first position
    df.insert(0, 'LME', df.pop('LME'))
        
    return(df)


def getLandscapeMetrics(ls):
    ###########################################################################
    # Landscape-level metrics: aggregating over all patches of the landscape
 
    # Stadistics across patches 
    # `_mn` -> mean
    # `_am` -> area-weighted mean
    # `_md` -> median
    # `_ra` -> range
    # `_sd` -> standard deviation
    # `_cv` -> coefficient of variation (=sigma/mu)
 
    # class_metrics.columns 
    #['total_area', 'number_of_patches', 'patch_density', 
    # 'largest_patch_index', 'total_edge', 'edge_density', 
    # 'landscape_shape_index', 'effective_mesh_size', 
    # 'area_mn', 'area_am', 'area_md', 'area_ra', 'area_sd', 'area_cv', 
    # 'perimeter_mn', 'perimeter_am', 'perimeter_md', 'perimeter_ra', 
    # 'perimeter_sd', 'perimeter_cv', 'perimeter_area_ratio_mn', 
    # 'perimeter_area_ratio_am', 'perimeter_area_ratio_md', 
    # 'perimeter_area_ratio_ra', 'perimeter_area_ratio_sd', 
    # 'perimeter_area_ratio_cv', 
    # 'shape_index_mn', 'shape_index_am', 'shape_index_md', 'shape_index_ra', 
    # 'shape_index_sd', 'shape_index_cv',
    # 'fractal_dimension_mn', 'fractal_dimension_am', 'fractal_dimension_md', 
    # 'fractal_dimension_ra', 'fractal_dimension_sd','fractal_dimension_cv', 
    # 'euclidean_nearest_neighbor_mn', 'euclidean_nearest_neighbor_am', 
    # 'euclidean_nearest_neighbor_md',
    # 'euclidean_nearest_neighbor_ra', 'euclidean_nearest_neighbor_sd', 
    # 'euclidean_nearest_neighbor_cv',
    # 'contagion', 'shannon_diversity_index']
    
    import warnings
    
    kws = {'total_area': {'hectares': False},
           'edge_density': {'hectares': False},
           'perimeter_area_ratio': {'hectares': False},
           'patch_density': {'hectares': False}}
    
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        df = ls.compute_landscape_metrics_df(metrics_kws=kws).T.squeeze()

    return(df.rename('value'))


def getSampleStats(sample, ls, adj_pairs, class_metrics, landscape_metrics):
    
    # gather sample statistics
    sample_out = sample.copy()
    
    npairs = np.sum(adj_pairs['total'].values)
    npairs_eq = 0
    for i, c in enumerate(adj_pairs['LME']):
        npairs_eq += (adj_pairs['total']*adj_pairs[c]).tolist()[i]
     
    idx = np.logical_or(sample_out.index.str.endswith('_file'),
                         sample_out.index.str.endswith('_dir'))
    sample_out = sample_out.drop(sample.index[idx].tolist())
    sample_out['num_lmes'] = len(ls.classes)
    sample_out['landscape_area'] = ls.landscape_area
    sample_out['adjacency_index'] = npairs_eq/npairs
    
    # Interspersion (or Contagion): measure of aggregation that 
    # represents the probability that two random adjacent cells belong 
    # to the same class.
    # => Approaches '0' when the classes are maximally disaggregated 
    #    (i.e., every cell is a patch of a different class) and interspersed 
    #    (i.e., equal proportions of all pairwise adjacencies)
    # => Approaches '100' when the whole landscape consists of a single patch.
    sample_out['lme_contagion'] = landscape_metrics['contagion']

    # Shannon Diversity Index (Entropy): measure of diversity that reflects 
    # the number of classes present in the landscape as well as the relative 
    # abundance of each class.
    # => Approaches 0 when the entire landscape consists of a single patch.
    # => Increases as the number of classes increases and/or the proportional
    #    distribution of area among classes becomes more equitable.
    sample_out['lme_Shannon'] = landscape_metrics['shannon_diversity_index']

    # Landscape Shape Index: measure of class aggregation that provides a 
    # standardized measure of edginess that adjusts for the size of the 
    # landscape.
    # => Equals 1 when the entire landscape consists of a single patch of the 
    #    corresponding class,
    # => Increases without limit as the patches become more disaggregated and 
    #    uniformly mixed.
    sample_out['lme_shape_index'] = landscape_metrics['landscape_shape_index']
    
    # Simpson Diversity Index (1 - Dominance): measure of diversity that 
    # reflects the dominance of species, taking into account the relative 
    # abundance of each class.
    # => Equals 1 large diversity (more even distribution with no species 
    #    dominating the spectrum) 
    # => Equals 0 small diversity (the entire landscape consists of a single 
    #    patch of the corresponding class)
    ni = np.array(class_metrics.total_area)
    N  = np.sum(ni)
    sample_out['lme_Simpson'] = np.sum(ni*(ni-1))/(N*(N-1))

    # Other metrics
    sample_out['lme_num_patches'] = landscape_metrics['number_of_patches']
    sample_out['lme_patch_density'] = landscape_metrics['patch_density']
    sample_out['lme_total_edge'] = landscape_metrics['total_edge']
    sample_out['lme_edge_density'] = landscape_metrics['edge_density']
    sample_out['lme_largest_patch_index'] = \
        landscape_metrics['largest_patch_index']

    return(sample_out.to_frame().T)
     

# %%%% Plotting functions


def lmeAdjacencyHeatmap(sid, adj_pairs, res_pth ):
    
    import seaborn as sns
    
    mat = adj_pairs.copy()
    mat = mat.set_index('LME') 
    mat = mat[adj_pairs.LME]
    
    mat[mat==0] = 10**(np.floor(np.log10(np.nanmin(mat[mat>0]))))
    mat = np.log10(mat)
    sns.set(rc = {'figure.figsize':(10,10)})
    f = sns.heatmap(mat, linewidth=0.5, cmap="jet", vmax = 0,
                    cbar_kws={'label': r'$\log_{10}(F)$',
                              'shrink': 0.87,
                              'pad': 0.01},
                    xticklabels = 1, yticklabels = 1)
    f.set(title='LME adjacenty odds',
          xlabel=None, ylabel=None);
    f.invert_yaxis()
    f.set_aspect('equal')
    plt.yticks(rotation=0)
    f.figure.savefig(os.path.join(res_pth, 
                                  sid +'_lme_adjacency_odds.png'),
                     bbox_inches='tight', dpi=300)
    plt.tight_layout()
    plt.close()

    return(0)


def plotDiscreteLandscape(sid, raster, roi, comps, compslabs, shape,
                          metric, cticks_in, lims, scale, units, binsiz,
                          do_plots=False):
    """
    Plot of discrete landscape from comparison raster

    Args:
        - sid: sample ID
        - raster: (numpy) raster image
        - comps: (list) class comparisons
        - shape: (tuple) shape in pixels of TLA landscape
        - metric: (str) title of metric ploted
        - lims: (tuple) limits of the metric
        - scale: (float) scale of physical units / pixel
        - units: (str) name of physical units (eg '[um]')
        - binsiz : (float) size of quadrats

    """
    
    import pylandstats as pls
    from myfunctions import plotRGB, plotEdges, tofloat

    dim = len(comps)
    cticks = np.array(cticks_in)
    [ar, redges, cedges, xedges, yedges] = plotEdges(shape, binsiz, scale)
    
    def countsin(x, bins):
        return([len(x[x==b]) for b in bins])
    
    cmap = plt.get_cmap('jet', len(cticks))

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        
        comp_metrics = pd.DataFrame()
        fig = []

        # plots sample image
        if do_plots:
            # plots sample image
            fig, ax = plt.subplots(dim, 2,
                                   figsize=(5*2, 0.5 + math.ceil(5*dim/ar)),
                                   facecolor='w', edgecolor='k')
        
        if len(cticks) > 0: 
        
            for i, comp in enumerate(comps):
    
                aux = raster[:, :, i]

                # generates discrete values of the factor
                auy = aux.copy()
                inx = np.digitize(auy, cticks)
                # sets all values under first bin to first bin
                inx[inx<1] = 1
                auy = tofloat(cticks[inx - 1])
                
                # sets background
                auy[np.isnan(aux)] = np.nan
                auy[~roi] = np.nan
                inx[np.isnan(aux)] = 0
                inx[~roi] = 0
                
                # create pylandscape object with binned factor values
                pl = pls.Landscape(inx, res=(1, 1), 
                                   nodata=0, neighborhood_rule='8')
                
                if (len(pl.classes) > 0):
                    # get all landscape-level metrics for (discrete) factor
                    metrics = getLandscapeMetrics(pl)
                    metrics = metrics.to_frame().transpose()
                    metrics = metrics.reset_index(drop=True)
                    metrics['comp'] =  comp
                    metrics['number_of_levels'] = len(pl.classes)
                    comp_metrics = pd.concat([comp_metrics, metrics],
                                             ignore_index=True)
                    
                if do_plots:
                    freq = countsin(aux[~np.isnan(aux)], cticks)
                    vals = freq/np.sum(roi)
                    vals[vals==0] = np.nan
                    
                    if (dim > 1):  
                        im = plotRGB(ax[i, 0], aux, units,
                                     cedges, redges, xedges, yedges, 
                                     fontsiz=12,
                                     vmin=lims[0], vmax=lims[1], cmap=cmap)
                        ax[i, 0].contour(np.flip(roi, 0), 
                                         [0.5], linewidths=2, colors='black')
                                        
                        cbar = plt.colorbar(im, ax=ax[i, 0], 
                                            fraction=0.046, pad=0.04)
                        cbar.set_ticks(cticks)
                        cbar.set_label(metric, rotation=90, labelpad=2)
        
                        ax[i, 1].bar(cticks, vals,
                                     align='center',
                                     alpha=0.75, color='b', edgecolor='k')
            
                        ax[i, 1].set_title(compslabs[i], fontsize=18, y=1.02)
                        ax[i, 1].set_xlabel(metric)
                        ax[i, 1].set_ylabel('Fraction of pixels in ROI')
                        #ax[i, 1].set_xlim(lims)
                        ax[i, 1].set_xticks(cticks)
                        if np.sum(~np.isnan(vals)) > 0:
                            ax[i, 1].set_yscale('log')
                
                    else:
                        im = plotRGB(ax[0], aux, units,
                                     cedges, redges, xedges, yedges, 
                                     fontsiz=12,
                                     vmin=lims[0], vmax=lims[1], cmap=cmap)
                        ax[0].contour(np.flip(roi, 0),
                                      [0.5], linewidths=2, colors='black')
                        cbar = plt.colorbar(im, ax=ax[0], 
                                            fraction=0.046, pad=0.04)
                        cbar.set_ticks(cticks)
                        cbar.set_label(metric, rotation=90, labelpad=2)
                        ax[1].bar(cticks, vals, align='center',
                                  alpha=0.75, color=cmap(), edgecolor='k')
                        ax[1].set_title(compslabs[i], fontsize=18, y=1.02)
                        ax[1].set_xlabel(metric)
                        ax[1].set_ylabel('Fraction of pixels in ROI')
                        #ax[1].set_xlim(lims)
                        ax[1].set_xticks(cticks)
                        if np.sum(~np.isnan(vals)) > 0:
                            ax[1].set_yscale('log')
            
            if do_plots:
                fig.suptitle('Sample ID: ' + str(sid), fontsize=24, y=.95)
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                
            # shift column 'comp' to first position
            if (len(comp_metrics) > 0):
                comp_metrics.insert(0, 'comp', comp_metrics.pop('comp'))
                comp_metrics.insert(1, 'number_of_levels', 
                                    comp_metrics.pop('number_of_levels'))
            
    return([fig, comp_metrics])


def plotCompLandscape(sid, raster, roi, comps, compsterm, compslabs, shape,
                      metric, nlevs, lims_in, scale, units, binsiz, 
                      do_plots=False, same_lims=True):
    """Calculate features fragmentation and plot of landscape from 
       comparison raster
    
    Args:
        - sid (str): sample ID
        - raster (numpy): raster array to plot
        - roi (numpu): roi
        - comps (list): class comparisons
        - compsterm (list): comparison string terms.
        - compslabs (list): comparison labels.
        - shape (tuple): shape in pixels of landscape
        - metric (str): metric label
        - nlevs (int): number of contour levels for fragmentation analysis
        - lims_in (list): limits for plotting.
        - scale (float): scale of physical units / pixel
        - units (str): ame of physical units (eg '[um]')
        - binsiz (float): scale of quadrats (long-scale)
        - do_plots (bool, optional): do plots. Defaults to False.
        - same_lims (bool, optional): use same limits in all subplots. 
                                      Defaults to True.

    Returns:
        [fig, metrics] : figure object and metrics table

    """
    
    import pylandstats as pls
    from myfunctions import plotRGB, plotEdges

    dim = len(comps)
    [ar, redges, cedges, xedges, yedges] = plotEdges(shape, binsiz, scale)
       
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        
        comp_metrics = pd.DataFrame()
        fig = []

        # plots sample image
        if do_plots:
            fig, ax = plt.subplots(dim, 2,
                                   figsize=(5*2, 0.5 + math.ceil(5*dim/ar)),
                                   facecolor='w', edgecolor='k')

        for i, comp in enumerate(comps):
            
            if same_lims:
                lims = lims_in
            else:
                lims = lims_in[i]
                
            bini = (lims[1] - lims[0])/(nlevs)
            bticks = np.round(np.arange(lims[0], 
                                        lims[1] + bini, bini), 3)
            cmap = plt.get_cmap('jet', nlevs)

            aux = raster[:, :, i]
                        
            # generates discrete values of the factor
            auy = aux.copy()
            inx = np.digitize(auy, bticks)
            # sets all values under first bin to first bin
            inx[inx<1] = 1
            auy = bticks[inx - 1]
            
            # sets background
            auy[np.isnan(aux)] = np.nan
            auy[~roi] = np.nan
            inx[np.isnan(aux)] = 0
            inx[~roi] = 0
            
            # create pylandscape object with binned factor values
            pl = pls.Landscape(inx, res=(1, 1), 
                               nodata=0, neighborhood_rule='8')
            
            if (len(pl.classes) > 0):
                # get all landscape-level metrics for (discrete) factor
                metrics = getLandscapeMetrics(pl)
                metrics = metrics.to_frame().transpose()
                metrics = metrics.reset_index(drop=True)
                metrics['comp'] =  compsterm[i]
                metrics['number_of_levels'] = len(pl.classes)
                comp_metrics = pd.concat([comp_metrics, metrics],
                                         ignore_index=True)
            if do_plots:
                freq, _ = np.histogram(aux[~np.isnan(aux)], 
                                       bins=bticks,
                                       density=False)
                vals = freq/np.sum(roi)
                vals[vals==0] = np.nan
                
                if (dim > 1):  
                    im = plotRGB(ax[i, 0], auy, units,
                                 cedges, redges, xedges, yedges, 
                                 fontsiz=12,
                                 vmin=lims[0], vmax=lims[1], cmap=cmap)
                    ax[i, 0].contour(np.flip(roi, 0), 
                                     [0.5], linewidths=2, colors='black')
        
                    cbar = plt.colorbar(im, ax=ax[i, 0], 
                                        fraction=0.046, pad=0.04)
                    cbar.set_ticks(bticks[::2])
                    cbar.set_label(metric, rotation=90, labelpad=2)
                    
                    ax[i, 1].bar(bticks[:-1], vals,
                                 width=bini, align='edge',
                                 alpha=0.75, 
                                 color=cmap(range(len(bticks))),
                                 edgecolor='k')
        
                    ax[i, 1].set_title(compslabs[i], 
                                       fontsize=18, y=1.02)
                    ax[i, 1].set_xlabel(metric)
                    ax[i, 1].set_ylabel('Fraction of pixels in ROI')
                    ax[i, 1].set_xlim([bticks[0], bticks[-1]])
                    ax[i, 1].set_xticks(bticks[::2])
                    ax[i, 1].set_xticklabels(bticks[::2], 
                                             rotation=90, fontsize=14)
                    if np.sum(~np.isnan(vals)) > 0:
                        ax[i, 1].set_yscale('log')
                    
                else:
                    im = plotRGB(ax[0], auy, units,
                                 cedges, redges, xedges, yedges, 
                                 fontsiz=12,
                                 vmin=lims[0], vmax=lims[1], cmap=cmap)
                    ax[0].contour(np.flip(roi, 0), 
                                     [0.5], linewidths=2, colors='black')
                    cbar = plt.colorbar(im, ax=ax[0], 
                                        fraction=0.046, pad=0.04)
                    cbar.set_ticks(bticks[::2])
                    cbar.set_label(metric, rotation=90, labelpad=2)
                    ax[1].bar(bticks[:-1], vals,
                              width=bini, align='edge', alpha=0.75, 
                              color=cmap(range(len(bticks))), edgecolor='k')
                    ax[1].set_title(compslabs[i], fontsize=18, y=1.02)
                    ax[1].set_xlabel(metric)
                    ax[1].set_ylabel('Fraction of pixels in ROI')
                    ax[1].set_xlim([bticks[0], bticks[-1]])
                    ax[1].set_xticks(bticks[::2])
                    ax[1].set_xticklabels(bticks[::2], 
                                          rotation=90, fontsize=14)
                    if np.sum(~np.isnan(vals)) > 0:
                        ax[1].set_yscale('log')
                    
        if do_plots:
            fig.suptitle('Sample ID: ' + str(sid), fontsize=24, y=.95)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # shift column 'comp' to first position
        if (len(comp_metrics) > 0):
            comp_metrics.insert(0, 'comp', comp_metrics.pop('comp'))
            comp_metrics.insert(1, 'number_of_levels', 
                                comp_metrics.pop('number_of_levels'))
    
    return([fig, comp_metrics])


def plotCompLandscape_simple(sid, raster, roi, idx, shape, 
                             metric, nlevs, lims, scale, units, binsiz):
    """
    Plot of landscape from comparison raster
    
    Args:
        - sid: sample ID
        - raster: (numpy) raster image
        - comps: (list) class comparisons
        - shape: (tuple) shape in pixels of TLA landscape
        - metric: (str) title of metric ploted
        - nlevs: (int) number of contour levels
        - lims: (tuple) limits of the metric
        - scale: (float) scale of physical units / pixel
        - units: (str) name of physical units (eg '[um]')
        - binsiz : (float) size of quadrats

    """
    
    from myfunctions import plotRGB, plotEdges

    bini = (lims[1] - lims[0])/(nlevs)
    bticks = np.arange(lims[0], lims[1] + bini, bini)
    
    [ar, redges, cedges, xedges, yedges] = plotEdges(shape, binsiz, scale)

    vmax = lims[1]
        
    cmap = plt.get_cmap('jet', len(bticks))

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        
        # plots sample image
        fig, ax = plt.subplots(1, 1,
                               figsize=(10, 0.5 + math.ceil(10/ar)),
                               facecolor='w', edgecolor='k')

        aux = raster[:, :, idx]
            
        # generates discrete values of the factor
        auy = aux.copy()
        auy[np.isnan(aux)] = 0
        
        inx = np.digitize(auy, bticks)
        inx[np.isnan(aux)] = 0
        
        auy = bticks[inx - 1]
        auy[np.isnan(aux)] = np.nan
            
        im = plotRGB(ax, auy, units,
                     cedges, redges, xedges, yedges, fontsiz=24,
                     vmin=lims[0], vmax=vmax, cmap=cmap)
        ax.contour(np.flip(roi, 0), 
                   [0.5], linewidths=2, colors='black')
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_ticks(bticks[::2])
        #cbar.set_label(metric, rotation=90, labelpad=2)
        
        ttl = 'Sample ID: ' + str(sid) + '\n' + metric
        fig.suptitle(ttl, fontsize=24, y=.95)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
    return(fig)


def plotCompCorrelations(classes, raster, comps, ttl, lims):

    import seaborn as sns
    import scipy.stats as sts

    nc = 0
    if len(comps)>1:
        nc = len(comps)*(len(comps)-1)/2
    ncols = int(np.ceil(np.sqrt(nc)))
    nrows = int(np.ceil(nc/ncols))

    fig, ax = plt.subplots(nrows, ncols,
                           figsize=(ncols*5, nrows*5),
                           facecolor='w', edgecolor='k')
    
    def corij(auxi, auxj, ax, compi, compj, lims):
        
        aux = np.stack((auxi, auxj)).T
        aux = aux[~np.isnan(aux).any(axis=1)]
        # calculate correlation in <=1000 sample points
        aux = aux[np.random.randint(len(aux),
                                    size=min(1000, len(aux))), :]
        
        cli1 = classes.loc[classes['class'] == compi[0]].class_name.item()
        cli2 = classes.loc[classes['class'] == compi[1]].class_name.item()
        clj1 = classes.loc[classes['class'] == compj[0]].class_name.item()
        clj2 = classes.loc[classes['class'] == compj[1]].class_name.item()
        xlab = '(' + cli1 + '::' + cli2 + ')'
        ylab = '(' + clj1 + '::' + clj2  + ')'

        sns.axes_style("whitegrid")
        sns.regplot(x=aux[:, 0], y=aux[:, 1],
                    ax=ax,
                    scatter_kws={'s':5, "color": "black"},
                    line_kws={"color": "red"})

        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.plot(np.linspace(lims[0], lims[1], 10),
                np.linspace(lims[0], lims[1], 10),
                color='k', linestyle='dashed')
        ax.set_xlim(lims[0], lims[1])
        ax.set_ylim(lims[0], lims[1])
        ax.axis('square')
            
        coefs = 'C = NA'
        if (len(aux > 5)):
            correlation, p_value = sts.pearsonr(aux[:, 0], aux[:, 1])
            star = 'NS'
            if (p_value < 0.05):
                star = '*'
            if (p_value < 0.01):
                star = '**'
            if (p_value < 0.001):
                star = '***'
            coefs = 'C = %.4f; p-value = %.2e' % (correlation, p_value) +\
                ' (' + star + ')'
            
        ax.set_title(coefs)
        
    if (nc > 1):
        shp = ax.shape
        k = 0
        for i, compi in enumerate(comps):
            auxi = raster[:, :, i].ravel()
            for j, compj in enumerate(comps):
                if j > i:
                    auxj = raster[:, :, j].ravel()
                    corij(auxi, auxj, 
                          ax[np.unravel_index(k, shp)], 
                          compi, compj, lims)
                    k = k+1
        for i in range(k, shp[0]*shp[1]):
            ax[np.unravel_index(i, shp)].set_xlim(lims[0], lims[1])
            ax[np.unravel_index(i, shp)].set_ylim(lims[0], lims[1])
            ax[np.unravel_index(i, shp)].axis('square')
            ax[np.unravel_index(i, shp)].remove()
            
    elif (nc == 1):
        auxi = raster[:, :, 0].ravel()
        auxj = raster[:, :, 1].ravel()
        corij(auxi, auxj, 
              ax, 
              comps[0], comps[1], lims)
        
    fig.subplots_adjust(hspace=0.4)
    fig.suptitle(ttl, fontsize=16, y=.95)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    return(fig)


def plotPatchHists(sid, df, res_pth):
    
    import matplotlib as mpl

    lmes = pd.unique(df['LME']).tolist()
    lab = ['LME class: ' + i for i in lmes]

    # plot some patch metrics distributions
    fig, axs = plt.subplots(2, 2, figsize=(10, 10),
                            facecolor='w', edgecolor='k')

    col = [mpl.cm.jet(x) for x in np.linspace(0, 1, len(lmes))]

    for i, lme in enumerate(lmes):

        mif = np.log10(np.min(df.area_fraction))
        maf = np.log10(np.max(df.area_fraction))
        dat = df.loc[df['LME'] == lme].area_fraction.apply(np.log10)
        hist, bins = np.histogram(dat,
                                  bins=np.arange(mif,
                                                 maf,
                                                 (maf-mif)/21),
                                  density=False)
        x = (bins[1:] + bins[:-1])/2
        axs[0, 0].plot(x, hist, label=lab[i], color=col[i])

        mir = np.log10(np.min(df.perimeter_area_ratio))
        mar = np.log10(np.max(df.perimeter_area_ratio))
        dat = df.loc[df['LME'] == lme].perimeter_area_ratio.apply(np.log10)
        hist, bins = np.histogram(dat,
                                  bins=np.arange(mir,
                                                 mar,
                                                 (mar-mir)/21),
                                  density=False)
        x = (bins[1:] + bins[:-1])/2
        axs[0, 1].plot(x, hist, label=lab[i], color=col[i])

        mis = np.log10(np.min(df.shape_index))
        mas = np.log10(np.max(df.shape_index))
        dat = df.loc[df['LME'] == lme].shape_index.apply(np.log10)
        hist, bins = np.histogram(dat,
                                  bins=np.arange(mis,
                                                 mas,
                                                 (mas-mis)/21),
                                  density=False)

        x = (bins[1:] + bins[:-1]) / 2
        axs[1, 0].plot(x, hist, label=lab[i], color=col[i])

        mie = np.log10(np.min(df.euclidean_nearest_neighbor))
        mae = np.log10(np.max(df.euclidean_nearest_neighbor))
        aux = df.loc[df['LME'] == lme]
        dat = aux.euclidean_nearest_neighbor.apply(np.log10)
        hist, bins = np.histogram(dat,
                                  bins=np.arange(mie,
                                                 mae,
                                                 (mae-mie)/21),
                                  density=False)
        x = (bins[1:] + bins[:-1])/2
        axs[1, 1].plot(x, hist, label=lab[i], color=col[i])

    #axs[0, 0].set_xticks(np.arange(-10, 10, 1))
    axs[0, 0].set_xlim(mif, maf)
    axs[0, 0].locator_params(tight=True, nbins=5)
    axs[0, 0].get_xaxis().set_major_formatter(mpl.ticker.FuncFormatter(
        lambda x, p: r'$10^{%.1f}$' % x))
    axs[0, 0].set_xlabel('Area fraction of patches (log)')
    axs[0, 0].set_ylabel('Frequency')

    #axs[0, 1].set_xticks(np.arange(-10, 10, 1))
    axs[0, 1].set_xlim(mir, mar)
    axs[0, 1].locator_params(tight=True, nbins=5)
    axs[0, 1].get_xaxis().set_major_formatter(mpl.ticker.FuncFormatter(
        lambda x, p: r'$10^{%.1f}$' % x))
    axs[0, 1].set_xlabel('Perimeter-Area ratio of patches (log)')
    axs[0, 1].set_ylabel('Frequency')

    axs[1, 0].set_xlim(mis, mas)
    axs[1, 0].locator_params(tight=True, nbins=5)
    axs[1, 0].get_xaxis().set_major_formatter(mpl.ticker.FuncFormatter(
        lambda x, p: r'$10^{%.1f}$' % x))
    axs[1, 0].set_xlabel('Shape index of patches (log)')
    axs[1, 0].set_ylabel('Frequency')

    axs[1, 1].set_xlim(mie, mae)
    axs[1, 1].locator_params(tight=True, nbins=5)
    axs[1, 1].get_xaxis().set_major_formatter(mpl.ticker.FuncFormatter(
        lambda x, p: r'$10^{%.1f}$' % x))
    axs[1, 1].set_xlabel('Euclidean Nearest Neighbor of patches (log)')
    axs[1, 1].set_ylabel('Frequency')


    fig.legend(lmes, 
               loc='center right', 
               bbox_to_anchor=(1.05,0.5), 
               ncol=1, 
               bbox_transform=fig.transFigure)
    fig.subplots_adjust(hspace=0.4)
    fig.suptitle('Sample ID: ' + str(sid), fontsize=24, y=.95)
    #plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(os.path.join(res_pth, sid +'_lme_patch_metrics_hists.png'),
                bbox_inches='tight', dpi=300)
    plt.close()

    return(fig)


def plotClassHists(sid, df, res_pth):
    # plot some patch metrics distributions
    
    import matplotlib as mpl

    lmes = pd.unique(df['LME']).tolist()
    cols = [mpl.cm.jet(x) for x in np.linspace(0, 1, len(lmes))]

    fig, axs = plt.subplots(2, 2, figsize=(10, 10),
                            facecolor='w', edgecolor='k')

    axs[0, 0].bar(lmes, df.patch_density, color=cols, align='center')
    axs[0, 0].set_title('Patch Density')
    axs[0, 0].set_xticks(lmes)
    axs[0, 0].set_xticklabels(lmes, rotation=90, fontsize=8)
    axs[0, 1].bar(lmes, df.largest_patch_index, color=cols, align='center')
    axs[0, 1].set_title('Largest Patch Index')
    axs[0, 1].set_xticks(lmes)
    axs[0, 1].set_xticklabels(lmes, rotation=90, fontsize=8)
    axs[1, 0].bar(lmes, df.edge_density, color=cols, align='center')
    axs[1, 0].set_title('Edge Density')
    axs[1, 0].set_xticks(lmes)
    axs[1, 0].set_xticklabels(lmes, rotation=90, fontsize=8)
    axs[1, 1].bar(lmes, df.landscape_shape_index, color=cols, align='center')
    axs[1, 1].set_title('Landscape Shape Index')
    axs[1, 1].set_xticks(lmes)
    axs[1, 1].set_xticklabels(lmes, rotation=90, fontsize=8)

    fig.subplots_adjust(hspace=0.4)
    fig.suptitle('Sample ID: ' + str(sid), fontsize=24, y=.95)
    fig.savefig(os.path.join(res_pth, sid +'_lme_class_metrics_hists.png'),
                bbox_inches='tight', dpi=300)
    #plt.tight_layout()
    plt.close()

    # get class coverage graphs
    fig2, axs = plt.subplots(1, 2, figsize=(10, 5),
                             facecolor='w', edgecolor='k')

    y = df.total_area
    f = 100.*y/y.sum()
    patches, texts = axs[0].pie(y, colors=cols, startangle=10)
    axs[0].axis('equal')
    axs[0].set_title('Proportion of landscape')
    labs = ['{0} - {1:1.2f} %'.format(i, j) for i, j in zip(lmes, f)]
    axs[0].legend(patches, labs, 
                  loc='center left',
                  bbox_to_anchor=(0.0, 0.5), 
                  fontsize=8,
                  prop={'size': 8},
                  bbox_transform=fig2.transFigure)

    y = df.number_of_patches
    f = 100.*y/y.sum()
    patches, texts = axs[1].pie(y, colors=cols, startangle=0)
    axs[1].axis('equal')
    axs[1].set_title('Number of Patches')
    labs = ['{0} - {1:1.2f} %'.format(i, j) for i, j in zip(lmes, f)]
    axs[1].legend(patches, labs, 
                  loc='center right',
                  bbox_to_anchor=(1.0, 0.5), 
                  fontsize=8,
                  prop={'size': 8},
                  bbox_transform=fig2.transFigure)

    fig2.subplots_adjust(hspace=0.5)
    fig2.suptitle('Sample ID: ' + str(sid), fontsize=24, y=.95)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig2.savefig(os.path.join(res_pth, sid +'_lme_class_coverage.png'),
                 bbox_inches='tight', dpi=300)
    plt.close()

    return([fig, fig2])


def plotViolins(tbl, grps, glab, signal, slab, fname):
    
    import seaborn as sns
    from statannot import add_stat_annotation
    from itertools import combinations
    
    aux = tbl[grps].to_list()
    auy = tbl[signal].to_list()
    grp = sorted(set(aux), reverse=False)
    
    fig, ax = plt.subplots(figsize=(10, 10), facecolor='w', edgecolor='k')
    
    _ = sns.axes_style("whitegrid")
    _ = sns.violinplot(x  = aux, 
                       y  = auy, 
                       ax = ax,
                       #palette ="jet", 
                       scale = 'count', 
                       inner = 'box',
                       order = grp)
    _ = sns.swarmplot(x  = aux, 
                      y  = auy, 
                      ax = ax,
                      color = "black",
                      order = grp)
    
    _ = add_stat_annotation(ax, x=aux, y=auy, order=grp,
                            box_pairs=combinations(grp, 2),
                            test='t-test_ind', 
                            line_offset_to_box=0.2,
                            #text_format='full',
                            text_format='star',
                            loc='inside', verbose=0);    
    ax.set_xlabel(glab)
    ax.set_ylabel(slab)
    sns.set(font_scale = 2)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(fname, bbox_inches='tight', dpi=300)
    plt.close()
         
    return(0)


def plotFactorCorrelation(rasteri, comps1, tti, limsi,
                          rasterj, comps2, ttj, limsj,
                          ttl, classes, do_plots=False):

    import seaborn as sns
    import scipy.stats as sts
    import warnings
    
    df = pd.DataFrame()
    fig = []
    
    # plots sample image
    if do_plots:
        nc = len(comps1)*len(comps2)
        ncols = int(np.ceil(np.sqrt(nc)))
    
        fig, ax = plt.subplots(int(np.ceil(nc/ncols)), ncols,
                               figsize=(ncols*(10), (nc/ncols)*(10)),
                               facecolor='w', edgecolor='k')
        shp = ax.shape

    k = 0
    for i, compi in enumerate(comps1):
        auxi = rasteri[:, :, i].ravel()
        for j, compj in enumerate(comps2):
            auxj = rasterj[:, :, j].ravel()
            aux = np.stack((auxi, auxj)).T
            aux = aux[~np.isnan(aux).any(axis=1)]
            aux = aux[np.random.randint(len(aux),
                                        size=min(1000, len(aux))), :]
            if (len(aux) > 5):
                
                compi_lab = classes.iloc[compi[0]].class_name + \
                    '::' + classes.iloc[compi[1]].class_name
                compj_lab = classes.iloc[compj[0]].class_name + \
                    '::' + classes.iloc[compj[1]].class_name 
                    
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    correlation, p_value = sts.pearsonr(aux[:, 0], aux[:, 1])
                star = 'NS'
                if (p_value < 0.05):
                    star = '*'
                if (p_value < 0.01):
                    star = '**'
                if (p_value < 0.001):
                    star = '***'
                coefs = 'C = %.4f; p-value = %.2e' % (correlation, p_value)
                
                dfij = pd.DataFrame({'factor_i': [tti], 
                                     'comp_i': [compi_lab],
                                     'factor_j': [ttj],
                                     'comp_j': [compj_lab],
                                     'pearson_cor': [correlation],
                                     'p_value:': [p_value]})
                df = pd.concat([df, dfij], ignore_index=True)

                if do_plots:
                    ij = np.unravel_index(k, shp)
                    sns.axes_style("whitegrid")
                    sns.regplot(x=aux[:, 0], y=aux[:, 1],
                                ax=ax[np.unravel_index(k, shp)],
                                scatter_kws={"color": "black"},
                                line_kws={"color": "red"})
                    ax[ij].set_xlabel(tti + '(' + compi_lab + ')', size=20)
                    ax[ij].set_ylabel(ttj + '(' + compi_lab + ')', size=20)
                    ax[ij].plot(np.linspace(limsi[0], limsi[1], 10),
                                np.linspace(limsj[0], limsj[1], 10),
                                color='k', linestyle='dashed')
                    ax[ij].set_xlim(limsi[0], limsi[1])
                    ax[ij].set_ylim(limsj[0], limsj[1])
                    ax[ij].axis('equal')
                    ax[ij].set_title(coefs + ' (' + star + ')', size=20)
                    
                k = k + 1
    for i in range(k, shp[0]*shp[1]):
        ax[np.unravel_index(i, shp)].axis('square')
        ax[np.unravel_index(i, shp)].remove()
        
    if do_plots:
        fig.subplots_adjust(hspace=0.4)
        fig.suptitle(ttl, fontsize=24, y=.95)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    return([fig, df])


    # %% Main function

def main(args):
    """
    *******  Main function  *******

    """
    # %% STEP 0: start, checks how the program was launched 
    debug = False
    try:
        args
    except NameError:
        #  if not running from the CLI, run in debug mode
        debug = True
    
    if debug:
        # running from the IDE
        # path of directory containing this script
        main_pth = os.path.dirname(os.getcwd())
        f = os.path.join(main_pth, 'pathAI.csv')
        REDO = True
        GRPH = True
        CASE = 11
    else:
        # running from the CLI using the bash script
        # path to working directory (above /scripts)
        main_pth = os.getcwd()
        f = os.path.join(main_pth, args.argsfile) 
        REDO = args.redo
        GRPH = args.graph
        CASE = args.casenum
        
    argsfile = filexists(f)

    print("-------------------------------------------------")
    print("==> The working directory is: " + main_pth)
    print("==> Is CUDA available: " + str(ISCUDA))

    if not os.path.exists(argsfile):
        print("ERROR: The specified argument file does not exist!")
        sys.exit()
        
    # NOTE: only ONE line in the argument table will be used
    study = Study( pd.read_csv(argsfile).iloc[0], main_pth)
    
    
    # %% STEP 1: create sample object
    sample = Sample(CASE, study)

    if (sample.num_cells > 0):
        
        print(sample.msg)
        
        # tracks time and memory usage
        progress = Progress(sample.sid)
        progress.dostep(debug, 'Sample object created')
        
        
        # %% STEP 2: calculate kernel-level space stats
        sample.spaceStats(REDO, study.analyses)
        progress.dostep(debug, 'Space Stats calculated')
        
        
        # %%STEP 3: coloc stats
        sample.plotColocLandscape(REDO, study.analyses, [0.0 , 1.0], 
                                  do_plots=GRPH)
        progress.dostep(debug, 'Coloc Stats calculated')
        
        
        # %%STEP 4: NNDist stats
        sample.plotNNDistLandscape(REDO, study.analyses, [-2.5, 2.5], 
                                   do_plots=GRPH)
        progress.dostep(debug, 'NNDist Stats calculated')
        
        
        # %%STEP 5: RHFunc stats        
        sample.plotRHFuncLandscape(REDO,  study.analyses, [-1.5, 1.5], 
                                   do_plots=GRPH)
        progress.dostep(debug, 'RHFunc Stats calculated')
        
        
        # %%STEP 6: GeOrd stats
        sample.plotGeOrdLandscape(REDO, study.analyses, [-2.5, 2.5], 
                                  do_zs=False, do_plots=GRPH)
        progress.dostep(debug, 'GeOrd Stats calculated')
        
        
        # %%STEP 7: inter-metric Correlations
        sample.plotFactorCorrelations(REDO, study.analyses, do_plots=GRPH)
        progress.dostep(debug, 'Inter-metric correlations')
        
        
        # %%STEP 8: simple plots
        if (GRPH):
            df = study.analyses
            
            if ~df.loc[df['name'] == 'coloc']['drop'].values[0]:
                comps = list(df.loc[df['name'] == 'coloc'].comps.values[0])
                for i, comp in enumerate(comps):
                    sample.plotColocLandscape_simple(i, comps, 
                                                   'Coloc index', [0, 1])
                
            if ~df.loc[df['name'] == 'nndist']['drop'].values[0]:
                comps = list(df.loc[df['name'] == 'nndist'].comps.values[0])
                for i, comp in enumerate(comps):
                    sample.plotNNDistLandscape_simple(i, comps,
                                                    'NND index', [-2.5, 2.5])
            
            if ~df.loc[df['name'] == 'rhfunc']['drop'].values[0]:
                comps = list(df.loc[df['name'] == 'rhfunc'].comps.values[0])
                for i, comp in enumerate(comps):
                    sample.plotRHFuncLandscape_simple(i, comps, 
                                                    'RHF index', [-1.5, 1.5])
            progress.dostep(debug, 'Simple Plots created')
    
    
        # %%STEP 9: assigns LME values to landscape
        sample.loadLMELandscape(REDO, do_plots=GRPH)
        progress.dostep(debug, 'LME Landscape created')
        
        
        # %%STEP 10: pylandstats analysis
        # Regular metrics can be computed at the patch, class and
        # landscape level. For list of all implemented metrics see: 
        # https://pylandstats.readthedocs.io/en/latest/landscape.html
        
        sample.landscapeAnalysis(REDO, do_plots=GRPH)
        progress.dostep(debug, 'Landscape Analysis calculated')
    
    
    # %% if sample has no data    
    else:
        
        print( "==> sample <" + sample.sid + \
              "> dropped from analysis (no cell data)..." )
    
        
    # %% LAST step: saves study stats results for sample 
    memmax = np.max(np.nanmax(progress.mbuse))
    print('====> Sample: ' + sample.sid + ' finished. Time elapsed: ', 
          ftime(progress.runtime), '[HH:MM:SS]')
    print("====> Max memory used: " + str(memmax) + "[MB]")
    
    with open(study.done_list, 'a') as f:
        f.write(sample.sid + '\n')
    
    
    #%%
    return(0)        
        

# %% Argument parser
if __name__ == "__main__":

    # Create the parser
    my_parser = ArgumentParser(prog="tla_sample",
                               description="# Single Sample Processing " +
                               "module for Tumor Landscape Analysis #",
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
                           help="Set case number to be processed (zero based)")

    my_parser.add_argument("--graph",
                           default=False,
                           action="store_true",
                           help="If <<--graph>> is used, then print graphs")

    my_parser.add_argument("--redo",
                           default=False,
                           action="store_true",
                           help="If --redo is used, re-do landscape analysis")

    # passes arguments object to main
    main(my_parser.parse_args())
