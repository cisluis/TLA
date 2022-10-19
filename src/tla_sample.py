'''
    Tumor Landscape Analysis (TLA):
    ##############################

        This script reads lines from a study set table
        Each line has parameters of a particular study

        This formating helps faster and more consistent running of the TLA


'''

# %% Imports
import os
import gc
import sys
import math

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pylandstats as pls
# import pickle

from skimage import io
from PIL import Image
from ast import literal_eval

from argparse import ArgumentParser

Image.MAX_IMAGE_PIXELS = 600000000

__version__  = "1.1.0"


# %% Private classes

class Study:
    
  def __init__(self, study, main_pth):
      
      # loads arguments for this study
      self.name = study['name']
      f = os.path.join(main_pth, study['data_path'])
      if not os.path.exists(f):
          print("ERROR: data folder " + f + " does not exist!")
          print("... Please run << TLA setup >> for this study!")
          sys.exit()
      self.dat_pth = f
      
      f = os.path.join(self.dat_pth, study['name'] + '_classes.csv')
      if not os.path.exists(f):
          print("ERROR: classes file " + f + " does not exist!")
          sys.exit()
      self.classes = getLMEclasses(f)
      
      f = os.path.join(self.dat_pth, study['name'] + '_samples.csv')
      if not os.path.exists(f):
          print("ERROR: samples file " + f + " does not exist!")
          sys.exit()
      self.samples = pd.read_csv(f)
      self.samples.fillna('', inplace=True)
     
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
      
      
  def getStudy(self, i):
      
      # get sample entry from samples df
      sample = self.samples.iloc[i].copy()
            
      # check that results dir exist
      f = os.path.join(self.dat_pth, sample['results_dir'])
      if not os.path.exists(f):
          print("ERROR: data folder " + f + " does not exist!")
          print("... Please run << TLA setup >> for this study!")
          sys.exit()
      sample['res_pth'] = f    
      
      return(sample)
      

class Landscape:
    
  def __init__(self, sample, study):
      
      from myfunctions import mkdirs
      
      dat_pth = study.dat_pth
      
      self.sid = sample.sample_ID
      self.res_pth = sample['res_pth']
      
      f = os.path.join(self.res_pth, self.sid +'_classes.csv')
      if not os.path.exists(f):
          print("ERROR: classes file " + f + " does not exist!")
          sys.exit()
      self.classes = pd.merge(pd.read_csv(f), 
                              study.classes, 
                              how="left",
                              on=['class', 'class_name',
                                  'class_val', 'class_color']).fillna(0)
      
      self.coord_file = os.path.join(dat_pth, sample['coord_file'])
       
      # general attributes
      self.binsiz = study.binsiz
      self.subbinsiz = study.subbinsiz 
      self.kernel =  study.kernel
      self.subkernel = study.subkernel
      self.scale = study.scale
      self.units = study.units
      
      # loads all raster images
      f = os.path.join(dat_pth, sample['roi_file'])
      if not os.path.exists(f):
          print("ERROR: raster file " + f + " does not exist!")
          sys.exit()
      aux = np.load(f)
      self.roiarr = aux['roi'] # ROI raster
      self.imshape = self.roiarr.shape
      self.kde_file = os.path.join(dat_pth, sample['kde_file'])
      self.abumix_file = os.path.join(dat_pth, sample['abumix_file'])
      
      # more raster attributes
      pth = mkdirs(os.path.join(study.dat_pth, 'rasters', self.sid))
      self.coloc_file = os.path.join(pth, self.sid + '_coloc.npz')  
      self.nndist_file = os.path.join(pth, self.sid + '_nndist.npz')  
      self.rhfunc_file = os.path.join(pth, self.sid + '_rhfunc.npz')  
      self.geordG_file = os.path.join(pth, self.sid + '_geordG.npz')  
      self.lmearr = []
      
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
          
      # pylandstats landscape object
      self.plsobj = []

    
  def getSpaceStats(self, analyses):
      """
      Calculates pixel resolution statistics in raster arrays:
          
      1- Colocalization index (spacial Morisita-Horn score):
         Symetric score between each pair of classes
         (*) M ~ 1 indicates the two classes are similarly distributed
         (*) M ~ 0 indicates the two classes are segregated

      2- Nearest Neighbor Distance index
         Bi-variate asymetric score between all classes ('ref' and 'test'). 
                        N = log10(D(ref, test)/D(ref, ref))
         (*) V > 0 indicates ref and test cells are segregated
         (*) V ~ 0 indicates ref and test cells are well mixed
         (*) V < 0 indicates ref cells are individually infiltrated
             (ref cells are closer to test cells than other ref cells)

      3- Ripley's H score
         Bi-variate asymetric version of the Ripley's H(r) function, evaluated 
         at r=subbw (subkernel scale) between all classes ('ref' and 'test').
         The value of this metric indicates how much larger (or shorter) is 
         the radious 'R' of a circle with a uniform point distribution and the 
         same point density as the evaluation kernel:
                              H = log10(R/r) 
         (*) H > 0 indicates clustering of 'test' cells around 'ref' cells
         (*) H ~ 0 indicates random mixing between 'test' and 'ref' cells
         (*) H < 0 indicates dispersion of 'test' cells around 'ref' cells

      4- Gets-Ord statistics (G* and HOT value)
         G* (Z statistic) and HOT for all classes, were:
         (*) HOT = +1 if the region is overpopulated (P < 0.05)
         (*) HOT =  0 if the region is average (P > 0.05)
         (*) HOT = -1 if the region is underpopulated (P < 0.05)
         
         
      NOTE: the specific analyzes to be done are set in the `analyses` data
            frame passed in (this is an attribute of the Study object). This
            is done to have the option of downsizing the amount of results
            produced (and hence reduce number of calculations and increase
            running speeed)

      """
      
      from scipy.signal import fftconvolve
      from myfunctions import getis_ord_g_array, morisita_horn_array
      from myfunctions import nndist_array, circle
      from myfunctions import ripleys_K_array, ripleys_K_array_biv
      
      f = self.coord_file
      if not os.path.exists(f):
          print("ERROR: data file " + f + " does not exist!")
          sys.exit()
      data = pd.read_csv(f)
      
      # analyses
      df = analyses.copy()
      iscoloc = ~df.loc[df['name'] == 'coloc']['drop'].values[0] and \
          not os.path.exists(self.coloc_file)
      isnndist = ~df.loc[df['name'] == 'nndist']['drop'].values[0] and \
          not os.path.exists(self.nndist_file)
      isrhfunc = ~df.loc[df['name'] == 'rhfunc']['drop'].values[0] and \
          not os.path.exists(self.rhfunc_file)
      isgeordG = ~df.loc[df['name'] == 'geordG']['drop'].values[0] and \
          not os.path.exists(self.geordG_file)
      
      # if not all analyses are dropped
      if (iscoloc or isnndist or isrhfunc or isgeordG):
          
          # lists of classes cases to be calculated (x is ref, y is target)
          cases_x = []
          cases_y = []
          # empty arrays for desider comparisons for each metric
          coloc_comps = []
          nndist_comps = []
          rhfunc_comps = []
          geordG_comps = []
          
          # get comparisons
          if iscoloc:
              # combinations of cases:
              coloc_comps = df.loc[df['name'] == 'coloc']['comps'].values[0]
              # Colocalization index (spacial Morisita-Horn score)
              colocarr = np.full((self.imshape[0], self.imshape[1], 
                                   len(coloc_comps)), np.nan)
              cases_x  = list(set(cases_x) | set([c[0] for c in coloc_comps]))
              cases_y  = list(set(cases_y) | set([c[1] for c in coloc_comps]))
          if isnndist:
              # combinations of cases:
              nndist_comps = df.loc[df['name'] == 'nndist']['comps'].values[0]
              # Nearest Neighbor Distance index
              nndistarr = np.full((self.imshape[0], self.imshape[1], 
                                    len(nndist_comps)), np.nan)
              cases_x  = list(set(cases_x) | set([c[0] for c in nndist_comps]))
              cases_y  = list(set(cases_y) | set([c[1] for c in nndist_comps]))
          if isrhfunc:
              # combinations of cases:
              rhfunc_comps = df.loc[df['name'] == 'rhfunc']['comps'].values[0]
              # Ripley's H score
              rhfuncarr = np.full((self.imshape[0], self.imshape[1], 
                                    len(rhfunc_comps)), np.nan)
              cases_x  = list(set(cases_x) | set([c[0] for c in rhfunc_comps]))
              cases_y  = list(set(cases_y) | set([c[1] for c in rhfunc_comps]))
          if isgeordG:
              # combinations of cases:
              geordG_comps = df.loc[df['name'] == 'geordG']['comps'].values[0]
              # Gets-Ord statistics (G* and HOT value)
              geordGarr = np.full((self.imshape[0], self.imshape[1], 
                                    len(geordG_comps)), np.nan)
              hotarr = np.full((self.imshape[0], self.imshape[1], 
                                 len(geordG_comps)), np.nan)
              cases_x  = list(set(cases_x) | set(geordG_comps))
          
          # list classes with non-zero abundance in this sample  
          valid_cases = self.classes.loc[self.classes['number_of_cells'] > 0, 
                                         'class'].to_list()
          
          cases_x  = list(set(cases_x) & set(valid_cases))
          cases_y  = list(set(cases_y) & set(valid_cases))
          
          # reduced list of all class cases
          cases = list(set(cases_x) | set(cases_y))  
          
          # indeces of ref and target list cases in overal list
          cases_xi  = [cases.index(c) for c in cases_x]
          cases_yj  = [cases.index(c) for c in cases_y]
          
          # cell locations and abundance in kernels and subkernels
          X = np.zeros((self.imshape[0], self.imshape[1], len(cases)))
          N = np.zeros((self.imshape[0], self.imshape[1], len(cases)))
          n = np.zeros((self.imshape[0], self.imshape[1], len(cases)))
          
          # produces a box-circle kernel
          circ = circle(self.kernel)
          subcirc = circle(self.subkernel)
          
          # precalculate convolutions for all required classes
          for i, case in enumerate(cases):
              # coordinates of cells in class case
              aux = data.loc[data['class'] == case]
              X[aux.row, aux.col, i] = 1
              
              N[:, :, i] = np.abs(np.rint(fftconvolve(X[:, :, i],
                                                      circ, mode='same')))
              n[:, :, i] = np.abs(np.rint(fftconvolve(X[:, :, i],
                                                      subcirc, mode='same')))
          
          # loop thru all combinations of classes (pair-wise comparisons)
          for i, case_x in enumerate(cases_x):
              
              # index of ref case in all cases list
              xi = cases_xi[i]
              # coordinates of cells in class x (ref)
              aux = data.loc[data['class'] == case_x]
              rcx = np.array(aux[['row', 'col']])
              
              # Getis-Ord stats on smooth abundance profile for this class
              if (case_x in geordG_comps):
                  G = getis_ord_g_array(N[:, :, xi], self.roiarr)
                  [geordGarr[:, :, geordG_comps.index(case_x)], 
                   hotarr[:, :, geordG_comps.index(case_x)]] = G
              
              for j, case_y in enumerate(cases_y):
                  
                  # index of target case in all cases list
                  yj = cases_yj[j]
                  # coordinates of cells in class y (target)
                  aux = data.loc[data['class'] == case_y]
                  rcy = np.array(aux[['row', 'col']])
                  
                  # comparison tuple
                  comp = (case_x, case_y) 
                  
                  if (comp in coloc_comps):
                      if (comp[0] != comp[1]):
                          # Morisita Index (colocalization score)
                          M = morisita_horn_array(n[:, :, xi], 
                                                  n[:, :, yj], 
                                                  circ)
                      else:
                          # Morisita Index (identity)
                          M = 1.0*(n[:, :, xi] > 0)
                      M[self.roiarr == 0] = np.nan
                      colocarr[:, :, coloc_comps.index(comp)] = M
                    
                  if (comp in nndist_comps):
                      if (comp[0] != comp[1]):
                          # Nearest Neighbor Distance index (bivariate)
                          D = nndist_array(rcx, rcy, N[:, :, xi], circ)
                      else:
                          # Nearest Neighbor Distance index (identity)
                          D = 1.0*(N[:, :, xi] > 0)
                      D[self.roiarr == 0] = np.nan
                      nndistarr[:, :, nndist_comps.index(comp)] = D
                  
                  if (comp in rhfunc_comps):
                      if (comp[0] != comp[1]):
                          # Ripleys H score (bivarite)
                          K = ripleys_K_array_biv(rcx, 
                                                  n[:, :, xi], N[:, :, xi],
                                                  n[:, :, yj], N[:, :, yj], 
                                                  circ) 
                      else:
                          # Ripleys H score (identity)
                          K = ripleys_K_array(rcx, 
                                              n[:, :, xi], N[:, :, xi], 
                                              circ)
                      
                      K[self.roiarr == 0] = np.nan    
                      K[K == 0] = np.nan    
                      # Old definition:
                      # H = (np.sqrt(K/np.pi) - self.subkernel)/self.subkernel
                      # New definition, should be easier to interpret
                      H = np.log10(np.sqrt(K/np.pi)/self.subkernel)
                      rhfuncarr[:, :, rhfunc_comps.index(comp)] = H              
    
          if iscoloc:
              # saves coloc cases:
              np.savez_compressed(self.coloc_file, coloc=colocarr)
              del colocarr
              
          if isnndist:
              # saves nndist cases:
              np.savez_compressed(self.nndist_file, nndist=nndistarr)
              del nndistarr
              
          if isrhfunc:
              # saves rhfunc cases:
              np.savez_compressed(self.rhfunc_file, rhfunc=rhfuncarr)
              del rhfuncarr
              
          if isgeordG:
              # saves geordG cases:
              np.savez_compressed(self.geordG_file, 
                                  geordG=geordGarr,
                                  hot=hotarr)
              del geordGarr
              del hotarr
 
          gc.collect()
          
          
  def loadLMELandscape(self):
      """
      Produces raster array with LME code values. 
      This is based on a KDE raster, which is a smoothed cell concentration 
      image (with a certain bandwidth), of the original landscape of cell 
      locations, and a smooth raster of cell mixing values calculated in 
      the pre-processing module.
      For each point in the array, the local abundance of cells of each class
      and the mixing value level are used to assign an LME code according to
      study-level edges estimated in pre-processing.

      This method generates a landscape of LME categories with pixel-resolution
      Values of '0' in the array indicates elements outside of the ROI

      """

      nedges = self.classes['abundance_edges']
      medges = self.classes['mixing_edges']

      dim = len(self.classes)
      lmearr = np.zeros(self.imshape)
      
      f = self.abumix_file
      if not os.path.exists(f):
          print("ERROR: raster file " + f + " does not exist!")
          sys.exit()
      aux = np.load(f)
      abuarr = aux['abu'] # local abundances of each class
      mixarr = aux['mix'] # local mixing values of each class

      # vectorized function for faster processing
      def indexvalue(x, edges):
          o = max([0] + [j for j, v in enumerate(edges[:-1]) if v < x])
          return(o)
      vindexvalue = np.vectorize(indexvalue, excluded=['edges'])

      for i, iclass in self.classes.iterrows():
          # if edges are not empty
          if (len(nedges[i])>0):
              # get abundance level
              aux = abuarr[:, :, i]
              aux[np.isnan(aux)] = 0
              abu = vindexvalue(x=aux, edges=nedges[i])
              # get mixing level
              aux = mixarr[:, :, i]
              aux[np.isnan(aux)] = 0
              mix = vindexvalue(x=aux, edges=medges[i])
              # produces a single digital (dim-digit) code
              j = dim - (i + 1)
              lmearr = lmearr + (10**(2*j + 1))*abu + (10**(2*j))*mix

      # sets out-regions to NAN
      lmearr[self.roiarr == 0] = np.nan

      # reduces the number of lme classes by grouping them
      self.lmearr = lmeRename(lmearr, dim)
      
      del abuarr
      del mixarr
      gc.collect()
      
      
  def plotLMELandscape(self, out_pth):
      """
      Plot LME landscape

      """
            
      from myfunctions import plotRGB, plotEdges
      from matplotlib.cm import get_cmap
      import warnings

      dim = len(self.classes)
      nlevs = 3**dim
      raster = self.lmearr
      
      lme_code = ''
      for i in np.arange(dim):
          lme_code = lme_code + self.classes['class'][i]
      
      icticks = np.arange(nlevs) + 1
      cticks = [lmeCode(x, dim) for x in icticks]
      ctitle = 'LME Categories (' + lme_code + ')'
      cmap = get_cmap('jet', nlevs)
      
      [ar, redges, cedges,  xedges, yedges] = plotEdges(self.imshape, 
                                                        self.binsiz, 
                                                        self.scale)

      with warnings.catch_warnings():
          
          warnings.simplefilter('ignore')
          
          # plots sample image
          fig, ax = plt.subplots(1, 1,
                                 figsize=(12*1, 0.5 + math.ceil(12*1/ar)),
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
                       fontsize=16, y=1.02)
          #fig.subplots_adjust(hspace=0.4)
          fig.suptitle('Sample ID: ' + str(self.sid), fontsize=18, y=.95)
          #plt.tight_layout()
          fig.savefig(os.path.join(out_pth,
                                   self.sid +'_lme_landscape.png'),
                      bbox_inches='tight', dpi=300)
          plt.close()

          # the histogram of LME frequencies
          fig, ax = plt.subplots(1, 1, figsize=(12, 12),
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
      
      
  def plotColocLandscape(self, out_pth, analyses, lims, do_plots):
      """
      Plots colocalization index landscape from raster

      """
      # from itertools import combinations
      # # generates a list of comparisons for coloc
      # comps = [list(c) for c in list(combinations(self.classes.index, 2))]
      
      # if this analysis is already done, skip it
      fout = os.path.join(out_pth, self.sid +'_coloc_stats.csv')
      
      if not os.path.exists(fout):
      
          # analyses
          df = analyses.copy()
          iscoloc = ~df.loc[df['name'] == 'coloc']['drop'].values[0]
          
          if iscoloc:
              
              aux = np.load(self.coloc_file)
              colocarr = aux['coloc']
              
              # combinations of cases:
              coloc_comps = df.loc[df['name'] == 'coloc']['comps'].values[0]
              
              # plots array of landscapes for these comparisons
              [fig, metrics] = plotCompLandscape(self.sid,
                                                 colocarr, 
                                                 coloc_comps, 
                                                 self.imshape,
                                                 'Colocalization index', 
                                                 lims,
                                                 self.scale, 
                                                 self.units, 
                                                 self.binsiz,
                                                 do_plots)
              
              # saves landscape metrics table
              metrics.to_csv(fout, sep=',', index=False, header=True)
        
              if do_plots:  
                  # saves to png file
                  fig.savefig(os.path.join(out_pth,
                                           self.sid +'_coloc_landscape.png'),
                              bbox_inches='tight', dpi=300)
                  plt.close()
                  
                  # plot correlations between comparisons
                  ttl = 'Colocalization index Correlations\nSample ID: ' + \
                      str(self.sid)
                  fig = plotCompCorrelations(colocarr, 
                                             coloc_comps, 
                                             ttl, 
                                             lims)
                  plt.savefig(os.path.join(out_pth,
                                           self.sid +'_coloc_correlations.png'),
                              bbox_inches='tight', dpi=300)
                  plt.close()
                  
              del colocarr
              gc.collect() 
      
      
  def plotNNDistLandscape(self, out_pth, analyses, lims, do_plots):
      """
      Plots nearest neighbor distance index landscape from raster

      """
      # from itertools import permutations
      # # generate list of comparisons for coloc
      # comps = [list(c) for c in list(permutations(self.classes.index, r=2))]
      
      # if this analysis is already done, skip it
      fout = os.path.join(out_pth, self.sid +'_nndist_stats.csv')
      
      if not os.path.exists(fout):
      
          # analyses
          df = analyses.copy()
          isnndist = ~df.loc[df['name'] == 'nndist']['drop'].values[0]
          
          if isnndist:
              
              aux = np.load(self.nndist_file)
              nndistarr = aux['nndist']
              
              # combinations of cases:
              nndist_comps = df.loc[df['name'] == 'nndist']['comps'].values[0]
    
              # plots array of landscapes for these comparisons
              [fig, metrics] = plotCompLandscape(self.sid, 
                                                 nndistarr, 
                                                 nndist_comps, 
                                                 self.imshape,
                                                 'Nearest Neighbor Distance index',
                                                 lims,
                                                 self.scale,
                                                 self.units, 
                                                 self.binsiz,
                                                 do_plots)
              
              # saves landscape metrics table
              metrics.to_csv(fout, sep=',', index=False, header=True)
              
              if do_plots:
                  
                  # saves to png file
                  fig.savefig(os.path.join(out_pth,
                                           self.sid +'_nndist_landscape.png'),
                              bbox_inches='tight', dpi=300)
                  plt.close()
      
                  # plot correlations between comparisons
                  ttl = 'Nearest Neighbor Dist index Correlations\nSample ID: ' + \
                      str(self.sid)
                      
                  fig = plotCompCorrelations(nndistarr, 
                                             nndist_comps,
                                             ttl, 
                                             lims)
                  plt.savefig(os.path.join(out_pth, 
                                           self.sid +'_nndist_correlations.png'),
                              bbox_inches='tight', dpi=300)
                  plt.close()
                  
              del nndistarr
              gc.collect()     
          

  def plotRHFuncLandscape(self, out_pth, analyses, lims, do_plots):
      """
      Plots Ripley`s H function score landscape from raster

      """
      # from itertools import product
      # # generate list of comparisons for coloc
      # comps = [list(c) for c in list(product(self.classes.index, repeat=2))]
      
      # if this analysis is already done, skip it
      fout = os.path.join(out_pth, self.sid +'_rhfunc_stats.csv')
      
      if not os.path.exists(fout):
          
          # analyses
          df = analyses.copy()
          isrhfunc = ~df.loc[df['name'] == 'rhfunc']['drop'].values[0]
          
          if isrhfunc:
              
              aux = np.load(self.rhfunc_file)
              rhfuncarr = aux['rhfunc']
              
              # combinations of cases:
              rhfunc_comps = df.loc[df['name'] == 'rhfunc']['comps'].values[0]
    
              # plots array of landscapes for these comparisons
              [fig, metrics] = plotCompLandscape(self.sid,
                                                 rhfuncarr,
                                                 rhfunc_comps, 
                                                 self.imshape,
                                                 'Ripley`s H function score', 
                                                 lims,
                                                 self.scale, 
                                                 self.units, 
                                                 self.binsiz,
                                                 do_plots)
              
              # saves landscape metrics table
              metrics.to_csv(fout, sep=',', index=False, header=True)           
                     
              if do_plots:
                  
                  # saves to png file
                  fig.savefig(os.path.join(out_pth,
                                           self.sid +'_rhfunc_landscape.png'),
                              bbox_inches='tight', dpi=300)
                  plt.close()
                  
                  # plot correlations between comparisons
                  ttl = 'Ripley`s H function score Correlations\nSample ID: ' + \
                         str(self.sid)
                  fig = plotCompCorrelations(rhfuncarr, rhfunc_comps, 
                                             ttl, lims)
                  plt.savefig(os.path.join(out_pth,
                                           self.sid +'_rhfunc_correlations.png'),
                              bbox_inches='tight', dpi=300)
                  plt.close()
                  
              del rhfuncarr
              gc.collect()       
              
      
  def plotGeOrdLandscape(self, out_pth, analyses, lims, do_plots):
      """
      Plots Getis-Ord Z score landscape from raster

      """

      # generate list of comparisons
      # comps = list(self.classes.index)
      
      # if this analysis is already done, skip it
      fout = os.path.join(out_pth, self.sid +'_geordG_stats.csv')
      
      if not os.path.exists(fout):
      
          # analyses
          df = analyses.copy()
          isgeordG = ~df.loc[df['name'] == 'geordG']['drop'].values[0]
          
          if isgeordG:
              
              aux = np.load(self.geordG_file)
              geordGarr = aux['geordG']
              hotarr = aux['hot']
              
              # combinations of cases:
              geordG_comps = df.loc[df['name'] == 'geordG']['comps'].values[0]
          
              # plots array of landscapes for these comparisons
              [fig, metrics] = plotCaseLandscape(self.sid,
                                                 geordGarr,
                                                 geordG_comps,
                                                 self.imshape,
                                                 'Getis-Ord Z score', 
                                                 lims,
                                                 self.scale,
                                                 self.units, 
                                                 self.binsiz,
                                                 do_plots)
              
              # saves landscape metrics table
              metrics.to_csv(fout, sep=',', index=False, header=True) 
              
              if do_plots:
                  
                  # saves to png file
                  fig.savefig(os.path.join(out_pth,
                                           self.sid +'_geordG_landscape.png'),
                              bbox_inches='tight', dpi=300)
                  plt.close()
                  
                  # plots array of landscapes for these comparisons
                  fig = plotDiscreteLandscape(self.sid, 
                                              hotarr, 
                                              geordG_comps, 
                                              self.imshape,
                                              'HOT score', 
                                              [-1, 1],
                                              self.scale, 
                                              self.units, 
                                              self.binsiz)
                  
                  # saves to png file
                  fig.savefig(os.path.join(out_pth, 
                                           self.sid +'_hots_landscape.png'),
                              bbox_inches='tight', dpi=300)
                  plt.close()
                  
              del geordGarr
              del hotarr
              gc.collect()      
  

  def plotFactorCorrelations(self, out_pth, analyses, do_plots):

      # from itertools import combinations, product, permutations
      # # generates a list of comparisons for coloc
      # comps1 = [list(c) for c in list(combinations(self.classes.index, 2))]
      # # generate list of comparisons for rhfunc
      # comps2 = [list(c) for c in list(product(self.classes.index, repeat=2))]
      # # generate list of comparisons for nndist
      # comps3 = [list(c) for c in list(permutations(self.classes.index, r=2))]
      
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
        
      corr = pd.DataFrame()
      colocarr = []
      nndistarr = []
      rhfuncarr = []
      
      if iscoloc:
          aux = np.load(self.coloc_file)
          colocarr = aux['coloc']
      if isnndist:
          aux = np.load(self.nndist_file)
          nndistarr = aux['nndist']   
      if isrhfunc:
          aux = np.load(self.rhfunc_file)
          rhfuncarr = aux['rhfunc']
          
      vmin = 0.25*round(np.nanquantile(nndistarr, .001)/0.25)
      vmax = 0.25*round(np.nanquantile(nndistarr, .999)/0.25)
      wmin = 0.25*round(np.nanquantile(rhfuncarr, .001)/0.25)
      wmax = 0.25*round(np.nanquantile(rhfuncarr, .999)/0.25)    
    
    
      if iscoloc and isrhfunc:
                      
          # plot correlations between coloc and RHindex comparisons
          ttl = 'Coloc - RHindex Correlations\nSample ID: ' + str(self.sid)
          [fig, df] = plotFactorCorrelation(colocarr, 
                                            coloc_comps, 
                                            'Coloc', 
                                            [0.0, 1.0],
                                            rhfuncarr, 
                                            rhfunc_comps, 
                                            'RHindex', 
                                            [wmin, wmax],
                                            ttl, 
                                            self.classes)
          if do_plots:
              fig.savefig(os.path.join(out_pth,
                                       self.sid + '_coloc-rhindex_corr.png'),
                                bbox_inches='tight', dpi=300)
              plt.close()
              
          # concatenate correlation tables
          corr =  pd.concat([corr, df], ignore_index=True)
                
      if iscoloc and isnndist:
          
          # plot correlations between coloc and NNindex comparisons
          ttl = 'Coloc - NNdist Correlations\nSample ID: ' + str(self.sid)
          [fig, df] = plotFactorCorrelation(colocarr, 
                                             coloc_comps, 
                                            'coloc', 
                                            [0.0, 1.0],
                                            nndistarr, 
                                            nndist_comps, 
                                            'RHindex', 
                                            [vmin, vmax],
                                            ttl,
                                            self.classes)
          if do_plots:
              fig.savefig(os.path.join(out_pth,
                                       self.sid + '_coloc-nnindex_corr.png'),
                          bbox_inches='tight', dpi=300)
              plt.close()
              
          # concatenate correlation tables
          corr =  pd.concat([corr, df], ignore_index=True)
                
      if iscoloc and isnndist:
          
          # plot correlations between NNindex and RHindex comparisons
          ttl = 'RHindex - NNdist Correlations\nSample ID: ' + str(self.sid)
          [fig, df] = plotFactorCorrelation(rhfuncarr, 
                                            rhfunc_comps, 
                                            'RHindex', 
                                            [wmin, wmax],
                                            nndistarr, 
                                            nndist_comps, 
                                            'NNdist', 
                                            [vmin, vmax],
                                            ttl,
                                            self.classes)
          if do_plots:
              fig.savefig(os.path.join(out_pth, 
                                       self.sid +'_nnindex-rhindex_corr.png'),
                          bbox_inches='tight', dpi=300)
              plt.close()
              
          # concatenate correlation tables
          corr =  pd.concat([corr, df], ignore_index=True)
      
      # saves correlations table
      if (len(corr) > 0):
          corr.to_csv(os.path.join(out_pth,
                                   self.sid +'_factor_correlations.csv'), 
                      sep=',', index=False, header=True)
      
      del colocarr 
      del nndistarr 
      del rhfuncarr 
      gc.collect()
          

  def landscapeAnalysis(self, sample, samplcsv, do_plots):
      
      from myfunctions import mkdirs
      
      lme_pth = mkdirs(os.path.join(self.res_pth, 'lmes'))
      
      # sets background to 0 and create a pls object
      
      aux = self.lmearr.copy()
      aux[np.isnan(aux)] = 0
      
      # create pylandscape object and assing to landscape
      self.plsobj = pls.Landscape(aux.astype(int), 
                                  res=(1, 1),
                                  neighborhood_rule='8')

      # get adjacency matrix (as related to LME classes)
      adj_pairs = lmeAdjacency(self.plsobj, self.classes)
      adj_pairs.to_csv(os.path.join(lme_pth, 
                                    self.sid +'_lme_adjacency_odds.csv'),
                       sep=',', index=False, header=True)
      
      # get all patch metrics 
      patch_metrics = getPatchMetrics(self.plsobj, self.classes)
      patch_metrics.to_csv(os.path.join(lme_pth, 
                                        self.sid +'_lme_patch_metrics.csv'),
                           sep=',', index=False, header=True)
      # get all class metrics 
      class_metrics = getClassMetrics(self.plsobj, self.classes)
      class_metrics.to_csv(os.path.join(lme_pth, 
                                        self.sid +'_lme_class_metrics.csv'),
                           sep=',', index=False, header=True)
      # get all landscape-level metrics 
      landscape_metrics = getLandscapeMetrics(self.plsobj)
      landscape_metrics.to_csv(os.path.join(lme_pth, 
                                            self.sid + \
                                                '_lme_landscape_metrics.csv'),
                           sep=',', index=True, index_label="metric")
      
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
      sample_out = getSampleStats(sample, self.plsobj, adj_pairs,
                                  class_metrics, landscape_metrics)
      sample_out.to_csv(samplcsv, sep=',', index=False, header=True)
  
    
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
    
    from myfunctions import printProgressBar
    
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


def lmeRename(lmearr, dim):
    """
    Re-label LME classes according to a coarser definition in order to
    reduce the total number of LME classes

    Coarse LME categories are assigned as a decimal number derived from a
    base-3 numeric value, in which each base-3 bit is:
        * 0, "Bare" environment, with none or very few cells
        * 1, "Segmented" environment, cells are not uniformly distributed
        * 2, "Mixed" environment, where cells are mixed uniformly

    Parameters
    ----------
    - lmearr: (numpy) original lmearr (dim-digit code)
    - dim: (int) number of cell classes
    """

    def coarse(lmeval, dim):
        
        def get_digit(number, n):
            return number // 10**n % 10

        # coarses LME categories from lme array to decimal
        v = 0
        for d in range(dim):
            j = dim - (d + 1)
            n = get_digit(lmeval, 2*j + 1)
            m = get_digit(lmeval, 2*j)
            if n == 0:
                k = 0
            elif (m == 0 or (n == 1 and m == 1)):
                k = 1
            else:
                k = 2
            v = v + (3**j)*k
        return(v + 1)

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


def lmeAdjacency(ls, classes):
    """
    Gets LME points adjacency matrix

    Parameters
    ----------
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
    
    landscape_metrics['contagion']

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
    f.set(title='LME adjacenty odds F',
          xlabel=None, ylabel=None);
    f.invert_yaxis()
    f.set_aspect('equal')
    f.figure.savefig(os.path.join(res_pth, 
                                  sid +'_lme_adjacency_odds.png'),
                     bbox_inches='tight', dpi=300)
    plt.tight_layout()
    plt.close()

    return(0)


def plotCaseLandscape(sid, raster, comps, shape,
                      metric, lims, scale, units, binsiz, do_plots):
    """
    Plot of landscape from comparison raster

    Parameters
    ----------
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
    
    from myfunctions import plotRGB, plotEdges

    dim = len(comps)
    nlevs = 10
    bini = (lims[1] - lims[0])/(2*nlevs)
    vmax = np.nanquantile(raster, 0.975) + 2*bini
    cticks = np.arange(lims[0], lims[1] + 2*bini, 2*bini)
    bticks = np.arange(lims[0], lims[1] + bini, bini)
    bbticks = bticks[bticks <= vmax] 

    [ar, redges, cedges, xedges, yedges] = plotEdges(shape, binsiz, scale)
    
    comp_metrics = pd.DataFrame()

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
 
        # plots sample image
        fig, ax = plt.subplots(dim, 2,
                               figsize=(12*2, 0.5 + math.ceil(12*dim/ar)),
                               facecolor='w', edgecolor='k')

        for i, comp in enumerate(comps):

            aux = raster[:, :, i]
            
            # generates discrete values of the factor
            auy = aux.copy()
            auy[np.isnan(aux)] = 0
            inx = np.digitize(auy, bticks[:-1])
            auy = bticks[inx]
            auy[np.isnan(aux)] = np.nan
            cmap = plt.get_cmap('jet', len(bbticks))
            
            # create pylandscape object with binned factor values
            pl = pls.Landscape(inx, res=(1, 1), neighborhood_rule='8')
            
            # get all landscape-level metrics for (discrete) factor
            metrics = getLandscapeMetrics(pl)
            metrics = metrics.to_frame().transpose().reset_index(drop=True)
            metrics['class'] =  comp
            comp_metrics = pd.concat([comp_metrics, metrics],
                                     ignore_index=True)
            
            if do_plots:
                im = plotRGB(ax[i, 0], aux, units,
                             cedges, redges, xedges, yedges, fontsiz=18,
                             vmin=lims[0], vmax=lims[1], cmap=cmap)
    
                cbar = plt.colorbar(im, ax=ax[i, 0], fraction=0.046, pad=0.04)
                cbar.set_ticks(cticks)
                cbar.set_label(metric, rotation=90, labelpad=2)
                # ax[i, 0].set_title(comp, fontsize=18, y=1.02)
    
                freq, _ = np.histogram(aux[~np.isnan(aux)], 
                                       bins=bticks,
                                       density=False)
                vals = freq/np.sum(~np.isnan(aux))
                # vmax = np.max(np.append(vmax, vals))
                ax[i, 1].bar(bticks[:-1], vals,
                             width=bini, align='edge',
                             alpha=0.75, color='b', edgecolor='k')
    
                ax[i, 1].set_title(comp, fontsize=18, y=1.02)
                ax[i, 1].set_xlabel(metric)
                ax[i, 1].set_ylabel('Fraction of pixels')
                ax[i, 1].set_xlim(lims)
                ax[i, 1].set_xticks(cticks)
                # ax[i, 1].set_yscale('log')

        # for i in np.arange(len(comps)):
        #    ax[i, 1].set_ylim([0, 1.05*vmax])

        fig.suptitle('Sample ID: ' + str(sid), fontsize=24, y=.95)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # shift column 'class' to first position
        comp_metrics.insert(0, 'class', comp_metrics.pop('class'))

    return([fig, comp_metrics])


def plotDiscreteLandscape(sid, raster, comps, shape,
                          metric, lims, scale, units, binsiz):
    """
    Plot of discrete landscape from comparison raster

    Parameters
    ----------
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
    
    from myfunctions import plotRGB, plotEdges

    dim = len(comps)
    cticks  = np.unique(raster[~np.isnan(raster)])

    [ar, redges, cedges, xedges, yedges] = plotEdges(shape, binsiz, scale)
    
    def countsin(x, bins):
        return([len(x[x==b]) for b in bins])
    
    cmap = plt.get_cmap('jet', len(cticks))

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        # plots sample image
        fig, ax = plt.subplots(dim, 2,
                               figsize=(12*2, 0.5 + math.ceil(12*dim/ar)),
                               facecolor='w', edgecolor='k')

        for i, comp in enumerate(comps):

            aux = raster[:, :, i]
            
            im = plotRGB(ax[i, 0], aux, units,
                         cedges, redges, xedges, yedges, fontsiz=18,
                         vmin=lims[0], vmax=lims[1], cmap=cmap)

            cbar = plt.colorbar(im, ax=ax[i, 0], fraction=0.046, pad=0.04)
            cbar.set_ticks(cticks)
            cbar.set_label(metric, rotation=90, labelpad=2)
            # ax[i, 0].set_title(comp, fontsize=18, y=1.02)

            freq = countsin(aux[~np.isnan(aux)], cticks)
            vals = freq/np.sum(~np.isnan(aux))
            # vmax = np.max(np.append(vmax, vals))
            ax[i, 1].bar(cticks, vals,
                         align='center',
                         alpha=0.75, color='b', edgecolor='k')

            ax[i, 1].set_title(comp, fontsize=18, y=1.02)
            ax[i, 1].set_xlabel(metric)
            ax[i, 1].set_ylabel('Fraction of pixels')
            #ax[i, 1].set_xlim(lims)
            ax[i, 1].set_xticks(cticks)
            # ax[i, 1].set_yscale('log')

        # for i in np.arange(len(comps)):
        #    ax[i, 1].set_ylim([0, 1.05*vmax])

        fig.suptitle('Sample ID: ' + str(sid), fontsize=24, y=.95)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        

    return(fig)


def plotCompLandscape(sid, raster, comps, shape,
                      metric, lims, scale, units, binsiz, do_plots):
    """
    Plot of landscape from comparison raster

    Parameters
    ----------
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
    
    from myfunctions import plotRGB, plotEdges

    dim = len(comps)
    
    nlevs = 20
    bini = (lims[1] - lims[0])/(nlevs)
    cticks = np.arange(lims[0], lims[1] + 2*bini, 2*bini)
    bticks = np.arange(lims[0], lims[1] + bini, bini)
    
    [ar, redges, cedges, xedges, yedges] = plotEdges(shape, binsiz, scale)
    
    vmax = np.nanquantile(raster, 0.99) + 2*bini
    bbticks = bticks[bticks <= vmax] 
    
    cmap = plt.get_cmap('jet', len(bbticks))

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        
        comp_metrics = pd.DataFrame()

        # plots sample image
        fig, ax = plt.subplots(dim, 2,
                               figsize=(12*2, 0.5 + math.ceil(12*dim/ar)),
                               facecolor='w', edgecolor='k')

        for i, comp in enumerate(comps):

            aux = raster[:, :, i]
            
            # generates discrete values of the factor
            auy = aux.copy()
            auy[np.isnan(aux)] = 0
            inx = np.digitize(auy, bticks[:-1])
            auy = bticks[inx]
            auy[np.isnan(aux)] = np.nan
            
            # create pylandscape object with binned factor values
            pl = pls.Landscape(inx, res=(1, 1), neighborhood_rule='8')
            
            # get all landscape-level metrics for (discrete) factor
            metrics = getLandscapeMetrics(pl)
            metrics = metrics.to_frame().transpose().reset_index(drop=True)
            metrics['comp'] =  comp[0] + '::' + comp[1]
            comp_metrics = pd.concat([comp_metrics, metrics],
                                     ignore_index=True)

            if do_plots:
                im = plotRGB(ax[i, 0], auy, units,
                             cedges, redges, xedges, yedges, fontsiz=18,
                             vmin=lims[0], vmax=vmax, cmap=cmap)
    
                cbar = plt.colorbar(im, ax=ax[i, 0], fraction=0.046, pad=0.04)
                cbar.set_ticks(cticks[cticks <= vmax])
                cbar.set_label(metric, rotation=90, labelpad=2)
                #ax[i, 0].set_title(comp[0] + '::' + comp[1],
                #                   fontsize=18, y=1.02)
    
                freq, _ = np.histogram(aux[~np.isnan(aux)], 
                                       bins=bbticks,
                                       density=False)
                vals = freq/np.sum(~np.isnan(aux))
                ax[i, 1].bar(bbticks[:-1], vals,
                             width=bini, align='edge',
                             alpha=0.75, 
                             color=cmap(range(len(bbticks[:-1]))),
                             edgecolor='k')
    
                ax[i, 1].set_title(comp[0] + '::' + comp[1],
                                   fontsize=18, y=1.02)
                ax[i, 1].set_xlabel(metric)
                ax[i, 1].set_ylabel('Fraction of pixels')
                ax[i, 1].set_xlim([lims[0], vmax])
                ax[i, 1].set_xticks(cticks[cticks <= vmax])
                # ax[i, 1].set_yscale('log')

        fig.suptitle('Sample ID: ' + str(sid), fontsize=24, y=.95)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # shift column 'comp' to first position
        comp_metrics.insert(0, 'comp', comp_metrics.pop('comp'))
        
    return([fig, comp_metrics] )



def plotCompCorrelations(raster, comps, ttl, lims):

    import seaborn as sns
    import scipy.stats as sts

    nc = len(comps)*(len(comps)-1)/2
    ncols = int(np.ceil(np.sqrt(nc)))

    fig, ax = plt.subplots(int(np.ceil(nc/ncols)), ncols,
                           figsize=(ncols*12, (nc/ncols)*12),
                           facecolor='w', edgecolor='k')
    shp = ax.shape
    k = 0
    for i, compi in enumerate(comps):
        auxi = raster[:, :, i].ravel()
        for j, compj in enumerate(comps):
            if j > i:
                ij = np.unravel_index(k, shp)
                auxj = raster[:, :, j].ravel()

                aux = np.stack((auxi, auxj)).T
                aux = aux[~np.isnan(aux).any(axis=1)]
                aux = aux[np.random.randint(len(aux),
                                            size=min(1000, len(aux))), :]

                xlab = '(' + compi[0] + ':' + compi[1] + ')'
                ylab = '(' + compj[0] + ':' + compj[1] + ')'

                sns.axes_style("whitegrid")
                sns.regplot(x=aux[:, 0], y=aux[:, 1],
                            ax=ax[np.unravel_index(k, shp)],
                            scatter_kws={"color": "black"},
                            line_kws={"color": "red"})

                ax[ij].set_xlabel(xlab)
                ax[ij].set_ylabel(ylab)
                ax[ij].plot(np.linspace(lims[0], lims[1], 10),
                            np.linspace(lims[0], lims[1], 10),
                            color='k', linestyle='dashed')
                ax[ij].set_xlim(lims[0], lims[1])
                ax[ij].set_ylim(lims[0], lims[1])

                correlation, p_value = sts.pearsonr(aux[:, 0], aux[:, 1])
                star = 'NS'
                if (p_value < 0.05):
                    star = '*'
                if (p_value < 0.01):
                    star = '**'
                if (p_value < 0.001):
                    star = '***'
                coefs = 'C = %.4f; p-value = %.2e' % (correlation, p_value)
                ax[ij].set_title(coefs + ' (' + star + ')')
                k = k+1

    fig.subplots_adjust(hspace=0.4)
    fig.suptitle(ttl, fontsize=24, y=.95)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    return(fig)


def plotPatchHists(sid, df, res_pth):

    lmes = pd.unique(df['LME']).tolist()
    lab = ['LME class: ' + i for i in lmes]

    # plot some patch metrics distributions
    fig, axs = plt.subplots(2, 2, figsize=(24, 24),
                            facecolor='w', edgecolor='k')

    col = [mpl.cm.jet(x) for x in np.linspace(0, 1, len(lmes))]

    for i, lme in enumerate(lmes):

        mif = np.log10(np.min(df.area_fraction))
        maf = np.log10(np.max(df.area_fraction))
        dat = df.loc[df['LME'] == lme].area_fraction.apply(np.log10)
        hist, bins = np.histogram(dat,
                                  bins=np.arange(mif,
                                                 maf,
                                                 (maf-mif)/11),
                                  density=False)
        x = (bins[1:] + bins[:-1])/2
        axs[0, 0].plot(x, hist, label=lab[i], color=col[i])

        mir = np.log10(np.min(df.perimeter_area_ratio))
        mar = np.log10(np.max(df.perimeter_area_ratio))
        dat = df.loc[df['LME'] == lme].perimeter_area_ratio.apply(np.log10)
        hist, bins = np.histogram(dat,
                                  bins=np.arange(mir,
                                                 mar,
                                                 (mar-mir)/11),
                                  density=False)
        x = (bins[1:] + bins[:-1])/2
        axs[0, 1].plot(x, hist, label=lab[i], color=col[i])

        mis = np.min(df.shape_index)
        mas = np.max(df.shape_index)
        dat = df.loc[df['LME'] == lme].shape_index
        hist, bins = np.histogram(dat,
                                  bins=np.arange(mis,
                                                 mas,
                                                 (mas-mis)/11),
                                  density=False)

        x = (bins[1:] + bins[:-1]) / 2
        axs[1, 0].plot(x, hist, label=lab[i], color=col[i])

        mie = np.min(df.euclidean_nearest_neighbor)
        mae = np.max(df.euclidean_nearest_neighbor)
        dat = df.loc[df['LME'] == lme].euclidean_nearest_neighbor
        hist, bins = np.histogram(dat,
                                  bins=np.arange(mie,
                                                 mae,
                                                 (mae-mie)/11),
                                  density=False)
        x = (bins[1:] + bins[:-1])/2
        axs[1, 1].plot(x, hist, label=lab[i], color=col[i])

    axs[0, 0].set_xticks(np.arange(-10, 0, .5))
    axs[0, 0].set_xlim(mif, maf)
    axs[0, 0].get_xaxis().set_major_formatter(mpl.ticker.FuncFormatter(
        lambda x, p: r'$10^{%.1f}$' % x))
    #axs[0, 0].legend(loc='upper right')
    axs[0, 0].set_xlabel('Area fraction of patches')
    axs[0, 0].set_ylabel('Frequency')

    axs[0, 1].set_xticks(np.arange(-10, 10, .25))
    axs[0, 1].set_xlim(mir, mar)
    axs[0, 1].get_xaxis().set_major_formatter(mpl.ticker.FuncFormatter(
        lambda x, p: r'$10^{%.1f}$' % x))
    axs[0, 1].legend(loc='upper left', bbox_to_anchor=(1.01, 1))
    axs[0, 1].set_xlabel('Perimeter-Area ratio of patches (log)')
    axs[0, 1].set_ylabel('Frequency')

    axs[1, 0].set_xlim(mis, mas)
    # axs[1, 0].legend(loc='upper right')
    axs[1, 0].set_xlabel('Shape index of patches')
    axs[1, 0].set_ylabel('Frequency')

    axs[1, 1].set_xlim(mie, mae)
    # axs[1, 1].legend(loc='upper right')
    axs[1, 1].set_xlabel('Euclidean Nearest Neighbor of patches')
    axs[1, 1].set_ylabel('Frequency')

    fig.subplots_adjust(hspace=0.4)
    fig.suptitle('Sample ID: ' + str(sid), fontsize=24, y=.95)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(os.path.join(res_pth, sid +'_lme_patch_metrics_hists.png'),
                bbox_inches='tight', dpi=300)
    plt.close()

    return(fig)


def plotClassHists(sid, df, res_pth):
    # plot some patch metrics distributions

    lmes = pd.unique(df['LME']).tolist()
    cols = [mpl.cm.jet(x) for x in np.linspace(0, 1, len(lmes))]

    fig, axs = plt.subplots(2, 2, figsize=(24, 24),
                            facecolor='w', edgecolor='k')

    axs[0, 0].bar(lmes, df.patch_density, color=cols, align='center')
    axs[0, 0].set_title('Patch Density')
    axs[0, 0].set_xticks(lmes)
    axs[0, 0].set_xticklabels(lmes, rotation=90)
    axs[0, 1].bar(lmes, df.largest_patch_index, color=cols, align='center')
    axs[0, 1].set_title('Largest Patch Index')
    axs[0, 1].set_xticks(lmes)
    axs[0, 1].set_xticklabels(lmes, rotation=90)
    axs[1, 0].bar(lmes, df.edge_density, color=cols, align='center')
    axs[1, 0].set_title('Edge Density')
    axs[1, 0].set_xticks(lmes)
    axs[1, 0].set_xticklabels(lmes, rotation=90)
    axs[1, 1].bar(lmes, df.landscape_shape_index, color=cols, align='center')
    axs[1, 1].set_title('Landscape Shape Index')
    axs[1, 1].set_xticks(lmes)
    axs[1, 1].set_xticklabels(lmes, rotation=90)

    fig.subplots_adjust(hspace=0.4)
    fig.suptitle('Sample ID: ' + str(sid), fontsize=24, y=.95)
    fig.savefig(os.path.join(res_pth, sid +'_lme_class_metrics_hists.png'),
                bbox_inches='tight', dpi=300)
    plt.tight_layout()
    plt.close()

    # get class coverage graphs
    fig2, axs = plt.subplots(1, 2, figsize=(24, 12),
                             facecolor='w', edgecolor='k')

    y = df.total_area
    f = 100.*y/y.sum()
    patches, texts = axs[0].pie(y, colors=cols, startangle=10)
    axs[0].axis('equal')
    axs[0].set_title('Proportion of landscape')
    labs = ['{0} - {1:1.2f} %'.format(i, j) for i, j in zip(lmes, f)]
    axs[0].legend(patches, labs, loc='center right',
                  bbox_to_anchor=(0, 0.5), fontsize=8)

    y = df.number_of_patches
    f = 100.*y/y.sum()
    patches, texts = axs[1].pie(y, colors=cols, startangle=0)
    axs[1].axis('equal')
    axs[1].set_title('Number of Patches')
    labs = ['{0} - {1:1.2f} %'.format(i, j) for i, j in zip(lmes, f)]
    axs[1].legend(patches, labs, loc='center right',
                  bbox_to_anchor=(0, 0.5), fontsize=8)

    fig2.subplots_adjust(hspace=0.4)
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
                          ttl, classes):

    import seaborn as sns
    import scipy.stats as sts

    nc = len(comps1)*len(comps2)
    ncols = int(np.ceil(np.sqrt(nc)))

    fig, ax = plt.subplots(int(np.ceil(nc/ncols)), ncols,
                           figsize=(ncols*(12), (nc/ncols)*(12)),
                           facecolor='w', edgecolor='k')
    shp = ax.shape
    
    df = pd.DataFrame()

    k = 0
    for i, compi in enumerate(comps1):
        auxi = rasteri[:, :, i].ravel()
        for j, compj in enumerate(comps2):
            ij = np.unravel_index(k, shp)
            auxj = rasterj[:, :, j].ravel()

            aux = np.stack((auxi, auxj)).T
            aux = aux[~np.isnan(aux).any(axis=1)]
            aux = aux[np.random.randint(len(aux),
                                        size=min(1000, len(aux))), :]
            
            compi_lab = classes.iloc[compi[0]].class_name + \
                ':' + classes.iloc[compi[1]].class_name
                
            compj_lab = classes.iloc[compj[0]].class_name + \
                ':' + classes.iloc[compj[1]].class_name 

            xlab = tti + '(' + compi_lab + ')'
            ylab = ttj + '(' + compi_lab + ')'

            sns.axes_style("whitegrid")
            sns.regplot(x=aux[:, 0], y=aux[:, 1],
                        ax=ax[np.unravel_index(k, shp)],
                        scatter_kws={"color": "black"},
                        line_kws={"color": "red"})

            ax[ij].set_xlabel(xlab)
            ax[ij].set_ylabel(ylab)
            ax[ij].plot(np.linspace(limsi[0], limsi[1], 10),
                        np.linspace(limsj[0], limsj[1], 10),
                        color='k', linestyle='dashed')
            ax[ij].set_xlim(limsi[0], limsi[1])
            ax[ij].set_ylim(limsj[0], limsj[1])

            correlation, p_value = sts.pearsonr(aux[:, 0], aux[:, 1])
            star = 'NS'
            if (p_value < 0.05):
                star = '*'
            if (p_value < 0.01):
                star = '**'
            if (p_value < 0.001):
                star = '***'
            coefs = 'C = %.4f; p-value = %.2e' % (correlation, p_value)
            ax[ij].set_title(coefs + ' (' + star + ')')
            
            dfij = pd.DataFrame({'factor_i': [tti], 
                                 'comp_i': [compi_lab],
                                 'factor_j': [ttj],
                                 'comp_j': [compj_lab],
                                 'pearson_cor': [correlation],
                                 'p_value:': [p_value]})
            df = pd.concat([df, dfij], ignore_index=True)
            k = k + 1

    fig.subplots_adjust(hspace=0.4)
    fig.suptitle(ttl, fontsize=24, y=.95)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    return([fig, df])


    # %% Main function

def main(args):
    """
    *******  Main function  *******

    """
    # %% debug starts
    debug = True
    
    if debug:
        # running from the IDE
        # path of directory containing this script
        main_pth = os.path.dirname(os.getcwd())
        argsfile = os.path.join(main_pth, 'BE_set_1.csv')
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
        
    # NOTE: only ONE line in the argument table will be used
    study = Study( pd.read_csv(argsfile).iloc[0], main_pth)
    
    # %% STEP 1: creates data directories and new sample table
    # creates sample object and data folders for processed data
    sample = study.getStudy(CASE)
    
    # SID for display
    sid = sample.sample_ID
    
    msg = "====> Case [" + str(CASE + 1) + \
          "/" + str(len(study.samples.index)) + \
          "] :: SID <- " + sid 
    
    # output sample data filenames
    # landpkl = os.path.join(sample['res_pth'], sid +'_landscape.pkl')
    samplcsv = os.path.join(sample['res_pth'], sid +'_lme_tbl.csv')
    
    GO = False
    if (sample['num_cells'] > 10000):
        GO = (REDO or (not os.path.exists(samplcsv)))
    
    # if processed landscape do not exist
    if (GO):
        
        from myfunctions import mkdirs
        
        print( msg + " >>> processing..." )
    
        # %% STEP 1: loading data
        land = Landscape(sample, study)

        # %% STEP 2: calculate kernel-level space stats
        land.getSpaceStats(study.analyses)
        
        # %% STEP 3: prints space stats
        fact_pth = mkdirs(os.path.join(land.res_pth, 'space_factors'))
        land.plotColocLandscape(fact_pth, study.analyses, [0.0 ,1.0], GRPH)
        land.plotNNDistLandscape(fact_pth, study.analyses, [-1, 1], GRPH)
        land.plotRHFuncLandscape(fact_pth, study.analyses, [-1, 1], GRPH)
        land.plotGeOrdLandscape(fact_pth, study.analyses, [-3, 3], GRPH)
        land.plotFactorCorrelations(fact_pth, study.analyses, GRPH)
        
        # saves a proxy table to flag end of process (in case next step is 
        # commented and don't want to redo all analysis up to here)
        #sample.to_csv(samplcsv, sep=',', index=False, header=True)
        
        # %% STEP 4: assigns LME values to landscape
        land.loadLMELandscape()
        
        # % STEP 5: pylandstats analysis
        # Regular metrics can be computed at the patch, class and
        # landscape level. For list of all implemented metrics see: 
        # https://pylandstats.readthedocs.io/en/latest/landscape.html
        land.landscapeAnalysis(sample, samplcsv, GRPH)      
        
        # pickle results of quadrats analysis (for faster re-runs)
        # with open(landpkl, 'wb') as f:  
        #      pickle.dump([land], f)  
        # del land
            
    # else:
        # %% STEP 6: loads landscape data
        # with open(landpkl, 'rb') as f:  
        #    [land] = pickle.load(f) 
        
    
    # %% end
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
