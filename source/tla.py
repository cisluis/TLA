'''
    Tumor Landscape Analysis (TLA):
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
import matplotlib as mpl
import matplotlib.pyplot as plt
import pylandstats as pls
import pickle

from skimage import io
from PIL import Image
from ast import literal_eval

from argparse import ArgumentParser

from myfunctions import printProgressBar, plotRGB
from myfunctions import comp_correlations, factor_correlations

Image.MAX_IMAGE_PIXELS = 600000000

__version__  = "1.0.0"


# %% Private classes

class Study:
    
  def __init__(self, study, main_pth):
      
      # loads arguments for this study
      self.name = study['name']
      self.dat_pth = os.path.join(main_pth, study['path'], 'data')
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
          
      
  def addToSamplesOut(self, sample_out):
      
      aux = pd.concat([self.samples_out,
                       sample_out.to_frame().T],
                      ignore_index=True)
      
      self.samples_out = aux


class Landscape:
    
  def __init__(self, sample, study):
      
      dat_pth = study.dat_pth
      
      self.sid = sample.sample_ID
      self.res_pth = os.path.join(dat_pth, sample['results_dir'])
      
      aux = os.path.join(self.res_pth, self.sid +'_classes.csv')
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

    
  def getSpaceStats(self):
      """
      Calculates pixel resolution statistics in raster arrays:
          
      1- Colocalization index (spacial Morisita-Horn score):
         Symetric score between each pair of classes
         (*) M ~ 1 indicates the two classes are similarly distributed
         (*) M ~ 0 indicates the two classes are segregated

      2- Nearest Neighbor Distance index
         Bi-variate asymetric score between all classes ('ref' and 'test'). 
         (*) V > 0 indicates ref and test cells are segregated
         (*) V ~ 0 indicates ref and test cells are well mixed
         (*) V < 0 indicates ref cells are individually infiltrated
             (ref cells are closer to test cells than other ref cells)

      3- Ripley's H score
         Bi-variate asymetric version of the Ripley's H(r) function, evaluated 
         at r=subbw (subkernel scale) between all classes ('ref' and 'test').
         (*) H > 0 indicates clustering of 'test' cells around 'ref' cells
         (*) H ~ 0 indicates random mixing between 'test' and 'ref' cells
         (*) H < 0 indicates dispersion of 'test' cells around 'ref' cells

      4- Gets-Ord statistics (G* and HOT value)
         G* (Z statistic) and HOT for all classes, were:
         (*) HOT = +1 if the region is overpopulated (P < 0.05)
         (*) HOT = 0 if the region is average (P > 0.05)
         (*) HOT = -1 if the region is underpopulated (P < 0.05)

      """
      
      from scipy.signal import fftconvolve
      from myfunctions import getis_ord_g_array, morisita_horn_array
      from myfunctions import nndist_array
      from myfunctions import ripleys_K_array, ripleys_K_array_biv

      def circle(r):
          y, x = np.ogrid[-r: r + 1, -r: r + 1]
          return(1*(x**2 + y**2 < r**2))
      
      nc = len(self.classes)

      # raster array of cell locations and abundance in kernels and subkernels
      X = np.zeros((self.imshape[0], self.imshape[1], nc))
      N = np.zeros((self.imshape[0], self.imshape[1], nc))
      n = np.zeros((self.imshape[0], self.imshape[1], nc))

      # Colocalization index (spacial Morisita-Horn score)
      colocarr = np.empty((self.imshape[0], self.imshape[1], nc, nc))
      # Nearest Neighbor Distance index
      nndistarr = np.empty((self.imshape[0], self.imshape[1], nc, nc))
      # Ripley's H score
      rhfuncarr = np.empty((self.imshape[0], self.imshape[1], nc, nc))
      # Gets-Ord statistics (G* and HOT value)
      geordGarr = np.empty((self.imshape[0], self.imshape[1], nc))
      hotarr = np.empty((self.imshape[0], self.imshape[1], nc))
      
      # produces a box-circle kernel
      circ = circle(self.kernel)
      subcirc = circle(self.subkernel)
      data = self.cell_data
      
      # precalculate convolutions for all classes
      for i, clsx in self.classes.iterrows():

          # coordinates of cells in class x
          aux = data.loc[data['class'] == clsx['class']]
          X[aux.row, aux.col, i] = 1
          N[:, :, i] = np.abs(np.rint(fftconvolve(X[:, :, i],
                                                  circ, mode='same')))
          n[:, :, i] = np.abs(np.rint(fftconvolve(X[:, :, i],
                                                  subcirc, mode='same')))
      
      # loop thru all combinations of classes (pair-wise comparisons)
      for i, clsx in self.classes.iterrows():

          # coordinates of cells in class x
          aux = data.loc[data['class'] == clsx['class']]
          rcx = np.array(aux[['row', 'col']])

          # Nearest Neighbor Distance index (identity)
          nndistarr[:, :, i, i] = 1.0*(N[:, :, i] > 0)

          # Ripleys H score (identity)
          ripley = ripleys_K_array(rcx, n[:, :, i], N[:, :, i], circ)
          ripley = np.sqrt(ripley/np.pi) - self.subkernel
          ripley[self.roiarr == 0] = np.nan
          rhfuncarr[:, :, i, i] = ripley

          # Getis-Ord stats on smooth abundance profile for this class
          gog, hot = getis_ord_g_array(N[:, :, i], self.roiarr)
          [geordGarr[:, :, i],
           hotarr[:, :, i]] = getis_ord_g_array(N[:, :, i], self.roiarr)

          for j, clsy in self.classes.iterrows():

              # coordinates of cells in class y
              aux = data.loc[data['class'] == clsy['class']]
              rcy = np.array(aux[['row', 'col']])

              if (j > i):

                  # Morisita Index (colocalization score)
                  M = morisita_horn_array(n[:, :, i], n[:, :, j], circ)
                  M[self.roiarr == 0] = np.nan
                  colocarr[:, :, i, j] = M
                  colocarr[:, :, j, i] = M

              if (i != j):

                  # Ripleys H score (bivarite)
                  ripley = ripleys_K_array_biv(rcx, n[:, :, i], N[:, :, i],
                                               n[:, :, j], N[:, :, j], circ)
                  ripley = np.sqrt(ripley/np.pi) - self.subkernel
                  ripley[self.roiarr == 0] = np.nan
                  rhfuncarr[:, :, i, j] = ripley

                  aux = nndist_array(rcx, rcy, N[:, :, i], circ)

                  # Nearest Neighbor Distance index
                  nndistarr[:, :, i, j] = nndist_array(rcx, rcy,
                                                       N[:, :, i], circ)

      self.colocarr = colocarr
      self.nndistarr = nndistarr
      self.rhfuncarr = rhfuncarr
      self.geordGarr = geordGarr
      self.hotarr = hotarr
                                

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

      # defines a vectorizing function (for faster processing)
      def indexvalue(x, edges):
          o = max([0] + [j for j, v in enumerate(edges[:-1]) if v < x])
          return(o)
      vindexvalue = np.vectorize(indexvalue, excluded=['edges'])

      for i, iclass in self.classes.iterrows():
          # get abundance level
          aux = self.abuarr[:, :, i]
          aux[np.isnan(aux)] = 0
          abu = vindexvalue(x=aux, edges=nedges[i])
          # get mixing level
          aux = self.mixarr[:, :, i]
          aux[np.isnan(aux)] = 0
          mix = vindexvalue(x=aux, edges=medges[i])
          # produces a single digital (dim-digit) code
          j = dim - (i + 1)
          lmearr = lmearr + (10**(2*j + 1))*abu + (10**(2*j))*mix

      # sets out-regions to NAN
      lmearr[self.roiarr == 0] = np.nan

      # reduces the number of lme classes by grouping them
      lmearr = lmeRename(lmearr, dim)

      self.lmearr = lmearr
      
      
  def plotLMELandscape(self, out_pth):
      """
      Plot LME landscape

      """
            
      from matplotlib.cm import get_cmap
      import warnings

      dim = len(self.classes)
      nlevs = 3**dim
      lme = ''
      raster = self.lmearr
      
      for i in np.arange(dim):
          
          lme = lme + self.classes['class'][i]

          icticks = np.arange(nlevs)
          cticks = [lmeCode(x, dim) for x in np.arange(1, nlevs+1)]
          ctitle = 'LME Categories (' + lme + ')'
          cmap = get_cmap('jet', nlevs)

          [ar, 
           redges, cedges, 
           xedges, yedges] = plotEdges(self.imshape, self.binsiz, self.scale)

          with warnings.catch_warnings():
              
              warnings.simplefilter('ignore')

              # plots sample image
              fig, ax = plt.subplots(1, 1, 
                                     figsize=(12*1, 0.5 + math.ceil(12*1/ar)),
                                     facecolor='w', edgecolor='k')
              
              im = plotRGB(ax, raster, self.units,
                           cedges, redges, xedges, yedges, fontsiz=18,
                           vmin=-0.5, vmax=(nlevs - 0.5), cmap=cmap)
              cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
              cbar.set_ticks(icticks)
              cbar.set_ticklabels(cticks)
              cbar.ax.tick_params(labelsize=14)
              cbar.set_label(ctitle, size = 16, rotation=90, labelpad=1)
              ax.set_title('Local Micro-Environments (LME)', 
                           fontsize=16, y=1.02)
              #fig.subplots_adjust(hspace=0.4)
              fig.suptitle('Sample ID: ' + str(self.sid), 
                           fontsize=18, y=.95)
              plt.tight_layout()
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
              ax.set_xlim([-0.5, (nlevs - 0.5)])
              ax.set_xticks(icticks)
              ax.set_xticklabels(cticks, rotation=90)
              ax.set_yscale('log')
              plt.tight_layout()
              fig.savefig(os.path.join(out_pth, 
                                       self.sid +'_lme_distribution.png'),
                          bbox_inches='tight', dpi=300)
              plt.close()
      
      
  def plotColocLandscape(self, out_pth):
      """
      Plots colocalization index landscape from raster

      """
      from itertools import combinations

      # generates a list of comparisons for coloc
      comps = [list(c) for c in list(combinations(self.classes.index, 2))]

      # plots array of landscapes for these comparisons
      fig = plotCompLandscape(self.sid, 
                              self.colocarr, 
                              self.classes, 
                              comps, 
                              self.imshape,
                              'Colocalization index', 
                              [0.0, 1.0],
                              self.scale, 
                              self.units, 
                              self.binsiz)
      
      # saves to png file
      fig.savefig(os.path.join(out_pth, 
                               self.sid +'_coloc_landscape.png'),
                  bbox_inches='tight', dpi=300)
      plt.close()

      # plot correlations between comparisons
      ttl = 'Colocalization index Correlations\nSample ID: ' + str(self.sid)
      fig = comp_correlations(self.colocarr, 
                              comps, 
                              self.classes, 
                              ttl, 
                              [0, 1])
      plt.savefig(os.path.join(out_pth,
                               self.sid +'_coloc_correlations.png'),
                  bbox_inches='tight', dpi=300)
      plt.close()
      
      
  def plotNNDistLandscape(self, out_pth):
      """
      Plots nearest neighbor distance index landscape from raster

      """
      from itertools import permutations

      # generate list of comparisons for coloc
      comps = [list(c) for c in list(permutations(self.classes.index, r=2))]

      vmin = 0.25*round(np.nanquantile(self.nndistarr, .001)/0.25)
      vmax = 0.25*round(np.nanquantile(self.nndistarr, .999)/0.25)

      # plots array of landscapes for these comparisons
      fig = plotCompLandscape(self.sid, 
                              self.nndistarr, 
                              self.classes, 
                              comps, 
                              self.imshape,
                              'Nearest Neighbor Distance index',
                              [vmin, vmax],
                              self.scale, 
                              self.units, 
                              self.binsiz)
      
      # saves to png file
      fig.savefig(os.path.join(out_pth, 
                               self.sid +'_nndist_landscape.png'),
                  bbox_inches='tight', dpi=300)
      plt.close()

      # plot correlations between comparisons
      ttl = 'Nearest Neighbor Distance index Correlations\nSample ID: ' + \
          str(self.sid)
      fig = comp_correlations(self.nndistarr, 
                              comps, 
                              self.classes, 
                              ttl, 
                              [vmin, vmax])
      plt.savefig(os.path.join(out_pth, 
                               self.sid +'_nndist_correlations.png'),
                  bbox_inches='tight', dpi=300)
      plt.close()
      

  def plotRHFuncLandscape(self, out_pth):
      """
      Plots Ripley`s H function score landscape from raster

      """
      from itertools import product

      # generate list of comparisons for coloc
      comps = [list(c) for c in list(product(self.classes.index, repeat=2))]

      vmin = 0.25*round(np.nanquantile(self.rhfuncarr, .001)/0.25)
      vmax = 0.25*round(np.nanquantile(self.rhfuncarr, .999)/0.25)

      # plots array of landscapes for these comparisons
      fig = plotCompLandscape(self.sid, 
                              self.rhfuncarr, 
                              self.classes, 
                              comps, 
                              self.imshape,
                              'Ripley`s H function score', 
                              [vmin, vmax],
                              self.scale, 
                              self.units, 
                              self.binsiz)
      
      # saves to png file
      fig.savefig(os.path.join(out_pth,
                               self.sid +'_rhfscore_landscape.png'),
                  bbox_inches='tight', dpi=300)
      plt.close()

      # plot correlations between comparisons
      ttl = 'Ripley`s H function score Correlations\nSample ID: ' + \
          str(self.sid)
      fig = comp_correlations(self.rhfuncarr, 
                              comps, 
                              self.classes, 
                              ttl, 
                              [vmin, vmax])
      plt.savefig(os.path.join(out_pth,
                               self.sid +'_rhfscore_correlations.png'),
                  bbox_inches='tight', dpi=300)
      plt.close()
      
      
  def plotGOrdLandscape(self, out_pth):
      """
      Plots Getis-Ord Z score landscape from raster

      """

      # generate list of comparisons
      comps = list(self.classes.index)

      vmin = 0.25*round(np.nanquantile(self.geordGarr, .01)/0.25)
      vmax = 0.25*round(np.nanquantile(self.geordGarr, .99)/0.25)

      # plots array of landscapes for these comparisons
      fig = plotCaseLandscape(self.sid, 
                              self.geordGarr, 
                              self.classes, 
                              comps, 
                              self.imshape,
                              'Getis-Ord Z score', 
                              [vmin, vmax],
                              self.scale, 
                              self.units, 
                              self.binsiz)
      
      # saves to png file
      fig.savefig(os.path.join(out_pth,
                               self.sid +'_gozscore_landscape.png'),
                  bbox_inches='tight', dpi=300)
      plt.close()

      # plots array of landscapes for these comparisons
      fig = plotDiscreteLandscape(self.sid, 
                                  self.hotarr, 
                                  self.classes, 
                                  comps, 
                                  self.imshape,
                                  'HOT score', 
                                  [-1, 1],
                                  self.scale, 
                                  self.units, 
                                  self.binsiz)

      # saves to png file
      fig.savefig(os.path.join(out_pth, 
                               self.sid +'_hotscore_landscape.png'),
                  bbox_inches='tight', dpi=300)
      plt.close()
  

  def plotFactorCorrelations(self, out_pth):

      from itertools import combinations, product, permutations

      # generates a list of comparisons for coloc
      comps1 = [list(c) for c in list(combinations(self.classes.index, 2))]
      # generate list of comparisons for rhfunc
      comps2 = [list(c) for c in list(product(self.classes.index, repeat=2))]
      # generate list of comparisons for nndist
      comps3 = [list(c) for c in list(permutations(self.classes.index, r=2))]

      vmin = 0.25*round(np.nanquantile(self.nndistarr, .001)/0.25)
      vmax = 0.25*round(np.nanquantile(self.nndistarr, .999)/0.25)
      wmin = 0.25*round(np.nanquantile(self.rhfuncarr, .001)/0.25)
      wmax = 0.25*round(np.nanquantile(self.rhfuncarr, .999)/0.25)

      # plot correlations between coloc and RHindex comparisons
      ttl = 'Coloc - RHindex Correlations\nSample ID: ' + str(self.sid)
      fig = factor_correlations(self.colocarr, 
                                comps1, 
                                'Coloc', 
                                [0.0, 1.0],
                                self.rhfuncarr, 
                                comps2, 
                                'RHindex', 
                                [wmin, wmax],
                                ttl, 
                                self.classes)
      fig.savefig(os.path.join(out_pth,
                               self.sid +'_coloc-rhindex_correlations.png'),
                  bbox_inches='tight', dpi=300)
      plt.close()

      # plot correlations between coloc and NNindex comparisons
      ttl = 'Coloc - NNdist Correlations\nSample ID: ' + str(self.sid)
      fig = factor_correlations(self.colocarr, 
                                comps1, 
                                'coloc', 
                                [0.0, 1.0],
                                self.nndistarr, 
                                comps3, 
                                'RHindex', 
                                [vmin, vmax],
                                ttl,
                                self.classes)
      fig.savefig(os.path.join(out_pth, 
                               self.sid +'_coloc-nnindex_correlations.png'),
                  bbox_inches='tight', dpi=300)
      plt.close()

      # plot correlations between NNindex and RHindex comparisons
      ttl = 'RHindex - NNdist Correlations\nSample ID: ' + str(self.sid)
      fig = factor_correlations(self.rhfuncarr, 
                                comps2, 
                                'RHindex', 
                                [wmin, wmax],
                                self.nndistarr, 
                                comps3, 
                                'NNdist', 
                                [vmin, vmax],
                                ttl,
                                self.classes)
      fig.savefig(os.path.join(out_pth, 
                               self.sid +'_nnindex-rhindex_correlations.png'),
                  bbox_inches='tight', dpi=300)
      plt.close()
      

  def landscapeAnalysis(self, sample, lme_pth):
      
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
                       sep=',', index=False)
      lmeAdjacencyHeatmap(self.sid, adj_pairs, lme_pth)
      

      # get all patch metrics and plot histograms
      patch_metrics = getPatchMetrics(self.plsobj, self.classes)
      _ = plotPatchHists(self.sid, patch_metrics, lme_pth)
      # saves patch metrics table
      patch_metrics.to_csv(os.path.join(lme_pth, 
                                        self.sid +'_lme_patch_metrics.csv'),
                           sep=',', index=False)

      # get all class metrics and plot histograms
      class_metrics = getClassMetrics(self.plsobj, self.classes)
      _ = plotClassHists(self.sid, class_metrics, lme_pth)
      # saves class metrics table
      class_metrics.to_csv(os.path.join(lme_pth, 
                                        self.sid +'_lme_class_metrics.csv'),
                           sep=',', index=False)
      
      # get all landscape-level metrics and plot histograms
      landscape_metrics = getLandscapeMetrics(self.plsobj)
      # _ = plotLandscapeHists(sid, landscape_metrics, res_pth)
      # saves landscape metrics table
      landscape_metrics.to_csv(os.path.join(lme_pth, 
                                            self.sid + \
                                                '_lme_landscape_metrics.csv'),
                           sep=',', index=True, index_label="metric")

      # get sample stats
      return(getSampleStats(sample, self.plsobj, adj_pairs, 
                            class_metrics, landscape_metrics))
  
    
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

    for i in range(len(lmes)):
        newarr[lmearr == lmes[i]] = coarse(lmes[i], dim)

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

    return(sample_out)
     

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

            freq, _ = np.histogram(aux[~np.isnan(aux)], 
                                   bins=bticks,
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

        #fig.subplots_adjust(hspace=0.4)
        fig.suptitle('Sample ID: ' + str(sid),
                     fontsize=24, y=.95)
        plt.tight_layout()

    return(fig)


def plotDiscreteLandscape(sid, raster, classes, comps, shape,
                          metric, lims, scale, units, binsiz):
    """
    Plot of discrete landscape from comparison raster

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

            aux = raster[:, :, comp]
            
            im = plotRGB(ax[i, 0], aux, units,
                         cedges, redges, xedges, yedges, fontsiz=18,
                         vmin=lims[0], vmax=lims[1], cmap=cmap)

            cbar = plt.colorbar(im, ax=ax[i, 0], fraction=0.046, pad=0.04)
            cbar.set_ticks(cticks)
            cbar.set_label(metric, rotation=90, labelpad=2)
            ax[i, 0].set_title(classes.class_name[comp],
                               fontsize=18, y=1.02)

            freq = countsin(aux[~np.isnan(aux)], cticks)
            vals = freq/np.sum(~np.isnan(aux))
            # vmax = np.max(np.append(vmax, vals))
            ax[i, 1].bar(cticks, vals,
                         align='center',
                         alpha=0.75, color='b', edgecolor='k')

            ax[i, 1].set_title(classes.class_name[comp],
                               fontsize=18, y=1.02)
            ax[i, 1].set_xlabel(metric)
            ax[i, 1].set_ylabel('Fraction of pixels')
            #ax[i, 1].set_xlim(lims)
            ax[i, 1].set_xticks(cticks)
            # ax[i, 1].set_yscale('log')

        # for i in np.arange(len(comps)):
        #    ax[i, 1].set_ylim([0, 1.05*vmax])

        #fig.subplots_adjust(hspace=0.85)
        fig.suptitle('Sample ID: ' + str(sid),
                     fontsize=24, y=.95)
        plt.tight_layout()
        

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

        #fig.subplots_adjust(hspace=0.85)
        fig.suptitle('Sample ID: ' + str(sid),
                     fontsize=24, y=.95)
        plt.tight_layout()
        

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
    fig.savefig(os.path.join(res_pth, sid +'_lme_patch_metrics_hists.png'),
                bbox_inches='tight', dpi=300)
    plt.tight_layout()
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
    fig2.savefig(os.path.join(res_pth, sid +'_lme_class_coverage.png'),
                 bbox_inches='tight', dpi=300)
    plt.tight_layout()
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
    fig.savefig(fname, bbox_inches='tight', dpi=300)
    plt.tight_layout()
    plt.close()
         
    return(0)

    # %% Main function

def main(args):

    # %% debug start
    debug = False
    
    if debug:
        # running from the IDE
        # path of directory containing this script
        main_pth = os.path.dirname(os.getcwd())
        argsfile = os.path.join(main_pth, 'test_set.csv')
        REDO = False
    else:
        # running from the CLI using bash script
        # path of directory containing this script
        main_pth = os.getcwd()
        argsfile = os.path.join(main_pth, args.argsfile)
        REDO = args.redo

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

        # number of processing steps
        Nsteps = 7
        printProgressBar(Nsteps*(0), Nsteps*(numsamples), suffix='')

        print(st)
        sys.exit()        


        # %% loops over samples in this study
        for index, sample in study.samples.iterrows():
            
            # SID for display
            sid = sample.sample_ID
            
            # creates a message to display in progress bar
            msg = '==> ' + sid + "; [{0}/{1}]".format(index + 1, numsamples)
            
            landpkl = os.path.join(study.dat_pth, 
                                   sample['results_dir'],
                                   sid +'.pkl')
            
            # if processed landscape picle file do not exist
            if REDO or (not os.path.exists(landpkl)):
            
                # STEP 1: loading data
                progressBar(index, numsamples, 1, Nsteps, msg, 
                            'loading sample data...')
                land = Landscape(sample, study)
    
                # STEP 2: assigns LME values to landscape
                progressBar(index, numsamples, 2, Nsteps, msg, 
                            'creating LME landscape...')
                land.loadLMELandscape()
                
                # STEP 3: calculate kernel-level space stats
                progressBar(index, numsamples, 3, Nsteps, msg, 
                            'computing space statistics...')
                land.getSpaceStats()
                
                # pickle results of quadrats analysis (for faster re-runs)
                with open(landpkl, 'wb') as f:  
                    pickle.dump([land], f)  
            
            else:
                # STEP 4: loads pickled landscape data
                progressBar(index, numsamples, 4, Nsteps, msg, 
                            'loading landscape data...')
                with open(landpkl, 'rb') as f:  
                    [land] = pickle.load(f) 
                    
            # STEP 5: pylandstats analysis
            # Regular metrics can be computed at the patch, class and
            # landscape level. For list of all implemented metrics see: 
            # https://pylandstats.readthedocs.io/en/latest/landscape.html
            progressBar(index, numsamples, 5, Nsteps, msg, 
                        'running pylandstat analysis on LME patches...')
            
            lme_pth = os.path.join(land.res_pth, 'LME_Analysis')
            if not os.path.exists(lme_pth):
                os.makedirs(lme_pth)
            sample_out = land.landscapeAnalysis(sample, lme_pth)
            
            land.plotLMELandscape(lme_pth)
            
            # update sample table
            study.addToSamplesOut(sample_out)
                
            # STEP 6: prints LMEs and kernel stats
            progressBar(index, numsamples, 6, Nsteps, msg, 
                        'printing LMEs and factor index maps...')
            
            fact_pth = os.path.join(land.res_pth, 'Space_Factors')
            if not os.path.exists(fact_pth):
                os.makedirs(fact_pth)
                
            land.plotColocLandscape(fact_pth)
            land.plotNNDistLandscape(fact_pth)
            land.plotRHFuncLandscape(fact_pth)
            land.plotGOrdLandscape(fact_pth)
            land.plotFactorCorrelations(fact_pth)

        # %% End Steps
        progressBar(index, numsamples, 7, Nsteps, msg, 
                   'saving summary tables...')
        
        cols = ['sample_ID', 'total_area', 'ROI_area', 'num_cells',
                'f_num_cells', 'l_num_cells', 't_num_cells']
        tblname = os.path.join(study.dat_pth, 
                               'results',
                               study.name + '_samples_stats.csv')
        if not os.path.exists(tblname):
            print("ERROR: data file " + tblname + " does not exist!")
            sys.exit()
        study.samples_out= pd.merge(pd.read_csv(tblname)[cols], 
                                    study.samples_out, 
                                    on=['sample_ID', 'num_cells'])
        
        study.samples_out.to_csv(tblname, index=False)
        
        cols = ['num_cells', 'num_lmes', 'landscape_area', 
                'adjacency_index', 'lme_contagion',
                'lme_Shannon', 'lme_shape_index', 'lme_Simpson', 
                'lme_num_patches',
                'lme_patch_density', 'lme_total_edge', 'lme_edge_density',
                'lme_largest_patch_index']
        
        labs = ['Number of Cells', 'Number of LMEs', 'Landscape Area', 
                'LME Adjacency Index', 'LME Contagion',
                'LME Shannon Index', 'LME Shape Index', 'LME Simpson Index', 
                'LME Number of Patches', 'LME Patch Density', 
                'LME Total Edge', 'LME Edge Density', 
                'LME Largest Patch Index']
        
        for i, c in enumerate(cols):
            
            _= plotViolins(study.samples_out,
                           'cohort', "Cohort",
                           c, labs[i], 
                           os.path.join(study.dat_pth, 
                                        'results',
                                        study.name + '_' + cols[i] + '.png'))



    # %% end
    
    print("==> Analysis finished!!! ")
    
    return(0)        
        

# %% Argument parser
if __name__ == "__main__":

    # Create the parser
    my_parser = ArgumentParser(prog="tla",
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
