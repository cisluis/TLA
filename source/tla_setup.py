'''
    TLA setup:
    #################

        This script reads lines from a study set table
        Each line has parameters of a particular study
        Then (raw) data for each study is read and extreme values calculated
        Cell classes are checked and new, curated, coordinate file are produced
        along with a croped version of the IHC image and a raster arrays
        for each sample (at hoc region mask, a mask that defines the ROI, and
                         a multilevel raster array with kde cell density info).
        A sample data summary and general stats are also produced.

        This process prepares and formats data for TLA
'''

# %% Imports

import os
import sys
import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from skimage import io
from PIL import Image
from argparse import ArgumentParser

from myfunctions import printProgressBar

Image.MAX_IMAGE_PIXELS = 600000000

__version__  = "1.0.0"


# %% Private classes

class Study:
    
  def __init__(self, study, main_pth):
      
      # loads arguments for this study
      self.name = study['name']
      self.raw_path = os.path.join(main_pth, study['raw_path'])
      
      samples_file = os.path.join(self.raw_path, study['raw_samples_table'])
      if not os.path.exists(samples_file):
          print("ERROR: samples file " + samples_file + " does not exist!")
          sys.exit()
      self.samples = pd.read_csv(samples_file)
      self.samples.fillna('', inplace=True)
      
      self.dat_path = os.path.join(main_pth, study['data_path'])
      if not os.path.exists(self.dat_path):
          os.makedirs(self.dat_path)

      # the size of quadrats and subquadrats
      self.binsiz = int((study['binsiz']))
      self.subbinsiz = int(self.binsiz/5)

      # bandwidth size for convolutions is half the quadrat size
      self.supbw = int(self.binsiz/2)
      self.bw = int(self.subbinsiz/2)

      # scale parameters
      self.scale = study['scale']
      self.units = study['units']

      # Filter parameters:
      # (-) cell class to run filter on (e.g. tumor cells)
      self.FILTER_CODE = study['FILTER_CODE']
      # (-) code to assign to cells in high density regions
      self.HIGH_DENS_CODE = study['HIGH_DENS_CODE']
      # (-) code to assign to cells in low density regions
      self.LOW_DENS_CODE = study['LOW_DENS_CODE']
      # (-) threshold high/low density regions (0 for no filtering)
      self.DTHRES = study['DTHRES']
      # (-) if true, uses 'blob' label in data to mask cells
      self.BLOBS = study['BLOBS']

      # reduces classes df to just the accepted types (i.e. `drop=False`)
      classes_file = os.path.join(self.raw_path, study['raw_classes_table'])
      if not os.path.exists(classes_file):
          print("ERROR: classes file " + classes_file + " does not exist!")
          sys.exit()
      classes = pd.read_csv(classes_file)
      classes.drop(classes.loc[classes['drop']].index, inplace=True)
      classes.drop(columns=['drop'], inplace=True)
      self.classes = classes

      # creates samples table for output
      self.samples_out = pd.DataFrame()
      self.allstats_out = pd.DataFrame()
      self.allpops_out = pd.DataFrame()
      
      
  def add_sample(self, samp, stats, pop):
      
      self.samples_out = pd.concat([self.samples_out, 
                                    samp.to_frame().T], 
                                   ignore_index=True)
       
      self.allstats_out = pd.concat([self.allstats_out, 
                                     stats.to_frame().T],
                                    ignore_index=True)
      
      self.allpops_out = pd.concat([self.allpops_out, 
                                    pop], ignore_index=True)
      
      
  def output(self):
      
      self.samples_out = self.samples_out.astype({'num_cells': int})
      self.samples_out = self.samples_out.astype(self.samples.dtypes)
      self.samples_out.to_csv(os.path.join(self.dat_path, 
                                           self.name + '_samples.csv'),
                              index=False)
      self.allstats_out.to_csv(os.path.join(self.dat_path, 
                                            'results',
                                            self.name + '_samples_stats.csv'),
                               index=False)
      self.allpops_out.to_csv(os.path.join(self.dat_path,
                                           'results',
                                           self.name + '_quadrat_stats.csv'), 
                              index=False)
      
      # plots distributions of quadrat stats
      self.classes = quadFigs(self.allpops_out, self.classes,
                              os.path.join(self.dat_path,
                                           self.name + '_quadrat_stats.png'))
      self.classes.to_csv(os.path.join(self.dat_path, 
                                       self.name + '_classes.csv'), 
                          index=False)
      
      
class Sample:
    
  def __init__(self, samp, study):
      
      # creates sample object
      self.sid = samp.sample_ID
      
      # raw data files
      self.raw_cell_data_file = os.path.join(study.raw_path, 
                                             samp.coord_file)
      
      self.classes = study.classes    # cell classes
      self.tbl = samp.copy()          # table of parameters (for output)
      self.cell_data = pd.DataFrame() # dataframe of cell data (coordinates)
      self.imshape = []               # shape of accepted image
      self.img = []                   # image array
      self.msk = []                   # mask array (blobs)
      self.roiarr = []                # ROI array
      self.kdearr = []                # KDE array (smoothed cell density)
      self.abuarr = []                # abundance array (coarsed)
      self.mixarr = []                # mixing array (coarsed)
      
      # the size of quadrats and subquadrats
      self.binsiz = study.binsiz
      self.subbinsiz = study.subbinsiz

      # bandwidth size for convolutions is half the quadrat size
      self.supbw = study.supbw
      self.bw = study.bw
     
      # scale parameters
      self.scale = study.scale
      self.units = study.units
      
      # creates results folder and add path to sample
      outpth = os.path.join(study.dat_path, 'results')
      if not os.path.exists(outpth):
          os.makedirs(outpth)
      res_pth = os.path.join(outpth, self.sid)
      if not os.path.exists(res_pth):
          os.makedirs(res_pth)
      self.tbl['results_dir'] = 'results/' + self.sid + '/'
      self.res_pth = res_pth
      
      # creates cellPos folder and add path to df
      outpth = os.path.join(study.dat_path, 'cellPos')
      if not os.path.exists(outpth):
          os.makedirs(outpth)
      self.tbl['coord_file'] = 'cellPos/' + self.sid + '.csv'
      self.cell_data_file = os.path.join(outpth, self.sid + '.csv')

      # creates images folder and add path to df
      outpth = os.path.join(study.dat_path, 'images')
      if not os.path.exists(outpth):
          os.makedirs(outpth)
      if (samp.image_file ==''):
          self.tbl['image_file'] = ''
          self.imfile = ''  
          self.raw_imfile = ''
      else: 
          self.tbl['image_file'] = 'images/' + self.sid + '_img.jpg'
          self.imfile = os.path.join(outpth, self.sid + '_img.jpg')
          self.raw_imfile = os.path.join(study.raw_path, samp.image_file)

      # creates blob mask folder and add path to df
      outpth = os.path.join(study.dat_path, 'rasters')
      if not os.path.exists(outpth):
          os.makedirs(outpth)
      self.tbl['mask_file'] = 'rasters/' + self.sid + '_mask.npz'
      self.mkfile = os.path.join(outpth, self.sid + '_mask.npz')    
      if (samp.mask_file ==''):    
          self.raw_mkfile = ''
      else:
          self.raw_mkfile = os.path.join(study.raw_path, samp.mask_file)
          
      # creates raster folder and add path to df
      self.tbl['raster_file'] = 'rasters/' + self.sid + '_raster.npz'
      self.raster_file = os.path.join(outpth, self.sid + '_raster.npz')
      
      # additional info
      self.classes_file = os.path.join(self.res_pth, 
                                       self.sid +'_classes.csv')
      self.rcell = 0
      
     
  def setup_data(self, edge):
      """
      Loads coordinates data, shift and convert values.
      Crops images to the corresponding convex hull
      """
     
      from skimage.transform import resize
      
      if not os.path.exists(self.raw_cell_data_file):
          print("ERROR: data file " + self.raw_cell_data_file + \
                " does not exist!")
          sys.exit()
      cxy = pd.read_csv(self.raw_cell_data_file)[['class', 'x', 'y']]
      cxy = cxy.loc[cxy['class'].isin(self.classes['class'])]

      # gets extreme pixel values
      xmin, xmax = int(np.min(cxy.x)), int(np.max(cxy.x))
      ymin, ymax = int(np.min(cxy.y)), int(np.max(cxy.y))

      imshape = [np.nan, np.nan]

      # reads image file (if exists)
      imfile_exist = os.path.exists(self.raw_imfile)
      if imfile_exist:
          ims = io.imread(self.raw_imfile)
          imshape = ims.shape

      # reads mask file (if exists)
      mkfile_exist = os.path.exists(self.raw_mkfile)
      if mkfile_exist:
          msc = io.imread(self.raw_mkfile)
          imshape = msc.shape

      # check for consistency in image and mask
      if ((imfile_exist and mkfile_exist) and
          ((ims.shape[0] != msc.shape[0]) or
           (ims.shape[1] != msc.shape[1]))):
          #print('\n =====> WARNING! sample_ID: ' + self.sid +
          #      '; image and mask are NOT of the same size, ' +
          #      'thus adopting mask field domain...')
          ims = np.rint(resize(ims, (msc.shape[0], msc.shape[1], 3),
                               anti_aliasing=True, 
                               preserve_range=True)).astype('uint8')
          
          #rspan = np.min([ims.shape[0], msc.shape[0]]) - 1
          #cspan = np.min([ims.shape[1], msc.shape[1]]) - 1
          #aux = np.zeros((imshape[0], imshape[1], 3))
          #aux[0:rspan, 0:cspan, :] = ims[0:rspan, 0:cspan, :]
          #ims = aux.copy()

      # limits for image cropping
      rmin = np.nanmax([0, ymin - edge])
      cmin = np.nanmax([0, xmin - edge])
      rmax = np.nanmin([imshape[0] - 1, ymax + edge])
      cmax = np.nanmin([imshape[1] - 1, xmax + edge])
      imshape = [int(rmax - rmin + 1), int(cmax - cmin + 1)]

      # shifts coordinates
      cell_data = xyShift(cxy, imshape, [rmin, cmin], self.scale)

      # create croped versions of image and mask raster
      img = np.zeros((imshape[0], imshape[1], 3))
      if imfile_exist:
          img[0:(rmax - rmin), 
              0:(cmax - cmin), :] = ims[rmin:rmax, cmin:cmax, :]
      img = img.astype('uint8')
      
      msk = np.zeros(imshape)
      if mkfile_exist:
          msk[0:(rmax - rmin), 
              0:(cmax - cmin)] = msc[rmin:rmax, cmin:cmax]
      [cell_data, msk_img] = getBlobs(cell_data, msk)
      
      self.cell_data = cell_data
      self.imshape = imshape
      self.img = img
      self.msk = msk_img
      
      
  def save_data(self):
      
      # saves main data files
      self.cell_data.to_csv(self.cell_data_file, index=False)
      self.classes.to_csv(self.classes_file, index=False)
      
      if (self.imfile != ''):
          io.imsave(self.imfile, self.img, check_contrast=False)
      np.savez_compressed(self.mkfile, mask=self.msk )
          
      np.savez_compressed(self.raster_file,
                          roi=self.roiarr,
                          kde=self.kdearr,
                          abu=self.abuarr,
                          mix=self.mixarr)
      
  def load_data(self):
      
      # load all data (if it was previously created)
      self.cell_data = pd.read_csv(self.cell_data_file)
      self.classes = pd.read_csv(self.classes_file)
      
      aux = np.load(self.mkfile)
      self.msk = aux['mask']
      
      aux = np.load(self.raster_file)
      self.roiarr = aux['roi']
      self.kdearr = aux['kde']
      self.abuarr = aux['abu']
      self.mixarr = aux['mix']
      
      self.imshape = [self.roiarr.shape[0], self.roiarr.shape[1]]
      
      if (self.imfile != ''):
          self.img = io.imread(self.imfile)
      else:    
          self.img = np.zeros((self.imshape[0], self.imshape[1], 3))
      
      
  def filter_class(self, study):
      """
      Applies a class filter. Returns filtered copy of cell_data and mask

      Parameters
      ----------
      - target_code: (str) cell class to run filter on (e.g. tumor cells)
      - hd_code: (str) code to assign to cells in high density regions
      - ld_code: (str) code to assign to cells in low density regions
      - denthr: (float) threshold to distinguish high and low density regions.
                make denthr=0 to have no filtering
      - blobs: (bool) if true, uses 'blob' label in data to mask cells. In this
               case cells outside of blobs are reassigned the ld_code and those
               inside are assigned the hd_code. If denthr>0 then cells in low
               denisity regions inside the blobs are assigned the hd_code, BUT
               all cells outside the blobs are assigned the hd_code.

      """
      from myfunctions import KDE, arrayLevelMask
      
      target_code = study.FILTER_CODE
      hd_code =  study.HIGH_DENS_CODE
      ld_code = study.LOW_DENS_CODE
      denthr = study.DTHRES
      blobs = study.BLOBS

      data = self.cell_data.copy()
      data['orig_class'] = data['class']

      # redefine all target cells
      data.loc[(data['class'] == target_code), 'class'] = hd_code
      if blobs:
          # redefine possibly misclasified target cells by external mask filter
          # (those inside mask blobs)
          data.loc[(data['orig_class'] == target_code) &
                   (data['blob'] > 0), 'class'] = hd_code

          # (those outside mask blobs)
          data.loc[(data['orig_class'] == target_code) &
                   (data['blob'] == 0), 'class'] = ld_code
          
      if (denthr > 0.0):
          
          data['i'] = data.index

          # subset data to just target cells, gets array of values
          aux = data.loc[data['class'] == hd_code]
          irc = np.array(aux[['i', 'row', 'col']])

          # do KDE on pixel locations of target cells (e.g. tumor cells)
          [_, _, z] = KDE(aux, self.imshape, self.bw)

          # generate a binary mask
          mask = arrayLevelMask(z, denthr, np.pi*(self.bw)*(self.bw))

          # tag masked out cells
          ii = np.ones(len(data))
          ii[irc[:, 0]] = mask[irc[:, 1], irc[:, 2]]
          data['masked'] = (ii < 1)

          # redefine possibly misclasified target cells by KDE filter
          # (those in low density regions)
          data.loc[data['masked'], 'class'] = ld_code
          data.drop(columns=['masked', 'i'], inplace=True)

      # drop cells not in the approved class list
      self.cell_data = data.loc[data['class'].isin(self.classes['class'])]
      

  def roi_mask(self):
      
      from myfunctions import kdeMask, filterCells
      
      [_, roiarr] = kdeMask(self.cell_data, self.imshape, self.bw)
      self.cell_data = filterCells(self.cell_data, roiarr)
      self.roiarr = roiarr
      
      
  def kde_mask(self):
      """
      Calculates a pixel resolution KDE profile for each cell type and record
      it into a raster array.

      """
      
      from myfunctions import kdeMask

      N = len(self.cell_data)
      classes = self.classes.copy()
      kdearr= np.zeros((self.imshape[0], self.imshape[1], len(classes)))

      classes['raster_index'] = classes.index
      classes['number_of_cells'] = 0

      for i, row in classes.iterrows():

          code = row['class']
          aux = self.cell_data.loc[self.cell_data['class'] == code]
          classes.loc[classes['class'] == code, 'number_of_cells'] = len(aux)

          if (len(aux) > 0):
              # do KDE on pixel locations of cells
              [z, _] = kdeMask(aux, self.imshape, self.bw)
              kdearr[:, :, i] = z

      classes['fraction_of_total'] = classes['number_of_cells'] / N
      
      self.kdearr = kdearr
      self.classes = classes
      
      
  def abumix_mask(self):
      """
      Calculates a pixel resolution abundance and mixing score profile for each
      cell type and record it into raster arrays.

      Mixing is scored by a univariate implementation of the Morisita-Horn index:

              M := 2*sum{x_i*y_i}/(sum{x_i*x_i} + sum{y_i*y_i})

      where {y_i} is a uniform distribution with the same volume as {x_i}:

              sum{y_i} = sum{x_i} = N and y_i = k = constant

      then k = N/A with A the area of integration (or number of elements)

      The constant k is the mean density of elements in the integration region,
      and thus M can be written as:

                      M  = 2 * N^2 / (A*sum{x_i*x_i} + N^2)

      This can be implemented spacially using a convolution function to get the
      local number of elements (N = sum{x_i}) and the dispersion (sum{x_i*x_i})
      inside the kernel in each position, giving us a pixel-level resolution
      metric of the level of uniformity (i.e. mixing) of elements. It is best
      if the argument 'raster' is a smooth version of the real array of
      discrete locations (typically a KDE with a small bandwidth).

      """
 
      from scipy.signal import fftconvolve

      def circle(r):
          y, x = np.ogrid[-r: r + 1, -r: r + 1]
          return(1*(x**2 + y**2 < r**2))

      classes = self.classes.copy()
      abuarr = np.zeros((self.imshape[0], self.imshape[1], len(classes)))
      mixarr = np.zeros((self.imshape[0], self.imshape[1], len(classes)))

      # produces a box-circle kernel (with size of quadrat)
      circ = circle(self.supbw)
      # area of kernel
      A = np.sum(circ)

      classes['mean_abundance'] = 0
      classes['std_abundance'] = 0
      classes['mean_mixing'] = 0
      classes['std_mixing'] = 0

      for i, clss in classes.iterrows():

          # smooth (binned) cell location field
          x = self.kdearr[:, :, i]

          # mask of cell locations
          msk = 1.0*(x > 0)

          # dispersion in local abundance
          xx = fftconvolve(np.multiply(x, x), circ, mode='same')
          xx[msk == 0] = 0

          # number of cells in kernel (local abundance)
          N = fftconvolve(x, circ, mode='same')
          N[msk == 0] = 0

          NN = np.multiply(N, N)
          M = 2 * np.divide(NN, (A*xx + NN),
                            out=np.zeros(msk.shape),
                            where=(msk > 0))

          N[msk == 0] = np.nan
          M[M > 1.0] = 1.0
          M[M <= 0.0] = np.nan
          M[msk == 0] = np.nan

          abuarr[:, :, i] = N
          mixarr[:, :, i] = M

          classes.loc[classes['class'] == clss['class'],
                      'mean_abundance'] = np.mean(N[msk > 0])
          classes.loc[classes['class'] == clss['class'],
                      'std_abundance'] = np.std(N[msk > 0])
          classes.loc[classes['class'] == clss['class'],
                      'mean_mixing'] = np.nanmean(M[msk > 0])
          classes.loc[classes['class'] == clss['class'],
                      'std_mixing'] = np.nanstd(M[msk > 0])

      self.abuarr = abuarr
      self.mixarr = mixarr
      self.classes = classes
      
     
  def quadrat_stats(self):
      """
      Gets a coarse grained representation of the sample based on quadrat
      estimation of cell abundances

      """
      
      from myfunctions import morisita_horn_univariate

      # define quadrats (only consider full quadrats)
      redges = np.arange(0, self.imshape[0], self.binsiz)
      if (max(redges) > (self.imshape[0] - self.binsiz)):
          redges = redges[:-1]
      cedges = np.arange(0, self.imshape[1], self.binsiz)
      if (max(cedges) > (self.imshape[1] - self.binsiz)):
          cedges = cedges[:-1]

      # dataframe for population abundances
      pop = pd.DataFrame()
      data = self.cell_data

      # get population vectors for each quadrat (square bin)
      for r, rbin in enumerate(redges):
          rmax = rbin + self.binsiz
          for c, cbin in enumerate(cedges):
              cmax = cbin + self.binsiz
              # drop quadrats outside the masked region
              m = self.mask[rbin:rmax, cbin:cmax]
              if (np.sum(m) == np.prod(m.shape)):
                  # subset data in quadrat
                  scdat = data.loc[(data['row'] >= rbin) &
                                   (data['row'] < rmax) &
                                   (data['col'] >= cbin) &
                                   (data['col'] < cmax)]

                  # count cells per class in quadrat, and total
                  ncells = len(scdat)

                  if ncells > 0:

                      # bin quadrat down to get spatial distributions
                      rowssubs = np.arange(rbin, rbin + self.binsiz, 
                                           self.subbinsiz)
                      colssubs = np.arange(cbin, cbin + self.binsiz, 
                                           self.subbinsiz)
                      subpops = np.zeros((self.classes['class'].size,
                                          len(rowssubs), len(colssubs)))

                      # get population vectors for each subquadrat
                      for r2, rbin2 in enumerate(rowssubs):
                          for c2, cbin2 in enumerate(colssubs):
                              # subset data in this subquadrat
                              sscdat = scdat.loc[(scdat['row'] >=
                                                  rbin2) & (scdat['row'] <
                                                  (rbin2 + self.subbinsiz)) &
                                                 (scdat['col'] >= cbin2) &
                                                 (scdat['col'] <
                                                  (cbin2 + self.subbinsiz))]

                              for i, code in enumerate(self.classes['class']):
                                  # get number of cells in each class
                                  # for each subsubarray
                                  m = sscdat.loc[sscdat['class'] == code]
                                  subpops[i, r2, c2] = len(m)

                      # get population fractions in each quadrant
                      aux = pd.DataFrame({'r': [r],
                                          'c': [c],
                                          'row_lo': [rbin],
                                          'row_hi': [rmax],
                                          'col_lo': [cbin],
                                          'col_hi': [cmax],
                                          'total':  [ncells]})

                      for i, code in enumerate(self.classes['class']):
                          # record abundance of this class
                          aux[code] = [sum(scdat['class'] == code)]

                          # record mixing level of this class
                          # (per Morisita-Horn univariate score)
                          m = morisita_horn_univariate(subpops[i, 
                                                               :, 
                                                               :].ravel())
                          aux[code + '_MH'] = [m]

                      pop = pd.concat([pop, aux], ignore_index=True)

      pop = pop.astype({'r': int, 'c': int})
      # pop = pop.fillna(0)
      pop['sample_id'] = self.sid
       
      return(pop)
   
    
  def local_stats(self):
      """
      Gets a coarse grained representation of the sample based on quadrat
      estimation of cell abundances. This is, abundance and mixing in a 
      discrete lattice of sice binsiz, which is a reduced sample of the sample
      in order to define regions for the LME in the TLA analysis.

      """

      abuarr = self.abuarr.copy()
      abuarr[np.isnan(abuarr)] = 0

      # define quadrats (only consider full quadrats)
      redges = np.arange(self.binsiz, self.imshape[0], self.binsiz)
      if (max(redges) > (self.imshape[0] - self.binsiz)):
          redges = redges[:-1]
      cedges = np.arange(self.binsiz, self.imshape[1], self.binsiz)
      if (max(cedges) > (self.imshape[1] - self.binsiz)):
          cedges = cedges[:-1]

      rcs = [(r, c) for r in redges for c in cedges]

      # dataframe for population abundances
      pop = pd.DataFrame()
      for rc in rcs:
          if self.roiarr[rc] > 0:
              n = np.nansum(self.abuarr[rc[0], rc[1], :])
              if n > 0:
                  aux = pd.DataFrame({'row': [rc[0]],
                                      'col': [rc[1]],
                                      'total': [n]})
                  for i, code in enumerate(self.classes['class']):
                      # record abundance of this class
                      aux[code] = [self.abuarr[rc[0], rc[1], i]]

                      # record mixing level of this class
                      # (per Morisita-Horn univariate score)
                      aux[code + '_MH'] = [self.mixarr[rc[0], rc[1], i]]

                  pop = pd.concat([pop, aux], ignore_index=True)

      pop = pop.astype({'row': int, 'col': int})
      pop['sample_id'] = self.sid

      return(pop)


  def gen_stats(self):
      """
      Aggregate general statistics for the sample, prints out and outputs 
      summary stats and a portion of a table to be constructued for each study

      """
      
      # general properties
      N = len(self.cell_data)
      A = np.sum(self.roiarr)
      roi_area = round(A*self.scale*self.scale, 4)
      
      # update sample table
      self.tbl['num_cells'] = N
      self.tbl['shape'] = self.imshape
         
      # estimate typical cell size based of density of points
      self.rcell = getCellSize(np.sum(self.kdearr, axis=2), self.binsiz)
       
      # Save a reference to the standard output
      original_stdout = sys.stdout
      with open(os.path.join(self.res_pth, 
                             self.sid +'_summary.txt'), 'w') as f:
          sys.stdout = f  # Change the standard output to the file we created.
          print('(*) Sample: ' + self.sid)
          print('(*) Landscape size (r,c)[pix]: ' + str(self.imshape) +
                '; (x,y)' + self.units + ": " +
                str([round(self.imshape[1]*self.scale, 2), 
                     round(self.imshape[0]*self.scale, 2)]))
          print('(*) ROI area [pix]^2: ' + str(A) + '; ' +
                self.units + "^2: " + str(roi_area))
          print('(*) Total cell density 1/[pix]^2: ' + str(round(N/A, 4)) +
                '; 1/' + self.units + "^2: " + str(round(N/A, 4)))
          print('(*) Composition: ' + str(N) +
                ' cells (uniquely identified, not overlaping):')
          print(self.classes[['class', 'class_name', 'number_of_cells',
                         'fraction_of_total']].to_markdown())
          print('(*) Typical radius of a cell [pix]: ' +
                str(round(self.rcell, 4)) + ' ; ' + self.units + ': ' +
                str(round(self.rcell*self.scale, 8)))
          # Reset the standard output to its original value
          sys.stdout = original_stdout
      del f

      stats = self.tbl[['sample_ID', 'num_cells']]
      stats['total_area'] = self.imshape[0]*self.imshape[1]
      stats['ROI_area'] = A
      for i, row in self.classes.iterrows():
          c = row['class']
          n = row['number_of_cells']
          stats[c + '_num_cells'] = n

      return(stats)
              
      
  def plot_landscape_scatter(self):
      """
      Produces scatter plot of landscape based on cell coordinates, with
      colors assigned to each cell type
      Also generates individual scatter plots for each cell type

      """
      from myfunctions import landscapeScatter

      [ar, redges, cedges, xedges, yedges] = plotEdges(self.imshape, 
                                                       self.binsiz, 
                                                       self.scale)

      fig, ax = plt.subplots(1, 1, figsize=(12, math.ceil(12/ar)),
                             facecolor='w', edgecolor='k')
      for i, row in self.classes.iterrows():
          aux = self.cell_data.loc[self.cell_data['class'] == row['class']]
          landscapeScatter(ax, aux.x, aux.y, row.class_color, row.class_name,
                           self.units, xedges, yedges, 
                           spoint=2*self.rcell, fontsiz=18)
      if (i % 2) != 0:
          ax.grid(which='major', linestyle='--', linewidth='0.3', color='black')
      ax.set_title('Sample ID: ' + str(self.sid), fontsize=20, y=1.02)
      ax.legend(labels=self.classes.class_name,
                loc='best',
                # loc='upper left',
                # loc='upper left', bbox_to_anchor=(1, 1),
                markerscale=10, fontsize=16, facecolor='w', edgecolor='k')

      plt.savefig(os.path.join(self.res_pth, 
                               self.sid + '_landscape_points.png'),
                  bbox_inches='tight', dpi=300)
      plt.close()

      fig, ax = plt.subplots(len(self.classes), 1,
                             figsize=(12, len(self.classes)*math.ceil(12/ar)),
                             facecolor='w', edgecolor='k')
      for i, row in self.classes.iterrows():
          aux = self.cell_data.loc[self.cell_data['class'] == row['class']]
          landscapeScatter(ax[i], aux.x, aux.y, 
                           row.class_color, row.class_name,
                           self.units, xedges, yedges, 
                           spoint=2*self.rcell, fontsiz=18)
          ax[i].set_title(row.class_name, fontsize=18, y=1.02)
      plt.suptitle('Sample ID: ' + str(self.sid), fontsize=24, y=1.001)
      plt.tight_layout()
      plt.savefig(os.path.join(self.res_pth, 
                               self.sid + '_landscape_classes.png'),
                  bbox_inches='tight', dpi=300)
      plt.close()


  def plot_landscape_props(self):
      """
      Produces general properties plot of landscape

      """
      from myfunctions import kdeLevels, landscapeLevels, plotRGB
      from myfunctions import landscapeScatter

      [ar, redges, cedges, xedges, yedges] = plotEdges(self.imshape, 
                                                       self.binsiz, 
                                                       self.scale)

      # gets kde profile of cell data
      [r, c, z, m, levs, th] = kdeLevels(self.cell_data, 
                                         self.imshape, 
                                         self.bw)
      x = c*self.scale
      y = (self.imshape[0] - r)*self.scale

      import warnings
      with warnings.catch_warnings():
          warnings.simplefilter('ignore')

          fig, ax = plt.subplots(2, 2, 
                                 figsize=(12*2, 0.5 + math.ceil(12*2/ar)),
                                 facecolor='w', edgecolor='k')

          # plots sample image
          plotRGB(ax[0, 0], self.img, self.units, 
                  cedges, redges, xedges, yedges,
                  fontsiz=18)
          ax[0, 0].set_title('Histology image', fontsize=18, y=1.02)

          # plots sample image
          plotRGB(ax[0, 1], 255*(self.msk == 0), self.units,
                  cedges, redges, xedges, yedges, fontsiz=18, cmap='gray')
          ax[0, 1].set_title('Mask image', fontsize=18, y=1.02)

          # plots sample scatter (with all cell classes)
          for i, row in self.classes.iterrows():
              aux = self.cell_data.loc[self.cell_data['class'] == row['class']]
              landscapeScatter(ax[1, 0], aux.x, aux.y,
                               row.class_color, row.class_name,
                               self.units, xedges, yedges,
                               spoint=2*self.rcell, fontsiz=18)
          if (i % 2) != 0:
              ax[1, 0].grid(which='major', linestyle='--',
                            linewidth='0.3', color='black')
          # plots roi contour over scatter
          ax[1, 0].contour(x, y, m, [0.5], linewidths=2, colors='black')
          ax[1, 0].set_title('Cell locations', fontsize=18, y=1.02)
          ax[1, 0].legend(labels=self.classes.class_name,
                          loc='best',
                          markerscale=10, fontsize=16,
                          facecolor='w', edgecolor='k')

          # plots kde levels
          landscapeLevels(ax[1, 1], x, y, z, m, levs,
                          self.units, xedges, yedges, fontsiz=18)
          ax[1, 1].set_title('KDE levels', fontsize=18, y=1.02)

          fig.subplots_adjust(hspace=0.4)
          fig.suptitle('Sample ID: ' + str(self.sid), fontsize=24, y=.95)
          fig.savefig(os.path.join(self.res_pth, 
                                   self.sid + '_landscape.png'),
                      bbox_inches='tight', dpi=300)
          plt.close()


  def plot_class_landscape_props(self):
      """
      Produces class properties plot of landscape

      """
      from myfunctions import plotRGB

      [ar, redges, cedges, xedges, yedges] = plotEdges(self.imshape, 
                                                       self.binsiz, 
                                                       self.scale)
      import warnings
      with warnings.catch_warnings():
          warnings.simplefilter('ignore')

          n = len(self.classes)
          vmin = np.floor(np.log10(np.quantile(self.abuarr[self.abuarr > 0], 
                                               0.25)))
          vmax = np.ceil(np.log10(np.quantile(self.abuarr[self.abuarr > 0], 
                                              0.75)))

          mixmin = np.floor(np.quantile(self.mixarr[self.mixarr > 0], 0.05))
          #mixmax = np.ceil(np.quantile(self.mixarr[self.mixarr > 0], 0.95))
          mixmax = 1.0

          fig, ax = plt.subplots(n, 2,
                                 figsize=(12*2, 0.5 + math.ceil(12*n/ar)),
                                 facecolor='w', edgecolor='k')

          # plots sample scatter (with all cell classes)
          for i, row in self.classes.iterrows():

              name = row['class_name']

              aux = self.abuarr[:, :, i].copy()
              msk = (aux == 0)
              aux[msk] = 0.00000001
              abuim = np.log10(aux)
              abuim[msk] = np.nan

              mixim = self.mixarr[:, :, i].copy()
              mixim[mixim == 0] = np.nan

              # plots kde image
              im = plotRGB(ax[i, 0], abuim, self.units,
                           cedges, redges, xedges, yedges,
                           fontsiz=18,
                           vmin=vmin, vmax=vmax, cmap='RdBu_r')
              plt.colorbar(im, ax=ax[i, 0], fraction=0.046, pad=0.04)
              ax[i, 0].set_title('Log Abundance image: ' + name,
                                 fontsize=18, y=1.02)

              # plots mix image
              im = plotRGB(ax[i, 1], mixim, self.units,
                           cedges, redges, xedges, yedges,
                           fontsiz=18,
                           vmin=mixmin, vmax=mixmax, cmap='RdBu_r')
              plt.colorbar(im, ax=ax[i, 1], fraction=0.046, pad=0.04)
              ax[i, 1].set_title('Mixing image: ' + name, fontsize=18, y=1.02)

          fig.subplots_adjust(hspace=0.4)
          fig.suptitle('Sample ID: ' + str(self.sid), fontsize=24, y=.95)
          fig.savefig(os.path.join(self.res_pth, 
                                   self.sid + '_abu_mix_landscape.png'),
                      bbox_inches='tight', dpi=300)
          plt.close()


      
# %% Private Functions
    
def progressBar(i, n, step, Nsteps,  msg, msg_i):
    
    printProgressBar(Nsteps*i + step, Nsteps*n,
                     suffix=msg + ' ; ' + msg_i, length=50)

def xyShift(data, shape, ref, scale):
    """
    Shifts coordinates and transforms into physical units

    Parameters
    ----------
    - data: (pandas) TLA dataframe of cell coordinates
    - shape: (tuple) shape in pixels of TLA landscape
    - ref: (tuple) reference location (upper-right corner)
    - scale: (float) scale of physical units / pixel

    """

    cell_data = data.copy()

    # first drops duplicate entries
    # (shuffling first to keep a random copy of each duplicate)
    cell_data = cell_data.sample(frac=1.0).drop_duplicates(['x', 'y'])
    cell_data = cell_data.reset_index(drop=True)

    # generate a cell id
    cell_data['cell_id'] = cell_data.index + 1

    # round pixel coordinates
    cell_data['col'] = np.uint32(np.rint(cell_data['x']))
    cell_data['row'] = np.uint32(np.rint(cell_data['y']))

    # shift coordinates to reference point
    cell_data['row'] = cell_data['row'] - ref[0]
    cell_data['col'] = cell_data['col'] - ref[1]

    # scale coordinates to physical units and transforms vertical axis
    cell_data['x'] = round(cell_data['col']*scale, 4)
    cell_data['y'] = round((shape[0] - cell_data['row'])*scale, 4)
    
    # drops data points outside of frame
    cell_data = cell_data.loc[(cell_data['row'] > 0) &
                              (cell_data['row'] < shape[0]) &
                              (cell_data['col'] > 0) &
                              (cell_data['col'] < shape[1])]

    return(cell_data)


def getCellSize(cell_arr, r):
    """
    Assuming close packing (max cell density), the typical size of a cell
    is based on the maximum number of cells found in a circle of radious `bw`
    (which is calculated using a fast convolution algorithm): the typical area
    of a cell is estimated as:
          (area of circle) / (max number of cells in circle)

    Parameters
    ----------
    - cell_arr: (pumpy) array with cell locations, or densities
    - r: (float) radious of circular kernel

    """

    from scipy.signal import fftconvolve

    def circle(r):
        y, x = np.ogrid[-r: r + 1, -r: r + 1]
        return(1*(x**2 + y**2 < r**2))

    # produces a box-circle kernel
    circ = circle(int(np.ceil(r)))

    # convolve array of cell locations with kernel
    # (ie. number of cells inside circle centered in each pixel)
    N = fftconvolve(cell_arr, circ)

    # the typical area of a cell
    # (given by maximun number of cells in any circle)
    acell = np.sum(circ)/np.max(N)

    # returns radius of a typical cell

    return(round(np.sqrt(acell/np.pi), 4))


def getBlobs(data, mask):
    """
    Gets labels from blob regions mask, and assing them to the cell data

    Parameters
    ----------
    - data: (pandas df) TLA dataframe of cell coordinates
    - mask: (numpy) binary mask defining blob regions

    """

    from skimage import measure

    # get a binary image of the mask
    msk_img = np.zeros(mask.shape)
    msk_img[mask > 127] = 1

    # label blobs in mask image
    blobs_labels = measure.label(msk_img, background=0, connectivity=2)

    # get data coordinates and labels for mask
    rows, cols = np.where(msk_img > 0)
    msk_data = pd.DataFrame({'blob': blobs_labels[rows, cols],
                             'row': rows,
                             'col': cols})

    aux = pd.merge(data, msk_data, how="left",
                   on=["row", "col"]).fillna(0)
    aux = aux.astype({'blob': int})

    return(aux, blobs_labels)


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


def quadDist(z, lab, var, xlims, ax):
    """
    Produces quadrat value distribution

    Parameters
    ----------
    - z: (numpy) values
    - lab: (str) label
    - var: (str) variable name
    - xlims:(tuple) limits for x values
    - ax: (axis) subplot axis handler

    """

    # from scipy.stats import expon

    db = (xlims[1] - xlims[0] + 1)/50
    bins = np.arange(xlims[0], xlims[1], db)

    mu = np.mean(z)
    sigma = np.std(z)

    ttl = r'{} ($\mu={}$, $\sigma={}$)'.format(lab,
                                               str(round(mu, 4)),
                                               str(round(sigma, 4)))

    # the histogram of the data
    n, bins, _ = ax.hist(z, bins, density=False,
                         facecolor='blue', alpha=0.5)
    # y = expon.pdf(bins, scale=mu)
    # c = expon.cdf(bins, scale=mu)
    # ax.plot(bins, y, 'r--')
    # ax.plot(bins, c, 'g--')
    ax.set_xlabel(var)
    ax.set_ylabel('Frequency')
    ax.set_title(ttl)
    ax.set_xlim([xlims[0], xlims[1]])
    ax.set_yscale('log')

    return(max(n))


def quadFigs(quadrats, classes, pngout):
    """
    Produces quadrat figures

    Parameters
    ----------
    - quadrats: (pandas) quadrats dataframe
    - classes: (pandas) dataframe with cell classes
    - pngout: (str) name of figure file

    """

    fig, ax = plt.subplots(2, len(classes),
                           figsize=(len(classes)*6, 2*6),
                           facecolor='w', edgecolor='k')

    # maxval = max(quadrats[classes['class'].tolist()].max(axis=0))
    maxfrq = [0, 0]

    tme_edges = pd.DataFrame()

    for i, row in classes.iterrows():

        aux = pd.DataFrame({'class': [row['class']]})

        vals = quadrats[row['class']]
        qats = np.nanquantile(vals, [0.0, 0.5, 0.87, 1.0]).tolist()
        # qats = [0.0, np.mean(vals),
        #         np.mean(vals) + np.std(vals), np.max(vals)]
        n = quadDist(vals, row['class_name'],
                     'Cells per quadrat',
                     [0, np.nanmax(vals)], ax[0, i])
        if (n > maxfrq[0]):
            maxfrq[0] = n
        aux['abundance_edges'] = [qats]
        for xc in qats:
            ax[0, i].axvline(x=xc,
                             color='red', linestyle='dashed', linewidth=1)

        vals = quadrats[row['class']+'_MH']
        qats = [0.0, 0.2, 0.67, 1.0]
        n = quadDist(vals, row['class_name'],
                     'Mixing index per quadrat',
                     [0.0, 1.0], ax[1, i])
        if (n > maxfrq[1]):
            maxfrq[1] = n
        aux['mixing_edges'] = [qats]
        for xc in qats:
            ax[1, i].axvline(x=xc,
                             color='red', linestyle='dashed', linewidth=1)

        tme_edges = pd.concat([tme_edges, aux], ignore_index=True)

    for i, row in classes.iterrows():
        ax[0, i].set_ylim([0.9, 1.1*maxfrq[0]])
        ax[1, i].set_ylim([0.9, 1.1*maxfrq[1]])

    plt.tight_layout()
    plt.savefig(pngout, bbox_inches='tight', dpi=300)
    plt.close()

    return(pd.merge(classes, tme_edges, how="left", on=['class']))


# %% Main function

def main(args):
    """
    *******  Main function  *******

    """

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
    # *** There is typically just one study in the a set, but this allows for
    #     running a set of analyses in a single call
    for k, stu in args_tbl.iterrows():
        
        # loads arguments for this study
        study = Study(stu, main_pth)
        
        if debug:
            study.samples = study.samples.iloc[:1]
            
        numsamples = len(study.samples.index)

        print("==> Processing study: " + study.name +
              "; [{0}/{1}]".format(k + 1, len(args_tbl.index)))

        # number of processing steps
        Nsteps = 11

        # %% loops over samples in this study
        for index, samp in study.samples.iterrows():

            sid = samp.sample_ID

            # creates a message to display in progress bar
            msg = '==> ' + sid + "; [{0}/{1}]".format(index + 1, numsamples)

            # %% STEP 1: creates data directories and new sample table
            progressBar(index, numsamples, 1, Nsteps, msg, 
                        'checking data folders...')
            
            # creates sample object and data folders for pre-processed data
            sample = Sample(samp, study)
            
            # %% if pre-processing files do not exist
            if REDO or ((not os.path.exists(sample.cell_data_file)) or
                        (not os.path.exists(sample.classes_file)) or
                        (not os.path.exists(sample.raster_file))):

                # STEP 2: loads and format coordinate data
                progressBar(index, numsamples, 2, Nsteps, msg, 
                            'loading data...')
                sample.setup_data(sample.supbw)

                # STEP 3: Filter cells according to density filters
                progressBar(index, numsamples, 3, Nsteps, msg, 
                            'filtering data...')
                sample.filter_class(study)
                
                # STEP 4: calculate a ROI mask for region with cells
                progressBar(index, numsamples, 4, Nsteps, msg, 
                            'creating ROI mask...')
                sample.roi_mask()
                
                # STEP 5: create raster images from density KDE profiles
                progressBar(index, numsamples, 5, Nsteps, msg, 
                            'creating KDE raster...')
                sample.kde_mask()

                # STEP 6: create raster images from cell mixing profiles
                progressBar(index, numsamples, 6, Nsteps, msg, 
                            'creating Abundance-Mix raster...')
                sample.abumix_mask()
                
                # STEP 7: saves main data files
                progressBar(index, numsamples, 7, Nsteps, msg, 
                            'saving data for this sample...')
                sample.save_data()

            else:
                # STEP 8: if sample is already pre-processed read data
                progressBar(index, numsamples, 8, Nsteps, msg, 
                           'loading data...')
                sample.load_data()


            # %% STEP 9: calculates quadrat populations for coarse graining
            progressBar(index, numsamples, 9, Nsteps, msg, 
                       'calculating stats...')
            
            # MUST run these first because they modify sample.tbl
            # qstats = sample.quadrat_stats(study.binsiz, study.subbinsiz)
            qstats = sample.local_stats()
            mstats = sample.gen_stats()
            
            # ads stats to study table
            study.add_sample(sample.tbl, mstats, qstats)

            
            # %% STEP 10: plots landscape data
            progressBar(index, numsamples, 10, Nsteps, msg, 
                       'ploting landscape props...')
            sample.plot_landscape_scatter()
            sample.plot_landscape_props()
            sample.plot_class_landscape_props()
            
        # %% saves study results
        progressBar(index, numsamples, Nsteps, Nsteps, msg, 
                   'saving summary tables...')
        study.output()

    print("\n ==> Pre-processing finished!")
    # %% the end
    return(0)


# %% Argument parser
if __name__ == "__main__":

    # Create the parser
    my_parser = ArgumentParser(prog="tla_setup",
                               description="### Pre-processing module for " +
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
                           help="If redo is used, then redo analysis")

    # passes arguments object to main
    main(my_parser.parse_args())
