'''
    TLA spit single sample:
    ########################
    
        This process prepares and formats sample data for TLA.

        This script reads parameters from a study set table (only the 1st line)
        Then (raw) data for one sample (given by a specified case number) is 
        read, and split into quadrats of a specified size and saved as 
        different samples, corresponding to the same initial large sample. This
        allows for using TLA in samples that are very large.
        
'''

# %% Imports

import os
import sys
import warnings
import pandas as pd
import numpy as np
from PIL import Image
from argparse import ArgumentParser

Image.MAX_IMAGE_PIXELS = 600000000

__version__  = "1.2.0"


# %% Private classes

class Study:
    
  def __init__(self, study, main_pth):
      
      from myfunctions import mkdirs
      
      # loads arguments for this study
      self.name = study['name']
      self.raw_path = os.path.join(main_pth, study['raw_raw_path'])
      
      # loads samples table for this study
      f = os.path.join(self.raw_path, study['raw_samples_table'])
      if not os.path.exists(f):
          print("ERROR: samples table file " + f + " does not exist!")
          sys.exit()
      self.samples = pd.read_csv(f)
      self.samples.fillna('', inplace=True)
      
      # sets path for processed data
      self.dat_path = mkdirs(os.path.join(main_pth, study['raw_path']))

      # creates tables for output
      self.samples_out = pd.DataFrame()
      
      
class Sample:
    
  def __init__(self, i, study):
      
      from myfunctions import mkdirs
      
      # creates sample object
      self.tbl = study.samples.iloc[i].copy()  # table of parameters
      self.sid = self.tbl.sample_ID            # sample ID
      self.classes = study.classes             # cell classes
      
      # raw data files
      self.raw_cell_data_file = os.path.join(study.raw_path, 
                                             self.tbl.coord_file)
      
      # other sample attributes (to be filled later)    
      self.cell_data = pd.DataFrame() # dataframe of cell data (coordinates)
      
      # creates cellPos folder and add path to sample tbl
      f = mkdirs(os.path.join(study.dat_path, 'cellPos'))
      self.tbl['coord_file'] = 'cellPos/' + self.sid + '.csv'
      self.cell_data_file = os.path.join(f, self.sid + '.csv')

      # creates images folder and add path to sample tbl
      if (self.tbl.image_file == ''):
          self.raw_imfile = ''
          self.imfile = ''  
          self.isimg = False
      else: 
          self.raw_imfile = os.path.join(study.raw_path, self.tbl.image_file)
          f = mkdirs(os.path.join(study.dat_path, 'images'))
          self.tbl['image_file'] = 'images/' + self.sid + '_img.jpg'
          self.imfile = os.path.join(f, self.sid + '_img.jpg')
          self.isimg = True
          
      # creates raster folder and add path to df
      f = mkdirs(os.path.join(study.dat_path, 'rasters', self.sid))
      pth = 'rasters/' + self.sid + '/'
      fmsk = os.path.join(study.raw_path, self.tbl.mask_file)
      if (self.tbl.mask_file == '' or not os.path.exists(fmsk)):    
          self.raw_mkfile = ''
          self.mask_file = ''
          self.ismk = False
      else:
          self.raw_mkfile = fmsk
          self.tbl['mask_file'] = pth + self.sid +'_mask.npz'
          self.mask_file = os.path.join(f, self.sid + '_mask.npz')
          self.ismk = True
      
     
  def setup_data(self, size_rows, size_cols):
      """
      Loads coordinates data, shift and convert values.
      Crops images to [size_rows, size_cols] sections
      
      """
     
      from skimage import io
      from skimage.transform import resize
      
      if not os.path.exists(self.raw_cell_data_file):
          print("ERROR: data file " + self.raw_cell_data_file + \
                " does not exist!")
          sys.exit()
      cxy = pd.read_csv(self.raw_cell_data_file)
      
      # updates coordinae values by conversion factor (from pix to xip)
      cxy.x, cxy.y = np.int32(cxy.x), np.int32(cxy.y)

      # gets extreme pixel values
      xmin, xmax = np.min(cxy.x), np.max(cxy.x)
      ymin, ymax = np.min(cxy.y), np.max(cxy.y)

      imshape = [np.nan, np.nan]
      
      # reads image file (if exists)
      if self.isimg:
          if os.path.exists(self.raw_imfile):
              ims = io.imread(self.raw_imfile)
              imsh = (ims.shape[0]*self.factor,
                      ims.shape[1]*self.factor,
                      ims.shape[2])
              ims = resize(ims, imsh, anti_aliasing=True, preserve_range=True)
              imshape = ims.shape
          else:
              print("WARNING: image: " + self.raw_imfile + " not found!")
              self.isimg = False

      # reads mask file (if exists)
      if self.ismk:
          if os.path.exists(self.raw_mkfile):
              msc = io.imread(self.raw_mkfile)
              imsh = (msc.shape[0]*self.factor,
                      msc.shape[1]*self.factor)
              msc = resize(msc, imsh, anti_aliasing=True, preserve_range=True)
              imshape = msc.shape
          else:
              print("WARNING: image: " + self.raw_mkfile + " not found!")
              self.ismk = False
              
      # check for consistency in image and mask
      if ((self.isimg  and self.ismk) and
          ((ims.shape[0] != msc.shape[0]) or
           (ims.shape[1] != msc.shape[1]))):
          #print('\n =====> WARNING! sample_ID: ' + self.sid +
          #      '; image and mask are NOT of the same size, ' +
          #      'thus adopting mask domain...')
          ims = np.rint(resize(ims, (msc.shape[0], msc.shape[1], 3),
                               anti_aliasing=True, 
                               preserve_range=True)).astype('uint8')

      # limits for image cropping
      with warnings.catch_warnings():
          warnings.simplefilter("ignore")
          rmin = np.nanmax([0, ymin])
          cmin = np.nanmax([0, xmin])
          rmax = np.nanmin([imshape[0] - 1, ymax])
          cmax = np.nanmin([imshape[1] - 1, xmax])
      
      dr = rmax - rmin
      dc = cmax - cmin
      if (np.isnan(dr) or np.isnan(dc)):
          print("ERROR: data file " + self.raw_cell_data_file + \
                " is an empty landscape!")
          sys.exit()
          
      imshape = [int(dr + 1), int(dc + 1)]

      # shifts coordinates
      cell_data = xyShift(cxy, imshape, [rmin, cmin], self.scale)

      # create croped versions of image and mask raster
      img = np.zeros((imshape[0], imshape[1], 3))
      if self.isimg:
          img[0:(rmax - rmin), 
              0:(cmax - cmin), :] = ims[rmin:rmax, cmin:cmax, :]
          self.img = img.astype('uint8')
          io.imsave(self.imfile, self.img, check_contrast=False)
      
      
      msk = np.zeros(imshape)
      if self.ismk:
          msk[0:(rmax - rmin), 
              0:(cmax - cmin)] = msc[rmin:rmax, cmin:cmax]
          io.imsave(msk, self.mask_file, check_contrast=False)
      
      
      self.cell_data = cell_data.reset_index(drop=True)
      self.imshape = imshape    
      
      
  def save_data(self):
      
      # saves main data files
      self.cell_data.to_csv(self.cell_data_file, index=False)
      self.classes.to_csv(self.classes_file, index=False)
      self.qstats.to_csv(os.path.join(self.res_pth, 
                                      self.sid + '_quadrat_stats.csv'),
                         index=False, header=True)
    
  def output(self, study): 
      
      samples_out = self.tbl.to_frame().T
      samples_out = samples_out.astype({'num_cells': int})
      samples_out = samples_out.astype(study.samples.dtypes)
      samples_out.to_csv(os.path.join(self.res_pth, 
                                      self.sid + '_samples.csv'), 
                         index=False, header=True)
     
      allstats_out = self.mstats.to_frame().T
      allstats_out.to_csv(os.path.join(self.res_pth, 
                                       self.sid + '_samples_stats.csv'), 
                          index=False, header=True)
   
      
# %% Private Functions

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
    #cell_data['cell_id'] = cell_data.index + 1

    # round pixel coordinates
    cell_data['col'] = np.uint32(np.rint(cell_data['x']))
    cell_data['row'] = np.uint32(np.rint(cell_data['y']))

    # shift coordinates to reference point
    cell_data['row'] = cell_data['row'] - ref[0]
    cell_data['col'] = cell_data['col'] - ref[1]

    # scale coordinates to physical units and transforms vertical axis
    cell_data['x'] = round(cell_data['col']*scale, 6)
    cell_data['y'] = round((shape[0] - cell_data['row'])*scale, 6)
    
    # drops data points outside of frames
    cell_data = cell_data.loc[(cell_data['row'] > 0) &
                              (cell_data['row'] < shape[0]) &
                              (cell_data['col'] > 0) &
                              (cell_data['col'] < shape[1])]

    return(cell_data)

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
        argsfile = os.path.join(main_pth, 'PC.csv')
        CASE = 0
        SIZROW = 2000
        SIZCOL = 2000
    else:
        # running from the CLI using the bash script
        # path to working directory (above /scripts)
        main_pth = os.getcwd()
        argsfile = os.path.join(main_pth, args.argsfile) 
        CASE = args.casenum
        SIZROW = args.size_rows
        SIZCOL = args.size_cols
    
    print("==> The working directory is: " + main_pth)
    
    if not os.path.exists(argsfile):
        print("ERROR: The specified argument file does not exist!")
        sys.exit()
        
    # NOTE: only ONE line in the argument table will be used
    study = Study( pd.read_csv(argsfile).iloc[0], main_pth)
    
    # %% STEP 1: creates data directories and new sample table
    # creates sample object and data folders for pre-processed data
    
    sample = Sample(CASE, study)
    msg = "====> Case [" + str(CASE + 1) + \
          "/" + str(len(study.samples.index)) + \
          "] :: SID <- " + sample.sid 
        
    print( msg + " >>> splitting samples..." )
       
    # %% STEP 2: loads and format coordinate data
    sample.setup_data(SIZROW, SIZCOL)
           
    # %% STEP 9: saves main data files
    sample.save_data()
            
    # LAST step: saves study stats results for sample 
    sample.output(study)

    # %% the end
    return(0)


# %% Argument parser
if __name__ == "__main__":

    # Create the parser
    my_parser = ArgumentParser(prog="tla_split_sample",
                               description="# Single Sample Split " + 
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
    
    my_parser.add_argument('size_rows',
                           metavar="size_rows",
                           type=int,
                           help="Set row size of quadrats")
    
    my_parser.add_argument('size_cols',
                           metavar="size_cols",
                           type=int,
                           help="Set col size of quadrats")
    
    my_parser.add_argument("--graph",
                           default=False,
                           action="store_true",
                           help="If <<--graph>> is used, then print graphs")

    my_parser.add_argument("--redo",
                           default=False,
                           action="store_true",
                           help="If <<--redo>> is used, then redo analysis")

    # passes arguments object to main
    main(my_parser.parse_args())
