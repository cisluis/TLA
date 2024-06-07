'''
    TLA setup single sample:
    ########################
    
        This process prepares and formats sample data for TLA.

        This script reads parameters from a study set table (only the 1st line)
        Then (raw) data for one sample (given by a specified case number) is 
        read, extreme values calculated and data is formatted. 
        Cell classes are checked out and a new, curated, coordinate file is 
        produced with a cropped version of the IHC image and raster arrays
        (at hoc region mask, a mask that defines the ROI, and a multilevel 
         raster array with kde cell density info).
        
        Also, a sample data summary and general stats are produced for the 
        sample, BUT aggregated statistics across samples should be calculated 
        using 'TLA_setup_sum' after pre-processing all samples in the study. 
        
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
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from argparse import ArgumentParser

if torch.cuda.is_available(): 
    ISCUDA = True
else:
    ISCUDA = False
    
Image.MAX_IMAGE_PIXELS = 600000000

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:10240"


__version__  = "2.0.0"

# %% Private classes

class Study:
    
    def __init__(self, study, main_pth):
        
        from myfunctions import mkdirs, tofloat, toint
        
        # loads arguments for this study
        self.name = study['name']
        self.raw_path = os.path.join(main_pth, study['raw_path'])
        
        # loads samples table for this study
        f = os.path.join(self.raw_path, study['raw_samples_table'])
        if not os.path.exists(f):
            print("ERROR: samples table file " + f + " does not exist!")
            sys.exit()
        self.samples = pd.read_csv(f)
        self.samples.fillna('', inplace=True)
        
        # sets path for processed data
        self.dat_path = mkdirs(os.path.join(main_pth, study['data_path']))
        
        # list of processed samples
        self.done_list = os.path.join(self.dat_path, 'setup_done_samples.csv')
        
        # scale parameters
        self.factor = 1.0
        if 'factor' in study:
            self.factor = study['factor']
        self.scale = tofloat(study['scale']/self.factor)
        self.units = study['units']
  
        # the size of quadrats and subquadrats
        aux = 4*np.ceil((study['binsiz']/self.scale)/4)
        self.binsiz = toint(np.rint(aux))
        self.subbinsiz = toint(np.rint(self.binsiz/4))
  
        # bandwidth size for convolutions is half the quadrat size
        self.supbw = toint(np.rint(self.binsiz/2))
        self.bw = toint(np.rint(self.subbinsiz/2))
        
        # r values for Ripleys' H function
        dr = toint(np.rint(self.bw/2))
        self.rs = list(tofloat(np.arange(dr, (self.binsiz/2)+dr, dr)))
        self.ridx = toint((np.abs(np.asarray(self.rs) - self.supbw)).argmin())
  
        # Filter parameters:
        # (-) cell class to run filter on (e.g. tumor cells)
        self.FILTER_CODE = str(study['FILTER_CODE'])
        # (-) code to assign to cells in high density regions
        self.HIGH_DENS_CODE = str(study['HIGH_DENS_CODE'])
        # (-) code to assign to cells in low density regions
        self.LOW_DENS_CODE = str(study['LOW_DENS_CODE'])
        # (-) threshold high/low density regions (0 for no filtering)
        self.DTHRES = study['DTHRES']
        # (-) if true, uses 'blob' label in data to mask cells
        self.BLOBS = study['BLOBS']
  
        # reduces classes df to just the accepted types (i.e. `drop=False`)
        f = os.path.join(self.raw_path, study['raw_classes_table'])
        if not os.path.exists(f):
            print("ERROR: classes file " + f + " does not exist!")
            sys.exit()
        classes = pd.read_csv(f)
        classes.drop(classes.loc[classes['drop']].index, inplace=True)
        classes.drop(columns=['drop'], inplace=True)
        classes.reset_index(inplace=True, drop=True)
        classes['class'] = classes['class'].astype(str)
        self.classes = classes
  
        # creates tables for output
        self.samples_out = pd.DataFrame()
        self.allstats_out = pd.DataFrame()
        self.allpops_out = pd.DataFrame()
      
      
class Sample:
    
    def __init__(self, i, study):
        
        from myfunctions import mkdirs, tofloat, toint
      
        # creates sample object
        self.tbl = study.samples.iloc[i].copy()  # table of parameters
        self.sid = self.tbl.sample_ID            # sample ID
        self.classes = study.classes             # cell classes
        
        # raw data files
        self.raw_cell_data_file = os.path.join(study.raw_path, 
                                               self.tbl.coord_file)
        
        # the size of quadrats and subquadrats
        self.binsiz = study.binsiz
        self.subbinsiz = study.subbinsiz
    
        # bandwidth size for convolutions is half the quadrat size
        self.supbw = study.supbw
        self.bw = study.bw
        
        # r values for Ripleys' H function
        self.rs = study.rs
        self.ridx = study.ridx
       
        # scale parameters
        self.factor = study.factor
        self.scale = study.scale
        self.units = study.units
        
        # other sample attributes (to be filled later)    
        self.cell_data = pd.DataFrame() # dataframe of cell data (coordinates)
        self.imshape = []               # shape of accepted image
        
        # sample arrays
        self.img = []                   # image array
        self.msk = []                   # mask array (blobs)
        self.roiarr = []                # ROI array
        
        # global spacial factors
        self.coloc = []
        self.nndist = []
        self.rhfunc = []
        
        # stats
        self.qstats = []                # quadrat statistics table 
        self.mstats = []                # general statistics table 
        
        
        # creates results folder and add path to sample tbl
        f = mkdirs(os.path.join(study.dat_path, 
                                'results', 'samples', self.sid))
        self.tbl['results_dir'] = 'results/samples/' + self.sid + '/'
        self.res_pth = f
        
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
            self.raw_imfile = os.path.join(study.raw_path, 
                                           self.tbl.image_file)
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
              
        # raster file names
        self.raster_folder = f
        self.roi_file = os.path.join(f, self.sid + '_roi.npz')
        self.kde_file = os.path.join(f, self.sid + '_kde.npz')
        self.abumix_file = os.path.join(f, self.sid + '_abumix.npz')
        self.spafac_file = os.path.join(f, self.sid + '_spafac.npz')
        
        # classes info file
        self.classes_file = os.path.join(self.res_pth, 
                                         self.sid + '_classes.csv')
      
        # total number of cells (after filtering)
        self.num_cells = toint(0)        
        
        # cell size attribute
        self.rcell = tofloat(np.nan)
      
     
    def setup_data(self, edge):
        """
        Loads coordinates data, shift and convert values.
        Crops images to the corresponding convex hull
        
        > edge : maximum size [pix] of edge around data extremes in rasters
        
        """
       
        from skimage import io
        from skimage.transform import resize
        from myfunctions import toint, tobit
        
        if not os.path.exists(self.raw_cell_data_file):
            print("ERROR: data file " + self.raw_cell_data_file + \
                  " does not exist!")
            sys.exit()
        cxy = pd.read_csv(self.raw_cell_data_file)[['class', 'x', 'y']]
        cxy['class'] = cxy['class'].astype(str)
        cxy = cxy.loc[cxy['class'].isin(self.classes['class'])]
        
        #############
        #### code to estimate the NN distances between all cells
        # from scipy.spatial import KDTree
        # rc = np.array(cxy[['x', 'y']])
        # dnn, _ = KDTree(rc).query(rc, k=[2])
        # dmin = np.min(dnn)  
        # if dmin > 2:
        #     # rescaling factor is such that min cell distance is sqrt(2)
        #     #(which is the max distance between adjcent cell in square grid)
        #     self.factor = np.sqrt(2)/dmin
        # NOTE: This part of the code is only to determine a good factor value
        #       DO NOT UNCOMMENT... use it find a value and edit in the 
        #       parameter file for the study (so all samples have the same value)
        #       then re-run the analysis
            
        # updates coordinae values by conversion factor (from pix to piy)
        cxy.x = toint(cxy.x*self.factor)
        cxy.y = toint(cxy.y*self.factor)  
            
        # gets extreme pixel values
        xmin, xmax = toint(np.min(cxy.x)), toint(np.max(cxy.x))
        ymin, ymax = toint(np.min(cxy.y)), toint(np.max(cxy.y))
  
        imshape = [toint(0), toint(0)]
  
        # reads image file (if exists)
        if self.isimg:
            if os.path.exists(self.raw_imfile):
                ims = io.imread(self.raw_imfile)
                imsh = (ims.shape[0]*self.factor,
                        ims.shape[1]*self.factor,
                        ims.shape[2])
                ims = tobit(resize(ims, imsh, 
                             anti_aliasing=True, 
                             preserve_range=True))
                imshape = ims.shape
            else:
                print("WARNING: image: " + self.raw_imfile + " not found!")
                self.isimg = False
  
        # reads mask file (if exists)
        if self.ismk:
            if os.path.exists(self.raw_mkfile):
                msc = io.imread(self.raw_mkfile)
                imsh = (toint(msc.shape[0]*self.factor),
                        toint(msc.shape[1]*self.factor))
                msc = tobit(resize(msc, imsh, 
                             anti_aliasing=True, 
                             preserve_range=True))
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
            ims = tobit(np.rint(resize(ims, 
                                       (msc.shape[0], msc.shape[1], 3),
                                       anti_aliasing=True,
                                       preserve_range=True)))
  
        # limits for image cropping
        rmin = toint(np.nanmax([0, ymin - edge]))
        cmin = toint(np.nanmax([0, xmin - edge]))
        rmax = toint(np.nanmin([imshape[0] - 1, ymax + edge]))
        cmax = toint(np.nanmin([imshape[1] - 1, xmax + edge]))
        
        dr = rmax - rmin
        dc = cmax - cmin
        if (np.isnan(dr) or np.isnan(dc)):
            print("ERROR: data file " + self.raw_cell_data_file + \
                  " is an empty landscape!")
            sys.exit()
        imshape = [toint(dr + 1), toint(dc + 1)]
        
        
        #############
        #### code to estimate the cell overlaps after subsampling with factor
        # factor = 8
        # data = cxy.copy()
        # data.x = np.rint(data.x/factor).astype('int64')
        # data.y = np.rint(data.y/factor).astype('int64')
        
        # N = len(data)
        # aux = data.groupby(['x','y', 'class'], as_index = False).size()
        # df = aux.pivot(index=['x','y'], 
        #               columns='class', 
        #               values = 'size').fillna(0).reset_index()  
        # df = df.rename(columns={"1": "n1", "2": "n2", "3": "n3"})
        # df["ntot"] = df["n1"]+df["n2"]+df["n3"]
        # df = df[df['ntot']>1]
        # df['numzeros'] = (df == 0).astype(int).sum(axis=1)
        # nmax = int(np.max(df["ntot"]))
              
        # cts, bins, fig1 = plt.hist(df["ntot"], bins=range(2,nmax+1))
        # cts = 100*np.multiply(cts, bins[:-1])/N
        
        # aux=df.loc[df['numzeros']==2]
        # cts2, bins, fig = plt.hist(aux["ntot"], bins=range(2,nmax+1))
        # cts2 = 100*np.multiply(cts2, bins[:-1])/N
        
        # aux=df.loc[df['numzeros']==1]
        # cts1, bins, fig = plt.hist(aux["ntot"], bins=range(2,nmax+1))
        # cts1 = 100*np.multiply(cts1, bins[:-1])/N
        
        # aux=df.loc[df['numzeros']==0]
        # cts0, bins, fig = plt.hist(aux["ntot"], bins=range(2,nmax+1))
        # cts0 = 100*np.multiply(cts0, bins[:-1])/N
        
        
        # shifts coordinates
        cell_data = xyShift(cxy, imshape, [rmin, cmin], self.scale)
  
        # create croped versions of image and mask raster
        if self.isimg:
            img = np.zeros((imshape[0], imshape[1], 3), dtype='uint8')
            img[0:(rmax - rmin), 
                0:(cmax - cmin), :] = tobit(ims[rmin:rmax,
                                                cmin:cmax, :])
            self.img = img
            io.imsave(self.imfile, self.img, check_contrast=False)
        else:
            self.img = []
        
        if self.ismk:
            msk = np.zeros(imshape, dtype='uint8')
            msk[0:(rmax - rmin), 
                0:(cmax - cmin)] = tobit(msc[rmin:rmax, cmin:cmax])
            [cell_data, msk_img] = getBlobs(cell_data, msk)
            np.savez_compressed(self.mask_file, roi=msk_img)
            self.msk = (msk_img > 0).astype('bool')
        else:
            self.msk = []
        
        self.cell_data = cell_data.reset_index(drop=True)
        self.imshape = [toint(imshape[0]), toint(imshape[1])] 
        self.num_cells = toint(len(self.cell_data))
        
      
    def save_data(self):
        
        # saves main data files
        self.cell_data.to_csv(self.cell_data_file, index=False)
        self.classes.to_csv(self.classes_file, index=False)
        self.qstats.to_csv(os.path.join(self.res_pth, 
                                        self.sid + '_quadrat_stats.csv'),
                           index=False, header=True)
        
      
    def load_data(self):
        
        from skimage import io
        from myfunctions import tofloat, toint, tobit
        
        # load all data (if it was previously created)
        self.cell_data = pd.read_csv(self.cell_data_file)
        self.cell_data['class'] = self.cell_data['class'].astype(str)
        self.classes = pd.read_csv(self.classes_file)
        self.classes['class'] = self.classes['class'].astype(str)
       
        aux = np.load(self.roi_file)
        self.roiarr = aux['roi'].astype('bool')
        
        aux = np.load(self.spafac_file)
        self.coloc = tofloat(aux['coloc'])
        self.nndist = tofloat(aux['nndist']) 
        self.aefunc = tofloat(aux['aefunc'])
        self.rhfunc = tofloat(aux['rhfunc'])
        
        f = os.path.join(self.res_pth, self.sid + '_quadrat_stats.csv')
        if not os.path.exists(f):
            print("ERROR: samples table file " + f + " does not exist!")
            sys.exit()
        self.qstats = pd.read_csv(f)
        
        self.imshape = [toint(self.roiarr.shape[0]), 
                        toint(self.roiarr.shape[1])]
        
        if (self.isimg ):
            self.img = tobit(io.imread(self.imfile))
        else:    
            self.img = np.zeros((self.imshape[0],
                                 self.imshape[1], 3), dtype='uint8')
            
        if self.ismk:
            aux = np.load(self.mask_file)
            self.msk = aux['roi'].astype('bool')
        else:
            self.msk = np.zeros(self.imshape, dtype='bool')
    
    
      
    def output(self, study, trun, memmax): 
        
        samples_out = self.tbl.to_frame().T
        samples_out = samples_out.astype(study.samples.dtypes)
        samples_out = samples_out.astype({'num_cells': 'int'})
        samples_out['setup_runtime'] = trun
        samples_out['setup_maxMBs'] = memmax
        samples_out.to_csv(os.path.join(self.res_pth, 
                                        self.sid + '_samples.csv'), 
                           index=False, header=True)
       
        allstats_out = self.mstats.to_frame().T
        allstats_out.to_csv(os.path.join(self.res_pth, 
                                         self.sid + '_samples_stats.csv'), 
                            index=False, header=True)
        
        
    def filter_class(self, study):
        """
        Applies a class filter
    
        Parameters
        ----------
        - target_code: (str) cell class to run filter on (e.g. tumor cells)
        - hd_code: (str) code to assign to cells in high density regions
        - ld_code: (str) code to assign to cells in low density regions
        - denthr: (float) threshold to distinguish high and low density regions.
                  If denthr=0 then no filtering
        - blobs: (bool) if true, uses 'blob' label in data to mask cells. In this
                 case cells outside of blobs are reassigned the ld_code and those
                 inside are assigned the hd_code. If denthr>0 then cells in low
                 denisity regions inside the blobs are assigned the hd_code, BUT
                 all cells outside the blobs are assigned the hd_code.
    
        """
        from myfunctions import KDE, arrayLevelMask, tofloat, toint
        
        target_code = study.FILTER_CODE
        hd_code =  study.HIGH_DENS_CODE
        ld_code = study.LOW_DENS_CODE
        denthr = tofloat(study.DTHRES)
        blobs = (self.ismk) and (study.BLOBS)
    
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
            
        if (denthr > 0):
            
            data['i'] = data.index
    
            # subset data to just target cells, gets array of values
            aux = data.loc[data['class'] == hd_code]
            
            if (len(aux) > 0):
                
                irc = toint(np.array(aux[['i', 'row', 'col']]))
                
                # do KDE on pixel locations of target cells (e.g. tumor cells)
                [_, _, z] = KDE(aux, self.imshape, self.supbw, cuda=ISCUDA)
      
                #### code to find a good 'denthr'
                # z[z<=1e-5] = 1e-5
                # lz = np.log10(z)
                # plt.hist(lz.ravel(), bins = np.arange(-5, 2, .1))
                # plt.show()
                # denthr = 5
                
                # generate a binary mask
                mask = arrayLevelMask(z, denthr, self.supbw, fill_holes=False)
      
                # plt.figure()
                # plt.imshow(z)
                # plt.figure()
                # plt.imshow(mask)
                
                # tag masked out cells
                ii = np.ones(len(data), dtype='bool')
                ii[irc[:, 0]] = mask[irc[:, 1], irc[:, 2]]
      
                # redefine possibly misclasified target cells by means of 
                # the KDE filter (i.e. those in low density regions)
                data.loc[np.logical_not(ii), 'class'] = ld_code
                data.drop(columns=['i'], inplace=True)
    
        # drop cells not in the approved class list
        self.cell_data = data.loc[data['class'].isin(self.classes['class'])]
        
    
    def roi_mask(self, redo):
      
      from myfunctions import filterCells
      
      fout = self.roi_file
      
      if redo or not os.path.exists(fout):
      
          from myfunctions import kdeMask, toint
          
          # gets a mask for the region that has cells inside
          [_, self.roiarr] = kdeMask(self.cell_data, 
                                     self.imshape, 
                                     self.bw, 
                                     cuda=ISCUDA)
          np.savez_compressed(fout, roi=self.roiarr)
      
      else:
            
          aux = np.load(fout)
          self.roiarr = aux['roi'].astype('bool')
      
      # filter out cells outside of ROI
      self.cell_data = filterCells(self.cell_data, self.roiarr)
      
      # total number of cells (after filtering) 
      self.num_cells = toint(len(self.cell_data)) 
      
      
    def kde_mask(self, redo):
        """
        Calculates a pixel resolution KDE profile for each cell type and record
        it into a raster array.
    
        """
        
        from myfunctions import toint, tofloat
        
        classes = self.classes.copy()
        classes['raster_index'] = classes.index
        classes['number_of_cells'] = toint(0)
        classes['fraction_of_total'] = tofloat(np.nan)
        aux = []
        for i, row in classes.iterrows():
            aux = self.cell_data.loc[self.cell_data['class'] == row['class']]
            classes.loc[classes['class'] == row['class'], 
                        'number_of_cells'] = toint(len(aux))
            
        if (self.num_cells > 0):
            f = classes['number_of_cells']/self.num_cells
            classes['fraction_of_total'] = tofloat(np.around(f, decimals=4))
        self.classes = classes    
            
        fout = self.kde_file
        if redo or not os.path.exists(fout):
        
            from myfunctions import kdeMask
      
            kdearr = tofloat(np.zeros((self.imshape[0],
                                       self.imshape[1],
                                       len(classes))))
            
            for i, row in classes.iterrows():
                c = row['class']
                aux = self.cell_data.loc[self.cell_data['class'] == c]
                if (len(aux) > 0):
                    # do KDE on pixel locations of cells
                    [z, _] = kdeMask(aux, self.imshape, self.bw, cuda=ISCUDA)
                    
                    kdearr[:, :, i] = tofloat(z*self.roiarr)
                    del z
                del aux 
            
            np.savez_compressed(fout, kde=kdearr)
                
        else:
            
            aux = np.load(fout)
            kdearr = tofloat(aux['kde'])
            del aux 
     
        # estimate typical cell size based of density of points
        self.rcell = getCellSize(np.sum(kdearr, axis=2), self.binsiz)
        
        # clean memory
        gc.collect()
        
        return(kdearr)
       
       
    def abumix_mask(self, redo, kdearr):
        """
        Calculates a pixel resolution abundance and mixing score profile for each
        cell type and record it into raster arrays.
    
        Mixing is scored by a univariate implementation of the Morisita-Horn 
        index:
    
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
        
        from myfunctions import tofloat
        
        classes = self.classes.copy() 
        classes['mean_abundance'] = tofloat(0.0)
        classes['std_abundance'] = tofloat(np.nan)
        classes['mean_mixing'] = tofloat(0.0)
        classes['std_mixing'] = tofloat(np.nan)
        
        fout = self.abumix_file
        if redo or not os.path.exists(fout):
     
            abuarr = tofloat(np.full((self.imshape[0], 
                                     self.imshape[1],
                                     len(classes)), np.nan))
            mixarr = tofloat(np.full((self.imshape[0], 
                                     self.imshape[1],
                                     len(classes)), np.nan))
                
            for i, clss in classes.iterrows():
      
                # get smooth (binned) cell location field for this class
                x = kdearr[:, :, i]
               
                if (np.sum(x) > 0):
      
                    [msk, N, M] = abumix(x, self.bw, cuda=ISCUDA)
          
                    # assign raster values
                    abuarr[:, :, i] = N
                    mixarr[:, :, i] = M
                    
                    # get basic stats
                    n = np.mean(N[msk])
                    if ( n > 0):
                        classes.loc[classes['class'] == clss['class'],
                                    'mean_abundance'] = np.round(n, 4)
                        aux = np.round(np.std(N[msk]), 6)
                        classes.loc[classes['class'] == clss['class'],
                                    'std_abundance'] = aux
                        aux = np.round(np.nanmean(M[msk]), 4)
                        classes.loc[classes['class'] == clss['class'],
                                    'mean_mixing'] = aux
                        aux = np.round(np.nanstd(M[msk]), 6)
                        classes.loc[classes['class'] == clss['class'],
                                    'std_mixing'] = aux
                        
                    del msk
                    del N
                    del M
                    
                del x
                
                # saves raster
                np.savez_compressed(fout, abu=abuarr, mix=mixarr)
     
        else:
            
            # loads raster
            aux = np.load(fout)
            abuarr = tofloat(aux['abu'])
            mixarr = tofloat(aux['mix'])
            
            for i, clss in classes.iterrows():
                
                # smooth (binned) cell location field for this class
                x = kdearr[:, :, i]
                
                if (np.sum(x) > 0):
      
                    # mask of cell locations
                    msk = (x > 0)
                
                    # assign raster values
                    N = abuarr[:, :, i]
                    M = mixarr[:, :, i]
                    
                    # get basic stats
                    n = np.mean(N[msk])
                    if ( n > 0):
                        classes.loc[classes['class'] == clss['class'],
                                    'mean_abundance'] = np.round(n, 4)
                        aux = np.round(np.std(N[msk]), 6)
                        classes.loc[classes['class'] == clss['class'],
                                    'std_abundance'] = aux
                        aux = np.round(np.nanmean(M[msk]), 4)
                        classes.loc[classes['class'] == clss['class'],
                                    'mean_mixing'] = aux
                        aux = np.round(np.nanstd(M[msk]), 6)
                        classes.loc[classes['class'] == clss['class'],
                                    'std_mixing'] = aux
                        
                    del msk
                    del N
                    del M
                    
                del x
     
        # updates classes df
        self.classes = classes  
        
        # clean memory
        gc.collect()
        
        return([abuarr, mixarr])
        
    
    def quadrat_stats(self, abuarr, mixarr):
        """
        Gets a coarse grained representation of the sample based on quadrat
        estimation of cell abundances. This is, abundance and mixing in a 
        discrete lattice of size binsiz, which is a reduced sample of the sample
        in order to define regions for the LME in the TLA analysis.
    
        """
        
        from myfunctions import toint, tofloat
    
        # abuarr[np.isnan(abuarr)] = 0
    
        # define quadrats (only consider full quadrats)
        redges = toint(np.arange(self.bw, self.imshape[0], 2*self.bw))
        #if (max(redges) > (self.imshape[0] - self.bw)):
        #    redges = redges[:-1]
        cedges = toint(np.arange(self.bw, self.imshape[1], 2*self.bw))
        #if (max(cedges) > (self.imshape[1] - self.bw)):
        #    cedges = cedges[:-1]
    
        rcs = [(r, c) for r in redges for c in cedges]
    
        # dataframe for population abundances
        pop = pd.DataFrame({'sample_ID': [],
                            'row': [],
                            'col': [],
                            'total': []})
        for rc in rcs:
            if self.roiarr[rc] > 0:
                n = np.nansum(abuarr[rc[0], rc[1], :])
                if n > 0:
                    aux = pd.DataFrame({'sample_ID': [self.sid],
                                        'row': [rc[0]],
                                        'col': [rc[1]],
                                        'total': [n]})
                    for i, code in enumerate(self.classes['class']):
                        # record abundance of this class
                        auy = tofloat([abuarr[rc[0], rc[1], i]])
                        aux[code] = np.round(auy, 4)
    
                        # record mixing level of this class
                        # (per Morisita-Horn univariate score)
                        auy = tofloat([mixarr[rc[0], rc[1], i]])
                        aux[code + '_MH'] = np.round(auy, 4)
    
                    pop = pd.concat([pop, aux], ignore_index=True)
                    
        pop['row']=toint(pop['row'])
        pop['col']=toint(pop['col'])
        pop['total']=tofloat(pop['total'])
    
        self.qstats = pop


    def space_stats(self, redo, dat_path):
          """
          Calculates global statistics in raster arrays:
              
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
    
          3- Attraction Enrichment Function Score 
             Bi-variate asymetric score, evaluated at r=subbw (subkernel scale) 
             between all classes ('ref' and 'test')
             (*) T = +1 0 indicates attraction of 'test' cells around 'ref' cells
             (*) T == 0 indicates random dipersion between 'test' and 'ref' cells
             (*) T = -1 indicates repulsion of 'test' cells from 'ref' cells
             
          4- Ripley's H function score
             Bi-variate asymetric version of Ripley's H(r) function, evaluated 
             at r=subbw (subkernel scale) between all classes ('ref' and 'test').
             (*) H > 0 indicates clustering of 'test' cells around 'ref' cells
             (*) H ~ 0 indicates random mixing between 'test' and 'ref' cells
             (*) H < 0 indicates dispersion of 'test' cells around 'ref' cells
    
          """
          
          from myfunctions import toint, tofloat, tobit
          
          fout = self.spafac_file
          if redo or not os.path.exists(fout):
          
              from myfunctions import fftconv2d
              from myfunctions import circle, nndist, attraction_T_biv
              from myfunctions import ripleys_K, ripleys_K_biv
              
              data = self.cell_data
              
              # number of classes
              nc = len(self.classes)
              
              # landscape area
              A = toint(np.sum(self.roiarr))
      
              # raster array of cell locations
              X = tobit(np.zeros((self.imshape[0], 
                                  self.imshape[1], 
                                  nc)))
              # raster array of cell abundance in subkernels
              n = toint(np.zeros((self.imshape[0], 
                                  self.imshape[1], 
                                  nc, 
                                  len(self.rs))))
      
              # cell global densities
              ptdens = tofloat(np.zeros(nc))
              
              # Colocalization index
              colocarr = tofloat(np.full((nc, nc), np.nan))
              
              # Nearest Neighbor Distance index
              nndistarr = tofloat(np.full((nc, nc), np.nan))
              
              # precalculate local abundance for all classes and radii
              # (needs to be precalculated for combination measures)
              for i, clsx in self.classes.iterrows():
                  # coordinates of cells in class x
                  aux = data.loc[data['class'] == clsx['class']]
                  if (len(aux) > 0):
                      ptdens[i] = len(aux)/A
                      X[aux.row, aux.col, i] = 1
                      for k, r in enumerate(self.rs):
                          auy = np.rint(fftconv2d(X[:, :, i], 
                                                  circle(r), 
                                                  cuda=ISCUDA))
                          auy[auy<0]=0
                          n[:,:,i,k] = toint(auy)
                          del auy
                          gc.collect()
                                                
              rsp = list(np.around(np.array(self.rs) * self.scale, decimals=2))
              df = pd.DataFrame({'sample_ID': [str(x) for x in self.rs],
                                 'r': self.rs, 'r_physical': rsp})
              df['a'] = np.around(np.pi*df.r*df.r, decimals=2)
              df['a_physical'] = np.around(np.pi*df.r_physical*df.r_physical, 
                                           decimals=2)
                                                               
              # Attraction Function Score 
              aefuncarr = tofloat(np.full((nc, nc, len(self.rs), 2), np.nan))
              aefuncarr[:, :, 0:len(self.rs), 0]= self.rs
              aefuncdf = df.copy()
              aefuncdf.sample_ID = self.sid            
                                                               
              # Ripley's H function
              rhfuncarr = tofloat(np.full((nc, nc, len(self.rs), 2), np.nan))
              rhfuncarr[:, :, 0:len(self.rs), 0]= self.rs
              rhfuncdf = df.copy()
              rhfuncdf.sample_ID = self.sid            
              
              # loop thru all combinations of classes (pair-wise comparisons)
              for i, clsx in self.classes.iterrows():
                  
                  # coordinates of cells in class x
                  aux = data.loc[data['class'] == clsx['class']]
                  
                  if (len(aux) > 0):
                  
                      rcx = np.array(aux[['row', 'col']])    
                      for k, r in enumerate(self.rs):
                          
                          # Ripleys H score (identity)
                          rk = ripleys_K(rcx, n[:, :, i, k])
                          rk = (np.sqrt(rk/np.pi) - r)/r
                          rhfuncarr[i, i, k, 1] = tofloat(rk)
                          del rk
                          
                          # Attraction Enrichment Score (identity)
                          dy = n[:, :, i, k]/(np.pi*r*r)
                          tk = attraction_T_biv(rcx,
                                                ptdens[i],
                                                dy)
                          aefuncarr[i, i, k, 1] = tofloat(tk)
                      
                      gc.collect()
                      
                      nam = clsx['class'] + '_' + clsx['class']
                      rhfuncdf['H_' + nam] = np.round(rhfuncarr[i, i, :, 1],4) 
                      aefuncdf['T_' + nam] = aefuncarr[i, i, :, 1]
          
                      # Colocalization index (identity)
                      colocarr[i, i] = 1.0
          
                      # Nearest Neighbor Distance index (identity)
                      nndistarr[i, i] = 0.0
                      
                      for j, clsy in self.classes.iterrows():
          
                          # coordinates of cells in class y
                          auy = data.loc[data['class'] == clsy['class']]
                          
                          if (len(auy) > 0):
                          
                              rcy = toint(np.array(auy[['row', 'col']]))
              
                              if (i != j):
                                  
                                  for k, r in enumerate(self.rs):
                                      
                                      # Ripleys H score (bivarite)
                                      rk = ripleys_K_biv(rcx, n[:, :, i, k], 
                                                         rcy, n[:, :, j, k])
                                      # Old definition:
                                      # rk = (np.sqrt(rk/np.pi) - r)/r
                                      # New definition, easier to interpret
                                      if (rk > 0):
                                          rk = np.log10(np.sqrt(rk/np.pi)/r)
                                      else:
                                          rk = np.nan
                                      rhfuncarr[i, j, k, 1] = tofloat(rk)
                                      
                                      # Attraction Function Score (bivarite)
                                      dy = n[:, :, j, k]/(np.pi*r*r)
                                      tk = attraction_T_biv(rcx,
                                                            ptdens[j],
                                                            dy)
                                      aefuncarr[i, j, k, 1] = tofloat(tk)
                                      
                                  nam =  clsx['class'] + '_' + clsy['class']
                                  v = np.round(rhfuncarr[i, j, :, 1], 4)
                                  rhfuncdf['H_' + nam] = v
                                  T = aefuncarr[i, j, :, 1]
                                  aefuncdf['T_' + nam] = T
              
                                  # Nearest Neighbor Distance index
                                  nndistarr[i, j] = tofloat(nndist(rcx, rcy))
                          
                              if (i > j):
                                  # MH index from quadrats sampling
                                  cx = clsx['class']
                                  cy = clsy['class']
                                  
                                  if ((cx in self.qstats.columns) and
                                      (cy in self.qstats.columns)):
                                      
                                      qs = self.qstats[[clsx['class'],
                                                        clsy['class']]].copy()
                                      qs = qs.dropna()
                                      qs['x'] =  qs[clsx['class']]/ \
                                          qs[clsx['class']].sum()
                                      qs['y'] =  qs[clsy['class']]/ \
                                          qs[clsy['class']].sum()
                                      
                                      xxyy = (qs['x']*qs['x']).sum() + \
                                          (qs['y']*qs['y']).sum()
                                  
                                      if (xxyy > 0):
                                          M = 2 * (qs['x']*qs['y']).sum()/xxyy
                                      else:
                                          M = np.nan
                                  else:
                                      M = np.nan
                                      
                                  colocarr[i, j]= tofloat(M)
                                  colocarr[j, i]= tofloat(M)
      
      
              f = os.path.join(dat_path, 'results', 'samples', 
                               self.sid, self.sid + '_Ripleys_H_function.csv')
              rhfuncdf.to_csv(f, index=False, header=True)
              
              f = os.path.join(dat_path, 'results', 'samples', 
                               self.sid, self.sid + '_Attraction_T_score.csv')
              aefuncdf.to_csv(f, index=False, header=True)
              
              np.savez_compressed(self.spafac_file, 
                                  coloc=colocarr,
                                  nndist=nndistarr,
                                  aefunc=aefuncarr,
                                  rhfunc=rhfuncarr)
              
              del rhfuncdf
              del aefuncdf
              del X
              del n
              gc.collect()
          
          else:
                
              aux = np.load(fout)
              colocarr = tofloat(aux['coloc'])
              nndistarr = tofloat(aux['nndist'])
              aefuncarr = tofloat(aux['aefunc'])
              rhfuncarr = tofloat(aux['rhfunc'])
              
          self.coloc = colocarr
          self.nndist = nndistarr
          self.aefunc = aefuncarr
          self.rhfunc = rhfuncarr
          
    
    def general_stats(self):
        """
        Aggregate general statistics for the sample, prints out and outputs 
        summary stats and a portion of a table to be constructued for each study
    
        """
        from itertools import combinations, permutations, product
        from scipy.spatial import KDTree
        from myfunctions import toint, tofloat
        # general properties
        N = toint(len(self.cell_data))
        A = toint(np.sum(self.roiarr))
        roi_area = tofloat(np.round(A*self.scale*self.scale, 4))
              
        # update sample table
        self.tbl['num_cells'] = N
        self.tbl['shape'] = self.imshape
        
        # estimates the NN distances between all cells
        rc = toint(np.array(self.cell_data[['row', 'col']]))
        dnn, _ = KDTree(rc).query(rc, k=[2])
        dnnqs = np.quantile(dnn, [0,0.25,0.5,0.75,1])     
        del dnn
        gc.collect()
        
        stats = self.tbl[['sample_ID', 'num_cells']]
        stats['total_area'] = toint(self.imshape[0]*self.imshape[1])
        stats['ROI_area'] = A
        
        for i, row in self.classes.iterrows():
            c = row['class']
            n = row['number_of_cells']
            stats[c + '_num_cells'] = n
            
            if (A>0):
                stats[c + '_den_cells'] = np.round(n/A, 6)
            else:
                stats[c + '_den_cells'] = np.nan
              
        # records overal Morisita-Horn index at pixel level
        comps = list(combinations(self.classes.index.values.tolist(), 2))
        for comp in comps:   
            ca = self.classes.iloc[comp[0]]['class']
            cb = self.classes.iloc[comp[1]]['class']
            aux = np.round(self.coloc[comp[0], comp[1]], 4)
            stats['coloc_' + ca + '_' + cb] = tofloat(aux)
            
        # records overal Nearest Neighbor Distance index at pixel level
        comps = list(permutations(self.classes.index.values.tolist(), 2))
        for comp in comps:   
            ca = self.classes.iloc[comp[0]]['class']
            cb = self.classes.iloc[comp[1]]['class']
            aux = np.round(self.nndist[comp[0], comp[1]], 4)
            stats['nndist_' + ca + '_' + cb] = tofloat(aux)
            
        # records overal Attraction Enrichment Function index at pixel level
        comps = list(product(self.classes.index.values.tolist(), repeat=2))
        for comp in comps:   
            ca = self.classes.iloc[comp[0]]['class']
            cb = self.classes.iloc[comp[1]]['class']
            aux = np.round(self.aefunc[comp[0], comp[1],
                                               self.ridx, 1], 4)
            stats['aefunc_' + ca + '_' + cb] = tofloat(aux)
            
        # records overal Ripley's H score at pixel level
        comps = list(product(self.classes.index.values.tolist(), repeat=2))
        for comp in comps:   
            ca = self.classes.iloc[comp[0]]['class']
            cb = self.classes.iloc[comp[1]]['class']
            aux = np.round(self.rhfunc[comp[0], comp[1],
                                               self.ridx, 1], 4)
            stats['rhfunc_' + ca + '_' + cb] = tofloat(aux)
            
        if ( A > 0 ):
            tot_dens = N/A
            tot_dens_units = tofloat(np.round(N/roi_area, 4))
        else:
            tot_dens = np.nan
            tot_dens_units = tofloat(np.nan)
            
        # Save a reference to the standard output
        original_stdout = sys.stdout
        with open(os.path.join(self.res_pth, 
                               self.sid +'_summary.txt'), 'w') as f:
            sys.stdout = f  # Change the standard output to the file we created.
            print('(*) Sample: ' + self.sid)
            print('(*) Landscape size (r,c)[pix]: ' + str(self.imshape) +
                  '; (x,y)' + self.units + ": " +
                  str([np.round(self.imshape[1]*self.scale, 2), 
                       np.round(self.imshape[0]*self.scale, 2)]))
            print('(*) ROI area [pix]^2: ' + str(A) + '; ' +
                  self.units + "^2: " + str(roi_area))
            print('(*) Total cell density 1/[pix]^2: ' + 
                  str(np.round(tot_dens, 4)) + '; 1/' + self.units + "^2: " + 
                  str(np.round(tot_dens_units, 4)))
            print('(*) Composition: ' + str(N) +
                  ' cells (uniquely identified, not overlaping):')
            print(self.classes[['class', 'class_name', 'number_of_cells',
                           'fraction_of_total']].to_markdown())
            print('(*) Typical radius of a cell [pix]: ' +
                  str(np.round(self.rcell, 4)) + ' ; ' + self.units + ': ' +
                  str(np.round(self.rcell*self.scale, 8)))
            print('(*) NNDistance quantiles [0, 0.25, 0.5, 0.75, 1] [pix]: ' +
                  str(np.round(dnnqs, 4)) + ' ; ' + self.units + ': ' +
                  str(np.round(dnnqs*self.scale, 8)))
            print('(*) Overall Morisita-Horn Index: ')
            aux = stats[[c for c in stats.index if c.startswith('coloc_')]]
            aux.name = "M"
            print(aux.to_markdown())
            print('(*) Overall Nearest-Neighbor Distance Index: ')
            aux = stats[[c for c in stats.index if c.startswith('nndist_')]]
            aux.name = "D"
            print(aux.to_markdown())
            print('(*) Overall Attraction Enrichment Score T(' + \
                  str(self.bw) + '[pix]):')
            aux = stats[[c for c in stats.index if c.startswith('aefunc_')]]
            aux.name = "T"
            print(aux.to_markdown())
            print('(*) Overall Ripleys Function H(' + str(self.bw) + '[pix]):')
            aux = stats[[c for c in stats.index if c.startswith('rhfunc_')]]
            aux.name = "H"
            print(aux.to_markdown())
            # Reset the standard output to its original value
            sys.stdout = original_stdout
        del f
    
        self.mstats = stats
        
        
    def plot_landscape_scatter(self, lims):
        """
        Produces scatter plot of landscape based on cell coordinates, with
        colors assigned to each cell type
        Also generates individual scatter plots for each cell type
    
        """
        from myfunctions import landscapeScatter, plotEdges
    
        [ar, redges, cedges, xedges, yedges] = plotEdges(self.imshape, 
                                                         self.binsiz, 
                                                         self.scale)
         
        C, R = np.meshgrid(np.arange(0, self.imshape[1], 1),
                           np.arange(0, self.imshape[0], 1))
        grid = np.stack([R.ravel(), C.ravel()]).T
        x = ( np.unique(grid[:, 1]))*self.scale
        y = (self.imshape[0] - ( np.unique(grid[:, 0])))*self.scale
    
        fig, ax = plt.subplots(1, 1, figsize=(12, math.ceil(12/ar)),
                               facecolor='w', edgecolor='k')
        
        classes = self.classes.iloc[::-1]
        
        j = 0
        for i, row in classes.iterrows():
            aux = self.cell_data.loc[self.cell_data['class'] == row['class']]
            if (len(aux) > 0):
                landscapeScatter(ax, aux.x, aux.y, 
                                 row.class_color, row.class_name,
                                 self.units, xedges, yedges, 
                                 spoint=5*self.rcell, 
                                 fontsiz=18)
                j = j + 1
        if (j % 2) != 0:
            ax.grid(which='major', linestyle='--', 
                    linewidth='0.3', color='black')
        ax.contour(x, y, self.roiarr, [.50], linewidths=2, colors='black')
        ax.set_xlim([lims[0], lims[1]])
        ax.set_ylim([lims[2], lims[3]])
        ax.set_title('Sample ID: ' + str(self.sid), fontsize=20, y=1.04)
        ax.legend(labels=classes.class_name,
                  loc='best',
                  # loc='upper left',
                  # loc='upper left', bbox_to_anchor=(1, 1),
                  markerscale=3, fontsize=16, facecolor='w', edgecolor='k')
        
    
        plt.savefig(os.path.join(self.res_pth, 
                                 self.sid + '_landscape_points.png'),
                    bbox_inches='tight', dpi=300)
        plt.close()
    
        fig, ax = plt.subplots(1, len(classes), 
                               figsize=(12*len(classes), math.ceil(12/ar)),
                               facecolor='w', edgecolor='k')
        for i, row in classes.iterrows():
            aux = self.cell_data.loc[self.cell_data['class'] == row['class']]
            if (len(aux) > 0):
                landscapeScatter(ax[i], aux.x, aux.y, 
                                 row.class_color, row.class_name,
                                 self.units, xedges, yedges, 
                                 spoint=5*self.rcell, 
                                 fontsiz=18)
            ax[i].contour(x, y, self.roiarr, [.50], linewidths=2, colors='black')
            ax[i].set_title(row.class_name, fontsize=18, y=1.02)
            ax[i].set_xlim([lims[0], lims[1]])
            ax[i].set_ylim([lims[2], lims[3]])
        plt.suptitle('Sample ID: ' + str(self.sid), fontsize=24, y=1.04)
        plt.tight_layout()
        plt.savefig(os.path.join(self.res_pth, 
                                 self.sid + '_landscape_classes.png'),
                    bbox_inches='tight', dpi=300)
        plt.close()


    def plot_landscape_props(self, lims):
        """
        Produces general properties plot of landscape
    
        """
        from myfunctions import kdeLevels, landscapeLevels, plotRGB
        from myfunctions import landscapeScatter, plotEdges
    
        [ar, redges, cedges, xedges, yedges] = plotEdges(self.imshape, 
                                                         self.binsiz, 
                                                         self.scale)
    
        # gets kde profile of cell data
        [r, c, z, m, levs, th] = kdeLevels(self.cell_data, 
                                           self.imshape, 
                                           self.bw)
        
        xv, yv = np.meshgrid(range(m.shape[1]), range(m.shape[0]))
        xv = xv*self.scale
        yv = (self.imshape[0] - yv)*self.scale
        
        classes = self.classes.iloc[::-1]
    
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
    
            if (self.isimg or self.ismk):
                
                fig, ax = plt.subplots(2, 2, 
                                   figsize=(12*2, 0.5 + math.ceil(12*2/ar)),
                                   facecolor='w', edgecolor='k')
    
                # plots sample image
                if (self.isimg):
                    plotRGB(ax[0, 0], self.img, self.units, 
                            cedges, redges, xedges, yedges,
                            fontsiz=18)
                ax[0, 0].set_title('Histology image', fontsize=18, y=1.02)
                ax[0, 0].set_xlim([0, self.imshape[0]])
                ax[0, 0].set_ylim([0, self.imshape[1]])
    
                # plots sample image
                if (self.ismk):
                    plotRGB(ax[0, 1], 255*(self.msk > 0), self.units,
                            cedges, redges, xedges, yedges, fontsiz=18, cmap='gray')
                ax[0, 1].set_title('Mask image', fontsize=18, y=1.02)
                ax[0, 1].set_xlim([0, self.imshape[0]])
                ax[0, 1].set_ylim([0, self.imshape[1]])
    
                # plots sample scatter (with all cell classes)
                for i, row in classes.iterrows():
                    aux = self.cell_data.loc[self.cell_data['class'] == row['class']]
                    landscapeScatter(ax[1, 0], aux.x, aux.y,
                                     row.class_color, row.class_name,
                                     self.units, xedges, yedges,
                                     spoint=2*self.rcell, fontsiz=18)
                if (i % 2) != 0:
                    ax[1, 0].grid(which='major', linestyle='--',
                                  linewidth='0.3', color='black')
                # plots roi contour over scatter
                ax[1, 0].contour(xv, yv, m, [0.5], linewidths=2, colors='black')
                ax[1, 0].set_title('Cell locations', fontsize=18, y=1.02)
                ax[1, 0].legend(labels=classes.class_name,
                                loc='best',
                                markerscale=3, fontsize=16,
                                facecolor='w', edgecolor='k')
                ax[1, 0].set_xlim([lims[0], lims[1]])
                ax[1, 0].set_ylim([lims[2], lims[3]])
      
                # plots kde levels
                landscapeLevels(ax[1, 1], xv, yv, z, m, levs,
                                self.units, xedges, yedges, fontsiz=18)
                ax[1, 1].set_title('KDE levels', fontsize=18, y=1.02)
                ax[1, 1].set_xlim([lims[0], lims[1]])
                ax[1, 1].set_ylim([lims[2], lims[3]])
    
            else:
                
                fig, ax = plt.subplots(1, 2, 
                                 figsize=(12*2, 0.5 + math.ceil(12*1/ar)),
                                 facecolor='w', edgecolor='k')
                
                # plots sample scatter (with all cell classes)
                for i, row in classes.iterrows():
                    aux = self.cell_data.loc[self.cell_data['class'] == row['class']]
                    landscapeScatter(ax[0], aux.x, aux.y,
                                     row.class_color, row.class_name,
                                     self.units, xedges, yedges,
                                     spoint=2*self.rcell, fontsiz=18)
                if (i % 2) != 0:
                    ax[0].grid(which='major', linestyle='--',
                               linewidth='0.3', color='black')
                # plots roi contour over scatter
                ax[0].contour(xv, yv, m, [0.5], linewidths=2, colors='black')
                ax[0].set_title('Cell locations', fontsize=18, y=1.02)
                ax[0].legend(labels=classes.class_name,
                             loc='best',
                             markerscale=3, fontsize=16,
                             facecolor='w', edgecolor='k')
                ax[0].set_xlim([lims[0], lims[1]])
                ax[0].set_ylim([lims[2], lims[3]])
    
                # plots kde levels
                landscapeLevels(ax[1], xv, yv, z, m, levs,
                              self.units, xedges, yedges, fontsiz=18)
                ax[1].set_title('KDE levels', fontsize=18, y=1.02)
                ax[1].set_xlim([lims[0], lims[1]])
                ax[1].set_ylim([lims[2], lims[3]])
    
            fig.subplots_adjust(hspace=0.4)
            fig.suptitle('Sample ID: ' + str(self.sid), fontsize=24, y=.95)
            fig.savefig(os.path.join(self.res_pth, 
                                     self.sid + '_landscape.png'),
                        bbox_inches='tight', dpi=300)
            plt.close()
            
            
    def plot_landscape_simple(self, inx, lims):
        """
        Produces simple plot of landscape
    
        """
        from myfunctions import kdeLevels, landscapeLevels
        from myfunctions import landscapeScatter, plotEdges
    
        [ar, redges, cedges, xedges, yedges] = plotEdges(self.imshape, 
                                                         self.binsiz, 
                                                         self.scale)
    
        tclass = self.classes.iloc[inx]
    
        # gets kde profile of cell data
        # aux = self.cell_data
        aux = self.cell_data.loc[self.cell_data['class'] == tclass['class']]
        [r, c, z, m, levs, th] = kdeLevels(aux, 
                                           self.imshape, 
                                           self.bw)
              
        xv, yv = np.meshgrid(range(m.shape[1]), range(m.shape[0]))
        xv = xv*self.scale
        yv = (self.imshape[0] - yv)*self.scale
    
        classes = self.classes.iloc[::-1]
    
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
    
            fig, ax = plt.subplots(1, 2, 
                                   figsize=(12*2, 0.5 + math.ceil(12/ar)),
                                   facecolor='w', edgecolor='k')
    
            # plots sample scatter (with all cell classes)
            for i, row in classes.iterrows():
                aux = self.cell_data.loc[self.cell_data['class'] == row['class']]
                landscapeScatter(ax[0], aux.x, aux.y,
                                 row.class_color, row.class_name,
                                 self.units, xedges, yedges,
                                 spoint=5*self.rcell, fontsiz=18)
            if (i % 2) != 0:
                ax[0].grid(which='major', linestyle='--',
                              linewidth='0.3', color='black')
            # plots roi contour over scatters
            ax[0].contour(xv, yv, self.roiarr, 
                          [.50], linewidths=2, colors='black')
            ax[0].set_title('Cell locations', fontsize=18, y=1.02)
            ax[0].legend(labels=classes.class_name,
                            loc='best',
                            markerscale=3, fontsize=18,
                            facecolor='w', edgecolor='k')
            ax[0].set_xlim([lims[0], lims[1]])
            ax[0].set_ylim([lims[2], lims[3]])
    
     
            # plots kde levels
            landscapeLevels(ax[1], xv, yv, z, m, levs,
                            self.units, xedges, yedges, fontsiz=18)
            ax[1].set_title(tclass['class_name'] + ' - KDE levels', 
                            fontsize=18, y=1.02)
            ax[1].set_xlim([lims[0], lims[1]])
            ax[1].set_ylim([lims[2], lims[3]])
    
            fig.subplots_adjust(hspace=0.4)
            fig.suptitle('Sample ID: ' + str(self.sid), fontsize=18, y=.95)
            nam = self.sid + '_' + tclass['class']
            fig.savefig(os.path.join(self.res_pth, 
                                     nam + '_simple_landscape.png'),
                        bbox_inches='tight', dpi=300)
            plt.close()
    
    
    def plot_class_landscape_props(self, lims):
        """
        Produces class properties plot of landscape
    
        """
        from myfunctions import plotRGB, plotEdges
    
        [ar, redges, cedges, xedges, yedges] = plotEdges(self.imshape, 
                                                         self.binsiz, 
                                                         self.scale)
        
        aux = np.load(self.abumix_file)
        abuarr = aux['abu']
        mixarr = aux['mix']
        
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
    
            n = len(self.classes)
            vmin = np.floor(np.log10(np.quantile(abuarr[abuarr > 0],0.1)))
            vmax = np.ceil(np.log10(np.quantile(abuarr[abuarr > 0], 0.9)))
    
            mixmin = np.floor(np.quantile(mixarr[mixarr > 0], 0.05))
            #mixmax = np.ceil(np.quantile(mixarr[mixarr > 0], 0.95))
            mixmax = 1.0
            
            fig, ax = plt.subplots(n, 2,
                                   figsize=(12*2, 0.5 + math.ceil(12*n/ar)),
                                   facecolor='w', edgecolor='k')
    
            # plots sample scatter (with all cell classes)
            for i, row in self.classes.iterrows():
    
                name = row['class_name']
    
                aux = abuarr[:, :, i].copy()
                msk = (aux == 0)
                aux[msk] = 0.00000001
                abuim = np.log10(aux)
                abuim[msk] = np.nan
    
                mixim = mixarr[:, :, i].copy()
                mixim[mixim == 0] = np.nan
    
                # plots kde image
                im = plotRGB(ax[i, 0], abuim, self.units,
                             cedges, redges, xedges, yedges,
                             fontsiz=18,
                             vmin=vmin, vmax=vmax, cmap='RdBu_r')
                plt.colorbar(im, ax=ax[i, 0], fraction=0.046, pad=0.04)
                ax[i, 0].set_title('Log Abundance image: ' + name,
                                   fontsize=18, y=1.02)
                #ax[i, 0].set_xlim([lims[0], lims[1]])
                #ax[i, 0].set_ylim([lims[2], lims[3]])
    
                # plots mix image
                im = plotRGB(ax[i, 1], mixim, self.units,
                             cedges, redges, xedges, yedges,
                             fontsiz=18,
                             vmin=mixmin, vmax=mixmax, cmap='RdBu_r')
                plt.colorbar(im, ax=ax[i, 1], fraction=0.046, pad=0.04)
                ax[i, 1].set_title('Mixing image: ' + name, fontsize=18, y=1.02)
                #ax[i, 1].set_xlim([lims[0], lims[1]])
                #ax[i, 1].set_ylim([lims[2], lims[3]])
    
            fig.subplots_adjust(hspace=0.4)
            fig.suptitle('Sample ID: ' + str(self.sid), fontsize=24, y=.95)
            fig.savefig(os.path.join(self.res_pth, 
                                     self.sid + '_abu_mix_landscape.png'),
                        bbox_inches='tight', dpi=300)
            plt.close()
            
            del abuarr
            del mixarr
            gc.collect()


      
# %% Private Functions

def memuse():
    gc.collect()
    m = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
    return(round(m, 2))


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
    from myfunctions import tofloat, toint
    
    cell_data = data.copy()

    # first drops duplicate entries
    # (shuffling first to keep a random copy of each duplicate)
    cell_data = cell_data.sample(frac=1.0).drop_duplicates(['x', 'y'])
    cell_data = cell_data.reset_index(drop=True)

    # generate a cell id
    #cell_data['cell_id'] = cell_data.index + 1

    # round pixel coordinates
    cell_data['col'] = toint(np.rint(cell_data['x']))
    cell_data['row'] = toint(np.rint(cell_data['y']))

    # shift coordinates to reference point
    cell_data['row'] = cell_data['row'] - ref[0]
    cell_data['col'] = cell_data['col'] - ref[1]

    # scale coordinates to physical units and transforms vertical axis
    cell_data['x'] = tofloat(cell_data['col']*scale)
    cell_data['y'] = tofloat((shape[0] - cell_data['row'])*scale)
    
    # drops data points outside of frames
    cell_data = cell_data.loc[(cell_data['row'] >= 0) &
                              (cell_data['row'] < shape[0]) &
                              (cell_data['col'] >= 0) &
                              (cell_data['col'] < shape[1])]

    return(cell_data)


def abumix(x, bw, cuda = False):
    
    from myfunctions import circle, tofloat, tocuda
    
    # produces a (circular) kernel with same size as quadrat
    kernel = circle(bw) # box-circle
        
    # area of kernel
    A = np.sum(kernel)
    
    # mask of cell locations
    msk = (x > 0)
    
    if cuda:
        
        from myfunctions import cudaconv2d
        
        # if GPU is available, uses pytorch cuda
        
        # create tensor objects
        xs = tocuda(x)
        ks = tocuda(kernel)
        
        # mask of cell locations
        mskt = (xs > 0)
        
        # dispersion in local abundance
        xx = cudaconv2d(torch.mul(xs, xs), ks)
        xx[mskt == 0] = 0
        
        # number of cells in kernel (local abundance)
        Nt = cudaconv2d(xs, ks)
        Nt[mskt == 0] = 0
        
        # calculates the MH univariate index
        NN = torch.mul(Nt, Nt)
        denom = torch.add(A*xx, NN)
        
        # initialize output tensor with desired value
        Mt = torch.full_like(xs, fill_value=np.nan)
        Mt[mskt] = 2 * NN[mskt] / denom[mskt]
        
        # revert NAN values
        Mt[Mt > 1.0] = 1.0
        Mt[Mt < 0.0] = 0.0
        Mt[mskt == 0] = np.nan
        
        # return to CPU as numpy
        M = Mt.cpu().numpy()
        N = Nt.cpu().numpy()
        
        del xs, ks, mskt, xx, Nt, NN, denom, Mt
        torch.cuda.empty_cache()
        
    else:
        
        from scipy.signal import convolve

        # dispersion in local abundance
        xx = convolve(np.multiply(x, x), kernel, mode='same')
        xx[msk == 0] = 0

        # number of cells in kernel (local abundance)
        N = convolve(x, kernel, mode='same')
        N[msk == 0] = np.nan
        
        # calculates the MH univariate index
        NN = np.multiply(N, N)
        M = 2 * np.divide(NN, (A*xx + NN),
                          out=np.zeros(msk.shape),
                          where=(msk))
        
        # revert NAN values
        M[M > 1.0] = 1.0
        M[M < 0.0] = 0.0
        M[msk == 0] = np.nan
        
    return([msk, tofloat(N), tofloat(M)])


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

    from myfunctions import circle, tofloat, fftconv2d

    # produces a box-circle kernel
    circ = circle(np.ceil(r))

    # convolve array of cell locations with kernel
    # (ie. number of cells inside circle centered in each pixel)
    N = fftconv2d(cell_arr, circ, cuda=ISCUDA)

    # the typical area of a cell
    # (given by maximun number of cells in any circle)
    acell = 0
    if ( np.max(N) > 0):
        acell = np.sum(circ)/np.max(N)

    # returns radius of a typical cell
    return(tofloat(round(np.sqrt(acell/np.pi), 4)))


def getBlobs(data, mask):
    """
    Gets labels from blob regions mask, and assing them to the cell data

    Parameters
    ----------
    - data: (pandas df) TLA dataframe of cell coordinates
    - mask: (numpy) binary mask defining blob regions

    """

    from myfunctions import toint
    from skimage import measure

    # get a binary image from the (thresholded) mask
    msk_img = np.zeros(mask.shape, dtype='uint8')
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
    aux['row'] = toint(aux['row'])
    aux['col'] = toint(aux['col'])
    aux['blob'] = toint(aux['blob'])
    
    blobs_labels = toint(blobs_labels)

    return(aux, blobs_labels)


# %% Main function

def main(args):
    """
    *******  Main function  *******

    """
    # %% start, checks how the program was launched 
    debug = False
    try:
        args
    except NameError:
        #  if not running from the CLI, run in debug mode
        debug = True
    
    # tracks time and memory 
    start = time.time()
    mbuse = [np.nan] * 10

    if debug:
        # running from the IDE
        # path of directory containing this script
        main_pth = os.path.dirname(os.getcwd())
        argsfile = os.path.join(main_pth, 'DCIS.csv')
        REDO = True
        GRPH = True
        CASE = 106
    else:
        # running from the CLI using the bash script
        # path to working directory (above /scripts)
        main_pth = os.getcwd()
        argsfile = os.path.join(main_pth, args.argsfile) 
        REDO = args.redo
        GRPH = args.graph
        CASE = args.casenum
    
    print("==> The working directory is: " + main_pth)
    print("==> Is CUDA available: " + str(ISCUDA))
        
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
    
    mbuse[0] = memuse()
    if debug:
        print("==> STEP 1 - Memory used: " + str(mbuse[0]) + "[MB]")
    
    # if pre-processed files do not exist
    if  (REDO or 
         (not os.path.exists(sample.cell_data_file)) or
         (not os.path.exists(sample.classes_file)) or
         (not os.path.exists(sample.raster_folder))):
        
        print( msg + " >>> pre-processing..." )
       
        # %% STEP 2: loads and format coordinate data
        #sample.setup_data(sample.supbw)
        sample.setup_data(0)
        
        extent = [0, np.round(sample.imshape[1]*sample.scale, 2),
                  0, np.round(sample.imshape[0]*sample.scale, 2)]
        
        mbuse[1] = memuse()
        if debug:
            print("==> STEP 2 - Memory used: " + str(mbuse[0]) + "[MB]")

        # %% STEP 3: Filter cells according to density filters
        sample.filter_class(study)
        
        mbuse[2] = memuse()
        if debug:
            print("==> STEP 3 - Memory used: " + str(mbuse[2]) + "[MB]")
        
        # %% STEP 4: calculate a ROI mask for region with cells
        sample.roi_mask(REDO)
        
        mbuse[3] = memuse()
        if debug:
            plt.figure()
            sns.scatterplot(data=sample.cell_data, s=10,
                            x='x', y='y', hue='class')
            ax = plt.gca()
            ax.set_aspect('equal', adjustable='box')
            del ax
            
            plt.figure()
            plt.imshow(sample.roiarr, extent=extent)
            
            print("==> STEP 4 - Memory used: " + str(mbuse[3]) + "[MB]")
            
        # %% STEP 5: create raster images from density KDE profiles
        kdearr = sample.kde_mask(REDO)
        if debug:
            for c, clss in sample.classes.iterrows():
                plt.figure()
                plt.imshow(kdearr[:,:,c], extent=extent)
                
        mbuse[4] = memuse()
        if debug:
            print("==> STEP 5 - Memory used: " + str(mbuse[4]) + "[MB]")
        
        # %% STEP 6: create rasters for cell abundance and mixing profiles
        abuarr, mixarr = sample.abumix_mask(REDO, kdearr)
        del kdearr
        gc.collect()
        
        mbuse[5] = memuse()
        if debug:
            for c, clss in sample.classes.iterrows():
                plt.figure()
                plt.imshow(abuarr[:,:,c], extent=extent)
                plt.figure()
                plt.imshow(mixarr[:,:,c], extent=extent)
                
            print("==> STEP 6 - Memory used: " + str(mbuse[4]) + "[MB]")
        
        # %% STEP 7: calculates quadrat populations for coarse graining
        sample.quadrat_stats(abuarr, mixarr);
        del abuarr
        del mixarr
        gc.collect()
        
        mbuse[6] = memuse()
        if debug:
            print("==> STEP 7 - Memory used: " + str(mbuse[6]) + "[MB]")
        
        # %% STEP 8: calculate global spacial statistics
        sample.space_stats(REDO, study.dat_path)
        
        mbuse[7] = memuse()
        if debug:
            print("==> STEP 8 - Memory used: " + str(mbuse[7]) + "[MB]")
        
        # %% STEP 9: saves main data files
        sample.save_data()
        
        mbuse[8] = memuse()
        if debug:
            print("==> STEP 9 - Memory used: " + str(mbuse[8]) + "[MB]")
    
    # %% else
    else:
        # %% STEP 10: if sample is already pre-processed read data
        print(msg + " >>> loading data..." )
        sample.load_data()
        
        extent = [0, np.round(sample.imshape[1]*sample.scale, 2),
                  0, np.round(sample.imshape[0]*sample.scale, 2)]
        
        mbuse[8] = memuse()
        if debug:
            print("==> STEP 10 - Memory used: " + str(mbuse[8]) + "[MB]")
    
    # %% STEP 11: calculates general stats
    sample.general_stats()
    
    mbuse[9] = memuse()
    if debug:
        print("==> STEP 11 - Memory used: " + str(mbuse[9]) + "[MB]")

    # plots landscape data
    if (GRPH and sample.num_cells > 0):
        sample.plot_landscape_scatter(extent)
        sample.plot_landscape_props(extent)
        sample.plot_landscape_simple(0, extent)
        sample.plot_landscape_simple(1, extent)
        sample.plot_landscape_simple(2, extent)
        sample.plot_class_landscape_props(extent)
        
    # %% LAST step: saves study stats results for sample 
    
    memmax = np.nanmax(mbuse)
    trun = time.strftime('%H:%M:%S', time.gmtime(time.time()-start))
    print('==> TLA-Setup finished. Time elapsed: ', trun, '[HH:MM:SS]')
    print("==> Max memory used: " + str(memmax) + "[MB]")
    
    with open(study.done_list, 'a') as f:
        f.write(sample.sid + '\n')
    
    sample.output(study, trun, memmax)    
    
    # %% end
    return(0)

# %% Argument parser
if __name__ == "__main__":

    # Create the parser
    my_parser = ArgumentParser(prog="tla_setup_sample",
                               description="# Single Sample Pre-processing " + 
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
                           help="If <<--redo>> is used, then redo analysis")

    # passes arguments object to main
    main(my_parser.parse_args())
