""" 
    TLA setup single sample:
    #######################

    This process prepares and formats sample data for TLA.
    
    1- Reads parameters from a study set table (only the 1st line)
    2- Raw-data for a slide, given by case number, is read, extreme 
       values are calculated and data is formatted and prepared for TLA.
    3- The data is split into different ROIs, given in raster file 
       pointed in input table. If no ROI is given, it's assumed the slide 
       is one sample. In this case small ROI fractions will be dropped out.
    4- Each ROI corresponds to a sample, which will be processed individually.
    5- Cell classes are checked out and a new, curated, coordinate file is
       produced with a cropped version of the IHC image and raster arrays
       (at hoc region mask, a mask that defines the ROI, a multilevel
       raster array with kde cell density info and a multilevel raster array
       of cell abundance at different neigborhood scales).
    6- Some global spatial statistics and spatial fields required for TLA are 
       calculated
    7- Sample data summary and general stats are produced.
    
    NOTE: Aggregated statistics across samples should be calculated
    using 'TLA_setup_sum' after pre-processing all slides in the study.
"""

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
from argparse import ArgumentParser

from myfunctions import mkdirs, filexists

if torch.cuda.is_available():
    ISCUDA = True
else:
    ISCUDA = False

Image.MAX_IMAGE_PIXELS = 600000000

__version__ = "2.0.2"


# %% Private classes

class Progress:
    """Progress class"""
    
    def __init__(self, name, maxsteps=10, sub=False):
        
        self.label = name
        self.start = time.time()
        self.mbuse = [np.nan] * maxsteps
        self.step = 0
        self.mbuse[0] = memuse()
        
        if sub:
            self.arrow = "====> " 
        else:
            self.arrow = "==> " 
            
            
    def dostep(self, display, lab=''):
        
        self.runtime = time.time() - self.start
        self.step = self.step + 1
        self.mbuse[self.step] = memuse()

        if display:
            print(self.arrow + self.label + \
                  "; STEP: " + str(self.step) + "; " + lab +'\n'\
                  " - Run Time: " + ftime(self.runtime) + "[HH:MM:SS]" +\
                  " - Mem used: " + str(self.mbuse[self.step] ) + "[MB]")
            

class Study:
    """Study class"""

    def __init__(self, study, pth):

        from myfunctions import tofloat, toint

        # loads arguments for this study
        self.name = study['name']
        self.raw_path = os.path.join(pth, study['raw_path'])

        # loads samples table for this study
        f = filexists(os.path.join(self.raw_path, study['raw_samples_table']))
        
        self.samples = pd.read_csv(f)
        self.samples.fillna('', inplace=True)

        # sets path for processed data
        self.dat_path = mkdirs(os.path.join(pth, study['data_path']))

        # list of processed samples
        self.done_list = os.path.join(self.dat_path, 'setup_done_samples.csv')

        # scale parameters
        self.factor = 1.0
        if 'factor' in study:
            self.factor = study['factor']
        self.scale = tofloat(study['scale']/self.factor)
        self.units = study['units']

        # the size of quadrats (to pixels as multiple of 10) (long scale)
        aux = np.rint(10*np.ceil((study['binsiz']/self.scale)/10))
        self.binsiz = toint(aux)
        
        # bandwidth size for convolutions (short scale)
        self.bw = toint(np.rint(self.binsiz/20))

        # r values for Ripleys' H function
        dr = toint(np.rint(self.bw))
        self.rs = list(tofloat(np.arange(dr, self.binsiz + dr, dr)))
        # index of bw in rs array
        self.ridx = toint(np.abs(np.asarray(self.rs) - self.bw).argmin())

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

        # loads classes info
        f = filexists(os.path.join(self.raw_path, study['raw_classes_table']))
        classes = pd.read_csv(f)
        classes['class'] = classes['class'].astype(str)
        self.classes = classes

        # creates tables for output
        self.samples_out = pd.DataFrame()
        self.allstats_out = pd.DataFrame()
        self.allpops_out = pd.DataFrame()


class Slide:
    """Slide class."""

    def __init__(self, i, study):

        from myfunctions import mkdirs, toint

        # creates sample object
        self.tbl = study.samples.iloc[i].copy()  # table of parameters
        self.sid = self.tbl.sample_ID            # sample ID
        self.classes = study.classes             # cell classes
        self.msg = "==> Slide [" + str(i + 1) + \
              "/" + str(len(study.samples.index)) + \
              "] :: Slide ID <- " + self.sid + " >>> pre-processing..."
        
        # paths
        self.raw_path = study.raw_path
        self.dat_path = study.dat_path

        # raw data file
        f = os.path.join(self.raw_path, self.tbl.coord_file)
        self.raw_cell_data_file = filexists(f)
        
        # roi mask image
        self.roi_file = os.path.join(self.raw_path, self.tbl.roi_file)
        # makes sure the path for roi file exists
        _ = mkdirs(os.path.dirname(self.roi_file))
        
        # the size of quadrats and subquadrats
        self.binsiz = study.binsiz
        
        # bandwidth size for convolutions
        self.bw = study.bw

        # r values for Ripleys' H function
        self.rs = study.rs
        self.ridx = study.ridx

        # scale parameters
        self.factor = study.factor
        self.scale = study.scale
        self.units = study.units

        # other sample attributes (to be filled later)
        self.cell_data = pd.DataFrame()  # dataframe of cell coordinates
        self.imshape = []                # shape of landscape
        
        # if ROI should be split 
        self.split_rois = self.tbl.split_rois
        
        # sample arrays
        self.img = []                    # image array (slide image)
        self.msk = []                    # mask array  (blobs mask)
        self.roiarr = []                 # ROI array   (roi mask array)

        # slide image
        if (self.tbl.image_file == ''):
            self.raw_imfile = ''
            self.isimg = False
        else:
            aux = os.path.join(study.raw_path, self.tbl.image_file)
            self.raw_imfile = aux
            self.isimg = True

        # blob mask image
        if (self.tbl.mask_file == ''):
            self.raw_mkfile = ''
            self.ismk = False
        else:
            aux = os.path.join(study.raw_path, self.tbl.mask_file)
            self.raw_mkfile = aux
            self.ismk = True

        # total number of cells (after filtering)
        self.num_cells = toint(0)
        # list of section labels in ROI array
        self.roi_sections = []


    def setup_data(self, edge=0):
        """
        Load coordinates data, shift and convert values.
        Crop images to the corresponding convex hull.

        Args:
            - edge (int, optional):size [pix] of edge around data extremes 
             in rasters. Defaults to 0.
             
        Returns:
            None.

        """
        
        from skimage import io
        from skimage.transform import resize
        from myfunctions import toint, tobit

        cxy = pd.read_csv(self.raw_cell_data_file)[['class', 'x', 'y']]
        cxy['class'] = cxy['class'].astype(str)
        cxy = cxy.loc[cxy['class'].isin(self.classes['class'])]

        """
        ###################################################################
        # code to estimate the NN distances between all cells
        # NOTE: This part of the code is here to find a good factor value
        #       DO NOT UNCOMMENT! Use it in a good sample and edit in the
        #       the parameter file so its the same for all samples
        from scipy.spatial import KDTree
        rc = np.array(cxy[['x', 'y']])
        dnn, _ = KDTree(rc).query(rc, k=[2])
        dmin = np.min(dnn)
        if dmin > 2:
            # rescaling factor is such that min cell distance is sqrt(2)
            #(which is the max distance between adjcent cell in square grid)
            self.factor = np.sqrt(2)/dmin
        ####################################################################
        """
        
        # updates coordinae values by conversion factor (from pix to piy)
        cxy.x = toint(cxy.x*self.factor)
        cxy.y = toint(cxy.y*self.factor)
        
        """
        ####################################################################
        # code to estimate how much cell overlap after subsampling with factor
        data = cxy.copy()
        N = len(data)
        aux = data.groupby(['x','y', 'class'], as_index = False).size()
        # number of overlaping cells of each type in eah location
        df = aux.pivot(index=['x','y'], 
                        columns='class',
                        values = 'size').fillna(0).reset_index()
        # number of different cell types overlaping per location 
        df['ndif'] = (df.drop(['x','y'], axis=1) != 0).sum(axis=1)
        # total number of cells overlaping per location 
        df['ntot'] = df.drop(['x','y', 'ndif'], axis=1).sum(axis=1)
        # reduces to actual overlaping cases
        df = df[df['ntot']>1]
        if len(df) > 0:
            nmax = int(np.max(df['ntot']))
            dmax = int(np.max(df['ndif']))
            cts, bins, fig1 = plt.hist(df['ntot'], bins=range(2, nmax+2))
            cts = 100*np.multiply(cts, bins[:-1])/N
            cts2, bins, fig = plt.hist(df['ndif'], bins=range(1, dmax+2))
            cts2 = 100*np.multiply(cts2, bins[:-1])/N
        ####################################################################
        """
        
        # gets extreme pixel values
        xmin, xmax = toint(np.min(cxy.x)), toint(np.max(cxy.x))
        ymin, ymax = toint(np.min(cxy.y)), toint(np.max(cxy.y))
        imshape = [ymax + edge, xmax + edge]

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
        if ((self.isimg and self.ismk) and
            ((ims.shape[0] != msc.shape[0]) or
             (ims.shape[1] != msc.shape[1]))):
            print('\n =====> WARNING! sample_ID: ' + self.sid +
                  '; image and mask are NOT of the same size...' +
                  'adopting mask domain...')
            ims = tobit(np.rint(resize(ims,
                                       (msc.shape[0], msc.shape[1], 3),
                                       anti_aliasing=True,
                                       preserve_range=True)))

        # limits for image cropping
        rmin = toint(np.nanmax([0, ymin - edge]))
        cmin = toint(np.nanmax([0, xmin - edge]))
        rmax = ymax + edge
        cmax = xmax + edge

        dr = rmax - rmin
        dc = cmax - cmin
        if (np.isnan(dr) or np.isnan(dc) or (dr <= 0) or (dc <= 0)):
            print("ERROR: data file " + self.raw_cell_data_file +
                  " is an empty or invalid landscape!")
            sys.exit()
        imshape = [toint(dr + 1), toint(dc + 1)]

        # shifts coordinates
        cell_data = xyShift(cxy, imshape, [rmin, cmin], self.scale)

        # create croped versions of image and mask raster
        if self.isimg:
            img = np.zeros((imshape[0], imshape[1], 3), dtype='uint8')
            img[0:(rmax - rmin),
                0:(cmax - cmin), :] = tobit(ims[rmin:rmax,
                                                cmin:cmax, :])
            self.img = img
        else:
            self.img = []

        if self.ismk:
            msk = np.zeros(imshape, dtype='uint8')
            msk[0:(rmax - rmin),
                0:(cmax - cmin)] = tobit(msc[rmin:rmax, cmin:cmax])
            self.msk = (msk).astype('bool')
        else:
            self.msk = []

        self.cell_data = cell_data.reset_index(drop=True)
        self.imshape = [toint(imshape[0]), toint(imshape[1])]
        self.num_cells = toint(len(self.cell_data))


    def filter_class(self, study):
        """Apply class filter.

        - Parameters:
            - target_code: (str) cell class to run filter on (e.g. tumor cells)
            - hd_code: (str) code to assign to cells in high-density regions
            - ld_code: (str) code to assign to cells in low-density regions
            - denthr: (float) threshold to distinguish high and low density
              regions. If denthr=0 then no filtering in performed
            - blobs: (bool) if true, uses 'blob' label in data to mask cells.
        - If bloobs are used, cells outside of blobs are labeled with `ld_code` 
          and  those inside are labeled with `hd_code`.
        - If `denthr` > 0 then this value is used to determine low from high 
          density regions. 
        - Cells in low-density regions AND inside the blobs are labeled with 
         `ld_code`
        - BUT all cells outside the blobs are labeled with `hd_code` regardless
          of their density as compared with `denthr`.
        """
        from myfunctions import KDE, arrayLevelMask, tofloat, toint

        target_code = study.FILTER_CODE
        hd_code = study.HIGH_DENS_CODE
        ld_code = study.LOW_DENS_CODE
        denthr = tofloat(study.DTHRES)
        blobs = (self.ismk) and (study.BLOBS)

        data = self.cell_data.copy()
        data['orig_class'] = data['class']

        # redefine all target cells
        data.loc[(data['class'] == target_code), 'class'] = hd_code
        if blobs:
            # redefine possibly misclasified target cells using the external
            # mask filter (those inside mask blobs)
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

                # do KDE on pixel locations of target cells (e.g. tumor cells)
                z = KDE(aux, self.imshape, self.bw, cuda=ISCUDA)

                """############################################################
                # code to find a good 'denthr'
                denthr = 0.01
                zz = z.copy()
                zz[zz<=denthr] = 1e-6
                lz = np.log10(zz)
                plt.imshow(lz, cmap='jet')
                plt.figure()
                v = lz[lz>-6].ravel()
                plt.hist(v, bins = np.arange(-6, 1.1, .1), density=True)
                ############################################################"""
                
                # generate a binary mask
                mask = arrayLevelMask(z, denthr, self.bw, fill_holes=False)

                # tag masked out cells
                irc = toint(np.array(aux[['i', 'row', 'col']]))
                ii = np.ones(len(data), dtype='bool')
                ii[irc[:, 0]] = mask[irc[:, 1], irc[:, 2]]

                # redefine possibly misclasified target cells by means of
                # the KDE filter (i.e. those in low density regions)
                data.loc[np.logical_not(ii), 'class'] = ld_code
                data.drop(columns=['i'], inplace=True)

        # drop cells not in the approved class list
        self.cell_data = data.loc[data['class'].isin(self.classes['class'])]
  
        
    def roi_mask(self, redo):
        """Generate ROI."""
        from myfunctions import filterCells, toint

        fout = self.roi_file
        if not os.path.exists(fout):
            # NOTE: if roi mask file does not exist, assumes this is a single
            #       sample slide, and filters out small disconnected regions
            from myfunctions import kdeMask_rois
            # gets a mask for the region that has cells inside
            self.roiarr = kdeMask_rois(self.cell_data,
                                       self.imshape,
                                       self.bw,
                                       100000,
                                       split=False,
                                       cuda=ISCUDA)
            np.savez_compressed(fout, roi=self.roiarr)
        else:
            aux = np.load(fout)
            self.roiarr = toint(aux['roi'])

        # filter out cells outside of ROI
        # adds a 'mask' tag for cells in different regions
        self.cell_data = filterCells(self.cell_data, self.roiarr)

        # total number of cells (after filtering)
        self.num_cells = toint(len(self.cell_data))
        # list of roi sections
        self.roi_sections = np.unique(self.roiarr[self.roiarr > 0]).tolist()


class Sample:
    """Sample class."""

    def __init__(self, i, slide):

        from myfunctions import mkdirs, tofloat, toint
        
        roi = slide.roi_sections[i]

        # creates sample object
        self.sid = str(slide.sid) + "_roi-" + str(roi) # sample ID
        self.tbl = slide.tbl.copy()                    # table of parameters
        self.tbl.sample_ID = self.sid
        self.msg = "====> Sample [" + str(i + 1) + \
              "/" + str(len(slide.roi_sections)) + \
              "] :: SID <- " + self.sid + " >>> pre-processing..."
        
        classes = slide.classes.copy()                 # cell classes
        # drops classes that are excluded from study
        classes.drop(classes.loc[classes['drop']].index, inplace=True)
        classes.drop(columns=['drop'], inplace=True)
        classes.reset_index(inplace=True, drop=True)
        self.classes = classes

        # raw data files
        self.raw_cell_data_file = slide.raw_cell_data_file

        # the size of quadrats and subquadrats
        self.binsiz = slide.binsiz

        # bandwidth size for convolutions
        self.bw = slide.bw

        # r values for Ripleys' H function
        self.rs = slide.rs
        self.ridx = slide.ridx

        # scale parameters
        self.factor = slide.factor
        self.scale = slide.scale
        self.units = slide.units

        # other sample attributes
        aux = slide.cell_data.copy()
        auy = aux.loc[aux['class'].isin(self.classes['class'])]
        df = auy.loc[auy['mask'] == roi]
        self.cell_data = df                   # cell coordinates
        self.imshape = slide.imshape.copy()   # shape of accepted image

        # sample arrays
        self.img = slide.img.copy()            # image array
        self.msk = slide.msk.copy()            # mask array (blobs)
        
        aux = np.zeros_like(slide.roiarr, dtype='bool')
        aux[slide.roiarr == roi] = True
        self.roiarr = aux # ROI array
        
        # ploting limits (physical)
        self.ext = []

        # global spacial factors
        self.coloc = []
        self.nndist = []
        self.rhfunc = []

        # stats
        self.qstats = []                # quadrat statistics table
        self.mstats = []                # general statistics table

        # creates results folder and add path to sample tbl
        f = mkdirs(os.path.join(slide.dat_path,
                                'results', 'samples', self.sid))
        self.tbl['results_dir'] = 'results/samples/' + self.sid + '/'
        self.res_pth = f

        # creates cellPos folder and add path to sample tbl
        f = mkdirs(os.path.join(slide.dat_path, 'cellPos'))
        self.tbl['coord_file'] = 'cellPos/' + self.sid + '.csv'
        self.cell_data_file = os.path.join(f, self.sid + '.csv')

        # creates images folder and add path to sample tbl
        if (slide.isimg):
            f = mkdirs(os.path.join(slide.dat_path, 'images'))
            self.tbl['image_file'] = 'images/' + self.sid + '_img.jpg'
            self.imfile = os.path.join(f, self.sid + '_img.jpg')
            self.isimg = True
        else:
            self.imfile = ''
            self.isimg = False
            
        # creates raster folder and add path to df
        f = mkdirs(os.path.join(slide.dat_path, 'rasters', self.sid))
        if (slide.ismk):
            self.tbl['mask_file'] = 'rasters/' + self.sid + '/' + \
                self.sid + '_mask.npz'
            self.mask_file = os.path.join(f, self.sid + '_mask.npz')
            self.ismk = True
        else:
            self.mask_file = ''
            self.ismk = False
            
        # raster file names
        self.raster_folder = f
        self.roi_file = os.path.join(f, self.sid + '_roi.npz')
        self.kde_file = os.path.join(f, self.sid + '_kde.npz')
        self.nrs_file = os.path.join(f, self.sid + '_nrs.npz')
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
        reset coordinates data, shift and convert values.

        Args:
            edge (int): size [pix] of edge around data extremes in rasters

        Returns:
            None.
        
        """
        
        from skimage import io
        from myfunctions import toint, tobit

        # resets coordinate values for cropped out sample
        cxy = self.cell_data[['class', 'orig_class', 'col', 'row']]
        cxy.columns = ['class', 'orig_class', 'x', 'y']
        
        # gets extreme pixel values
        xmin, xmax = toint(np.min(cxy.x)), toint(np.max(cxy.x))
        ymin, ymax = toint(np.min(cxy.y)), toint(np.max(cxy.y))

        # limits for image cropping
        rmin = toint(np.nanmax([0, ymin - edge]))
        cmin = toint(np.nanmax([0, xmin - edge]))
        rmax = ymax + edge
        cmax = xmax + edge

        imshape = [toint(rmax - rmin), toint(cmax - cmin)]

        # shifts coordinates
        cell_data = xyShift(cxy, imshape, [rmin, cmin], self.scale)

        # create croped versions of image and mask raster
        if self.isimg:
            ims = self.img.copy()
            img = np.zeros((imshape[0], imshape[1], 3), dtype='uint8')
            img[0:(rmax - rmin),
                0:(cmax - cmin), :] = tobit(ims[rmin:rmax, cmin:cmax, :])
            self.img = img
            io.imsave(self.imfile, self.img, check_contrast=False)
        else:
            self.img = []

        if self.ismk:
            msc = self.msk.copy()
            msk = np.zeros(imshape, dtype='uint8')
            msk[0:(rmax - rmin),
                0:(cmax - cmin)] = tobit(msc[rmin:rmax, cmin:cmax])
            [cell_data, msk_img] = getBlobs(cell_data, msk)
            np.savez_compressed(self.mask_file, roi=msk_img)
            self.msk = (msk_img > 0).astype('bool')
        else:
            self.msk = []

        self.roiarr = self.roiarr[rmin:rmax, cmin:cmax]
        fout = self.roi_file
        np.savez_compressed(fout, roi=self.roiarr)
        
        self.cell_data = cell_data.reset_index(drop=True)
        self.imshape = [toint(imshape[0]), toint(imshape[1])]
        self.num_cells = toint(len(self.cell_data))
        
        # ploting limits (physical)
        self.ext = [0, np.round(self.imshape[1]*self.scale, 2),
                    0, np.round(self.imshape[0]*self.scale, 2)]


    def save_data(self):
        """Save main data files."""
        self.cell_data.to_csv(self.cell_data_file, index=False)
        self.classes.to_csv(self.classes_file, index=False)
        self.qstats.to_csv(os.path.join(self.res_pth,
                                        self.sid + '_quadrat_stats.csv'),
                           index=False, header=True)


    def load_data(self):
        """Load data."""
        from skimage import io
        from myfunctions import tofloat, toint, tobit

        # load all data (if it was previously created)
        self.cell_data = pd.read_csv(self.cell_data_file)
        self.cell_data['class'] = self.cell_data['class'].astype(str)
        self.cell_data['orig_class'] = self.cell_data['orig_class'].astype(str)
        self.classes = pd.read_csv(self.classes_file)
        self.classes['class'] = self.classes['class'].astype(str)

        aux = np.load(self.roi_file)
        self.roiarr = aux['roi'].astype('int64')

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

        if (self.isimg):
            self.img = tobit(io.imread(self.imfile))
        else:
            self.img = np.zeros((self.imshape[0],
                                 self.imshape[1], 3), dtype='uint8')

        if self.ismk:
            aux = np.load(self.mask_file)
            self.msk = aux['roi'].astype('int64')
        else:
            self.msk = np.zeros(self.imshape, dtype='bool')

        aux = np.load(self.kde_file)
        kdearr = tofloat(aux['kde'])
        del aux
                
        # estimate typical cell size based of density of points
        self.rcell = getCellSize(np.sum(kdearr, axis=2), self.binsiz)
        
        self.ext = [0, np.round(self.imshape[1]*self.scale, 2),
                    0, np.round(self.imshape[0]*self.scale, 2)]
        

    def output(self, study, trun, memmax):
        """Record output."""
        samples_out = self.tbl.to_frame().T
        samples_out = samples_out.astype(study.samples.dtypes)
        samples_out = samples_out.astype({'num_cells': 'int'})
        samples_out['setup_runtime'] = trun
        samples_out['setup_maxMBs'] = memmax
        samples_out['index'] = np.nan
        samples_out.to_csv(os.path.join(self.res_pth,
                                        self.sid + '_samples.csv'),
                           index=False, header=True)

        allstats_out = self.mstats.to_frame().T
        allstats_out.to_csv(os.path.join(self.res_pth,
                                         self.sid + '_samples_stats.csv'),
                            index=False, header=True)
  
        
    def kde_arrays(self, redo):
        """ calculate several raster arrays needed during the TLA
        
        NOTE: KDE is normalized to the total number of points, so it 
        represents a coarse grainned version of the data: number of points per
        pixel, taking the "points" as spreaded out with a Gaussian smoothing 
        opperation. So it replicates the distribution of points (continuously)
        On the other hand the raster of abundances KAE represents a number of 
        points withing a certain distance, and its NOT regularized. This is 
        calculated with a kernel of weights, with a maximun value of 1. It's a 
        practical way to calculate indicator function integrals. 
        If the KAE is done using a normalized kernel, it represents the local
        mean of point counts in each location, we call it nrKDE (which stands
        for non-regularized KDE). As the convolution is the running average
        of point abundance for each location. For these two rasters, there is 
        NO CONSISTENT representation of spatial distributions as the freq 
        normalization changes from point to point.
        
            - (kdearr): KDE raster at bw for each cell type (Gaussian kernel)
            - (nrsarr): raster of abundances for different r (KAE)
        """
        
        from myfunctions import toint, tofloat

        # Calculate cell fractions for each cell type
        classes = self.classes.copy()
                
        cla = self.classes['class']
        df = self.cell_data
        n = toint(len(df))
        
        classes['raster_index'] = classes.index
        classes['number_of_cells'] = toint(0)
        classes['fraction_of_total'] = tofloat(np.nan)

        if (n > 0):
            aux = []
            for i, c in enumerate(cla):
                aux = df.loc[df['class'] == c]
                classes.loc[classes['class'] == c,
                            'number_of_cells'] = toint(len(aux))
            f = np.around(classes['number_of_cells']/n, decimals=4)
            classes['fraction_of_total'] = tofloat(f)
    
        # update classes dataframe
        self.classes = classes

        # generate kde and nrs arrays for each cell type
        do_kde = redo or not os.path.exists(self.kde_file)
        do_nrs = redo or not os.path.exists(self.nrs_file)
        
        if do_kde:
            kdearr = tofloat(np.zeros((self.imshape[0],
                                       self.imshape[1],
                                       len(cla))))
        if do_nrs:
            nrsarr = tofloat(np.zeros((self.imshape[0],
                                       self.imshape[1],
                                       len(cla), len(self.rs))))
            nrsvals = tofloat(np.zeros((2, len(self.rs))))
            
        if do_kde or do_nrs :
            # precalculate kde and local abundances for all classes (and radii)
            for i, c in enumerate(cla):
                aux = df.loc[df['class'] == c]
                if (len(aux) > 0):
                    if do_kde:
                        from myfunctions import KDE
                        # do KDE on pixel locations of cells
                        z = KDE(aux, self.imshape, self.bw,  cuda=ISCUDA)
                        # set background to NA
                        z[self.roiarr == 0] = np.nan
                        kdearr[:, :, i] = z
                        del z
                    if do_nrs:
                        from myfunctions import gkern, fftconv2d, tobit
                        # get (strict) abuyndance around each location
                        X = tobit(np.zeros((self.imshape[0], self.imshape[1])))
                        X[aux.row, aux.col] = 1
                        for k, r in enumerate(self.rs):
                            # get number of cells in each neighborhood
                            kernel = gkern(r, normal=False)
                            nr = np.rint(fftconv2d(X, kernel, cuda=ISCUDA))
                            # regularize small values and sets bg to nan
                            nr[nr < np.min(kernel[kernel>0])] = 0
                            nr[self.roiarr == 0] = np.nan
                            nrsarr[:, :, i, k] = tofloat(nr)
                            nrsvals[0, k] = r
                            nrsvals[1, k] = np.sum(kernel)
                            del nr
                            gc.collect()
                        del X
                del aux
            # saves raster arrays
            if do_kde:
                np.savez_compressed(self.kde_file, kde=kdearr)
            if do_nrs:
                np.savez_compressed(self.nrs_file, nrs=nrsarr, rs=nrsvals)
                del nrsarr, nrsvals
        else:
            # if raster exists, load it
            aux = np.load(self.kde_file)
            kdearr = tofloat(aux['kde'])
            del aux

        # estimate typical cell size based of density of points
        self.rcell = getCellSize(np.sum(kdearr, axis=2), self.binsiz)

        # clean memory
        gc.collect()

        return kdearr


    def abumix_mask(self, kdearr, redo):
        """ABUMIX Mask.

        Generate rasters for abundance (N) and mixing (M) for cell locations
        for a (large) scale given by bandwith binsiz. 
        """
        from myfunctions import tofloat

        classes = self.classes.copy()
        classes['mean_abundance'] = tofloat(0.0)
        classes['std_abundance'] = tofloat(np.nan)
        classes['mean_mixing'] = tofloat(0.0)
        classes['std_mixing'] = tofloat(np.nan)
        
        cla = np.unique(classes['class'])

        fout = self.abumix_file
        if redo or not os.path.exists(fout):

            abuarr = tofloat(np.full((self.imshape[0],
                                      self.imshape[1],
                                      len(cla)), np.nan))
            mixarr = tofloat(np.full((self.imshape[0],
                                      self.imshape[1],
                                      len(cla)), np.nan))

            for i, c in enumerate(cla):
                # cell location array (smooth at short scale)
                x = (kdearr[:, :, i])
                
                if (np.nansum(x) > 0):
                    # get abumix arrays at long scale 
                    [N, M] = abumix(x, self.binsiz, cuda=ISCUDA)

                    # assign raster values
                    abuarr[:, :, i] = tofloat(N)
                    mixarr[:, :, i] = tofloat(M)
                    
                    n = np.nanmean(N)
                    if (n > 0):
                        classes.loc[classes['class'] == c,
                                    'mean_abundance'] = np.round(n, 4)
                        aux = np.round(np.nanstd(N), 6)
                        classes.loc[classes['class'] == c,
                                    'std_abundance'] = aux
                        aux = np.round(np.nanmean(M), 4)
                        classes.loc[classes['class'] == c,
                                    'mean_mixing'] = aux
                        aux = np.round(np.nanstd(M), 6)
                        classes.loc[classes['class'] == c,
                                    'std_mixing'] = aux
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

            # get basic stats for each cell class
            for i, c in enumerate(cla):
                
                # assign raster values
                N = abuarr[:, :, i]
                M = mixarr[:, :, i]
                
                n = np.nanmean(N)
                if (n > 0):
                    classes.loc[classes['class'] == c,
                                'mean_abundance'] = np.round(n, 4)
                    aux = np.round(np.nanstd(N), 6)
                    classes.loc[classes['class'] == c,
                                'std_abundance'] = aux
                    aux = np.round(np.nanmean(M), 4)
                    classes.loc[classes['class'] == c,
                                'mean_mixing'] = aux
                    aux = np.round(np.nanstd(M), 6)
                    classes.loc[classes['class'] == c,
                                'std_mixing'] = aux
                del N
                del M
            
        # updates classes df
        self.classes = classes

        # clean memory
        gc.collect()

        return [abuarr, mixarr]


    def quadrat_stats(self, abuarr, mixarr):
        """Quadrat Statistics.

        Gets a coarse grained representation of the sample based on quadrat
        estimation of cell abundances. This is, abundance and mixing in a
        discrete lattice of size binsiz, which is a reduced sample of the
        sample in order to define regions for the LME in the TLA analysis.
        """
        from myfunctions import toint, tofloat

        # define quadrats (only consider full quadrats)
        redges = toint(np.arange(self.binsiz/2, self.imshape[0], self.binsiz))
        cedges = toint(np.arange(self.binsiz/2, self.imshape[1], self.binsiz))

        rcs = [(r, c) for r in redges for c in cedges]

        # dataframe for population abundances
        pop = pd.DataFrame({'sample_ID': [],
                            'row': [],
                            'col': [],
                            'total': []})
        for rc in rcs:
            if self.roiarr[rc] > 0:
                n = np.sum(abuarr[rc[0], rc[1], :])
                if ~np.isnan(n):
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

        pop['row'] = toint(pop['row'])
        pop['col'] = toint(pop['col'])
        pop['total'] = tofloat(pop['total'])

        self.qstats = pop


    def space_stats(self, redo, dat_path):
        """Spacial Statistics.

        Calculate global statistics:

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
           This is asymetric and normalized by r (ie. H(r) = log(L(r)/r)) for 
           interpretation.
        
           - H > 0 indicates clustering of 'test' cells around 'ref' cells
           - H ~ 0 indicates random mixing between 'test' and 'ref' cells
           - H < 0 indicates dispersion of 'test' cells around 'ref' cells
        """
        from myfunctions import toint, tofloat

        fout = self.spafac_file
        if redo or not os.path.exists(fout):
            
            from myfunctions import nndist, attraction_T
            from myfunctions import ripleys_K

            data = self.cell_data.copy()

            # number of classes
            nc = len(self.classes)
            # landscape area
            A = toint(np.sum(self.roiarr))
            
            # raster array of cell abundance at different radii
            aux = np.load(self.nrs_file)
            n = tofloat(aux['nrs'])
            del aux

            # Colocalization index
            colocarr = tofloat(np.full((nc, nc), np.nan))

            # Nearest Neighbor Distance index
            nndistarr = tofloat(np.full((nc, nc), np.nan))
            
            # physical values of radii
            rsp = list(np.around(np.array(self.rs) * self.scale, decimals=2))
            df = pd.DataFrame({'sample_ID': [str(x) for x in self.rs],
                               'r': self.rs, 'r_physical': rsp})
            df['a'] = np.around(np.pi*df.r*df.r, decimals=2)
            df['a_physical'] = np.around(np.pi*df.r_physical*df.r_physical,
                                         decimals=2)

            # Attraction Function Score
            aefuncarr = tofloat(np.full((nc, nc, len(self.rs), 2), np.nan))
            aefuncarr[:, :, 0:len(self.rs), 0] = self.rs
            aefuncdf = df.copy()
            aefuncdf.sample_ID = self.sid

            # Ripley's H function
            rhfuncarr = tofloat(np.full((nc, nc, len(self.rs), 2), np.nan))
            rhfuncarr[:, :, 0:len(self.rs), 0] = self.rs
            rhfuncdf = df.copy()
            rhfuncdf.sample_ID = self.sid

            # loop thru all combinations of classes (pair-wise comparisons)
            for i, clsx in self.classes.iterrows():
                
                # class label
                cx = clsx['class']
                # coordinates of cells in class x
                aux = data.loc[data['class'] == cx]
                # density of points
                ptdens_i = len(aux)/A

                if (len(aux) > 0):
                    rcx = np.array(aux[['row', 'col']])
                    for k, r in enumerate(self.rs):
                        # Ripleys H score (identity)
                        npairs = len(rcx)*(len(rcx) - 1)
                        rk = ripleys_K(rcx, n[:, :, i, k], npairs, A)
                        if ~np.isnan(rk):
                            # original Ripley's H def (A = pi*r^2)
                            #rk = np.sqrt(rk/np.pi) - r
                            # area of a Gaussian is 2*pi*sigma^2
                            rk = np.log10(np.sqrt(rk/(2*np.pi))/r)
                        rhfuncarr[i, i, k, 1] = tofloat(rk)
                        del rk

                        # Attraction Enrichment Score (identity)
                        dy = n[:, :, i, k]/(2*np.pi*r*r)
                        tk = attraction_T(rcx, ptdens_i, dy)
                        aefuncarr[i, i, k, 1] = tk
                    gc.collect()

                    nam = cx + '_' + cx
                    rhfuncdf['H_' + nam] = np.round(rhfuncarr[i, i, :, 1], 4)
                    aefuncdf['T_' + nam] = aefuncarr[i, i, :, 1]

                    # Colocalization index (identity)
                    colocarr[i, i] = 1.0

                    # Nearest Neighbor Distance index (identity)
                    nndistarr[i, i] = 0.0

                    for j, clsy in self.classes.iterrows():
                        
                        # class label
                        cy = clsy['class']
                        # coordinates of cells in class y
                        auy = data.loc[data['class'] == cy]
                        # density of points
                        ptdens_j = len(auy)/A

                        if (len(auy) > 0):
                            rcy = toint(np.array(auy[['row', 'col']]))
                            if (i != j):
                                for k, r in enumerate(self.rs):
                                    # Ripleys H score (bivarite)
                                    npairs = len(rcx)*len(rcy)
                                    rk = ripleys_K(rcx, n[:, :, j, k], 
                                                   npairs, A)
                                    if ~np.isnan(rk):
                                        # original Ripley's H def (A = pi*r^2)
                                        #rk = np.sqrt(rk/np.pi) - r
                                        # area of a Gaussian is 2*pi*sigma^2
                                        rk = np.log10(np.sqrt(rk/(2*np.pi))/r)
                                    rhfuncarr[i, j, k, 1] = tofloat(rk)

                                    # Attraction Function Score (bivarite)
                                    dy = n[:, :, j, k]/(2*np.pi*r*r)
                                    tk = attraction_T(rcx, ptdens_j, dy)
                                    aefuncarr[i, j, k, 1] = tk

                                nam = cx + '_' + cy
                                v = np.round(rhfuncarr[i, j, :, 1], 4)
                                rhfuncdf['H_' + nam] = v
                                T = aefuncarr[i, j, :, 1]
                                aefuncdf['T_' + nam] = T

                                # Nearest Neighbor Distance index
                                nndistarr[i, j] = nndist(rcx, rcy)

                            if (i > j):
                                # MH index from quadrats sampling
                                cols = self.qstats.columns
                                if ((cx in cols) and (cy in cols)):
                                    qs = self.qstats[[cx, cy]].copy()
                                    qs = qs.dropna()
                                    qs['x'] = qs[cx] / qs[cx].sum()
                                    qs['y'] = qs[cy] / qs[cy].sum()
                                    xxyy = (qs['x']*qs['x']).sum() + \
                                           (qs['y']*qs['y']).sum()
                                    if (xxyy > 0):
                                        M = 2 * (qs['x']*qs['y']).sum()/xxyy
                                    else:
                                        M = np.nan
                                else:
                                    M = np.nan
                                colocarr[i, j] = tofloat(M)
                                colocarr[j, i] = tofloat(M)

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

            del rhfuncdf, aefuncdf, n
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
        """General Statistics.

        Aggregate general statistics for the sample.
        Prints out and outputs summary stats
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
        dnnqs = np.quantile(dnn, [0.0, 0.25, 0.5, 0.75, 1.0])
        del dnn
        gc.collect()

        stats = self.tbl[['sample_ID', 'num_cells']]
        stats['total_area'] = toint(self.imshape[0]*self.imshape[1])
        stats['ROI_area'] = A

        for i, row in self.classes.iterrows():
            c = row['class']
            n = row['number_of_cells']
            stats[c + '_num_cells'] = n

            if (A > 0):
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
            aux = np.round(self.aefunc[comp[0], comp[1], self.ridx, 1], 4)
            stats['aefunc_' + ca + '_' + cb] = tofloat(aux)

        # records overal Ripley's H score at pixel level
        comps = list(product(self.classes.index.values.tolist(), repeat=2))
        for comp in comps:
            ca = self.classes.iloc[comp[0]]['class']
            cb = self.classes.iloc[comp[1]]['class']
            aux = np.round(self.rhfunc[comp[0], comp[1], self.ridx, 1], 4)
            stats['rhfunc_' + ca + '_' + cb] = tofloat(aux)

        if (A > 0):
            tot_dens = N/A
            tot_dens_units = tofloat(np.round(N/roi_area, 4))
        else:
            tot_dens = np.nan
            tot_dens_units = tofloat(np.nan)

        # Save a reference to the standard output
        original_stdout = sys.stdout
        with open(os.path.join(self.res_pth,
                               self.sid + '_summary.txt'), 'w') as f:
            sys.stdout = f  # Change the standard output to file
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
            print('(*) Overall Attraction Enrichment Score T(' +
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
        """Plot Scatter Plots.

        Produces scatter plot of landscape based on cell coordinates, with
        colors assigned to each cell type
        Also generates individual scatter plots for each cell type
        """
        from myfunctions import plotEdges, landscapeScatter

        [ar, redges, cedges, xedges, yedges] = plotEdges(self.imshape,
                                                         self.binsiz,
                                                         self.scale)

        C, R = np.meshgrid(np.arange(0, self.imshape[1], 1),
                           np.arange(0, self.imshape[0], 1))
        grid = np.stack([R.ravel(), C.ravel()]).T
        x = (np.unique(grid[:, 1]))*self.scale
        y = (self.imshape[0] - (np.unique(grid[:, 0])))*self.scale
        
        classes = self.classes.iloc[::-1]

        fig, ax = plt.subplots(1, 1, figsize=(12, math.ceil(12/ar)),
                               facecolor='w', edgecolor='k')
        for i, row in classes.iterrows():
            aux = self.cell_data.loc[self.cell_data['class'] == row['class']]
            if (len(aux) > 0):
                landscapeScatter(ax, aux.x, aux.y,
                                 row.class_color, row.class_name,
                                 self.units, xedges, yedges,
                                 spoint=self.rcell,
                                 fontsiz=16, grid=False)
        ax.legend(labels=classes.class_name,
                  loc='upper left', bbox_to_anchor=(1, 1),
                  markerscale=5, fontsize=18, facecolor='w', edgecolor='k')
        ax.grid(which='major', linestyle='--',linewidth='0.3', color='black')
        ax.contour(x, y, self.roiarr, [.50], linewidths=2, colors='black')
        ax.set_xlim([lims[0], lims[1]])
        ax.set_ylim([lims[2], lims[3]])
        ax.set_aspect('equal', adjustable='box')
        ax.set_title('Sample ID: ' + str(self.sid), fontsize=22, y=1.04)
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
                                 spoint=self.rcell,
                                 fontsiz=16)
            ax[i].contour(x, y, self.roiarr, [.50], linewidths=2, 
                          colors='black')
            ax[i].set_title(row.class_name, fontsize=20, y=1.02)
            ax[i].set_xlim([lims[0], lims[1]])
            ax[i].set_ylim([lims[2], lims[3]])
            ax[i].set_aspect('equal', adjustable='box')
        plt.suptitle('Sample ID: ' + str(self.sid), fontsize=22, y=1.04)
        plt.tight_layout()
        plt.savefig(os.path.join(self.res_pth,
                                 self.sid + '_landscape_classes.png'),
                    bbox_inches='tight', dpi=300)
        plt.close()


    def plot_landscape_props(self, lims):
        """Produce general properties plot of landscape."""
        from myfunctions import plotEdges, KDE, plotRGB
        from myfunctions import landscapeScatter, landscapeLevels
        
        df = self.cell_data
        
        [ar, redges, cedges, xedges, yedges] = plotEdges(self.imshape,
                                                         self.binsiz,
                                                         self.scale)
        
        # get all cells KDE and contour levels
        z = KDE(df, self.imshape, self.binsiz/2, cuda=ISCUDA)
        z[self.roiarr==0] = np.nan
        
        gamma = np.ceil(np.log10(np.nanmax(z))) + 1
        delta = np.max((np.floor(np.log10(np.nanmin(z[z > 0]))),
                        gamma - 5))
        # linearly spaced contour levels
        #levs = np.linspace(10**delta, 10**gamma, 5)
        ## log spaced contour levels
        levs = 10**np.arange(delta, gamma)
        
        xv, yv = np.meshgrid(range(self.imshape[1]), range(self.imshape[0]))
        xv = xv*self.scale
        yv = (self.imshape[0] - yv)*self.scale

        classes = self.classes.iloc[::-1]

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            if (self.isimg or self.ismk):
                fig, ax = plt.subplots(2, 2,
                                       figsize=(12*2, 0.5 +
                                                math.ceil(12*2/ar)),
                                       facecolor='w', edgecolor='k')
                # plots sample image
                if (self.isimg):
                    plotRGB(ax[0, 0], self.img, self.units,
                            cedges, redges, xedges, yedges,
                            fontsiz=18)
                ax[0, 0].set_title('Histology image', fontsize=18, y=1.02)
                ax[0, 0].set_xlim([0, self.imshape[0]])
                ax[0, 0].set_ylim([0, self.imshape[1]])
                ax[0, 0].set_aspect('equal', adjustable='box')
                
                # plots sample image
                if (self.ismk):
                    plotRGB(ax[0, 1], 255*(self.msk > 0), self.units,
                            cedges, redges, xedges, yedges,
                            fontsiz=18, cmap='gray')
                ax[0, 1].set_title('Mask image', fontsize=18, y=1.02)
                ax[0, 1].set_xlim([0, self.imshape[0]])
                ax[0, 1].set_ylim([0, self.imshape[1]])
                ax[0, 1].set_aspect('equal', adjustable='box')

            else:
                 fig, ax = plt.subplots(1, 2,
                                        figsize=(12*2,
                                                 0.5 + math.ceil(12*1/ar)),
                                        facecolor='w', edgecolor='k')

            if (self.isimg or self.ismk):
                axi = (1, 0)
            else:
                axi = 0
                
            # plots sample scatter (with all cell classes)
            for i, row in classes.iterrows():
                aux = df.loc[df['class'] == row['class']]
                landscapeScatter(ax[axi], aux.x, aux.y,
                                 row.class_color, row.class_name,
                                 self.units, xedges, yedges,
                                 spoint=self.rcell, fontsiz=18, 
                                 grid=False)
            ax[axi].grid(which='major', linestyle='--',
                          linewidth='0.3', color='black')
            # plots roi contour over scatter
            ax[axi].contour(xv, yv, self.roiarr, [0.5],
                             linewidths=2, colors='black')
            ax[axi].set_title('Cell locations', fontsize=18, y=1.02)
            ax[axi].legend(labels=classes.class_name,
                            loc='best',
                            markerscale=3, fontsize=16,
                            facecolor='w', edgecolor='k')
            ax[axi].set_xlim([lims[0], lims[1]])
            ax[axi].set_ylim([lims[2], lims[3]])
            ax[axi].set_aspect('equal', adjustable='box')
            
            if (self.isimg or self.ismk):
                axi = (1, 1)
            else:
                axi = 1

            # plots kde levels
            landscapeLevels(ax[axi], xv, yv, z, levs,
                            self.units, xedges, yedges, fontsiz=18)
            ax[axi].contour(xv, yv, self.roiarr, [0.5],
                             linewidths=2, colors='black')
            ax[axi].set_title('KDE levels', fontsize=18, y=1.02)
            ax[axi].set_xlim([lims[0], lims[1]])
            ax[axi].set_ylim([lims[2], lims[3]])
            ax[axi].set_aspect('equal', adjustable='box')

            fig.subplots_adjust(hspace=0.4)
            fig.suptitle('Sample ID: ' + str(self.sid), fontsize=24, y=.95)
            fig.savefig(os.path.join(self.res_pth,
                                     self.sid + '_landscape.png'),
                        bbox_inches='tight', dpi=300)
            plt.close()


    def plot_landscape_simple(self, inx, lims):
        """Produce simple plot of landscape."""
        from myfunctions import KDE, landscapeLevels
        from myfunctions import landscapeScatter, plotEdges
        
        df = self.cell_data

        [ar, redges, cedges, xedges, yedges] = plotEdges(self.imshape,
                                                         self.binsiz,
                                                         self.scale)

        tclass = self.classes.iloc[inx]

        # gets kde profile of cell data
        aux = df.loc[df['class'] == tclass['class']]
        z = KDE(aux, self.imshape, self.binsiz/2, cuda=ISCUDA)
        #z[self.roiarr==0] = np.nan
        
        gamma = np.ceil(np.log10(np.nanmax(z))) + 1
        delta = np.max((np.floor(np.log10(np.nanmin(z[z > 0]))),
                        gamma - 5))
        # linearly spaced contour levels
        #levs = np.linspace(10**delta, 10**gamma, 10)
        ## log spaced contour levels
        levs = 10**np.arange(delta, gamma)

        xv, yv = np.meshgrid(range(self.imshape[1]), range(self.imshape[0]))
        xv = xv*self.scale
        yv = (self.imshape[0] - yv)*self.scale

        classes = self.classes.iloc[::-1]

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            fig, ax = plt.subplots(1, 3,
                                   figsize=(12*3, 0.5 + math.ceil(12/ar)),
                                   facecolor='w', edgecolor='k')

            # plots sample scatter (with all cell classes)
            
            for i, row in classes.iterrows():
                aux = df.loc[df['class'] == row['class']]
                landscapeScatter(ax[0], aux.x, aux.y,
                                 row.class_color, row.class_name,
                                 self.units, xedges, yedges,
                                 spoint=self.rcell, fontsiz=18, grid=False)
            ax[0].grid(which='major', linestyle='--',
                       linewidth='0.3', color='black')
            # plots roi contour 
            ax[0].contour(xv, yv, self.roiarr,
                          [.50], linewidths=2, colors='black')
            ax[0].set_title('All cells', fontsize=18, y=1.02)
            ax[0].legend(labels=classes.class_name,
                         loc='best',
                         markerscale=3, fontsize=18,
                         facecolor='w', edgecolor='k')
            ax[0].set_xlim([lims[0], lims[1]])
            ax[0].set_ylim([lims[2], lims[3]])
            ax[0].set_aspect('equal', adjustable='box')
            
            aux = df.loc[df['class'] == tclass['class']]
            landscapeScatter(ax[1], aux.x, aux.y,
                             tclass.class_color, tclass.class_name,
                             self.units, xedges, yedges,
                             spoint=self.rcell, fontsiz=18, grid=False)
            ax[1].grid(which='major', linestyle='--',
                       linewidth='0.3', color='black')
            # plots roi contour 
            ax[1].contour(xv, yv, self.roiarr,
                          [.50], linewidths=2, colors='black')
            ax[1].set_title(tclass['class_name'], 
                            fontsize=18, y=1.02)
            ax[1].set_xlim([lims[0], lims[1]])
            ax[1].set_ylim([lims[2], lims[3]])
            ax[1].set_aspect('equal', adjustable='box')
        
            # plots kde levels
            landscapeLevels(ax[2], xv, yv, z, levs,
                            self.units, xedges, yedges, fontsiz=18)
            # plots roi contour 
            ax[2].contour(xv, yv, self.roiarr,
                          [.50], linewidths=2, colors='black')
            ax[2].set_title(tclass['class_name'] + ' - KDE levels',
                            fontsize=18, y=1.02)
            ax[2].set_xlim([lims[0], lims[1]])
            ax[2].set_ylim([lims[2], lims[3]])
            ax[2].set_aspect('equal', adjustable='box')

            fig.subplots_adjust(hspace=0.4)
            fig.suptitle('Sample ID: ' + str(self.sid), fontsize=18, y=.95)
            nam = self.sid + '_' + tclass['class']
            fig.savefig(os.path.join(self.res_pth,
                                     nam + '_simple_landscape.png'),
                        bbox_inches='tight', dpi=300)
            plt.close()


    def plot_class_landscape_props(self):
        """Produce class properties plot of landscape."""
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
            vmin = np.floor(np.log10(np.quantile(abuarr[abuarr > 0], 0.1)))
            vmax = np.ceil(np.log10(np.quantile(abuarr[abuarr > 0], 0.9)))

            mixmin = np.floor(np.quantile(mixarr[mixarr > 0], 0.05))
            # mixmax = np.ceil(np.quantile(mixarr[mixarr > 0], 0.95))
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

                # plots mix image
                im = plotRGB(ax[i, 1], mixim, self.units,
                             cedges, redges, xedges, yedges,
                             fontsiz=18,
                             vmin=mixmin, vmax=mixmax, cmap='RdBu_r')
                plt.colorbar(im, ax=ax[i, 1], fraction=0.046, pad=0.04)
                ax[i, 1].set_title('Mixing image: ' + name,
                                   fontsize=18, y=1.02)

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
    """Memory use"""
    gc.collect()
    m = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
    return(round(m, 2))


def ftime(t):
    """Formated time string"""
    return(time.strftime('%H:%M:%S', time.gmtime(t)))
   

def xyShift(data, shape, ref, scale):
    """
    Shift coordinates and transforms into physical units.

    Args:
        - data (pandas): TLA dataframe of cell coordinates
        - shape (tuple): shape in pixels of TLA landscape
        - ref (tupe): reference location (upper-right corner)
        - scale (float): scale of physical units / pixel

    Returns:
        cell_data (pandas): data dable with shifted coordinates

    """
    from myfunctions import tofloat, toint

    cell_data = data.copy()

    # first drops duplicate entries
    # (shuffling first to keep a random copy of each duplicate)
    cell_data = cell_data.sample(frac=1.0).drop_duplicates(['x', 'y'])
    cell_data = cell_data.reset_index(drop=True)

    # generate a cell id
    # cell_data['cell_id'] = cell_data.index + 1

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

    return cell_data


def abumix(x, bw, cuda=False):
    """
    Calculates pixel resolution abundance and mixing profiles for \
    each cell type.
    
            N := sum{x_i*w_i}

    Mixing is scored by a univariate implementation of the Morisita-Horn
    index:

            M := 2*sum{x_i*y_i*w_i}/(sum{x_i*x_i*w_i} + sum{y_i*y_i*w_i})

    where {y_i} is a uniform distribution with the same volume as {x_i}:

            sum{y_i*w_i} = sum{x_i*w_i} = N and y_i = k = constant

    then k = N/A with A the area of integration

    The constant k is the mean density of elements in the integration region, 
    and thus M can be written as:

                    M  = 2 * N^2 / (A*sum{x_i*x_i*w_i} + N^2)

    This can be implemented spatially using a convolution function to get
    the local number of elements (N = sum{x_i*w_i}) and the dispersion
    (sum{x_i*x_i*w_i}) inside the kernel in each position, giving us a
    pixel-level resolution metric of the level of uniformity (i.e. mixing)
    of elements. So a general convolution with an arbitrary weights kernel 
    is a valid implementation.
    
    The argument 'raster' is a smoothed out evaluation of of the array of
    discrete locations (e.g. KDE) so values {x_i} are local abundances at 
    a short scale. The background is assumed to be NA.

    Args:
        - x (numpy array): cell locations array (or kdearr)
        - binsiz (int): bandwith scale for coarse grainning.
        - cuda (bool, optional): use cuda. Defaults to False.

    Returns:
        - list = [N, M]

    """

    from myfunctions import tocuda
    from myfunctions import gkern

    # produces a gaussian kernel weights for long scale coarse grainning
    kernel = gkern(bw, normal=False)

    # area of kernel
    A = np.sum(kernel)
    
    # sets background to zero
    bg = np.isnan(x) 
    x[bg]=0
    
    # small values
    delta = np.min(x[x>0])
    deltak = np.min(kernel[kernel>0])
    
    if cuda:
        from myfunctions import cudaconv2d

        # create tensor objects
        xs = tocuda(x)
        ks = tocuda(kernel)

        # dispersion in local abundance
        xx = cudaconv2d(torch.mul(xs, xs), ks)
        xx[xx < (delta*delta*deltak)] = 0

        # number of cells in kernel (local abundance)
        Nt = cudaconv2d(xs, ks)
        Nt[Nt < delta*deltak] = 0

        # calculates the MH univariate index
        NN = torch.mul(Nt, Nt)
        denom = torch.add(A*xx, NN)
        
        # calculates the MH univariate index
        mskt = (denom > 0)
        Mt = torch.full_like(xs, fill_value=np.nan)
        Mt[mskt] = 2 * NN[mskt] / denom[mskt]

        # return to CPU as numpy
        M = Mt.cpu().numpy()
        N = Nt.cpu().numpy()

        del xs, ks, mskt, xx, Nt, NN, denom, Mt
        torch.cuda.empty_cache()

    else:

        from scipy.signal import convolve

        # dispersion in local abundance
        xx = convolve(np.multiply(x, x), kernel, mode='same')
        xx[xx < (delta*delta*deltak)] = 0

        # number of cells in kernel (local abundance)
        N = convolve(x, kernel, mode='same')
        N[N < delta*deltak] = 0

        # calculates the MH univariate index
        NN = np.multiply(N, N)
        denom = (A*xx + NN)
        
        M = 2 * np.divide(NN, denom,
                          out=np.full(x.shape, np.nan),
                          where=(denom > 0))

    # reverts NA values in bg
    N[bg] = np.nan
    M[bg] = np.nan
    # M is not well defined when N==0
    M[N==0] = np.nan
    M[M<0] = np.nan
    M[M>1] = np.nan

    return [N, M]


def getCellSize(cell_arr, r):
    """Calculate Cell Size. Assuming a circular cell shape

    The typical size of a cell is based on the maximum number of cells found 
    in a circle of radious `r` (which is calculated using a convolution): 
    the typical area of a cell is estimated as:
          (area of circle) / (max number of cells in circle)

    Args:
        - cell_arr: (pumpy) array with cell locations, or densities
        - r: (float) radius of circular kernel
    """
    
    from myfunctions import circle, tofloat, fftconv2d
    
    arr =  cell_arr.copy()
    arr[np.isnan(arr)]=0

    # produces a box-circle kernel
    circ = circle(np.ceil(r))

    # convolve array of cell locations with kernel
    # (ie. number of cells inside circle centered in each pixel)
    N = fftconv2d(arr, circ, cuda=ISCUDA)

    # the typical area of a cell
    # (given by maximun number of cells in any circle)
    acell = 0
    if (np.max(N) > 0):
        acell = np.sum(circ)/np.max(N)

    # returns radius of a typical cell
    return tofloat(round(np.sqrt(acell/np.pi), 2))


def getBlobs(data, mask):
    """
    Get labels from blob regions mask, and assing them to the cell data.

    Args:
        data (pandas): TLA dataframe of cell coordinates
        mask (numpy): binary mask defining blob regions

    Returns:
        aux (pandas): dataframe with extra column for `blob`
        blobs_labels (numpy): array with blob labels

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

    return (aux, blobs_labels)


# %% Main function

def main(args):
    """TLA Setup Main."""
    
    # %% STEP 0: start, checks how the program was launched
    debug = False
    try:
        args
    except NameError:
        #  if not running from the CLI, run in debug mode
        debug = True

    if debug:
        # running from the IDE
        import seaborn as sns
        # path of directory containing this script
        main_pth = os.path.dirname(os.getcwd())
        f = os.path.join(main_pth, 'pathAI.csv')
        REDO = True
        GRPH = True
        CASE = 294      # case number to process
        
    else:
        # running from the CLI, (eg. using the bash script)
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

    # NOTE: only the FIRST line in the argument table will be used
    study = Study(pd.read_csv(argsfile).iloc[0], main_pth)
    
    
    # %% STEP 1: create slide object
    slide = Slide(CASE, study)
    print(slide.msg)
    
    # tracks time and memory usage
    progress = Progress(slide.sid)
    progress.dostep(debug, 'Slide object created')
       
    
    # %% STEP 2: loads and format coordinate data for slide
    slide.setup_data()
    slide.filter_class(study)
    
    if debug:
        plt.close('all')
        plt.figure()
        sns.scatterplot(data=slide.cell_data, s=1,
                        x='x', y='y', hue='class',
                        hue_order=slide.classes['class'])
        plt.legend(bbox_to_anchor=(1.02, 1), 
                   loc='upper left',
                   borderaxespad=0,
                   markerscale=5)
        plt.title("Slide ID: " + slide.sid)
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        del ax
        

    # %% STEP 3: load or calculate ROI mask for regions with cells. 
    #            ROIs are large unconnected sections of tissue 
   
    slide.roi_mask(REDO)
    
    if debug:
        plt.close('all')
        plt.figure()
        ext = [0, np.round(slide.imshape[1]*slide.scale, 2),
               0, np.round(slide.imshape[0]*slide.scale, 2)]
        plt.imshow(slide.roiarr, extent=ext)
        plt.title("Slide ID: " + slide.sid)
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        del ax

    progress.dostep(debug, 'ROIs loaded')
    
    # runtime up to here
    thead = progress.runtime
    print("==> ROI sections detected: " + str(slide.roi_sections))
    

    # %% STEP 4: for each ROI in the slide, create a new sample

    for i, roi in enumerate(slide.roi_sections):
        
        print("-------------------------------------------------")
        sample = Sample(i, slide)
        print(sample.msg)
        
        # tracks time and memory usage for this sample
        subprogress = Progress(sample.sid, sub=True)
        subprogress.dostep(debug, 'Sample object created')
    
        # if pre-processed files do not exist
        if (REDO or
                (not os.path.exists(sample.cell_data_file)) or
                (not os.path.exists(sample.classes_file)) or
                (not os.path.exists(sample.raster_folder))):
            
            # %% STEP 5: setup coordinate data (pick out sample data)
            sample.setup_data(0)
            
            if debug:
                plt.close('all')
                plt.figure()
                sns.scatterplot(data=sample.cell_data, s=1,
                                x='x', y='y', hue='class',
                                hue_order=sample.classes['class'])
                plt.legend(bbox_to_anchor=(1.02, 1), 
                           loc='upper left',
                           borderaxespad=0,
                           markerscale=5)
                plt.title("Sample_ID: " + sample.sid)
                ax = plt.gca()
                ax.set_aspect('equal', adjustable='box')
                del ax
                
                plt.figure()
                plt.imshow(sample.roiarr, extent=sample.ext)
                plt.title("ROI for Sample_ID: " + sample.sid)
                ax = plt.gca()
                ax.set_aspect('equal', adjustable='box')
                del ax
                
            subprogress.dostep(debug, 'Cell data formated')
                
            # %% STEP 6: raster images from density KDE profiles
            kdearr = sample.kde_arrays(REDO)
            
            if debug:
                plt.close('all')
                for c, clss in sample.classes.iterrows():
                    plt.figure()
                    plt.imshow(kdearr[:, :, c], extent=sample.ext)
                    plt.title("Sample_ID: " + sample.sid + \
                              "\nKDE for: " + clss['class_name'])
                    ax = plt.gca()
                    ax.set_aspect('equal', adjustable='box')
                    del ax
                    
            subprogress.dostep(debug, 'KDE calculated')


            # %% STEP 7: rasters for cell abundance and mixing profiles
            abuarr, mixarr = sample.abumix_mask(kdearr, REDO)
            del kdearr
            gc.collect()
            
            if debug:
                plt.close('all')
                for c, clss in sample.classes.iterrows():
                    plt.figure()
                    plt.imshow(abuarr[:, :, c], extent=sample.ext)
                    plt.title("Abundance for: " + clss['class_name'])
                    plt.figure()
                    plt.imshow(mixarr[:, :, c], extent=sample.ext)
                    plt.title("Mixing for: " + clss['class_name'])
                    
            subprogress.dostep(debug, 'ABU-MIX calculated')
    

            # %% STEP 8: calculates quadrat populations for coarse graining
            sample.quadrat_stats(abuarr, mixarr)
            del abuarr
            del mixarr
            gc.collect()
            subprogress.dostep(debug, 'Quadrat stats calculated')


            # %% STEP 9: calculate global spacial statistics
            sample.space_stats(REDO, study.dat_path)
            subprogress.dostep(debug, 'Space stats calculated')
            

            # %% STEP 10: saves main data files
            sample.save_data()
            subprogress.dostep(debug, 'Sample saved')
            

        # %% else
        else:
            # %% STEP 11: if sample is already pre-processed read data
            print(sample.msg + " >>> loading data...")
            sample.load_data()
            subprogress.dostep(debug, 'Sample loaded')
            
        
        # %% STEP 12: calculates general stats
        sample.general_stats()
    
        # plots landscape data
        if (GRPH and sample.num_cells > 0):

            sample.plot_landscape_scatter(sample.ext)
            sample.plot_landscape_props(sample.ext)
            
            sample.plot_landscape_simple(0, sample.ext)
            sample.plot_landscape_simple(1, sample.ext)
            sample.plot_landscape_simple(2, sample.ext)
                
            sample.plot_class_landscape_props()
                            
        subprogress.dostep(debug, 'General stats calculated')
        

        # %% LAST step: saves study stats results for sample
        memmax = np.max((np.nanmax(subprogress.mbuse), 
                         np.nanmax(progress.mbuse)))
        trun = ftime(thead + subprogress.runtime)
        print('====> Sample: ' + sample.sid + ' finished. Time elapsed: ', 
              trun, '[HH:MM:SS]')
        print("====> Max memory used: " + str(memmax) + "[MB]")
    
        with open(study.done_list, 'a') as f:
            f.write(sample.sid + '\n')
    
        sample.output(study, trun, memmax)
    
    # %% no ROI        
    if (np.sum(slide.roiarr)==0):
        print("WARNING: Slide <" + slide.sid + "> doesn't have any valid ROIs")
        sys.exit()
        
    progress.dostep(False)
    trun = ftime(progress.runtime)
    print("-------------------------------------------------")
    print('==> TLA-setup finished. Total time elapsed: ', trun, '[HH:MM:SS]')
    plt.close('all')
    
    # %% end
    return (0)


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
