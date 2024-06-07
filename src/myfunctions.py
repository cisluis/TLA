'''
     TLA general functions

'''

import os
import sys
import math
import pandas as pd
import numpy as np
import matplotlib as mpl
import torch

__version__  = "2.0.2"

###############################################################################
# %% Misc functions

def filexists(f):
    """
    Check if file exists, if it doesn't exit with an error.

    Args:
        -f (str): full file path and name

    Returns:
        - string: returns same value of 'f' if file exists

    """
    if not os.path.exists(f):
        print("ERROR: file <" + f +"> does not exist!")
        sys.exit()
    return(f)


def mkdirs(path):
    """
    Create a folder if it doesn't exist already.

    Args:
        - path (str): folder path

    Returns:
        - string: returns same value of 'path'

    """
    if not os.path.exists(path):
        # try-except in case a parallel process is creating the same folder
        try:
            os.makedirs(path)
        except OSError:
            pass
    return(path)


def tofloat(x):
    """
    Convert numpy object to float64

    Args:
        - x: numpy object

    Returns:
        - converted numpy object.

    """
    return(np.float64(x))


def toint(x):
    """
    Convert numpy object to int64

    Args:
        - x: numpy object

    Returns:
        - converted numpy object.

    """
    return(np.int64(x))


def tobit(x):
    """
    Convert numpy object to uint8

    Args:
        - x: numpy object

    Returns:
        - converted numpy object.

    """
    return(np.uint8(x))


def tocuda(x):
    """
    Convert numpy object to pytorch tensor

    Args:
        - x: numpy object

    Returns:
        - pytorch tensor

    """
    return(torch.from_numpy(x).type(torch.float64).cuda())



###############################################################################
# %% Array functions

def arrayLevelMask(z, th, rmin, fill_holes=True):
    """
    Return a binary mask: True for values above threshold, False otherwise

    Args:
        - z (numpy): array of float values
        - th (float): threshold value; mask = (z > th)
        - rmin (float): regions with scale r < rmin are removed
        - fill_holes (bool, optional): holes are filled out. Defaults to True.

    Returns:
        - mask (numpy): boolean array.

    """
    from skimage.morphology import remove_small_objects
    from scipy.ndimage import binary_fill_holes

    msk = (z > th).astype('int32')
    aux = remove_small_objects(msk,
                               min_size=np.pi*(rmin)*(rmin),
                               connectivity=2)
    if fill_holes:
        aux = binary_fill_holes(aux)

    return(aux > 0)


def filterCells(data, mask):
    """
    Filter out points from the data table according to a binary mask.
    Returns a reduced dataframe, with an additional columm `mask` indicating
    the value of the mask label correspondint to each point.

    Args:
        - data: (pandas) TLA dataframe of cell coordinates
        - mask: (numpy) Labeled mask (eg. a ROI binary mask)

    Returns:
        - (numpy) boolean array.

    """
    
    # load coordinates to a numpy array
    aux = data.copy().reset_index(drop=True)
    aux['i'] = aux.index
    irc = toint(np.array(aux[['i', 'row', 'col']]))

    # label cells in data that are in each mask region
    ii = np.zeros(len(aux), dtype='int64')
    ii[irc[:, 0]] = mask[irc[:, 1], irc[:, 2]]
    aux['mask'] = ii
    
    # drop index column
    aux.drop(columns=['i'], inplace=True)

    # drop cells outside of masks
    aux.drop(aux.loc[aux['mask']==0].index, inplace=True)
    
    return(aux.reset_index(drop=True))


###############################################################################
# %% Convolution functions

def circle(rin, normal=False):
    """
    Disc kernel of radius 'r'. This is a 2D radial heavyside function H_2(r).
    If the kernel is normalized then its total area is A = np.sum(kernel) = 1.
    Normalization is of use when the convolution is used to calculate a local 
    mean, equivalent to a running average.
    If NOT normalized it gives a Kernel Abundance Estimator (KAE), which is the
    sum of points inside each neigborhood defined by the kernel

    Args:
        - rin (int): radius of disc
        - normal (bool, optional): is normalized. Defaults to False.

    Returns:
        - (numpy array) kernel array.

    """

    r = int(rin)
    y, x = np.ogrid[-r: r + 1, -r: r + 1]
    
    kernel = 1.0*(x**2 + y**2 < r**2)
    
    if normal:
        kernel = kernel / np.sum(kernel)
        
    return(tofloat(kernel))


def gkern(sig, normal = True):
    """
    Gaussian kernel with sigma =`sig` (up to 5*sigma tail)
    If the kernel is normalized then its total area is A = np.sum(kernel) = 1.
    Normalization is of use when the convolution is used to calculate a local 
    (weighted) mean, equivalent to a running average. This is the common use.
    If NOT normalized it gives a Kernel Abundance Estimator (KAE), which is the
    sum of points inside each neigborhood defined by the (weighted) kernel

    Args:
        - sig (float): sigma of Gaussian kernel
        - normal (bool, optional): is normalized. Defaults to True.

    Returns:
        (numpy array) kernel array.

    """
    
    l = int(5.0*sig)
    if l % 2 == 0:
        # makes sure the kernel has a single center point
        l = l + 1    
    aux = np.arange(-(l - 1) / 2., ((l - 1) / 2.) + 1)
    gauss = np.exp(-0.5 * np.square(aux) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    kernel[kernel<0] = 0
    
    if normal:
        kernel = kernel / np.sum(kernel)
        
    return (tofloat(kernel))


def cuda_complex_product(x, y):
    """
    Fast complex multiplication between two complex 2D arrays (cuda tensors) 
    using the Karatsuba multiplication method.
    
    Refs:    
        - https://stackoverflow.com/questions/19621686/complex-numbers-product-using-only-three-multiplications
        - https://en.wikipedia.org/wiki/Karatsuba_algorithm
    
    Args:
        - x (cuda array): pytorch tensor with complex values
        - y (cuda array): pytorch tensor with complex values
    
    Returns:
        - (cuda array): x*y = pytorch tensor of complex values

    """
    # Extract the real and imaginary parts
    a, b = x.real, x.imag
    c, d = y.real, y.imag

    # term by term products
    ac = torch.mul(a, c)
    bd = torch.mul(b, d)
    ab_cd = torch.mul(torch.add(a, b), torch.add(c, d))
    
    return torch.complex(ac - bd, ab_cd - ac - bd)


def cudaconv2d(imgs, filts):
    """
    Calculate 2D FFT convolution using GPU with CUDA 

    Args:
        -imgs (cuda array): original image
        -filts (cuda array): kernel image

    Returns:
        -cuda array: convoluted image

    """
    
    from torch.fft import fft2, ifft2
    from torch.nn.functional import pad
    
    imgsshape = imgs.shape
    filtsshape = filts.shape

    # Pad and transform the image and filter
    # Pad arg = (last dim pad left side, last dim pad right side, 
    #            2nd last dim left side, etc..)
    f_imgs = fft2(pad(imgs, (0, filtsshape[1] - 1, 0, filtsshape[0] - 1)))
    f_filts = fft2(pad(filts, (0, imgsshape[1] - 1, 0, imgsshape[0] - 1)))
    
    # element wise complex multiplication and reverse fourier transform
    img = ifft2(cuda_complex_product(f_imgs, f_filts)).real

    # get non-padded part of arrays
    dc = int(filtsshape[1]/2)
    dr = int(filtsshape[0]/2)
    
    return img[dr:(imgsshape[0] + dr), dc:(imgsshape[1] + dc)]


def fftconv2d(img, filt, cuda=False):
    """
    Fast Fourier Transform 2D Convolution, with option to use CUDA

    Args:
        - img (np array): original image to convolute
        - filt (np array): kernel filter to convolute with
        - cuda (bool, optional): use cuda GPU processing. Defaults to False.

    Returns:
        - np array: Convoluted (smoothed) image

    """
    
    if cuda:
        # if GPU is available, uses pytorch cuda
        
        # create tensor objects
        imgs = tocuda(img)
        filts = tocuda(filt)
        
        # does fft convolution in cuda
        aux = cudaconv2d(imgs, filts)
        
        # return to CPU as numpy
        imgout = aux.cpu().numpy()
        
        del imgs, filts, aux
        torch.cuda.empty_cache()
        
    else:
        from scipy.signal import fftconvolve
        
        # if GPU not available, uses scipy (faster than pytorch in CPU)
        imgout = fftconvolve(img, filt, mode='same')
        
    return tofloat(imgout)


def KDE(data, shape, bw, cuda=False):
    """
    Evaluate a REGULARIZED KDE using a convolution with a Fast Fourier 
    Transform (FFT) using array convolution 'fftconvolve' from scipy.signal 
    (in CPU) or torch (in GPU).The KDE is normalized such that the total sum 
    equals the total number of points in the data, so the output represents a 
    (fractional) number of points per pixel in each pixel location.
    
    NOTE: This is NOT the KDE as the abundance density, which is the 
    number of cells in a kernel divided by the area of the kernel.

    Args:
        - data: (pandas df) TLA dataframe of cell coordinates
        - shape: (tuple) shape in pixels of TLA landscape
        - bw : (float) bandwidth; std dev of the KDE (gaussian) kernel
        - cuda (bool, optional): use cuda GPU processing. Defaults to False.

    Returns:
        - (numpy array) array of floats for normalized KDE

    """
    
    # generate Gaussian kernel
    kern = gkern(bw)
    # generate raster array of point locations
    arr = tofloat(np.zeros((shape[0], shape[1])))
    coords = np.array(data[['row', 'col']])
    arr[coords[:,0], coords[:,1]] = 1.0
    # Compute the kernel density estimate
    z = fftconv2d(arr, kern, cuda=cuda)
    # regularize negative and small values
    z[z < np.min(kern)] = 0
    # re-normalizes the output (to adjust for zeroed values)
    # output represents points per pixel, with total sum of points conserved
    z = len(data)*(z/np.sum(z))
    
    return(z)


def kdeMask(data, shape, bw, cuda=False):
    """
    Calculate a pixel resolution KDE profile from cell location data and
    generates a mask for the bakground region (where no cells are found).

    Args:
        - data (pandas df): TLA dataframe of cell coordinates
        - shape (tuple): shape in pixels of TLA landscape
        - bw (float): bandwidth; standard deviation of the gaussian kernel
        - cuda (TYPE, bool): use cuda for GPU processing. Defaults to False.

    """
    
    if (len(data) > 0):
        # Compute the KDE and levels
        z = KDE(data, shape, bw, cuda=cuda)
        # get threshold for background
        th = np.nanmin(z[z > 0])
        m = arrayLevelMask(z, th, bw, fill_holes=True)
    else:
        z = tofloat(np.zeros(shape))
        m = np.zeros(shape, dtype='bool')
    return(z, m)


def kdeMask_rois(data, shape, bw, minsiz, split=False, cuda=False):
    """
    Calculates a pixel resolution KDE profile from cell location data and
    generates a mask to contrast regions with cells versus the bakground.
    Mask is an array of ROI labels (integers). Background has label = 0

    Args:
        - data (pandas df): TLA dataframe of cell coordinates
        - shape (tuple): shape in pixels of TLA landscape
        - bw (float): bandwidth; standard deviation of the gaussian kernel
        - minsiz (int): minimum size of accepted ROI region
        - split (TYPE, bool): whether ROI would be split. Defaults to False.
        - cuda (TYPE, bool): use cuda for GPU processing. Defaults to False.

    Returns:
        - (numpy array) array of integers for roi region labels

    """

    if (len(data) > 0):
       
        from skimage.measure import label, regionprops 
        
        # Compute the kde and region where z>0
        z, m = kdeMask(data, shape, bw, cuda)
        # create a label mask for different sections in the sample
        msk = label(m).astype('int64')
        for region in regionprops(msk):
            lab = (msk==region.label)
            if (region.area < minsiz):
                # drops sections that are too small
                msk[lab] = 0
        if split:
            mask = msk.copy()
        else:
            mask = (msk>0).astype('int64')
        
    else:
        mask = np.zeros(shape)

    return(mask)


###############################################################################
# %% Spatial Statistics

def ripleys_K(rc, n, npairs, A):
    """Ripley's K function K(r). Global metric. 
    
    Using whole ensamble of points to calculate the background point density
    (null hypothesis). A is the area of the landscape where points are located.
    
    Calculated by using a KAE array 'n' of local abundances for each location 
    (NOTE: this is NOT the normalized KDE).
    This is a convenient way to implement the Indicator function I(r) integral,
    which is 'I(x,y) = n(x,y) - 1' (central point is substracted).
    The size of the neighborhood is given by the scale of the kernel at which 
    'n' was calculated.
    
        - K ~ pi*r^2 indicates uniform distribution of points
        - K > pi*r^2 indicates clumping of points
        - K < pi*r^2 indicates dispersion of points (typically periodic)
    
    A weighted sum could be used to calculate 'n', for instance using a 
    Gaussian kernel. In this case the interpretation of K will change as the 
    expected value is now the area under the Gaussian (2*pi*sig^2), 
    instead of the area of the circle (pi*r^2).
    
    NOTE: if 'rc' are coordinates of points of different type as the ones in
          'n' this statistics is bivariate. Just adjust the 'npairs' 
          accordingly:
              - Univariate case: npairs = N*(N-1)/2
              - Bi-variate case: npairs = Nx*Ny
        
    Args:
        - rc (numpy): array of 'ref' coordinates (Nx2)
        - n (numpy): 2D array with test point local abundance (KAE)
        - npairs (int): number of pair point comparisons
        - A (float): total area of the landscape.

    Returns:
        (float) Ripley's K(r) value 

    """

    ripley = np.nan
    
    if (len(rc) > 1):
        # number of neighbors in kernel around each point
        # (Indicator function integral)
        Ir = np.zeros((n.shape[0], n.shape[1]))
        Ir[rc[:, 0], rc[:, 1]] = n[rc[:, 0], rc[:, 1]] - 1
        
        # Ripley's K sum (do sum of Ir for each kernel)
        ripley = A * np.sum(Ir)/npairs
        
        if ripley < 0:
            ripley = np.nan
    
    return(tofloat(ripley))


def ripleys_K_array(rc, n, npairs, kernel, roi, cuda=False):
    """ Ripley's K function K(r). Local metric.
    
    Using local ensamble of points to calculate the background point density
    (null hypothesis) at each location. Thus, local metric values are estimated
    by using a local (long-scale) background, defined in the KAE array N.
    
    Calculated by using a KAE array 'n' of local abundances for each location.
    This is a convenient way to implement the Indicator function I(r) integral,
    which is 'I(x,y) = n(x,y) - 1' (central point substracted).
        
        - K ~ pi*r^2 indicates uniform distribution of points
        - K > pi*r^2 indicates clumping of points
        - K < pi*r^2 indicates dispersion of points (typically periodic)
    
    A weigted sum could be used to calculate 'n', for instance using a 
    Gaussian kernel. In this case the interpretation will change as the 
    expected value is now the area under the Gaussian (2*pi*sig^2), 
    instead of the area of the circle (pi*r^2)
    
    Args:
        - rc (numpy): array of coordinates
        - n (numpy): 2D array with point abundance at short scale (r)
        - npairs (numpy): 2D array with number of pairs at long scale (R)
        - kernel (numpy): kernel array for convolution smoothing (long scale)
        - roi (numpy): ROI mask
        - cuda (bool, optional): use cuda GPU processing. Defaults to False.

    NOTE: if 'rc' are coordinates of points of different type as the ones in
          'n' this statistics is bivariate. Just adjust the 'npoits' 
          accordingly:
              - Univariate case: npoints = N*(N-1)/2
              - Bi-variate case: npoints = Nx*Ny
              
    Returns:
        (numpy) Array with Ripley's K(r) values in each pixel 

    """
 
    ripley = np.ones(n.shape)*np.nan
    
    if (len(rc) > 1):
        # number of neighbors (at subkernel scale) around each point
        Ir = np.zeros((n.shape[0], n.shape[1]))
        Ir[rc[:, 0], rc[:, 1]] = n[rc[:, 0], rc[:, 1]] - 1
        
        # local parameters
        delta = np.min(kernel[kernel>0])
        A = np.sum(kernel)
    
        if cuda:
            # transfer arrays to GPU
            Nps = tocuda(npairs)
            Irs = tocuda(Ir)
            ks = tocuda(kernel)
            dtf = Nps > 0
        
            # Ripley's K sum (do sum of Ir for each kernel)
            aux = cudaconv2d(Irs, ks)
            aux[aux < delta]=0
            ripley = torch.full_like(Nps, fill_value=np.nan)
            ripley[dtf] = A * aux[dtf] / npairs[dtf] 
            ripley = ripley.cpu().numpy()
            
            del Nps, Irs, ks, dtf, aux
            torch.cuda.empty_cache()
            
        else:      
            
            from scipy.signal import fftconvolve
            
            dtf = npairs > 0
        
            # Ripley's K sum (do sum of Ir for each kernel)
            aux = fftconvolve(Ir, kernel, mode='same')
            aux[aux < delta]=0
            ripley = np.full_like(npairs, fill_value=np.nan)
            ripley[dtf] = A * aux[dtf] / npairs[dtf] 

        ripley[~roi] = np.nan
        ripley[ripley<=0] = np.nan
        
    return(tofloat(ripley))


def attraction_T(rcx, lambday, dy):
    """Attraction Enrichment Function score T. Global 
    
    A t-test statistic comparing the mean density of 'test' points  around 
    'ref' cells to the mean density of 'test' points in the whole region. 
    This is somewhat similar to Ripley's K statistic, implemented as a test 
    similar to Getis-Ord G*, as a bivariate statistics.
    
    NOTE: Ripley's K tests specifically for Complete Spatial Randomness (CSR), 
    while this statistic is less strong and only test whether 'test' points 
    tend to aggregate around 'ref' points in a statistical significant way as
    compared with the global point density (null hypothesis).
    
    Output is a significance score, showing whether or not the mean 
    local densities are above, bellow or about as expected.

        - T = 0 indicates uniform mixing between 'test' and 'ref' cells
        - T = +1 indicates clustering of 'test' cells around 'ref' cells
             (i.e. more attraction than 'ref' cells around 'ref' cells)
        - T = -1 indicates dispersion of 'test' cells around 'ref' cells
             (i.e. less attraction than 'ref' cells around 'ref' cells)

    This is an array form, calculated by using an KDE-type array 'dy' of 
    local densities for each location.
    
    Args:
        - rcx (numpy): array of 'ref' coordinates (sample points)
        - lambday (float): total density of 'test' points
        - dy (numpy): nrKDE of 'test' cells (abundance density)

    Returns:
        (float) value of T

    """
    from scipy import stats
    import warnings
    
    out = np.nan
        
    if (len(rcx) > 0):
        # local densities of 'test' cells around all 'ref' cells
        aux = dy[rcx[:, 0], rcx[:, 1]].ravel()
        # p is the probability of the t-test with the null 
        # hypothesis that the two point distributions are equivalent
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = stats.ttest_1samp(aux, popmean=lambday, nan_policy='omit')
        t = res.statistic
        p = res.pvalue
        out = tofloat(np.sign(t)*(p < 0.05))
        
    return(out)


def attraction_T_array(rcx, dy, Ny, kernel, roi, cuda=False):
    """Attraction Enrichment Functions score T, local (using convolution)
    
    A t-test statistic comparing the mean density of 'test' points  around 
    'ref' cells to the mean density of 'test' points in a long-scale region. 
    This is somewhat similar to Ripley's K statistic, implemented as a test 
    similar to Getis-Ord G*, as a bivariate statistics.
    
    NOTE: Ripley's K tests specifically for Complete Spatial Randomness (CSR), 
    while this statistic is less strong and only test whether 'test' points 
    tend to aggregate around 'ref' points in a statistical significant way as
    compared with the global point density (null hypothesis).
    
    Output is a significance score, showing whether or not the mean local 
    densities, at short scale, are above, bellow or about as expected given the
    long-scale local density.

        - T = 0 indicates uniform mixing between 'test' and 'ref' cells
        - T = +1 indicates clustering of 'test' cells around 'ref' cells
             (i.e. more attraction than 'ref' cells around 'ref' cells)
        - T = -1 indicates dispersion of 'test' cells around 'ref' cells
             (i.e. less attraction than 'ref' cells around 'ref' cells)

    This is an array form, calculated by using an KDE-type array 'dy' of 
    local densities for each location and compared to the density of those 
    cells in a long scale region given by Ny (also a KDE).
    
    Args:
        - rcx (numpy): array of 'ref' coordinates (sample points)
        - dy (numpy): nrKDE array of density of 'test' points at short scale
        - Ny (numpy): KAE array of abundance of 'test' points at long scale 
        - kernel (numpy): kernel at which stats are summarized 
        - roi (numpy): ROI mask
        - cuda (bool, optional): use cuda GPU processing. Defaults to False.

    Returns:
        T (numpy) raster array

    """

    pout = np.full(dy.shape, np.nan)

    if (len(rcx) > 0):
        
        from scipy import special
        
        A = np.sum(kernel)
        
        # densities in locations of 'ref' points (short scale)
        x = np.full_like(dy, fill_value=0)
        x[rcx[:, 0], rcx[:, 1]] = dy[rcx[:, 0], rcx[:, 1]]
        
        if (np.sum(x)>0):
                
            deltax = np.nanmin(x[x>0])
            deltak = np.nanmin(kernel[kernel>0])
            
            if cuda:
                # create tensor objects
                xs = tocuda(x)
                ks = tocuda(kernel)
                ns = tocuda(Ny)
                nz = ns > 0
                
                # mean density around ref points (only)
                sx = cudaconv2d(xs, ks)
                sx[sx < deltax*deltak] = 0
                mx = torch.full_like(sx, fill_value=0)
                mx[nz] = sx[nz]/ns[nz]
                sxx = cudaconv2d(xs*xs, ks)
                sxx[sxx < deltax*deltax*deltak] = 0
                mxx = torch.full_like(sxx, fill_value=0)
                mxx[nz] = sxx[nz]/ns[nz]
                
                # std error of density around ref points (only)
                dif = mxx - mx*mx
                dif[dif<0] = 0
                se = torch.full_like(dif, fill_value=0)
                se[nz] = torch.sqrt(dif[nz]/ns[nz])
                
                # t-statistics
                dtf = se > 0
                t = torch.full_like(se, fill_value=np.nan)
                t[dtf] = (mx[dtf] - (ns[dtf]/A))/se[dtf]
                
                t = t.cpu().numpy()
                
                del xs, ks, ns, nz, sx, mx, sxx, mxx, dif, se, dtf
                torch.cuda.empty_cache()
                
            else:
                
                from scipy.signal import fftconvolve
                
                nz = Ny > 0
                
                # mean density around ref points
                sx = fftconvolve(x, kernel, mode='same')
                sx[sx < deltax*deltak] = 0
                mx = np.full_like(sx, fill_value=0)
                mx[nz] = sx[nz]/Ny[nz]
                sxx = fftconvolve(x*x, kernel, mode='same')
                sxx[sxx < deltax*deltax*deltak] = 0
                mxx = np.full_like(sxx, fill_value=0)
                mxx[nz] = sxx[nz]/Ny[nz]
                
                # std error of density around ref points
                dif = mxx - mx*mx
                dif[dif<0] = 0
                se = np.full_like(dif, fill_value=0)
                se[nz] = np.sqrt(dif[nz]/Ny[nz])
                # t-statistics
                dtf = se > 0
                t = np.full_like(se, fill_value=np.nan)
                t[dtf] = (mx[dtf] - (Ny[dtf]/A))/se[dtf]
            
            t[~roi] = np.nan
            p = tofloat(np.full_like(Ny, fill_value=np.nan))
        
            # for elements with N > 30 use normal dist approx
            (r,c) = np.where(Ny>=30)
            p[r,c] = 2*(1 - special.ndtr(np.abs(t[r,c])))
            
            # for elements with N < 30 use standard t dist approx
            (r,c) = np.where((Ny>0) & (Ny<30))
            p[r,c] =2*(1 - special.stdtr(Ny[r,c], np.abs(t[r,c])))
            
            pout = np.sign(t)*(p < 0.05)
        
    return(tofloat(pout))


def nndist(rcx, rcy):
    """Nearest Neighbor Distance index. Global
    
    Calculate the mean NN-Distance for all pairs given by two lists of 
    coordinates. Defined for two different point classes. This measure is
    NOT symmetric and has trivial identity (ie. the index for a cell class
    against itself is equal to 0).
    
        - 'ref_NNDist' is the mean NNDist of ref cells to other ref cells
        - 'test_NNDist' is the mean NNDist of ref cells to test cells
        - Index: v = log(test_NNDist/ref_NNDist)

        - v > 0 indicates ref and test cells are segregated (test cells are 
          prefentially distanced from ref cells)
        - v ~ 0 indicates ref and test cells are well mixed (equally distant 
          from each other)
        - v < 0 indicates ref cells are individually infiltrated (ref cells 
          are closer,thus preferentially surrounded to test cells than other
          ref cells)
                
    Args:
        - rcx (numpy): array of coordinates for ref cells (Nx2)
        - rcy (numpy): array of coordinates for test cells (Nx2)

    Returns:
        (float) value of v.

    """
    from scipy.spatial import KDTree
    
    v = 0
    if (len(rcx) > 1) and (len(rcy) > 0):
        # get mean nearest neighbor distances of ref cells with their own type
        dnnxx, _ = KDTree(rcx).query(rcx, k=[2])
        mdnnxx = np.mean(dnnxx)
        if (mdnnxx > 0):
            # get nearest neighbor distances to test cells
            dnnxy, _ = KDTree(rcy).query(rcx, k=[1])
            # gets ratio of mean NNDist
            v = np.mean(dnnxy) / mdnnxx
    if (v > 0):
        nndi = np.log10(v)
    else:
        nndi = np.nan

    return(tofloat(nndi))
    
    
def nndist_array(rcx, rcy, N, kernel, roi, cuda=False):
    """Nearest Neighbor Distance index. Local

    Calculate the mean NN-Distance for all pairs given by two lists of 
    coordinates. Defined for two different point classes. This measure is
    NOT symmetric and has trivial identity (ie. the index for a cell class
    against itself is equal to 0). 
    
    This index is calculated in each location of the lanscape, with the index 
    being the statistics calculated in each region defined by the kernel 
    (region where the mean NNDs are calculated). For this mean, the array `N` 
    carries the total number of points per kernel.
    
        - 'ref_NNDist' is the mean NNDist of ref cells to other ref cells
        - 'test_NNDist' is the mean NNDist of ref cells to test cells
        - Index: v = log(test_NNDist/ref_NNDist)

        - v > 0 indicates ref and test cells are segregated (test cells are 
          prefentially distanced from ref cells)
        - v ~ 0 indicates ref and test cells are well mixed (equally distant 
          from each other)
        - v < 0 indicates ref cells are individually infiltrated (ref cells 
          are closer,thus preferentially surrounded to test cells than other
          ref cells)

    Args:
        - rcx (numpy): Nx2 array of coordinates for reference class.
        - rcy (numpy): Nx2 array of coordinates for test class. 
        - N (numpy): KAE array with ref cell abundances
        - kernel (numpy): kernel array for convolution smoothing
        - roi (numpy): ROI mask
        - cuda (bool, optional): use cuda GPU processing. Defaults to False.
        
    Returns:
        (float) value of v.

    """

    from scipy.spatial import KDTree
    
    v = np.nan*np.ones(N.shape)
    
    if ((len(rcx) > 1) & (len(rcy) > 0)):

        # get nearest neighbor distances of ref cells to ref cells
        dnnxx, innxx = KDTree(rcx).query(rcx, k=[2])
        # turns into array form
        nnxx = tofloat(np.zeros((N.shape[0], N.shape[1])))
        nnxx[rcx[:, 0], rcx[:, 1]] = tofloat(dnnxx[:, 0])
        
        # get nearest neighbor distances of ref cells to test cells
        dnnxy, innxy = KDTree(rcy).query(rcx, k=[1])
        # turns to array form
        nnxy = tofloat(np.zeros((N.shape[0], N.shape[1])))
        nnxy[rcx[:, 0], rcx[:, 1]] = tofloat(dnnxy[:, 0])
        
        if cuda:
            # transfer arrays to GPU
            xs = tocuda(nnxx)
            ys = tocuda(nnxy)
            ks = tocuda(kernel)
            Ns = tocuda(N)
            dtf = Ns > 0
            
            # local mean of ref-ref nndist(div by local number of ref cells)
            aux = cudaconv2d(xs, ks)
            aux[aux < 0] = 0
            mdnnxx = torch.full_like(xs, fill_value=np.nan)
            mdnnxx[dtf] = aux[dtf] / Ns[dtf]
            
            # local mean of ref-test nndist (div by local number of ref cells)
            aux = cudaconv2d(ys, ks)
            aux[aux < 0] = 0
            mdnnxy = torch.full_like(ys, fill_value=np.nan)
            mdnnxy[dtf] = aux[dtf] / Ns[dtf]
    
            # gets (local) ratio of mean NNDists
            aux = torch.full_like(xs, fill_value=np.nan)
            aux[mdnnxx > 0] = mdnnxy[mdnnxx > 0] / mdnnxx[mdnnxx > 0]
            aux[aux <= 0] = np.nan
            v = torch.log10(aux).cpu().numpy()
            
            del xs, ys, ks, Ns, aux, mdnnxx, mdnnxy
            torch.cuda.empty_cache()
        
        else:  
            
            from scipy.signal import fftconvolve
            
            # local mean of ref-ref nndist(div by local number of ref cells)
            aux = fftconvolve(nnxx, kernel, mode='same')
            aux[aux < 0] = 0
            mdnnxx = np.divide(aux, N, out=np.zeros(N.shape), where=(N > 0))
        
            # local mean of ref-test nndist (div by local number of ref cells)
            aux = fftconvolve(nnxy, kernel, mode='same')
            aux[aux < 0] = 0
            mdnnxy = np.divide(aux, N, out=np.zeros(N.shape), where=(N > 0))

            # gets (local) ratio of mean NNDists
            v = np.divide(mdnnxy, mdnnxx, out=np.zeros(N.shape), 
                          where=(mdnnxx > 0))
            v[v <= 0] = np.nan
            v = np.log10(v)
        
        v[~roi] = np.nan
        
    return(tofloat(v))

    
def getis_ord_g_array(x, box, roi, mu, sigma, A, muxy, sigmaxy, 
                      bw=0, cuda=False):
    """Getis-Ord G* (array form, using kernel neigborhood)
    
    Calculates the Z-score given by the Getis-Ord G* hotspot statistics, 
    accounting for the number of events in a local neighborhood (given as a 
    pre-calculated kde) in relation to the global densitiy of events. 
        - If the associated p-value (according to the z-statistics) is 
          significant and pos, the location is 'hot' (hot = +1)
        - If significant and neg then its 'cold' (hot = -1)
        - If not-significant is 'normal' (hot = 0).
    
    The definition:
                       G_i^* = \frac{A_i}{B_i}
    with:
            
        - A_i = \sum_{j=1}^n w_{i,j}x_j - \mu*\sum_{j=1}^n w_{i,j}
            
        - B_i = \sigma*\sqrt{\frac{D_i}{n-1)}}
            
        - D_i =  n*\sum_{j=1}^n w_{i,j}^2 - \left(\sum_{j=1}^n w_{i,j}\right)^2
    
    and (\mu, \sigma) are the total mean and std of event numbers: 
        
        - \mu = \frac{\sum_{j=1}^n x_j}{n}
        - \sigma = \sqrt(\frac{sum_{j=1}^n x_j^2}{n} - \mu^2))
    
    NOTE: if event are understood as the presence of a point in a location
          then the values of x_j = 1 or 0 and then the std can be reduced to
          
          \sigma = \sqrt(\mu*(1-\mu))
          
    The neighborhood weights w{i,j} are given as an arbitrary kernel 'w', 
    usually a disc or a gaussian kernel (short scale)
    
    By definition this index is locally evaluated BUT defined in  terms of 
    global event density as the reference (null hypothesis).
    
    We can use a local reference defined at a long scale to calculate the 
    hotspot status. Thus the Gi* statistic in each location would be tested 
    for density of events in a small neigborhood around the 
    location in relation to the event density in a long scale region. 
    This can easily be implemented in this array format by replacing the mean
    and std of events by spatial factors: 
    
        n = A(\Omega(r))
        \mu_i_j = \frac{\sum_{j\in\Omega(r)} x_j}{n}
        \sigma = \sqrt(\frac{sum_{j\in\Omega(r)} x_j^2}{n} - \mu^2))
    
    with \Omega(r) a medium scale region (e.g. a disc of radius r)
    and A(\Omega(r)) it's area (as a number of grid elements)
    

    Args:
        - x (numpy): array of point locations
        - box (numpy): short-scale kernel defining neighborhoods
        - roi (numpy): ROI mask
        - mu (float): value of the mean of number of events per pixel
        - sigma (float): value of the std of number of events per pixel
        - A (float): area of long scale region for local references
        - muxy (numpy): kde-type array reflecting the mean number of events in
                        each reference region, at pixel resolution
        - sigmaxy (TYPE): kde-type array reflecting the std number of events
                          in each reference region, at pixel resolution
        - cuda (bool, optional): use cuda GPU processing. Defaults to False.

    Returns:
        (z, h, z_local, h_local): arrays for z-scores and hot indeces

    Refs:
        - https://pro.arcgis.com/en/pro-app/latest/tool-reference/spatial-statistics/hot-spot-analysis.htm
        - https://en.wikipedia.org/wiki/Getis%E2%80%93Ord_statistics

    """
    from scipy import special
        
    # zeroes center point for appropiate G* calculation
    box[int(np.floor(box.shape[0]/2)), int(np.floor(box.shape[1]/2))] = 0
    
    # calculate weigted sums using convolutions
    if cuda:
        # transfer arrays to GPU
        xs = tocuda(x)
        ms = tocuda(roi)
        bs = tocuda(box)
        # weighted sum of x in each neighborhood
        wx = cudaconv2d(xs, bs)[ms > 0]
        # local number of neighbors in each pixel
        sw = cudaconv2d(ms, bs)[ms > 0]
        # the sum of the square of kernel terms for each location
        sw2 = cudaconv2d(ms, torch.mul(bs, bs))[ms > 0]
        # transfer arrays back to CPU
        wx = tofloat(wx.cpu().numpy())
        sw = tofloat(sw.cpu().numpy())
        sw2 = tofloat(sw2.cpu().numpy())
        del xs, ms, bs
        torch.cuda.empty_cache()
    else:        
        from scipy.signal import fftconvolve
        # weighted sum of x in each neighborhood
        wx = fftconvolve(x, box, mode='same')[roi > 0]
        # local number of neighbors in each pixel
        sw = fftconvolve(roi, box, mode='same')[roi > 0]
        # the sum of the square of kernel terms for each location
        sw2 = fftconvolve(roi, box*box, mode='same')[roi > 0]
    
    # regularize negative and small values of calculated convolutions
    wx[wx < np.min(box)] = 0
    sw[sw < np.min(box)] = 0
    sw2[sw < np.min(box)] = 0
       
    # get (flat) indices of terms in the roi
    inx = np.arange(np.prod(roi.shape)).reshape(roi.shape)[roi > 0]
    
    ###### do global stats (inside ROI)
    n = np.sum(roi)                       # number of elements in global stat
    Ui = np.sqrt((n*sw2 - (sw**2))/(n-1)) # dispersion in sums of weights
    Ai = wx - mu*sw                       # numerator of Gi*
    Bi = sigma*Ui                         # denominator of Gi*
    z = np.divide(Ai, Bi, out=np.ones(Ai.shape)*np.nan, where=(Bi != 0))
    
    # revert to full array forms (padded with np.nan)
    aux = np.ones(np.prod(roi.shape))*np.nan
    aux[inx] = z
    Z = aux.reshape(roi.shape)

    # smooths out space factor
    if  bw > 0:
        msk = np.isnan(Z)
        Z[msk] = 0
        Z = fftconv2d(Z, gkern(bw, normal=True), cuda=False)
        Z[msk] = np.nan
        z = Z[roi > 0]
        
    # pvalue is the sum of the two tails (ie. 2 time positive tail)
    p = 2*(1 - special.ndtr(np.abs(z)))
    p[np.isnan(p)] = 1
    
    # get the hotspot index for each location
    sig = np.sign(z)
    sig[p > 0.05] = 0
    aux = np.ones(np.prod(roi.shape))*np.nan
    aux[inx] = sig
    hot = aux.reshape(roi.shape)
    
    ######  now do local stats (inside ROI)
    
    # list terms in input arrays 
    muxy_ = muxy[roi > 0]
    sigmaxy_ = sigmaxy[roi > 0]
    
    Ui = np.sqrt((A*sw2 - (sw**2))/(A-1))  # dispersion in sums of weights
    Ai = wx - muxy_*sw                      # numerator of Gi*
    Bi = sigmaxy_*Ui                        # denominator of Gi*
    z_local = np.divide(Ai, Bi, out=np.ones(Ai.shape)*np.nan, where=(Bi != 0))
    
    # revert to full array forms (padded with np.nan)
    aux = np.ones(np.prod(roi.shape))*np.nan
    aux[inx] = z_local
    Z_local = aux.reshape(roi.shape)
    
    # smooths out space factor
    if  bw > 0:
        msk = np.isnan(Z_local)
        Z_local[msk] = 0
        Z_local = fftconv2d(Z_local, gkern(bw, normal=True), cuda=False)
        Z_local[msk] = np.nan
        z_local = Z_local[roi > 0]
        
    # pvalue is the sum of the two tails (ie. 2* one tail)
    p = 2*(1 - special.ndtr(np.abs(z_local)))
    p[np.isnan(p)] = 1
    
    # get the hotspot index for each location
    sig = np.sign(z_local)
    sig[p > 0.05] = 0
    aux = np.ones(np.prod(roi.shape))*np.nan
    aux[inx] = sig
    hot_local = aux.reshape(roi.shape)

    return(tofloat(Z), tofloat(hot), tofloat(Z_local), tofloat(hot_local))


def morisita_horn_array(x, y, kernel, roi, cuda=False):
    """Morisita-Horn score. Local
    
    M := 2*sum{x_i*y_i*w_i}/(sum{x_i*x_i*w_i} + sum{y_i*y_i*w_i})
    
    Calculated locally, bewteen two 2D arrays of point abundances using spatial
    convolution. The kernel defines a local region (long scale) in which the
    Morisita-Horn index is calculated. The value of the index is assigned to 
    the center of each region. 
    
    Args:
        - x (numpy): KDE of density of events 'x' per pixel (short scale)
        - y (numpy): KDE of density of events 'y' per pixel (short scale)
        - kernel (numpy): kernel array for convolution sum (long-scale)
        - roi (numpy): ROI mask
        - cuda (bool, optional): use cuda GPU processing. Defaults to False.

    Returns:
        M (numpy) array of MH values in each pixel

    """
    
    deltax = np.nanmin(x[x>0])
    deltay = np.nanmin(y[y>0])
    deltak = np.nanmin(kernel[kernel>0])
    
    if cuda:
        
        # transfer arrays to GPU
        xs = tocuda(x)
        ys = tocuda(y)
        ks = tocuda(kernel)
        
        # dispersions in x and y
        xx = cudaconv2d(xs*xs, ks)
        xx[xx < deltax*deltax*deltak] = 0
        yy = cudaconv2d(ys*ys, ks)
        yy[yy < deltay*deltay*deltak] = 0
        xy = cudaconv2d(xs*ys, ks)
        xy[xy < deltax*deltay*deltak] = 0
        
        num = xy
        den = xx + yy
        dtf = (den>0) 
        
        # Morisita Index (colocalization score)
        M = torch.full_like(xs, fill_value=np.nan)
        M[dtf] = 2 * num[dtf]/ den[dtf]
        
        # transfer arrays to CPU
        M = M.cpu().numpy()
        
        del xs, ys, ks, nxs, nys, xx, yy, xy, den, aux, dtf
        torch.cuda.empty_cache()
        
    else:    
        
        from scipy.signal import fftconvolve
        
        # dispersions in x and y
        xx = fftconvolve(x*x, kernel, mode='same')
        xx[xx < deltax*deltax*deltak] = 0
        yy = fftconvolve(y*y, kernel, mode='same')
        yy[yy < deltay*deltay*deltak] = 0
        xy = fftconvolve(x*y, kernel, mode='same')
        xy[xy < deltax*deltay*deltak] = 0
        
        num = xy
        den = xx + yy
        dtf = (den>0) 
        
        # Morisita Index (colocalization score)
        M = np.full_like(x, fill_value=np.nan)
        M[dtf] = 2 * num[dtf]/ den[dtf]

    M[M < 0] = np.nan
    M[M > 1] = np.nan
    M[~roi] = np.nan
    
    return(tofloat(M))


###############################################################################
# %% Plotting functions

def fmt(x, pos):
    a, b = '{:.2e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)


def fmtSimple(x, pos):
    a, b = '{:.2e}'.format(x).split('e')
    b = int(b)
    return r'$10^{{{}}}$'.format(b)


def setTicklabelSpacing(ax, num):
    """ Set the number of ticks to 'num'
    
    Args:
        - ax (pyplot axis): axis to apply ticks too
        - num (int): number of tick marks to use

    Returns:
        num

    """
    # get number of ticks that fit in each axis
    n = min([math.ceil(len(ax.xaxis.get_ticklabels())/num),
             math.ceil(len(ax.yaxis.get_ticklabels())/num)])

    for index, lab in enumerate(ax.xaxis.get_ticklabels()):
        if index % n != 0:
            lab.set_visible(False)

    for index, lab in enumerate(ax.yaxis.get_ticklabels()):
        if index % n != 0:
            lab.set_visible(False)

    return(num)


def plotEdges(shape, binsiz, scale):
    """ Get grid edges for ploting landscapes

    Args:
        - shape (tuple): shape in pixels of TLA landscape.
        - binsiz (int): size of quadrats.
        - scale (float): scale of physical units / pixel.

    Returns:
        - ar: aspect ratio of figure
        - redges, cedges: row and col grid edges (pixels)
        - xedges, yedges: x and y grid edges (units)

    """
    # define quadrats
    redges = np.arange(0, shape[0] + binsiz, binsiz)
    cedges = np.arange(0, shape[1] + binsiz, binsiz)
    
    # fix upper bounds
    redges[-1] = shape[0]
    cedges[-1] = shape[1]

    # aspect ratio
    ar = np.max(cedges)/np.max(redges)

    # coordinates of quadrats (um)
    xedges = [np.around(b*scale, 2) for b in cedges]
    yedges = [np.around(b*scale, 2) for b in redges]

    return([ar, redges, cedges, xedges, yedges])


def landscapeScatter(ax, xs, ys, col, lab, units, xedges, yedges,
                     spoint=1, fontsiz=16, grid=True):
    """ Create a scatter plot of data points
    

    Args:
        - ax (plt axis): axis pyplot object 
        - xs (list): x coordinates
        - ys (list): y coordinates
        - col (color): color
        - lab (str): class label (for legend)
        - units (str): name of physical units
        - xedges (list): x edges of grid
        - yedges (list): y edges of grid
        - spoint (float, optional): size of points. Defaults to 1.
        - fontsiz (float, optional): size of fonts Defaults to 16.
        - grid (bool, optional): DESCRIPTION. show grid? Defaults to True.

    Returns:
        - scatter figure (pyplot object).

    """
    
    ax.plot(xs, ys, color=col, label=lab, 
            marker='.', ms=spoint, linewidth=0)
    ax.set_facecolor('white')
    ax.set_xticks(xedges, minor=False)
    ax.set_yticks(yedges, minor=False)
    ax.minorticks_on()
    if grid:
        ax.grid(which='major', linestyle='--', linewidth='0.3', color='black')
        ax.grid(which='minor', linestyle=':', linewidth='0.1', color='black')
    ax.set_xlim(min(xedges), max(xedges))
    ax.set_ylim(min(yedges), max(yedges))
    ax.set_xticklabels(xedges, rotation=90, fontsize=fontsiz)
    ax.set_yticklabels(yedges, fontsize=fontsiz)
    ax.set_xlabel(units, fontsize=fontsiz)
    ax.set_ylabel(units, fontsize=fontsiz)
    _ = setTicklabelSpacing(ax, 10)

    return(0)


def plotRGB(ax, rgb, units, xedges, yedges, xticks, yticks,
            fontsiz=16, vmin=None, vmax=None, cmap=None):
    """Plot raster as RGB image
    
    Args:
        - ax (plt axis): axis pyplot object 
        - rgb (numpy): raster image
        - units (str): physical units.
        - xedges (list): x edges of grid
        - yedges (list): y edges of grid
        - xticks (list): x tickmarks
        - yticks (list): y tickmarks
        - fontsiz (TYPE, optional): font size. Defaults to 16.
        - vmin (TYPE, optional): min value. Defaults to None.
        - vmax (TYPE, optional): max value. Defaults to None.
        - cmap (TYPE, optional): colormap. Defaults to None.

    Returns:
        plot (pyplot image object)

    """
    
    i = ax.imshow(np.flip(rgb, 0), vmin=vmin, vmax=vmax, 
                  cmap=cmap, interpolation='none')

    current_cmap = mpl.cm.get_cmap()
    current_cmap.set_bad(color='k')
    ax.set_xticks(xedges, minor=False)
    ax.set_yticks(yedges, minor=False)
    ax.minorticks_on()
    ax.grid(which='major', linestyle='--', linewidth='0.3', color='black')
    ax.grid(which='minor', linestyle=':',  linewidth='0.1', color='black')
    ax.set_xlim(min(xedges), max(xedges))
    ax.set_ylim(max(yedges), min(yedges))
    ax.set_xticklabels(xticks, rotation=90, fontsize=fontsiz)
    ax.set_yticklabels(yticks, fontsize=fontsiz)
    ax.set_xlabel(units, fontsize=fontsiz)
    ax.set_ylabel(units, fontsize=fontsiz)
    ax.set_aspect('equal', adjustable='box')
    setTicklabelSpacing(ax, 10)
    ax.invert_yaxis()

    return i


def landscapeLevels(ax, x, y, z, levs, units, xedges, yedges,
                    fontsiz=16):
    """ Plot levels of landscape
    

    Args:
        - ax (plt axis): axis pyplot object 
        - x (numpy array): x values
        - y (numpy array): y values
        - z (numpy array): z values of landscape factor
        - levs (list): contour levels to plot
        - units (str): physical units.
        - xedges (list): x edges of grid
        - yedges (list): y edges of grid
        - fontsiz (TYPE, optional): font size. Defaults to 16.

    Returns:
        None.

    """

    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator, FuncFormatter
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    
    CS1 = ax.contourf(x, y, z, levs,
                      locator=LogLocator(), cmap='YlOrRd', extend='max')
    cbbox = inset_axes(ax, '10%', '50%', loc='upper right')
    [cbbox.spines[k].set_visible(False) for k in cbbox.spines]
    cbbox.tick_params(axis='both',
                      left=False, top=False, right=False, bottom=False,
                      labelleft=False, labeltop=False,
                      labelright=False, labelbottom=False)
    cbbox.set_facecolor([1, 1, 1, 0.75])
    axins = inset_axes(cbbox, '30%', '90%',
                        loc='center left')
    cbar = plt.colorbar(CS1, cax=axins, format=FuncFormatter(fmtSimple))
    axins.yaxis.set_ticks_position('right')
    cbar.ax.tick_params(labelsize=12)
    
    # create contour levels
    #ax.contourf(x, y, z, levs, locator=LogLocator(), cmap='jet_r')
    ax.set_facecolor('white')
    ax.set_xticks(xedges, minor=False)
    ax.set_yticks(yedges, minor=False)
    ax.minorticks_on()
    ax.grid(which='major', linestyle='--', linewidth='0.3', color='black')
    ax.grid(which='minor', linestyle=':', linewidth='0.1', color='black')
    ax.set_xlim(min(xedges), max(xedges))
    ax.set_ylim(min(yedges), max(yedges))
    ax.set_xticklabels(xedges, rotation=90, fontsize=fontsiz)
    ax.set_yticklabels(yedges, fontsize=fontsiz)
    ax.set_xlabel(units, fontsize=fontsiz)
    ax.set_ylabel(units, fontsize=fontsiz)
    setTicklabelSpacing(ax, 10)

    return(0)



"""
###############################################################################
"""
# %% Archived functions (NOT being used)

def str2bool(v):
    return v.lower() in ("true", "t", "1")


def printProgressBar(iteration, total,
                     prefix='Progress', 
                     suffix='Completed',
                     decimals=1, length=50,
                     fill='█'):
    """
    Call in a loop to create terminal progress bar

    Args:
        - iteration (int): current iteration
        - total (int): total number of iterations 
        - prefix (str, optional): prefix string. Defaults to 'Progress'.
        - suffix (str, optional): suffix string. Defaults to 'Completed'.
        - decimals (int, optional): decimals in 'percent'. Defaults to 1.
        - length (int, optional): Dcharacter length of bar. Defaults to 50.
        - fill (str, optional): bar fill character . Defaults to '█'.

    Returns:
        None.

    """
    percent = ("{0:." + str(decimals) +
               "f}").format(100 * (iteration / float(total)))
    filledLength = np.rint(length * iteration // total).astype('int16')
    bar = fill * filledLength + '-' * (length - filledLength)
    suf = suffix[0:80].ljust(80, " ")
    out = f'\r{prefix} |{bar}| {percent}% {suf}'
    print(out, end='\r')
    
    # Print New Line on Complete
    if iteration == total:
        print()


def silverman_bw(data):
    """
    Estimates bw for a 2D KDE using Silverman’s rule of thumb

    """
    
    from scipy import stats
    
    h = np.nan
    coords = np.array(data[['row', 'col']])
    n = coords.shape[0]
    
    if (n>0):
    
        # standard deviations of coordinates
        sigcol = np.std(coords[:,1])
        sigrow = np.std(coords[:,0])
        # iqr of coordinates
        iqrcol = stats.iqr(coords[:,1])
        iqrrow = stats.iqr(coords[:,0])
        # Silverman’s formula
        k = 0.9*np.power(n, -0.2)
        hcol = k*np.min([sigcol, iqrcol/1.35])
        hrow = k*np.min([sigrow, iqrrow/1.35])
        h=np.sqrt(hcol**2 +  hrow**2)
    
    return(h)


def getis_ord_g(f, roi, r = 0, cuda=False):
    """Getis-Ord G* (at pixel resolution using spacial deconvolutions,
       using standard lattice neighborhood)
    
    Calculates the Z-score given by the Getis-Ord G* hotspot statistics, 
    accounting for the number of events in a local neighborhood in the 
    lattice in relation to the global densitiy of events. 
        - If the associated p-value (according to the z-statistics) is 
          significant and pos, the location is 'hot' (hot = +1)
        - If significant and neg then its 'cold' (hot = -1)
        - If not-significant is 'normal' (hot = 0).
    
    The definition:
                       G_i^* = \frac{A_i}{B_i}
    with:
            
        A_i = \sum_{j=1}^n w_{i,j}x_j - \mu*\sum_{j=1}^n w_{i,j}
            
        B_i = \sigma*\sqrt{\frac{D_i}{n-1)}}
            
        D_i =  n*\sum_{j=1}^n w_{i,j}^2 - \left(\sum_{j=1}^n w_{i,j}\right)^2
    
    and (\mu, \sigma) are the total mean and std of event numbers: 
        
        \mu = \frac{\sum_{j=1}^n x_j}{n}
        \sigma = \sqrt(\frac{sum_{j=1}^n x_j^2}{n} - \mu^2))
    
    NOTE: if event are understood as the presence of a point in a location
          then the values of x_j = 1 or 0 and then the std can be reduced to
          
          \sigma = \sqrt(\mu*(1-\mu))
          
    The neighborhood is defined as w_{i,j} = 1 if i and j are neighbors in a 
    square lattice, and zero otherwise (also zero if i=j)
    
    By definition this index is locally evaluated BUT defined in  terms of 
    global event density as the reference (null hypothesis).
    
    We can use a local reference defined at a medium scale to calculate the 
    hotspot status. Thus the Gi* statistic in each location would be tested 
    for density of events in a small neigborhood around the 
    location in relation to the event density in a medium size region. 
    This can easily be implemented in this array format by replacing the mean
    and std of events by spatial factors: 
    
        n = A(\Omega(r))
        \mu_i_j = \frac{\sum_{j\in\Omega(r)} x_j}{n}
        \sigma = \sqrt(\frac{sum_{j\in\Omega(r)} x_j^2}{n} - \mu^2))
    
    with \Omega(r) a medium scale region (e.g. a disc of radious r)
    and A(\Omega(r)) it's area (as a number of grid elements)
    
    The argument 'r' defines the medium scale. If r==0 this is NOT done.
    
    NOTE: both methods, global and local, are calculated when r>0
    
    Args:
        - f (numpy): array of data containing frequency of events in each loc
        - roi (numpy): ROI mask
        - r (TYPE, optional): DESCRIPTION. Defaults to 0.
        - local (TYPE, optional): DESCRIPTION. Defaults to True.
        - cuda (bool, optional): use cuda GPU processing. Defaults to False.

    Returns:
        (z, h, z_local, h_local): arrays for z-scores and hot indeces

    Refs:
        - https://pro.arcgis.com/en/pro-app/latest/tool-reference/spatial-statistics/hot-spot-analysis.htm
        - https://en.wikipedia.org/wiki/Getis%E2%80%93Ord_statistics

    """
    from scipy import special

    # local neighborhood (up to first neighbors in lattice grid)
    box = np.ones((3, 3))
    box[1, 1] = 0
    
    if r>0:
        # kernel for local mean densities (medium scale)
        circ = circle(r)
        A = np.sum(circ)
    
    if cuda:
        # transfer arrays to GPU
        fs = tocuda(f)
        ms = tocuda(roi)
        bs = tocuda(box)
        
        # local number of neighbors in each pixel
        sw = cudaconv2d(ms, bs)[ms > 0]
        # weighted sum of x in each neighborhood
        wx = cudaconv2d(fs, bs)[ms > 0]
        
        # transfer arrays back to CPU
        sw = tofloat(sw.cpu().numpy())
        wx = tofloat(wx.cpu().numpy())
        
        if r>0:
            # transfer arrays to GPU
            cs = tocuda(circle)
            
            # stats terms (mean and std for each circ)
            mx = cudaconv2d(ms, cs)/A
            mxx = cudaconv2d(torch.mul(ms, ms), cs)/A
            mx2 = torch.mul(mx, mx)
            mx = mx[ms > 0] 
            sx = torch.sqrt(mxx - mx2)[ms > 0]
            
            # transfer arrays back to CPU
            mx =  tofloat(mx.cpu().numpy())
            sx =  tofloat(sx.cpu().numpy())
        
            del cs, mxx, mx2
        
        del fs, ms, bs
        torch.cuda.empty_cache()
        
    else:        
        
        from scipy.signal import fftconvolve

        # local number of neighbors in each pixel
        sw = fftconvolve(roi, box, mode='same')[roi > 0]
        # weighted sum of x in each neighborhood
        wx = fftconvolve(f, box, mode='same')[roi > 0]

        if r>0:
            # stats terms (mean and std for each circ)
            mx = fftconvolve(roi, circ)/A
            mxx = fftconvolve(np.multiply(roi, roi), circ)/A
            mx2 = np.multiply(mx, mx)
            mx = mx[roi > 0]
            sx = np.sqrt(mxx - mx2)[roi > 0]
            
            del mxx, mx2
            
    # x values in the roi
    x = tofloat(f[roi > 0])
    
    # get (flat) indices of terms in the roi
    inx = np.arange(np.prod(roi.shape)).reshape(roi.shape)[roi > 0]
    
    # do stats (inside ROI)
    n = len(x)
    mu = np.mean(x)
    sigma = np.std(x)
    Ui = np.sqrt((n*sw - (sw**2))/(n-1))
    Ai = wx - mu*sw
    Bi = sigma*Ui
    z = np.divide(Ai, Bi, out=np.ones(Ai.shape)*np.nan, where=(Bi!= 0))
    z[roi==0] = np.nan
        
    # pvalue is the sum of the two tails (ie. 2* one tail)
    p = 2*(1 - special.ndtr(np.abs(z)))
    p[np.isnan(p)] = 1
    
    # get the hotspot index for each location
    sig = np.sign(z)
    sig[p > 0.05] = 0

    # revert to full array forms (padded with np.nan)
    aux = np.ones(np.prod(roi.shape))*np.nan
    aux[inx] = z
    z = aux.reshape(roi.shape)

    aux = np.ones(np.prod(roi.shape))*np.nan
    aux[inx] = sig
    hot = aux.reshape(roi.shape)
    
    z_local = None
    hot_local = None
    
    if r>0:
        Ui = np.sqrt((A*sw - (sw**2))/(A-1))
        Ai = wx - mx*sw
        Bi = sx*Ui
        z = np.divide(Ai, Bi, out=np.ones(Ai.shape)*np.nan, where=(Bi != 0))
        z[roi==0] = np.nan
            
        # pvalue is the sum of the two tails (ie. 2* one tail)
        p = 2*(1 - special.ndtr(np.abs(z)))
        p[np.isnan(p)] = 1
        
        # get the hotspot index for each location
        sig = np.sign(z)
        sig[p > 0.05] = 0
        
        # revert to full array forms (padded with np.nan)
        aux = np.ones(np.prod(roi.shape))*np.nan
        aux[inx] = z
        z_local = aux.reshape(roi.shape)

        aux = np.ones(np.prod(roi.shape))*np.nan
        aux[inx] = sig
        hot_local = aux.reshape(roi.shape)

    return(tofloat(z), tofloat(hot), tofloat(z_local), tofloat(hot_local))


def morisita_horn_simple(x, y):
    """
    Bivariate Morisita-Horn score: bewteen two 1D arrays of the same order
    
    M := 2*sum{x_i*y_i*w_i}/(sum{x_i*x_i*w_i} + sum{y_i*y_i*w_i})

    Data are 1D arrays of data containing counts of events. 
        - They can be arrays of species abundances in given locations x and y
          (distribution of abundances in tow locations)
        - Or abundances of two given species at a sequence of locations
          (spatial distributions of two species)
        
    Args:
        x (numpy): 1D array of abundances
        y (numpy): 1D array of abundances

    Returns:
        None.

    """
    out = np.nan
    
    # normalizes arrays
    p1 = x/np.sum(x)
    p2 = y/np.sum(y)
    p1p2 = np.dot(p1, p1) + np.dot(p2, p2)
    
    if (p1p2 > 0):
        out = 2*np.dot(p1, p2)/(p1p2)
    
    return(out)


def morisita_horn_univariate(x, min_x=5, min_b=3):
    """Morisita-Horn score, univariate
    
    Univariate implementation of the Morisita-Horn index:

            M := 2*sum{x_i*y_i*w_i}/(sum{x_i*x_i*w_i} + sum{y_i*y_i*w_i})

    where {y_i} is a uniform distribution with the same volume as {x_i}:

            sum{y_i*w_i} = sum{x_i*w_i} = N and y_i = k = constant

    then k = N/A with A the area of integration

    The constant k is the mean density of elements in the integration region, 
    and thus M can be written as:

                    M  = 2 * N^2 / (A*sum{x_i*x_i*w_i} + N^2)
                    
    This metric measures the MH score against an array of the same size with 
    a uniform distribution (Total Spatial Randomness). This score gives the 
    degree of mixing of a sample. 
    
    In case the data density is too low (MH score is stable for data density 
    over ~5 points per bin), it will coarsen the array in order to increase the
    density (by lowering the order), as long as the final order is at least
    3 bins (min_b). Then the accepted array is compared to an array of the
    same order, which is uniformly distributed, using the MH score.

    Args:
        - x (numpy): 1D array with counts of events. This can be a array of
             species abundances or a binned spatial distribution of points
        - min_x (int, optional): min density (events per bin). Defaults to 5.
        - min_b (int, optional): min order (num of bins). Defaults to 3.

    Returns:
        M (float)

    """

    M = np.nan
    if sum(x) > 0:
        # get bin size for a 'min_x' counts of the density of x (if uniform)
        bi = toint(np.ceil(min_x*(len(x)/sum(x))))
        # copy of data
        z = x.copy()

        # if density is such that a typical bin has less than 'min_x' counts
        if bi > 1:
            # coarsen array using add.reduce at frequency 'bi'
            z = np.add.reduceat(x, np.arange(0, x.shape[0], bi), axis=0)

        # gets the MH score if the accepted array has at least 'min_b' bins
        if (len(z) >= min_b):
            # the second array is a uniform array of the same order
            M = morisita_horn_simple(z, np.ones(len(z)))
            
    return(M)


def morisita_horn(x, y):
    """
    Morisita-Horn score: bewteen two 2D arrays of spatial density
    
    M := 2*sum{x_i*y_i*w_i}/(sum{x_i*x_i*w_i} + sum{y_i*y_i*w_i})
    
    NOTE: it is important to use REGULARIZED KDEs for this metric as
    the calculation is added across spatial locations, so this is not a
    proper "local" metric. The normalization of spatial distributions must
    be consistent across the landscape.

    Args:
        - x (numpy): KDE array containing number of 'x' events per pixel.
        - y (numpy): KDE array containing number of 'y' events per pixel.

    Returns:
        M

    """
    
    M = np.ones(x.shape)*np.nan

    # dispersion in x and y abundances
    xx = np.sum(np.multiply(x, x))
    yy = np.sum(np.multiply(y, y))
    xy = np.sum(np.multiply(x, y))
    roi = (xx + yy)>0
    
    M[roi] = 2 * xy[roi]/(xx[roi] + yy[roi])

    # Morisita Index (colocalization score)
    return(M)






























def SSH_factor_detector(Y, factor_name,
                        X, strata_name,
                        THR=0.05):
    """
    The factor detector q-statistic measures the Spatial Stratified 
    Heterogeneity (SSH) of a spatial variable Y (given as an array of numeric
    values) in relation to a categorical variable X (given as an array of 
    labels or blobs). This is also known as the determinant power of a 
    covariate X of Y. 
   
    Ported from R source: https://CRAN.R-project.org/package=geodetector
    Ref: https://cran.r-project.org/web/packages/
                             geodetector/vignettes/geodetector.html
   
    Parameters:
    1- 'Y' : array of values of explained (numerical) variable
    2- 'X' : array of explanatory (categorical) variable. It must be labels
   
    Outputs: Results is a dataframe with the factor detector wich includes
    1- q-statistic   : SSH q-statistic
    2- F-statistic   : F-value (assuming a random non-central F-distribution)
    3- p-value       : Prob that the q-value is observed by random chance
   
    Interpretation of q-test: 
        The q-statistic measures the degree of SSH.
        - (q~1) indicates a strong stratification, small within-strata 
          variance and/or large between-strata variance. Thus there is a 
          strong association between the explanatory variable and the 
          explained variable (ie. strata categories explain the data)
        - (q~0) means within-strata variance is large and/or between-strata
          variance is small. Thus there is no relationship between the strata 
          categories and the data.
   
    The null hypothesis is defined as absence of within-stratum
    heterogeneity (q~0):
         H_0: there is no SSH (stratification is not significant), thus
              within and between strata heterogeneity are similar
         H_1: there is SSH (stratification is significant), thus
              within-strata heterogeneity is significantly smaller than
              between-strata heterogeneity.
   
    For more details of Geodetector method, please refer:
    [1] Wang JF, Li XH, Christakos G, Liao YL, Zhang T, Gu X, Zheng XY.
        Geographical detectors-based health risk assessment and its
        application in the neural tube defects study of the Heshun Region,
        China. International Journal of Geographical. Information Science,
        2010, 24(1): 107-127.
    [2] Wang JF, Zhang TL, Fu BJ. A measure of spatial stratified
        heterogeneity. Ecological. Indicators,2016, 67(2016): 250-256.
    [3] Wang JF, Xu CD. Geodetector:Principle and prospective. Geographica
        Sinica, 2017, 72(1):116-134.
    """
    
    import scipy.stats as sts

    # select valid data
    inx = ~np.isnan(Y)

    # number of valid data points (population size)
    N_popu = np.sum(inx)

    # variance of all samples
    N_var = np.var(Y[inx])

    # unique strata values (can be categorical values)
    #strata = list(set(X[inx]))
    strata = np.unique(X[inx])
    # number of strata (levels) for this variable
    N_stra = len(strata)

    # run stats on each strata
    strataVarSum = 0
    lamda_1st_sum = 0
    lamda_2nd_sum = 0

    for s in strata:

        yi = Y[(inx) & (X == s)]
        LenInter = len(yi)
        strataVar = 0
        lamda_1st = 0
        lamda_2nd = 0

        if (LenInter <= 1):
            strataVar = 0
            lamda_1st = yi**2
            lamda_2nd = yi
        else:
            strataVar = (LenInter - 1) * np.var(yi)
            lamda_1st = (np.mean(yi))**2
            lamda_2nd = np.sqrt(LenInter) * np.mean(yi)

        strataVarSum += strataVar
        lamda_1st_sum += lamda_1st
        lamda_2nd_sum += lamda_2nd

    TotalVar = (N_popu - 1)*N_var

    # q statistic
    q = 1 - (strataVarSum/TotalVar)

    # lamda value
    lamda = (lamda_1st_sum - (lamda_2nd_sum**2 / N_popu)) / N_var

    # F value
    F_value = (N_popu - N_stra) * q / ((N_stra - 1) * (1 - q))

    # p value (positive tail of the cumulative non-centered F statistic)
    p_value = sts.ncf.sf(x=F_value,
                         dfn=N_stra - 1,
                         dfd=N_popu - N_stra,
                         nc=lamda)
    if isinstance(p_value, (list, tuple, np.ndarray)):
        p_value = p_value[0]

    # Create Result
    factor_detector = {'factor': factor_name,
                       'X': strata_name,
                       'q_statistic': q,
                       'F_statistic': F_value,
                       'p_value': p_value,
                       'is_SSH': p_value < THR}

    return(factor_detector)


def SSH_interaction_detector(Y, factor_name,
                             X1, strata_name1,
                             X2, strata_name2,
                             THR=0.05):
    """
    The interaction detector function reveals whether the risk factors
    {X1, X2} have an interactive influence on a factor Y.
   
    Ported from R source: https://CRAN.R-project.org/package=geodetector
    Ref: https://cran.r-project.org/web/packages/
                              geodetector/vignettes/geodetector.html
   
    Parameters:
    1- 'Y' : array of values of explained (numerical) variable
    2- 'X1' : array of arrays of explanatory (categorical) variable 1
    3- 'X2' : array of arrays of explanatory (categorical) variable 2
   
    Outputs: table for the interactive q satistic between variables
   
    For more details of Geodetector method, please refer:
    [1] Wang JF, Li XH, Christakos G, Liao YL, Zhang T, Gu X, Zheng XY.
        Geographical detectors-based health risk assessment and its
        application in the neural tube defects study of the Heshun Region,
        China. International Journal of Geographical
        Information Science, 2010, 24(1): 107-127.
    [2] Wang JF, Zhang TL, Fu BJ. A measure of spatial stratified
        heterogeneity. Ecological. Indicators,2016, 67(2016): 250-256.
    """

    # unique strata values (can be categorical values)
    # strata1 = np.unique(X1)
    # number of strata (levels) for this variable
    # N_stra1 = len(strata1)

    # unique strata values (can be categorical values)
    # strata2 = np.unique(X2)
    # number of strata (levels) for this variable
    # N_stra2 = len(strata2)

    X1X2 = np.multiply(X1, X2)
    strata_name12 = strata_name1 + ":" + strata_name2

    # combined factor detector
    factor1 = SSH_factor_detector(Y, factor_name,
                                  X1, strata_name1, THR=THR)
    factor2 = SSH_factor_detector(Y, factor_name,
                                  X2, strata_name2, THR=THR)
    factor12 = SSH_factor_detector(Y, factor_name,
                                   X1X2, strata_name12, THR=THR)

    # q-statistics
    q = factor12['q_statistic']
    q1 = factor1['q_statistic']
    q2 = factor2['q_statistic']

    # relationship of interactions
    if (q1 < q2):
        qlo = q1
        qhi = q2
        xlo = 'X1'
        xhi = 'X2'
    else:
        qlo = q2
        qhi = q1
        xlo = 'X2'
        xhi = 'X1'

    if (qlo == qhi):
        outputRls = "equivalent"
        description = "q(X1∩X2) = q(X1) = q(X2)"
    else:
        # if (q < q1 + q2):
        #    outputRls   = "weaken"
        #    description = "q(X1∩X2) < q(X1) + q(X2)"
        if (q < qlo):
            outputRls = "weaken, nonlinear"
            description = "q(X1∩X2) < q(X1) and q(X2)"
        if (q < qhi and q == qlo):
            outputRls = xhi + " weaken (uni-)"
            description = "q(X1∩X2) < q(" + xhi + ")"
        if (q < qhi and q > qlo):
            outputRls = xhi + " weaken; " + xlo + " enhance "
            description = "q(" + xlo + ") < q(X1∩X2) < q(" + xhi + ")"
        if (q == qhi and q > qlo):
            outputRls = xlo + " enhance (uni-)"
            description = "q(" + xlo + ") < q(X1∩X2)"
        if (q == q1 + q2):
            outputRls = "independent"
            description = "q(X1∩X2) = q(X1) + q(X2)"
        if (q > q1 + q2):
            outputRls = "enhance, nonlinear"
            description = "q(X1∩X2) > q(X1) + q(X2)"
        if (q > qhi):
            outputRls = "enhance, bi-"
            description = "q(X1∩X2) > q(X1) and q(X2)"

    # Create Result
    interaction_detector = {'factor': factor_name,
                            'X1': strata_name1,
                            'X2': strata_name2,
                            'X1_X2': strata_name12,
                            'q_statistic_1': factor1['q_statistic'],
                            'p_value_1': factor1['p_value'],
                            'q_statistic_2': factor2['q_statistic'],
                            'p_value_2': factor2['p_value'],
                            'q_statistic': factor12['q_statistic'],
                            'p_value': factor12['p_value'],
                            'significance': factor12['p_value'] < THR,
                            'description': outputRls + "; " + description}

    return(interaction_detector)


def SSH_risk_detector(Y, factor_name,
                      X, strata_name,
                      THR=0.05):
    """
    This function calculates the average values in each stratum of the
    explanatory variable (X), and reports if a significant difference
    between two strata exists.
    
    Ported from R source: https://CRAN.R-project.org/package=geodetector
    Ref: https://cran.r-project.org/web/packages/
                             geodetector/vignettes/geodetector.html
   
    Parameters:
    1- 'Y' : array of values of explained (numerical) variable
    2- 'X' : array of explanatory (categorical) variable. It must be labels
    
    Outputs: Results of risk detector include the means of explained variable
             in each stratum and the t-test for differences every pair of
             strata.
    
    For more details of Geodetector method, please refer:
    [1] Wang JF, Li XH, Christakos G, Liao YL, Zhang T, Gu X, Zheng XY.
        Geographical detectors-based health risk assessment and its
        application in the neural tube defects study of the Heshun Region,
        China. International Journal of Geographical. Information Science,
        2010, 24(1): 107-127.
    [2] Wang JF, Zhang TL, Fu BJ. A measure of spatial stratified
        heterogeneity. Ecological. Indicators,2016, 67(2016): 250-256.
    """

    import scipy.stats as sts
    from statsmodels.sandbox.stats.multicomp import multipletests

    # select valid data
    inx = ~np.isnan(Y)

    # unique strata values (can be categorical values)
    strata = np.unique(X[inx])
    # number of strata
    # N_stra = len(strata)

    # for all strata combinations
    risk_detector = pd.DataFrame()
    for i, si in enumerate(strata):
        # data in stratum 'i'
        yi = Y[(inx) & (X == si)]
        for j, sj in enumerate(strata):
            if (j > i):
                # data in stratum 'j'
                yj = Y[(inx) & (X == sj)]

                if (len(yi) > 1 and len(yj) > 1):
                    # Welch’s t-test
                    # (does not assume equal population variances)
                    [tij, pij] = sts.ttest_ind(yi, yj, equal_var=False)

                    aux = pd.DataFrame({'factor': [factor_name],
                                        'strata': [strata_name],
                                        'stratum_i': [str(si)],
                                        'stratum_j': [str(sj)],
                                        'num_Y_i': [len(yi)],
                                        'mean_Y_i': [np.mean(yi)],
                                        'num_Y_j': [len(yj)],
                                        'mean_Y_j': [np.mean(yj)],
                                        't_statistic': [tij],
                                        'p_value': [pij]})
                    risk_detector = pd.concat([risk_detector, aux],
                                              ignore_index=True)
    [sig, adj, a, b] = multipletests(risk_detector['p_value'].tolist(),
                                     alpha=THR,
                                     method='bonferroni')
    risk_detector['p_adjust'] = adj
    risk_detector['significance'] = sig
    risk_detector = risk_detector.astype({'num_Y_i': int,
                                          'num_Y_j': int})

    return(risk_detector)


def SSH_ecological_detector(Y, factor_name,
                            X1, strata_name1,
                            X2, strata_name2,
                            THR=0.05):
    """
    This function identifies the impact of differences between 
    factors  X1 ~ X2
    
    Ported from R source: https://CRAN.R-project.org/package=geodetector
    Ref: https://cran.r-project.org/web/packages/
                              geodetector/vignettes/geodetector.html
   
    Parameters:
    1- 'Y' : array of values of explained (numerical) variable
    2- 'X1' : array of arrays of explanatory (categorical) variable 1
    3- 'X2' : array of arrays of explanatory (categorical) variable 2
    
    Outputs: Results of ecological detector is the significance test of
             impact difference between two explanatory variables.
    
    For more details of Geodetector method, please refer:
    [1] Wang JF, Li XH, Christakos G, Liao YL, Zhang T, Gu X, Zheng XY.
        Geographical detectors-based health risk assessment and its
        application in the neural tube defects study of the Heshun Region,
        China. International Journal of Geographical. Information Science,
        2010, 24(1): 107-127.
    [2] Wang JF, Zhang TL, Fu BJ. A measure of spatial stratified
        heterogeneity. Ecological. Indicators,2016, 67(2016): 250-256.
    """
    
    import scipy.stats as sts

    # number of valid data points (population size)
    N_popu = np.sum(~np.isnan(Y))

    # individual factors
    f_numerator = SSH_factor_detector(Y, factor_name,
                                      X2, strata_name2,
                                      THR=THR)['F_statistic']
    f_denominator = SSH_factor_detector(Y, factor_name,
                                        X1, strata_name1,
                                        THR=THR)['F_statistic']

    F1_value = f_numerator / f_denominator
    F2_value = f_denominator / f_numerator

    # p value (positive tail of the cumulative F-statistic)
    p1_value = sts.f.sf(x=F1_value, dfn=N_popu - 1, dfd=N_popu - 1)
    p2_value = sts.f.sf(x=F2_value, dfn=N_popu - 1, dfd=N_popu - 1)
    F_case = 'X2/X1'

    if p2_value < p1_value:
        F1_value = F2_value
        p1_value = p2_value
        F_case = 'X1/X2'

    # Create Result
    ecological_detector = {'factor': factor_name,
                           'X1': strata_name1,
                           'X2': strata_name2,
                           'F_statistic': F1_value,
                           'X_case/X_ref': F_case,
                           'p_value': p1_value,
                           'significance': p1_value < THR}

    return(ecological_detector)


def SSH_factor_detector_df(data, y_column, x_column_nn, THR=0.05):
    """
     The factor detector q-statistic measures the Spatial Stratified 
     Heterogeneity (SSH) of a variable Y in relation to variables {X}, 
     also known as the determinant power of a covariate X of Y
    
     Ported from R source: https://CRAN.R-project.org/package=geodetector
     Ref: https://cran.r-project.org/web/packages/
                              geodetector/vignettes/geodetector.html
    
     Parameters:
     1- 'data'        : dataframe containing all variables
     2- 'y_column'    : name of explained (numerical) variable
     3- 'x_column_nn' : list of explanatory (categorical) variables
     4- 'THR'         : significance threshold (default 0.05)
     
     Outputs: Results is a dataframe with the factor detector wich includes
     1- variable name : name of explanatory variable
     2- q-statistic   : SSH q-statistic
     3- F-statistic   : F-value (assuming a random non-central F-distribution)
     4- p-value       : Prob that the q-value is observed by random chance
    
     Interpretation of q-test: 
         The q-statistic measures the degree of SSH.
         - (q~1) indicates a strong stratification, small within-strata 
           variance and/or large between-strata variance. Thus there is a 
           strong association between the explanatory variable and the 
           explained variable (ie. strata categories explain the data)
         - (q~0) means within-strata variance is large and/or between-strata
           variance is small. Thus there is no relationship between the strata 
           categories and the data.
    
      The null hypothesis is defined as absence of within-stratum
      heterogeneity (q~0):
           H_0: there is no SSH (stratification is not significant), thus
                within and between strata heterogeneity are similar
           H_1: there is SSH (stratification is significant), thus
                within-strata heterogeneity is significantly smaller than
                between-strata heterogeneity.
    
     For more details of Geodetector method, please refer:
     [1] Wang JF, Li XH, Christakos G, Liao YL, Zhang T, Gu X, Zheng XY.
         Geographical detectors-based health risk assessment and its
         application in the neural tube defects study of the Heshun Region,
         China. International Journal of Geographical. Information Science,
         2010, 24(1): 107-127.
     [2] Wang JF, Zhang TL, Fu BJ. A measure of spatial stratified
         heterogeneity. Ecological. Indicators,2016, 67(2016): 250-256.
     [3] Wang JF, Xu CD. Geodetector:Principle and prospective. Geographica
         Sinica, 2017, 72(1):116-134.
    """
    
    import scipy.stats as sts

    factor_detector = pd.DataFrame()

    # number of valid data points (population size)
    N_popu = len(data)
    # variance of all samples
    N_var = np.var(data[y_column])

    # for each explanatory variable
    for x_column in x_column_nn:

        # unique strata values (can be categorical values)
        # strata = list(set(data[x_column]))
        strat = np.unique(data[x_column])
        # number of strata (levels) for this variable
        N_stra = len(strat)
        
        q = np.nan
        F_value = np.nan
        p_value = np.nan
        
        if ((N_popu > 1) & (N_var > 0) & (N_stra > 1)):

            # run stats on each strata
            strataVarSum = 0
            lamda_1st_sum = 0
            lamda_2nd_sum = 0
            for s in strat:
                yi = data.loc[data[x_column]== s][y_column].to_numpy()
                LenInter = len(yi)
                strataVar = 0
                lamda_1st = 0
                lamda_2nd = 0
                if (LenInter > 0):
                    strataVar = (LenInter - 1) * np.var(yi)
                    lamda_1st = (np.mean(yi))**2
                    lamda_2nd = np.sqrt(LenInter) * np.mean(yi)
    
                strataVarSum += strataVar
                lamda_1st_sum += lamda_1st
                lamda_2nd_sum += lamda_2nd
    
            TotalVar = (N_popu - 1)*N_var
    
            # q statistic
            q = 1 - (strataVarSum/TotalVar)
    
            # lamda value
            lamda = (lamda_1st_sum - (lamda_2nd_sum**2 / N_popu)) / N_var
    
            # F value
            F_value = (N_popu - N_stra) * q / ((N_stra - 1) * (1 - q))
    
            # p value (positive tail of the cumulative non-centered F statistic)
            p_value = sts.ncf.sf(x=F_value,
                                 dfn=N_stra - 1,
                                 dfd=N_popu - N_stra,
                                 nc=lamda)
            if isinstance(p_value, (list, tuple, np.ndarray)):
                p_value = p_value[0]

        # Create Result
        aux = pd.DataFrame({'factor': [y_column],
                            'strata': [x_column],
                            'q_statistic': [q],
                            'F_statistic': [F_value],
                            'p_value': [p_value],
                            'is_SSH': [p_value < THR]})
        factor_detector = pd.concat([factor_detector, aux], ignore_index=True)

    return(factor_detector)


def SSH_interaction_detector_df(data, y_column, x_column_nn, 
                                qs = None, ps = None, THR=0.05):
    """
    The interaction detector function reveals whether the risk factors
    {X_i} have an interactive influence on a factor Y.
   
    Ported from R source: https://CRAN.R-project.org/package=geodetector
   
    Parameters:
    1- 'data' : is the dataframe that contains all variables.
    2- 'y_column': is the name of explained (numerical) variable
    3- 'x_column_nn' : list of names of (categorical) explanatory variables
    4- 'qs' : q-values if ssh is already calculated for each variable
    5- 'ps' : p-values if ssh is already calculated for each variable
    6- 'THR' : significance threshold (default 0.05)
    
    Outputs: table for the interactive q satistic between variables
   
    For more details of Geodetector method, please refer:
    [1] Wang JF, Li XH, Christakos G, Liao YL, Zhang T, Gu X, Zheng XY.
        Geographical detectors-based health risk assessment and its
        application in the neural tube defects study of the Heshun Region,
        China. International Journal of Geographical
        Information Science, 2010, 24(1): 107-127.
    [2] Wang JF, Zhang TL, Fu BJ. A measure of spatial stratified
        heterogeneity. Ecological. Indicators,2016, 67(2016): 250-256.
    """

    from itertools import combinations
    from statsmodels.sandbox.stats.multicomp import multipletests
    
    if ((qs is not None) or (ps is not None)):
        if ((len(qs) != len(x_column_nn)) or (len(ps) != len(x_column_nn))):
            print("ERROR: qs and ps must have same lentgh as x_column_nn")
            return(None)

    # number of variables
    n_x = len(x_column_nn)
    if(n_x < 2):
        print("ERROR: X input should be more than one variable")
        return(None)

    # combination  for X1, X2...
    x_column_ii =  range(len(x_column_nn))
    x_x = [list(i) for i in list(combinations(x_column_nn, 2))]
    x_i = [list(i) for i in list(combinations(x_column_ii, 2))]
    # n_x_x = len(x_x)

    # output data frame
    interaction_detector = pd.DataFrame()
    table = data.copy()
    
    for n, [x1_colnam, x2_colnam] in enumerate(x_x):
        
        nam = x1_colnam + ":" + x2_colnam
        # m1 = table[x1_colnam].astype(str)
        # m2 = table[x2_colnam].astype(str)
        # table[nam] = m1.str.cat(m2, sep='_')
        table[nam] = table[x1_colnam]*table[x2_colnam]

        # combined factor detector
        if qs is None:
            sshtbl = SSH_factor_detector_df(table, 
                                            y_column, 
                                            [x1_colnam, 
                                             x2_colnam, 
                                             nam], THR=THR)
            # q-statistics
            q1 = sshtbl['q_statistic'][0]
            q2 = sshtbl['q_statistic'][1]
            q = sshtbl['q_statistic'][2]
            
            # p-values
            p1 = sshtbl['p_value'][0]
            p2 = sshtbl['p_value'][1]
            p = sshtbl['p_value'][2]
            
        else:
            sshtbl = SSH_factor_detector_df(table, 
                                            y_column, 
                                            [nam], THR=THR)
            # q-statistics
            q1 = qs[x_i[n][0]]
            q2 = qs[x_i[n][1]]
            q = sshtbl['q_statistic'][0]
            
            # p-values
            p1 = ps[x_i[n][0]]
            p2 = ps[x_i[n][1]]
            p = sshtbl['p_value'][0]
        
        # relationship of interactions
        if (q1 < q2):
            qlo = q1
            qhi = q2
            xlo = 'X1'
            xhi = 'X2'
        else:
            qlo = q2
            qhi = q1
            xlo = 'X2'
            xhi = 'X1'

        if (qlo == qhi):
            outputRls = "equivalent"
            description = "q(X1∩X2) = q(X1) = q(X2)"
        else:
            # if (q < q1 + q2):
            #    outputRls   = "weaken"
            #    description = "q(X1∩X2) < q(X1) + q(X2)"
            if (q < qlo):
                outputRls = "weaken, nonlinear"
                description = "q(X1∩X2) < q(X1) and q(X2)"
            if (q < qhi and q == qlo):
                outputRls = xhi + " weaken (uni-)"
                description = "q(X1∩X2) < q(" + xhi + ")"
            if (q < qhi and q > qlo):
                outputRls = xhi + " weaken; " + xlo + " enhance "
                description = "q(" + xlo + ") < q(X1∩X2) < q(" + xhi + ")"
            if (q == qhi and q > qlo):
                outputRls = xlo + " enhance (uni-)"
                description = "q(" + xlo + ") < q(X1∩X2)"
            if (q == q1 + q2):
                outputRls = "independent"
                description = "q(X1∩X2) = q(X1) + q(X2)"
            if (q > q1 + q2):
                outputRls = "enhance, nonlinear"
                description = "q(X1∩X2) > q(X1) + q(X2)"
            if (q > qhi):
                outputRls = "enhance, bi-"
                description = "q(X1∩X2) > q(X1) and q(X2)"
            else:
                outputRls = "NULL"
                description = ""

        desc = outputRls + "; " + description

        # Create Result
        aux = pd.DataFrame({'factor_Y': [y_column],
                            'X1': [x1_colnam],
                            'X2': [x2_colnam],
                            'X1_X2': [nam],
                            'q_statistic_1': [q1],
                            'p_value_1': [p1],
                            'q_statistic_2': [q2],
                            'p_value_2': [p2],
                            'q_statistic': [q],
                            'p_value': [p],
                            'description': [desc]})

        interaction_detector = pd.concat([interaction_detector, aux],
                                         ignore_index=True)

    [sig, adj, a, b] = multipletests(interaction_detector['p_value'].tolist(),
                                     alpha=THR,
                                     method='bonferroni')
    interaction_detector['p_adjust'] = adj
    interaction_detector['significance'] = sig

    return(interaction_detector)


def SSH_risk_detector_df(data, y_column, x_column, THR=0.05):
    """
    This function calculates the average values in each stratum of the
    explanatory variable (X), and reports if a significant difference
    between two strata exists.
    
    Ported from R source: https://CRAN.R-project.org/package=geodetector
    
    Parameters:
    1- 'data': is the dataframe that contains variable 
    2- 'y_column': is the name of explained (numerical) variable 
    3- 'x_column': is the name of (categorical) explanatory variable 
    4- 'THR' : significance threshold (default 0.05)
    
    Outputs: Results of risk detector include the means of explained variable
             in each stratum and the t-test for differences every pair of
             strata.
    
    For more details of Geodetector method, please refer:
    [1] Wang JF, Li XH, Christakos G, Liao YL, Zhang T, Gu X, Zheng XY.
        Geographical detectors-based health risk assessment and its
        application in the neural tube defects study of the Heshun Region,
        China. International Journal of Geographical. Information Science,
        2010, 24(1): 107-127.
    [2] Wang JF, Zhang TL, Fu BJ. A measure of spatial stratified
        heterogeneity. Ecological. Indicators,2016, 67(2016): 250-256.
    """
    
    import scipy.stats as sts
    from statsmodels.sandbox.stats.multicomp import multipletests
    import warnings
    
    # unique strata values (can be categorical values)
    # strata = list(set(data[x_column]))
    strata = np.unique(data[x_column])
    # number of strata
    # N_stra = len(strata)

    risk_detector = pd.DataFrame()

    # for all strata combinations
    for i, si in enumerate(strata):
        # data in stratum 'i'
        yi = data.loc[data[x_column]== si][y_column].dropna()
        for j, sj in enumerate(strata):
            if (j > i):
                # data in stratum 'j'
                yj = data.loc[data[x_column]== sj][y_column].dropna()
                if (len(yi) > 1 and len(yj) > 1):
                    # Welch’s t-test
                    # (does not assume equal population variances)
                    with warnings.catch_warnings():
                        
                        warnings.simplefilter('ignore')
                        [tij, pij] = sts.ttest_ind(yi, yj, equal_var=False)

                    aux = pd.DataFrame({'strata': [x_column],
                                        'stratum_i': [si],
                                        'stratum_j': [sj],
                                        'factor_Y': [y_column],
                                        'num_Y_i': [len(yi)],
                                        'mean_Y_i': [np.mean(yi)],
                                        'std_Y_i': [np.std(yi)],
                                        'num_Y_j': [len(yj)],
                                        'mean_Y_j': [np.mean(yj)],
                                        'std_Y_j': [np.std(yj)],
                                        't_statistic': [tij],
                                        'p_value': [pij]})

                    risk_detector = pd.concat([risk_detector, aux],
                                              ignore_index=True)
                       
    if len(risk_detector) > 0:
        [sig, adj, _, _] = multipletests(risk_detector['p_value'].tolist(),
                                         alpha=THR,
                                         method='bonferroni')
        risk_detector['p_adjust'] = adj
        risk_detector['significance'] = sig
        risk_detector = risk_detector.astype({'num_Y_i': int,
                                              'num_Y_j': int})

    return(risk_detector)


def SSH_ecological_detector_df(data, y_column, x_column_nn, 
                               fs = None, THR=0.05):
    """
    This function identifies the impact of differences between factors  X1 ~ X2
    
    Translated from R,
    Source: https://CRAN.R-project.org/package=geodetector
    
    Parameters:
    1- 'tabledata'   : dataframe that contains all variables
    2- 'y_column'    : name of explained (numerical) variable in dataset
    3- 'x_column_nn' : list of names of (categorical) explanatory variables
    4- 'THR' : significance threshold (default 0.05)
    
    Outputs: Results of ecological detector is the significance test of
             impact difference between two explanatory variables.
    
    For more details of Geodetector method, please refer:
    [1] Wang JF, Li XH, Christakos G, Liao YL, Zhang T, Gu X, Zheng XY.
        Geographical detectors-based health risk assessment and its
        application in the neural tube defects study of the Heshun Region,
        China. International Journal of Geographical. Information Science,
        2010, 24(1): 107-127.
    [2] Wang JF, Zhang TL, Fu BJ. A measure of spatial stratified
        heterogeneity. Ecological. Indicators,2016, 67(2016): 250-256.
    """

    from itertools import combinations
    from statsmodels.sandbox.stats.multicomp import multipletests
    
    if ((fs is not None)):
        if (len(fs) != len(x_column_nn)):
            print("ERROR: fs must have same lentgh as x_column_nn")
            return(None)

    # number of interactions
    n_x = len(x_column_nn)
    if(n_x < 2):
        print("ERROR: X variables input should be more than one variable")
        return(None)

    # combination  for X1, X2...
    x_column_ii =  range(len(x_column_nn))
    x_x = [list(i) for i in list(combinations(x_column_nn, 2))]
    x_i = [list(i) for i in list(combinations(x_column_ii, 2))]
    # n_x_x = len(x_x)

    # output data frame
    ecological_detector = pd.DataFrame()

    for n, [x1_colnam, x2_colnam] in enumerate(x_x):
        
        # combined factor detector
        if fs is None:
            sshtbl = SSH_factor_detector_df(data, 
                                            y_column, 
                                            [x1_colnam, 
                                             x2_colnam], THR=THR)
            # individual factors
            f_numerator = sshtbl['F_statistic'][0]
            f_denominator = sshtbl['F_statistic'][1]
            
        else:
            
            # individual factors
            f_numerator = fs[x_i[n][0]]
            f_denominator = fs[x_i[n][1]]

        F1_value = f_numerator / f_denominator
        F2_value = f_denominator / f_numerator

        # p value (positive tail of the cumulative F-statistic)
        p1_value = sts.f.sf(x=F1_value,
                            dfn=len(data.index) - 1,
                            dfd=len(data.index) - 1)
        p2_value = sts.f.sf(x=F2_value,
                            dfn=len(data.index) - 1,
                            dfd=len(data.index) - 1)

        if p2_value < p1_value:
            F1_value = F2_value
            p1_value = p2_value

        # Create Result
        aux = pd.DataFrame({'factor_Y': [y_column],
                            'X1': [x1_colnam],
                            'X2': [x2_colnam],
                            'F_statistic': [F1_value],
                            'p_value': [p1_value]})

        ecological_detector = pd.concat([ecological_detector, aux],
                                        ignore_index=True)

    [sig, adj, a, b] = multipletests(ecological_detector['p_value'].tolist(),
                                     method='bonferroni')
    ecological_detector['p_adjust'] = adj
    ecological_detector['significance'] = sig

    return(ecological_detector)



