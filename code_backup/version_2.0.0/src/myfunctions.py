'''
     TLA general functions

'''

import os
import math
import pandas as pd
import numpy as np
import scipy.stats as sts
import matplotlib as mpl
# import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
import torch
from torch.fft import fft2, ifft2 
from torch.nn.functional import pad

__version__  = "2.0.0"


def tofloat(x):
    return(np.float64(x))

def toint(x):
    return(np.int64(x))

def tobit(x):
    return(np.uint8(x))

def tocuda(x):
    return(torch.from_numpy(x).type(torch.float64).cuda())


def mkdirs(path):
    # create folder if it doesn't exist
    if not os.path.exists(path):
        # try-except in case a parallel process is simultaneously 
        # creating the same folder folder
        try:
            os.makedirs(path)
        except OSError:
            pass
        
    return(path)


def str2bool(v):
    return v.lower() in ("true", "t", "1")


def printProgressBar(iteration, total,
                     prefix='Progress', 
                     suffix='Completed',
                     decimals=1, length=50,
                     fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : number of decimals in percent (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    #import sys
    
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


def circle(rin):
    """
    Creates hard disc kernel of radious 'r'
    (this is equivalent to a 2D heavyside function H_2(r))
    """
    r = int(rin)
    y, x = np.ogrid[-r: r + 1, -r: r + 1]
    return(tofloat(1.0*(x**2 + y**2 < r**2)))


def gkern(sig):
    """
    Creates normalized gaussian kernel with sigma value `sig`
    (up to 5*sigma)
    """
    
    l = 5.0*sig
    
    aux = np.arange(-(l - 1) / 2., ((l - 1) / 2.) + 1)
    gauss = np.exp(-0.5 * np.square(aux) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    kernel[kernel<0] = 0
    return (tofloat(kernel / np.sum(kernel)))


def complex_product(x, y):
    '''
    Fast complex multiplication (Karatsuba multiplication)
    (*) https://stackoverflow.com/questions/19621686/complex-numbers-product-using-only-three-multiplications
    (*) https://en.wikipedia.org/wiki/Karatsuba_algorithm
    '''
    
    # Extract the real and imaginary parts
    a, b = x.real, x.imag
    c, d = y.real, y.imag

    # term by term products
    ac = torch.mul(a, c)
    bd = torch.mul(b, d)
    ab_cd = torch.mul(torch.add(a, b), torch.add(c, d))
    
    return torch.complex(ac - bd, ab_cd - ac - bd)

def cudaconv2d(imgs, filts):
    # calculates 2D convolution using FFT in GPU using CUDA 

    imgsshape = imgs.shape
    filtsshape = filts.shape

    # Pad and transform the image and filter
    # Pad arg = (last dim pad left side, last dim pad right side, 
    #            2nd last dim left side, etc..)
    f_imgs = fft2(pad(imgs, (0, filtsshape[1] - 1, 0, filtsshape[0] - 1)))
    f_filts = fft2(pad(filts, (0, imgsshape[1] - 1, 0, imgsshape[0] - 1)))
    
    # Do element wise complex multiplication and then reverse transform
    img = ifft2(complex_product(f_imgs, f_filts)).real

    # get back not-padded part
    dc = int(filtsshape[1]/2)
    dr = int(filtsshape[0]/2)
    
    return img[dr:(imgsshape[0]+dr), dc:(imgsshape[1]+dc)]


def fftconv2d(img, filt, cuda=False):
    
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
        # if GPU not available, uses scipy (faster than pytorch in CPU)
        imgout = fftconvolve(img, filt, mode='same')
        
    return tofloat(imgout)


def KDE(data, shape, bw, cuda=False):
    """
    Evaluates a KDE using a convolution with a Fast Fourier Transform (FFT)
    using array convolution 'fftconvolve' from scipy.signal (in CPU)
    or torch (in GPU)

    Parameters
    ----------
    - data: (pandas df) TLA dataframe of cell coordinates
    - shape: (tuple) shape in pixels of TLA landscape
    - bw : (float) bandwidth; standard deviation of the KDE (gaussian) kernel

    """
    kern = gkern(bw)

    arr = tofloat(np.zeros((shape[0], shape[1])))
    coords = np.array(data[['row', 'col']])
    arr[coords[:,0], coords[:,1]] = 1.0
    
    # Compute the kernel density estimate
    points = fftconv2d(arr, kern, cuda=cuda)
    
    # regularize negative and small values
    v = 10**(np.around(np.log10(0.01/np.sum(kern>0)), 0)) 
    points[points<v] = 0

    # normalizes the output to cells per grid unit
    points = len(data)*(points/np.sum(points))
    
    return([toint(coords[:,0]), 
            toint(coords[:,1]), 
            tofloat(points)])


def maskCells(cell_data, mask):
    """
    Tags cells from cell_data table that are in the zero-region in binary mask

    Parameters
    ----------
    - data: (pandas df) TLA dataframe of cell coordinates
    - mask: (numpy array) ROI binary mask

    """

    aux = cell_data.copy().reset_index(drop=True)
    aux['i'] = aux.index

    irc = toint(np.array(aux[['i', 'row', 'col']]))

    # get cells in data that are outside the mask
    ii = np.ones(len(aux), dtype='bool')
    ii[irc[:, 0]] = mask[irc[:, 1], irc[:, 2]]
    aux['masked'] = np.logical_not(ii)

    return(aux.drop(columns=['i']))


def filterCells(cell_data, mask):
    """
    Filters out cells from the cell_data table according to a binary mask

    Parameters
    ----------
    - data: (pandas df) TLA dataframe of cell coordinates
    - mask: (numpy array) ROI binary mask

    """

    # tag masked out cells
    aux = maskCells(cell_data, mask)

    # drop cells outside of mask
    aux.drop(aux.loc[aux['masked']].index, inplace=True)

    return(aux.drop(columns=['masked']).reset_index(drop=True))


def arrayLevelMask(z, th, rmin, fill_holes=True):
    """
    returns an array binary mask: 0 for values below a threshold, 1 otherwise

    Parameters
    ----------
    - z: (numpy array) array of float values
    - th: (float) threshold value; z[z<th] = 0
    - rmin: (float) regions with r<rmin are filtered out
    - fill_holes: (bool) if true, holes are fillled out

    """

    from skimage.morphology import remove_small_objects, label
    from scipy.ndimage import binary_fill_holes

    lab = label((z > th).astype('int32'), connectivity=2) > 0
    aux = remove_small_objects(lab,
                               min_size=np.pi*(rmin)*(rmin),
                               connectivity=2)
    if fill_holes:
        aux = binary_fill_holes(aux)

    return((aux > 0).astype('bool'))


def kdeLevels(data, shape, bw, all_cells=True, toplev=1.0, cuda=False):

    # Compute the kernel density estimate
    [r, c, z] = KDE(data, shape, bw, cuda=cuda)

    # threshold for landscape edge
    aux = tofloat(z.copy())
    aux[aux<=0]=np.nan
    aux = np.around(np.log10(aux, where=aux > 0), 2)
    
    if (all_cells):
        if len(aux[aux > np.nanmin(aux)]) > 0:
            # first value above background
            th = 10**np.nanmin(aux[aux > np.nanmin(aux)])
        else:
            th = 10**np.nanmin(aux)
    else:
        # get top order of magnitud
        th = 10**np.nanmin(aux[aux > (np.nanmax(aux) - toplev)])
    
    levs = 10**np.arange(np.floor(np.log10(th)), np.ceil(np.nanmax(aux)) + 1)

        
    # get mask at 'th' level (min size radius == bw)
    m = arrayLevelMask(z, th, bw, fill_holes=False)

    return([r, c, z, m, levs, th])


def kdeMask(data, shape, bw, cuda=False):
    """
    Calculates a pixel resolution KDE profile from cell location data and
    generates a mask for the bakground region (where no cells are found).
    This corresponds to the ROI for TLA

    Parameters
    ----------
    - data: (pandas df) TLA dataframe of cell coordinates
    - shape: (tuple) shape in pixels of TLA landscape
    - bw : (float) bandwidth; standard deviation of the KDE (gaussian) kernel

    """

    if (len(data) > 0):
        # Compute the kernel density estimate and levels
        [r, c, z, mask, levs, th] = kdeLevels(data, shape, bw, cuda=cuda)
     
        # renormalizes z (to cells per unit pixel)
        z[np.logical_not(mask)] = 0
        z = tofloat(len(data)*(z/np.sum(z)))
        
    else:
        z = tofloat(np.zeros(shape))
        mask = np.zeros(shape, dtype='bool')

    return(z, mask)


def fmt(x, pos):

    a, b = '{:.2e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)


def fmtSimple(x, pos):
    
    a, b = '{:.2e}'.format(x).split('e')
    b = int(b)
    return r'$10^{{{}}}$'.format(b)


def setTicklabelSpacing(ax, num):

    n = min([math.ceil(len(ax.xaxis.get_ticklabels())/num),
             math.ceil(len(ax.yaxis.get_ticklabels())/num)])

    for index, lab in enumerate(ax.xaxis.get_ticklabels()):
        if index % n != 0:
            lab.set_visible(False)

    for index, lab in enumerate(ax.yaxis.get_ticklabels()):
        if index % n != 0:
            lab.set_visible(False)

    return(num)


def landscapeScatter(ax, xs, ys, col, lab, units, xedges, yedges,
                     spoint=1, fontsiz=16):

    scatter = ax.scatter(x=xs, y=ys, c=col,
                         label=lab, s=spoint,
                         marker='.', linewidths=0)
    ax.axis('square')
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

    return scatter


def landscapeLevels(ax, x, y, z, m, levs, units, xedges, yedges,
                    fontsiz=16):

    from matplotlib.ticker import LogLocator
    # from matplotlib.ticker import FuncFormatter
    # from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    # zeroes background
    z[np.logical_not(m)] = 0.0
    ax.contourf(x, y, z, levs, locator=LogLocator(), cmap='RdBu_r')
    
    # CS1 = ax.contourf(x, y, z, levs,
    #                   locator=LogLocator(), cmap='RdBu_r')
    # cbbox = inset_axes(ax, '15%', '50%', loc='lower right')
    # [cbbox.spines[k].set_visible(False) for k in cbbox.spines]
    # cbbox.tick_params(axis='both',
    #                   left=False, top=False, right=False, bottom=False,
    #                   labelleft=False, labeltop=False,
    #                   labelright=False, labelbottom=False)
    # cbbox.set_facecolor([1, 1, 1, 0.75])
    # axins = inset_axes(cbbox, '35%', '90%',
    #                    loc='center left')

    # cbar = plt.colorbar(CS1, cax=axins, format=FuncFormatter(fmtSimple))
    # axins.yaxis.set_ticks_position('right')
    # cbar.ax.tick_params(labelsize=fontsiz)
    
    ax.set_aspect(aspect=1)
    ax.contour(x, y, m, [0.5], linewidths=2, colors='green')
    ax.axis('square')
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


def plotRGB(ax, rgb, units, xedges, yedges, xticks, yticks,
            fontsiz=16, vmin=None, vmax=None, cmap=None):

    i = ax.imshow(np.flip(rgb, 0), vmin=vmin, vmax=vmax, cmap=cmap)

    current_cmap = mpl.cm.get_cmap()
    current_cmap.set_bad(color='k')
    ax.axis('square')
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
    setTicklabelSpacing(ax, 10)
    ax.invert_yaxis()

    return i


def morisita_horn_simple(x, y):
    """
    Bivariate Morisita-Horn score: bewteen two 1D arrays of the same order


    Parameters
    ----------
    - x: array of data containing counts of events. This can be a array of
         species abundances in a location "X", or binned spatial locations of
         individuals of one species "X"
    - y: array of data containing counts of events. This can be a array of
         species abundances in a location "Y", or binned spatial locations of
         individuals of one species "Y"

    """
    # normalizes arrays
    p1 = x/np.sum(x)
    p2 = y/np.sum(y)
    return(2*np.dot(p1, p2)/(np.dot(p1, p1) + np.dot(p2, p2)))


def morisita_horn_univariate(x, min_x=5, min_b=3):
    """
    Morisita-Horn score, univariate version: measures the MH score against
    an array of the same size with a uniform distribution.
    This score gives the degree of mixing of a sample. In case the data
    density is too low (MH score is stable for data density over ~5 (min_x)
    points per bin), it will coarsen the array in order to increase the
    density (by lowering the order), as long as the final order is at least
    3 bins (min_b). Then the accepted array is compared to an array of the
    same order, which is uniformly distributed, using the MH score.

    Parameters
    ----------
    - x: array of data containing counts of events. This can be a array of
         species abundances or binned spatial locations of individuals
    - min_x: minimun data density (events per bin)
    - min_b: minimun order (number of bins) of data array
    """

    m = np.nan
    if sum(x) > 0:
        # get bin size of 'min_x' counts for the data density of x (if uniform)
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
            m = morisita_horn_simple(z, np.ones(len(z)))
    return(m)


def morisita_horn(x, y):
    """
    Bivariate Morisita-Horn score: bewteen two 2D arrays

    Parameters
    ----------
    - x: array of data containing counts of events per pixel.
    - y: array of data containing counts of events per pixel.
    """

    # dispersion in x and y abundances
    xx = np.sum(np.multiply(x, x))
    yy = np.sum(np.multiply(y, y))
    xy = np.sum(np.multiply(x, y))

    # Morisita Index (colocalization score)
    return(2 * xy/(xx + yy))


def morisita_horn_array(x, y, kernel, cuda=False):
    """
    Bivariate Morisita-Horn score: bewteen two 2D arrays using spacial
    deconvolution

    Parameters
    ----------
    - x: array of data containing counts of events per pixel.
    - y: array of data containing counts of events per pixel.
    - kernel: small kernel array for deconvolution smoothing
    """
    if cuda:
        
        # transfer arrays to GPU
        xs = tocuda(x)
        ys = tocuda(y)
        ks = tocuda(kernel)
        
        # dispersion in x and y abundance
        xx = torch.round(cudaconv2d(torch.mul(xs, xs), ks))
        yy = torch.round(cudaconv2d(torch.mul(ys, ys), ks))
        xy = torch.round(cudaconv2d(torch.mul(xs, ys), ks))
        den = torch.add(xx, yy)
        
        # Morisita Index (colocalization score)
        # initialize output tensor with desired value
        aux = torch.full_like(xs, fill_value=np.nan)
        aux[den > 0] = 2 * xy[den > 0]/ den[den > 0]
        
        # transfer arrays to CPU
        out= aux.cpu().numpy()
        
        del xs, ys, ks, xx, yy, xy, den, aux
        torch.cuda.empty_cache()
        
    else:    
        # dispersion in x and y abundance
        xx = np.rint(fftconvolve(np.multiply(x, x), kernel, mode='same'))
        yy = np.rint(fftconvolve(np.multiply(y, y), kernel, mode='same'))
        xy = np.rint(fftconvolve(np.multiply(x, y), kernel, mode='same'))
        den = xx + yy
        
        # Morisita Index (colocalization score)
        out = 2 * np.divide(xy, den, 
                            out=np.ones(den.shape)*np.nan,
                            where=(den > 0))

    return(tofloat(out))


def getis_ord_g_array(X, msk, cuda=False):
    """
    Gets-Ord G* at pixel resolution, using spacial deconvolution


    Parameters
    ----------
    - X: array of data containing counts of events per kernel.
    - msk: ROI mask
    """
    #from scipy.stats import norm
    from scipy import special

    # local neighborhood (to first rectangular neighbors )
    box = np.ones((3, 3))
    box[1, 1] = 0
    
    if cuda:
        
        # transfer arrays to GPU
        Xs = tocuda(X)
        ms = tocuda(msk)
        bs = tocuda(box)
        
        # local number of neighbors in each pixel
        sw = torch.round(cudaconv2d(ms, bs)[ms > 0])
        sw = tofloat(sw.cpu().numpy())
        
        # weighted sum of x in each neighborhood
        wx = torch.round(cudaconv2d(Xs, bs)[ms > 0])
        wx = tofloat(wx.cpu().numpy())
    
        # x values in the roi
        aux = Xs[msk > 0]
        x = tofloat(aux.cpu().numpy())
        
        del Xs, ms, bs, aux
        torch.cuda.empty_cache()
        
    else:        

        # local number of neighbors in each pixel
        sw = tofloat(np.rint(fftconvolve(msk, box, mode='same')[msk > 0]))
    
        # weighted sum of x in each neighborhood
        wx = tofloat(np.rint(fftconvolve(X, box, mode='same')[msk > 0]))
    
        # x values in the roi
        x = tofloat(X[msk > 0])

    # do stats (inside ROI)
    n = len(x)
    s = np.std(x)
    u = np.sqrt((n*sw - (sw**2))/(n-1))
    aux = wx - np.mean(x)*sw
    auy = s*u
    z = np.divide(aux, auy, out=np.ones(aux.shape)*np.nan, where=(auy != 0))
        
    #p = 2*norm.sf(np.abs(z))
    p = 2*(1 - special.ndtr(np.abs(z)))
    p[np.isnan(p)] = 1
    sig = np.sign(z)
    sig[p > 0.05] = 0

    # get (flat) indices of terms in the roi
    inx = np.arange(np.prod(msk.shape)).reshape(msk.shape)[msk > 0]

    # revert to full array forms (padded with np.nan)
    aux = np.ones(np.prod(msk.shape))*np.nan
    aux[inx] = z
    z = aux.reshape(msk.shape)

    # aux = np.ones(np.prod(msk.shape))*np.nan
    # aux[inx] = p
    # p = aux.reshape(msk.shape)

    aux = np.ones(np.prod(msk.shape))*np.nan
    aux[inx] = sig
    hot = aux.reshape(msk.shape)

    return(tofloat(z), tofloat(hot))


def nndist(rcx, rcy):
    """
    Nearest Neighbor Distance index of all points in landscape

    ref_nndist is the mean NNDist of ref cells to other ref cells
    test_nndist is the mean NNDist of ref cells to test cells

    Index: v = log(test_NNDist/ref_NNDist)

    (*) v > 0 indicates ref and test cells are segregated
    (*) v ~ 0 indicates ref and test cells are well mixed
    (*) v < 0 indicates ref cells are individually infiltrated
              (ref cells are closer to test cells than other ref cells)

    Parameters
    ----------
    - rcx array of coordinates for reference class.
    - rcy: array of coordinates for test class.
    """

    from scipy.spatial import KDTree
    
    v = 0
    
    if (len(rcx) > 1):

        # get nearest neighbor distances of ref cells with their own type
        dnnxx, _ = KDTree(rcx).query(rcx, k=[2])
        mdnnxx = np.mean(dnnxx)
    
        if ((mdnnxx > 0) and (len(rcy) > 0)):
            
            # get nearest neighbor distances to test cells
            dnnxy, _ = KDTree(rcy).query(rcx, k=[1])
    
            # gets ratio of mean NNDist
            v = np.mean(dnnxy) / mdnnxx
        
    if (v > 0):
        nndi = np.log10(v)
    else:
        nndi = np.nan

    return(nndi)


def nndist_array(rcx, rcy, N, kernel, cuda=False):
    """
    Nearest Neighbor Distance index using deconvolution method

    ref_nndist is the mean NNDist of ref cells to other ref cells
    test_nndist is the mean NNDist of ref cells to test cells

    Index: v = log(test_NNDist/ref_NNDist)

    (*) v > 0 indicates ref and test cells are segregated
    (*) v ~ 0 indicates ref and test cells are well mixed
    (*) v < 0 indicates ref cells are individually infiltrated
              (ref cells are closer to test cells than other ref cells)

    Parameters
    ----------
    - rcx array of coordinates for reference class.
    - rcy: array of coordinates for test class.
    - N: 2D array with smooth reference cell abundance
    - kernel: small kernel array for deconvolution smoothing
    """

    from scipy.spatial import KDTree
    
    v = np.nan*np.ones(N.shape)
    
    if ((len(rcx) > 1) & (len(rcy) > 0)):

        # get nearest neighbor distances of ref cells with their own type
        dnnxx, innxx = KDTree(rcx).query(rcx, k=[2])
        # turns into array form
        nnxx = tofloat(np.zeros((N.shape[0], N.shape[1])))
        nnxx[rcx[:, 0], rcx[:, 1]] = tofloat(dnnxx[:, 0])
        
        # get nearest neighbor distances to test cells
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
            
            # local mean of NNdistance (dividing by local number of ref cells)
            aux = cudaconv2d(xs, ks)
            aux[aux < 0] = 0
            mdnnxx = torch.full_like(xs, fill_value=np.nan)
            mdnnxx[Ns > 0] = aux[Ns > 0]/Ns[Ns > 0]
            
            # local mean of NNdistance (dividing by local number of ref cells)
            aux = cudaconv2d(ys, ks)
            aux[aux < 0] = 0
            mdnnxy = torch.full_like(ys, fill_value=np.nan)
            mdnnxy[Ns > 0] = aux[Ns > 0]/Ns[Ns > 0]
    
            # gets (locally) ratio of mean NNDist
            aux = torch.full_like(xs, fill_value=np.nan)
            aux[mdnnxx > 0] = mdnnxy[mdnnxx > 0] / mdnnxx[mdnnxx > 0]
            aux[aux <= 0] = np.nan
            v = torch.log10(aux).cpu().numpy()
            
            del xs, ys, ks, Ns, aux, mdnnxx, mdnnxy
            torch.cuda.empty_cache()
        
        else:  

            # local mean of NNdistance (dividing by local number of ref cells)
            aux = fftconvolve(nnxx, kernel, mode='same')
            aux[aux < 0] = 0
            mdnnxx = np.divide(aux, N, out=np.zeros(N.shape), where=(N > 0))
        
            # local mean of NNdistance (dividing by local number of ref cells)
            aux = fftconvolve(nnxy, kernel, mode='same')
            aux[aux < 0] = 0
            mdnnxy = np.divide(aux, N, out=np.zeros(N.shape), where=(N > 0))

            # gets (locally) ratio of mean NNDist
            v = np.divide(mdnnxy, mdnnxx, out=np.zeros(N.shape), 
                          where=(mdnnxx > 0))
            v[v <= 0] = np.nan
            v = np.log10(v)
        
        
    return(tofloat(v))


def attraction_T_biv(rcx, lambday, ny):
    """
    Attraction Enrichment Functions score T

    Array form, evaluated in each kernel, of Attraction T(r) function
    evaluated at r given by subkernel scale. Done between 'ref' points and
    'test' points, this measure is NOT symetrical:
    (*) T = 0 indicates uniform mixing between 'test' and 'ref' cells
    (*) T = +1 indicates clustering of 'test' cells around 'ref' cells
            (i.e. more attraction than 'ref' cells around 'ref' cells)
    (*) T = -1 indicates dispersion of 'test' cells around 'ref' cells
            (i.e. less attraction than 'ref' cells around 'ref' cells)

    Parameters
    ----------
    - rcx: array of 'ref' coordinates (sample points)
    - lambday: total density of 'test' points (parental mean density)
    - ny: 2D array with test point density at subkernel scale (raster of 
                                                               densities to be
                                                               sampled)
    """
    
    from scipy import stats
    import warnings
    
    t = np.nan
    p = np.nan
    Nx = len(rcx)
    
    if (Nx > 0):
        # mean 'test' point density around 'ref' points 
        # relative to their global density 'lambday'
        aux = ny[rcx[:, 0], rcx[:, 1]].ravel()
        
        # p is the probability of the t-test with the null 
        # hypothesis that the two point distributions are equivalent
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = stats.ttest_1samp(aux, 
                                    popmean=lambday,
                                    nan_policy='omit')
        t = res.statistic
        p = res.pvalue
    
    return(np.sign(t)*(p < 0.05))


def attraction_T_array_biv(rcx, Ny, dy, kernel, cuda=False):
    """
    Attraction Enrichment Functions score T

    Array form, evaluated in each kernel, of Attraction T(r) function
    evaluated at r given by subkernel scale. Done between 'ref' points and
    'test' points, this measure is NOT symetrical:
    (*) T = 0 indicates uniform mixing between 'test' and 'ref' cells
    (*) T = +1 indicates clustering of 'test' cells around 'ref' cells
            (i.e. more attraction than 'ref' cells around 'ref' cells)
    (*) T = -1 indicates dispersion of 'test' cells around 'ref' cells
            (i.e. less attraction than 'ref' cells around 'ref' cells)

    Parameters
    ----------
    - rcx: array of 'ref' coordinates (sample points)
    - Ny: array of abundance of 'test' points at large scale 
          (parental mean density)
    - dy: 2D array with test point density at subkernel scale (raster of 
                                                               densities to be
                                                               sampled)
    - kernel: kernel at which stats are calculated
    """
    
    from scipy import special
    
    pout = np.full(Ny.shape, np.nan)
    
    D = Ny/np.sum(kernel)
    N = toint(Ny)
    
    if (len(rcx) > 0):
        
        if cuda:
            # create tensor objects
            dys = tocuda(dy)
            ks = tocuda(kernel)
            Ds = tocuda(D)
            Ns = tocuda(N)
            
            # do t-test on array terms
            m2d = torch.full_like(Ns, fill_value=np.nan)
            m2d[Ns > 0] = torch.mul(dys, dys)[Ns > 0] / Ns[Ns > 0]  
            
            numer = cudaconv2d(dys, ks) - torch.mul(Ds, Ns)
            denom = torch.sqrt(cudaconv2d(torch.mul(dys, dys), ks) - m2d)
            
            inx = (Ns > 0) & (denom != 0)
            aux = torch.full_like(Ns, fill_value=np.nan)
            aux[inx] = numer[inx] / denom[inx]
            t = aux.cpu().numpy()
            
            del dys, ks, Ds, Ns, m2d, numer, denom, aux, inx
            torch.cuda.empty_cache()
            
        else:
            
            # NOTE: THERE IS A BUG HERE... 
            # THE TERM (md2 - m2d) IS NEGATIVE SOMETIMES!
            
            # do t-test on array terms
            md = fftconvolve(dy, kernel, mode='same')
            md2 = fftconvolve(np.multiply(dy, dy), kernel, mode='same')
            m2d = np.divide(np.multiply(md, md), N, 
                            out=np.full(N.shape, np.nan), where=(N > 0))   
            t = np.divide(md - np.multiply(D, N), 
                          np.sqrt(md2 - m2d), 
                          out=np.full(N.shape, np.nan), where=(N > 0))  
            
        p = tofloat(np.full(Ny.shape, np.nan))
    
        # for elements with N > 30 use normal distribution approximation
        (r,c) = np.where(N>30)
        p[r,c] = 2*(1 - special.ndtr(np.abs(t[r,c])))
        
        # for elements with N < 30 use standard t distribution approximation
        (r,c) = np.where(N<30)
        p[r,c] =2*(1 - special.stdtr(N[r,c], np.abs(t[r,c])))
        
        # this part is slow, so using only 100 replicants
        #s = np.random.standard_t(N, (100, N.shape[0], N.shape[1]))
        #p = np.divide(np.sum(s < t, axis=0), N, 
        #              out=np.full(N.shape, np.nan), where=(N > 0))  
        
        pout = np.sign(t)*(p < 0.05)
        
    return(tofloat(pout))


def ripleys_K(rc, n):
    """
    Ripley's K index of all points in landscape

    Array form, evaluated in each kernel, of Regular Ripley's K(r) function
    evaluated at r given by subkernel scale:
    (*) K ~ pi*r^2 indicates uniform distribution of points across kernel
    (*) K > pi*r^2 indicates clustering of points at subkernel scale in kernel
    (*) K < pi*r^2 indicates dispersion of points at subkernel scale in kernel

    Parameters
    ----------
    - rc: array of coordinates
    - n: 2D array with point abundance at subkernel scale
    """

    # Number of points 
    N = len(rc)
    
    ripley = np.nan
    
    if (N > 1):
        
        # area of landscape
        A = np.sum(n>0)
    
        # number of neighbors in kernel around each point
        Ir = np.zeros((n.shape[0], n.shape[1]))
        Ir[rc[:, 0], rc[:, 1]] = n[rc[:, 0], rc[:, 1]] - 1
        
        # number of pair comparisons 
        npairs = N*(N - 1)/2
        
        # Ripley's K sum (do sum of Ir for each kernel)
        ripley = A * np.sum(Ir)/npairs
          
    return(ripley)


def ripleys_K_array(rc, n, N, kernel, cuda=False):
    """
    Ripley's K index using deconvolution method

    Array form, evaluated in each kernel, of Regular Ripley's K(r) function
    evaluated at r given by subkernel scale:
    (*) K ~ pi*r^2 indicates uniform distribution of points across kernel
        i.e. indicates evidence of Complete Spatial Randomness (CSR)
    (*) K > pi*r^2 indicates clustering of points at subkernel scale in kernel
    (*) K < pi*r^2 indicates dispersion  of points at subkernel scale in kernel
        i.e. this indicates a more regular process than expected from CSR

    Parameters
    ----------
    - rc: array of coordinates
    - n: 2D array with point abundance at subkernel scale
    - N: 2D array with point abundance at kernel scale
    - kernel: kernel array for deconvolution smoothing
    """

    ripley = np.ones(N.shape)*np.nan
    
    if (len(rc) > 1):
        
        # number of neighbors (at subkernel scale) around each point
        Ir = np.zeros((n.shape[0], n.shape[1]))
        Ir[rc[:, 0], rc[:, 1]] = n[rc[:, 0], rc[:, 1]] - 1
    
        if cuda:
            # transfer arrays to GPU
            Ns = tocuda(N)
            Irs = tocuda(Ir)
            ks = tocuda(kernel)
            
            # number of pair comparisons in each kernel (degenerate)
            npairs = torch.mul(Ns, Ns - 1)/2
        
            # Ripley's K sum (do sum of Ir for each kernel)
            aux = torch.round(cudaconv2d(Irs, ks))
            aux[aux<0]=0
            
            auy = torch.full_like(npairs, fill_value=np.nan)
            auy[npairs > 0] = aux[npairs > 0] / npairs[npairs > 0] 
            ripley = (auy*torch.sum(ks)).cpu().numpy()
            
            del Ns, Irs, ks, npairs, aux, auy
            torch.cuda.empty_cache()
            
        else:      
            # number of pair comparisons in each kernel (degenerate)
            npairs = np.multiply(N, N - 1)/2
        
            # Ripley's K sum (do sum of Ir for each kernel)
            aux = np.rint(fftconvolve(Ir, kernel, mode='same'))
            aux[aux<0]=0
            ripley = np.sum(kernel) * np.divide(aux, npairs,
                                                out=np.zeros(Ir.shape),
                                                where=(npairs > 0))     
    
    return(tofloat(ripley))


def ripleys_K_biv(rcx, nx, rcy, ny):
    """
    Bivariate Ripley's K index of all points in landscape

    Array form, evaluated in each kernel, of Bivariate Ripley's K(r) function
    evaluated at r given by subkernel scale. Done between 'ref' points and
    'test' points, this measure is NOT symetrical:
    (*) K ~ pi*r^2 indicates uniform mixing between 'test' and 'ref' cells
    (*) K < pi*r^2 indicates clustering of 'test' cells around 'ref' cells
                   (i.e. more clustering than 'ref' cells around 'ref' cells)
    (*) K > pi*r^2 indicates dispersion of 'test' cells around 'ref' cells
                   (i.e. less clustering than 'ref' cells around 'ref' cells)

    Parameters
    ----------
    - rcx: array of 'ref' coordinates
    - nx: 2D array with ref point abundance at subkernel scale
    - rcx: array of 'test' coordinates
    - ny: 2D array with test point abundance at subkernel scale
    """
    
    ripley = np.nan
    
    if (len(rcx) > 0 and len(rcy) > 0):
      
        # area of landscape
        A = np.sum(nx > 0)
        
        # 'test' neighbors (at subkernel scale) around each 'ref' point
        Ir = np.zeros((nx.shape[0], nx.shape[1]))
        Ir[rcx[:, 0], rcx[:, 1]] = ny[rcx[:, 0], rcx[:, 1]]
        
        # number of pair comparisons in each kernel (non-degenerate)
        npairs = len(rcx)*len(rcy)
    
        # Ripley's K sum (do sum of Ir for each kernel)
        ripley = A * np.sum(Ir)/npairs
    
    return(ripley)


def ripleys_K_array_biv(rcx, nx, Nx, ny, Ny, kernel, cuda=False):
    """
    Bivariate Ripley's K index using deconvolution method

    Array form, evaluated in each kernel, of Bivariate Ripley's K(r) function
    evaluated at r given by subkernel scale. Done between 'ref' points and
    'test' points, this measure is NOT symetrical:
    (*) K ~ pi*r^2 indicates uniform mixing between 'test' and 'ref' cells
    (*) K < pi*r^2 indicates clustering of 'test' cells around 'ref' cells
                   (i.e. more clustering than 'ref' cells around 'ref' cells)
    (*) K > pi*r^2 indicates dispersion of 'test' cells around 'ref' cells
                   (i.e. less clustering than 'ref' cells around 'ref' cells)

    Parameters
    ----------
    - rcx array of 'ref' coordinates
    - nx: 2D array with ref point abundance at subkernel scale
    - Nx: 2D array with ref point abundance at kernel scale
    - ny 2D array with test point abundance at subkernel scale
    - Ny: 2D array with test point abundance at kernel scale
    - kernel: kernel array for deconvolution smoothing
    """
    
    ripley = np.ones(nx.shape)*np.nan
    
    if (len(rcx) > 0):
        
        # 'test' neighbors (at subkernel scale) around each 'ref' point
        Ir = np.zeros((Nx.shape[0], Nx.shape[1]))
        Ir[rcx[:, 0], rcx[:, 1]] = ny[rcx[:, 0], rcx[:, 1]]
        
        if cuda:
            # transfer arrays to GPU
            Nxs = tocuda(Nx)
            Nys = tocuda(Ny)
            Irs = tocuda(Ir)
            ks = tocuda(kernel)
            
            # number of pair comparisons in each kernel (non-degenerate)
            npairs = torch.mul(Nxs, Nys)
    
            # Ripley's K sum (do sum of Ir for each kernel)
            aux = torch.round(cudaconv2d(Irs, ks))
            aux[aux<0]=0
            
            auy = torch.full_like(npairs, fill_value=np.nan)
            auy[npairs > 0] = aux[npairs > 0] / npairs[npairs > 0] 
            ripley = (auy*torch.sum(ks)).cpu().numpy()
            
            del Nxs, Nys, Irs, ks, npairs, aux, auy
            torch.cuda.empty_cache()
            
        else:

            # 'test' neighbors (at subkernel scale) around each 'ref' point
            Ir = np.zeros((Nx.shape[0], Nx.shape[1]))
            Ir[rcx[:, 0], rcx[:, 1]] = ny[rcx[:, 0], rcx[:, 1]]
            # number of pair comparisons in each kernel (non-degenerate)
            npairs = np.multiply(Nx, Ny)
    
            # Ripley's K sum (do sum of Ir for each kernel)
            aux = np.abs(np.rint(fftconvolve(Ir, kernel, mode='same')))
            aux[aux<0]=0
            ripley = np.sum(kernel) * np.divide(aux, npairs,
                                                out=np.zeros(Ir.shape),
                                                where=(npairs > 0))
    return(tofloat(ripley))


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



