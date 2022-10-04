import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
import os
import math
import seaborn as sns
import imageio as imio
import scipy.stats as sts
import warnings

from KDEpy import FFTKDE
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from skimage.morphology import remove_small_objects, label
from scipy.ndimage.morphology import binary_fill_holes
from scipy.spatial import KDTree
from scipy.signal import fftconvolve
from IPython.display import clear_output
from itertools import combinations
# from itertools import combinations_with_replacement
# from itertools import permutations, product
from statsmodels.sandbox.stats.multicomp import multipletests


def fmt(x, pos):

    a, b = '{:.2e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)


def fmt_simple(x, pos):
    a, b = '{:.2e}'.format(x).split('e')
    b = int(b)
    return r'$10^{{{}}}$'.format(b)


def ax_index(i, siz):

    if (siz[0] == 1):
        return i
    else:
        return np.unravel_index(i, siz)


def update_progress(progress, msg):
    bar_length = 25

    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
    if progress < 0:
        progress = 0
    if progress >= 1:
        progress = 1

    block = int(round(bar_length * progress))
    clear_output(wait=True)
    text = "[{0}] {1:.1f}%".format("#" * block + "-" * (bar_length - block),
                                   progress * 100)
    print(text + "  <<< " + msg)


def round_down(num, divisor):
    return (np.around(num - (num % divisor)).astype(int))


def circle(r):
    y, x = np.ogrid[-r: r, -r: r]
    return(1*(x**2 + y**2 < r**2))


def raster_edges(sample, edge, binsiz):

    # Pixel coverage of coordinates in data (row, col)
    mins = [int(np.ceil(sample.row_min - edge)),
            int(np.ceil(sample.col_min - edge))]
    maxs = [int(np.ceil(sample.row_max + edge)),
            int(np.ceil(sample.col_max + edge))]

    # reference lower corner (row, col)
    refpix = [round_down(mins[0], 10), round_down(mins[1], 10)]

    # adjusted size of landscape (row, col)
    maxpix = [maxs[0] - refpix[0], maxs[1] - refpix[1]]

    # coordinates of quadrats (pixels)
    redges = np.arange(0, maxpix[0] + binsiz, binsiz)
    cedges = np.arange(0, maxpix[1] + binsiz, binsiz)

    # final size of landscape (pix)
    syspix = [redges[-1] - redges[0], cedges[-1] - cedges[0]]

    return([refpix, maxpix, syspix, redges, cedges])


def scarce_to_raster(msg, cell_data, classes, namtiff,
                     shape, do_raster=False):
    # read scarce coordinates from file and create a raster matrix

    # Create an array with cell data coordinates
    if do_raster or not os.path.exists(namtiff):
        imout = np.zeros([shape[0], shape[1], 3], dtype=np.int32)
        for i, row in cell_data.iterrows():
            if (i % 1000 == 0):
                update_progress(i / len(cell_data.index), msg)
            val = classes.loc[classes['class_code'] ==
                              row['class_code']].class_val
            imout[row['row'], row['col'], 0] = np.uint32(row.cell_id)
            imout[row['row'], row['col'], 1] = np.uint32(val)[0]
            # imout[row['row'], row['col'], 2] = 0
        # saves a raster image with phenotype values
        imio.imwrite(namtiff, imout)

    # array of cell indices
    cell_img = imio.imread(namtiff)[:, :, 0]
    # array of cell classes
    class_img = imio.imread(namtiff)[:, :, 1]
    update_progress(1, msg + '... done!')

    return([cell_img, class_img])


def scarce_csv(namcsv, shape, ref, scale):
    # read scarce coordinates from file and create a raster matrix

    cell_data = pd.read_csv(namcsv)
    # drop duplicate entries
    # (shuffling first to keep a random copy of each duplicate)
    cell_data = cell_data.sample(frac=1.0).drop_duplicates(['x', 'y'])
    cell_data = cell_data.reset_index(drop=True)
    # cell_data = cell_data[['class_code', 'x', 'y']]
    # generate cell info
    cell_data['cell_id'] = cell_data.index + 1

    # round coordinates
    cell_data['col'] = np.uint32(np.rint(cell_data['x']))
    cell_data['row'] = np.uint32(np.rint(cell_data['y']))

    # shift coordinates to reference point
    cell_data['row'] = cell_data['row'] - ref[0]
    cell_data['col'] = cell_data['col'] - ref[1]
    # scale coordinates into um and transforms y axis
    cell_data['x'] = cell_data['col']*scale
    cell_data['y'] = (shape[0] - cell_data['row'])*scale

    return(cell_data)


def load_mask(msg, namfig, namtiff, shape, ref, scale, do_raster=False):
    # read scarce coordinates from file and create a raster matrix

    from skimage import measure

    # Create an array with mask data coordinates
    if do_raster or not os.path.exists(namtiff):

        th = 127
        aux = imio.imread(namfig)

        # get a binary image of the mask
        msk_img = np.zeros(aux.shape)
        msk_img[aux > th] = 1

        # label blobs in mask image
        blobs_labels = measure.label(msk_img, background=0, connectivity=2)

        # get data coordinates and labels for mask
        rows, cols = np.where(msk_img > 0)
        msk_data = pd.DataFrame({'blob': blobs_labels[rows, cols],
                                 'row': rows,
                                 'col': cols})

        # shift coordinates to reference point
        msk_data['row'] = msk_data['row'] - ref[0]
        msk_data['col'] = msk_data['col'] - ref[1]

        msk_img = np.zeros([shape[0], shape[1], 3], dtype=np.int32)
        for i, row in msk_data.iterrows():
            if (i % 1000 == 0):
                update_progress(i / len(msk_data.index), msg)
            r = np.uint32(np.rint(row['row']))
            c = np.uint32(np.rint(row['col']))
            msk_img[r, c, 0] = 1
            msk_img[r, c, 1] = np.uint32(row.blob)
            # msk_img[r, c, 2] = 0
        # saves a raster image with phenotype values
        imio.imwrite(namtiff, msk_img)

    # array of cell indices
    msk_img = imio.imread(namtiff)

    # array data
    rows, cols = np.where(msk_img[:, :, 0] > 0)
    msk_data = pd.DataFrame({'blob': msk_img[rows, cols, 1],
                             'row': rows,
                             'col': cols})

    # round coordinates
    msk_data['col'] = np.uint32(np.rint(msk_data['col']))
    msk_data['row'] = np.uint32(np.rint(msk_data['row']))

    CS = plt.contour(msk_img[:, :, 1], [0.99], linewidths=2, colors='black')
    plt.close()
    msk_contour = np.vstack(CS.allsegs[0])

    return(msk_data, msk_img, msk_contour)


def csv_mask_filter(data, target_code, new_code, nor_code,
                    shape, bw, toplev=1.5, blobs=True):

    cell_data = data.copy()
    cell_data['orig_class_code'] = cell_data['class_code']
    cell_data['i'] = cell_data.index

    # redefine all target cells
    cell_data.loc[(cell_data['class_code'] == target_code),
                  'class_code'] = new_code

    # subset data to just target cells
    aux = cell_data.loc[cell_data['class_code'] == new_code]
    irc = np.array(aux[['i', 'row', 'col']])

    # do KDE on pixel locations of target cells (e.g. tumor cells)
    [r, c, z] = KDE(aux, shape, 1, 1, bw)

    # get top orders of magnitud in density
    aux = np.around(np.log10(z), 2)
    th = 10**(np.min(aux[aux > (np.max(aux) - toplev)]))

    # generate a binary mask
    mask = kde_mask(z, th, np.pi*(bw)*(bw), fill_holes=True)

    # get target cells outside the mask (low density regions)
    ii = np.ones(len(cell_data))
    ii[irc[:, 0]] = mask[irc[:, 1], irc[:, 2]]
    cell_data['tp'] = (ii < 1)

    # redefine posibly misclasified target cells by KDE filter
    # (those in low density regions)
    cell_data.loc[cell_data['tp'], 'class_code'] = nor_code

    if blobs:
        # redefine possibly misclasified target cells by mask filter
        # (those inside mask blobs)
        cell_data.loc[(cell_data['orig_class_code'] == target_code) &
                      (cell_data['blob'] > 0), 'class_code'] = target_code

    return(cell_data.drop(columns=['i', 'tp']).reset_index(drop=True), mask)


def filter_cells(cell_data, mask):

    aux = cell_data.copy().reset_index()
    aux['i'] = aux.index

    irc = np.array(aux[['i', 'row', 'col']])

    # get target cells in data that are outside the mask
    ii = np.ones(len(aux))
    ii[irc[:, 0]] = mask[irc[:, 1], irc[:, 2]]
    aux['tp'] = (ii < 1)

    # drop cells outside of mask
    aux.drop(aux[aux['tp']].index, inplace=True)

    return(aux.drop(columns=['i', 'tp']).reset_index(drop=True))


def kde_mask(z, th, minsize, fill_holes):

    # clean up the tumor mask from small regions and holes
    lab = label((z > th).astype(int), connectivity=2) > 0
    if fill_holes:
        aux = binary_fill_holes(remove_small_objects(lab,
                                                     min_size=minsize,
                                                     connectivity=2))
    else:
        aux = remove_small_objects(lab, min_size=minsize, connectivity=2)
    mask = (aux > 0).astype(int)
    return(mask)


def mask_filter(mask, centers):

    # extract coordinate points in mask
    maskpoints = np.array(np.where(mask > 0)).T
    maskpointslist = [(r, c) for [r, c] in maskpoints]
    pointslist = [(int(np.around(r, decimals=-1)),
                   int(np.around(c, decimals=-1))) for [r, c] in centers]
    overpointslist = list(set(pointslist).intersection(set(maskpointslist)))

    return(np.array(overpointslist))


def set_ticklabels_spacing(ax, num):

    n = min([math.ceil(len(ax.xaxis.get_ticklabels())/num),
             math.ceil(len(ax.yaxis.get_ticklabels())/num)])

    for index, lab in enumerate(ax.xaxis.get_ticklabels()):
        if index % n != 0:
            lab.set_visible(False)

    for index, lab in enumerate(ax.yaxis.get_ticklabels()):
        if index % n != 0:
            label.set_visible(False)

    return(num)


def plot_landscape_scarce(ax, dat, col, lab, units,
                          xedges, yedges, spoint=1, fontsiz=20):

    scatter = ax.scatter(dat.x, dat.y, c=col,
                         label=lab, s=spoint, marker='.', linewidths=0)
    ax.axis('square')
    ax.set_xticks(xedges, minor=False)
    ax.set_yticks(yedges, minor=False)
    ax.minorticks_on()
    ax.grid(which='major', linestyle='--', linewidth='0.3', color='black')
    ax.grid(which='minor', linestyle=':', linewidth='0.1', color='black')
    ax.set_xlim(min(xedges), max(xedges))
    ax.set_ylim(min(yedges), max(yedges))
    ax.set_xticklabels(xedges, rotation=90, fontsize=fontsiz)
    ax.set_yticklabels(yedges, fontsize=fontsiz)
    ax.set_facecolor('white')
    ax.set_xlabel(units, fontsize=fontsiz)
    ax.set_ylabel(units, fontsize=fontsiz)
    set_ticklabels_spacing(ax, 10)

    return scatter


def plot_rgb(ax, rgb, units, xedges, yedges, xticks, yticks, fontsiz=20):

    i = ax.imshow(rgb)
    ax.axis('square')
    ax.set_xticks(xedges, minor=False)
    ax.set_yticks(yedges, minor=False)
    ax.minorticks_on()
    ax.grid(which='major', linestyle='--', linewidth='0.3', color='black')
    ax.grid(which='minor', linestyle=':',  linewidth='0.1', color='black')
    ax.set_xlim(min(xedges), max(xedges))
    ax.set_ylim(max(yedges), min(yedges))
    ax.set_xticklabels(xticks, rotation=90, fontsize=fontsiz)
    ax.set_yticklabels(yticks[::-1], fontsize=fontsiz)
    ax.set_xlabel(units, fontsize=fontsiz)
    ax.set_ylabel(units, fontsize=fontsiz)
    set_ticklabels_spacing(ax, 10)

    return i


def plot_landscape(ax, dat, dmin, dmax, units,
                   xedges, yedges, xticks, yticks,
                   colmap=20, fontsiz=24):

    i = ax.imshow(dat, vmin=dmin, vmax=dmax, cmap=colmap)
    ax.axis('square')
    ax.set_xticks(xedges, minor=False)
    ax.set_yticks(yedges, minor=False)
    ax.minorticks_on()
    ax.grid(which='major', linestyle='--', linewidth='0.3', color='black')
    ax.set_xlim(min(xedges), max(xedges))
    ax.set_ylim(max(yedges), min(yedges))
    ax.set_xticklabels(xticks, rotation=90, fontsize=fontsiz)
    ax.set_yticklabels(yticks[::-1], fontsize=fontsiz)
    ax.set_xlabel(units, fontsize=fontsiz)
    ax.set_ylabel(units, fontsize=fontsiz)
    set_ticklabels_spacing(ax, 10)

    return i


def shuffle_batch_ids(arr):

    import random

    ids_1 = np.unique(arr[arr > 0])
    ids_2 = ids_1.copy()
    random.shuffle(ids_2)

    idmap = np.transpose(np.array([ids_1, ids_2]))
    idmap = np.append([[0, 0]], idmap, axis=0)

    def remap(i):
        return idmap[i, 1]
    remap_vectorized = np.vectorize(remap)

    aux = remap_vectorized(arr).astype(np.float32)

    return(aux)


def estimate_cell_size(class_img, binsiz):

    # generates a box-circle kernel
    circ = circle(int(np.ceil(binsiz/2)))

    # convolve array of all-cell locations with kernel
    A = fftconvolve(1.0*(class_img > 0), circ)

    # the typical area of a cell is estimated as
    # area of circle / max number of cells in a circle
    acell = np.sum(circ)/np.max(A)
    # max radius of a typical cell
    # (assuming close packing in region with max cell density)
    rcell = round(np.sqrt(acell/np.pi), 2)

    return([acell, rcell])


def KDE(data, shape, rbins, cbins, bw):

    C, R = np.meshgrid(np.arange(-cbins, shape[1] + cbins, cbins),
                       np.arange(-rbins, shape[0] + rbins, rbins))
    grid = np.stack([R.ravel(), C.ravel()]).T

    # Compute the kernel density estimate
    coords = np.array(data[['row', 'col']])
    points = FFTKDE(kernel='gaussian', bw=bw).fit(coords).evaluate(grid)
    points = points/sum(points)

    # grid has shape (obs, dims), points has shape (obs, 1)
    row, col = np.unique(grid[:, 0]), np.unique(grid[:, 1])
    z = points.reshape([len(row), len(col)])
    row, col, z = row[1:-1], col[1:-1], z[1:-1, 1:-1]

    return([row, col, z])


def KDE_regions_mask_only(data, shape, subbinsiz, bw, minsize,
                          fill_holes=True, all_cells=True):

    # Compute the kernel density estimate
    [row, col, z] = KDE(data, shape, subbinsiz, subbinsiz, bw)

    # threshold for landscape edge
    aux = np.around(np.log10(z), 2)

    if (all_cells):
        # get values above background
        bux = np.min(aux[aux > np.min(aux)])
        th = 10**bux
    else:
        # get top order of magnitud
        cux = np.min(aux[aux > (np.max(aux) - 1.5)])
        th = 10**cux

    # generate a binary mask
    mask = kde_mask(z, th, minsize, fill_holes=fill_holes)

    return(mask)


def KDE_regions(data, classes, shape, cedges, redges, xedges, yedges,
                spoint, ar, scale, units, subbinsiz, bw,
                minsize, fontsiz=20, titsiz=24,
                fill_holes=True, all_cells=True, toplev=1.0):

    # Compute the kernel density estimate
    [row, col, z] = KDE(data, shape, subbinsiz, subbinsiz, bw)

    # threshold for landscape edge
    aux = np.around(np.log10(z), 2)
    if (all_cells):
        # get values above background
        th = 10**np.min(aux[aux > np.min(aux)])
    else:
        # get top order of magnitud
        th = 10**np.min(aux[aux > (np.max(aux) - toplev)])
    levs = 10**np.arange(np.floor(np.log10(th)), np.ceil(np.max(aux))+1)

    # generate a binary mask
    mask = kde_mask(z, th, minsize, fill_holes=fill_holes)

    # plot mask regions
    fig, contourpix = plot_kde_mask(data, classes, scale, units,
                                    row, col, z, th, levs, mask,
                                    cedges, redges, xedges, yedges, spoint, ar,
                                    fontsiz, titsiz)
    contourpoints = np.array((contourpix[:, 0] * scale,
                              (mask.shape[0] - contourpix[:, 1])*scale)).T

    return([mask, contourpix, contourpoints, fig])


def plot_kde_mask(cellData, classes, scale, units,
                  r, c, z, th, levs, mask,
                  cedges, redges, xedges, yedges, spoint, ar,
                  fontsiz=20, titsiz=24):

    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    x = c*scale
    y = (mask.shape[0] - r)*scale

    fig, ax = plt.subplots(2, 2, figsize=(2*12, 2*math.ceil(12/ar)),
                           facecolor='w', edgecolor='k')

    aux = z.ravel()
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # plot (ranked) density values in the landscape
    ax[0, 0].set_title('Ranked KDE values', fontsize=titsiz)
    ax[0, 0].plot(np.arange(0, len(aux)), np.sort(aux), 'ok', ms=2)
    ax[0, 0].axhline(y=th, linewidth=1, color='r', linestyle='--')
    ax[0, 0].set_xticks(np.arange(0, len(aux)+1, len(aux)/5).tolist())
    ax[0, 0].tick_params(axis='x', labelsize=fontsiz, labelrotation=90)
    ax[0, 0].tick_params(axis='y', labelsize=fontsiz)
    ax[0, 0].set_yscale('log')
    ax[0, 0].set_xlabel('rank', fontsize=fontsiz)
    ax[0, 0].set_ylabel('KDE', fontsize=fontsiz)
    ax[0, 0].text(0.05, 0.95,
                  "th = " + np.format_float_scientific(th,
                                                       unique=False,
                                                       precision=2),
                  transform=ax[0, 0].transAxes, fontsize=14,
                  verticalalignment='top', bbox=props)

    # Plot the kernel density estimate with cells
    z[z < 0.975 * th] = 0.0

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        CS1 = ax[0, 1].contourf(x, y, z, levs,
                                locator=ticker.LogLocator(),
                                cmap='RdBu_r')
        # cbar = plt.colorbar(CS1, ax=ax[0,1],
        #             format = ticker.FuncFormatter(fmt),
        #             fraction=0.046, pad=0.04);
        axins = inset_axes(ax[0, 1],
                           width="5%",
                           height="25%",
                           loc='upper right',
                           borderpad=2)
        cbar = plt.colorbar(CS1, cax=axins,
                            format=ticker.FuncFormatter(fmt_simple))
        axins.yaxis.set_ticks_position('left')
        cbar.ax.tick_params(labelsize=fontsiz/1.25)
        ax[0, 1].set_aspect(aspect=1)
        ax[0, 1].contour(x, y, z, [th], linewidths=2, colors='yellow')
        ax[0, 1].set_title('KDE contours', fontsize=titsiz)
        # ax[0, 1].plot(cellData.x, cellData.y, 'ok', ms=spoint)
        ax[0, 1].set_xticks(xedges, minor=False)
        ax[0, 1].set_yticks(yedges, minor=False)
        ax[0, 1].minorticks_on()
        ax[0, 1].grid(which='major', linestyle='-', linewidth='0.3',
                      color='black')
        ax[0, 1].grid(which='minor', linestyle=':', linewidth='0.1',
                      color='black')
        ax[0, 1].set_xlim(min(xedges), max(xedges))
        ax[0, 1].set_ylim(min(yedges), max(yedges))
        ax[0, 1].set_xlabel(units, fontsize=fontsiz)
        ax[0, 1].set_ylabel(units, fontsize=fontsiz)
        ax[0, 1].set_xticklabels(xedges, rotation=90, fontsize=fontsiz)
        ax[0, 1].set_yticklabels(yedges, fontsize=fontsiz)
        set_ticklabels_spacing(ax[0, 1], 10)

        # Plot the th-level kde mask with cells
        for i, row in classes.iterrows():
            aux = cellData.loc[cellData['class_code'] == row['class_code']]
            fig = plot_landscape_scarce(ax[1, 0], aux,
                                        row.class_color, row.class_name,
                                        units, xedges, yedges,
                                        spoint=spoint, fontsiz=fontsiz)
        if (i % 2) != 0:
            ax[1, 0].grid(which='major', linestyle='--', linewidth='0.3',
                          color='black')
        ax[1, 0].set_aspect(aspect=1)
        ax[1, 0].contour(x, y, z, [th], linewidths=2, colors='black')
        ax[1, 0].set_title('Cell locations', fontsize=titsiz)
        ax[1, 0].legend(labels=classes.class_name,
                        loc='upper right',
                        markerscale=5, fontsize=30)

        # plot landscape with mask contour
        ax[1, 1].imshow(mask, 'RdBu_r')
        CS = ax[1, 1].contour(c, r, mask, [0.5],
                              linewidths=2, colors='yellow')
        # ax[1,1].plot(cellData.col, cellData.row, 'ow', ms=spoint);
        ax[1, 1].set_title('KDE Mask', fontsize=titsiz)
        ax[1, 1].set_aspect(aspect=1)
        ax[1, 1].set_xticks(cedges, minor=False)
        ax[1, 1].set_yticks(redges, minor=False)
        ax[1, 1].grid(which='major', linestyle='-', linewidth='0.5',
                      color='white')
        ax[1, 1].set_xlim(min(cedges), max(cedges))
        ax[1, 1].set_ylim(max(redges), min(redges))
        ax[1, 1].set_xticklabels(xedges, rotation=90, fontsize=fontsiz)
        ax[1, 1].set_yticklabels(yedges[::-1], fontsize=fontsiz)
        ax[1, 1].set_xlabel(units, fontsize=fontsiz)
        ax[1, 1].set_ylabel(units, fontsize=fontsiz)
        set_ticklabels_spacing(ax[1, 1], 10)

        plt.tight_layout()
        contourpoints = np.vstack(CS.allsegs[0])

    return([fig, contourpoints])


def KDE_levels(data, shape, scale, units, subbinsiz, bw,
               all_cells=True, toplev=1.0):

    # Compute the kernel density estimate
    [r, c, z] = KDE(data, shape, subbinsiz, subbinsiz, bw)

    # threshold for landscape edge
    aux = np.around(np.log10(z), 2)
    if (all_cells):
        # get values above background
        th = 10**np.min(aux[aux > np.min(aux)])
    else:
        # get top order of magnitud
        th = 10**np.min(aux[aux > (np.max(aux) - toplev)])
    levs = 10**np.arange(np.floor(np.log10(th)), np.ceil(np.max(aux))+1)

    # coordenate values
    x = c*scale
    y = (z.shape[0] - r)*scale

    return([x, y, z, levs, th])


def plot_KDE_levels(data, shape, xedges, yedges,
                    ar, scale, units, subbinsiz, bw,
                    minsize, fontsiz=20, titsiz=24,
                    all_cells=True, toplev=1.0):

    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    # Compute the kernel density estimate
    [r, c, z] = KDE(data, shape, subbinsiz, subbinsiz, bw)

    # threshold for landscape edge
    aux = np.around(np.log10(z), 2)
    if (all_cells):
        # get values above background
        th = 10**np.min(aux[aux > np.min(aux)])
    else:
        # get top order of magnitud
        th = 10**np.min(aux[aux > (np.max(aux) - toplev)])
    levs = 10**np.arange(np.floor(np.log10(th)), np.ceil(np.max(aux))+1)

    # plot mask levels
    x = c*scale
    y = (z.shape[0] - r)*scale
    z[z < 0.975 * th] = 0.0

    fig, ax = plt.subplots(1, 1, figsize=(12, math.ceil(12/ar)),
                           facecolor='w', edgecolor='k')

    # Plot the kernel density estimate with cells
    CS1 = ax.contourf(x, y, z, levs,
                      locator=ticker.LogLocator(), cmap='RdBu_r')
    axins = inset_axes(ax,
                       width="5%",
                       height="25%",
                       loc='lower left',
                       borderpad=2.5)
    cbar = plt.colorbar(CS1, cax=axins,
                        format=ticker.FuncFormatter(fmt))

    cbar.ax.tick_params(labelsize=fontsiz/1.1)
    ax.set_aspect(aspect=1)
    ax.contour(x, y, z, [th], linewidths=2, colors='green')
    ax.set_title('KDE levels', fontsize=titsiz, y=1.02)
    ax.set_xticks(xedges, minor=False)
    ax.set_yticks(yedges, minor=False)
    ax.minorticks_on()
    ax.grid(which='major', linestyle='-', linewidth='0.3', color='black')
    ax.grid(which='minor', linestyle=':', linewidth='0.1', color='black')
    ax.set_xlim(min(xedges), max(xedges))
    ax.set_ylim(min(yedges), max(yedges))
    ax.set_xlabel(units, fontsize=fontsiz)
    ax.set_ylabel(units, fontsize=fontsiz)
    ax.set_xticklabels(xedges, rotation=90, fontsize=fontsiz)
    ax.set_yticklabels(yedges, fontsize=fontsiz)
    set_ticklabels_spacing(ax, 10)
    plt.tight_layout()

    return(fig)


def pairwise_diffs(data):
    # get all pairwise diferences of points in data
    npts = len(data)
    diff = np.zeros(shape=(npts * (npts - 1) // 2, 2), dtype=np.double)
    k = 0
    for i in range(npts - 1):
        size = npts - i - 1
        diff[k:k + size] = abs(data[i] - data[i+1:])
        k += size

    return diff


def pairwise_diffs_biv(data_a, data_b):
    # get all diferences of points in data_a to points in data_b
    # assuming data_a and data_b are different sets
    npts_a = len(data_a)
    npts_b = len(data_b)
    diff = np.zeros(shape=(npts_a * npts_b, 2), dtype=np.double)
    k = 0
    for i in range(npts_a):
        diff[k:k + npts_b] = abs(data_a[i] - data_b[:])
        k += npts_b

    return diff


def ripleysKest(diff, w, h, radii, mode='none'):

    # Estimators for Ripley’s K function for two-dimensional spatial data
    #
    # Adapted from Astropy 4.0 implementation:
    # - Documentation:
    # https://docs.astropy.org/en/stable/api/astropy.stats.RipleysKEstimator.html
    # - Source code:
    # https://docs.astropy.org/en/stable/_modules/astropy/stats/spatial.html#RipleysKEstimator
    #
    # mode == 'none': this method does not take into account any edge effects
    #          whatsoever
    # mode == 'translation': computes the intersection of rectangular areas
    #          centered at the given points provided the upper bounds of the
    #          dimensions of the rectangular area of study. It assumes that
    #          all the points lie in a bounded rectangular region satisfying
    #          x_min < x_i < x_max; y_min < y_i < y_max.
    #          A detailed description of this method can be found on ref [4]
    #
    # References
    # ----------
    # [1] Peebles, P.J.E. *The large scale structure of the universe*.
    #   <http://adsabs.harvard.edu/cgi-bin/nph-bib_query?bibcode=1980lssu.book.....P&db_key=AST>
    # [2] Spatial descriptive statistics.
    #   <https://en.wikipedia.org/wiki/Spatial_descriptive_statistics>
    # [3] Package spatstat.
    #   <https://cran.r-project.org/web/packages/spatstat/spatstat.pdf>
    # [4] Cressie, N.A.C. (1991). Statistics for Spatial Data,
    #   Wiley, New York.
    # [5] Stoyan, D., Stoyan, H. (1992). Fractals, Random Shapes and
    #   Point Fields, Akademie Verlag GmbH, Chichester.

    if (len(diff) > 0):
        ripley = np.zeros(len(radii))
        if mode == 'none':
            for r in range(len(radii)):
                Ir = (np.hypot(diff[:, 0], diff[:, 1]) < radii[r])
                ripley[r] = Ir.sum()
            ripley = (w*h) * (1/len(diff)) * ripley
        elif mode == 'translation':
            intersec_area = ((w - diff[:, 0]) * (h - diff[:, 1]))
            for r in range(len(radii)):
                Ir = (np.hypot(diff[:, 0], diff[:, 1]) < radii[r])
                ripley[r] = ((1 / intersec_area) * Ir).sum()
            ripley = ((w*h)**2) * (1/len(diff)) * ripley
    else:
        ripley = np.nan

    # return the K function
    return ripley


def Kest_Kfunction(data, w, h, radii, mode='none'):

    data = np.asarray(data)
    if not data.shape[1] == 2:
        raise ValueError('data must be an n by 2 array, where n is the '
                         'number of observed points.')

    diff = pairwise_diffs(data)
    ripley = ripleysKest(diff, w, h, radii, mode)

    # return the H function
    return ripley


def Kest_Hfunction(data, w, h, radii, mode='none'):

    data = np.asarray(data)
    if not data.shape[1] == 2:
        raise ValueError('data must be an n by 2 array, where n is the '
                         'number of observed points.')

    diff = pairwise_diffs(data)
    ripley = ripleysKest(diff, w, h, radii, mode)

    # return the H function
    return (np.sqrt(ripley/np.pi) - radii)


def Kest_Hfunction_biv(data_a, data_b, w, h, radii, mode='none'):

    data_a = np.asarray(data_a)
    data_b = np.asarray(data_b)

    # Bivariate version of `Kest_Hfunction()`
    if not (data_a.shape[1] == 2 and data_b.shape[1] == 2):
        raise ValueError('data must be an n by 2 array, where n is the '
                         'number of observed points.')
    # check is two arrays are the same
    if np.array_equal(data_a, data_b):
        diff = pairwise_diffs(data_a)
    else:
        diff = pairwise_diffs_biv(data_a, data_b)
    ripley = ripleysKest(diff, w, h, radii, mode)

    # return the H function
    return (np.sqrt(ripley/np.pi) - radii)


def Kest_Hfunction_biv_sample_density(data_a, data_b, density, radii):

    # calculates the Bivariate `Kest_Hfunction()` using a global cell
    # density of test cells as the completely spatially random point pattern
    # instead of using the data to estimate it.
    # Useful for cases where a local measure of H is required
    # This corresponds to some form of bivarite hot spot analysis
    # NOTE: no edge effects are taken into account

    data_a = np.asarray(data_a)
    data_b = np.asarray(data_b)

    # Bivariate version of `Kest_Hfunction()`
    if not (data_a.shape[1] == 2 and data_b.shape[1] == 2):
        raise ValueError('data must be an n by 2 array, where n is the '
                         'number of observed points.')
    # check is two arrays are the same
    if np.array_equal(data_a, data_b):
        diff = pairwise_diffs(data_a)
    else:
        diff = pairwise_diffs_biv(data_a, data_b)

    if (len(diff) > 0):
        ripley = np.zeros(len(radii))
        for r in range(len(radii)):
            Ir = (np.hypot(diff[:, 0], diff[:, 1]) < radii[r])
            ripley[r] = Ir.sum()/density
    else:
        ripley = np.nan

    # return the H function
    return (np.sqrt(ripley/np.pi) - radii)


def pairwise_diffs_pairs(data):
    # returns 0 if two data points are equal, 1 is different
    # data is a set of labels s
    npts = len(data)
    diff = np.zeros(shape=(npts * (npts - 1) // 2, ), dtype=np.double)
    k = 0
    for i in range(npts - 1):
        size = npts - i - 1
        diff[k:k + size] = 1*(data[i] != data[i+1:])
        k += size

    return diff


def Kest_Hfunction_pairs(data_a, data_b, w, h, radii, mode='none'):

    # Bivariate version of `Kest_Hfunction()`
    data = np.append(data_a, data_b, axis=0)
    if not data.shape[1] == 2:
        raise ValueError('data must be an n by 2 array, where n is the '
                         'number of observed points.')

    codes = np.append(np.zeros(len(data_a)), np.ones(len(data_b)))
    area = (w)*(h)
    ripley = np.zeros(len(radii))
    diff = pairwise_diffs(data)
    delt = pairwise_diffs_pairs(codes)
    nprs = delt.sum()

    if mode == 'none':
        for r in range(len(radii)):
            Ir = (np.hypot(diff[:, 0], diff[:, 1]) < radii[r]) * delt
            ripley[r] = Ir.sum()

        ripley = area * (1/nprs) * ripley

    elif mode == 'translation':
        intersec_area = ((w - diff[:, 0]) * (h - diff[:, 1]))
        for r in range(len(radii)):
            Ir = (np.hypot(diff[:, 0], diff[:, 1]) < radii[r]) * delt
            ripley[r] = ((1 / intersec_area) * Ir).sum()

        ripley = (area**2) * (1/nprs) * ripley

    # return the H function
    return (np.sqrt(ripley/np.pi) - radii)


def quadrats_distances(pop, classes, metric, max_d, ttl):

    # Use a distance metric to compare population contents in the
    # different quadrants
    D = np.zeros((pop.shape[0], pop.shape[0]))

    for i, row in pop.iterrows():
        update_progress(i / len(pop), '...quadrat distances...')
        xi = row[[typ for typ in classes['class_code']]].values
        for j, col in pop.iterrows():
            if (i < j):
                yi = col[[typ for typ in classes['class_code']]].values
                dij = metric(xi, yi)
                D[i, j] = dij
                D[j, i] = dij
    update_progress(1, '...dDone!!!\n')

    plt.figure(figsize=(10, 10), facecolor='w', edgecolor='k')
    N = D[np.triu_indices(D.shape[0], k=1)].size
    hist, bins = np.histogram(D[np.triu_indices(D.shape[0], k=1)],
                              bins=np.arange(0, 1, 0.05), density=False)
    x = (bins[1:] + bins[:-1])/2
    plt.plot(x, hist/N)
    plt.axhline(y=0.05, linewidth=1, color='r', linestyle='--')
    plt.axvline(x=max_d, linewidth=1, color='r', linestyle='--')
    plt.title(ttl)
    plt.xlabel('Distance')
    plt.ylabel('Frequency')

    return(D)


def quadrat_MH_SH_distances(pop, classes):

    # Use a distance metric to compare population distribution in the
    # different quadrants for all combinations of quadrants
    D_MH = np.zeros((pop.shape[0], pop.shape[0]))
    D_SH = np.zeros((pop.shape[0], pop.shape[0]))

    # Get the pair-wise distance between all quadrats
    for i, row in pop.iterrows():
        update_progress(i / len(pop), '...population stats...')
        xi = row[[typ for typ in classes['class_code']]].values
        for j, col in pop.iterrows():
            yj = col[[typ for typ in classes['class_code']]].values

            if (j > i):
                # simple distances based on population frequencies
                mh = morisita_horn_not_normal(xi, yj)
                sh = heterogeneity(xi, yj)
                D_MH[i, j] = mh
                D_MH[j, i] = mh
                D_SH[i, j] = sh
                D_SH[j, i] = sh

    update_progress(1, '...done!...')

    return([D_MH, D_SH])


def quadrats_tree(D, metric, max_d, ttl):

    Z = linkage(D[np.triu_indices(D.shape[0], k=1)], 'complete', metric=metric)
    clusters = fcluster(Z, max_d, criterion='distance')

    plt.figure(figsize=(10, 10), facecolor='w', edgecolor='k')
    dn = fancy_dendrogram(Z, max_d=max_d, title=ttl,
                          above_threshold_color='grey',
                          truncate_mode='lastp', p=20,
                          show_leaf_counts=True,
                          show_contracted=True)

    return(Z, clusters, dn)


def quadrat_population(classes, data, mask, binsiz):

    # reduce cell data
    cell_data = data[['class_code', 'row', 'col']].copy()

    # define subquadrats
    redges = np.arange(0, mask.shape[0], binsiz)
    cedges = np.arange(0, mask.shape[1], binsiz)

    # quadrat mask
    msk = np.zeros((len(redges), len(cedges)))

    # dataframe for population fractions vectors
    pop = pd.DataFrame({'row': [], 'col': []})
    for i, rowf in classes.iterrows():
        pop[rowf['class_code']] = []

    # get population vectors for each quadrat (square bin)
    k = 0
    for r, rbin in enumerate(redges):
        update_progress(k / len(redges), '...quadrat counts...')
        k = k + 1
        for c, cbin in enumerate(cedges):
            # drop quadrats outside the masked region
            # (quadrat must be at least half covered)
            m = mask[rbin:(rbin + binsiz), cbin:(cbin + binsiz)]
            if np.sum(m) > np.prod(m.shape)/2:
                msk[r, c] = 1
                # subset in this quadrat
                s_c_data = cell_data.loc[(cell_data['row'] >= rbin) &
                                         (cell_data['row'] < (rbin + binsiz)) &
                                         (cell_data['col'] >= cbin) &
                                         (cell_data['col'] < (cbin + binsiz))]

                # count cells per class in quadrat, and total
                ncells = len(s_c_data)
                # if there is enough cells in quadrat add to population stats
                if (ncells > 20):
                    # get population fractions in each quadrant
                    aux = pd.DataFrame({'row': [rbin], 'col': [cbin]})
                    for i, code in enumerate(classes['class_code']):
                        C = len(s_c_data.loc[s_c_data['class_code'] == code])
                        aux[code] = [C/ncells]
                    pop = pop.append(aux, ignore_index=True)

    update_progress(1, '...done!\n')

    pop = pop.astype({'r': int, 'c': int})

    return(pop)


def quadrat_analysis_scarce(classes, data, mask, binsiz, subbinsiz):

    # reduce cell data
    cell_data = data[['class_code', 'row', 'col']].copy()

    # define subquadrats
    redges = np.arange(0, mask.shape[0], binsiz)
    cedges = np.arange(0, mask.shape[1], binsiz)

    # populations per quadrant
    C = np.zeros((classes['class_code'].size, len(redges), len(cedges)))
    C[:] = np.nan

    # quadrat mask
    msk = np.zeros((len(redges), len(cedges)))

    # dataframe for population fractions vectors
    pop = pd.DataFrame({'r': [], 'c': [],
                        'row_lo': [], 'row_hi': [],
                        'col_lo': [], 'col_hi': [],
                        'total': []})
    # dataframe for colocalization indices
    coloc = pd.DataFrame({'r': [], 'c': [],
                          'row_lo': [], 'row_hi': [],
                          'col_lo': [], 'col_hi': []})
    # dataframe for nearest neighbor distance indices
    nndist = pd.DataFrame({'r': [], 'c': [],
                           'row_lo': [], 'row_hi': [],
                           'col_lo': [], 'col_hi': []})
    # dataframe for Ripley's H indices
    rhfunc = pd.DataFrame({'r': [], 'c': [],
                           'row_lo': [], 'row_hi': [],
                           'col_lo': [], 'col_hi': []})
    # dataframe for modified Ripley's H indices
    # modrhfunc = pd.DataFrame({'r': [], 'c': [],
    #                          'row_lo': [], 'row_hi': [],
    #                          'col_lo': [], 'col_hi': []})

    for i, rowf in classes.iterrows():
        pop[rowf['class_code']] = []
        for j, rowt in classes.iterrows():
            if (j > i):
                coloc[rowf['class_code'] + ':' + rowt['class_code']] = []
            if (i != j):
                nndist[rowf['class_code'] + ':' + rowt['class_code']] = []
            rhfunc[rowf['class_code'] + ':' + rowt['class_code']] = []
            # modrhfunc[rowf['class_code'] + ':' + rowt['class_code']] = []

    # get population vectors for each quadrat (square bin)
    k = 0
    for r, rbin in enumerate(redges):

        update_progress(k / len(redges), '...quadrat stats...')
        k = k + 1

        for c, cbin in enumerate(cedges):

            # drop quadrats outside the masked region
            # (quadrat must be at least half covered)
            m = mask[rbin:(rbin + binsiz), cbin:(cbin + binsiz)]
            if np.sum(m) > np.prod(m.shape)/2:
                msk[r, c] = 1
                # subset data in quadrat
                s_c_data = cell_data.loc[(cell_data['row'] >= rbin) &
                                         (cell_data['row'] < (rbin + binsiz)) &
                                         (cell_data['col'] >= cbin) &
                                         (cell_data['col'] < (cbin + binsiz))]

                # subset data in (extended) quadrat
                s_c_e_data = cell_data.loc[(cell_data['row'] >= (rbin -
                                                                 subbinsiz)) &
                                           (cell_data['row'] < (rbin +
                                                                binsiz +
                                                                subbinsiz)) &
                                           (cell_data['col'] >= (cbin -
                                                                 subbinsiz)) &
                                           (cell_data['col'] < (cbin +
                                                                binsiz +
                                                                subbinsiz))]

                # count cells per class in quadrat, and total
                ncells = len(s_c_data)
                for i, code in enumerate(classes['class_code']):
                    C[i, r,
                      c] = len(s_c_data.loc[s_c_data['class_code'] == code])

                # if there is enough cells in quadrat add to population stats
                if (ncells > 20):

                    # bin quadrat down to get spatial distributions
                    rowssubs = np.arange(rbin, rbin + binsiz, subbinsiz)
                    colssubs = np.arange(cbin, cbin + binsiz, subbinsiz)
                    subpops = np.zeros((classes['class_code'].size,
                                        len(rowssubs), len(colssubs)))

                    # get population vectors for each subquadrat
                    for r2, rbin2 in enumerate(rowssubs):
                        for c2, cbin2 in enumerate(colssubs):
                            # subset data in this subquadrat
                            ss_c_data = s_c_data.loc[(s_c_data['row'] >=
                                                      rbin2) &
                                                     (s_c_data['row'] <
                                                      (rbin2 + subbinsiz)) &
                                                     (s_c_data['col'] >=
                                                      cbin2) &
                                                     (s_c_data['col'] <
                                                      (cbin2 + subbinsiz))]

                            for i, c in enumerate(classes['class_code']):
                                # get number of cells in each class
                                # for each subsubarray
                                m = ss_c_data.loc[ss_c_data['class_code'] == c]
                                subpops[i, r2, c2] = len(m)

                    # get population fractions in each quadrant
                    aux = pd.DataFrame({'r': [r], 'c': [c],
                                        'row_lo': [rbin], 'row_hi': [(rbin +
                                                                      binsiz)],
                                        'col_lo': [cbin], 'col_hi': [(cbin +
                                                                      binsiz)],
                                        'total':  [ncells]})
                    for i, code in enumerate(classes['class_code']):
                        aux[code] = [C[i, r, c]/ncells]
                    pop = pop.append(aux, ignore_index=True)

                    # get colocalization index values in each quadrat
                    aux = pd.DataFrame({'r': [r],
                                        'c': [c],
                                        'row_lo': [rbin],
                                        'row_hi': [(rbin + binsiz)],
                                        'col_lo': [cbin],
                                        'col_hi': [(cbin + binsiz)]})

                    # get nearest neighbor distance indeces in each quadrat
                    auy = pd.DataFrame({'r': [r],
                                        'c': [c],
                                        'row_lo': [rbin],
                                        'row_hi': [(rbin + binsiz)],
                                        'col_lo': [cbin],
                                        'col_hi': [(cbin + binsiz)]})

                    # get Ripley's H indices in each quadrat
                    auz = pd.DataFrame({'r': [r],
                                        'c': [c],
                                        'row_lo': [rbin],
                                        'row_hi': [(rbin + binsiz)],
                                        'col_lo': [cbin],
                                        'col_hi': [(cbin + binsiz)]})

                    # get modified Ripley's H indices in each quadrat
                    # (uses global density as H_0: completely spatially
                    #  random point pattern)
                    # auw = pd.DataFrame({'r':[r],
                    #                    'c':[c],
                    #                    'row_lo': [rbin],
                    #                    'row_hi': [(rbin + binsiz)],
                    #                    'col_lo': [cbin],
                    #                    'col_hi': [(cbin + binsiz)]})

                    for i, ref in enumerate(classes['class_code']):

                        # get array of coordinates of ref cells in the quadrat
                        fs = np.array(s_c_data.loc[s_c_data['class_code'] ==
                                                   ref][['row', 'col']])
                        # get array of ref cell count distribution in quadrat
                        fi = subpops[i, :, :].ravel()
                        ni = len(fs)

                        # if there are ref cells in the quadrat
                        if (ni > 0):
                            # generate KD-Tree for ref cells
                            kdtree = KDTree(fs)
                            # get nearest neighbor distances to other ref cells
                            dffs, iffs = kdtree.query(fs, k=2)
                            dff = np.mean(dffs[:, 1])

                            # get relationships with test cells
                            for j, test in enumerate(classes['class_code']):
                                lab = ref + ':' + test
                                # get coordinates of test cells in the quadrat
                                m = s_c_data.loc[s_c_data['class_code'] ==
                                                 test]
                                ts = np.array(m[['row', 'col']])
                                nj = len(ts)

                                if (i < j):
                                    aux[lab] = [float('nan')]
                                    # get array of test cell count
                                    # distribution in quadrat
                                    fj = subpops[j, :, :].ravel()
                                    if (nj > 0):
                                        aux[lab] = [morisita_horn(fi, fj)]

                                if (i != j):
                                    auy[lab] = [float('nan')]
                                    # if there are test cells in the quadrat
                                    if (ni > 1 and nj > 1):
                                        # get nearest neighbor distances to
                                        # test cells
                                        dfts, ifts = kdtree.query(ts, k=1)
                                        dft = np.nanmean(dfts)
                                        # - get v = log ratio of mean
                                        #   test_distance/ref_distance
                                        # - v > 0 indicates ref and
                                        #   test cells are segregated
                                        # - v ~ 0 indicates ref and test
                                        #   cells are well mixed
                                        # - v < 0 indicates ref cells are
                                        #   individually infiltrated
                                        auy[lab] = [np.log(dft/dff)]

                                auz[lab] = [float('nan')]
                                # auw[lab] = [float('nan')]
                                # dens = classes.loc[classes['class_code']==
                                #        test].num_cells/np.sum(mask)
                                m = s_c_e_data.loc[s_c_e_data['class_code'] ==
                                                   test]
                                ts = np.array(m[['row', 'col']])
                                if (len(ts) > 0):
                                    # Minimizes the edges effects by expanding
                                    # region of tests cells to
                                    #  [-subbinsiz, binsiz + subbinsiz]
                                    # (over to neighboring quadrat)
                                    # and using mode='none'...
                                    # Also, a local version of the H function
                                    # is used
                                    auz[lab] = [Kest_Hfunction_biv(fs, ts,
                                                                   binsiz +
                                                                   2*subbinsiz,
                                                                   binsiz +
                                                                   2*subbinsiz,
                                                                   [subbinsiz],
                                                                   'none')[0]]
                                    # auw[lab] =
                                    # [Kest_Hfunction_biv_sample_density(fs,
                                    #             ts, dens, [subbinsiz])]

                    coloc = coloc.append(aux, ignore_index=True)
                    nndist = nndist.append(auy, ignore_index=True)
                    rhfunc = rhfunc.append(auz, ignore_index=True)
                    # modrhfunc = modrhfunc.append(auw, ignore_index = True)

    # Do Getis-Ord hot-spot analysis
    Z = np.zeros((classes['class_code'].size, len(redges), len(cedges)))
    P = np.zeros((classes['class_code'].size, len(redges), len(cedges)))
    HOT = np.zeros((classes['class_code'].size, len(redges), len(cedges)))
    for i, row in classes.iterrows():
        (z, p) = getis_ord_mask(C[i, :, :], msk)
        Z[i, :, :] = z.copy()
        P[i, :, :] = p.copy()
        p[np.isnan(p)] = 1
        aux = np.sign(z)
        aux[p > 0.05] = 0
        HOT[i, :, :] = aux[:, :]
        HOT[i, msk == 0] = np.nan

    pop = pop.astype({'r': int, 'c': int})
    coloc = coloc.astype({'r': int, 'c': int})
    nndist = nndist.astype({'r': int, 'c': int})
    rhfunc = rhfunc.astype({'r': int, 'c': int})

    update_progress(1, '...done!\n')

    return([C, Z, P, HOT, pop, coloc, nndist, rhfunc, msk])


def quadrat_analysis_infiltrate(classes, data, mask, binsiz, subbinsiz):

    # reduce cell data
    cell_data = data[['class_code', 'row', 'col']].copy()

    # define subquadrats
    redges = np.arange(0, mask.shape[0], binsiz)
    cedges = np.arange(0, mask.shape[1], binsiz)

    # populations per quadrant
    C = np.zeros((classes['class_code'].size, len(redges), len(cedges)))
    C[:] = np.nan

    # quadrat mask
    msk = np.zeros((len(redges), len(cedges)))

    # dataframe for population fractions vectors
    pop = pd.DataFrame({'r': [], 'c': [],
                        'row_lo': [], 'row_hi': [],
                        'col_lo': [], 'col_hi': [],
                        'total': []})
    # dataframe for colocalization indices
    coloc = pd.DataFrame({'r': [], 'c': [],
                          'row_lo': [], 'row_hi': [],
                          'col_lo': [], 'col_hi': []})
    # dataframe for nearest neighbor distance indices
    nndist = pd.DataFrame({'r': [], 'c': [],
                           'row_lo': [], 'row_hi': [],
                           'col_lo': [], 'col_hi': []})
    # dataframe for ripley's H indices
    rhfunc = pd.DataFrame({'r': [], 'c': [],
                           'row_lo': [], 'row_hi': [],
                           'col_lo': [], 'col_hi': []})

    for i, rowf in classes.iterrows():
        pop[rowf['class_code']] = []
        for j, rowt in classes.iterrows():
            if (i < j):
                coloc[rowf['class_code'] + ':' + rowt['class_code']] = []
            if (i != j):
                nndist[rowf['class_code'] + ':' + rowt['class_code']] = []
            rhfunc[rowf['class_code'] + ':' + rowt['class_code']] = []

    # get population vectors for each quadrat (square bin)
    k = 0
    for r, rbin in enumerate(redges):

        update_progress(k / len(redges), '...quadrat stats...')
        k = k + 1

        for c, cbin in enumerate(cedges):

            # drop quadrats outside the masked region
            # (quadrat must be at least half covered)
            m = mask[rbin:(rbin + binsiz), cbin:(cbin + binsiz)]
            if np.sum(m) > np.prod(m.shape)/2:
                msk[r, c] = 1
                # subset in this quadrat
                s_c_data = cell_data.loc[(cell_data['row'] >= rbin) &
                                         (cell_data['row'] < (rbin + binsiz)) &
                                         (cell_data['col'] >= cbin) &
                                         (cell_data['col'] < (cbin + binsiz))]

                # count cells per class in quadrat, and total
                ncells = len(s_c_data)
                for i, code in enumerate(classes['class_code']):
                    C[i,
                      r, c] = len(s_c_data.loc[s_c_data['class_code'] == code])

                # if there is enough cells in quadrat add to population stats
                if (ncells > 20):

                    # bin quadrat into smaller boxes for spatial distributions
                    rowssubs = np.arange(rbin, rbin + binsiz, subbinsiz)
                    colssubs = np.arange(cbin, cbin + binsiz, subbinsiz)
                    subpops = np.zeros((classes['class_code'].size,
                                        len(rowssubs),
                                        len(colssubs)))

                    # get population vectors for each subquadrat
                    for r2, rbin2 in enumerate(rowssubs):
                        for c2, cbin2 in enumerate(colssubs):
                            # subset in this subquadrat
                            ss_c_data = s_c_data.loc[(s_c_data['row'] >=
                                                      rbin2) &
                                                     (s_c_data['row'] <
                                                      (rbin2 + subbinsiz)) &
                                                     (s_c_data['col'] >=
                                                      cbin2) &
                                                     (s_c_data['col'] <
                                                      (cbin2 + subbinsiz))]

                            for i, code in enumerate(classes['class_code']):
                                # get number of cells in each class for
                                # each subsubarray
                                m = ss_c_data.loc[ss_c_data['class_code'] ==
                                                  code]
                                subpops[i, r2, c2] = len(m)

                    # get population fractions in each quadrant
                    aux = pd.DataFrame({'r': [r],
                                        'c': [c],
                                        'row_lo': [rbin],
                                        'row_hi': [(rbin + binsiz)],
                                        'col_lo': [cbin],
                                        'col_hi': [(cbin + binsiz)],
                                        'total':  [ncells]})
                    for i, code in enumerate(classes['class_code']):
                        aux[code] = [C[i, r, c]/ncells]
                    pop = pop.append(aux, ignore_index=True)

                    # get colocalization index values in each quadrat
                    aux = pd.DataFrame({'r': [r],
                                        'c': [c],
                                        'row_lo': [rbin],
                                        'row_hi': [(rbin + binsiz)],
                                        'col_lo': [cbin],
                                        'col_hi': [(cbin + binsiz)]})

                    # get nearest neighbor distance indeces in each quadrat
                    auy = pd.DataFrame({'r': [r],
                                        'c': [c],
                                        'row_lo': [rbin],
                                        'row_hi': [(rbin + binsiz)],
                                        'col_lo': [cbin],
                                        'col_hi': [(cbin + binsiz)]})

                    # get ripley's H indices in each quadrat
                    auz = pd.DataFrame({'r': [r],
                                        'c': [c],
                                        'row_lo': [rbin],
                                        'row_hi': [(rbin + binsiz)],
                                        'col_lo': [cbin],
                                        'col_hi': [(cbin + binsiz)]})

                    for i, ref in enumerate(classes['class_code']):

                        # get array of coordinates of ref cells in the quadrat
                        m = s_c_data.loc[s_c_data['class_code'] == ref]
                        fs = np.array(m[['row', 'col']])
                        # get array of ref cell count distribution in quadrat
                        fi = subpops[i, :, :].ravel()
                        ni = C[i, r, c]

                        # if there are ref cells in the quadrat
                        if (ni > 0):
                            # generate KD-Tree for ref cells
                            kdtree = KDTree(fs)
                            # get nearest neighbor distances to other ref cells
                            dffs, iffs = kdtree.query(fs, k=2)
                            dff = np.mean(dffs[:, 1])

                            # get relationships with test cells
                            for j, test in enumerate(classes['class_code']):
                                lab = ref + ':' + test
                                # get coordinates of test cells in the quadrat
                                m = s_c_data.loc[s_c_data['class_code'] ==
                                                 test]
                                ts = np.array(m[['row', 'col']])
                                nj = len(ts)

                                if (i < j):
                                    aux[lab] = [float('nan')]
                                    # get array of test cell count
                                    # distribution in quadrat
                                    fj = subpops[j, :, :].ravel()
                                    nj = np.sum(fj)
                                    if (nj > 0):
                                        aux[lab] = [morisita_horn(fi, fj)]

                                if (i != j):
                                    auy[lab] = [float('nan')]
                                    # if there are test cells in the quadrat
                                    if (ni > 1 and nj > 1):
                                        # get nearest neighbor distances
                                        # to test cells
                                        dfts, ifts = kdtree.query(ts, k=1)
                                        dft = np.nanmean(dfts)
                                        # get log ratio of mean
                                        # test_distance/ref_distance in quadrat
                                        # * an index > 0 indicates ref and test
                                        #   cells are segregated
                                        # * an index ~ 0 indicates ref and test
                                        #   cells are well mixed
                                        # * an index < 0 indicates test cells
                                        #   are individually infiltrated
                                        auy[lab] = [np.log(dft/dff)]

                                auz[lab] = [float('nan')]
                                if (nj > 0):
                                    # NOTE: This can be improved by expanding
                                    #       region of tests cells to
                                    #       [-subbinsiz, binsiz + subbinsiz]
                                    #       (over to neighboring quadrat)
                                    #       and using mode='none'...
                                    #       this way, the value of H is less
                                    #       impacted by binning edge effects
                                    #       (and more precise)
                                    m = Kest_Hfunction_biv(fs, ts,
                                                           binsiz, binsiz,
                                                           [subbinsiz],
                                                           'translation')
                                    auz[lab] = [m]

                    coloc = coloc.append(aux, ignore_index=True)
                    nndist = nndist.append(auy, ignore_index=True)
                    rhfunc = rhfunc.append(auz, ignore_index=True)

    update_progress(1, '...done!\n')

    # Do Getis-Ord hot-spot analysis
    Z = np.zeros((classes['class_code'].size, len(redges), len(cedges)))
    P = np.zeros((classes['class_code'].size, len(redges), len(cedges)))
    HOT = np.zeros((classes['class_code'].size, len(redges), len(cedges)))

    for i, row in classes.iterrows():
        (z, p) = getis_ord_mask(C[i, :, :], msk)
        Z[i, :, :] = z.copy()
        P[i, :, :] = p.copy()
        p[np.isnan(p)] = 1
        aux = np.sign(z)
        aux[p > 0.05] = 0
        HOT[i, :, :] = aux[:, :]
        HOT[i, msk == 0] = np.nan

    pop = pop.astype({'r': int, 'c': int})
    coloc = coloc.astype({'r': int, 'c': int})
    nndist = nndist.astype({'r': int, 'c': int})
    rhfunc = rhfunc.astype({'r': int, 'c': int})

    return([C, Z, P, HOT, pop, coloc, nndist, rhfunc, msk])


def comp_correlations(dataframe, comps, classes, ttl):

    nc = len(comps)*(len(comps)-1)/2

    mins = np.zeros(len(comps))
    maxs = np.zeros(len(comps))
    for i, ci in enumerate(comps):
        aux = dataframe[ci[0] + ':' + ci[1]].dropna().tolist()
        mins[i] = np.min(aux)
        maxs[i] = np.max(aux)
    lims = [0.8*np.min(mins), 1.2*np.max(maxs)]

    ncols = int(np.ceil(np.sqrt(nc)))

    fig, ax = plt.subplots(int(np.ceil(nc/ncols)), ncols,
                           figsize=(ncols*6, (nc/ncols)*6),
                           facecolor='w', edgecolor='k')
    shp = ax.shape

    k = 0
    for i, ci in enumerate(comps):
        for j, cj in enumerate(comps):
            if j > i:
                si = ci[0] + ':' + ci[1]
                sj = cj[0] + ':' + cj[1]

                ti1 = classes.loc[classes['class_code'] == ci[0]].class_name
                ti2 = classes.loc[classes['class_code'] == ci[1]].class_name
                tj1 = classes.loc[classes['class_code'] == cj[0]].class_name
                tj2 = classes.loc[classes['class_code'] == cj[1]].class_name

                aux = dataframe[[si, sj]].dropna()

                sns.axes_style("whitegrid")
                sns.regplot(x=si, y=sj, data=aux,
                            ax=ax[np.unravel_index(k, shp)],
                            scatter_kws={"color": "black"},
                            line_kws={"color": "red"})
                ax[np.unravel_index(k, shp)].set_xlabel(ttl + '(' +
                                                        ti1.values[0] + ':' +
                                                        ti2.values[0] + ')')
                ax[np.unravel_index(k, shp)].set_ylabel(ttl + '(' +
                                                        tj1.values[0] + ':' +
                                                        tj2.values[0] + ')')
                ax[np.unravel_index(k, shp)].plot(np.arange(lims[0],
                                                            lims[1]+0.1, 0.1),
                                                  np.arange(lims[0],
                                                            lims[1]+0.1, 0.1),
                                                  color='k',
                                                  linestyle='dashed')
                ax[np.unravel_index(k, shp)].set_xlim(lims[0], lims[1])
                ax[np.unravel_index(k, shp)].set_ylim(lims[0], lims[1])

                correlation, p_value = sts.pearsonr(aux[si], aux[sj])
                star = 'NS'
                if (p_value < 0.05):
                    star = '*'
                if (p_value < 0.01):
                    star = '**'
                if (p_value < 0.001):
                    star = '***'
                coefs = 'C = %.4f; p-value = %.2e' % (correlation, p_value)
                ax[np.unravel_index(k, shp)].set_title(coefs +
                                                       ' (' + star + ')')

                k = k+1

    plt.tight_layout()

    return([fig, ax])


def factor_correlations(df1, comps1, tti, df2, comps2, ttj, classes):

    nc = len(comps1)*len(comps2)

    mins = np.zeros(len(comps1))
    maxs = np.zeros(len(comps1))
    for i, ci in enumerate(comps1):
        aux = df1[ci[0] + ':' + ci[1]].dropna().tolist()
        mins[i] = np.min(aux)
        maxs[i] = np.max(aux)
    limsi = [0.8*np.min(mins), 1.2*np.max(maxs)]

    mins = np.zeros(len(comps2))
    maxs = np.zeros(len(comps2))
    for i, ci in enumerate(comps2):
        aux = df2[ci[0] + ':' + ci[1]].dropna().tolist()
        mins[i] = np.min(aux)
        maxs[i] = np.max(aux)
    limsj = [0.8*np.min(mins), 1.2*np.max(maxs)]

    ncols = int(np.ceil(np.sqrt(nc)))

    fig, ax = plt.subplots(int(np.ceil(nc/ncols)), ncols,
                           figsize=(ncols*(6), (nc/ncols)*(6)),
                           facecolor='w', edgecolor='k')
    shp = ax.shape

    k = 0
    for i, ci in enumerate(comps1):
        for j, cj in enumerate(comps2):

            ti1 = classes.loc[classes['class_code'] == ci[0]].class_name
            ti2 = classes.loc[classes['class_code'] == ci[1]].class_name
            tj1 = classes.loc[classes['class_code'] == cj[0]].class_name
            tj2 = classes.loc[classes['class_code'] == cj[1]].class_name

            aux = pd.DataFrame({'X': df1[ci[0] + ':' +
                                         ci[1]].tolist(),
                                'Y': df2[cj[0] + ':' +
                                         cj[1]].tolist()}).dropna()

            sns.axes_style("whitegrid")
            sns.regplot(x='X', y='Y', data=aux,
                        ax=ax[np.unravel_index(k, shp)],
                        scatter_kws={"color": "black"},
                        line_kws={"color": "red"})
            ax[np.unravel_index(k, shp)].set_xlabel(tti + '(' +
                                                    ti1.values[0] + ':' +
                                                    ti2.values[0] + ')')
            ax[np.unravel_index(k, shp)].set_ylabel(ttj + '(' +
                                                    tj1.values[0] + ':' +
                                                    tj2.values[0] + ')')
            ax[np.unravel_index(k, shp)].plot([limsi[0], limsi[1]],
                                              [limsj[0], limsj[1]],
                                              color='k', linestyle='dashed')
            ax[np.unravel_index(k, shp)].set_xlim(limsi[0], limsi[1])
            ax[np.unravel_index(k, shp)].set_ylim(limsj[0], limsj[1])

            correlation, p_value = sts.pearsonr(aux.X, aux.Y)
            star = 'NS'
            if (p_value < 0.05):
                star = '*'
            if (p_value < 0.01):
                star = '**'
            if (p_value < 0.001):
                star = '***'
            coefs = 'C = %.4f; p-value = %.2e' % (correlation, p_value)
            ax[np.unravel_index(k, shp)].set_title(coefs + ' (' + star + ')')

            k = k+1

    plt.tight_layout()

    return([fig, ax])


def getis_ord(C):
    # get the Getis-Ord hotspot Z score for an array of counts

    # get neighbor weights for the masked array terms
    w = w_neighbors(C, 1)

    # do stats
    c = C.ravel()
    n = len(c)
    s = np.std(c)
    u = np.sqrt((n*np.sum(w**2, axis=0) - (np.sum(w, axis=0))**2)/(n-1))
    z = (np.dot(w, c) - np.mean(c)*np.sum(w, axis=0))/(s*u)
    p = 2*sts.norm.sf(abs(z))
    Z = z.reshape(C.shape)
    P = p.reshape(C.shape)

    return (Z, P)


def getis_ord_mask(C, msk):
    # Getis-Ord hotspot Z score for a MASKED array of counts

    # get (flat) indices of terms in the mask
    i = np.arange(np.prod(C.shape)).reshape(C.shape)
    k = i[msk > 0].ravel()

    # get neighbor weights for the masked array terms
    W = w_neighbors(C, 1)
    w = W[:, k][k, :]

    # filter terms in the mask
    c = C[msk > 0].ravel()

    # do stats
    n = len(c)
    s = np.std(c)
    u = np.sqrt((n*np.sum(w**2, axis=0) - (np.sum(w, axis=0))**2)/(n-1))
    z = (np.dot(w, c) - np.mean(c)*np.sum(w, axis=0))/(s*u)
    p = 2*sts.norm.sf(abs(z))

    # revert to full array forms (padded with np.nan)
    aux = np.empty(C.ravel().shape)
    aux[:] = np.nan
    aux[k] = z
    Z = aux.reshape(C.shape)

    aux = np.empty(C.ravel().shape)
    aux[:] = np.nan
    aux[k] = p
    P = aux.reshape(C.shape)

    return (Z, P)


def w_neighbors(m, n):

    # if n==1 this is whole function is the same the same as
    # using the pysal weights function 'lat2W'
    #
    # from libpysal.weights import lat2W
    # w = lat2W(4,3, rook=False).sparse.toarray()

    # produce neighbor matrix for an array of squared regions of size (a, b)
    (a, b) = m.shape
    # w = np.identity(a*b)
    w = np.zeros([a*b, a*b])
    # gets neighbor regions for each one
    for i in np.arange(0, a*b):
        for j in np.arange(i+1, a*b):
            (ci, ri) = np.unravel_index(i, (a, b))
            (cj, rj) = np.unravel_index(j, (a, b))
            if (abs(ci - cj) <= n) & (abs(ri - rj) <= n):
                w[i, j] = 1
                w[j, i] = 1
    return w


def morisita_horn_not_normal(x, y):
    # (not normalized) Morisita-Horn metric
    return (1 - 2.0 * np.dot(x, y) / (np.dot(x, x) + np.dot(y, y)))


def morisita_horn(x, y):
    # Morisita-Horn metric
    p1 = x/np.sum(x)
    p2 = y/np.sum(y)
    return (2.0 * np.dot(p1, p2) / (np.dot(p1, p1) + np.dot(p2, p2)))


def shuffle_array(A):
    a = A.ravel()
    np.random.shuffle(a)
    return a.reshape(A.shape)


def jaccard(p1, p2):
    # distance based on the Jaccard similarity index
    # (calculated using the element-wise min and max of population vectors
    return 1 - np.sum(np.minimum(p1, p2)) / np.sum(np.maximun(p1, p2))


def shannon(p):
    # Shannon entropy
    h = np.zeros(p.shape)
    p = p.astype(float)
    h[p != 0] = -(p[p != 0])*np.log(p[p != 0])
    return (np.sum(h))


def heterogeneity(x, y):
    # fraction of the posible range of entropy between two sets
    # = 0 if sets have the same distribution
    # = 1 if sets have distinct distributions (no overlap)
    def h(x, y):
        return shannon((x + y) / np.sum(x + y))

    def hmax(x, y):
        px = x / (np.sum(x) + np.sum(y))
        py = y / (np.sum(x) + np.sum(y))
        return shannon(px) + shannon(py)

    def hmin(x, y):
        X = np.sum(x)
        Y = np.sum(y)
        return (X*shannon(x/X) + Y*shannon(y/Y))/(X + Y)

    return ((h(x, y) - hmin(x, y)) / (hmax(x, y) - hmin(x, y)))


def distance_maps(D_MH, D_SH, max_d_MH, max_d_SH):

    fig, axs = plt.subplots(3, 2, figsize=(14, 21),
                            facecolor='w', edgecolor='k')

    im = axs[0, 0].imshow(D_MH, vmin=0, vmax=1, cmap='jet')
    axs[0, 0].invert_yaxis()
    plt.colorbar(im, ax=axs[0, 0], fraction=0.046, pad=0.04)
    axs[0, 0].set_title('Morisita-Horn pairwise distance')

    N = D_MH[np.triu_indices(D_MH.shape[0], k=1)].size
    hist, bins = np.histogram(D_MH[np.triu_indices(D_MH.shape[0], k=1)],
                              bins=np.arange(0, 1, 0.05), density=False)
    x = (bins[1:] + bins[:-1])/2
    axs[0, 1].plot(x, hist/N)
    # axs[0, 1].axhline(y=0.05, linewidth=1, color='r', linestyle='--');
    axs[0, 1].axvline(x=max_d_MH, linewidth=1, color='r', linestyle='--')
    axs[0, 1].set_title('Morisita-Horn distance distribution')
    axs[0, 1].set_xlabel('Morisita-Horn distance')
    axs[0, 1].set_ylabel('Frequency')

    im = axs[1, 0].imshow(D_SH, vmin=0, vmax=1, cmap='jet')
    axs[1, 0].invert_yaxis()
    plt.colorbar(im, ax=axs[1, 0], fraction=0.046, pad=0.04)
    axs[1, 0].set_title('Shannon Heterogeneity pairwise distance')

    N = D_SH[np.triu_indices(D_SH.shape[0], k=1)].size
    hist, bins = np.histogram(D_SH[np.triu_indices(D_SH.shape[0], k=1)],
                              bins=np.arange(0, 1, 0.05), density=False)
    x = (bins[1:] + bins[:-1])/2
    axs[1, 1].plot(x, hist/N)
    # axs[1, 1].axhline(y=0.05, linewidth=1, color='r', linestyle='--');
    axs[1, 1].axvline(x=max_d_SH, linewidth=1, color='r', linestyle='--')
    axs[1, 1].set_title('Shannon Heterogeneity distance distribution')
    axs[1, 1].set_xlabel('Shannon Heterogeneity distance')
    axs[1, 1].set_ylabel('Frequency')

    x = D_MH[np.triu_indices(D_MH.shape[0], k=1)].ravel()
    y = D_SH[np.triu_indices(D_SH.shape[0], k=1)].ravel()
    axs[2, 0].scatter(x, y, c='k', s=2, marker='o')
    axs[2, 0].set_title('Distances correlation')
    axs[2, 0].set_xlabel('Morisita-Horn distance')
    axs[2, 0].set_ylabel('Shannon Heterogeneity distance')
    sns.regplot(x=x, y=y, color="r", ci=68, scatter=False, ax=axs[2, 0])

    d = x-y
    N = d.size
    hist, bins = np.histogram(d, bins=np.arange(np.min(d), np.max(d), 0.01),
                              density=False)
    b = (bins[1:] + bins[:-1])/2
    axs[2, 1].plot(b, hist/N)
    axs[2, 1].set_title('Morisita-Horn - Shannon Heterogeneity Difference')
    axs[2, 1].set_xlabel('MH dist - SH dist')
    axs[2, 1].set_ylabel('Frequency')
    axs[2, 1].axvline(x=0, linewidth=1, color='k', linestyle='--')

    return(fig)


def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    tlt = kwargs.pop('title', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated) - ' + tlt)
        plt.xlabel('(group size)')
        plt.ylabel('distance')
        for k, (i, d, c) in enumerate(zip(ddata['icoord'],
                                          ddata['dcoord'],
                                          ddata['color_list'])):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
            # plt.xticks(rotation=90)
    return ddata


def distance_tree(D, metric, max_d, ttl):

    Z = linkage(D[np.triu_indices(D.shape[0], k=1)], 'complete', metric=metric)
    clusters = fcluster(Z, max_d, criterion='distance')
    plt.figure(figsize=(15, 10), facecolor='w', edgecolor='k')
    dn = fancy_dendrogram(Z,
                          max_d=max_d,
                          title=ttl,
                          above_threshold_color='grey',
                          truncate_mode='lastp',
                          p=20,
                          show_leaf_counts=True,
                          show_contracted=True)
    return(Z, clusters, dn)


def cluster_regions(pop, M, shape):

    clusters_regs = np.zeros(shape)
    clusters_regs[:] = np.nan
    for idx, m in enumerate(M):
        aux = pop.iloc[idx]
        row_lo = np.round(aux.row_lo).astype('uint16')
        row_hi = np.round(aux.row_hi).astype('uint16')
        col_lo = np.round(aux.col_lo).astype('uint16')
        col_hi = np.round(aux.col_hi).astype('uint16')
        clusters_regs[row_lo:row_hi, col_lo:col_hi] = m
    return(clusters_regs)


def cluster_regions_types(pop, classes, shape):

    labs = []
    for i, typf in enumerate(classes['class_code']):
        for j, typt in enumerate(classes['class_code']):
            if (i < j):
                labs = labs + [typf + ':' + typt]

    nprs = (len(classes['class_code'])) * (len(classes['class_code']) - 1) // 2
    clusters_regs = np.zeros((nprs, shape[0], shape[1]))

    clusters_regs[:] = np.nan
    for idx, row in pop.iterrows():
        row_lo = np.round(row.row_lo).astype('uint16')
        row_hi = np.round(row.row_hi).astype('uint16')
        col_lo = np.round(row.col_lo).astype('uint16')
        col_hi = np.round(row.col_hi).astype('uint16')

        k = 0
        for i, typf in enumerate(classes['class_code']):
            for j, typt in enumerate(classes['class_code']):
                if (i < j):
                    clusters_regs[k,
                                  row_lo:row_hi,
                                  col_lo:col_hi] = row[labs[k]]
                    k = k+1

    return(clusters_regs, labs)


def regions_merge_mask_label(regs_in, label, mask_in):
    # merges quadrat dataframe with mask blobs

    regs = regs_in.copy()
    mask = mask_in.copy()
    mask[np.isnan(mask)] = 0
    mask = mask.astype(int)

    regs[label] = 0
    for i, row in regs.iterrows():
        # evaluates mask in quadrat
        m = mask[int(row.row_lo):int(row.row_hi),
                 int(row.col_lo):int(row.col_hi)].ravel()
        # assigns the most frequent blob value in each quadrat
        regs.at[i, label] = np.argmax(np.bincount(m))
    regs['is_' + label] = (regs[label] > 0)
    return(regs)


def reg_frame_to_array(regs, shap, fac):
    # turns a quadrat (region) dataframe into an array
    # 'fac' is the variable (factor) values in the array

    out = np.empty(shap)
    out[:] = np.nan

    for i, row in regs.iterrows():
        out[int(row.r), int(row.c)] = row[fac]

    return(out)


def points_mask_label(regs, factors, label, blob_mask, mask):
    # merges quadrat dataframe with mask blobs
    # returns a dataframe with pixel-resolution

    blobs = blob_mask.copy()
    blobs[np.isnan(blobs)] = 0
    blobs = blobs.astype(int)

    m, n = blobs.shape
    R, C = np.mgrid[:m, :n]
    out = pd.DataFrame({'row': R.ravel(),
                        'col': C.ravel(),
                        'roi': mask.ravel(),
                        label: blobs.ravel()})
    out['is_' + label] = (out[label] > 0)

    facs = np.empty((m, n, len(factors)))
    facs[:] = np.nan
    for i, row in regs.iterrows():
        for f, fac in enumerate(factors):
            facs[int(row.row_lo):int(row.row_hi),
                 int(row.col_lo):int(row.col_hi), f] = row[fac]

    for f, fac in enumerate(factors):
        out[fac] = facs[:, :, f].ravel()

    out.dropna(inplace=True)
    out.drop(out[out.roi == 0].index, inplace=True)
    out.drop(['roi'], axis=1, inplace=True)
    out.reset_index(drop=True, inplace=True)

    return(out)


def lme_patterns(M, pop, classes):

    aux = pop.drop([x for x in pop.columns
                    if x not in classes['class_code'].to_list()],
                   axis=1)
    # transforms frequencies to cell numbers
    for x in aux.columns:
        aux[x] = aux[x]*pop['total']

    # max_n = 10**np.ceil(np.log10(max(aux.max())*1.1))
    max_n = max(aux.max())*1.1

    # includes the total population in the diagram
    # (normalized to max for plotting)
    # lis = classes['class_code'].to_list()
    # cls = lis.copy()
    # lis.append('total')
    # aux = pop.drop([x for x in pop.columns if x not in lis], axis=1)
    # aux['total'] = aux['total']/np.max(aux['total'])
    # names = [classes.loc[classes['class_code'] ==x].class_name.values[0]
    #          if x in cls else 'Total' for x in aux.columns ]

    aux['lme'] = M
    lme = aux.groupby('lme').mean().reset_index()
    lme['num'] = aux.groupby('lme')['lme'].count().to_list()

    fig, ax = plt.subplots(int(np.ceil(len(lme)/2)), 2,
                           figsize=(12, 3.0 * len(lme)),
                           facecolor='w', edgecolor='k')
    shp = ax.shape

    for i, row in lme.iterrows():
        aux2 = aux.loc[aux['lme'] == row.lme].drop(['lme'], axis=1)
        sns.axes_style("whitegrid")
        sns.violinplot(data=aux2, ax=ax[np.unravel_index(i, shp)],
                       palette=classes.class_color,
                       scale='count', inner='box')
        # sns.boxplot(data=aux2, ax=ax[np.unravel_index(i, shp)],
        #               palette=classes.class_color)
        ax[np.unravel_index(i, shp)].set_title('Local Micro-Environment lme=' +
                                               str(int(row.lme)) +
                                               '\nNumber of quadrats: ' +
                                               str(int(row.num)))
        # ax[np.unravel_index(i, shp)].set_ylim([0, 1])
        ax[np.unravel_index(i, shp)].set_ylim([0, max_n])
        ax[np.unravel_index(i, shp)].set_ylabel('Counts')
        ax[np.unravel_index(i, shp)].set_xticklabels(classes.class_name)
        ax[np.unravel_index(i, shp)].set_xticklabels(classes.class_name)
        # ax[np.unravel_index(i, shp)].set_xticklabels(names, fontsize=8)
        # ax[np.unravel_index(i, shp)].set_xticklabels(names, fontsize=8)

    return([lme, fig])


def hot_regions(M, class_1, class_2, shape, binsiz):

    hot1_regs = np.zeros(shape)
    hot1_regs[:] = np.nan
    hot2_regs = np.zeros(shape)
    hot2_regs[:] = np.nan
    hothot_regs = np.zeros(shape)
    hothot_regs[:] = np.nan
    coldcold_regs = np.zeros(shape)
    coldcold_regs[:] = np.nan
    hot1_cold2_regs = np.zeros(shape)
    hot1_cold2_regs[:] = np.nan
    cold1_hot2_regs = np.zeros(shape)
    cold1_hot2_regs[:] = np.nan

    reg1 = M[class_1, :, :]
    reg2 = M[class_2, :, :]

    for (i, j), val in np.ndenumerate(reg1*reg2):
        if ~np.isnan(val):
            row = i*binsiz
            col = j*binsiz
            vhh = np.logical_and(reg1[i, j] > 0, reg2[i, j] > 0)
            vcc = np.logical_and(reg1[i, j] < 0, reg2[i, j] < 0)
            vhc = np.logical_and(reg1[i, j] > 0, reg2[i, j] < 0)
            vch = np.logical_and(reg1[i, j] < 0, reg2[i, j] > 0)

            hot1_regs[row:(row + binsiz), col:(col + binsiz)] = reg1[i, j]
            hot2_regs[row:(row + binsiz), col:(col + binsiz)] = reg2[i, j]
            hothot_regs[row:(row + binsiz), col:(col + binsiz)] = vhh
            coldcold_regs[row:(row + binsiz), col:(col + binsiz)] = vcc
            hot1_cold2_regs[row:(row + binsiz), col:(col + binsiz)] = vhc
            cold1_hot2_regs[row:(row + binsiz), col:(col + binsiz)] = vch

    return([hot1_regs, hot2_regs,
            hothot_regs, coldcold_regs,
            hot1_cold2_regs, cold1_hot2_regs])


def get_comparisons_df(c, classes):

    comps = pd.DataFrame(c, columns=['code_1', 'code_2'])
    comps['inx_1'] = ''
    comps['inx_2'] = ''
    comps['name_1'] = ''
    comps['name_2'] = ''
    comps['tt'] = ''
    comps['ttl'] = ''

    for i, row in comps.iterrows():
        row.inx_1 = classes.loc[classes['class_code'] ==
                                row.code_1].index[0]
        row.name_1 = classes.loc[classes['class_code'] ==
                                 row.code_1].class_name.values[0]
        row.inx_2 = classes.loc[classes['class_code'] ==
                                row.code_2].index[0]
        row.name_2 = classes.loc[classes['class_code'] ==
                                 row.code_2].class_name.values[0]
        tt = row.code_1 + ':' + row.code_2
        row.tt = tt
        row.ttl = row.name_1 + ':' + row.name_2

    return comps


def colorbar(mappable, ticks=None):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax, ticks=ticks)
    plt.sca(last_axes)
    return cbar


def plot_coloc_regions_scarce(cell_data, classes, scale, contourpoints,
                              ttl1, coloc_regs1,
                              ttl2, coloc_regs2,
                              ttl3, coloc_regs3,
                              units, xedges, yedges, redges, cedges, ar):

    fig, ax = plt.subplots(2, 2, figsize=(2*6, 2*np.ceil(6/ar)),
                           facecolor='w', edgecolor='k')

    plt.set_cmap('jet')
    for i, row in classes.iterrows():
        aux = cell_data.loc[cell_data['class_code'] == row['class_code']]
        fig = plot_landscape_scarce(ax[0, 0], aux,
                                    row.class_color, row.class_name,
                                    units, xedges, yedges)
    ax[0, 0].set_aspect(aspect=1)
    ax[0, 0].scatter(contourpoints[:, 0],
                     contourpoints[:, 1], c='y', s=.1, marker='o')
    ax[0, 0].set_title('All cells', fontsize=12)

    im4 = plot_landscape(ax[0, 1], coloc_regs1[::-1], 0, 1, units,
                         cedges, redges, xedges, yedges, 'jet')
    ax[0, 1].set_title('Colocalization Index: ' + ttl1, fontsize=12, )
    colorbar(im4)

    im5 = plot_landscape(ax[1, 0], coloc_regs2[::-1], 0, 1, units,
                         cedges, redges, xedges, yedges, 'jet')
    ax[1, 0].set_title('Colocalization Index: ' + ttl2, fontsize=12, )
    colorbar(im5)

    im6 = plot_landscape(ax[1, 1], coloc_regs3[::-1], 0, 1, units,
                         cedges, redges, xedges, yedges, 'jet')
    ax[1, 1].set_title('Colocalization Index: ' + ttl3, fontsize=12, )
    colorbar(im6)

    plt.tight_layout()

    return(fig)


def plot_regions_scarce(cell_data, classes, scale, contourpoints,
                        MH_clusters_regs, SH_clusters_regs,
                        comps, coloc_regs,
                        units, xedges, yedges, redges, cedges, ar,
                        spoint=1, fontsiz=20):

    ros = (len(comps)//3) + 1

    fig, ax = plt.subplots(ros, 3, figsize=(3*6, ros*np.ceil(6/ar)),
                           facecolor='w', edgecolor='k')

    plt.set_cmap('jet')
    for i, row in classes.iterrows():
        aux = cell_data.loc[cell_data['class_code'] == row['class_code']]
        fig = plot_landscape_scarce(ax[0, 0],
                                    aux, row.class_color, row.class_name,
                                    units, xedges, yedges,
                                    spoint=spoint, fontsiz=fontsiz)
    ax[0, 0].set_aspect(aspect=1)
    ax[0, 0].scatter(contourpoints[:, 0],
                     contourpoints[:, 1], c='y', s=3, marker='o')
    ax[0, 0].set_title('All cells', fontsize=24, y=1.02)

    flev = np.nanmin(MH_clusters_regs).astype('uint16')
    llev = np.nanmax(MH_clusters_regs).astype('uint16')
    nlevs = llev - flev + 1
    cmap = plt.get_cmap('tab20', nlevs)
    im2 = plot_landscape(ax[0, 1], MH_clusters_regs, flev-0.5, llev+0.5, units,
                         cedges, redges, xedges, yedges, cmap, fontsiz=fontsiz)
    ax[0, 1].set_title('MH Cluster regions', fontsize=24, y=1.02)
    colorbar(im2, ticks=np.arange(flev, llev+1))

    flev = np.nanmin(SH_clusters_regs).astype('uint16')
    llev = np.nanmax(SH_clusters_regs).astype('uint16')
    nlevs = llev - flev + 1
    cmap = plt.get_cmap('tab20', nlevs)
    im3 = plot_landscape(ax[0, 2], SH_clusters_regs, flev-0.5, llev+0.5, units,
                         cedges, redges, xedges, yedges, cmap, fontsiz=fontsiz)
    ax[0, 2].set_title('SH Cluster regions', fontsize=24, y=1.02)
    colorbar(im3, ticks=np.arange(flev, llev+1))

    for i, row in comps.iterrows():
        rc = np.unravel_index(i+3, (ros, 3))
        crs = coloc_regs[row.tti, :, :]
        imi = plot_landscape(ax[rc], crs, 0, 1, units,
                             cedges, redges, xedges, yedges, 'jet',
                             fontsiz=fontsiz)
        ax[rc].set_title('Colocalization Index:\n' + row.ttl,
                         fontsize=fontsiz, y=1.02)
        colorbar(imi)

    plt.tight_layout()

    return(fig)


def plot_TME(cell_data, classes, scale, clusters_regs,
             units, xedges, yedges, redges, cedges, ttl, ar):

    fig, ax = plt.subplots(1, 1, figsize=(12, np.ceil(12/ar)),
                           facecolor='w', edgecolor='k')

    flev = np.nanmin(clusters_regs).astype('uint16')
    llev = np.nanmax(clusters_regs).astype('uint16')
    nlevs = llev - flev + 1
    cmap = plt.get_cmap('tab10', nlevs)
    im2 = plot_landscape(ax, clusters_regs, flev-0.5, llev+0.5, units,
                         cedges, redges, xedges, yedges, cmap)
    ax.set_title(ttl, fontsize=12)
    colorbar(im2, ticks=np.arange(flev, llev+1))
    plt.tight_layout()

    return(fig)


def plot_hot_regions_scarce(h1_regs, h2_regs,
                            hh_regs, cc_regs, hc_regs, ch_regs,
                            units, xedges, yedges, redges, cedges,
                            class_name_1, class_name_2,
                            ar, fontsiz=20, titsiz=24):

    cmap1 = mpl.colors.ListedColormap(['#3182bd', '#31a354', '#de2d26'])
    cmap2 = mpl.colors.ListedColormap(['#fec44f', '#d95f0e'])
    ttl = class_name_1 + ' : ' + class_name_2

    fig, ax = plt.subplots(3, 2, figsize=(2*10, 3*np.ceil(10/ar)),
                           facecolor='w', edgecolor='k')

    im1 = plot_landscape(ax[0, 0], h1_regs, -1.5, 1.5, units,
                         cedges, redges, xedges, yedges,
                         cmap1, fontsiz=fontsiz)
    ax[0, 0].set_title(class_name_1 + ' Getis-Ord regions',
                       fontsize=titsiz, y=1.02)
    cbar = colorbar(im1, ticks=np.arange(-1, 2))
    cbar.ax.set_yticklabels(['Cold', 'Reg', 'Hot'])

    im2 = plot_landscape(ax[0, 1], h2_regs, -1.5, 1.5, units,
                         cedges, redges, xedges, yedges,
                         cmap1, fontsiz=fontsiz)
    ax[0, 1].set_title(class_name_2 + ' Getis-Ord regions',
                       fontsize=titsiz, y=1.02)
    cbar = colorbar(im2, ticks=np.arange(-1, 2))
    cbar.ax.set_yticklabels(['Cold', 'Reg', 'Hot'])

    im3 = plot_landscape(ax[1, 0], hh_regs, -0.5, 1.5, units,
                         cedges, redges, xedges, yedges,
                         cmap2, fontsiz=fontsiz)
    ax[1, 0].set_title('Hot - Hot (' + ttl + ')',
                       fontsize=titsiz, y=1.02)
    cbar = colorbar(im3, ticks=np.arange(-0, 2))
    cbar.ax.set_yticklabels(['No', 'Yes'])

    im4 = plot_landscape(ax[1, 1], cc_regs, -0.5, 1.5, units,
                         cedges, redges, xedges, yedges,
                         cmap2, fontsiz=fontsiz)
    ax[1, 1].set_title('Cold - Cold (' + ttl + ')', fontsize=titsiz, y=1.02)
    cbar = colorbar(im4, ticks=np.arange(0, 2))
    cbar.ax.set_yticklabels(['No', 'Yes'])

    im5 = plot_landscape(ax[2, 0], hc_regs, -0.5, 1.5, units,
                         cedges, redges, xedges, yedges,
                         cmap2, fontsiz=fontsiz)
    ax[2, 0].set_title('Hot - Cold (' + ttl + ')', fontsize=titsiz, y=1.02)
    cbar = colorbar(im5, ticks=np.arange(-0, 2))
    cbar.ax.set_yticklabels(['No', 'Yes'])

    im6 = plot_landscape(ax[2, 1], ch_regs, -0.5, 1.5, units,
                         cedges, redges, xedges, yedges,
                         cmap2, fontsiz=fontsiz)
    ax[2, 1].set_title('Cold - Hot (' + ttl + ')', fontsize=titsiz, y=1.02)
    cbar = colorbar(im6, ticks=np.arange(0, 2))
    cbar.ax.set_yticklabels(['No', 'Yes'])

    plt.tight_layout()

    return(fig)


def df_regions_type(df, typ, shape):

    regs = np.zeros(shape)
    regs[:] = np.nan
    for i, row in df.iterrows():
        row_lo = np.round(row.row_lo).astype('uint16')
        row_hi = np.round(row.row_hi).astype('uint16')
        col_lo = np.round(row.col_lo).astype('uint16')
        col_hi = np.round(row.col_hi).astype('uint16')
        val = row[typ]
        regs[row_lo:row_hi, col_lo:col_hi] = val
    return(regs)


def plot_nnregions(NNindex, comps, shape, units,
                   xedges, yedges, redges, cedges,
                   ar, fontsiz=20, titsiz=24):

    nrows = int(np.ceil(len(comps)/6))
    fig, ax = plt.subplots(nrows, 6, figsize=(6*6, nrows*np.ceil(6/ar)),
                           facecolor='w', edgecolor='k')
    shp = ax.shape

    for i, row in comps.iterrows():
        aux = df_regions_type(NNindex, row.tt, shape)
        rc = ax_index(i, shp)
        im = plot_landscape(ax[rc], aux, -1, 1, units,
                            cedges, redges, xedges, yedges, 'jet',
                            fontsiz=fontsiz)
        ax[rc].set_title('Nearest Neighbor Index:\n' +
                         row.ttl, fontsize=titsiz, y=1.02)
        colorbar(im)
    plt.tight_layout()

    return(fig)


def plot_rhregion(RHindex, comp, shape, units,
                  xedges, yedges, redges, cedges,
                  ar, fontsiz=20, titsiz=24):

    fig, ax = plt.subplots(1, 1, figsize=(12, np.ceil(12/ar)),
                           facecolor='w', edgecolor='k')

    aux = df_regions_type(RHindex, comp.tt[0], shape)
    im = plot_landscape(ax, aux, -5, 5, units,
                        cedges, redges, xedges, yedges,
                        'jet', fontsiz=fontsiz)
    ax.set_title('Ripley`s H Index: ' + comp.ttl[0], fontsize=titsiz, y=1.02)
    colorbar(im)
    plt.tight_layout()

    return([fig, ax])


def plot_rhregions(RHindex, comps, shape, units,
                   xedges, yedges, redges, cedges,
                   ar, fontsiz=20, titsiz=24):

    nrows = int(np.ceil(len(comps)/3))
    fig, ax = plt.subplots(nrows, 3, figsize=(3*6, nrows*np.ceil(6/ar)),
                           facecolor='w', edgecolor='k')
    shp = ax.shape

    # minv = min(RHindex.loc[:, comps.tt].min(axis=1))
    # maxv = max(RHindex.loc[:, comps.tt].max(axis=1))

    for i, row in comps.iterrows():

        aux = df_regions_type(RHindex, row.tt, shape)
        rc = ax_index(i, shp)
        im = plot_landscape(ax[rc], aux, -5, 5, units,
                            cedges, redges, xedges, yedges,
                            'jet', fontsiz=fontsiz)
        ax[rc].set_title('Ripley`s H Index:\n' + row.ttl,
                         fontsize=titsiz, y=1.02)
        colorbar(im)
    plt.tight_layout()

    return([fig, ax])


def plot_infiltration_scarce(cent_points, cent_name, cent_color,
                             test_points, test_name, test_color,
                             r, c, z, th, levs, mask,
                             redges, cedges, xedges, yedges,
                             scale, units, spoint, ar):

    fig, ax = plt.subplots(2, 2, figsize=(2*6, 2.0 * np.ceil(6/ar)),
                           facecolor='w', edgecolor='k')

    # plot (ranked) density values in the landscape
    zpoints = z.ravel()
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax[0, 0].set_title('Ranked KDE values for ' + cent_name, fontsize=12)
    ax[0, 0].plot(np.arange(0, len(zpoints)), np.sort(zpoints), 'ok', ms=2)
    ax[0, 0].axhline(y=th, linewidth=1, color='r', linestyle='--')
    ax[0, 0].set_xticks(np.arange(0, len(zpoints)+1, len(zpoints)/5).tolist())
    ax[0, 0].set_yscale('log')
    ax[0, 0].set_xlabel('rank')
    ax[0, 0].set_ylabel('log(KDE)')
    ax[0, 0].text(0.05, 0.95, "th = " +
                  np.format_float_scientific(th, unique=False, precision=2),
                  transform=ax[0, 0].transAxes, fontsize=14,
                  verticalalignment='top', bbox=props)

    # transform mask coordinates to (x,y)
    x = c*scale
    y = (mask.shape[0] - r)*scale

    # Plot the kernel density estimate with cells
    CS1 = ax[0, 1].contourf(x, y, z, levs, locator=ticker.LogLocator(),
                            cmap='RdBu_r')
    plt.colorbar(CS1, ax=ax[0, 1],
                 format=ticker.FuncFormatter(fmt),
                 fraction=0.046, pad=0.04)
    ax[0, 1].set_aspect(aspect=1)
    CS2 = ax[0, 1].contour(x, y, mask, [0.5], linewidths=1, colors='yellow')
    ax[0, 1].set_title('KDE contours for ' + cent_name, fontsize=12)
    ax[0, 1].set_xticks(xedges, minor=False)
    ax[0, 1].set_yticks(yedges, minor=False)
    ax[0, 1].grid(which='major', linestyle='-', linewidth='0.3', color='black')
    ax[0, 1].set_xlim(min(xedges), max(xedges))
    ax[0, 1].set_ylim(min(yedges), max(yedges))
    ax[0, 1].set_xlabel(units)
    ax[0, 1].set_ylabel(units)
    ax[0, 1].set_xticklabels(xedges, rotation=90, fontsize=6)
    ax[0, 1].set_yticklabels(yedges, fontsize=6)
    set_ticklabels_spacing(ax[0, 1], 10)

    # Plot the th-level kde mask with center cells
    ax[1, 0].imshow(mask, 'RdBu_r')
    ax[1, 0].contour(c, r, mask, [0.5], linewidths=1, colors='green')
    ax[1, 0].plot(cent_points.col, cent_points.row, 'ow', ms=spoint)
    ax[1, 0].set_title('Landscape for ' + cent_name, fontsize=12)
    ax[1, 0].set_xticks(cedges, minor=False)
    ax[1, 0].set_yticks(redges, minor=False)
    ax[1, 0].grid(which='major', linestyle='-', linewidth='0.5', color='white')
    ax[1, 0].set_xlim(min(cedges), max(cedges))
    ax[1, 0].set_ylim(max(redges), min(redges))
    ax[1, 0].set_xticklabels(xedges, rotation=90, fontsize=6)
    ax[1, 0].set_yticklabels(yedges[::-1], fontsize=6)
    ax[1, 0].set_xlabel(units)
    ax[1, 0].set_ylabel(units)
    set_ticklabels_spacing(ax[1, 0], 10)

    # Plot the th-level kde mask with cells
    ax[1, 1].imshow(mask, 'RdBu_r')
    ax[1, 1].contour(c, r, mask, [0.5], linewidths=1, colors='green')
    ax[1, 1].plot(test_points.col, test_points.row, 'ow', ms=spoint)
    ax[1, 1].set_title('Landscape for ' + test_name, fontsize=12)
    ax[1, 1].set_xticks(cedges, minor=False)
    ax[1, 1].set_yticks(redges, minor=False)
    ax[1, 1].grid(which='major', linestyle='-', linewidth='0.5', color='white')
    ax[1, 1].set_xlim(min(cedges), max(cedges))
    ax[1, 1].set_ylim(max(redges), min(redges))
    ax[1, 1].set_xticklabels(xedges, rotation=90, fontsize=6)
    ax[1, 1].set_yticklabels(yedges[::-1], fontsize=6)
    ax[1, 1].set_xlabel(units)
    ax[1, 1].set_ylabel(units)
    set_ticklabels_spacing(ax[1, 1], 10)

    plt.tight_layout()
    contourpoints = np.vstack(CS2.allsegs[0])

    return([fig, contourpoints])


def KDE_mask_(cell_data, classes, cent_code, test_code, shape,
              scale, units, subbinsiz,
              redges, cedges, xedges, yedges, ar, all_cells):

    cent_data = cell_data.loc[cell_data['class_code'] == cent_code]
    test_data = cell_data.loc[cell_data['class_code'] == test_code]
    cent_name = classes.loc[classes['class_code'] ==
                            cent_code].class_name.values[0]
    cent_color = classes.loc[classes['class_code'] ==
                             cent_code].class_color.values[0]
    test_name = classes.loc[classes['class_code'] ==
                            test_code].class_name.values[0]
    test_color = classes.loc[classes['class_code'] ==
                             test_code].class_color.values[0]

    # do KDE based on pixel locations of center cells (e.g. tumor cells)
    [r, c, z] = KDE(cent_data, shape, 1, 1, subbinsiz/2)

    # threshold for landscape edge
    aux = np.around(np.log10(z), 2)
    levs = 10**np.arange(np.around(np.min(aux)), np.around(np.max(aux)+1))

    if (all_cells):
        # get values above background
        th = 10.0 ** np.min(aux[aux > np.min(aux)])
    else:
        # get top order of magnitud
        th = 10.0 ** np.min(aux[aux > (np.max(aux) - 1.5)])

    # generate a binary mask for the tumor
    cent_mask = kde_mask(z, th, 10000, fill_holes=False)

    # apply mask filter to test points overpoints are test cells
    # (e.g. lynphocytes) infiltrated of the regions with
    # abundant center cells (e.g cancer)
    overpix = mask_filter(cent_mask, np.array(test_data[['row', 'col']]))
    overpoints = np.array((overpix[:, 1] * scale,
                           (shape[0] - overpix[:, 0]) * scale)).T

    # Plot kde regions
    fig, test_contourpix = plot_infiltration_scarce(cent_data[['row', 'col']],
                                                    cent_name, cent_color,
                                                    test_data[['row', 'col']],
                                                    test_name, test_color,
                                                    r, c, z, th,
                                                    levs, cent_mask,
                                                    redges, cedges,
                                                    xedges, yedges,
                                                    scale, units, 0.1, ar)

    test_contourpoints = np.array((test_contourpix[:, 0] * scale,
                                   (shape[0] - test_contourpix[:, 1])*scale)).T

    # points inside tumor
    test_pix = np.array(np.where(cent_mask > 0)).T
    test_points = np.array((test_pix[:, 1]*scale,
                            (cent_mask.shape[0] - test_pix[:, 0])*scale)).T

    return([cent_mask, test_points, test_contourpoints, overpoints])


def infiltration_profile(cell_data, classes, cent_code, test_code, shape,
                         scale, units, subbinsiz,
                         redges, cedges, xedges, yedges, ar, all_cells):

    cent_data = cell_data.loc[cell_data['class_code'] == cent_code]
    test_data = cell_data.loc[cell_data['class_code'] == test_code]
    cent_name = classes.loc[classes['class_code'] ==
                            cent_code].class_name.values[0]
    cent_color = classes.loc[classes['class_code'] ==
                             cent_code].class_color.values[0]
    test_name = classes.loc[classes['class_code'] ==
                            test_code].class_name.values[0]
    test_color = classes.loc[classes['class_code'] ==
                             test_code].class_color.values[0]

    # do KDE based on pixel locations of center cells (e.g. tumor cells)
    [r, c, z] = KDE(cent_data, shape, 1, 1, subbinsiz/2)

    # threshold for landscape edge
    aux = np.around(np.log10(z), 2)
    levs = 10**np.arange(np.around(np.min(aux)), np.around(np.max(aux)+1))

    if (all_cells):
        # get values above background
        th = 10**np.min(aux[aux > np.min(aux)])
    else:
        # get top order of magnitud
        th = 10**np.min(aux[aux > (np.max(aux) - 1.5)])

    # generate a binary mask for the tumor
    cent_mask = kde_mask(z, th, 10000, fill_holes=False)

    # apply mask filter to test points
    # overpoints are test cells (e.g. lynphocytes) infiltrated of the
    # regions with abundant center cells (e.g cancer)
    overpix = mask_filter(cent_mask, np.array(test_data[['row', 'col']]))
    overpoints = np.array((overpix[:, 1]*scale,
                           (shape[0] - overpix[:, 0])*scale)).T

    # Plot kde regions
    fig, test_contourpix = plot_infiltration_scarce(cent_data[['row', 'col']],
                                                    cent_name, cent_color,
                                                    test_data[['row', 'col']],
                                                    test_name, test_color,
                                                    r, c, z, th, levs,
                                                    cent_mask,
                                                    redges, cedges,
                                                    xedges, yedges,
                                                    scale, units, 0.1, ar)

    test_contourpoints = np.array((test_contourpix[:, 0]*scale,
                                   (shape[0] - test_contourpix[:, 1])*scale)).T

    # points inside tumor
    test_pix = np.array(np.where(cent_mask > 0)).T
    test_points = np.array((test_pix[:, 1]*scale,
                            (cent_mask.shape[0] - test_pix[:, 0])*scale)).T

    return([cent_mask, test_points, test_contourpoints, overpoints])


def plot_infiltration_dist(inpoints, contourpoints, overpoints,
                           reps, bins, tt1, tt2, units, ar):

    # build a KDTree with contourpoint and query distance to each overpoint
    kdetree = KDTree(contourpoints)
    distance, index = kdetree.query(overpoints)
    x = (bins[1:] + bins[:-1])/2

    # built a histogram of closest distances of infiltrated cells
    hist, b = np.histogram(distance, bins=bins, density=True)

    # get replicants of equal size with inpoints to check for significance
    L = overpoints.shape[0]
    M = np.arange(inpoints.shape[0])
    His = np.zeros((reps, bins.shape[0]-1))

    for r in np.arange(reps):
        update_progress(r / reps, 'Infiltration bootstrap...')
        i = np.random.choice(M, L)
        di, ii = kdetree.query(inpoints[i, :])
        hi, b = np.histogram(di, bins=bins, density=True)
        His[r, :] = hi[:]

    Hmean = np.mean(His, axis=0)
    Hstd = np.std(His, axis=0)

    fig, ax = plt.subplots(1, 2, figsize=(2*6, np.ceil(6/ar)),
                           facecolor='w', edgecolor='k')

    ax[0].scatter(contourpoints[:, 0], contourpoints[:, 1],
                  c='r', s=.1, marker='o')
    ax[0].scatter(overpoints[:, 0], overpoints[:, 1],
                  c='g', s=.1, marker='o')
    ax[0].set_aspect(aspect=1)
    ax[0].set_title(tt1, fontsize=12)
    ax[0].set_xlabel(units)
    ax[0].set_ylabel(units)

    ax[1].plot(x, hist, linestyle='-', color='k', linewidth=2)
    ax[1].plot(x, Hmean, linestyle='-', color='r', linewidth=1)
    ax[1].plot(x, Hmean + Hstd,  linestyle='--', color='g', linewidth=1)
    ax[1].plot(x, Hmean - Hstd,  linestyle='--', color='g', linewidth=1)
    ax[1].set_title(tt2, fontsize=12)
    ax[1].set_xlabel('Distance to edge ' + units)
    ax[1].set_ylabel('Density')

    return fig


def SSH_factor_detector(y_column, x_column_nn, tabledata):

    # The factor detector q-statistic measures the
    # Spatial Stratified Heterogeneity (SSH) of a variable Y,
    # (also known as the determinant power of a covariate X of Y)
    #
    # Translated from R source: https://CRAN.R-project.org/package=geodetector
    #
    # Ref: https://cran.r-project.org/web/packages/
    #                          geodetector/vignettes/geodetector.html
    #
    # Parameters:
    # 1- 'y_column'    : name of explained (numerical) variable
    # 2- 'x_column_nn' : list of explanatory (categorical) variables
    # 3- 'tabledata'   : data-frame that contains all variables
    #
    # Outputs: Results is a data frame with factor detector include
    # 1- variable name : name of explanatory variable
    # 2- q-statistic   : SSH q-statistic
    # 3- F-statistic   : F value (assuming a random non-central F-distribution)
    # 4- p-value       : Prob that the q-value is observed by random chance
    #
    # Interpretation of q-test: The q-statistic measures the degree of SSH.
    #                           - Larger values (q~1) indicate a stronger
    #                             stratified heterogeneity effect. This means
    #                             very small within-strata variations and/or
    #                             very large between-strata variations.
    #                             Thus a strong association between the
    #                             explanatory variable and the explained
    #                             variable (ie. strata categories explain data)
    #                           - Small values (q~0) mean then within-strata
    #                             variations are large and/or between-strata
    #                             variations are small. Thus there is not
    #                             relationship between the strata categories
    #                             and the data.
    #
    #  The null hypothesis is defined as absence of within-stratum
    #  heterogeneity (q~0):
    #       H_0: there is no SSH (stratification is not significant), thus
    #            within and between strata heterogeneity are similar
    #       H_1: there is SSH (stratification is significant), thus
    #            within-strata heterogeneity is significantly smaller than
    #            between-strata heterogeneity.
    #
    # For more details of Geodetector method, please refer:
    # [1] Wang JF, Li XH, Christakos G, Liao YL, Zhang T, Gu X, Zheng XY.
    #     Geographical detectors-based health risk assessment and its
    #     application in the neural tube defects study of the Heshun Region,
    #     China. International Journal of Geographical. Information Science,
    #     2010, 24(1): 107-127.
    # [2] Wang JF, Zhang TL, Fu BJ. A measure of spatial stratified
    #     heterogeneity. Ecological. Indicators,2016, 67(2016): 250-256.
    # [3] Wang JF, Xu CD. Geodetector:Principle and prospective. Geographica
    #     Sinica, 2017, 72(1):116-134.

    factor_detector = pd.DataFrame({'factor': [],
                                    'stratum': [],
                                    'q_statistic': [],
                                    'F_statistic': [],
                                    'p_value': []})

    # obtain data for explained variable
    y_all = np.array(tabledata[y_column])
    inx = ~np.isnan(y_all)
    y = y_all[inx].tolist()

    for x_column in x_column_nn:

        # obtain data for explanatory variable
        x_all = tabledata[x_column]
        x = x_all[inx].tolist()

        # unique strata values (can be categorical values)
        strata = list(set(x))
        # number of strata
        N_stra = len(strata)

        # number of data points (population size)
        N_popu = len(tabledata.index)

        # variance of all samples
        N_var = np.var(y)
        # N_var = var(y)*(N_popu - 1)/N_popu

        # run stats on each strata
        strataVarSum = 0
        lamda_1st_sum = 0
        lamda_2nd_sum = 0

        for s in strata:
            yi = np.array([y[i] for i, si in enumerate(x) if si == s])
            LenInter = len(yi)
            strataVar = 0
            lamda_1st = 0
            lamda_2nd = 0

            if(LenInter <= 1):
                strataVar = 0
                lamda_1st = yi**2
                lamda_2nd = yi
            else:
                strataVar = (LenInter - 1) * np.var(yi)
                # strataVar = LenInter * np.var(yi)*(LenInter - 1)/LenInter
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
        aux = pd.DataFrame({'factor':      [y_column],
                            'stratum':     [x_column],
                            'q_statistic': [q],
                            'F_statistic': [F_value],
                            'p_value':     [p_value]})
        factor_detector = factor_detector.append(aux, ignore_index=True)

    factor_detector['is_SSH'] = factor_detector['p_value'] < 0.05

    return(factor_detector)


def SSH_interaction_detector(y_column, x_column_nn, tabledata):

    # The interaction detector function reveals whether the risk factors
    # {X_i} have an interactive influence on a factor Y.
    #
    # Translated from R,
    # Source: https://CRAN.R-project.org/package=geodetector
    #
    # Parameters:
    # 1- 'y_column'    : is the name of explained (numerical) variable in
    #    input dataset.
    # 2- 'x_column_nn' : is a list of names of (categorical) explanatory
    #    variables in input dataset.
    # 3- 'tabledata'   : is the dataframe that contains explained variable and
    #    explanatory variables.
    #
    # Outputs: table for the interactive q satistic between variables
    #
    # For more details of Geodetector method, please refer:
    # [1] Wang JF, Li XH, Christakos G, Liao YL, Zhang T, Gu X, Zheng XY.
    #     Geographical detectors-based health risk assessment and its
    #     application in the neural tube defects study of the Heshun Region,
    #     China. International Journal of Geographical
    #     Information Science, 2010, 24(1): 107-127.
    # [2] Wang JF, Zhang TL, Fu BJ. A measure of spatial stratified
    #     heterogeneity. Ecological. Indicators,2016, 67(2016): 250-256.

    # number of interctions
    n_x = len(x_column_nn)
    if(n_x < 2):
        print("ERROR: X input should be more than one variable")
        return(None)

    # combination  for X1,X2...
    x_x = [list(i) for i in list(combinations(x_column_nn, 2))]
    # n_x_x = len(x_x)

    # output data frame
    interaction_detector = pd.DataFrame({'factor': [],
                                         'X1': [],
                                         'X2': [],
                                         'X1_X2': [],
                                         'q_statistic_1': [],
                                         'p_value_1': [],
                                         'q_statistic_2': [],
                                         'p_value_2': [],
                                         'q_statistic': [],
                                         'p_value': [],
                                         'description': []})

    table = tabledata.copy()

    for n, [x1_colnam, x2_colnam] in enumerate(x_x):

        nam = x1_colnam + ":" + x2_colnam
        m1 = table[x1_colnam].astype(str)
        m2 = table[x2_colnam].astype(str)
        table[nam] = m1.str.cat(m2, sep='_')

        # combined factor detector
        factor = SSH_factor_detector(y_column, [nam], table)
        factor1 = SSH_factor_detector(y_column, [x1_colnam], table)
        factor2 = SSH_factor_detector(y_column, [x2_colnam], table)

        # q-statistics
        q = factor['q_statistic'][0]
        q1 = factor1['q_statistic'][0]
        q2 = factor2['q_statistic'][0]

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

        desc = outputRls + "; " + description

        # Create Result
        aux = pd.DataFrame({'factor': [y_column],
                            'X1': [x1_colnam],
                            'X2': [x2_colnam],
                            'X1_X2': [nam],
                            'q_statistic_1': [factor1['q_statistic'][0]],
                            'p_value_1': [factor1['p_value'][0]],
                            'q_statistic_2': [factor2['q_statistic'][0]],
                            'p_value_2': [factor2['p_value'][0]],
                            'q_statistic': [factor['q_statistic'][0]],
                            'p_value': [factor['p_value'][0]],
                            'description': [desc]})

        interaction_detector = interaction_detector.append(aux,
                                                           ignore_index=True)

    [sig, adj, a, b] = multipletests(interaction_detector['p_value'].tolist(),
                                     method='bonferroni')
    interaction_detector['p_adjust'] = adj
    interaction_detector['significance'] = sig

    return(interaction_detector[['factor', 'X1', 'X2', 'X1_X2',
                                 'q_statistic_1', 'p_value_1',
                                 'q_statistic_2', 'p_value_2',
                                 'q_statistic', 'p_value', 'p_adjust',
                                 'significance', 'description']])


def SSH_risk_detector(y_column, x_column, tabledata):

    # This function calculates the average values in each stratum of the
    # explanatory variable (X), and reports if a significant difference
    # between two strata exists.
    #
    # Translated from R,
    # Source: https://CRAN.R-project.org/package=geodetector
    #
    # Parameters:
    # 1- 'y_column'    : is the name of explained (numerical) variable in
    #                    input dataset.
    # 2- 'x_column'    : is the name of (categorical) explanatory variable in
    #                    input dataset.
    # 3- 'tabledata'   : is the dataframe that contains explained variable and
    #                    explanatory variables.
    #
    # Outputs: Results of risk detector include the means of explained variable
    #          in each stratum and the t-test for differences every pair of
    #          strata.
    #
    # For more details of Geodetector method, please refer:
    # [1] Wang JF, Li XH, Christakos G, Liao YL, Zhang T, Gu X, Zheng XY.
    #     Geographical detectors-based health risk assessment and its
    #     application in the neural tube defects study of the Heshun Region,
    #     China. International Journal of Geographical. Information Science,
    #     2010, 24(1): 107-127.
    # [2] Wang JF, Zhang TL, Fu BJ. A measure of spatial stratified
    #     heterogeneity. Ecological. Indicators,2016, 67(2016): 250-256.

    # obtain data for explained variable
    y_all = np.array(tabledata[y_column])
    inx = ~np.isnan(y_all)
    y = y_all[inx].tolist()

    # obtain data for explanatory variable
    x_all = tabledata[x_column]
    x = x_all[inx].tolist()

    # unique strata values (can be categorical values)
    strata = list(set(x))
    # number of strata
    # N_stra = len(strata)

    # run stats on each strata
    # mean_risk = np.zeros(N_stra)

    risk_detector = pd.DataFrame({'factor': [],
                                  'stratum_i': [],
                                  'stratum_j': [],
                                  'num_Y_i': [],
                                  'mean_Y_i': [],
                                  'num_Y_j': [],
                                  'mean_Y_j': [],
                                  't_statistic': [],
                                  'p_value': []})

    # for all strata combinations
    for i, si in enumerate(strata):
        # data in stratum 'i'
        yi = np.array([y[k] for k, s in enumerate(x) if s == si])

        for j, sj in enumerate(strata):

            if (i != j):
                # data in stratum 'j'
                yj = np.array([y[k] for k, s in enumerate(x) if s == sj])

                if (len(yi) > 1 and len(yj) > 1):
                    # Welch’s t-test
                    # (does not assume equal population variances)
                    [tij, pij] = sts.ttest_ind(yi, yj, equal_var=False)

                    aux = pd.DataFrame({'factor': [y_column],
                                        'stratum_i': [str(si)],
                                        'stratum_j': [str(sj)],
                                        'num_Y_i': [len(yi)],
                                        'mean_Y_i': [np.mean(yi)],
                                        'num_Y_j': [len(yj)],
                                        'mean_Y_j': [np.mean(yj)],
                                        't_statistic': [tij],
                                        'p_value': [pij]})

                    risk_detector = risk_detector.append(aux,
                                                         ignore_index=True)

    [sig, adj, a, b] = multipletests(risk_detector['p_value'].tolist(),
                                     method='bonferroni')
    risk_detector['p_adjust'] = adj
    risk_detector['significance'] = sig
    risk_detector = risk_detector.astype({'num_Y_i': int,
                                          'num_Y_j': int})

    return(risk_detector)


def SSH_ecological_detector(y_column, x_column_nn, tabledata):

    # This function identifies the impact of differences
    # between two factors  X1 ~ X2
    #
    # Translated from R,
    # Source: https://CRAN.R-project.org/package=geodetector
    #
    # Parameters:
    # 1- 'y_column'    : name of explained (numerical) variable in dataset
    # 2- 'x_column_nn' : list of names of (categorical) explanatory variables
    # 3- 'tabledata'   : dataframe that contains all variables
    #
    # Outputs: Results of ecological detector is the significance test of
    #          impact difference between two explanatory variables.
    #
    # For more details of Geodetector method, please refer:
    # [1] Wang JF, Li XH, Christakos G, Liao YL, Zhang T, Gu X, Zheng XY.
    #     Geographical detectors-based health risk assessment and its
    #     application in the neural tube defects study of the Heshun Region,
    #     China. International Journal of Geographical. Information Science,
    #     2010, 24(1): 107-127.
    # [2] Wang JF, Zhang TL, Fu BJ. A measure of spatial stratified
    #     heterogeneity. Ecological. Indicators,2016, 67(2016): 250-256.

    # number of interactions
    n_x = len(x_column_nn)
    if(n_x < 2):
        print("ERROR: X variables input should be more than one variable")
        return(None)

    # combination  for X1,X2...
    x_x = [list(k) for k in list(combinations(x_column_nn, 2))]
    # n_x_x = len(x_x)

    # output data frame
    ecological_detector = pd.DataFrame({'factor': [],
                                        'X1': [],
                                        'X2': [],
                                        'F_statistic': [],
                                        'p_value': []})

    for n, [x1_colnam, x2_colnam] in enumerate(x_x):

        # individual factors
        f_numerator = SSH_factor_detector(y_column,
                                          [x2_colnam],
                                          tabledata)['F_statistic'][0]
        f_denominator = SSH_factor_detector(y_column,
                                            [x1_colnam],
                                            tabledata)['F_statistic'][0]

        F1_value = f_numerator / f_denominator
        F2_value = f_denominator / f_numerator

        # p value (positive tail of the cumulative F-statistic)
        p1_value = sts.f.sf(x=F1_value,
                            dfn=len(tabledata.index) - 1,
                            dfd=len(tabledata.index) - 1)
        p2_value = sts.f.sf(x=F2_value,
                            dfn=len(tabledata.index) - 1,
                            dfd=len(tabledata.index) - 1)

        if p2_value < p1_value:
            F1_value = F2_value
            p1_value = p2_value

        # Create Result
        aux = pd.DataFrame({'factor': [y_column],
                            'X1': [x1_colnam],
                            'X2': [x2_colnam],
                            'F_statistic': [F1_value],
                            'p_value': [p1_value]})

        ecological_detector = ecological_detector.append(aux,
                                                         ignore_index=True)

    [sig, adj, a, b] = multipletests(ecological_detector['p_value'].tolist(),
                                     method='bonferroni')
    ecological_detector['p_adjust'] = adj
    ecological_detector['significance'] = sig

    return(ecological_detector)


def SSH_plot_stratification(ssh_table, data, varttl, stratattl):

    fig, ax = plt.subplots(int(np.ceil(len(ssh_table)/2)), 2,
                           figsize=(2*(6), (len(ssh_table)/2)*(6)),
                           facecolor='w', edgecolor='k')
    shp = ax.shape

    for i, row in ssh_table.iterrows():
        sns.axes_style("whitegrid")
        sns.violinplot(x=row.stratum,
                       y=row.factor,
                       data=data,
                       ax=ax[np.unravel_index(i, shp)],
                       palette="Set3",
                       scale='count',
                       inner='box')
        sig = '(ns)'
        if (row.p_value < 0.05):
            sig = '(*)'
        if (row.p_value < 0.01):
            sig = '(**)'
        if (row.p_value < 0.001):
            sig = '(***)'
        txt = 'N = ' + str(len(data)) + '; p_value = %.2e ' % row.p_value
        ax[np.unravel_index(i, shp)].set_title(txt + sig)
        ax[np.unravel_index(i, shp)].set_ylabel(varttl)
        ax[np.unravel_index(i, shp)].set_xlabel(stratattl[i])

    plt.tight_layout()


def SSH_all(data, name, facts, titls, strata, stlabs, outpth):

    if not os.path.exists(outpth):
        os.makedirs(outpth)

    sshtab = pd.DataFrame({'factor': [],
                           'stratum': [],
                           'q_statistic': [],
                           'F_statistic': [],
                           'p_value': []})

    for i, f in enumerate(facts):

        ft = f.replace(":", ".")

        update_progress(i / len(facts), 'SSH Stats...')

        sshfac = SSH_factor_detector(f, strata, data)
        SSH_plot_stratification(sshfac, data, titls[i], stlabs)
        plt.savefig(outpth + '/SSH_factor_' + name + '_' + ft + '.png',
                    bbox_inches='tight', dpi=300)
        plt.close()
        sshtab = sshtab.append(sshfac, ignore_index=True)

        # SSH, interaction between strata
        sshint = SSH_interaction_detector(f, strata, data)
        sshint.to_csv(outpth + '/SSH_interaction_' + name + '_' + ft + '.csv',
                      index=False)

        # SSH, significant risk differences between strata levels
        for stt in strata:
            sshrsk = SSH_risk_detector(f, stt, data)
            auy = sshrsk.loc[sshrsk['significance']]
            auy.to_csv(outpth + '/SSH_risk_' + name + '_' + ft +
                       '-' + stt + '.csv', index=False)

        # SSH, overall significance test of impact difference between strata
        ssheco = SSH_ecological_detector(f, strata, data)
        ssheco.to_csv(outpth + '/SSH_ecological_' + name + '_' + ft + '.csv',
                      index=False)

    update_progress(1, 'SSH Stats...... Done!!!\n')

    return(sshtab)


def plot_patches(arr, ar, units,
                 cedges, redges,
                 xedges, yedges,
                 ttl, fontsiz=20, titsiz=28):

    flev = np.nanmin(arr).astype('uint16')
    llev = np.nanmax(arr).astype('uint16')
    nlevs = llev - flev + 1
    cmap = plt.get_cmap('tab20', nlevs)
    fig, ax = plt.subplots(1, 1, figsize=(12, np.ceil(12/ar)),
                           facecolor='w', edgecolor='k')
    im = plot_landscape(ax, arr, flev-0.5, llev+0.5, units,
                        cedges, redges, xedges, yedges, cmap,
                        fontsiz=fontsiz)
    ax.set_title(ttl, fontsize=titsiz, y=1.02)
    cbar = colorbar(im, ticks=np.arange(flev, llev+1))
    cbar.ax.tick_params(labelsize=fontsiz)
    set_ticklabels_spacing(ax, 10)
    return([fig, ax])


def get_patch_metrics(ls, classes, lab):

    df = ls.compute_patch_metrics_df(metrics_kws={'area': {'hectares': False},
                                                  'perimeter_area_ratio': {'hectares': False}}).reset_index()
    df['area_fraction'] = df['area']/ls.landscape_area

    # plot some patch metrics distributions
    fig, axs = plt.subplots(2, 2, figsize=(12, 12),
                            facecolor='w', edgecolor='k')

    col = list(mpl.colors.TABLEAU_COLORS)
    for i, class_val in enumerate(classes):

        mif = np.log10(np.min(df.area_fraction))
        maf = np.log10(np.max(df.area_fraction))
        dat = df.loc[df['class_val'] ==
                     class_val].area_fraction.apply(np.log10)
        hist, bins = np.histogram(dat,
                                  bins=np.arange(mif,
                                                 maf,
                                                 (maf-mif)/11),
                                  density=False)
        x = (bins[1:] + bins[:-1])/2
        axs[0, 0].plot(x, hist, label=lab[i], color=col[i])

        mir = np.log10(np.min(df.perimeter_area_ratio))
        mar = np.log10(np.max(df.perimeter_area_ratio))
        dat = df.loc[df['class_val'] ==
                     class_val].perimeter_area_ratio.apply(np.log10)
        hist, bins = np.histogram(dat,
                                  bins=np.arange(mir,
                                                 mar,
                                                 (mar-mir)/11),
                                  density=False)
        x = (bins[1:] + bins[:-1])/2
        axs[0, 1].plot(x, hist, label=lab[i], color=col[i])

        mis = np.min(df.shape_index)
        mas = np.max(df.shape_index)
        dat = df.loc[df['class_val'] == class_val].shape_index
        hist, bins = np.histogram(dat,
                                  bins=np.arange(mis,
                                                 mas,
                                                 (mas-mis)/11),
                                  density=False)

        x = (bins[1:] + bins[:-1]) / 2
        axs[1, 0].plot(x, hist, label=lab[i], color=col[i])

        mie = np.min(df.euclidean_nearest_neighbor)
        mae = np.max(df.euclidean_nearest_neighbor)
        dat = df.loc[df['class_val'] == class_val].euclidean_nearest_neighbor
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
        lambda x, p: '10^%.1f' % x))
    axs[0, 0].legend(loc='upper right')
    axs[0, 0].set_xlabel('Area fraction of patches')
    axs[0, 0].set_ylabel('Frequency')

    axs[0, 1].set_xticks(np.arange(0, 10, .1))
    axs[0, 1].set_xlim(mir, mar)
    axs[0, 1].get_xaxis().set_major_formatter(mpl.ticker.FuncFormatter(
        lambda x, p: '10^%.1f' % x))
    # axs[0, 1].legend(loc='upper right')
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

    return([df, fig, axs])


def get_class_metrics(ls, lab):

    df = ls.compute_class_metrics_df(metrics_kws={'total_area': {'hectares': False},
                                                  'perimeter_area_ratio': {'hectares': False},
                                                  'patch_density': {'hectares': False},
                                                  'edge_density': {'hectares': False},
                                                  'effective_mesh_size': {'hectares': False}}).reset_index()

    # plot some patch metrics distributions
    cols = list(mpl.colors.TABLEAU_COLORS)[0:len(lab)]
    fig, axs = plt.subplots(2, 2, figsize=(15, 15),
                            facecolor='w', edgecolor='k')

    axs[0, 0].bar(lab, df.patch_density, color=cols, align='center')
    axs[0, 0].set_title('Patch Density')

    axs[0, 1].bar(lab, df.largest_patch_index, color=cols, align='center')
    axs[0, 1].set_title('Largest Patch Index')

    axs[1, 0].bar(lab, df.edge_density, color=cols, align='center')
    axs[1, 0].set_title('Edge Density')

    axs[1, 1].bar(lab, df.landscape_shape_index, color=cols, align='center')
    axs[1, 1].set_title('Landscape Shape Index')

    plt.tight_layout()

    return([df, fig, axs])
