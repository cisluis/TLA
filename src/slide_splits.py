"""TLA 
"""

# %% Imports

import os
import sys
import math
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from matplotlib.widgets import AxesWidget, Button


# %% Private classes

class bttnwidget(AxesWidget):
    def __init__(self, ax, lab):
        super().__init__(ax)
        
        self.lab = lab
        self.clicked = False
            
        self.button = Button(ax, lab)
        self.button.on_clicked(self.on_click)

    def on_click(self, event):
        #print(self.lab)
        self.clicked = True
 
        
class Study:
    """Study class."""

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
        
        # creates results folder and add path to sample tbl
        f = mkdirs(os.path.join(self.raw_path, 'slides'))
        self.res_pth = f

        # scale parameters
        self.factor = 1.0
        if 'factor' in study:
            self.factor = study['factor']
        self.scale = tofloat(study['scale']/self.factor)
        self.units = study['units']

        # the size of quadrats and subquadrats
        aux = 10*np.ceil((study['binsiz']/self.scale)/10)
        self.binsiz = toint(np.rint(aux))
        
        # bandwidth size for convolutions is half the quadrat size
        self.supbw = toint(np.rint(self.binsiz/5))
        self.bw = toint(np.rint(self.binsiz/10))

        # cell types classes df 
        f = os.path.join(self.raw_path, study['raw_classes_table'])
        if not os.path.exists(f):
            print("ERROR: classes file " + f + " does not exist!")
            sys.exit()
        self.classes = pd.read_csv(f)
        self.classes['class'] = self.classes['class'].astype(str)
        
    def getcmap(self):
        
        import matplotlib.colors as colors
        
        cols = self.classes['class_color']
        cmap = colors.LinearSegmentedColormap.from_list('mycolormap',
                                                        cols,
                                                        len(cols))
        color_list = [colors.rgb2hex(cmap(i)[:3]) for i in range(cmap.N)]
        color_list = ['#ffffff'] + color_list
        cmap = colors.LinearSegmentedColormap.from_list('mycolormap', 
                                                        color_list, 
                                                        len(cols)+1)
        return(cmap)


class Slide:
    """Slide class."""

    def __init__(self, i, study):

        from myfunctions import mkdirs, toint

        # creates sample object
        self.tbl = study.samples.iloc[i].copy()  # table of parameters
        self.sid = self.tbl.sample_ID            # sample ID
        self.classes = study.classes             # cell classes
        
        # paths
        self.raw_path = study.raw_path
        

        # raw data files
        self.raw_cell_data_file = os.path.join(self.raw_path,
                                               self.tbl.coord_file)
        # roi mask images
        self.roi_file = os.path.join(self.raw_path,
                                     self.tbl.roi_file)
        self.scatter_file = os.path.join(self.raw_path,'slides',
                                         self.sid + '_landscape_points.png')
        
        # makes sure the path for roi file exists
        _ = mkdirs(os.path.dirname(self.roi_file))
        _ = mkdirs(os.path.dirname(self.scatter_file))
        
        # the size of quadrats and subquadrats
        self.binsiz = study.binsiz
        
        # bandwidth size for convolutions is half the quadrat size
        self.supbw = study.supbw
        self.bw = study.bw

        # scale parameters
        self.factor = study.factor
        self.scale = study.scale
        self.units = study.units

        # other sample attributes (to be filled later)
        self.cell_data = pd.DataFrame()  # dataframe of cell data (coordinates)
        self.imshape = []                # shape of accepted image
        
        # if ROI should be split 
        self.split_rois = self.tbl.split_rois
        
        # sample arrays
        self.roiarr = []                 # ROI array

        # total number of cells (after filtering)
        self.num_cells = toint(0)
        # list of roi sections
        self.roi_sections = []
        
        # colormap for cells
        self.cmap = study.getcmap()


    def setup_data(self, edge=0):
        """
        Load coordinates data, shift and convert values.

        Crop images to the corresponding convex hull:
        > edge : maximum size [pix] of edge around data extremes in rasters
        """
        from myfunctions import toint

        if not os.path.exists(self.raw_cell_data_file):
            print("ERROR: data file " + self.raw_cell_data_file +
                  " does not exist!")
            sys.exit()
        cxy = pd.read_csv(self.raw_cell_data_file)[['class', 'x', 'y']]
        cxy['class'] = cxy['class'].astype(str)
        cxy = cxy.loc[cxy['class'].isin(self.classes['class'])]
        
        # updates coordinae values by conversion factor (from pix to piy)
        cxy.x = toint(cxy.x*self.factor)
        cxy.y = toint(cxy.y*self.factor)

        # gets extreme pixel values
        xmin, xmax = toint(np.min(cxy.x)), toint(np.max(cxy.x))
        ymin, ymax = toint(np.min(cxy.y)), toint(np.max(cxy.y))

        imshape = [ymax + edge, xmax + edge]
        
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

        self.cell_data = cell_data.reset_index(drop=True)
        self.imshape = [toint(imshape[0]), toint(imshape[1])]
        self.num_cells = toint(len(self.cell_data))

        
    def roi_mask(self):
        """Generate ROI."""
        from myfunctions import kdeMask_rois

        # gets a mask for the region that has cells inside
        self.roiarr = kdeMask_rois(self.cell_data,
                                   self.imshape,
                                   self.bw,
                                   minsiz=100000,
                                   split=self.split_rois,
                                   cuda=False)
    
    def cell_array(self):
        
        cell_data = self.cell_data
        cellarr = np.zeros(self.imshape)
        
        # make scatter plot 
        for i, row in self.classes.iterrows():
            aux = cell_data.loc[cell_data['class'] == row['class']]
            if (len(aux) > 0):
                cellarr[aux.row, aux.col] = row.class_val
                
        return(cellarr)        


    def split_roi_mask(self):
        """ Create mask image for sample
        
        from myfunctions import landscapeScatter, plotEdges
        from skimage.measure import label
        
        cell_data = slide.cell_data
        classes = slide.classes.iloc[::-1]
        sid = slide.sid
        imshape = slide.imshape
        split_rois = slide.split_rois
        binsiz = slide.binsiz
        
        msk = label(slide.roiarr>0).astype('int64')
        roi_sections = np.unique(msk[slide.roiarr > 0]).tolist()
        
        labelswitch = pick_rois(cell_data, classes, 
                                sid, imshape, 
                                split_rois, binsiz,
                                msk, roi_sections)    
        
        """
        from skimage.measure import label
        
        msk = (self.roiarr>0).astype('int64')
        
        if (np.sum(self.roiarr)>0) and (self.split_rois):
            
            msk = label(self.roiarr>0).astype('int64')
            roi_sections = np.unique(msk[self.roiarr > 0]).tolist()
            
            labelswitch = pick_rois(self.cell_data, self.classes, 
                                    self.sid, self.imshape, 
                                    self.split_rois, self.binsiz,
                                    msk, roi_sections)   
        
            # reset mask labels
            for i, row in labelswitch.iterrows():
                msk[msk==row['from']]=row['to']
            
        #plt.imsave(self.roi_file, msk, vmin=0, vmax=np.max(msk), cmap='gray')
        np.savez_compressed(self.roi_file, roi=msk)
        
        return(msk)
        
    def plot_landscape_scatter(self, msk):
        """Plot Scatter Plots.

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
        x = (np.unique(grid[:, 1]))*self.scale
        y = (self.imshape[0] - (np.unique(grid[:, 0])))*self.scale

        fig, ax = plt.subplots(1, 1, figsize=(12, math.ceil(12/ar)),
                               facecolor='w', edgecolor='k')

        classes = self.classes.iloc[::-1]

        for i, row in classes.iterrows():
            aux = self.cell_data.loc[self.cell_data['class'] == row['class']]
            if (len(aux) > 0):
                landscapeScatter(ax, aux.x, aux.y,
                                 row.class_color, row.class_name,
                                 self.units, xedges, yedges,
                                 spoint=2, fontsiz=18, grid=False)
        ax.grid(which='major', linestyle='--',linewidth='0.3', color='black')
        ax.contour(x, y, 1*(msk==1), [.50], linewidths=2, colors='orangered')
        ax.contour(x, y, 1*(msk==2), [.50], linewidths=2, colors='royalblue')
        ax.set_xlim([np.min(xedges), np.max(xedges)])
        ax.set_ylim([np.min(yedges), np.max(yedges)])
        ax.set_title('Sample ID: ' + str(self.sid), fontsize=20, y=1.04)
        ax.legend(labels=classes.class_name,
                  loc='upper left', bbox_to_anchor=(1, 1),
                  markerscale=3, fontsize=16, facecolor='w', edgecolor='k')

        plt.savefig(self.scatter_file, bbox_inches='tight', dpi=300)
        plt.close()

# %% Private Functions

def getmskcmap(n, col1, col2):
    
    import matplotlib.colors as colors
    cmap = colors.LinearSegmentedColormap.from_list('mycolormap',
                                                    ['gold', 
                                                     'darkgreen'],
                                                    n-2)
    color_list = [colors.rgb2hex(cmap(i)[:3]) for i in range(cmap.N)]
    color_list = ['#ffffff', col1, col2] + color_list
    cmap = colors.LinearSegmentedColormap.from_list('mycolormap', 
                                                    color_list, 
                                                    n+1)
    return(cmap)


def ynPopUp(question):
    
    fig = plt.figure(figsize=(5, 2), facecolor='w', edgecolor='k')
    axq = plt.subplot2grid((2, 1), (0, 0), rowspan=1, colspan=2)
    axy = plt.subplot2grid((2, 2), (1, 0))
    axn = plt.subplot2grid((2, 2), (1, 1))
    
    axq.set_axis_off()
    axq.text(0.5, 0.5, question, 
             fontsize=14, 
             horizontalalignment='center', verticalalignment='center')
       
    yes_button= bttnwidget(axy, 'Yes')
    no_button = bttnwidget(axn, 'No')
    
    plt.show()
    
    fig.canvas.setFocus()

    # wait for a button to be pressed   
    wa = True
    while wa: 
        while plt.waitforbuttonpress(timeout=-1): pass
        wa = not yes_button.clicked and not no_button.clicked
    
    plt.close()
    del fig
    
    return yes_button.clicked


def getCellSize(cell_arr, r):
    """Calculate Cell Size.

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
    N = fftconv2d(cell_arr, circ, cuda=False)

    # the typical area of a cell
    # (given by maximun number of cells in any circle)
    acell = 0
    if (np.max(N) > 0):
        acell = np.sum(circ)/np.max(N)

    # returns radius of a typical cell
    return tofloat(round(np.sqrt(acell/np.pi), 4))


def xyShift(data, shape, ref, scale):
    """Shift coordinates and transforms into physical units.

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



def pick_rois(cell_data, classes, sid, imshape, split_rois, binsiz, 
              msk, roi_sections):
       """
       cell_data = slide.cell_data
       classes = slide.classes.iloc[::-1]
       sid = slide.sid
       imshape = slide.imshape
       split_rois = slide.split_rois
       binsiz = slide.binsiz
   
       """
       from myfunctions import landscapeScatter, plotEdges, toint
       global picked, smsk, roi_label, labelswitch
       
       #plt.get_backend()
       #mpl.use('macosx')
       mpl.use('Qt5Agg')
       mpl.rcParams['toolbar'] = 'none'
       
       picked = False
       
       # scale images for faster display: to 500 "points" (pixels) wide
       scale = 500/np.max(cell_data['col'])
       data = cell_data[['class', 'col', 'row']].copy()
       # reduce data: coarse grain and duplicate removal
       data['col'] = toint(data['col']*scale)
       data['row'] = toint(data['row']*scale)
       data = data.sample(frac=1.0).drop_duplicates(['row', 'col'])
       data = data.reset_index(drop=True)
       shape = [toint(np.ceil(imshape[0]*scale)), 
                toint(np.ceil(imshape[1]*scale))]
       siz = toint(binsiz*scale)
       
       # plot features        
       [ar, redges, cedges, _, _] = plotEdges(shape, 5*siz, scale)

       C, R = np.meshgrid(np.arange(0, shape[1], 1),
                          np.arange(0, shape[0], 1))
       grid = np.stack([R.ravel(), C.ravel()]).T
       
       # private variables for GUI and list of roi sections
       roi_labels = [0, 1, 2]
       roi_label = roi_labels[0]
       labelswitch = pd.DataFrame({'from': roi_sections,
                                  'to': roi_sections} )
       
       # sample down roiarr
       smsk = np.zeros(shape) 
       sroiarr = np.zeros(shape) 
       for row in np.arange(0, shape[0]):
           for col in np.arange(0, shape[1]):
               if smsk[row, col]==0:
                   r = toint(row/scale)
                   c = toint(col/scale)
                   smsk[row, col] = msk[r,c]
                   sroiarr[row, col] = 1*(msk[r,c]>0)
       omsk = smsk.copy()
       
       col1 = ['tomato', 'orangered'] 
       col2 = ['cornflowerblue', 'royalblue']  
       if (not split_rois):
           col2 = ['grey', 'grey']
           
       # colormap for initial msk
       mskcmap = getmskcmap(len(roi_sections), col1[1], col2[1])
       
       # create figure
       fig = plt.figure(figsize=(6*2, math.ceil(6/ar)))
       axa = plt.subplot2grid((5, 12), (0, 0), rowspan=5, colspan=5)
       axb = plt.subplot2grid((5, 12), (0, 5), rowspan=5, colspan=5)
       axt = plt.subplot2grid((5, 12), (0, 11))
       ax0 = plt.subplot2grid((5, 12), (1, 11))
       ax1 = plt.subplot2grid((5, 12), (2, 11))
       ax2 = plt.subplot2grid((5, 12), (3, 11))
       ax3 = plt.subplot2grid((5, 12), (4, 11))
       
       fig.suptitle('Sample ID: ' + str(sid) + 
                     '\n(ROIs split = ' +  str(split_rois) + ')', 
                     fontsize=16)
       # events
       def update_text(txtstr, color):
           # Get text object
           text_obj = axt.texts[0]
           # Set new text
           txtprs = dict(edgecolor='black',
                         facecolor= color, 
                         boxstyle='round,pad=1')
           text_obj.set_text(txtstr)
           text_obj.set_bbox(txtprs)
           # Redraw 
           axt.figure.canvas.draw()    
             
       def onclick(event):
           # if clicks inside plot
           if event.inaxes == axa or event.inaxes == axb :
               #print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
               #      ('double' if event.dblclick else 'single', event.button,
               #       event.x, event.y, event.xdata, event.ydata))
               c = toint(event.xdata)
               r = toint(event.ydata)
               global smsk, labelswitch
               aux = omsk[r,c]
               if (aux > 0):
                   smsk[omsk == aux] = roi_label
                   labelswitch.loc[labelswitch['from']==aux, 'to'] = roi_label
                   axb.cla()
                   plotmsk(axb)
       
       def rbutton_click(event, lab, txt, col):
           update_text(txt, col)
           global roi_label
           roi_label = lab
           
       def done_click(event):
           # if ynPopUp('Do you want to save this ROI filter?'):            
           #     global picked
           #     picked = True
           #     fig.canvas.setFocus()
           if np.max(labelswitch['to']) < 3:
               global picked
               picked = True
           else:
               print("WARNING: only 1 or 2 ROIs are allowed!!!")
               
       def do_axes(ax):
           from myfunctions import setTicklabelSpacing
           
           ax.set_facecolor('white')
           ax.set_xticks(cedges, minor=False)
           ax.set_yticks(redges, minor=False)
           ax.minorticks_on()
           ax.grid(which='major', linestyle='--', 
                   linewidth='0.2', color='black')
           #ax.set_xticklabels(cedges, rotation=90, fontsize=8)
           #ax.set_yticklabels(redges, fontsize=8)     
           ax.set_xlim(0, shape[1])
           ax.set_ylim(0, shape[0])
           setTicklabelSpacing(ax, 10)
           ax.contour(np.unique(grid[:, 1]), 
                      np.unique(grid[:, 0]), 
                      sroiarr, [.50], linewidths=1.5, colors='black')
       
       def plotmsk(ax):
           ax.imshow(smsk, cmap = mskcmap, vmin=0, vmax=np.max(roi_sections))
           do_axes(ax)
           ax.set_xticklabels(cedges, rotation=90, fontsize=8)
           ax.set_yticklabels([])
           ax.set_title('ROIs Mask', y=1.01)
           axb.figure.canvas.draw()
        
       # make scatter plot pannel
       for i, row in classes.iterrows():
           aux = data.loc[data['class'] == row['class']]
           if (len(aux) > 0):
               landscapeScatter(axa, aux.col, aux.row,
                                row.class_color, row.class_name,
                                '', cedges, redges,
                                spoint=2, fontsiz=18, grid=False)
       do_axes(axa)
       axa.set_xticklabels(cedges, rotation=90, fontsize=8)
       axa.set_yticklabels(redges, fontsize=8)     
       axa.set_title('Cell Data', y=1.01)
       axa.figure.canvas.draw()
       
       # plot mask pannel
       plotmsk(axb)
       axb.set_xticklabels(cedges, rotation=90, fontsize=8)
       axb.set_yticklabels([])
       
       # make text box
       txtstr = '< NO ROI >'
       txtprs = dict(edgecolor='black',
                     facecolor='white', 
                     boxstyle='round,pad=1')
       axt.set_axis_off()
       axt.text(0.5, 0.5, txtstr, fontsize=12, bbox=txtprs,
                horizontalalignment='center', verticalalignment='center')
       axt.set_title('Selection', fontsize=12, y=1.01)
       
       # Create button widgets
       axes_widget = AxesWidget(axb)
       roi0_button = Button(ax0, 'NO ROI')
       roi1_button = Button(ax1, 'ROI_1', color=col1[0], hovercolor=col1[1])
       roi2_button = Button(ax2, 'ROI_2', color=col2[0], hovercolor=col2[1])
       done_button = Button(ax3, 'DONE')
       
       # Connect events to functions
       axes_widget.connect_event('button_press_event', onclick)
       #roi0_button.on_clicked(roi0_click)
       roi0_button.on_clicked(lambda event: rbutton_click(event, 
                                                          roi_labels[0], 
                                                          '< NO ROI >', 
                                                          'white'))
       roi1_button.on_clicked(lambda event: rbutton_click(event, 
                                                          roi_labels[1], 
                                                          '< ROI_1 >', 
                                                          col1[0]))
       if (split_rois):
           roi2_button.on_clicked(lambda event: rbutton_click(event, 
                                                              roi_labels[2],
                                                              '< ROI_2 >',
                                                              col2[0]))
       done_button.on_clicked(done_click)
       
       # Start the event loop.
       plt.show()    
       
       while not picked: 
           #fig.canvas.setFocus()
           while plt.waitforbuttonpress(timeout=-1): pass
    
       plt.close()
      
       
       return labelswitch
   
    
# %% Main function

def main(args):
    """TLA Slide Plots Main."""
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
        argsfile = os.path.join(main_pth, 'pathAI.csv')

    else:
        # running from the CLI using the bash script
        # path to working directory (above /scripts)
        main_pth = os.getcwd()
        argsfile = os.path.join(main_pth, args.argsfile)


    print("==> The working directory is: " + main_pth)

    if not os.path.exists(argsfile):
        print("ERROR: The specified argument file does not exist!")
        sys.exit()

    # NOTE: only ONE line in the argument table will be used
    study = Study(pd.read_csv(argsfile).iloc[0], main_pth)

    # %% create slide objects and plots data

    for c in  range(len(study.samples)):
        
        slide = Slide(c, study)
        msg = "====> Slide [" + str(c + 1) + \
            "/" + str(len(study.samples.index)) + \
                "] :: Slide ID = " + slide.sid
        print(msg + " >>> generating ROI filter...")
       
        # setup data
        slide.setup_data(0)   
        slide.roi_mask()

        # plot scatter plot
        roimask = slide.split_roi_mask()
        
        # plot scatter plot
        slide.plot_landscape_scatter(roimask)

    # %% end
    return (0)


# %% Argument parser
if __name__ == "__main__":

    # Create the parser
    my_parser = ArgumentParser(prog="tla_slide_plots",
                               description="# plots scatter data" +
                               "module for Tumor Landscape Analysis #",
                               allow_abbrev=False)

    # Add the arguments
    my_parser.add_argument('argsfile',
                           metavar="argsfile",
                           type=str,
                           help="Argument table file (.csv) for study set")

    # passes arguments object to main
    main(my_parser.parse_args())
