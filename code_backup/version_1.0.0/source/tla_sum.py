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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ast import literal_eval

from argparse import ArgumentParser


__version__  = "1.0.0"


# %% Private classes

class Study:
    
  def __init__(self, study, main_pth):
      
      # loads arguments for this study
      self.name = study['name']
      self.dat_pth = os.path.join(main_pth, study['data_path'])
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
 
    
# %% Private Functions


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


# %%%% Plotting functions

def plotViolins(tbl, grps, glab, signal, slab, fname):
    
    import seaborn as sns
    from statannot import add_stat_annotation
    from itertools import combinations
    
    aux = [str(x) for x in tbl[grps].to_list()]
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
                      order = grp,
                      size = 2)
    
    _ = add_stat_annotation(ax, x=aux, y=auy, order=grp,
                            box_pairs=list(combinations(grp, 2)),
                            test='Mann-Whitney', 
                            line_offset_to_box=0.2,
                            #text_format='full',
                            text_format='star',
                            loc='inside', verbose=0)    
    ax.set_xlabel(glab)
    ax.set_ylabel(slab)
    sns.set(font_scale = 2)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(fname, bbox_inches='tight', dpi=300)
    plt.close()
         
    return(0)

    # %% Main function

def main(args):

# %% start

    main_pth = os.getcwd()
    argsfile = os.path.join(main_pth, args.argsfile) 

    print("==> The working directory is: " + main_pth)
    
    if not os.path.exists(argsfile):
        print("ERROR: The specified argument file does not exist!")
        sys.exit()
        
    # only the first study in the argument table will be used
    study = Study( pd.read_csv(argsfile).iloc[0], main_pth)
    
    out_pth = os.path.join(study.dat_pth, 'results', 'LME_stats')
    if not os.path.exists(out_pth):
        os.makedirs(out_pth)
    
    # %% Summary Steps
    f = os.path.join(study.dat_pth, 'results',
                     study.name + '_samples_all_stats.csv')
    if not os.path.exists(f):
        print("ERROR: data file " + f + " does not exist!")
        sys.exit()
    else:    
        study.samples_out = pd.read_csv(f)
    
    if (len(study.samples_out) > 4):
        
        cols = ['num_cells', 
                'f_num_cells', 
                'l_num_cells', 
                't_num_cells',
                'num_lmes', 
                'landscape_area', 
                'adjacency_index', 
                'lme_contagion',
                'lme_Shannon', 
                'lme_shape_index', 
                'lme_Simpson', 
                'lme_num_patches',
                'lme_patch_density', 
                'lme_total_edge', 
                'lme_edge_density',
                'lme_largest_patch_index']
        
        labs = ['Toptal Number of Cells', 
                'Number of Fibroblasts',
                'Number of Lymphocytes',
                'Number of DCIS cells',
                'Number of LMEs', 
                'Landscape Area', 
                'LME Adjacency Index', 
                'LME Contagion',
                'LME Shannon Index', 
                'LME Shape Index', 
                'LME Simpson Index', 
                'LME Number of Patches', 
                'LME Patch Density', 
                'LME Total Edge', 
                'LME Edge Density', 
                'LME Largest Patch Index']
        
        for i, c in enumerate(cols):
            
            _= plotViolins(study.samples_out,
                           'cohort', "Cohort",
                           c, labs[i], 
                           os.path.join(out_pth,
                                        study.name + '_' + \
                                            cols[i] + '.png')) 
    else:
        print("More samples are required for cohort analysis...")   


    # %% end
    return(0)        
        

# %% Argument parser
if __name__ == "__main__":

    # Create the parser
    my_parser = ArgumentParser(prog="tla_sum",
                               description="### Processing-sum module for " +
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

    # passes arguments object to main
    main(my_parser.parse_args())
