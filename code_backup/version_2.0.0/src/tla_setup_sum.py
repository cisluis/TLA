'''
    TLA setup sum:
    ########################

        This script reads parameters from a study set table (only 1st line)
        Aggregated statistics across samples is calculated, specifically 
        quadrat statistics necessary to define cohort-level LMEs and global
        space stats to be used in a general cohort analysis.

'''

# %% Imports

import os
import psutil
import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

__version__  = "2.0.0"


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
                 
        # reduces classes df to just the accepted types (i.e. `drop=False`)
        classes_file = os.path.join(self.raw_path, study['raw_classes_table'])
        if not os.path.exists(classes_file):
            print("ERROR: classes file " + classes_file + " does not exist!")
            sys.exit()
        classes = pd.read_csv(classes_file)
        classes.drop(classes.loc[classes['drop']].index, inplace=True)
        classes.drop(columns=['drop'], inplace=True)
        classes.reset_index(inplace=True, drop=True)
        classes['class'] = classes['class'].astype(str)
        self.classes = classes
  
        # creates samples table for output
        self.samples_out = pd.DataFrame()
        self.allstats_out = pd.DataFrame()
        self.allpops_out = pd.DataFrame()
      

    def readTables(self):
        
        # loops over samples in this study
        for index, sample in self.samples.iterrows():
            
            # SID for display
            sid = sample.sample_ID
            res_pth = os.path.join(self.dat_path, 'results', 'samples', sid) 
            
            f = os.path.join(res_pth, sid + '_samples.csv')
            if not os.path.exists(f):
                print("WARNING: sample: " + sid + " was droped from TLA!")
            else:        
                # read table for this sample
                aux = pd.read_csv(f)
                
                # if sample has actual data
                if (aux['num_cells'].values[0] > 0):
                    # adds sample to study table
                    self.samples_out = pd.concat([self.samples_out, aux], 
                                                 ignore_index=True)
                
                    f = os.path.join(res_pth, sid + '_samples_stats.csv')
                    if not os.path.exists(f):
                        print("ERROR: stats file " + f + " does not exist!")
                        sys.exit()
                    aux = pd.read_csv(f)
                    self.allstats_out = pd.concat([self.allstats_out, aux],
                                                  ignore_index=True)
                
                    f = os.path.join(res_pth, sid + '_quadrat_stats.csv')
                    if not os.path.exists(f):
                        print("ERROR: quadrat file " + f + " does not exist!")
                        sys.exit()
                    aux = pd.read_csv(f)
                    self.allpops_out = pd.concat([self.allpops_out, aux], 
                                                 ignore_index=True)
                else:
                    print("WARNING: sample: " + sid + " was dropped from TLA!")
                    
        # saves study tables
        self.samples_out = self.samples_out.astype({'num_cells': int})
        f = os.path.join(self.dat_path, self.name + '_samples.csv')      
        self.samples_out.to_csv(f, index=False, header=True)      
              
        f = os.path.join(self.dat_path, 'results', 
                         self.name + '_samples_stats.csv')
        self.allstats_out.to_csv(f, index=False, header=True)
        
        f = os.path.join(self.dat_path, 'results', 
                         self.name + '_quadrat_stats.csv')
        self.allpops_out.to_csv(f, index=False, header=True)
        
  
    def summarize(self):
        # plots distributions of quadrat stats
        self.classes = quadFigs(self.allpops_out, 
                                self.classes,
                                os.path.join(self.dat_path,
                                             self.name + '_quadrat_stats.png'))
        self.classes.to_csv(os.path.join(self.dat_path, 
                                         self.name + '_classes.csv'), 
                            index=False)
        
        # analyses info file
        from itertools import combinations, permutations, product
        
        cases = self.classes['class'].tolist()
        combs = list(combinations(self.classes['class'].tolist(), 2))
        perms = list(permutations(self.classes['class'].tolist(), 2))
        prods = list(product(self.classes['class'].tolist(), repeat=2))
      
        aux = pd.DataFrame(data = {'name': ['coloc',
                                            'nndist',
                                            'aefunc',
                                            'rhfunc',
                                            'geordG'], 
                                   'drop': [False, 
                                            False, 
                                            False, 
                                            False,
                                            False],
                                   'comps':[combs, 
                                            perms, 
                                            prods, 
                                            prods,
                                            cases]})
        aux.to_csv(os.path.join(self.dat_path,
                                self.name + '_analyses.csv'), 
                   index=False)
      
      
    def summarize_by_category(self, cat):
        
        if (cat in self.samples.columns):
    
            tbl = self.samples.copy()
            pop = self.allpops_out.copy()
            
            cats = tbl[cat].unique()
            
            for c in cats:
                ids = list(tbl.loc[tbl[cat] == c]['sample_ID'])
                aux = pop.loc[pop['sample_ID'].isin(ids)]
                
                f = self.name + '_quadrat_stats_' + cat + '_' + str(c) + '.png'
            
                # plots distributions of quadrat stats
                _ = quadFigs(aux,self.classes, os.path.join(self.dat_path,f))
        else:
           
           print('WARNING: Category <<'+ cat + '>> is not included ' + \
                 'in samples table...' )
   
        
# %% Private Functions

def memuse():
    m = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
    return(round(m, 2))

    
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
    n, bins, _ = ax.hist(z, bins, density=False, facecolor='blue', alpha=0.5)
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
        aux['class'] = aux['class'].astype(str)

        vals = quadrats[row['class']]
        vals = vals[vals>0]
        qats = []
        
        if (len(vals) > 0):
            qats = [0] + \
                np.around(np.nanquantile(vals, 
                                         [0.125, 0.875, 1.0]), 2).tolist()
            # qats = [0.0, np.mean(vals),
            #         np.mean(vals) + np.std(vals), np.max(vals)]
            
            n = quadDist(vals, 
                         row['class_name'],
                         'Cells per quadrat',
                         [0, np.nanmax(vals)], 
                         ax[0, i])
            if (n > maxfrq[0]):
                maxfrq[0] = n
                
            for xc in qats:
                ax[0, i].axvline(x=xc,
                                 color='red', linestyle='dashed', linewidth=1)
                
        aux['abundance_edges'] = [qats]
 
        vals = quadrats[row['class']+'_MH']
        vals = vals[~np.isnan(vals)]
        qats = []
        
        if (len(vals) > 0):
            qats = [0.0, 0.5, 0.9, 1.0]
            n = quadDist(vals, 
                         row['class_name'],
                         'Mixing index per quadrat',
                         [0.0, 1.0], 
                         ax[1, i])
            if (n > maxfrq[1]):
                maxfrq[1] = n
            
            for xc in qats:
                ax[1, i].axvline(x=xc,
                                 color='red', linestyle='dashed', linewidth=1)
                
        aux['mixing_edges'] = [qats]
    
        # concat to table
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
    # %% start, checks how the program was launched 
    debug = False
    try:
        args
    except NameError:
        #  if not running from the CLI, run in debug mode
        debug = True
    
    # tracks time and memory 
    start = time.time()
    mbuse = [np.nan] * 3

    if debug:
        # running from the IDE
        # path of directory containing this script
        main_pth = os.path.dirname(os.getcwd())
        argsfile = os.path.join(main_pth, 'DCIS.csv')
        cat = 'cohort'
    else:
        # running from the CLI using the bash script
        # path to working directory (above /scripts)
        main_pth = os.getcwd()
        argsfile = os.path.join(main_pth, args.argsfile) 
        cat = args.category

    print("==> The working directory is: " + main_pth)
    
    if not os.path.exists(argsfile):
        print("ERROR: The specified argument file does not exist!")
        sys.exit()
        
    # %% only the first study in the argument table will be used
    study = Study(pd.read_csv(argsfile).iloc[0], main_pth)
    
    mbuse[0] = memuse()
    if debug:
        print("==> STEP 1 - Memory used: " + str(mbuse[0]) + "[MB]")
    
    # %% summarize stats in study tables
    study.readTables()
    
    mbuse[1] = memuse()
    if debug:
        print("==> STEP 2 - Memory used: " + str(mbuse[1]) + "[MB]")
    
    # %% summarize abu-mix in study tables
    study.summarize()
    
    if (cat!=''): 
        study.summarize_by_category(cat)
        
    mbuse[2] = memuse()
    if debug:
        print("==> STEP 3 - Memory used: " + str(mbuse[2]) + "[MB]")
    
    # %% shows performance
    memmax = np.nanmax(mbuse)
    trun = time.strftime('%H:%M:%S', time.gmtime(time.time()-start))
    print('==> TLA-Setup-Sum finished. Time elapsed: ', trun, '[HH:MM:SS]')
    print("==> Max memory used: " + str(memmax) + "[MB]")
    

    # %% the end
    return(0)


# %% Argument parser
if __name__ == "__main__":

    # Create the parser
    my_parser = ArgumentParser(prog="tla_setup_sum",
                               description="# Sum Pre-processing module " + 
                               "for Tumor Landscape Analysis #",
                               allow_abbrev=False)

    # Add the arguments
    my_parser.add_argument('-v', '--version', 
                           action='version', 
                           version="%(prog)s " + __version__)
    
    my_parser.add_argument('argsfile',
                           metavar="argsfile",
                           type=str,
                           help="Argument table file (.csv) for study set")
 
    my_parser.add_argument('-c', '--category', 
                           nargs='?', default='', const='',
                           type=str,
                           help="Samples category for study set")
    
    # passes arguments object to main
    main(my_parser.parse_args())
