'''
    Tumor Landscape Analysis (TLA):
    ##############################

        This script reads lines from a study set table
        Each line has parameters of a particular study

        This formating helps faster and more consistent running of the TLA


'''

# %% Imports
import os
import psutil
import sys
import time
import pandas as pd
#import numpy as np
from ast import literal_eval
from argparse import ArgumentParser

#from myfunctions import tofloat, toint

__version__  = "2.0.1"


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
      
      f= os.path.join(self.dat_pth, self.name + '_classes.csv')
      if not os.path.exists(f):
          print("ERROR: classes file " + f + " does not exist!")
          sys.exit()
      self.classes = pd.read_csv(f,
                            converters={'abundance_edges': literal_eval,
                                        'mixing_edges': literal_eval})
      self.classes['class'] = self.classes['class'].astype(str)

      f = os.path.join(self.dat_pth, self.name + '_samples.csv')
      if not os.path.exists(f):
          print("ERROR: samples file " + f + " does not exist!")
          sys.exit()
      self.samples = pd.read_csv(f)
      self.samples.fillna('', inplace=True)
      
      # # scale parameters
      # self.factor = tofloat(1.0)
      # if 'factor' in study:
      #     self.factor = study['factor']
      # self.scale = tofloat(study['scale']/self.factor)
      # self.units = study['units']
     
      # # the size of quadrats and subquadrats
      # aux = 10*np.ceil((study['binsiz']/self.scale)/10)
      # self.binsiz = toint(np.rint(aux))
      
      # # bandwidth size for convolutions is half the quadrat size
      # self.kernel = toint(np.rint(self.binsiz/5))
      # self.subkernel = toint(np.rint(self.binsiz/10))
      
      # analyses table
      self.analyses = pd.read_csv(os.path.join(self.dat_pth,
                                     self.name + '_analyses.csv'),
                        converters={'comps': literal_eval}) 
      
      # creates samples table for output
      self.lme_stats = pd.DataFrame()
      self.abu_stats = pd.DataFrame()
      self.clt_stats = pd.DataFrame()
      self.mix_stats = pd.DataFrame()
      self.coloc_stats = pd.DataFrame()
      self.nndist_stats = pd.DataFrame()
      self.rhfunc_stats = pd.DataFrame()
      #self.geordG_stats = pd.DataFrame()
      #self.geordG_local_stats = pd.DataFrame()
      self.hots_stats = pd.DataFrame()
      self.hots_local_stats = pd.DataFrame()
 
    
  def mergeTables(self): 
      
      from myfunctions import mkdirs
                  
      # loops over samples in this study
      for index, sample in self.samples.iterrows():
            
            sid = sample.sample_ID
            res_pth = os.path.join(self.dat_pth, 'results', 'samples', sid) 
            sfs_pth = os.path.join(res_pth, 'space_factors')
            
            f = os.path.join(res_pth, sid + '_lme_tbl.csv')
            if not os.path.exists(f):
                print("WARNING: sample: " + sid + " was dropped from TLA!")
            else: 
                
                self.lme_stats = pd.concat([self.lme_stats, 
                                            pd.read_csv(f)],
                                           ignore_index=True)
                
                # abu features
                f = os.path.join(sfs_pth, sid +'_abu_stats.csv')
                if os.path.exists(f):
                    aux = loadToWide(f, sid, 'abu')
                    self.abu_stats = pd.concat([self.abu_stats, aux],
                                                 ignore_index=True)
                    
                # clt features
                f = os.path.join(sfs_pth, sid +'_clt_stats.csv')
                if os.path.exists(f):
                    aux = loadToWide(f, sid, 'clt')
                    self.clt_stats = pd.concat([self.clt_stats, aux],
                                                 ignore_index=True)
                    
                # mix features
                f = os.path.join(sfs_pth, sid +'_mix_stats.csv')
                if os.path.exists(f):
                    aux = loadToWide(f, sid, 'mix')
                    self.mix_stats = pd.concat([self.mix_stats, aux],
                                                 ignore_index=True)
                
                # colocalization features
                f = os.path.join(sfs_pth, sid +'_coloc_stats.csv')
                if os.path.exists(f):
                    aux = loadToWide(f, sid, 'coloc')
                    self.coloc_stats = pd.concat([self.coloc_stats, aux],
                                                 ignore_index=True)
                    
                # NN distance features
                f = os.path.join(sfs_pth, sid +'_nndist_stats.csv')
                if os.path.exists(f):
                    aux = loadToWide(f, sid, 'nndist')
                    self.nndist_stats = pd.concat([self.nndist_stats, aux],
                                                  ignore_index=True)
                    
                # Ripley's H features
                f = os.path.join(sfs_pth, sid +'_rhfunc_stats.csv')
                if os.path.exists(f):
                    aux = loadToWide(f, sid, 'rhfunc')
                    self.rhfunc_stats = pd.concat([self.rhfunc_stats, aux],
                                                  ignore_index=True)
                    
                # Getis-Ord features 
                # f = os.path.join(sfs_pth, sid +'_geordG_stats.csv')
                # if os.path.exists(f):
                #     aux = loadToWide(f, sid, 'geordG')
                #     self.geordG_stats = pd.concat([self.geordG_stats, aux],
                #                                  ignore_index=True)
                # f = os.path.join(sfs_pth, sid +'_geordG_local_stats.csv')
                # if os.path.exists(f):
                #     aux = loadToWide(f, sid, 'geordG_local')
                #     self.geordG_local_stats = pd.concat([self.geordG_local_stats, aux],
                #                                  ignore_index=True)
                f = os.path.join(sfs_pth, sid +'_hots_stats.csv')
                if os.path.exists(f):
                    aux = loadToWide(f, sid, 'hots')
                    self.hots_stats = pd.concat([self.hots_stats, aux],
                                                 ignore_index=True)
                f = os.path.join(sfs_pth, sid +'_hots_stats.csv')
                if os.path.exists(f):
                    aux = loadToWide(f, sid, 'hots_local')
                    self.hots_local_stats = pd.concat([self.hots_local_stats, 
                                                       aux],
                                                 ignore_index=True)
            
            
      # saves study tables
      pth = mkdirs(os.path.join(self.dat_pth, 'results', 'lmes'))     
      f = os.path.join(pth, self.name + '_lme_tbl.csv')      
      self.lme_stats.to_csv(f, index=False, header=True)      
      
      pth = mkdirs(os.path.join(self.dat_pth, 'results', 'space_factors'))
           
      f = os.path.join(pth, self.name + '_abu_tbl.csv')      
      self.abu_stats.to_csv(f, index=False, header=True)  
      
      f = os.path.join(pth, self.name + '_clt_tbl.csv')      
      self.clt_stats.to_csv(f, index=False, header=True)  
      
      f = os.path.join(pth, self.name + '_mix_tbl.csv')      
      self.mix_stats.to_csv(f, index=False, header=True)  
      
      f = os.path.join(pth, self.name + '_coloc_tbl.csv')      
      self.coloc_stats.to_csv(f, index=False, header=True)  
      
      f = os.path.join(pth, self.name + '_nndist_tbl.csv')      
      self.nndist_stats.to_csv(f, index=False, header=True)  
      
      f = os.path.join(pth, self.name + '_rhfunc_tbl.csv')      
      self.rhfunc_stats.to_csv(f, index=False, header=True)  
      
      # f = os.path.join(pth, self.name + '_georG_tbl.csv')      
      # self.geordG_stats.to_csv(f, index=False, header=True)  
      
      # f = os.path.join(pth, self.name + '_georG_local_tbl.csv')      
      # self.geordG_local_stats_local.to_csv(f, index=False, header=True)  
      
      f = os.path.join(pth, self.name + '_hots_tbl.csv')      
      self.hots_stats.to_csv(f, index=False, header=True)  
      
      f = os.path.join(pth, self.name + '_hots_local_tbl.csv')      
      self.hots_local_stats.to_csv(f, index=False, header=True)  
      
          
# %% Private Functions

def memuse():
    m = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
    return(round(m, 2))

def loadToWide(f, sid, metric):
    
    covs =["patch_density", "largest_patch_index", 
           "edge_density", "landscape_shape_index", 
           "entropy", "shannon_diversity_index", "contagion"]
    cols = ['comp'] + covs
    
    tbl = pd.read_csv(f)[cols]
    tbl['comp'] = tbl['comp'].astype(str)
    tbl.index=[0]*len(tbl)
    dfout =  pd.DataFrame({'sample_ID': [sid]})
    
    for c in covs:
        aux = tbl[['comp', c]].copy()
        aux['comp'] = aux['comp'].str.replace('::','_')
        aux['cc'] = metric + "_" + aux['comp'].astype(str) + "_" + c
        auy = pd.pivot(aux, 
                       #index=None,
                       columns='cc', 
                       values=c)
        auy.insert(0, 'sample_ID', sid)
        dfout = dfout.merge(auy, how='left', on='sample_ID')
        
    return(dfout)
    

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
        
    # only the first study in the argument table will be used
    study = Study( pd.read_csv(argsfile).iloc[0], main_pth)
    
    # %% Summarize stats in study tables
    study.mergeTables()
    
    trun = time.strftime('%H:%M:%S', time.gmtime(time.time()-start))
    print('==> TLA-Sum finished. Time elapsed: ', trun, '[HH:MM:SS]')
    print("==> Max memory used: " + str(memuse()) + "[MB]")
    
    #%%
    return(0)        
        

# %% Argument parser
if __name__ == "__main__":

    # Create the parser
    my_parser = ArgumentParser(prog="tla_sum",
                               description="# Sum Processing module for " +
                               "Tumor Landscape Analysis #",
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
