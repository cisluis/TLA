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
from ast import literal_eval
from argparse import ArgumentParser

__version__  = "1.0.1"


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
      
      f= os.path.join(self.dat_pth, study['name'] + '_classes.csv')
      if not os.path.exists(f):
          print("ERROR: classes file " + f + " does not exist!")
          sys.exit()
      self.classes = pd.read_csv(f,
                            converters={'abundance_edges': literal_eval,
                                        'mixing_edges': literal_eval})

      f = os.path.join(self.dat_pth, study['name'] + '_samples.csv')
      if not os.path.exists(f):
          print("ERROR: samples file " + f + " does not exist!")
          sys.exit()
      self.samples = pd.read_csv(f)
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
      self.lme_stats = pd.DataFrame()
      self.coloc_stats = pd.DataFrame()
      self.nndist_stats = pd.DataFrame()
      self.rhfunc_stats = pd.DataFrame()
      self.gordG_stats = pd.DataFrame()
 
    
  def mergeTables(self): 
      
      # loops over samples in this study
      for index, sample in self.samples.iterrows():
            
            # SID for display
            sid = sample.sample_ID
            res_pth = os.path.join(self.dat_path, 'results', 'samples', sid) 
            sfs_pth = os.path.join(res_pth, 'space_factors')
            
            f = os.path.join(res_pth, sid + '_lme_tbl.csv')
            if not os.path.exists(f):
                print("ERROR: samples table file " + f + " does not exist!")
                sys.exit()
            self.lme_stats = pd.concat([self.lme_stats, 
                                        pd.read_csv(f)],
                                       ignore_index=True)
            
            # colocalization features
            f = os.path.join(sfs_pth, self.sid +'_coloc_stats.csv')
            if os.path.exists(f):
                self.coloc_stats = pd.concat([self.coloc_stats, 
                                              pd.read_csv(f)],
                                             ignore_index=True)
                
            # NN distance features
            f = os.path.join(sfs_pth, self.sid +'_nndist_stats.csv')
            if os.path.exists(f):
                self.nndist_stats = pd.concat([self.nndist_stats, 
                                               pd.read_csv(f)],
                                              ignore_index=True)
                
            # Ripley's H features
            f = os.path.join(sfs_pth, self.sid +'_rhfunc_stats.csv')
            if os.path.exists(f):
                self.rhfunc_stats = pd.concat([self.rhfunc_stats, 
                                               pd.read_csv(f)],
                                              ignore_index=True)
                
            # Getis-Ord features 
            f = os.path.join(sfs_pth, self.sid +'_georG_stats.csv')
            if os.path.exists(f):
                self.gordG_stats = pd.concat([self.gordG_stats, 
                                              pd.read_csv(f)],
                                             ignore_index=True)
            
      # saves study tables
      f = os.path.join(self.dat_path, self.name + '_lme_tbl.csv')      
      self.lme_stats.to_csv(f, index=False, header=True)      
            
      f = os.path.join(self.dat_path, self.name + '_coloc_tbl.csv')      
      self.coloc_stats.to_csv(f, index=False, header=True)  
      
      f = os.path.join(self.dat_path, self.name + '_nndist_tbl.csv')      
      self.nndist_stats.to_csv(f, index=False, header=True)  
      
      f = os.path.join(self.dat_path, self.name + '_rhfunc_tbl.csv')      
      self.rhfunc_stats.to_csv(f, index=False, header=True)  
      
      f = os.path.join(self.dat_path, self.name + '_georG_tbl.csv')      
      self.gordG_stats.to_csv(f, index=False, header=True)  
          
# %% Private Functions

    # %% Main function

def main(args):

# %% start

    # %% debug starts
    debug = False
    
    if debug:
        # running from the IDE
        # path of directory containing this script
        main_pth = os.path.dirname(os.getcwd())
        argsfile = os.path.join(main_pth, 'DCIS_252_set.csv')
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
    
    # %% Summary Steps
    
    # summarize stats in study tables
    study.mergeTables()

    # %% end
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
