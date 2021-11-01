from numpy import *;#导入numpy的库函数
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import math
from scipy import stats
from sklearn import preprocessing
import sys
import seaborn as sns
from scipy.optimize import curve_fit
from statsmodels.stats.weightstats import DescrStatsW
import csv
from scipy.stats import norm
from   netCDF4 import Dataset as open_ncfile
from outfunctions import random_choice_with_sameid

## model name
namef=["ACCESS-CM2","ACCESS-ESM1-5","BCC-CSM2-MR","CAMS-CSM1-0","CanESM5","CESM2", "CESM2-WACCM",
       "CNRM-CM6-1", "CNRM-ESM2-1","CNRM-CM6-1-HR","EC-Earth3", "FGOALS-f3-L","FGOALS-g3",
       "GISS-E2-1-G","HadGEM3-GC31-LL","IPSL-CM6A-LR","KACE-1-0-G",
       "MIROC-ES2L","MIROC6","MPI-ESM1-2-HR","MPI-ESM1-2-LR",
       "MRI-ESM2-0","NESM3","NorESM2-LM","NorESM2-MM","UKESM1-0-LL"]

metric_name=['LTMI','MBLC','BCA','GT','BCS']
label=['Lower tropospheric mixing index (LTMI)','MBLC metric','Brient cloud albedo','GSAT trend (GT)','Brient cloud shallowness (BCS)']
color=['-k8','-g8','-m8','-c8','-b8']
## path of historical metric and projected GSAT
path_his='**'
path_fu = '**'
times=100000

##cor_metrics= np.arange(len(metric_name)*len(times)).reshape(len(metric_name),len(times))
##cor_metrics = cor_metrics.astype(np.float64)

## loop for selected metrics
for m in range(len(metric_name)):
    cor_t_random=[]
    p_t_random=[]

    ## ten thousands times of random selection on one ensemble per model
    for r in range(times):
        ##  define an historical metric array 't_all' and 'warm_all' for one random choice of one one
        ##    ensemble per model, the size of each array is the model number    
        t_all=[]
        warm_all=[]
        
        for i in range(len(namef)):
            ## read historical metric and future GSAT change with their runid for each model and loop for each model        
            cl_t=np.loadtxt(path_his+metric_name[m]+namef[i]+'.txt',delimiter='\t')
            fu=np.loadtxt(path_fu+metric_name[m]+namef[i]+'.txt',delimiter='\t')
            runid_his=np.loadtxt(path_his+metric_name[m]+namef[i]+'.txt',delimiter='\t')
            runid_fu=np.loadtxt(path_fu+metric_name[m]+namef[i]+'.txt',delimiter='\t')

            ## one random choice to slect a ensemble for the model looped, and use function 'random_chooice_with_sameid'
            ## to chose a random ensemble in his and future with the same id        
            select_fu_his=random_choice_with_sameid(cl_t,fu, runid_his,runid_fu)
            clt_t1=select_fu_his[0]
            fu1=select_fu_his[1]
            ## to create the array with multi models 
            t_all=np.append(t_all,clt_t1)
            warm_all=np.append(warm_all,fu1)
        ## calculate correlation cofficient between historical metric and projected GSAT change across multi models for one time
        cor_t=stats.pearsonr(t_all,warm_all)
        ## append correlation cofficient and corresponding p vlaue to get the array for all ten thousands times of choice
        cor_t_random=np.append(cor_t_random,cor_t[0])
        p_t_random=np.append(p_t_random,cor_t[1])
    ## plot 5-95 range of correlation cofficient for each metric   
    plt.plot([np.percentile(cor_t_random,5),np.percentile(cor_t_random,95)], [m,m], color[m],label=label[m])

plt.yticks([1,2,3, 4, 5], ['LTMI', 'BCS', 'BCA','MBLC','GT'])
plt.title('5-95% percentile of correlation cofficient for constraints and future \n warming(SSP5-8.5) in 2081-2100 (base period 1995-2014)',fontsize=11)

##plot the crtical correlation cofficinet with p vavlue equal 0.05 and consider the number of dependent models (refer to Appendix 1.3) 
plt.axvline(x=0.444, alpha=0.5, color='grey')
                                                            
plt.show()

