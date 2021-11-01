## This code show a function/example how to do imperfect mdoel test to evaluate
## weighting method compared with unweighted results by RMSE and correlation
## suppose you have wu and wq already with the metric calculated by ensemble mean of each model

import numpy as np
import math
import sys
sys.path.append("c:/python")
from func_cal_weights import calc_weights

prd_series_w=[]
prd_series_e=[]
obs_series=[]

for i in range(len(model_names)):
    ## create one dimension arry for pseudo observation
    ## series which can compared with predicted mean series    
    obs_pro=pro[i]
    obs_series=np.append(obs_series,obs_pro)

    model_pro=delete(pro,i)
    model_nahi=delete(model_names,i)

    for j in range(len(obs_pro)):
        predict_w=0

        # when the jth member of model i as pseudo observation
        for k in range(len(model_pro)):
            weights[i,j,k]=calc_weights(wu[i,k],wq[i,k],model_nahi)

            ## the loop below give equal way to each ensemble of per model
            for kk in range(len(model_pro[k])):
                predict_w=model_pro[k][kk]*(1/np.size(model_pro[k]))*weights[i,j,k]+predict_w
                predict_e=model_pro[k][kk]*(1/np.size(model_pro[k]))*(1/np.size(model_nahi))+predict_e

        prd_series_w=np.append(prd_series_w,predict_w)
        prd_series_e=np.append(prd_series_e,predict_e)

## evaulation section by comparing predicted series using weighting
##  method and observation series
        
## first calculate correlation
cor_w=stats.pearsonr(obs_series,prd_series_w)
cor_e=stats.pearsonr(obs_series,prd_series_e)

## calculate RMSE between pseudo observation series and weighted
##mean or unweighted prediction

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


rmse_weight = rmse(np.array(obs_series), np.array(prd_series_w))
rmse_unweight = rmse(np.array(obs_series), np.array(prd_series_e)) 

#plot scatter using plt.scatter
