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
decile_all=[]
decile_weights=[]

for i in range(len(model_names)):
    ## create one dimension arry for pseudo observation
    ## series which can compared with predicted mean series
    ##observation use all ensmeble dataset
    obs_pro=pro[i]
    obs_his=his[i]
    obs_series=np.append(obs_series,obs_pro)

    ##models to weight (historical) use ensemble mean dataset
    model_his=delete(his_mean,i)

    model_pro=delete(pro,i)
    model_nahi=delete(model_names,i)

    for j in range(len(obs_his)):
        
        # when the jth member of model i as pseudo observation
        #model pro are the projection value calculated by ensemble mean of
        # each model except the model including pseudo observation
        #here use all ensemble members that historical experiment provide,
        #whose number should be more than corresponding model in fiture experiments 
        
        for k in range(len(model_his)): 
            weights[j,k]=calc_weights(wu[k],wq[k],model_nahi)
            
            ## for models with more than one ensemble member, we give equal weight to each ensemble
            
            for l in range(len(model_pro[k])):
                #here gets the weights from all ensembles of historical experiment(number is j),
                #but models to weight using ensemble mean(number is k), for ensemble of each model will get equal weight(number is l)
                wei_allens[j,k,l]=weights[j,k]*(len(model_his[k]))
                
    ## here give example how we export decile of each pseudo observation 
    for m in range(len(obs_pro)):
        for n in range(len(obs_his)):
            index=np.argsort(model_pro)
            p = np.cumsum(wei_allens[n,:,:][index])
            data_sorted = np.sort(model_pro)
            
            ## find the pseudo observation locate in frequency based on weighted CDF distribution
            
            if size(obsf)==1:
                p0=find_nearest(data_sorted,obs_pro)
            else:
                p0=find_nearest(data_sorted,obs_pro[m])
            decile[m,j]=p[np.argwhere(data_sorted==p0)]

            ## since we want get relative frequency,
            ##  we give equal weight for each model and divide by model size for each model           
            decile_model[m,j]=(1/size(model_nahi))*(1/size(obs_pro))*(1/size(obs_his))
    decile_all=np.append(decile_all,decile)
    decile_weights=np.append(decile_weights,decile_model)     

## here give example how to sort each decile of pseudo observation into each quntile

np.histogram(decile_all,bins=10,weights=decile_weights,density=False);
his = plt.hist(decile_all, bins=10,weights=decile_weights)
 
plt.show()

##or draw a histogram plot with 5 bar 
#index=0
#v1=np.arange(5*1).reshape(5, 1)
#v1 = v1.astype(np.float64)
#for s in np.arange(0, 1, 0.2):
#    v1[index]=0
#    value1=[]
#    for i in range(len(decile_all)):
#        if(decile_all[i] >s  and decile_all[i]<=s+0.2):
#           value1=np.append(value1,decile_weights[i])
#    v1[index]=sum(value1)
#    index=index+1

#f#eq_20 =  [num for elem in v1 for num in elem]

################################################################
