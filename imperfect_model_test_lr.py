## This code show a function/example how to do imperfect mdoel test to evaluate
## weighting method compared with unweighted results by RMSE and correlation
## suppose you have wu and wq already with the metric calculated by ensemble mean of each model

import numpy as np
import math
import sys
sys.path.append("c:/python")

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())




def imperfect_model_test_lr(predictions, targets):
    pre_series=[]
    obs_series=[]

    for i in range(len(model_names)):
        ## create one dimension arry for pseudo observation
        ## series which can compared with predicted mean series
        pseudo_obs=metric[i]
        obs_pro=pro[i]
        obs_series=np.append(obs_series,obs_pro)

        model_pro=delete(pro,i)
        model_metric=delete(metric,i)

        list_m = [list(a) for a in zip(model_metric)]
        list_m = sm.add_constant(list_m)
        m = sm.OLS(pro, list_m).fit()
        pre= m.predict(pseudo_obs)
        pre_series=np.append(pre_series,pre)

    ## evaulation section by comparing predicted series using weighting
    ##  method and observation series
        
    ## first calculate correlation
    cor=stats.pearsonr(obs_series,pre_series)

    ## calculate RMSE between pseudo observation series and weighted
    ##mean or unweighted prediction
    rmse_weight = rmse(obs_series, pre_series)


