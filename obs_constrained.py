## This code show a function/example how to do imperfect mdoel test to evaluate
## weighting method compared with unweighted results by RMSE and correlation
## suppose you have wu and wq already with the metric calculated by ensemble mean of each model

import numpy as np
import math
import sys
sys.path.append("c:/python")


def obs_cons_lr(pro_,edf,metric_,obs_m,p,s):

    select_fu_his=random_choice_with_sameid(metric_,pro_, runid_his,runid_fu)
    metric=select_fu_his[0]
    pro=select_fu_his[1]

    list_m = [list(a) for a in zip(metric)]
    list_m = sm.add_constant(list_m)
    m = sm.OLS(pro, list_m).fit()
    cons_best= m.predict(obs_m)

    y_prd=m.predict(list_m)
    S=np.sqrt(np.sum(np.square(pro-y_prd)/(len(pro)-p-s-1)))

    aa=np.dot(list_m.transpose(),list_m)
    aa_re=np.linalg.inv(aa)
    bb=np.dot(Xobs.transpose(),aa_re)
    cc=np.dot(bb,Xobs)
    scale=S*np.sqrt(cc+1)

    x = np.linspace(1,7, 100)
    samp = norm.rvs(loc=predict_mean, scale=scale, size=150)
    param = norm.fit(samp)
    pdf_fitted = norm.pdf(x,loc=param[0],scale=param[1])

    return pdf_fitted

    
        
    
    


