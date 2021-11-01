from numpy import *;#导入numpy的库函数
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import math
from scipy import stats
from sklearn import preprocessing
import sys
import scipy.io as sio
import statsmodels.api as sm # import statsmodels
from   netCDF4 import Dataset as open_ncfile
import seaborn as sns
from scipy.stats import norm
from matplotlib.lines import Line2D



def calc_prediction_with_lr_2v (m1,m2,pro):
    list2 = [list(a) for a in zip(m1,m2)]
    list2 = sm.add_constant(list2)
    m2 = sm.OLS(pro, list2).fit()
    pred = m2.predict(list2)

    return pred

def calc_F (m1,m2,pro,p,n):
    pred2=calc_prediction_with_lr_2v (m1,m2,pro)
    pred_m1=calc_prediction_with_lr (m1,pro)
    pred_m2=calc_prediction_with_lr (m2,pro)

    rss2  = np.sum(np.square(pred2-np.mean(pro)))
    rssm1  = np.sum(np.square(pred_m1-np.mean(pro)))
    rssm2  = np.sum(np.square(pred_m2-np.mean(pro)))

    sse= np.sum(np.square(pred2-pro))

    F_m1=(rssm1-rss2)/sse/(n-p-1)
    F_m2=(rssm1-rss2)/sse/(n-p-1)

    return F_m1
    return F_m2


def calc_F_with_dependence (m1,m2,pro,p,m):
    pred2=calc_prediction_with_lr_2v (m1,m2,pro)
    pred_m1=calc_prediction_with_lr (m1,pro)
    pred_m2=calc_prediction_with_lr (m2,pro)

    rss2  = np.sum(np.square(pred2-np.mean(pro)))
    rssm1  = np.sum(np.square(pred_m1-np.mean(pro)))
    rssm2  = np.sum(np.square(pred_m2-np.mean(pro)))

    sse= np.sum(np.square(pred2-pro))

    F_m1=(rssm1-rss2)/sse/(m-p-1)
    F_m2=(rssm1-rss2)/sse/(m-p-1)

    return F_m1
    return F_m2
    
    


