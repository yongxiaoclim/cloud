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
from scipy.stats import t

from test import find_nearest, model_with_ensembles,performance_weights,similarity_weights
from test import weights, equal_weights,d_distance
from test import cv_decide_sigma_d
from test import percent_05_95_mean

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]

min_range_v=0
max_range_v=25
min_year=1

trend_all=model_with_ensembles(min_range_v,max_range_v,min_year)

model_namef=["ACCESS-CM22081","ACCESS-ESM1-52081","bcc_change2081","CAMS-CSM1-02081","Cesm2_change2081", "Cesm2_waccm_change2081",
             "CNRM_CM62081", "CNRM_ESM22081","CNRM-CM6-1-HR2081","EC_earth32081","FGOALS-f3-L2081","FGOALS-g32081",
             "GISS-E2-1-G2081","HadGEM3-GC31-LL2081", "IPSL_change2081","KACE-1-0-G2081",
             "MIROC_es2l_chang2081","MIROC6_2081","MPI-ESM1-2-HR2081","MPI-ESM1-2-LR2081",
             "MRI_ESM2_change2081","NESM3_change2081","NorESM2-LM2081","NorESM2-MM2081","UKESM3_change2081"]

namehis=["ACCESS-CM2","ACCESS-ESM1-5","BCC-CSM2-MR","CAMS-CSM1-0","CESM2", "CESM2-WACCM",
       "CNRM-CM6-1", "CNRM-ESM2-1","CNRM-CM6-1-HR","EC-Earth3", "FGOALS-f3-L","FGOALS-g3",
       "GISS-E2-1-G","HadGEM3-GC31-LL","IPSL-CM6A-LR","KACE-1-0-G",
       "MIROC-ES2L","MIROC6","MPI-ESM1-2-HR","MPI-ESM1-2-LR",
       "MRI-ESM2-0","NESM3","NorESM2-LM","NorESM2-MM","UKESM1-0-LL"]

model_namehis=["ACCESS-CM2_trend","ACCESS-ESM1-5_trend","BCCSM2_trend",
            "CAMS-CSM1_trend","ce_trend","cesm2_waccm_trend","CNRM_CM6_trend",
            "CNRM-CM6-1-HR_trend","CNRMESM2_trend","EC_earth3","FGOALS-f3-L","FGOALS-f3-g3",
            "GISS-E2-1-G_trend","HadGEM3_trend","IPSL_trend","KACE-1-0-G_trend",
            "MIROC_es2l_trend","MI_trend","MPI-ESM1","MPI-ESM1-2-LR_trend",
            "MRIESM2_trend","NESM3_trend","NorESM2-LM_trend","NorESM2-MM_cl_trend_","uk_trend"]

letter=["A","B","C","D", "E","F", "G","H", "I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y"]

print(len(namehis))
un05_all=[]
un95_all=[]
unmean_all=[]
w05_all=[]
w95_all=[]
wmean_all=[]

##obs_zhai_pool=np.arange(-2, -1.6, 0.01)
obs_zhai_pool=np.arange(-1.7, -1.3, 0.01)
##obs_zhai_pool=np.arange(-1.5, -1.1, 0.01)
##obs_zhai_pool=np.random.normal(loc=-1.5, scale=0.19, size=200)
##obs_zhai_pool=np.arange(-1.5, -1.1, 0.01)
##obs_zhai_pool=np.arange(-1.61, -1.26, 0.01)
obs_swcre_pool=np.arange(-0.01, 0.01, 0.001)
obs_tas_pool=np.arange(0.018, 0.020, 0.001)
obs_sha_pool=np.arange(43, 48, 0.1)

length=[]

path_his='F:/cmip6/trend_for_fu/fu_same585/'

for i in range(size(model_namef)): ##第i个model与观测的距离
    trend_all[i]=np.loadtxt(path_his+model_namehis[i]+'.txt', delimiter='\t')

p5_c_all=[]
p95_c_all=[]
p5_u_all=[]
p95_u_all=[]
pmean_c_all=[]
pmean_u_all=[]

pb=[]
for r in range(100):
    projection=[]
    sw_all=[]
    sha_all=[]
    lc_all=[]
    zh_all=[]

    pro_r=[]
    sw_r=[]
    sha_r=[]
    lc_r=[]
    zh_r=[]
    tas_r=[]

    for i in range(size(model_namef)):
        pro=np.loadtxt('F:/study3/code/NA_tas_changes/NA_change_585/'+namehis[i]+'.txt',delimiter='\t')
        projection=np.append(projection,np.mean(pro))
        if size(pro)==1:
            pro_r=np.append(pro_r,pro)
        else:
            pro_r=np.append(pro_r,np.random.choice(pro))

        if size(trend_all[i])==1:
            tas_r=np.append(tas_r,trend_all[i])
        else:
            tas_r=np.append(tas_r,np.random.choice(trend_all[i]))

    ## read factor zhai
##        zhai=open_ncfile('F:/paper2/data/zhai/up/up/'+namehis[i]+'_cloud.nc')
        zhai=open_ncfile('F:/paper2/data/zhai/up/up/average_first/w_gt10/'+namehis[i]+'_cloud.nc')
##        zhai=open_ncfile('G:/paper2/data/zhai/up/up/average_first/'+namehis[i]+'_cloud.nc')
##        zh=zhai.variables['lcc_sst_g_ave']
        zh=zhai.variables['lcc_sst_g']
        zh_all=np.append(zh_all,np.mean(zh))
        zh_r=np.append(zh_r,np.random.choice(zh))

    ## read factor SWCRE
        swcre=open_ncfile('F:/paper2/data/shallow/up/up/area_first/as_published/0820/0821/'+namehis[i]+'_cloud.nc')
        sw=swcre.variables['index']
        sw_all=np.append(sw_all,np.mean(sw))
        sw_r=np.append(sw_r,np.random.choice(sw)*100)
    
    list2_r = [list(a) for a in zip(zh_r, sw_r)]
    list2_r = sm.add_constant(list2_r)
    model2_r = sm.OLS(pro_r, list2_r).fit()
    predictions2_r = model2_r.predict(list2_r)

    m1=np.random.choice(obs_zhai_pool)
    m2=np.random.choice(obs_sha_pool)
    Xobs=np.array([1,m1, m2])
    
##    y_hat=model2_r.params[0]+model2_r.params[1]*m1+model2_r.params[2]*m2
    predict_mean=model2_r.predict(Xobs)
    pmean_c_all=np.append(pmean_c_all,predict_mean)

    y_prd=model2_r.predict(list2_r)
    S=np.sqrt(np.sum(np.square(pro_r-y_prd)/(len(namehis)-2-1)))

    aa=np.dot(list2_r.transpose(),list2_r)
    aa_re=np.linalg.inv(aa)
    bb=np.dot(Xobs.transpose(),aa_re)
    Xobs1=np.array([1,m1, m2])
    cc=np.dot(bb,Xobs1)
    scale=S*np.sqrt(cc+1)
##    scale=np.square(S)*cc +np.square(S)

    df=22
##    x = np.linspace(t.ppf(0.000001, df),t.ppf(0.999999, df), 100)
##    a = np.linspace(t.ppf(0.000001, df)*scale+predict_mean,t.ppf(0.999999, df)*scale+predict_mean, 100)
    x = np.linspace(1.5,8.5, 100)
##    y1 = t.cdf(x, df, predict_mean, scale)
##    y2 = t.pdf(x, df, predict_mean, scale)

    samp = t.rvs(loc=predict_mean, scale=scale, df=df, size=150)
    param = t.fit(samp)
    cdf_fitted = t.cdf(x,loc=param[1],scale=param[2],df=param[0])
    pdf_fitted = t.pdf(x,loc=param[1],scale=param[2],df=param[0])
    pb.append(pdf_fitted)

    p5_u=np.percentile(pro_r,5)
    p95_u=np.percentile(pro_r,95)

    loc05=find_nearest(cdf_fitted,0.05)
    index05=np.argwhere(cdf_fitted==loc05)
    p5_c=x[index05[0][0]]
    
    loc95=find_nearest(cdf_fitted,0.95)
    index95=np.argwhere(cdf_fitted==loc95)
    p95_c=x[index95[0][0]]
    
    p5_c_all=np.append(p5_c_all,p5_c)
    p95_c_all=np.append(p95_c_all,p95_c)
    p5_u_all=np.append(p5_u,p5_u_all)
    p95_u_all=np.append(p95_u,p95_u_all)
    pmean_u_all=np.append(pmean_u_all,mean(pro_r))

pb_mean=np.arange(len(x)).reshape(len(x),1)
pb_max=np.arange(len(x)).reshape(len(x),1)
pb_min=np.arange(len(x)).reshape(len(x),1)
pb_mean = pb_mean.astype(np.float64)
pb_max = pb_max.astype(np.float64)
pb_min = pb_min.astype(np.float64)

for m in range(100):
    pb_tem=[]
    for n in range(100):
        pb_tem=np.append(pb_tem,pb[n][m])
    pb_mean[m]=mean(pb_tem)
    pb_max[m]=max(pb_tem)
    pb_min[m]=min(pb_tem)

    
##a1=sns.distplot(pro_r,bins=10,kde=False,fit=stats.t,color="white",fit_kws={'color': 'k','linestyle':'--'},label=' ')
ax = plt.subplot(111)
ax.plot(x, pb_mean,color="blue", alpha=0.6, label='t pdf')
ax.fill_between(x, [num for elem in pb_min for num in elem],[num for elem in pb_max for num in elem],color='blue',alpha=.3)

plt.xlim((1, 6.5))
plt.ylim((0, 0.9))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
##plt.hlines(0.9,mean(p5_c_all),mean(p95_c_all),color='red')
##plt.hlines(0.95,mean(p5_u_all),mean(p95_u_all),color='k')
  


np.savetxt('zs_x.txt', x, delimiter=',')    
np.savetxt('zs_pb_mean.txt', pb_mean, delimiter=',')
np.savetxt('zs_pb_min.txt', [num for elem in pb_min for num in elem], delimiter=',')
np.savetxt('zs_pb_max.txt', [num for elem in pb_max for num in elem], delimiter=',')

np.savetxt('zs_p5_c_all_norm.txt', p5_c_all, delimiter=',')
np.savetxt('zs_p95_c_all_norm.txt', p95_c_all, delimiter=',')
np.savetxt('zs_pmean_c_all_norm.txt', pmean_c_all, delimiter=',')

np.savetxt('zs_p5_u_all_norm.txt', p5_u_all, delimiter=',')
np.savetxt('zs_p95_u_all_norm.txt', p95_u_all, delimiter=',')
np.savetxt('zs_pmean_u_all_norm.txt', pmean_u_all, delimiter=',')

plt.show()
