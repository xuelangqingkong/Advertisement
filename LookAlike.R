import pandas as pd
import sys, re, os
import numpy as np
from sklearn.decomposition import PCA

#--- 1. import data
l_dtype = ['CNT_TRIP', 'CNT_LPK', 'AMT_TXN',  'P12_AMT_RV_DERMA', 'P12_AMT_RV_LASER', 'P12_AMT_RV_OTHER',  'P12_AMT_RV_FILLER', 'P12_AMT_HF', 'P12_AMT_WP_NORMAL',  'P12_AMT_WP_LASER', 'P12_AMT_WP_CLINICAL', 'P12_AMT_WP_MIRACLE',  'P12_AMT_UV', 'P12_AMT_AP_BASIC', 'P12_AMT_AP_CELL',  'P12_AMT_AP_LUMI', 'P12_AMT_AP_OTHER', 'P12_AMT_YC',  'P12_AMT_CLEANSER', 'P12_AMT_ESSENCE', 'P12_AMT_MOISTURIZER',  'P12_AMT_TONER', 'P12_AMT_SPCL_CARE_EYE', 'P12_AMT_MU_FACE',  'P12_AMT_MU_EYE', 'P12_AMT_MU_LIP', 'P12_AMT_MU_NAIL',  'P12_AMT_MENS', 'P12_AMT_HAIR_COLOR', 'P12_AMT_HAIR_CARE',  'P12_CNT_RV_DERMA', 'P12_CNT_RV_LASER', 'P12_CNT_RV_OTHER',  'P12_CNT_RV_FILLER', 'P12_CNT_HF', 'P12_CNT_WP_NORMAL',  'P12_CNT_WP_LASER', 'P12_CNT_WP_CLINICAL', 'P12_CNT_WP_MIRACLE',  'P12_CNT_UV', 'P12_CNT_AP_BASIC', 'P12_CNT_AP_CELL',  'P12_CNT_AP_LUMI', 'P12_CNT_AP_OTHER', 'P12_CNT_YC',  'P12_CNT_CLEANSER', 'P12_CNT_ESSENCE', 'P12_CNT_MOISTURIZER',  'P12_CNT_TONER', 'P12_CNT_SPCL_CARE_EYE', 'P12_CNT_MU_FACE',  'P12_CNT_MU_EYE', 'P12_CNT_MU_LIP', 'P12_CNT_MU_NAIL',  'P12_CNT_MENS', 'P12_CNT_HAIR_COLOR', 'P12_CNT_HAIR_CARE',  'PCT_AMT_RV_DERMA', 'PCT_AMT_RV_LASER', 'PCT_AMT_RV_OTHER',  'PCT_AMT_RV_FILLER', 'PCT_AMT_HF', 'PCT_AMT_WP_NORMAL',  'PCT_AMT_WP_LASER', 'PCT_AMT_WP_CLINICAL', 'PCT_AMT_WP_MIRACLE',  'PCT_AMT_UV', 'PCT_AMT_AP_BASIC', 'PCT_AMT_AP_CELL',  'PCT_AMT_AP_LUMI', 'PCT_AMT_AP_OTHER', 'PCT_AMT_YC',  'PCT_AMT_CLEANSER', 'PCT_AMT_ESSENCE', 'PCT_AMT_MOISTURIZER',  'PCT_AMT_TONER', 'PCT_AMT_SPCL_CARE_EYE', 'PCT_AMT_MU_FACE',  'PCT_AMT_MU_EYE', 'PCT_AMT_MU_LIP', 'PCT_AMT_MU_NAIL',  'PCT_AMT_MENS', 'PCT_AMT_HAIR_COLOR', 'PCT_AMT_HAIR_CARE',  'PCT_CNT_RV_DERMA', 'PCT_CNT_RV_LASER', 'PCT_CNT_RV_OTHER',  'PCT_CNT_RV_FILLER', 'PCT_CNT_HF', 'PCT_CNT_WP_NORMAL',  'PCT_CNT_WP_LASER', 'PCT_CNT_WP_CLINICAL', 'PCT_CNT_WP_MIRACLE',  'PCT_CNT_UV', 'PCT_CNT_AP_BASIC', 'PCT_CNT_AP_CELL',  'PCT_CNT_AP_LUMI', 'PCT_CNT_AP_OTHER', 'PCT_CNT_YC',  'PCT_CNT_CLEANSER', 'PCT_CNT_ESSENCE', 'PCT_CNT_MOISTURIZER',  'PCT_CNT_TONER', 'PCT_CNT_SPCL_CARE_EYE', 'PCT_CNT_MU_FACE',  'PCT_CNT_MU_EYE', 'PCT_CNT_MU_LIP', 'PCT_CNT_MU_NAIL',  'PCT_CNT_MENS', 'PCT_CNT_HAIR_COLOR', 'PCT_CNT_HAIR_CARE']
d_dtype = dict()
for index in l_dtype:
    d_dtype[index] = np.float64

pop_tot = pd.read_csv("C:\\Users\\wizhan\\Documents\\Project\\NNS\\Input_NNS.csv", sep = ",", dtype = d_dtype)
pop_tot.dtypes
pop_mat = pop_tot.head(10000)


#--- 2. preparation

#-- cap the data set 

pop_app = pop_tot[10000:]
pctl_01 = pop_tot[l_dtype].quantile(0.01).to_dict()
pctl_99 = pop_tot[l_dtype].quantile(0.99).to_dict()
def capper(x,col):
    if x < pctl_01[col]:
        return pctl_01[col]
    elif x > pctl_99[col]:
        return pctl_99[col]
    else:
        return x

for col in l_dtype:
    pop_app[col].apply(lambda x: capper(x,col))
    
#-- get the mean and p25, p75 of each column
d_pctl_10 = pop_app[l_dtype].quantile(0.1).to_dict()
d_medean = pop_app[l_dtype].quantile(0.5).to_dict()
d_pctl_90 = pop_app[l_dtype].quantile(0.9).to_dict()

#-- standardize each column
for col in l_dtype:
    if d_pctl_90[col] - d_pctl_10[col] > 0:
        pop_app[col] = (pop_app[col]-d_medean[col])/(d_pctl_90[col] - d_pctl_10[col])
    else:
        pop_app[col] = 0

for col in l_dtype:
    if d_pctl_90[col] - d_pctl_10[col] > 0:
        pop_mat[col] = (pop_mat[col]-d_medean[col])/(d_pctl_90[col] - d_pctl_10[col])
    else:
        pop_mat[col] = 0

#--- 3. PCA application
pop_mat.columns.values
pca = PCA()
pca.fit(pop_mat[l_dtype])
sum = 0
col_sel = 0
for value in pca.explained_variance_ratio_:
    sum+=value;
    col_sel+=1
    if sum > 0.95:
        print(sum,value, col_sel)
        break
    

print(sum)
pop_mat_tf = pca.transform(pop_mat[l_dtype])
pop_mat_df = pd.DataFrame(pop_mat_tf)
pop_mat_df_sel = pop_mat_df.iloc[:,0:col_sel]
dpca_mat_medean =pop_mat_df_sel.quantile(0.5).to_dict()
dpca_mat_pctl_25 =pop_mat_df_sel.quantile(0.25).to_dict()
dpca_mat_pctl_75 =pop_mat_df_sel.quantile(0.75).to_dict()

dpca_mat_radius = dict()    
for ind in range(0,col_sel):
    dpca_mat_radius[ind] = dpca_mat_pctl_75[ind] - dpca_mat_pctl_25[ind]

for ind in range(0,col_sel):
    pop_mat_df_sel.iloc[:,ind] = pca.explained_variance_ratio_[ind]*(pop_mat_df_sel.iloc[:,ind] - dpca_mat_medean[ind])/dpca_mat_radius[ind]

pop_mat_df_sel['dist'] = pop_mat_df_sel.apply(lambda values: np.sqrt(np.sum([v**2 for v in values])), axis=1)

pop_app_tf = pca.transform(pop_app[l_dtype])
pop_app_df = pd.DataFrame(pop_app_tf)
pop_app_df_sel = pop_app_df.iloc[:,0:col_sel]

for ind in range(0,col_sel):
    pop_app_df_sel.iloc[:,ind] = pca.explained_variance_ratio_[ind]*(pop_app_df_sel.iloc[:,ind] - dpca_mat_medean[ind])/dpca_mat_radius[ind]

pop_app_df_sel['dist'] = pop_app_df_sel.apply(lambda values: np.sqrt(np.sum([v**2 for v in values])), axis=1)

df_pctl_app = pd.qcut(pop_app_df_sel['dist'],10, labels = range(0,10))

df_tag_app = pd.concat([pop_app.reset_index(drop=True),df_pctl_app],axis = 1)

df_tag_app.to_csv('C:\\Users\\wizhan\\Documents\\Project\\NNS\\df_tag_app.csv', sep = '|', index = False)
pop_mat.to_csv('C:\\Users\\wizhan\\Documents\\Project\\NNS\\pop_mat_df_sel.csv', sep = '|', index = False)

