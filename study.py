#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 14:20:18 2022

@author: mnabian
"""

import pandas as pd
import pickle

with open('/Users/mnabian/Desktop/draft/results_dict.pkl', 'rb') as handle:
    b = pickle.load(handle)

b.keys()

b['mean']['3']['ENSG00000100316']


signal=b['mean']['3']['ENSG00000100316']


import matplotlib.pyplot as plt
plt.plot(signal)

##########################################
import numpy as np
df=pd.read_csv("/Users/mnabian/Desktop/draft/sc_counts_final_small.csv")
df.shape
df2=pd.read_csv("/Users/mnabian/Desktop/draft/df_reconstructed.csv")
df2.shape
np.mean(np.abs(df-df2))
df3=pd.read_csv("/Users/mnabian/Desktop/draft/df_reconstructed_decoder.csv")
df3.shape
df2=df2.drop(columns=['Unnamed: 0'])
df3=df3.drop(columns=['Unnamed: 0'])
df_delta=np.abs(df-df2)
df_delta=np.abs(df2-df3)
import seaborn as sns;
sns.heatmap(df_delta)
df_delta.median().median()
df_delta.mean().mean()
df3.median().median()
df2.median().median()
df3.mean().mean()
df2.mean().mean()

df['ENSG00000008988'].head()
df2['ENSG00000008988'].head()
from pickle import load
scaler = load(open('/Users/mnabian/Documents/GitHub/ci_vae/scaler.pkl', 'rb'))
def de_normalize(df,has_labels=True):
    if (has_labels):
        y=df['Y']  
        yy=df['YY'] 
        df=df.drop(columns=['Y','YY'])
    df = pd.DataFrame(scaler.inverse_transform(df), columns=df.columns)
    if (has_labels):
        df['Y']=y
        df['YY']=yy
    return(df)

df = de_normalize(df,has_labels=True)
df2 = de_normalize(df2,has_labels=False)
df3=de_normalize(df3,has_labels=False)

(df-df2)
df_copy=df.drop(columns=['Y','YY'])
df_delta=np.abs(df_copy-df2)
df_delta.median().median()
df_delta.mean().mean()
import seaborn as sns
df_delta.shape
sns.heatmap(df_delta.values)
sns.heatmap(df.values, vmax=10000)
sns.heatmap(df2.values,vmax=10000)
df.loc[df['Y']==1,'ENSG00000100316']
###################################
###################################
######################
for k1 in b.keys():
    for k2 in b[k1].keys():
        dd=b[k1][k2]
        dd = pd.DataFrame(scaler.inverse_transform(dd),columns=dd.columns)
        b[k1][k2]=dd
######################
def find_error(df,b):   
    gene_list=list(paper_genes_name.values())
    cell_types= list(range(1,28))
    #gene_list=['ENSG00000100316']
    ll=[]
    for stat in ['mean','median']:
        for gene in gene_list:
            for cell_type in cell_types:
                for disease_state in [0,10]:
                    try:
                        if disease_state==0:
                            index=0
                        else:
                            index=-1
                        if stat=='median':
                            true_val=df.loc[(df['Y']==cell_type)&(df['YY']==disease_state),gene].median()
                            pred_val=b['med'][str(cell_type)][gene].tolist()[index]
                        elif stat=='mean':
                            true_val=df.loc[(df['Y']==cell_type)&(df['YY']==disease_state),gene].mean()
                            pred_val=b['mean'][str(cell_type)][gene].tolist()[index]
                        ll.append([gene,cell_type,disease_state,stat,true_val,pred_val,abs(true_val-pred_val),abs(true_val-pred_val)/(true_val+1)*100])
                        df_st=pd.DataFrame(ll,columns=['gene','celltype','disease_state','statistics','true_val','pred_val','diff','diff_ratio'])
                        
                    except Exception as e: 
                        #print(e)
                        continue
    return(df_st)
######################
df_st = find_error(df,b)
df_st.shape
df_st_high_quality=df_st.loc[(df_st['diff_ratio']<20)&(df_st['true_val']>1)&(df_st['statistics']=='mean')]
df_st_high_quality.shape
df_st_high_quality
plt.scatter(df_st['true_val'],df_st['pred_val'],s=1)
plt.xlim([0,1000])
plt.ylim([0,1000])
plt.show()
##################################################################
b['mean']['2']['ENSG00000271503']
df.loc[(df['Y']==2) & (df['YY']==10),'ENSG00000271503'].mean()

##################################################################
true_val=df.loc[df['Y']==4&(df['YY']==0),'ENSG00000271503'].median()
pred_val=b['med']['4']['ENSG00000271503'].tolist()[0]

    



g='ENSG00000100316'
df.loc[(df['Y']==1)&(df['YY']==10),[g]].hist()
df.loc[(df['Y']==1)&(df['YY']==5),[g]].hist()
df.loc[(df['Y']==1)&(df['YY']==0),[g]].hist()
df.iloc[3,:]


g=df.columns.tolist()[11]
g='ENSG00000204291'
g='ENSG00000143878'
g='ENSG00000142798'
g='ENSG00000106366'
g='ENSG00000134871'
g='ENSG00000112769'
g='ENSG00000187498'
g='ENSG00000113140' #paper


g='ENSG00000113140'
g='ENSG00000275302'
g='ENSG00000271503'
g='ENSG00000204291'
g = b['med']['1'].columns.tolist()[10]

g='ENSG00000019582'
g='ENSG00000100316'

index=1


for index in range(len(paper_genes_name)-1,0,-1):
    g=list(paper_genes_name.values())[index]
    g_id=list(paper_genes_name.keys())[index]
    for i in range(28):
        cell_type=str(i)
        signal=b['mean'][cell_type][g]
        print(max(signal))
        if max(signal)>100:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            #fig.suptitle(g_id+"__"+str(i))
            fig.suptitle(g_id+" ["+cell_types[i]+"]")
            
            #signal_n=[x / signal[0] for x in signal]
            #ax1.plot(signal)
            signal=b['med'][cell_type][g]
            #signal_n=[x / signal[0] for x in signal]
            ax1.plot(signal,'-bo',markersize=3)
            ax1.set_xlabel('Healthy to Cancer (Prediction)', fontsize=10)
            ax1.set_ylabel('Gene Expression', fontsize=10)
            ax1.axes.xaxis.set_ticks([])
            ax1.axes.yaxis.set_ticks([])
            #ax1.axis('off')
            #ax1.xticks([])
            #ax1.yticks([])
            #df.groupby('Y').count()
            
            a1=df.loc[(df['Y']==int(cell_type))&(df['YY']==0),[g]].median()
            a2=df.loc[(df['Y']==int(cell_type))&(df['YY']==5),[g]].median()
            a3=df.loc[(df['Y']==int(cell_type))&(df['YY']==10),[g]].median()
            b1=df.loc[(df['Y']==int(cell_type))&(df['YY']==0),[g]].mean()
            b2=df.loc[(df['Y']==int(cell_type))&(df['YY']==5),[g]].mean()
            b3=df.loc[(df['Y']==int(cell_type))&(df['YY']==10),[g]].mean()
            #ax2.plot([a1/a1,a2/a1,a3/a1])
            #ax2.plot([b1,b2,b3])
            ax2.plot([a1,a2,a3],'-gD')
            ax2.set_xlabel('Healthy to Cancer (Ground Truth)', fontsize=10)
            ax2.set_ylabel('Gene Expression', fontsize=10)
            ax2.axes.xaxis.set_ticks([])
            ax2.axes.yaxis.set_ticks([])
            #ax2.axis('off')
            #ax2.xticks([])
            #ax2.yticks([])


f1=df.loc[(df['Y']==1),[g]]
f2=b['std']['1']

df2 = pd.read_csv("/Users/mnabian/Documents/GitHub/sc_data/sc_counts_filtered_final_2.csv")
df
df['ENSG00000008988']
df2['ENSG00000008988']


paper_genes_name={'COL15A1':'ENSG00000204291',
                  'RHOB':'ENSG00000143878',
                  'HSPG2':'ENSG00000142798',
                  'SERPINE1':'ENSG00000106366',
                  'COL4A2':'ENSG00000134871',
                  'LAMA4':'ENSG00000112769',
                  'COL4A1':'ENSG00000187498',
                  'SPARC':'ENSG00000113140',
                  'PFN1':'ENSG00000108518',
                  'LGALS1':'ENSG00000100097',
                  'S100A6':'ENSG00000197956',
                  'PRKCDBP':'ENSG00000170955',
                  'KLF6':'ENSG00000067082',
                  'HTRA1':'ENSG00000166033',
                  'C4orf48':'ENSG00000243449',
                  'SH3BGRL3':'ENSG00000142669',
                  'CD74':'ENSG00000019582',
                  'CD320':'ENSG00000167775',
                  'FABP5':'ENSG00000164687',
                  'SOCS3':'ENSG00000184557',
                  'IGHA1':'ENSG00000211895',
                  'SDPR':'ENSG00000168497',
                  'IRF1':'ENSG00000125347',
                  'SPARCL1':'ENSG00000152583',
                  'JCHAIN':'ENSG00000132465',
                  'GADD45B':'ENSG00000099860',
                  'ADIRF':'ENSG00000148671',
                  'EGR1':'ENSG00000120738',
                  'KLF4':'ENSG00000136826',
                  'IGHA2':'ENSG00000211890',
                  'CLDN5':'ENSG00000184113',
                  'H3F3B':'ENSG00000132475',
                  'ITM2B':'ENSG00000136156',
                  'CEBPD':'ENSG00000221869',
                  'DNAJA1':'ENSG00000086061'}

cell_types=['consensus molecular subtype 1 epithelial cell',
 'myeloid cell',
 'regulatory T cell',
 'myofibroblast',
 'stromal cell',
 'consensus molecular subtype 3 epithelial cell',
 'CD4-positive, alpha-beta T cell',
 'smooth muscle cell of large intestine',
 'endothelial tip cell',
 'pericyte',
 'mast cell',
 'endothelial stalk cell',
 'T cell',
 'consensus molecular subtype 2 epithelial cell',
 'natural killer cell',
 'conventional dendritic cell',
 'B cell',
 'T follicular helper cell',
 'endothelial cell of lymphatic vessel',
 'glial cell',
 'IgA plasma cell',
 'CD8-positive, alpha-beta T cell',
 'epithelial cell of large intestine',
 'enterocyte of epithelium of large intestine',
 'T-helper 17 cell',
 'goblet cell',
 'plasma cell',
 'gamma-delta T cell',
 'brush cell of epithelium proper of large intestine']




cell_type=19
g1='ENSG00000100097'
g2='ENSG00000067082'


cell_type=11
g1='ENSG00000019582'
g2='ENSG00000100097'
g2='ENSG00000142798'



a1=float(df.loc[(df['Y']==int(cell_type))&(df['YY']==0),[g1]].median())
a2=float(df.loc[(df['Y']==int(cell_type))&(df['YY']==5),[g1]].median())
a3=float(df.loc[(df['Y']==int(cell_type))&(df['YY']==10),[g1]].median())
b1=float(df.loc[(df['Y']==int(cell_type))&(df['YY']==0),[g2]].median())
b2=float(df.loc[(df['Y']==int(cell_type))&(df['YY']==5),[g2]].median())
b3=float(df.loc[(df['Y']==int(cell_type))&(df['YY']==10),[g2]].median())
v1=b['med'][str(cell_type)][g1]/b['med'][str(cell_type)][g1][0]
v2=b['med'][str(cell_type)][g2]/b['med'][str(cell_type)][g2][0]
plt.plot(v2,v1)
plt.plot([b1/b1,b2/b1,b3/b1],[a1/a1,a2/a1,a3/a1])




