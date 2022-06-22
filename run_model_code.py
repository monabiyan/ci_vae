#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 13:41:42 2022

@author: mnabian
"""

print("start of the code")

import ivae

import pandas as pd
df_XY=pd.read_csv("sc_counts_final_small.csv")
##############################################################   
labels1=df_XY['Y'].tolist()
labels2=df_XY['YY'].tolist()
df_XY=df_XY
df_XY['Y']=labels1
df_XY=df_XY.drop(columns=['YY'])
df_XY.shape
df_XY.head()
##############################################################   
##############################################################
model_init=True
model_tobe_trained=True



import sklearn
model_init=True
model_tobe_trained=True
model_file_address='./test_model_20220425.pt'
save_address="bb"
#sc_df_filtered2=sc_df_filtered2.drop(columns=['Unnamed: 0','Y','Y','Y'])


obj1=ivae.IVAE(df_XY=df_XY,
               reconst_coef=100000,
               latent_size=20,
               kl_coef=0.0001*512,
               classifier_coef=1000,
               test_ratio=1)


obj=obj1

##########
if model_init:
    obj.model_initialiaze()
##########
if model_tobe_trained:
    lr=1e-3
    print(lr)
    obj.model_training(epochs=200,learning_rate=lr)

    lr=1e-3
    print(lr)
    #obj.model_training(epochs=70,learning_rate=lr)

    lr=1e-3
    print(lr)
    #obj.model_training(epochs=200,learning_rate=lr)

    obj.model_save(address=save_address+".pt")
    obj.save_residuals(address=save_address+'_residuals.pkl')
    lr=1e-3
    print(lr)
    #obj.model_training(epochs=70,learning_rate=lr)

    lr=5e-4
    print(lr)
    obj.model_training(epochs=400,learning_rate=lr)

    obj.model_save(address=save_address+".pt")
    obj.save_residuals(address=save_address+'_residuals.pkl')

    lr=1e-5
    print(lr)
    obj.model_training(epochs=400,learning_rate=lr)

    lr=5e-6
    print(lr)
    obj.model_training(epochs=400,learning_rate=lr)

                
    ##########

print("running the neural network")
#run(obj1,save_address)
obj1.model_save(address=save_address+".pt")
obj1.save_residuals(address=save_address+'_residuals.pkl')



