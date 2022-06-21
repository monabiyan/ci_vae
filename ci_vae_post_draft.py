print("start of the code")

import ivae
import pandas as pd
import numpy as np
import sklearn
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
df_XY=pd.read_csv("sc_counts_final_small.csv")

##############################################################   
labels1=df_XY['Y'].tolist()
labels2=df_XY['YY'].tolist()
#df_XY=df_XY/100000
df_XY['Y']=labels1
df_XY=df_XY.drop(columns=['YY'])
df_XY.shape
df_XY.head()
##############################################################   
##############################################################
model_init=True
model_tobe_trained=False



model_init=True
model_file_address='./bb.pt'
save_address1="./"

obj1=ivae.IVAE(df_XY=df_XY,
               reconst_coef=100000,
               latent_size=10,
               kl_coef=0.0001*512,
               classifier_coef=1000,
               test_ratio=1)

obj1.model_initialiaze()

obj1.model_load(address="bb.pt")


with torch.no_grad():
    obj1.model.eval()


    obj1.load_residuals(address='bb_residuals.pkl')
    print("model loaded")
    
    
    obj1.generate_test_results()
    print("test data generated")

with torch.no_grad():
    obj1.model.eval()
    
    df_reconstructed = pd.DataFrame(obj1.x_last.cpu().detach().numpy(), columns=df_XY.drop(columns=['Y']).columns)
    df_latent=pd.DataFrame(obj1.zs.cpu().detach().numpy())
    print(obj1.zs)
    print(obj1.zs.size())
    
    obj1.model.eval()
    
    zs_tensor=obj1.zs.to(device)
    
    
    
    df_reconstructed_decoder=pd.DataFrame(obj1.model.decoder(zs_tensor).cpu().detach().numpy(), columns=df_XY.drop(columns=['Y']).columns)
    print(obj1.model.decoder)
    df_reconstructed.to_csv('df_reconstructed.csv')
    df_latent.to_csv('df_latent.csv')
    df_reconstructed_decoder.to_csv('df_reconstructed_decoder.csv')
    print("Full_data_reconstructed...")
    
    print("========df_reconstructed========")
    print(df_reconstructed)
    print("========df_reconstructed_decoder========")
    print(df_reconstructed_decoder)
    
    
    obj1.plot_residuals(init_index=110)
    print("regression analysis")
    obj1.regression_analysis(obj1.zs,df_XY['Y'])
    df_XY['Y']
