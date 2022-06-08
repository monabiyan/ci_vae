print("start of the code")

import ivae
import pandas as pd
import numpy as np
import sklearn

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
               reconst_coef=1000000,
               latent_size=10,
               kl_coef=0.0001*512,
               classifier_coef=10,
               test_ratio=1)

obj1.model_initialiaze()

obj1.model_load(address="bb.pt")

obj1.load_residuals(address='bb_residuals.pkl')
print("model loaded")





obj1.generate_test_results()
print("test data generated")


df_reconstructed = pd.DataFrame(obj1.x_last.cpu().detach().numpy(), columns=df_XY.drop(columns=['Y']).columns)
df_reconstructed.to_csv('df_reconstructed.csv')
print("Full_data_reconstructed...")


obj1.plot_residuals(init_index=110)
print("regression analysis")
obj1.regression_analysis(obj1.zs,df_XY['Y'])
df_XY['Y']
print("calculate tsne_umap_pca")
tsne_mat,umap_mat,pca_mat,Y=obj1.calculate_lower_dimensions(obj1.zs,obj1.y_last,N=20000)
obj1.plot_lower_dimension(tsne_mat,Y,projection='3d',save_str='tsne3d.pdf')
obj1.plot_lower_dimension(tsne_mat,Y,projection='2d',save_str='tsne2d.pdf')
obj1.plot_lower_dimension(umap_mat,Y,projection='3d',save_str='umap3d.pdf')
obj1.plot_lower_dimension(umap_mat,Y,projection='2d',save_str='umap2d.pdf')
obj1.plot_lower_dimension(pca_mat,Y,projection='3d',save_str='pca3d.pdf')
obj1.plot_lower_dimension(pca_mat,Y,projection='2d',save_str='pca2d.pdf')

print("finished")

ant_df=pd.DataFrame({'Y':labels1,'YY':labels2,'index':list(range(0, len(labels1)))})

def mean_traversal(cell_type_id):
    import random
    healthy = list(ant_df.loc[ant_df['YY']==0].loc[ant_df['Y']==cell_type_id]['index'])
    cancer = list(ant_df.loc[ant_df['YY']==10].loc[ant_df['Y']==cell_type_id]['index'])
    #healthy = [i for i, x in enumerate(YY) if x == 0]
    #cancer = [i for i, x in enumerate(YY) if x == 10]
    print(len(healthy),len(cancer))
    h_max=min(50,len(healthy))
    c_max=min(50,len(cancer))
    traversal_step=50
    line_decoded=np.zeros(shape=(traversal_step, df_XY.shape[1]-1,h_max*c_max))
    index=0
    
    for h in random.sample(healthy, h_max):
        for c in random.sample(cancer, c_max):
            #print((sc_df_filtered2.iloc[c,:-1]-sc_df_filtered2.iloc[h,:-1]).mean())
            ss =obj1.traverse(number_of_images=traversal_step, start_id=h, end_id=c,model_name="supervised_")
            #print(ss.shape)
            #ss = ss/ss[0]
            #print(ss)
            #line_decoded = np.add(line_decoded,ss)
            line_decoded[:,:,index]=ss
            index=index+1
    #line_decoded = line_decoded/(50*50)
    line_decoded_med = np.median(line_decoded,axis=2)
    line_decoded_mean = np.mean(line_decoded,axis=2)
    line_decoded_std = np.std(line_decoded,axis=2)
    print(line_decoded_med)
    
    gg_med=pd.DataFrame(line_decoded_med)
    gg_med.columns=df_XY.columns[0:-1]
    
    gg_mean=pd.DataFrame(line_decoded_mean)
    gg_mean.columns=df_XY.columns[0:-1]
    
    gg_std=pd.DataFrame(line_decoded_std)
    gg_std.columns=df_XY.columns[0:-1]
    
    #gg= gg.div(gg.iloc[0])
    return(gg_med,gg_mean,gg_std)
    #return(line_decoded)
    
ff=dict()
ff['mean']=dict()
ff['med']=dict()
ff['std']=dict()

for i in range(len(set(ant_df['Y']))):
    print(i)
    ff['mean'][str(i)],ff['med'][str(i)],ff['std'][str(i)]=mean_traversal(i)

import pickle
with open('results_dict.pkl', 'wb') as f:
    pickle.dump(ff, f)
        
with open('results_dict.pkl', 'rb') as f:
    ff = pickle.load(f)
