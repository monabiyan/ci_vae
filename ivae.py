#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 14:26:21 2021

@author: mnabian
"""
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.autograd import Variable
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch
import random
random.seed(1234)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MyDataset(Dataset):
    def __init__(self,df,y_label=["Y"] ,mode = 'train'):
          self.mode = mode    
          if self.mode == 'train':
              self.oup = df.loc[:, y_label].values
              self.inp  = df.drop(columns=y_label)
              self.x_features=self.inp.columns.tolist()
              self.inp = self.inp.values      # df.values == Return a Numpy representation of the DataFrame
          else:
              self.inp = df.values
              self.x_features = self.inp.columns.tolist()
    def __len__(self):
        return (self.inp).shape[0]
    def __dim__(self):
        return (self.inp).shape[1]
    def __getitem__(self, idx):
        if self.mode == 'train':
            inpt  = torch.Tensor(self.inp[idx])
            oupt  = torch.Tensor(self.oup[idx])
            return (inpt,oupt)
        else:
            inpt = torch.Tensor(self.inp[idx])
            return inpt

class IVAE_ARCH(nn.Module):
    def __init__(self,input_size,n_classes,latent_size,dropout_rate=0.10):
        super().__init__()
        self.latent_size=latent_size
        self.dropout_rate = dropout_rate
        self.input_size=input_size
        second_layer_size = int((self.input_size+latent_size ** 2)/2)

        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, second_layer_size),
            nn.ReLU(),
            nn.BatchNorm1d(second_layer_size),
            nn.Dropout(p=dropout_rate),
            ##################
            nn.Linear(second_layer_size, latent_size ** 2),
            nn.ReLU(),
            nn.BatchNorm1d(latent_size ** 2),
            nn.Dropout(p=dropout_rate),
            ##################
            nn.Linear(latent_size ** 2, latent_size ** 2),
            nn.ReLU(),
            nn.BatchNorm1d(latent_size ** 2),
            nn.Dropout(p=dropout_rate),
            ##################
            nn.Linear(latent_size ** 2, latent_size ** 2),
            nn.ReLU(),
            nn.BatchNorm1d(latent_size ** 2),
            nn.Dropout(p=dropout_rate),
            ##################
            nn.Linear(latent_size ** 2, latent_size ** 2),
            nn.ReLU(),
            nn.BatchNorm1d(latent_size ** 2),
            nn.Dropout(p=dropout_rate),
            ##################
            nn.Linear(latent_size ** 2, latent_size ** 2),
            nn.ReLU(),
            nn.BatchNorm1d(latent_size ** 2),
            nn.Dropout(p=dropout_rate),
            ##################
            nn.Linear(latent_size ** 2, latent_size ** 2),
            nn.ReLU(),
            nn.BatchNorm1d(latent_size ** 2),
            nn.Dropout(p=dropout_rate),
            ##################
            nn.Linear(latent_size ** 2, latent_size * 2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_size, latent_size ** 2),
            nn.ReLU(),
            nn.BatchNorm1d(latent_size ** 2),
            nn.Dropout(p=dropout_rate),
            ##################
            nn.Linear(latent_size ** 2, latent_size ** 2),
            nn.ReLU(),
            nn.BatchNorm1d(latent_size ** 2),
            nn.Dropout(p=dropout_rate),
            ##################
            nn.Linear(latent_size ** 2, latent_size ** 2),
            nn.ReLU(),
            nn.BatchNorm1d(latent_size ** 2),
            nn.Dropout(p=dropout_rate),
            ##################
            nn.Linear(latent_size ** 2, latent_size ** 2),
            nn.ReLU(),
            nn.BatchNorm1d(latent_size ** 2),
            nn.Dropout(p=dropout_rate),
            ##################
            nn.Linear(latent_size ** 2, latent_size ** 2),
            nn.ReLU(),
            nn.BatchNorm1d(latent_size ** 2),
            nn.Dropout(p=dropout_rate),
            ##################
            nn.Linear(latent_size ** 2, latent_size ** 2),
            nn.ReLU(),
            nn.BatchNorm1d(latent_size ** 2),
            nn.Dropout(p=dropout_rate),
            ##################
            nn.Linear(latent_size ** 2, second_layer_size),
            nn.Sigmoid(),
            #nn.BatchNorm1d(1000),
            nn.Dropout(p=dropout_rate),
            nn.Linear(second_layer_size , self.input_size)
        )
        
        self.classifier = nn.Sequential (
            nn.Linear(latent_size, n_classes),
            #nn.BatchNorm1d(10),
            nn.Dropout(p = 0.80)
            #nn.Softmax()
        )
    def reparameterise(self, mu, logvar):
        #if self.training:
        if True:
            std = torch.exp(logvar / 2)+0.0000001
            q = torch.distributions.Normal(mu, std)
            z = q.rsample()
            #std = logvar.mul(0.5).exp_()
            #eps = std.data.new(std.size()).normal_()
            return z
        else:
            return mu

    def encode(self, x):
      mu_logvar = self.encoder(x.view(-1, self.input_size)).view(-1, 2, self.latent_size)
      mu = mu_logvar[:, 0, :]
      logvar = mu_logvar[:, 1, :]
      return mu, logvar

    def decode(self, z):
      return self.decoder(z)

    def sample(self, n_samples):
      z = torch.randn((n_samples, self.latent_size)).to(device)
      return self.decode(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        y_hat=self.classifier(z)
        x_hat = self.decode(z)
        return x_hat,y_hat, mu, logvar, z
    
    def decoding_from_latent(self, mu, logvar):
        z = self.reparameterise(mu, logvar)
        x_hat = self.decode(z)
        return x_hat

    
    

class IVAE(MyDataset,IVAE_ARCH):
#############################################################

    def __init__(self,df_XY,latent_size=20,reconst_coef=100000,kl_coef=0.001*512,classifier_coef=1000,test_ratio=0.2):
        ##########
        self.reconst_coef = reconst_coef
        self.kl_coef = kl_coef
        self.classifier_coef = classifier_coef
        ##########
        self.df_XY=df_XY
        #self.df_XY = self.MNIST_data()
        ##########
        self.latent_size=latent_size
        #obj.organize_data(df_XY)
        self.input_size = self.df_XY.shape[1]-1

        IVAE_ARCH.__init__(self,
            input_size=self.input_size,
            n_classes=len(set(list(self.df_XY['Y']))),
            latent_size=self.latent_size)
        
        MyDataset.__init__(self,df=self.df_XY)
        self.organize_data(test_ratio)
#############################################################
    def model_initialiaze(self):
        self.model=IVAE_ARCH(input_size=self.input_size,
            n_classes=len(set(list(self.df_XY['Y']))),
            latent_size=self.latent_size
            ).to(device)
        self.train_tracker=[]
        self.test_tracker=[]
        self.test_BCE_tracker=[]
        self.test_KLD_tracker=[]
        self.test_CEP_tracker=[]
#############################################################        
    def model_save(self,address):
        torch.save(self.model.state_dict(),address)
#############################################################   
    def model_load(self,address):
        random.seed(1234)
        self.model_initialiaze()
        self.model.load_state_dict(torch.load(address))
#############################################################
    def save_residuals(self,address='residuals.pkl'):
        import pickle
        residuals={'train_tracker':self.train_tracker,
           'test_BCE_tracker':self.test_BCE_tracker,
           'test_KLD_tracker':self.test_KLD_tracker,
           'test_CEP_tracker':self.test_CEP_tracker,
           'test_tracker':self.test_tracker}
        with open(address, 'wb') as f:
            pickle.dump(residuals, f)
#############################################################
    def load_residuals(self,address='residuals.pkl'):
        import pickle
        with open(address, 'rb') as f:
            residuals = pickle.load(f)
            self.train_tracker = residuals['train_tracker']
            self.test_BCE_tracker = residuals['test_BCE_tracker']
            self.test_KLD_tracker = residuals['test_KLD_tracker']
            self.test_CEP_tracker = residuals['test_CEP_tracker']
            self.test_tracker = residuals['test_tracker']
#############################################################   
    def visualize_model_architecture(self):
        pass
############################################################# 
    def plot_residuals(self,init_index=0,save_fig_address="./residuals.pdf"):
        import matplotlib.pyplot as plt
        plt.plot(self.train_tracker[init_index:], label='Training Total loss')
        plt.plot(self.test_tracker[init_index:], label='Test Total loss')
        plt.plot(self.test_BCE_tracker[init_index:], label='Test BCE loss')
        plt.plot(self.test_KLD_tracker[init_index:], label='Test KLD loss')
        plt.plot(self.test_CEP_tracker[init_index:], label='Test CEP loss')
        plt.legend()
        plt.show()
        plt.savefig(save_fig_address)
        
#############################################################        
    def pipeline(self,
                 model_init=True,
                 model_tobe_trained=True,
                 epochs=1000,
                 learning_rate= 1e-4,
                 model_file_address='./test_model.pt'):
        # Training the model
        self.df_XY = self.MNIST_data()
        self.organize_data(self.df_XY)
        self.input_size = self.df_XY.shape[1]-1
        
        if model_init:
            self.model_initialiaze(input_size=self.input_size)
        if model_tobe_trained:
            self.model_training(epochs,learning_rate)
        self.model_save(address=model_file_address)
        self.model_load(address=model_file_address)
        self.plot_residuals()
#############################################################
    def model_training(self,epochs,learning_rate):
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=learning_rate,weight_decay=2e-5)
        iteration_no = 0
        loss_scale_show = 1
        #codes = dict(μ=list(), logσ2=list(), y=list(), x=list())
        for epoch in range(1, epochs+1):
            iteration_no = iteration_no+1
            # train for one epocha
            self.train_total_loss = self.train(self.model)
            self.test_BCE_loss, self.test_KLD_loss, self.test_CEP_loss, self.test_total_loss, self.means, self.logvars, self.labels, self.images,z = self.test(self.model)
            #self.miu_last = torch.cat(self.means)
            #self.var_last = torch.cat(self.logvars)
            #self.y_last = torch.cat(self.labels)
            #self.x_last = torch.cat(self.images)
        
            train_total_loss_scaled = self.train_total_loss*loss_scale_show/ len(self.trainloader.dataset)
            test_total_loss_scaled = self.test_total_loss*loss_scale_show/ len(self.testloader.dataset)
            test_BCE_loss_scaled = self.test_BCE_loss*loss_scale_show/ len(self.testloader.dataset)
            test_KLD_loss_scaled = self.test_KLD_loss*loss_scale_show/ len(self.testloader.dataset)
            test_CEP_loss_scaled = self.test_CEP_loss*loss_scale_show/ len(self.testloader.dataset)
            
        
            self.train_tracker.append(train_total_loss_scaled)
            self.test_tracker.append(test_total_loss_scaled)
            self.test_BCE_tracker.append(test_BCE_loss_scaled)
            self.test_KLD_tracker.append(test_KLD_loss_scaled)
            self.test_CEP_tracker.append(test_CEP_loss_scaled)
            # print the test loss for the epoch
            print(f'====> Epoch: {iteration_no} total_train_loss: {train_total_loss_scaled:.6f} Total_test_loss: {test_total_loss_scaled:.6f} Total_BCE_test_loss: {test_BCE_loss_scaled:.6f} Total_KLD_test_loss: {test_KLD_loss_scaled:.6f} Total_CEP_test_loss: {test_CEP_loss_scaled:.6f}')
          
#############################################################   
    def MNIST_data(self):
        from keras.datasets import mnist
        (train_X, train_y), (test_X, test_y) = mnist.load_data()
        train_X = np.reshape(train_X, (60000,28*28))
        test_X = np.reshape(test_X, (10000,28*28))
        X = np.concatenate((train_X,test_X),axis=0)
        X = X/255
        Y = np.concatenate((train_y,test_y),axis=0)
        df_X=pd.DataFrame(X)
        df_Y=pd.DataFrame(Y)
        df_Y.columns=["Y"]
        df_XY=pd.concat([df_X,df_Y],axis=1)
        return(df_XY)
#############################################################    
    def organize_data(self,test_ratio):
        from sklearn.model_selection import train_test_split
        if (test_ratio==1):
            df_XY_train = self.df_XY
            df_XY_test = self.df_XY
        else:   
            self.df_XY = self.df_XY.sample(frac = 1,random_state=1234)
            
            df_XY_train, df_XY_test = train_test_split(self.df_XY, test_size=test_ratio, random_state=1234)
        
        data_train = MyDataset(df=df_XY_train,y_label=["Y"])
        data_test = MyDataset(df=df_XY_test,y_label=["Y"])
        import random
        random.seed(1234)
        self.BATCH_SIZE=512
        trainloader = torch.utils.data.DataLoader(dataset = data_train,
                                                   batch_size = self.BATCH_SIZE,
                                                  shuffle=False)
        testloader = torch.utils.data.DataLoader(dataset = data_test,
                                                  batch_size = self.BATCH_SIZE,
                                                 shuffle=False)
        self.trainloader = trainloader
        self.testloader = testloader
        # Reconstruction + KL divergence losses summed over all elements and batch
    #############################################################
    def loss_function(self,x_hat, x,y_hat,y, mu, logvar):
        # reconstruction loss (pushing the points apart)
        #BCE = nn.functional.binary_cross_entropy(x_hat, x.view(-1, input_size), reduction='sum')
        #mse_loss = nn.MSELoss()
        mae_loss = nn.L1Loss()
        crs_entrpy = nn.CrossEntropyLoss()
        
        #BCE = nn.functional.binary_cross_entropy_with_logits(x_hat, x.view(-1, self.input_size))
        BCE = mae_loss(x_hat, x.view(-1, self.input_size))
        #BCE = F.binary_cross_entropy(x_hat, x.view(-1, self.input_size))
        CEP = crs_entrpy(y_hat.to(device),y.to(device))
        # KL divergence loss (the relative entropy between two distributions a multivariate gaussian and a normal)
        # (enforce a radius of 1 in each direction + pushing the means towards zero)
        #KLD = 0.5 * torch.sum(logvar.exp() - logvar - 1 + mu.pow(2))
        #KLD=torch.mean(torch.abs(KLD))
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD = KLD/self.BATCH_SIZE
        #KLD2= torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)
        #print(KLD)

        #BCE = BCE*reconst_coef
        #KLD = KLD * kl_coef
        #CEP = CEP*classifier_coef
        
    
        #Tot_Loss = BCE + KLD
        
        Tot_Loss = BCE * self.reconst_coef + KLD * self.kl_coef + CEP * self.classifier_coef
        return BCE * self.reconst_coef, KLD * self.kl_coef, CEP * self.classifier_coef, Tot_Loss  # we can use a beta parameter here (BCE + beta * KLD)
    
    #############################################################   
    # performs one epoch of training and returns the training loss for this epoch
    def train(self,model):
      model.train()
      train_loss = 0
      for x, y in self.trainloader:
        x = x.to(device)
        y = y.to(device)
        y=y.to(device)
        y=torch.tensor(torch.reshape(y, (-1,)), dtype=torch.long)
        # ===================forward=====================
        self.optimizer.zero_grad()  # make sure gradients are not accimulated.
        x_hat,y_hat, mu, logvar,z = model(x)
        BCE_loss, KLD_loss, CEP_loss, total_loss = self.loss_function(x_hat, x,y_hat,y, mu, logvar)
        # ===================backward====================
        #optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        train_loss = train_loss + total_loss.item()
      return train_loss
    #############################################################    
    # evaluates the model on the test set
    def test(self,model):
      means, logvars, labels, images = list(), list(), list(), list()
      zs=[]
      test_BCE_loss=0
      test_KLD_loss=0
      test_CEP_loss=0
      test_total_loss = 0
      import random
      random.seed(1234)
      with torch.no_grad():
        model.eval()
        for x, y in self.testloader:
          x = x.to(device)
          y = y.to(device)
          y=y.to(device)
          y=torch.tensor(torch.reshape(y, (-1,)), dtype=torch.long)
          # forward
          x_hat,y_hat, mu, logvar,z = model(x)
          BCE_loss, KLD_loss, CEP_loss, total_loss = self.loss_function(x_hat, x, y_hat,y, mu, logvar)
          test_total_loss = test_total_loss + total_loss.item()
          test_BCE_loss = test_BCE_loss + BCE_loss.item()
          test_KLD_loss = test_KLD_loss + KLD_loss.item()
          test_CEP_loss = test_CEP_loss + CEP_loss.item()
          # log
          means.append(mu.detach())
          logvars.append(logvar.detach())
          labels.append(y.detach())
          images.append(x.detach())
          zs.append(z.detach())
      return test_BCE_loss, test_KLD_loss, test_CEP_loss, test_total_loss, means, logvars, labels, images,zs
    #############################################################
    def generate_test_results(self):
      
      loss_scale_show=1
    
      test_tracker=[]
      test_BCE_tracker=[]
      test_KLD_tracker=[]
      test_CEP_tracker=[]
      z_list=[]
    
      for epoch in range(1):
          # the following line, will read test data from tes
          test_BCE_loss, test_KLD_loss, test_CEP_loss, test_total_loss, means, logvars, labels, images,z = self.test(self.model)
          
          self.miu_last=torch.cat(means)
          self.var_last=torch.cat(logvars)
          self.y_last=torch.cat(labels)
          self.x_last=torch.cat(images)
          self.zs=torch.cat(z)
    
          test_total_loss_scaled = test_total_loss*loss_scale_show/ len(self.testloader.dataset)
          test_BCE_loss_scaled = test_BCE_loss*loss_scale_show/ len(self.testloader.dataset)
          test_KLD_loss_scaled = test_KLD_loss*loss_scale_show/ len(self.testloader.dataset)
          test_CEP_loss_scaled = test_CEP_loss*loss_scale_show/ len(self.testloader.dataset)
          
          self.test_tracker.append(test_total_loss_scaled)
          self.test_BCE_tracker.append(test_BCE_loss_scaled)
          self.test_KLD_tracker.append(test_KLD_loss_scaled)
          self.test_CEP_tracker.append(test_CEP_loss_scaled)
          
          #return(test_tracker,test_BCE_tracker,test_KLD_tracker,test_CEP_tracker)
    #############################################################
    #############################################################
    def generate_test_results(self):
      
      loss_scale_show=1
    
      test_tracker=[]
      test_BCE_tracker=[]
      test_KLD_tracker=[]
      test_CEP_tracker=[]
      z_list=[]
    
      for epoch in range(1):
          # the following line, will read test data from tes
          test_BCE_loss, test_KLD_loss, test_CEP_loss, test_total_loss, means, logvars, labels, images,z = self.test(self.model)
          
          self.miu_last=torch.cat(means)
          self.var_last=torch.cat(logvars)
          self.y_last=torch.cat(labels)
          self.x_last=torch.cat(images)
          self.zs=torch.cat(z)
    
          test_total_loss_scaled = test_total_loss*loss_scale_show/ len(self.testloader.dataset)
          test_BCE_loss_scaled = test_BCE_loss*loss_scale_show/ len(self.testloader.dataset)
          test_KLD_loss_scaled = test_KLD_loss*loss_scale_show/ len(self.testloader.dataset)
          test_CEP_loss_scaled = test_CEP_loss*loss_scale_show/ len(self.testloader.dataset)
          
          self.test_tracker.append(test_total_loss_scaled)
          self.test_BCE_tracker.append(test_BCE_loss_scaled)
          self.test_KLD_tracker.append(test_KLD_loss_scaled)
          self.test_CEP_tracker.append(test_CEP_loss_scaled)
          
          #return(test_tracker,test_BCE_tracker,test_KLD_tracker,test_CEP_tracker)
    #############################################################

    def plot_zs_pca2(self):
        #self.zs
        #self.y_last
        pass
        
    
    
    def regression_analysis(self,means,labels):
      #means_all_test=torch.empty_like(means[0])
      #for i in range(len(means)):
        #means_all_test = torch.cat((means_all_test,means[i]),0)
      #means_all_test=means_all_test.cpu().detach().numpy()
      #labels_all_test=torch.empty_like(labels[0])
      #for i in range(len(labels)):
        #labels_all_test = torch.cat((labels_all_test,labels[i]),0)
      #labels_all_test=labels_all_test.cpu().detach().numpy()
      means_all_test = means.cpu().detach().numpy()
      labels_all_test = labels.cpu().detach().numpy()
      from sklearn.linear_model import LogisticRegression
      reg = LogisticRegression(solver='liblinear',max_iter=500).fit(means_all_test, labels_all_test.reshape(-1, 1).ravel())
      reg.predict(means_all_test)
      print(reg.score(means_all_test, labels_all_test.reshape(-1, 1)))
    #############################################################
    def calculate_lower_dimensions(self,miu_last,y_last,N=1000):
      import random
      index_rand=random.sample(range(1, miu_last.shape[0]), N)
      import umap.umap_ as umap
      from sklearn.manifold import TSNE
      from sklearn.decomposition import PCA
      #X=x_last
      X = miu_last
      Y = y_last
      X=X[index_rand,]
      Y=Y[index_rand]
      Y = list(map(int, Y))
      E_TSNE = TSNE(n_components=3).fit_transform(X.cpu())
      reducer = umap.UMAP(random_state=42,n_components=20)
      E_UMAP = reducer.fit_transform(X.cpu())
      pca = PCA(n_components=3)
      E_PCA=pca.fit_transform(X.cpu())
      return(E_TSNE,E_UMAP,E_PCA,Y)
    #############################################################
    def plot_lower_dimension(self,EE,Y,size_dot=1,projection='2d',save_str='myplot.pdf'):
      
        if projection == '2d':
            ax = plt.axes()
            sc=ax.scatter(EE[:,0], EE[:,1],s=size_dot,c=Y,marker=".",cmap='tab20')
        elif projection == '3d':
            ax = plt.axes(projection='3d')
            sc=ax.scatter(EE[:,0], EE[:,1],EE[:,2],s=size_dot,c=Y,marker=".",cmap='tab20')
            # Hide grid lines
            ax.grid(False)
            # Hide axes ticks
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])

        #legend = ax.legend(#handles=sc.legend_elements()[1],
                          #title="class"
                          #framealpha=0,
                          #ncol=10
                          #loc=2,
                          #fontsize='xx-small'
        #                   )
        #ax.add_artist(legend)

        ax.legend()
        plt.savefig(save_str)
        plt.show()
#############################################################      
    def display_images_real_vs_synthetic(self,number_class=3,image_number=40,image_shape=28,normalized_factor=256):
        fig = plt.figure(figsize=(10, 7))
        self.model.eval()
        
        img_real=self.x_last[self.y_last==number_class][image_number].cpu().detach().numpy().reshape(image_shape,image_shape)*normalized_factor
        fig.add_subplot(3, 1, 1)
        plt.imshow(img_real,vmin=0,vmax=normalized_factor-1)
        plt.axis('off')
        plt.title("Real")
        
        
        img_vae=self.model.decoder(self.miu_last[self.y_last==number_class][image_number:image_number+2]).cpu().detach().numpy().reshape(2,image_shape,image_shape)[0]*normalized_factor
        fig.add_subplot(3, 1, 2)
        plt.imshow(img_vae,vmin=0,vmax=normalized_factor-1)
        plt.axis('off')
        plt.title("Reconstructed")
        
        img_diference=img_real-img_vae
        fig.add_subplot(3, 1, 3)
        plt.imshow(img_diference,vmin=0,vmax=normalized_factor-1)
        plt.axis('off')
        plt.title("Difference")
        print("mean difference = "+str(np.mean(np.abs(img_diference))))
#############################################################
    def distance_kl(self,mean1,std1,mean2,std2):
            n=std1.shape[0]
            std1_mat=np.zeros((n,n))
            np.fill_diagonal(std1_mat, list(std1))
            std2_mat=np.zeros((n,n))
            np.fill_diagonal(std2_mat, list(std2))
            expression1 = np.log(np.linalg.det(std2_mat)/np.linalg.det(std1_mat))
            expression2 = np.trace(np.matmul(np.linalg.inv(std2_mat),std1_mat))
            expression3 = np.matmul(np.matmul((mean1-mean2).T,np.linalg.inv(std2_mat)),(mean1-mean2)) 
            distance = 1/2*(expression1-n+expression2+expression3)
            return(distance)
#############################################################
    def latent_traversal(self,
                         points_mean,
                         points_std,
                         start_id,
                         end_id,
                         k_neighbor_ratio=0.1,
                         distance_eucludian=False,
                         plot_results_2d=True):
        n_samples=points_mean.shape[0]
        k=int(0.05*n_samples)
        dist_array = np.zeros([n_samples,n_samples])
        for i in range(n_samples):
            for j in range(n_samples):
                #print(i,j)
                if distance_eucludian:
                    dist_array[i,j]=np.linalg.norm(points_mean[i,:]-points_mean[j,:])
                else:
                    d1=self.distance_kl(points_mean[i,:],points_std[i,:],points_mean[j,:],points_std[j,:])
                    d2=self.distance_kl(points_mean[j,:],points_std[j,:],points_mean[i,:],points_std[i,:])
                    dist_array[i,j]=abs(d1+d2)/2
                
        
        adjucency_mat = np.zeros([n_samples,n_samples])
        for i in range(n_samples):
            nearest_ids = dist_array[i,:].argsort()[:k]
            nearest_ids = list(np.delete(nearest_ids,np.where(nearest_ids == i)))
            for j in nearest_ids:
                adjucency_mat[i,j] = dist_array[i,j]
        
            
        import igraph
        A = adjucency_mat
        g = igraph.Graph.Adjacency((A > 0).tolist())
        # Add edge weights and node labels.
        g.es['weight'] = A[A.nonzero()]
        #g.vs['label'] = node_names  # or a.index/a.columns
    
        #g=igraph.Graph.Weighted_Adjacency(adjucency_mat,mode='undirected')
        g.is_weighted()
        
        shrt_pth=g.get_shortest_paths(start_id, to=end_id,weights=g.es["weight"])
        shrt_pth
    
        if(plot_results_2d):
            import matplotlib.pyplot as plt
            size_dot=1
            plt.scatter(x=points_mean[:,0],y=points_mean[:,1],s=size_dot)
            plt.plot(points_mean[shrt_pth,0].tolist()[0],points_mean[shrt_pth,1].tolist()[0],'-r')
            plt.scatter(x=points_mean[shrt_pth[0],0],y=points_mean[shrt_pth[0],1],c='green',s=size_dot)
            plt.scatter(x=points_mean[shrt_pth[-1],0],y=points_mean[shrt_pth[-1],1],c='yellow',s=size_dot)
            plt.show()
            
        return(list(shrt_pth[0]))
#############################################################
    def getEquidistantPoints(self,p1, p2, parts):
        return (list(zip(*[np.linspace(p1[i], p2[i], parts+1) for i in range(len(p1))])))
#############################################################
    def latent_traversal_interpolated(self,
                                      points_mean,
                                      points_std,
                                      steps,
                                      linespace_k=5):
        #############################
        ll=[]
        for i in list(range(len(steps)-1)):
            mm=self.getEquidistantPoints(points_mean[steps[i]], points_mean[steps[i+1]], linespace_k)
            ll=ll+mm
        for i in range(len(ll)):
            ll[i]=list(ll[i])
        points_mean_interpolated=np.array(ll) 
        #############################.        I think interpolating standard deviation is not meaningful. We calculated here but we are not going to use it. 
        ll=[]
        for i in list(range(len(steps)-1)):
            mm=self.getEquidistantPoints(points_std[steps[i]], points_std[steps[i+1]], linespace_k)
            ll=ll+mm
        for i in range(len(ll)):
            ll[i]=list(ll[i])
        points_std_interpolated=np.array(ll)
        
        return((points_mean_interpolated,points_std_interpolated))
#############################################################    
    # Simply traverse between two end points and create some equally spaced points on the line.
    def sample_data_on_a_line(self,x0,x1,number_of_images):
      space_dim=20
      line_distance=x1-x0
      n=number_of_images
      delta=line_distance/n
      line = torch.empty(size=(number_of_images, space_dim))
      for i in range(number_of_images):
        line[i]=x0+i*delta
      line=line.cpu()
      return(line)
#############################################################
    def generate_data_linear_from_a_to_b(self,model,miu_last,y_last,number_of_images,start_id,end_id,flat=False):
      model.eval()
      with torch.no_grad():
        x0=miu_last[start_id]
        x1=miu_last[end_id]
        line = self.sample_data_on_a_line(x0,x1,number_of_images)
        if (flat):
          line_decoded = np.fliplr(((model.decoder(line).cpu().detach().numpy())))
        else:
          line_decoded = np.fliplr(((model.decoder(line).cpu().detach().numpy().reshape(number_of_images,28,28)*256)))
      return(line_decoded)
#############################################################  
    def save_GIF(self,decoded_objects,file_path_root,indicator,speed=5):
      import numpy as np
      import matplotlib.pyplot as plt
      import imageio
      import os
    
      #a = line_decoded
      #a = line_decoded2
      a = decoded_objects
      a = np.array(a)
    
      images = []
    
      for array_ in a:
          #file_path = "/content/drive/MyDrive/coh_pm/image.png"
          file_path = file_path_root+indicator+".png"
          #img = plt.figure(figsize = (8,8))
          plt.imshow(array_, origin = 'lower',vmin=20,vmax=200)
          plt.colorbar(shrink = 0.5)
          plt.savefig(file_path) #Saves each figure as an image
          images.append(imageio.imread(file_path)) #Adds images to list
          plt.clf()
    
      plt.close()
      os.remove(file_path)
      imageio.mimsave(file_path_root +indicator+ ".gif", images, fps=speed) #Creates gif out of list of images
#############################################################    
    def generate_synthetic_data(self,model,miu_last,y_last,number_of_additional_data):
          import numpy as np
          number_of_images_per_traversal=20
        
          k=int(number_of_additional_data/number_of_images_per_traversal)
          synthetic_data_all = torch.empty(0, 28*28)
          for i in range(k):
            m = np.random.choice(range(500), 2, replace=False) 
            start_id = m[0]
            end_id = m[1]
            synthetic_data = self.generate_data_linear_from_a_to_b(model,miu_last,y_last,number_of_images_per_traversal,start_id,end_id,flat=True)
        
            if (i==0):
              synthetic_data_all=synthetic_data
            else:
              synthetic_data_all=np.append(synthetic_data_all,synthetic_data, axis=0)
            #synthetic_data_all = torch.cat((synthetic_data_all, synthetic_data), 0)
          return(synthetic_data_all)
          #return(synthetic_data)
############################################################# 
    def traverse(self,number_of_images,start_id,end_id,file_path_root="traverse",model_name="supervised_",flat=True):
        line_decoded = self.generate_data_linear_from_a_to_b(self.model,self.miu_last,self.y_last,number_of_images,start_id,end_id,flat=flat)
        decoded_objects=line_decoded
        indicator = model_name+"_"+str(start_id)+"_"+str(end_id)
        if (flat==False):
            self.save_GIF(decoded_objects,file_path_root,indicator)
        return(line_decoded)
#############################################################
    def traverse_multiple(self,number_class,number_of_images,start_id,end_id,file_path_root="multiple_traverse"):
        for i in range(10):
          number_class=2
          number_of_images=20
          start_id=10
          end_id=80+i
          model_name="supervised_"
          #model_name="UNsupervised_"
          line_decoded = self.generate_data_linear_from_a_to_b(self.model,self.miu_last,self.y_last,number_class,number_of_images,start_id,end_id,flat=False)
          decoded_objects=line_decoded
          indicator = "multiple_"+model_name+str(number_class)+"_"+str(start_id)+"_"+str(end_id)
          self.save_GIF(decoded_objects,file_path_root,indicator)
          print("successful!")
#############################################################   
    def append_augmented_data_to_original(self,synthetic_physical_data,number_class,number_of_additional_data):
        physical_data_all=np.append(self.x_last.cpu().numpy(),synthetic_physical_data,axis=0)
        physical_data_all_lables=np.append(self.y_last.cpu().numpy(),np.repeat(number_class, number_of_additional_data))
        self.original_with_augmented_data_all_X=torch.from_numpy(physical_data_all)
        self.original_with_augmented_data_all_lables=torch.from_numpy(physical_data_all_lables)
#############################################################
#############################################################
class CLS_ARCH(nn.Module):
    def __init__(self,input_size,dropout_rate=0.30):
        super().__init__()
        self.input_size=input_size
        dropout_rate = dropout_rate
        self.nn_predictor = nn.Sequential(
            nn.Linear(self.input_size,500),
            nn.ReLU(),
            nn.BatchNorm1d(500),
            nn.Dropout(p=dropout_rate),
            ##################
            nn.Linear(500, 300),
            nn.ReLU(),
            nn.BatchNorm1d(300),
            nn.Dropout(p=dropout_rate),
            ##################
            nn.Linear(300, 100),
            nn.ReLU(),
            nn.BatchNorm1d(100),
            nn.Dropout(p=dropout_rate),
            ##################
            nn.Linear(100, 20),
            nn.ReLU(),
            nn.BatchNorm1d(20),
            nn.Dropout(p=dropout_rate),
            ##################
            nn.Linear(20, 10),
            #nn.Softmax()
        )
    def forward(self, x):
        y_hat = self.nn_predictor(x)
        return y_hat


#############################################################

#############################################################
class CLS(MyDataset,CLS_ARCH):
#############################################################
    def __init__(self):
        ##########
        self.df_XY = self.MNIST_data()
        ##########
        #obj.organize_data(df_XY)
        self.input_size = self.df_XY.shape[1]-1
        CLS_ARCH.__init__(self,self.input_size)
        MyDataset.__init__(self,df=self.df_XY)
        self.organize_data()
#############################################################
    def model_initialiaze(self):
        self.model=CLS_ARCH(self.input_size).cpu()
        self.train_tracker=[]
        self.test_tracker=[]
        self.test_BCE_tracker=[]
        self.test_KLD_tracker=[]
        self.test_CEP_tracker=[]
#############################################################        
    def model_save(self,address):
        torch.save(self.model.state_dict(),address)
#############################################################   
    def model_load(self,address):
        random.seed(1234)
        self.model_initialiaze()
        self.model.load_state_dict(torch.load(address))
#############################################################
    def save_residuals(self,address='residuals.pkl'):
        import pickle
        residuals={'train_tracker':self.train_tracker,
           'test_BCE_tracker':self.test_BCE_tracker,
           'test_KLD_tracker':self.test_KLD_tracker,
           'test_CEP_tracker':self.test_CEP_tracker,
           'test_tracker':self.test_tracker}
        with open(address, 'wb') as f:
            pickle.dump(residuals, f)
#############################################################
    def load_residuals(self,address='residuals.pkl'):
        import pickle
        with open(address, 'rb') as f:
            residuals = pickle.load(f)
            self.train_tracker = residuals['train_tracker']
            self.test_BCE_tracker = residuals['test_BCE_tracker']
            self.test_KLD_tracker = residuals['test_KLD_tracker']
            self.test_CEP_tracker = residuals['test_CEP_tracker']
            self.test_tracker = residuals['test_tracker']
#############################################################   
    def visualize_model_architecture(self):
        pass
############################################################# 
    def plot_residuals(self,init_index=0):
        import matplotlib.pyplot as plt
        plt.plot(self.train_tracker[init_index:], label='Training Total loss')
        plt.plot(self.test_tracker[init_index:], label='Test Total loss')
        plt.plot(self.test_BCE_tracker[init_index:], label='Test BCE loss')
        plt.plot(self.test_KLD_tracker[init_index:], label='Test KLD loss')
        plt.plot(self.test_CEP_tracker[init_index:], label='Test CEP loss')
        plt.legend()
        plt.show()
#############################################################
    def model_training(self,epochs,learning_rate):
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=learning_rate,weight_decay=2e-5)
        iteration_no = 0
        loss_scale_show = 1
        #codes = dict(μ=list(), logσ2=list(), y=list(), x=list())
        for epoch in range(1, epochs+1):
            iteration_no = iteration_no+1
            # train for one epocha
            self.train_total_loss = self.train(self.model)
        
            self.test_total_loss,self.labels, self.images,y_pred = self.test(self.model)

            train_total_loss_scaled = self.train_total_loss*loss_scale_show/ len(self.trainloader.dataset)
            test_total_loss_scaled = self.test_total_loss*loss_scale_show/ len(self.testloader.dataset)

            self.train_tracker.append(train_total_loss_scaled)
            self.test_tracker.append(test_total_loss_scaled)
            # print the test loss for the epoch
            print(f'====> Epoch: {iteration_no} total_train_loss: {train_total_loss_scaled:.6f} Total_test_loss: {test_total_loss_scaled:.6f}')
          
#############################################################   
    def MNIST_data(self):
        from keras.datasets import mnist
        (train_X, train_y), (test_X, test_y) = mnist.load_data()
        train_X = np.reshape(train_X, (60000,28*28))
        test_X = np.reshape(test_X, (10000,28*28))
        X = np.concatenate((train_X,test_X),axis=0)
        X = X/255
        Y = np.concatenate((train_y,test_y),axis=0)
        df_X=pd.DataFrame(X)
        df_Y=pd.DataFrame(Y)
        df_Y.columns=["Y"]
        df_XY=pd.concat([df_X,df_Y],axis=1)
        return(df_XY)
#############################################################    
    def organize_data(self):
        from sklearn.model_selection import train_test_split
        self.df_XY = self.df_XY.sample(frac = 1,random_state=1234)
        
        df_XY_train, df_XY_test = train_test_split(self.df_XY, test_size=0.2, random_state=1234)
        
        data_train = MyDataset(df=df_XY_train,y_label=["Y"])
        data_test = MyDataset(df=df_XY_test,y_label=["Y"])
        import random
        random.seed(1234)
        self.BATCH_SIZE=512
        trainloader = torch.utils.data.DataLoader(dataset = data_train,
                                                   batch_size = self.BATCH_SIZE,
                                                  shuffle=False)
        testloader = torch.utils.data.DataLoader(dataset = data_test,
                                                  batch_size = self.BATCH_SIZE,
                                                 shuffle=False)
        self.trainloader = trainloader
        self.testloader = testloader
        # Reconstruction + KL divergence losses summed over all elements and batch
    #############################################################
    def loss_function(self,y_pred,y):
        crs_entrpy = nn.CrossEntropyLoss()
        CEP = crs_entrpy(y_pred.cpu(),y.cpu())
        return CEP
    #############################################################   
    # performs one epoch of training and returns the training loss for this epoch
    def train(self,model):
      model.train()
      train_loss = 0
      for x, y in self.trainloader:
        x = x.cpu()
        y = y.cpu()
        y=torch.tensor(torch.reshape(y, (-1,)), dtype=torch.long)
        # ===================forward=====================
        self.optimizer.zero_grad()  # make sure gradients are not accimulated.
        y_pred = model(x)
        total_loss = self.loss_function(y_pred,y)
        # ===================backward====================
        #optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        train_loss = train_loss + total_loss.item()
      return train_loss
    #############################################################    
    # evaluates the model on the test set
    def test(self,model):
      labels, images = list(), list()
      ys_pred=[]
      test_total_loss = 0
      import random
      random.seed(1234)
      with torch.no_grad():
        model.eval()
        for x, y in self.testloader:
          x = x.cpu()
          y = y.cpu()
          y=torch.tensor(torch.reshape(y, (-1,)), dtype=torch.long)
          # forward
          y_pred = model(x)
          total_loss = self.loss_function(y_pred,y)
          test_total_loss = test_total_loss + total_loss.item()

          labels.append(y.detach())
          images.append(x.detach())
          ys_pred.append(y_pred.detach())
      return test_total_loss,labels, images,ys_pred
    #############################################################
    def read_gifs(self,filename, asNumpy=True):
        import PIL
        import os
        """ readGif(filename, asNumpy=True)
        
        Read images from an animated GIF file.  Returns a list of numpy 
        arrays, or, if asNumpy is false, a list if PIL images.
        
        """
    
        # Check PIL
        if PIL is None:
            raise RuntimeError("Need PIL to read animated gif files.")
        
        # Check Numpy
        if np is None:
            raise RuntimeError("Need Numpy to read animated gif files.")
        
        # Check whether it exists
        if not os.path.isfile(filename):
            raise IOError('File not found: '+str(filename))
        
        # Load file using PIL
        pilIm = PIL.Image.open(filename)    
        pilIm.seek(0)
        
        # Read all images inside
        images = []
        try:
            while True:
                # Get image as numpy array
                tmp = pilIm.convert() # Make without palette
                a = np.asarray(tmp)
                if len(a.shape)==0:
                    raise MemoryError("Too little memory to convert PIL image to array")
                # Store, and next
                images.append(a)
                pilIm.seek(pilIm.tell()+1)
        except EOFError:
            pass
        
        # Convert to normal PIL images if needed
        if not asNumpy:
            images2 = images
            images = []
            for im in images2:            
                images.append( PIL.Image.fromarray(im) )
        
        # Done
        return images 
    
############################################ 
############################################ 
############################################    
class Utils():
 def linear_traversal_original_coordinates(self):
    from keras.datasets import mnist
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    
    from matplotlib import pyplot
    
    three=train_X[train_y==3,]
    pyplot.imshow(three[1], cmap=pyplot.get_cmap('gray'))
    pyplot.show()
    
    pyplot.imshow(three[2], cmap=pyplot.get_cmap('gray'))
    pyplot.show()
    
    delta=(three[2]-three[1])/10
    for i in range(11):
        pyplot.imshow(three[1]+(i)*delta, cmap=pyplot.get_cmap('gray'))
        pyplot.show()
 def linear_traverse_original_space(x0_png_address,x1_png_address,output_address,number_of_images=20):
     
     from PIL import Image
     import numpy as np
     import matplotlib.pyplot as plt
     import imageio
     import os
     
     x0 = Image.open(x0_png_address)
     x0 = x0.convert('L')
     x0 = np.array(x0)
     x0=x0.astype(np.int16)
     
     x1 = Image.open(x1_png_address)
     x1 = x1.convert('L')
     x1=np.array(x1)
     x1=x1.astype(np.int16)
     

     line = np.zeros((number_of_images+1,x0.shape[0], x0.shape[1]))
     delta=(x1-x0)/number_of_images
     
     for i in range(number_of_images+1):
         print(i)
         line[i]=x0+i*delta
         
     plt.imshow(line[20], origin = 'lower')
     plt.imshow(delta, origin = 'lower')
     
     


    #line=line+np.random.rand(21, 168, 168)*20
     images = []
    
     for array_ in line:
        #file_path = "/content/drive/MyDrive/coh_pm/image.png"
        file_path = './'
        #img = plt.figure(figsize = (8,8))
        plt.imshow(array_)
        plt.colorbar(shrink = 0.5)
        plt.savefig(file_path) #Saves each figure as an image
        images.append(imageio.imread(file_path)) #Adds images to list
        plt.clf()
      
     plt.close()

     os.remove(file_path)
     imageio.mimsave(output_address+'.gif', images, fps=7)
    
############################################ 
############################################ 
############################################     
