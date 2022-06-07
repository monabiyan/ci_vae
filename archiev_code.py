#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 13:25:32 2022

@author: mnabian
"""
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