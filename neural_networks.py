import numpy as np
import keras.backend as K
from helper_functions import do_normalization, do_inverse_norm
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from keras.layers import Dense
from keras.models import Sequential, load_model


class training():
    
    def __init__(self, model, X, Y, normW):
        self.model = model
        self.normW = normW
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X,Y, test_size=0.2)
        
        self.X_trainN = do_normalization(self.X_train,self.X_train,normW)
        self.X_testN  = do_normalization(self.X_test,self.X_train,normW)
    
        self.y_trainN = do_normalization(self.y_train,self.y_train,normW)
        self.y_testN  = do_normalization(self.y_test,self.y_train,normW)
        
    def do_training(self, batch_sz, epchs):
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10,restore_best_weights=True, min_delta=1e-4)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.000001,verbose=1,mode='min')
        self.h=self.model.fit(self.X_trainN,self.y_trainN,epochs=epchs,batch_size=batch_sz, callbacks = [es, reduce_lr], \
                              validation_data= (self.X_testN, self.y_testN))
        
    
    def get_predictions(self,X_pred):
        X_predN = do_normalization(X_pred,self.X_train,self.normW)
        pred = self.model.predict(X_predN)
        pred = do_inverse_norm(self.y_train, pred, self.normW)
        
        return pred
    
    def get_errors(self, X_pred, Y_pred):
        pred = self.get_predictions(X_pred)
        errR2 = np.zeros(Y_pred.shape[1])
        errMSE = np.zeros(Y_pred.shape[1])

        for i in range(0,Y_pred.shape[1]):
            errR2[i] = r2_score(Y_pred[:,i],pred[:,i])
            errMSE[i] = np.sqrt(mean_squared_error(Y_pred[:,i],pred[:,i])/np.mean(Y_pred[:,i]**2))
        return errR2, errMSE

    def save_model(self,name):
       self.model.save(name)

    

def custom_loss(y_true, y_pred):
    alpha=0.2
    return  abs(1-K.sum(y_pred*(maxs-mins)+means)/500)*alpha + K.mean((y_true-y_pred)**2)*(1.0-alpha)

def get_model_species(dimi, nc):
  
    model = Sequential()
    model.add(Dense(110,activation='relu',input_dim=dimi ))
    model.add(Dense(110,activation='relu'))
    model.add(Dense(110,activation='relu'))
    model.add(Dense(nc,activation='linear'))
    model.compile(optimizer='nadam',loss='mean_squared_error')
    
    return model
    
def get_model_prop(dimi,nc):
    model = Sequential()
    model.add(Dense(10,input_dim=dimi, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(nc,activation='linear'))
    model.compile(optimizer='nadam',loss='mean_squared_error')
    return model

def get_model_reac(dimi):
    model = Sequential()
    model.add(Dense(20,activation='relu',input_dim=dimi))
    model.add(Dense(20, activation = 'relu'))
    model.add(Dense(1,activation='linear'))
    model.compile(optimizer='nadam',loss='mean_squared_error')
    return model



     
