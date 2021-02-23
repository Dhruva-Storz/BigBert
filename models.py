from scipy.stats.stats import pearsonr
from sklearn.neural_network import MLPRegressor
import numpy as np
import torch
import torch.nn as nn
from sklearn.svm import SVR
from zipfile import ZipFile

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

class MLP_Regressor():

    def __init__(self, activation='relu', regularization = 0.005, batch_size=128,
        hidden_layer_sizes=(4096, 2048, 1024,512, 256, 128), learning_rate='adaptive',
        learning_rate_init=0.001, max_iter=25, n_iter_no_change=10,
        optimizer='adam', early_stopping=True, tol=0.0001, validation_fraction=0.15):
        
        self.model = MLPRegressor(activation=activation, alpha=regularization, 
                                  batch_size=batch_size,
                                  hidden_layer_sizes=hidden_layer_sizes, 
                                  learning_rate=learning_rate,
                                  learning_rate_init=learning_rate_init, 
                                  max_iter=max_iter,
                                  n_iter_no_change=n_iter_no_change, solver=optimizer, 
                                  early_stopping=early_stopping,
                                  tol=tol, validation_fraction=validation_fraction,
                                  verbose=True)
        
    def fit(self, x, y):

        print('Training...')
        self.model.fit(x, y)
    
    def predict(self, x_test, y_test=None):
        
        y_pred = self.model.predict(x_test)
        
        if y_test != None:
            return y_pred
        else:
            pearson, rmse_score = self.get_scores(y_test, y_pred)
            return y_pred, pearson, rmse_score
            
    
    def get_scores(self,  y_test, y_pred):
        pearson = pearsonr(y_test, y_pred)[0]
        rmse_score = rmse(y_pred, y_test)
        
        return pearson, rmse_score
        
class MLP(nn.Module):

    def __init__(self, layers_sizes):
        super(MLP, self).__init__()
        self.layers_sizes = layers_sizes
        self.network = nn.Sequential()
        for i, (in_dims, out_dims) in enumerate(zip(self.layers_sizes[:-1], self.layers_sizes[1:])):
            self.network.add_module(
                name=f'Linear {i}',module=nn.Linear(in_dims, out_dims))
            if i != len(self.layers_sizes) - 2:
                self.network.add_module(name=f'Activation {i}', module=nn.ReLU())
                self.network.add_module(name=f'Dropout {i}', module=nn.Dropout(0.4))
            else:
                self.network.add_module(name='Identity', module=nn.Identity())
            
    def forward(self, x):
        
        return self.network(x)
    

class SVR_regression():
    
    def __init__(self, X_train, y_train, X_val, y_val, X_test, c=0.1, epsilon=0.1, kernel='rbf'):
        # Default initializations above are the optimal hyperparameters for translation dataset
        print("Initializing...")
        self.c=c
        self.epsilon=epsilon
        self.kernel=kernel
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val
        self.X_test = X_test
    

    def run_model(self):
        '''
        Runs model through entire process. From training to testing and printing validation results
        '''

        # constructs and fits svr to input embedding and scores
        print('Training...')
        self.svr = SVR(kernel = self.kernel, C=self.c, epsilon=self.epsilon, verbose=True)
        self.svr.fit(self.X_train, self.y_train)

        # predicts scores for validation set
        print('Predicting...')
        predictions = self.svr.predict(self.X_val)
        
        pearson = pearsonr(self.y_val, predictions)
        RMSE = np.sqrt(((predictions - self.y_val) ** 2).mean())
        
        print(f'RMSE: {RMSE} Pearson {pearson[0]}')
        print()

    def save_model(self, name, mode = 1):
        '''
        saves model predictions for submission to codalab
        '''

        print("Saving model...")

        # mode1 = save model trained on test only
        if mode == 1:
            predictions = self.svr.predict(self.X_test)

        #mode 2 = save model trained on test + val
        elif mode == 2:
            X_deploy = np.concatenate((self.X_train, self.X_val), axis=0)
            y_deploy = np.concatenate((self.y_train, self.y_val), axis=0)
            self.svr.fit(X_deploy, y_deploy)
            predictions = self.svr.predict(self.X_test)


        fn = "predictions.txt"
        print("")
        with open(fn, 'w') as output_file:
            for idx,x in enumerate(predictions):
                output_file.write(f"{x}\n")

        with ZipFile(name+".zip","w") as newzip:
            newzip.write("predictions.txt")