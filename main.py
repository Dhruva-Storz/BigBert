import torch
import models
from save_zip import writeScores
import numpy as np
import torch.utils.data as Data
from torch.autograd import Variable
from scipy.stats.stats import pearsonr

### Import max pooling and average pooling embeddings as torch tensors

# Max pooling
english_m = torch.load('./Embeddings/Embeddings_batch100/english_train_embeddings_max.pt')
german_m = torch.load('./Embeddings/Embeddings_batch100/german_embeddings_max.pt')
english_val_m = torch.load('./Embeddings/Embeddings_batch100/english_val_embeddings_max.pt')
german_val_m = torch.load('./Embeddings/Embeddings_batch100/german_val_embeddings_max.pt')
english_test_m = torch.load('./Embeddings/Embeddings_batch100/english_test_embeddings_max.pt')
german_test_m = torch.load('./Embeddings/Embeddings_batch100/german_test_embeddings_max.pt')
# Avg pooling
english_avg = torch.load('./Embeddings/Embeddings_batch100/english_train_embeddings_mean.pt')
german_avg = torch.load('./Embeddings/Embeddings_batch100/german_embeddings_mean.pt')
english_val_avg = torch.load('./Embeddings/Embeddings_batch100/english_val_embeddings_mean.pt')
german_val_avg = torch.load('./Embeddings/Embeddings_batch100/german_val_embeddings_mean.pt')
english_test_avg = torch.load('./Embeddings/Embeddings_batch100/english_test_embeddings_mean.pt')
german_test_avg = torch.load('./Embeddings/Embeddings_batch100/german_test_embeddings_mean.pt')



### LOAD Scores

f_train_scores = open("./data_en_de/train.ende.scores",'r')
de_train_scores = f_train_scores.readlines()
f_val_scores = open("./data_en_de/dev.ende.scores",'r')
de_val_scores = f_val_scores.readlines()

train_scores = np.array(de_train_scores).astype(float)
# Shape (7000,)
y_train =train_scores
# Shape (1000,)
val_scores = np.array(de_val_scores).astype(float)
y_val =val_scores

# Shape (7000, 768 x 2)
english = torch.cat((english_m, english_avg), dim=1)
german = torch.cat((german_m, german_avg), dim=1)

# Shape (1000, 768 x 2)
english_val = torch.cat((english_val_m, english_val_avg), dim=1)
german_val = torch.cat((german_val_m, german_val_avg), dim=1)

# Shape (1000, 768 x 2)
english_test = torch.cat((english_test_m, english_test_avg), dim=1)
german_test = torch.cat((german_test_m, german_test_avg), dim=1)


###### Create Feature Vectors ########################################
###### (en, ge, |en-ge|, en*ge) ######################################

### ** TRAIN SET ** ###
en_ge_cat = torch.cat((english,german), dim=1)
en_ge_product = english * german
en_ge_abs_dif = (english - german).abs()

# Train Tensor of Shape (7000, 768 x 8)
X_train = torch.cat((en_ge_cat, en_ge_product, en_ge_abs_dif), dim=1)

### ** VALIDATION SET ** ### 
en_ge_cat_val = torch.cat((english_val,german_val), dim=1)
en_ge_product_val = english_val * german_val
en_ge_abs_dif_val = (english_val - german_val).abs()

# Validation Tensor of Shape (1000, 768 x 8)
X_val = torch.cat((en_ge_cat_val, en_ge_product_val, en_ge_abs_dif_val), dim=1)

### ** TEST SET ** ###

# Test Tensor of Shape (1000, 768 x 8)
en_ge_cat_test = torch.cat((english_test,german_test), dim=1)
en_ge_product_test = english_test * german_test
en_ge_abs_dif_test = (english_test - german_test).abs()

X_test = torch.cat((en_ge_cat_test, en_ge_product_test, en_ge_abs_dif_test), dim=1)



def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def main(mode):
    '''
    Main method for choosing, training, and evaluating models. 
    Every model is instantiated with optimal hyperparamaters for the data.
    Hyperparameters were found using gridsearch and manual fine tuning.

    Modes:

    - MLP_scikit:

        Simple neural network from sklearn

    - MLP_torch:

        A larger custom neural network created in pytorch

    - SVR:

        Support Vector Regression from sklearn

    '''
    
    if mode == 'MLP_scikit':
        model = models.MLP_Regressor(activation='relu', regularization = 0.005, 
                        batch_size=128, hidden_layer_sizes=(4096, 2048, 1024,512, 256, 128), 
                learning_rate='adaptive',learning_rate_init=0.001, max_iter=25, n_iter_no_change=10,
                 optimizer='adam', early_stopping=True, tol=0.0001, validation_fraction=0.15)
        
        model.fit(X_train, y_train)
        
        _, pearson, r_mse = model.predict(X_val, y_val)
        
        print('Pearson: {}'.format(pearson))
        print('RMSE: {}'.format(r_mse))
        
        predictions_de = model.predict(X_test)
        writeScores('MLP_scikit', predictions_de)
        
    elif mode == 'MLP_torch':
        layers_sizes = [6144, 128, 64, 1]
        model = models.MLP(layers_sizes)
        GPU = True
        device_idx = 0
        if GPU:
            device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")
        BATCH_SIZE = 64
        learning_rate = 0.01 
        epochs = 1000
        train_dataset = Data.TensorDataset(X_train, torch.Tensor(y_train))
        if device == 0:
            num_workers = 2
        else:
            num_workers = 0
        loader_train = Data.DataLoader(
            dataset=train_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=True, num_workers=num_workers)
        
        
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        lmbda = lambda epoch: 0.9
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lmbda)
        
        # TRAIN
        epochs = 200
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for i, (x, y) in enumerate(loader_train):
                
            # Converting inputs and labels to Variable
        
                inputs = Variable(x.to(device))
                labels = Variable(y.to(device))
        
                # Clear gradient buffers because we don't want any gradient 
                # from previous epoch to carry forward, dont want to cummulate gradients
                optimizer.zero_grad()
        
                # get output from the model, given the inputs
                outputs = model(inputs)
                
                # Regularization
                reg = 0
                for param in model.parameters():
                    reg += 0.5 * (param ** 2).sum()
                reg_lambda = 0.01
                
                # get loss for the predicted output
                loss = criterion(outputs.reshape(outputs.shape[0]), labels) + \
                    reg_lambda * reg
                
                   
                train_loss += loss.item()
                
                # get gradients w.r.t to parameters
                loss.backward()
                
                # update parameters
                optimizer.step()
            with torch.no_grad():
                model.eval()
                out = model(Variable(X_val.to(device))).detach().cpu().numpy()
                
                pearson = pearsonr(out.reshape(out.shape[0]), y_val)[0]
                r_mse = rmse(out.reshape(out.shape[0]), y_val)
                print('Validation Metrics:\n')
                print('\tPearson: {}: '.format(pearson))
                print('\tRMSE: {}\n'.format(r_mse))
            model.train()
            scheduler.step()
        
            print('epoch [{}/{}], Training loss:{:.6f}'.format(
                epoch + 1, 
                epochs, 
                train_loss / len(loader_train.dataset)))
        with torch.no_grad():
            model.eval()
            out = model(Variable(X_test.to(device))).detach().cpu().numpy()
        
        predictions_de =out.reshape(out.shape[0])
        writeScores('MLP_torch', predictions_de)
    
    elif mode == 'SVR':
        
        model = models.SVR_regression(X_train, y_train,X_val, y_val, X_test) 
        model.run_model()
        model.save_model(name="SVR_prediction")



        
if __name__ == "__main__":
    # mode = 'MLP_torch'
    mode = 'SVR'
    main(mode=mode)
    

