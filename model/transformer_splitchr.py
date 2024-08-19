import json
import copy
import torch
import optuna
import sklearn
import random
import os

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold

from torch import optim
from torch.nn import Sequential, MaxPool1d, Flatten, LeakyReLU, BatchNorm1d, Dropout, Linear, ReLU, Tanh, GELU

from torch.optim import SGD, Adam
from torch.utils.data import DataLoader, Dataset

from model import embed_layer, mh_attention, encoder, utils

from sklearn.manifold import LocallyLinearEmbedding, SpectralEmbedding, TSNE

# os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# ==============================================================
# Utils/Help function
# ==============================================================
def get_activation_func(name):
    if name == 'ReLU':
        act_func = ReLU()
    elif name == 'LeakyReLU':
        act_func = LeakyReLU()
    elif name == 'GELU':
        act_func = GELU()
    else:
        act_func = Tanh()
    return act_func

def preprocess_mimax_scaler(y_train, y_val):
    minmax_scaler = MinMaxScaler()
    y_train = np.expand_dims(y_train, axis=1)
    y_val = np.expand_dims(y_val, axis=1)
    y_train_scaled = minmax_scaler.fit_transform(y_train)
    y_val_scaled   = minmax_scaler.transform(y_val)
    return y_train_scaled, y_val_scaled, minmax_scaler

# decomposition PCA
def decomposition_PCA(X_train, X_val, tuning_params):
    pca = PCA(tuning_params['pca'])
    pca.fit(X_train)
    X_train_scaled = pca.transform(X_train)
    X_val_scaled = pca.transform(X_val)
    # pk.dump(pca, open('./pca.pkl', 'wb'))
    # print('shape after PCA: train ={}, val={}'.format(X_train.shape, X_val.shape))
    return X_train_scaled, X_val_scaled

# Create a custom dataset class to handle list of tensors
class ListTensorDataset(Dataset):
    def __init__(self, X_list, y_train):
        self.X_list = X_list
        self.y_train = y_train

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        # return a list containing all tensors in the batch and the corresponding label
        return [X[idx] for X in self.X_list], self.y_train[idx]

# set seeds for reproducibility
def set_seeds(seed: int=42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # if gpu cuda available
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cuda.matmul.allow_tf32 = True
    # torch.use_deterministic_algorithms(True)
    # torch.utils.deterministic.fill_uninitialized_memory = True


# Create mask to ignore Padding token
def create_mask_each_chr(X):
    mask_all_chr = []
    for xi in X:
        mask = (xi != 0).astype(np.int32) # Shape: (batch_size, sequence_length)
        # Reshape the mask to the desired shape: (batch_size, 1, 1, sequence_length)
        final_mask = mask[:, np.newaxis, np.newaxis, :]  # Shape: (batch_size, 1, 1, sequence_length)
        mask_tensor = torch.LongTensor(final_mask)
        mask_all_chr.append(mask_tensor)
    return mask_all_chr

# Average transformer output along sequence-dim to process with linear head regression
class Pooling_Transformer_output(torch.nn.Module):
    def __init__(self):
        super(Pooling_Transformer_output, self).__init__()

    def forward(self, x):
        return torch.mean(x, dim=1)

# ==============================================================
# Define Transformer Model
# ==============================================================

class TransformerSNP(torch.nn.Module):
    def __init__(self, device, src_vocab_size, seq_len, max_seq_len, tuning_params):
        super(TransformerSNP, self).__init__()

        self.max_seq_len = max_seq_len
        embedding_dimension = int(tuning_params['n_heads'] * tuning_params['d_k'])
        self.embedding = embed_layer.Embedding(src_vocab_size, embedding_dimension)
        self.positional_encoding = embed_layer.PositionalEncoding(embedding_dimension, seq_len, tuning_params['dropout'])
        # self.positional_encoding = embed_layer.PositionalEmbedding(embedding_dimension, seq_len, tuning_params['dropout'])
        self.encoder_blocks = torch.nn.ModuleList([
            encoder.EncoderChunk(device, embedding_dimension, tuning_params['n_heads'], tuning_params['mlp_factor'], tuning_params['dropout'], use_rope=False)
            for _ in range(tuning_params['n_blocks'])
        ])

    def split_sequence(self, sequence, max_length):
        chunks = []
        for i in range(0, sequence.size(1), max_length):
            chunk = sequence[:, i:i + max_length]
            # if chunk.size(1) < max_length:
            #     padding = max_length - chunk.size(1)
            #     chunk = torch.nn.functional.pad(chunk, (0, padding), "constant", 0)
            chunks.append(chunk)
        return chunks
    
    def forward(self, x, mask):

        x = self.embedding(x)
        x = self.positional_encoding(x)

        if x.size(1) > self.max_seq_len:
            chunks = self.split_sequence(x, self.max_seq_len)
            mask_chunks = self.split_sequence(mask, self.max_seq_len)
            
            chunk_outputs = []
            for chunk, mask_chunk in zip(chunks, mask_chunks):
                for block in self.encoder_blocks:
                    chunk_encoder = block(chunk, mask_chunk)
                chunk_outputs.append(chunk_encoder)
            
            x = torch.cat(chunk_outputs, dim=1)
        
        else: 
            i = 0
            for block in self.encoder_blocks:
                x = block(x, mask)
                i = i + 1
        return x
    
class RegressionBlock(torch.nn.Module):
    def __init__(self, tuning_params):
        super(RegressionBlock, self).__init__()
        embedding_dimension = int(tuning_params['n_heads'] * tuning_params['d_k'])
        self.pooling_layer  = Pooling_Transformer_output()
        self.dropout = Dropout(tuning_params['dropout2'])
        self.linear  = Linear(in_features=embedding_dimension, out_features=1)
        self.act = get_activation_func(tuning_params['activation'])

    def forward(self, x):
        x = self.pooling_layer(x)
        x = self.dropout(x)
        x = self.linear(x)
        x = self.act(x)
        return x

class EnsembledModel(torch.nn.Module):
    def __init__(self, device, src_vocab_size, seq_lens, max_seq_len, tuning_params):
        super(EnsembledModel, self).__init__()
        self.src_vocab_size = src_vocab_size
        self.tuning_params  = tuning_params
        self.max_seq_len = max_seq_len

        self.transformer_chr1 = TransformerSNP(device, src_vocab_size, seq_lens[0], max_seq_len, tuning_params)
        self.transformer_chr2 = TransformerSNP(device, src_vocab_size, seq_lens[1], max_seq_len, tuning_params)
        self.transformer_chr3 = TransformerSNP(device, src_vocab_size, seq_lens[2], max_seq_len, tuning_params)
        self.transformer_chr4 = TransformerSNP(device, src_vocab_size, seq_lens[3], max_seq_len, tuning_params)
        self.transformer_chr5 = TransformerSNP(device,src_vocab_size, seq_lens[4], max_seq_len, tuning_params)
        # seq_len = np.sum(seq_lens)
        # self.transformer_block = TransformerSNP(src_vocab_size, seq_len, tuning_params)
        self.regression_block = RegressionBlock(tuning_params)
    
    # create mask to ignore Padding token
    def create_mask_list(self, list_X):
        mask_all_chr = []
        for xi in list_X:
            mask = (xi != 0).int()
            final_mask = mask[:, np.newaxis, np.newaxis, :]
            mask_all_chr.append(final_mask)
        return mask_all_chr
    
    def create_mask(self, x):
        mask = (x != 0).int()
        final_mask = mask[:, np.newaxis, np.newaxis, :]
        return final_mask
    
    # concatenate each chromosome (output of Transformer) into 1 input for Regression block
    def concatenate_chr(self, args):
        output = torch.cat(args, 1)
        return output
    
    def forward(self, list_X):

        mask_X = self.create_mask_list(list_X)
        
        # for i in range(len(mask_X)):
        #     print('  + Masking Chr{}: list_X[{}]={} mask_X[{}]={}'.format(i, i, list_X[i].shape, i, mask_X[i].shape))

        # input_X = self.concatenate_chr([list_X[0], list_X[1], list_X[2], list_X[3], list_X[4]])
        # input_X = list_X[0]
        # input_mask = self.create_mask(input_X)
        # input_mask = self.create_mask(list_X[0])
        
        # print("[EnsembledModel] foward: input_X shape={}, input_mask shape={}".format(input_X.shape, input_mask.shape))
        # for i in range(input_X.shape[0]):
        #     print("  + Sample {}: shape={}, {}".format(i, input_X[i].shape, input_X[i]))

        transformer_output1 = self.transformer_chr1(list_X[0], mask_X[0])
        transformer_output2 = self.transformer_chr2(list_X[1], mask_X[1])
        transformer_output3 = self.transformer_chr3(list_X[2], mask_X[2])
        transformer_output4 = self.transformer_chr4(list_X[3], mask_X[3])
        transformer_output5 = self.transformer_chr5(list_X[4], mask_X[4])
        # transformer_output = self.transformer_block(input_X, input_mask)
        # print("  + transformer_output: shape={}".format(transformer_output.shape))

        concatenated_output = self.concatenate_chr([transformer_output1, transformer_output2, transformer_output3, transformer_output4, transformer_output5])
        # concatenated_output = concatenated_output.detach_()
        # transformer_output = transformer_output.detach_()
        # print("  + transformer_output detach: shape={}".format(transformer_output.shape))

        regression_output  = self.regression_block(concatenated_output)
        # regression_output  = self.regression_block(transformer_output)
        # print("  + regression_output: shape={}".format(regression_output.shape))

        return regression_output
    
# ==============================================================
# Define training and validation loop
# ==============================================================
# Function to train the model for one epoch
def train_one_epoch(model, train_loader, loss_function, optimizer, device):
    
    model.train()
    
    # iterate through the train loader
    for i, (inputs, targets) in enumerate(train_loader):
        inputs  = [in_data.to(device) for in_data in inputs]
        targets = targets.to(device)

        # forward pass
        pred_outputs = model(inputs)

        # calculate training loss
        loss_training = loss_function(pred_outputs, targets)
        # print('\t [train_one_epoch] train_loss={:.5f}'.format(loss_training))

        # backward pass and optimization
        optimizer.zero_grad()
        loss_training.backward()
        optimizer.step()

# Function to validate the model for one epoch
def validate_one_epoch(model, val_loader, loss_function, device):

    # arrays for tracking eval results
    avg_loss = 0.0
    arr_val_losses = []

    # evaluate the trained model
    model.eval()
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(val_loader):
            inputs  = [in_data.to(device) for in_data in inputs]
            targets = targets.to(device)
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            arr_val_losses.append(loss.item())
    
    # calculate average validation loss
    avg_loss = np.average(arr_val_losses)
    # print('\t [validate_one_epoch] val_avg_loss={:.5f}'.format(avg_loss))

    return avg_loss

# Function to make predictions using the given model
def predict(model, val_loader, device):
    model.eval()
    predictions = None
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(val_loader):
            inputs  = [in_data.to(device) for in_data in inputs]
            targets = targets.to(device)
            outputs = model(inputs)

            # concatenate the predictions
            predictions = torch.clone(outputs) if predictions is None else torch.cat((predictions, outputs))

    # convert predictions to numpy array based on device
    if device == torch.device('cpu'):
        ret_output = predictions.detach().numpy()
    else:
        ret_output = predictions.cpu().detach().numpy()
    
    return ret_output

# Function to train model on train loader and evaluate on validation loader
def train_val_loop(model, training_params, tuning_params, X_train_list, y_train, X_val_list, y_val, device):

    # transform data to tensor format
    list_tensor_X_train = [torch.from_numpy(item).long() for item in X_train_list]
    list_tensor_X_val = [torch.from_numpy(item).long() for item in X_val_list]
    tensor_y_train = torch.Tensor(y_train)
    tensor_y_val = torch.Tensor(y_val)
    
    # squeeze y to get suitable y dims for training Transformer
    tensor_y_train, tensor_y_val = tensor_y_train.view(len(tensor_y_train),1), tensor_y_val.view(len(tensor_y_val),1)

    # create the list dataset
    train_dataset = ListTensorDataset(list_tensor_X_train, tensor_y_train)
    val_dataset   = ListTensorDataset(list_tensor_X_val, tensor_y_val)
    
    # define data loaders for training and validation data
    train_loader = DataLoader(train_dataset, batch_size=training_params['batch_size'], shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=training_params['batch_size'], shuffle=False)

    # define loss function and optimizer
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=tuning_params['learning_rate'], weight_decay=tuning_params['weight_decay'])
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=tuning_params['lr_decay'])

    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.data) # weight, bias

    # track the best loss value and best model
    best_model = copy.deepcopy(model)
    best_loss  = None

    # track the epoch with best values
    epochs_no_improvement = 0
    early_stopping_point = None

    # training loop over epochs
    num_epochs = training_params['num_epochs']
    early_stop_patience = tuning_params['early_stop']
    for epoch in range(num_epochs):

        train_one_epoch(model, train_loader, loss_function, optimizer, device)
        val_loss = validate_one_epoch(model, val_loader, loss_function, device)
        # Step the scheduler to adjust the learning rate
        # scheduler.step()
        
        # check if the current validation loss is the best observed so far
        # if current val loss is not the best, increase the count of epochs with no improvement
        if best_loss == None or val_loss < best_loss:
            best_loss = val_loss
            best_model = copy.deepcopy(model)
            epochs_no_improvement = 0
        else:
            epochs_no_improvement += 1
        
        print('Epoch {}/{}: current_loss={:.5f} | best_loss={:.5f}'.format(epoch, num_epochs, val_loss, best_loss))

        # check if early stopping criteria are met
        # if the current epoch is greater than or equal to 20 
        # and epochs with no improvement surpass early stopping patience
        if epoch >= 20 and epochs_no_improvement >= early_stop_patience:
            # set the early stopping point
            early_stopping_point = epoch - early_stop_patience
            print("Stopped at epoch " + str(epoch) + "| " + "Early stopping point = " + str(early_stopping_point))
            # predict using the best model 
            model = best_model
            y_pred = predict(model, val_loader, device)
            return y_pred, early_stopping_point
    
    # return the best predicted values
    y_pred = predict(best_model, val_loader, device)
    return y_pred, early_stopping_point

# ==============================================================
# Define objective function for tuning hyperparameters
# ==============================================================
def objective(trial, list_X_train, src_vocab_size, y, data_variants, training_params_dict, avg_stop_epochs, device):
    
    # for tuning parameters
    tuning_params_dict = {
        'learning_rate': trial.suggest_categorical('learning_rate', [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]), 
        # 'weight_decay': trial.suggest_categorical('weight_decay', [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]),
        'weight_decay': trial.suggest_float('weight_decay', 1e-8, 1e-2),
        # 'lr_decay': trial.suggest_float('lr_decay', 0.95, 0.99, step=0.01),
        'activation': trial.suggest_categorical('activation', ['LeakyReLU', 'ReLU', 'Tanh', 'GELU']),
        'early_stop': trial.suggest_int("early_stop", 5, 20, step=5),

        # 'max_seq_len': trial.suggest_int('max_seq_len', 384, 512, step=64),
        'n_blocks': trial.suggest_int("n_blocks", 2, 8, step=1),
        'n_heads': trial.suggest_int("n_heads", 2, 8, step=1),
        # 'd_k': trial.suggest_categorical('d_k', [16, 32, 64]),
        'd_k':  trial.suggest_int('d_k', 8, 96, step=4),
        'mlp_factor': trial.suggest_int("mlp_factor", 2, 6, step=1),

        # 'pca': trial.suggest_float('pca', 0.85, 0.95, step=0.05),
        'dropout2': trial.suggest_float('dropout2', 0.1, 0.5, step=0.05),
        'dropout': trial.suggest_float('dropout', 0.1, 0.5, step=0.05)
    }

     # extract preprocessed data variants for tuning
    minmax_scaler_mode = data_variants[0]
    # standard_scaler_mode = data_variants[1]
    # pca_fitting_mode = data_variants[2]

    # log early stopping point at each fold
    early_stopping_points = []
    
    # iterate for training and tuning
    print('----------------------------------------------')
    print("Params for Trial " + str(trial.number))
    print(trial.params)
    print('----------------------------------------------\n')

    # tracking the results
    first_obj_values = []
    second_obj_values = []

    # forl cross-validation kfolds, default = 5 folds
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        
    # Split the data into train and validation sets for all chromosomes at once
    all_folds = [kfold.split(X_chr) for X_chr in list_X_train]

    # Pipeline-like structure
    for fold, fold_indices in enumerate(zip(*all_folds), start=1):
        X_train_list = []
        X_val_list = []

        print("Preprocessing Fold No.{}: ".format(fold))
        for i, (train_ids, val_ids) in enumerate(fold_indices, start=1):
            
            X_chr = list_X_train[i-1]
            X_train, X_val = X_chr[train_ids], X_chr[val_ids]
            y_train, y_val = y[train_ids], y[val_ids]
            X_train_list.append(X_train)
            X_val_list.append(X_val)
        
        # preprocessing data
        if minmax_scaler_mode == True:
            y_train_scale, y_val_scale, minmax_scaler = preprocess_mimax_scaler(y_train, y_val)
        # if standard_scaler_mode == True:
        #     X_train, X_val = preprocess_standard_scaler(X_train, X_val)
        # if pca_fitting_mode == True:
        #     X_train, X_val = decomposition_PCA(X_train, X_val, tuning_params=tuning_params_dict)

        # get the max sequence length of each chrobosome
        seq_lens = []
        for i in range(len(X_train_list)):
            seq_lens.append(X_train_list[i].shape[1])
        
        max_seq_len= int(512)

        set_seeds(seed=int(42+fold))
        try:
            model = EnsembledModel(device, src_vocab_size, seq_lens, max_seq_len, tuning_params=tuning_params_dict).to(device)

        except Exception as err:
            print('Trial failed. Error in model creation, {}'.format(err))
            raise optuna.exceptions.TrialPruned()

        # call training model over each fold
        try:
            y_pred, stopping_point = train_val_loop(model, training_params_dict, tuning_params_dict,
                                     X_train_list, y_train_scale, X_val_list, y_val_scale, device)
            y_pred = minmax_scaler.inverse_transform(y_pred)
            
            # record the early-stopping points
            if stopping_point is not None:
                early_stopping_points.append(stopping_point)
            else:
                early_stopping_points.append(training_params_dict['num_epochs'])

            # calculate objective value
            obj_value1 = sklearn.metrics.mean_squared_error(y_true=y_val, y_pred=y_pred)
            obj_value2 = sklearn.metrics.explained_variance_score(y_true=y_val, y_pred=y_pred)
            print('Explained variance = {:.5f} | MSE loss = {:.5f}'.format(obj_value2, obj_value1))

            # report pruned values
            trial.report(value=obj_value2, step=fold)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            
            # accumulate the obj val losses
            first_obj_values.append(obj_value1)
            second_obj_values.append(obj_value2)

        # for pruning the tuning process
        except (RuntimeError, TypeError, ValueError) as exc:
            print(exc)
            if 'out of memory' in str(exc):
                print('Out of memory')
            else:
                print('Trial failed. Error in optim loop.')
            raise optuna.exceptions.TrialPruned()
    
    # return the average val loss
    current_val_loss = float(np.mean(first_obj_values))
    current_val_expv = float(np.mean(second_obj_values))

    # average value of early stopping points of all innerfolds for refitting of final model
    early_stopping_point = int(np.mean(early_stopping_points))
    print('----------------------------------------------')
    print("Average early_stopping_point: {}| avg_exp_var={:.5f}| avg_loss={:.5f}".format(early_stopping_point, current_val_expv, current_val_loss))
    print('----------------------------------------------\n')

    # try to return avg stop epochs for a specific trial 
    avg_stop_epochs[trial.number] = early_stopping_point

    # return current_val_expv
    return current_val_loss

# ==============================================================
# Call tuning function
# ==============================================================
def tuning_splitchr_transformer(datapath, list_X_train, src_vocab_size, y, data_variants, training_params_dict, device):

    # set seeds for reproducibility
    set_seeds()

    # for tracking the tuning information
    minmax = '_minmax' if data_variants[0] == True else ''
    # standard = '_standard' if data_variants[1] == True else ''
    # pcafitting = '_pca' if data_variants[2] == True else ''
    # pheno = str(data_variants[3])

    # create a list to keep track all stopping epochs during a hyperparameter search
    # because in some cases, the trial is pruned so the stopping epochs default = num_epochs
    avg_stopping_epochs = [training_params_dict['num_epochs']] * training_params_dict['num_trials']

    # create an optuna tuning object, num trials default = 100
    num_trials = training_params_dict['num_trials']
    study = optuna.create_study(
        study_name='transformer'+'mseloss_'+'data',
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=training_params_dict['optunaseed']),
        pruner=optuna.pruners.PercentilePruner(percentile=training_params_dict['percentile'], n_min_trials=training_params_dict['min_trials'])
    )
    
    # searching loop with objective tuning
    study.optimize(lambda trial: objective(trial, list_X_train, src_vocab_size, y, data_variants, training_params_dict, avg_stopping_epochs, device), n_trials=num_trials)
    set_seeds()

    # get early stopping of the best trial
    num_avg_stop_epochs = avg_stopping_epochs[study.best_trial.number]
    
    # print statistics after tuning
    print("Optuna study finished, study statistics:")
    print("  Finished trials: ", len(study.trials))
    print("  Pruned trials: ", len(study.get_trials(states=(optuna.trial.TrialState.PRUNED,))))
    print("  Completed trials: ", len(study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))))
    print("  Best Trial: ", study.best_trial.number)
    print("  Value: ", study.best_trial.value)
    print("  AVG stopping: ", num_avg_stop_epochs)
    print("  Params: ")
    for key, value in study.best_trial.params.items():
        print("    {}: {}".format(key, value))
    print('----------------------------------------------\n')

    best_params = study.best_trial.params
    best_params['avg_epochs'] = num_avg_stop_epochs
    print('Check best params: {}'.format(best_params))

    # record best parameters to file
    # with open(f"./tuned_transformer_" + "pheno" + ".json", 'w') as fp:
    #     json.dump(best_params, fp)

    return best_params

# ==============================================================
# Evaluation the performance on test set
# ==============================================================
def evaluate_result_splitchr_transformer(datapath, list_X_train, src_vocab_size, y_train, list_X_test, y_test, best_params, data_variants, device):

    # set seeds for reproducibility
    set_seeds()
    
    # for tracking the tuning information
    minmax = '_minmax' if data_variants[0] == True else ''
    # standard = '_standard' if data_variants[1] == True else ''
    # pcafitting = '_pca' if data_variants[2] == True else ''
    # pheno = str(data_variants[3])

    # extract preprocessed data variants for tuning
    minmax_scaler_mode = data_variants[0]
    # standard_scaler_mode = data_variants[1]
    # pca_fitting_mode = data_variants[2]

    # preprocessing data
    if minmax_scaler_mode == 1: # minmax scaler
        y_train_scale, y_test_scale, minmax_scaler = preprocess_mimax_scaler(y_train, y_test)
    # if standard_scaler_mode == 1: # standard scaler
    #     X_train, X_test = preprocess_standard_scaler(X_train, X_test)
    # if pca_fitting_mode == 1: # pca fitting
    #     X_train, X_test = decomposition_PCA(X_train, X_test, best_params['pca'])

    # extract training and tuned parameters
    batch_size = 32
    num_epochs = best_params['avg_epochs']
    learning_rate = best_params['learning_rate']
    momentum = best_params['weight_decay']

    # get the max sequence length of each chrobosome
    seq_lens = []
    for i in range(len(list_X_train)):
        seq_lens.append(list_X_train[i].shape[1])

    max_seq_len= int(512)
    # create model
    model = EnsembledModel(device, src_vocab_size, seq_lens, max_seq_len, tuning_params=best_params).to(device)

    # transform data to tensor format
    list_tensor_X_train = [torch.from_numpy(item).long() for item in list_X_train]
    list_tensor_X_test = [torch.from_numpy(item).long() for item in list_X_test]
    tensor_y_train = torch.Tensor(y_train_scale)
    tensor_y_test = torch.Tensor(y_test_scale)

    # squeeze y for training Transformer to tensor
    tensor_y_train, tensor_y_test = tensor_y_train.view(len(y_train_scale),1), tensor_y_test.view(len(y_test_scale),1)

    # create the list dataset
    train_dataset = ListTensorDataset(list_tensor_X_train, tensor_y_train)
    test_dataset   = ListTensorDataset(list_tensor_X_test, tensor_y_test)

    # define data loaders for training and testing data
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader   = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # define loss function and optimizer
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=momentum)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=best_params['lr_decay'])

    # training loop over epochs
    for epoch in range(num_epochs):
        train_one_epoch(model, train_loader, loss_function, optimizer, device)
        # Step the scheduler to adjust the learning rate
        # scheduler.step()
    
    # predict result test 
    y_pred = predict(model, test_loader, device)
    y_pred = minmax_scaler.inverse_transform(y_pred)

    # collect mse, r2, explained variance
    test_mse = sklearn.metrics.mean_squared_error(y_true=y_test, y_pred=y_pred)
    test_exp_variance = sklearn.metrics.explained_variance_score(y_true=y_test, y_pred=y_pred)
    test_r2 = sklearn.metrics.r2_score(y_true=y_test, y_pred=y_pred)
    test_mae = sklearn.metrics.mean_absolute_error(y_true=y_test, y_pred=y_pred)

    print('--------------------------------------------------------------')
    print('Test Transformer results: avg_loss={:.4f}, avg_expvar={:.4f}, avg_r2score={:.4f}, avg_mae={:.4f}'.format(test_mse, test_exp_variance, test_r2, test_mae))
    print('--------------------------------------------------------------')

    return test_exp_variance