import json 
import sklearn
import argparse
import numpy as np
import warnings
import time

from preprocess.snp_splitting import *
from preprocess.snp_tokenize import *
from preprocess.snp_embed import *

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold

from torch.optim import SGD, Adam
from torch.profiler import profile, record_function, ProfilerActivity
# from torchinfo import summary
from fvcore.nn import FlopCountAnalysis, parameter_count_table
# from torch.utils.flop_counter import FlopCounterMode
from torch.utils.data import DataLoader, Dataset

from model.transformer_allchr import EnsembledModel as Transformer_Allchr
from model.transformer_splitchr import EnsembledModel as Transformer_Splitchr

from model.mlp_mixer_allchr import MixerModel as MLPMixer_Allchr
from model.mlp_mixer_splitchr import MixerModel as MLPMixer_Splitchr

from model.hyper_mixer_allchr import MixerModel as HyperMixer_Allchr
from model.hyper_mixer_splitchr import MixerModel as HyperMixer_Splitchr

from model.cnn_transformer_allchr import CNN_Transformer as CNN_Transformer_Allchr
from model.cnn_transformer_splitchr import CNN_Transformer as CNN_Transformer_Splitchr

from model.transformer_infinite_allchr import EnsembledModel as Transformer_Inf_Allchr
from model.transformer_infinite_splitchr import EnsembledModel as Transformer_Inf_Splitchr

from model.word2vec_transformer_splitchr import EnsembledModel as Transformer_W2V_Splitchr
from model.word2vec_hyper_mixer_splitchr import MixerModel as HyperMixer_W2V_Splitchr

# Suppress specific warnings
warnings.filterwarnings("ignore")

class ListTensorDataset(Dataset):
    def __init__(self, X_list, y_train):
        self.X_list = X_list
        self.y_train = y_train

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        # return a list containing all tensors in the batch and the corresponding label
        return [X[idx] for X in self.X_list], self.y_train[idx]

def preprocess_mimax_scaler(y_train, y_val):
    minmax_scaler = MinMaxScaler()
    y_train = np.expand_dims(y_train, axis=1)
    y_val = np.expand_dims(y_val, axis=1)
    y_train_scaled = minmax_scaler.fit_transform(y_train)
    y_val_scaled   = minmax_scaler.transform(y_val)
    return y_train_scaled, y_val_scaled, minmax_scaler

def train_one_epoch(model, train_loader, loss_function, optimizer, device):
    
    model.train()
    
    # iterate through the train loader
    for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        # forward pass
        pred_outputs = model(inputs)

        # calculate training loss
        loss_training = loss_function(pred_outputs, targets)
        # print('\t [train_one_epoch] train_loss={:.5f}'.format(loss_training))

        # backward pass and optimization
        optimizer.zero_grad()
        loss_training.backward()
        optimizer.step()

def train_one_epoch_splitchr(model, train_loader, loss_function, optimizer, device):
    
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


def predict(model, val_loader, device):
    model.eval()
    predictions = None
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            # concatenate the predictions
            predictions = torch.clone(outputs) if predictions is None else torch.cat((predictions, outputs))

    # convert predictions to numpy array based on device
    if device == torch.device('cpu'):
        ret_output = predictions.detach().numpy()
    else:
        ret_output = predictions.cpu().detach().numpy()
    
    return ret_output

def predict_splitchr(model, val_loader, device):
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

def count_all_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_trainabled_parameters(model):
    return sum(p.numel() for p in model.parameters())

def log_gpu_memory():
    allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # Convert bytes to MB
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)  # Convert bytes to MB
    print(f"GPU Memory Allocated: {allocated:.2f} MB")
    print(f"GPU Memory Reserved: {reserved:.2f} MB")

def evaluate_result_regular(datapath, model_name, X_train, src_vocab_size, y_train, X_test, y_test, best_params, data_variants, device):
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
    max_len = X_train.shape[1]
   
    # create model
    if model_name == 'MLPMixer':
        print('Evaluate Performance: {}'.format(model_name))
        max_seq_lens = max_len
        model = MLPMixer_Allchr(src_vocab_size, max_seq_lens, tuning_params=best_params).to(device)
    elif model_name == 'Test1_CNN':
        print('Evaluate Performance: {}'.format(model_name))
        max_seq_lens = max_len
        model = CNN_Transformer_Allchr(device, src_vocab_size, max_seq_lens, tuning_params=best_params).to(device)
    elif model_name == 'Test1_Transformer':
        print('Evaluate Performance: {}'.format(model_name))
        max_seq_lens = int(512)
        model = Transformer_Allchr(device, src_vocab_size, max_len, max_seq_lens, tuning_params=best_params).to(device)
    elif model_name == 'Test2_Infinite_Transformer':
        print('Evaluate Performance: {}'.format(model_name))
        max_seq_lens = max_len
        model = Transformer_Inf_Allchr(src_vocab_size, max_seq_lens, device, tuning_params=best_params).to(device)
    else:
        print('Evaluate Performance: {}'.format(model_name))
        max_seq_lens = max_len
        model = HyperMixer_Allchr(src_vocab_size, max_seq_lens, tuning_params=best_params).to(device)

    # transform data to tensor format
    tensor_X_train, tensor_y_train = torch.LongTensor(X_train), torch.Tensor(y_train_scale)
    tensor_X_test, tensor_y_test = torch.LongTensor(X_test), torch.Tensor(y_test_scale)

    # squeeze y for training Transformer to tensor
    tensor_y_train, tensor_y_test = tensor_y_train.view(len(y_train_scale),1), tensor_y_test.view(len(y_test_scale),1)

    # define data loaders for training and testing data
    train_loader = DataLoader(dataset=list(zip(tensor_X_train, tensor_y_train)), batch_size=batch_size, shuffle=True)
    test_loader   = DataLoader(dataset=list(zip(tensor_X_test, tensor_y_test)), batch_size=batch_size, shuffle=False)

    # define loss function and optimizer
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=momentum)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=best_params['lr_decay'])

    # Clear cache and reset peak memory stats before the forward pass
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    total_time = 0  # Initialize total time
    # training loop over epochs
    for epoch in range(num_epochs):
        start_time = time.time()  # Start timer
        if epoch < 1:
            # with profile(activities=[ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:
            #     train_one_epoch(model, train_loader, loss_function, optimizer, device)
            # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=5))
            train_one_epoch(model, train_loader, loss_function, optimizer, device)
            print('1. Log memory usage for one epoch:')
            log_gpu_memory()
        else:
            train_one_epoch(model, train_loader, loss_function, optimizer, device)
        
        epoch_time = time.time() - start_time  # End timer
        total_time += epoch_time  # Accumulate total time
        # print(f"Epoch {epoch+1} took {epoch_time:.2f} seconds")
    
    # Calculate the average time per epoch
    average_time = total_time / num_epochs
    print(f"2. Average time per epoch: {average_time:.2f} seconds")
    
    # Calculating the number of parameters
    num_all_params = count_all_parameters(model)
    num_trainabled_params = count_trainabled_parameters(model)
    print(f"3.1. Number of all parameters: {num_all_params}")
    print(f"3.2. Number of only num_trainabled_params parameters: {num_trainabled_params}")
    
    # Calculating FLOPs using fvcore
    # inputs = next(iter(train_loader))[0].to(device)
    # flops = FlopCountAnalysis(model, inputs)
    # print(f"FLOPs: {flops.total()}")

    # inputs = next(iter(train_loader))[0].to(device) # Fetch a sample input from your DataLoader
    # model = model.to(device)
    # # Use FlopCounterMode to calculate FLOPs
    # with FlopCounterMode(model) as flops_mode:
    #     _ = model(inputs)  # Perform a forward pass
    #     flops = flops_mode.get_total_flops()
    # print(f"Floating point per second FLOPs: {flops}")

    # Printing detailed parameter count table
    print("Parameter Count Table:")
    print(parameter_count_table(model))


    # predict result test
    # with profile(activities=[ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:
    # print("Prediction Profile:")
    # print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=5))
    y_pred = predict(model, test_loader, device)
    print('Log memory usage for predict:')
    log_gpu_memory()

    # convert the predicted y values
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

def evaluate_result_splitchr(datapath, model_name, list_X_train, src_vocab_size, y_train, list_X_test, y_test, best_params, data_variants, device):
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
    max_len = []
    for i in range(len(list_X_train)):
        max_len.append(list_X_train[i].shape[1])
   
    # create model
    if model_name == 'MLPMixer':
        print('Evaluate Performance: {}'.format(model_name))
        max_seq_lens = max_len
        model = MLPMixer_Splitchr(src_vocab_size, max_seq_lens, tuning_params=best_params).to(device)
    elif model_name == 'Test1_CNN':
        print('Evaluate Performance: {}'.format(model_name))
        max_seq_lens = max_len
        model = CNN_Transformer_Splitchr(device, src_vocab_size, max_seq_lens, tuning_params=best_params).to(device)
    elif model_name == 'Transformer':
        print('Evaluate Performance: {}'.format(model_name))
        max_seq_lens = int(512)
        model = Transformer_Splitchr(device, src_vocab_size, max_len, max_seq_lens, tuning_params=best_params).to(device)
    elif model_name == 'Test2_Infinite_Transformer':
        print('Evaluate Performance: {}'.format(model_name))
        max_seq_lens = max_len
        model = Transformer_Inf_Splitchr(device, src_vocab_size, max_seq_lens, tuning_params=best_params).to(device)
    else:
        print('Evaluate Performance: {}'.format(model_name))
        max_seq_lens = max_len
        model = HyperMixer_Splitchr(src_vocab_size, max_seq_lens, tuning_params=best_params).to(device)

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

    # Clear cache and reset peak memory stats before the forward pass
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    total_time = 0  # Initialize total time
    # training loop over epochs
    for epoch in range(num_epochs):
        start_time = time.time()  # Start timer
        if epoch < 1:
            # with profile(activities=[ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:
            #     train_one_epoch(model, train_loader, loss_function, optimizer, device)
            # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=5))
            train_one_epoch_splitchr(model, train_loader, loss_function, optimizer, device)
            print('1. Log memory usage for one epoch:')
            log_gpu_memory()
        else:
            train_one_epoch_splitchr(model, train_loader, loss_function, optimizer, device)
        
        epoch_time = time.time() - start_time  # End timer
        total_time += epoch_time  # Accumulate total time
        # print(f"Epoch {epoch+1} took {epoch_time:.2f} seconds")
    
    # Calculate the average time per epoch
    average_time = total_time / num_epochs
    print(f"2. Average time per epoch: {average_time:.2f} seconds")
    
    # Calculating the number of parameters
    num_all_params = count_all_parameters(model)
    num_trainabled_params = count_trainabled_parameters(model)
    print(f"3.1. Number of all parameters: {num_all_params}")
    print(f"3.2. Number of only num_trainabled_params parameters: {num_trainabled_params}")

    # Printing detailed parameter count table
    print("Parameter Count Table:")
    print(parameter_count_table(model))

    # predict result test
    # with profile(activities=[ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:
    # print("Prediction Profile:")
    # print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=5))
    y_pred = predict_splitchr(model, test_loader, device)
    print('Log memory usage for predict:')
    log_gpu_memory()

    # convert the predicted y values
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

if __name__ == '__main__':

    # ----------------------------------------------------
    # Process the arguments
    # ----------------------------------------------------
    parser = argparse.ArgumentParser()

    parser.add_argument("-par", "--best_param", type=str,
                        default='./best_params/best_param_testtransformer49_nonoverlap_regular.json',
                        help="Path to the input file containing best params")
    
    parser.add_argument("-ddi", "--data_dir", type=str,
                        default='./data/easypheno_indices/ratio_7_3/pheno1',
                        help="Path to the directory of data")
    
    parser.add_argument("-mod", "--model_type", type=str,
                        default='Chromosomesplit',
                        help="Type of model for training the phenotype predictor: Chromosomesplit, Regular, Augmentation_Chromosome")
    
    parser.add_argument("-llm", "--model_name", type=str,
                        default='HyperMixer',
                        help="Name of LL model for training the phenotype prediction: Transformer, MLPMixer, HyperMixer, CNN_Transformer, \
                            Test_CNN, Test1_Transformer, Test2_Infinite_Transformer")
    
    parser.add_argument("-kme", "--kmer", type=int,
                        default=6,
                        help="The number of kmer to tokenize sequence X")
    
    parser.add_argument("-tok", "--tokenize_type", type=str,
                        default='overlap',
                        help="Type of tokenizing methods: overlap, non_overlap, nuc")
    
    parser.add_argument("-emb", "--embedding_type", type=str,
                        default='kmer_nonoverlap',
                        help="Type of Embedding methods: BPE, word2vec, kmer_nonoverlap, kmer_overlap")

    parser.add_argument("-voc", "--vocab_size_bpe", type=int,
                        default=4096,
                        help="The vocab size when using BPE: 1024, 2048, 4096")
    
    parser.add_argument("-min", "--minmax", type=int,
                        default=1,
                        help="Nomalizing y with min-max scaler")
    
    parser.add_argument("-sta", "--standa", type=int,
                        default=0,
                        help="Nomalizing X with min-max scaler")
    
    parser.add_argument("-pca", "--pcafit", type=int,
                        default=0,
                        help="Reducing and fitting X with PCA")
    
    parser.add_argument("-dat", "--dataset", type=int,
                        default=1,
                        help="The set of data using for training")
    
    parser.add_argument("-rat", "--ratio", type=int,
                        default=0.7,
                        help="The ratio o train_data when splitting: 0.7; 0.8; 0.9")
    
    parser.add_argument("-gpu", "--gpucuda", type=int,
                        default=0,
                        help="Training the model on GPU")
    
    # parse argument values
    args = vars(parser.parse_args())
    best_param_file = args['best_param']
    datapath = args["data_dir"]
    model_type = args["model_type"]
    model_name = args["model_name"]
    kmer = args["kmer"]
    # tokenize_type = args["tokenize_type"]
    embedding_type = args["embedding_type"]
    vocab_size_bpe = args['vocab_size_bpe']
    minmax_scale = args["minmax"]
    standa_scale = args["standa"]
    pca_fitting  = args["pcafit"]
    dataset = args["dataset"]
    ratio = args["ratio"]
    gpucuda = args["gpucuda"]

    # check available GPUs, if not, run on CPUs
    dev = "cpu"
    if args["gpucuda"] >= 1 and torch.cuda.is_available(): 
        print("GPU CUDA available, using GPU for training the models.")
        dev = "cuda:" + str(args["gpucuda"]-1)
    else:
        print("GPU CUDA not available, using CPU instead.")
    device = torch.device(dev)

    # different types of scaling data
    data_variants = [minmax_scale, standa_scale, pca_fitting, dataset]

    # parameters for the training process
    training_params_dict = {
        'num_trials': 60,
        'min_trials': 30,
        'percentile': 65,
        'optunaseed': 42,
        'num_epochs': 100,
        'early_stop': 20,
        'batch_size': 32
    }
    
    # reading the data from the file 
    with open(best_param_file) as f: 
        best_params_input = f.read()
        best_params_input = best_params_input.replace("'", "\"")
    best_params = json.loads(best_params_input)

    print('----------------------------------------------------')
    print('Best parameters:')
    print('----------------------------------------------------')
    print(best_params)
    print('')

    print('----------------------------------------------------')
    print('Run evaluating the model:')
    print('----------------------------------------------------')
    if model_type == 'Chromosomesplit':
        print('Type: Chromosomesplit')
        if dataset == 4:
            X_chr1_train, X_chr2_train, X_chr3_train, X_chr4_train, X_chr5_train = split_into_chromosome_train_realworld(datapath, dataset, ratio)
            X_chr1_test, X_chr2_test, X_chr3_test, X_chr4_test, X_chr5_test = split_into_chromosome_test_realworld(datapath, dataset, ratio)
            y_train, y_test= load_split_y_data(datapath, dataset, ratio)
        else:
            X_chr1_train, X_chr2_train, X_chr3_train, X_chr4_train, X_chr5_train = split_into_chromosome_train(datapath, dataset, ratio)
            X_chr1_test, X_chr2_test, X_chr3_test, X_chr4_test, X_chr5_test = split_into_chromosome_test(datapath, dataset, ratio)
            y_train, y_test= load_split_y_data(datapath, dataset, ratio)

        if embedding_type == 'kmer_nonoverlap':
            print('Embedding: {} | kmer={}'.format(embedding_type, kmer))

            X_chr1_kmer = seqs2kmer_nonoverlap(X_chr1_train, kmer=kmer)
            X_chr2_kmer = seqs2kmer_nonoverlap(X_chr2_train, kmer=kmer)
            X_chr3_kmer = seqs2kmer_nonoverlap(X_chr3_train, kmer=kmer)
            X_chr4_kmer = seqs2kmer_nonoverlap(X_chr4_train, kmer=kmer)
            X_chr5_kmer = seqs2kmer_nonoverlap(X_chr5_train, kmer=kmer)

            X_chr1_tokenizer = load_kmer_tokenizer(kmer=kmer)

            X_test_chr1_kmer = seqs2kmer_nonoverlap(X_chr1_test, kmer=kmer)
            X_test_chr2_kmer = seqs2kmer_nonoverlap(X_chr2_test, kmer=kmer)
            X_test_chr3_kmer = seqs2kmer_nonoverlap(X_chr3_test, kmer=kmer)
            X_test_chr4_kmer = seqs2kmer_nonoverlap(X_chr4_test, kmer=kmer)
            X_test_chr5_kmer = seqs2kmer_nonoverlap(X_chr5_test, kmer=kmer)

            x1_maxlen = find_max_kmer_length(X_chr1_kmer, X_chr1_tokenizer) #401
            x2_maxlen = find_max_kmer_length(X_chr2_kmer, X_chr1_tokenizer) #283
            x3_maxlen = find_max_kmer_length(X_chr3_kmer, X_chr1_tokenizer) #336
            x4_maxlen = find_max_kmer_length(X_chr4_kmer, X_chr1_tokenizer) #288
            x5_maxlen = find_max_kmer_length(X_chr5_kmer, X_chr1_tokenizer) #361


            embedded_X_chr1 = np.array(encode_sequences(X_chr1_kmer, X_chr1_tokenizer, max_length=x1_maxlen))
            embedded_X_chr2 = np.array(encode_sequences(X_chr2_kmer, X_chr1_tokenizer, max_length=x2_maxlen))
            embedded_X_chr3 = np.array(encode_sequences(X_chr3_kmer, X_chr1_tokenizer, max_length=x3_maxlen))
            embedded_X_chr4 = np.array(encode_sequences(X_chr4_kmer, X_chr1_tokenizer, max_length=x4_maxlen))
            embedded_X_chr5 = np.array(encode_sequences(X_chr5_kmer, X_chr1_tokenizer, max_length=x5_maxlen))
            embedded_X_test_chr1 = np.array(encode_sequences(X_test_chr1_kmer, X_chr1_tokenizer, max_length=x1_maxlen))
            embedded_X_test_chr2 = np.array(encode_sequences(X_test_chr2_kmer, X_chr1_tokenizer, max_length=x2_maxlen))
            embedded_X_test_chr3 = np.array(encode_sequences(X_test_chr3_kmer, X_chr1_tokenizer, max_length=x3_maxlen))
            embedded_X_test_chr4 = np.array(encode_sequences(X_test_chr4_kmer, X_chr1_tokenizer, max_length=x4_maxlen))
            embedded_X_test_chr5 = np.array(encode_sequences(X_test_chr5_kmer, X_chr1_tokenizer, max_length=x5_maxlen))

            list_X_train = [embedded_X_chr1, embedded_X_chr2, embedded_X_chr3, embedded_X_chr4, embedded_X_chr5]
            list_X_test = [embedded_X_test_chr1, embedded_X_test_chr2, embedded_X_test_chr3, embedded_X_test_chr4, embedded_X_test_chr5]

            src_vocab_size = len(X_chr1_tokenizer)

        elif embedding_type == 'kmer_overlap':
            print('Embedding: {} | kmer={}'.format(embedding_type, kmer))

            X_chr1_kmer = seqs2kmer_overlap(X_chr1_train, kmer=kmer)
            X_chr2_kmer = seqs2kmer_overlap(X_chr2_train, kmer=kmer)
            X_chr3_kmer = seqs2kmer_overlap(X_chr3_train, kmer=kmer)
            X_chr4_kmer = seqs2kmer_overlap(X_chr4_train, kmer=kmer)
            X_chr5_kmer = seqs2kmer_overlap(X_chr5_train, kmer=kmer)

            # X_chr1_tokenizer = kmer_embed(X_chr1_kmer, kmer=6)
            X_chr1_tokenizer = load_kmer_tokenizer(kmer=kmer)

            X_test_chr1_kmer = seqs2kmer_overlap(X_chr1_test, kmer=kmer)
            X_test_chr2_kmer = seqs2kmer_overlap(X_chr2_test, kmer=kmer)
            X_test_chr3_kmer = seqs2kmer_overlap(X_chr3_test, kmer=kmer)
            X_test_chr4_kmer = seqs2kmer_overlap(X_chr4_test, kmer=kmer)
            X_test_chr5_kmer = seqs2kmer_overlap(X_chr5_test, kmer=kmer)

            x1_maxlen = find_max_kmer_length(X_chr1_kmer, X_chr1_tokenizer) #401
            x2_maxlen = find_max_kmer_length(X_chr2_kmer, X_chr1_tokenizer) #283
            x3_maxlen = find_max_kmer_length(X_chr3_kmer, X_chr1_tokenizer) #336
            x4_maxlen = find_max_kmer_length(X_chr4_kmer, X_chr1_tokenizer) #288
            x5_maxlen = find_max_kmer_length(X_chr5_kmer, X_chr1_tokenizer) #361


            embedded_X_chr1 = np.array(encode_sequences(X_chr1_kmer, X_chr1_tokenizer, max_length=x1_maxlen))
            embedded_X_chr2 = np.array(encode_sequences(X_chr2_kmer, X_chr1_tokenizer, max_length=x2_maxlen))
            embedded_X_chr3 = np.array(encode_sequences(X_chr3_kmer, X_chr1_tokenizer, max_length=x3_maxlen))
            embedded_X_chr4 = np.array(encode_sequences(X_chr4_kmer, X_chr1_tokenizer, max_length=x4_maxlen))
            embedded_X_chr5 = np.array(encode_sequences(X_chr5_kmer, X_chr1_tokenizer, max_length=x5_maxlen))

            list_X_train = [embedded_X_chr1, embedded_X_chr2, embedded_X_chr3, embedded_X_chr4, embedded_X_chr5]

            embedded_X_test_chr1 = np.array(encode_sequences(X_test_chr1_kmer, X_chr1_tokenizer, max_length=x1_maxlen))
            embedded_X_test_chr2 = np.array(encode_sequences(X_test_chr2_kmer, X_chr1_tokenizer, max_length=x2_maxlen))
            embedded_X_test_chr3 = np.array(encode_sequences(X_test_chr3_kmer, X_chr1_tokenizer, max_length=x3_maxlen))
            embedded_X_test_chr4 = np.array(encode_sequences(X_test_chr4_kmer, X_chr1_tokenizer, max_length=x4_maxlen))
            embedded_X_test_chr5 = np.array(encode_sequences(X_test_chr5_kmer, X_chr1_tokenizer, max_length=x5_maxlen))

            list_X_train = [embedded_X_chr1, embedded_X_chr2, embedded_X_chr3, embedded_X_chr4, embedded_X_chr5]
            list_X_test = [embedded_X_test_chr1, embedded_X_test_chr2, embedded_X_test_chr3, embedded_X_test_chr4, embedded_X_test_chr5]

            src_vocab_size = len(X_chr1_tokenizer)

        if model_name == 'MLPMixer':
            print('Run: MLPMixer | Split chromosome - Embedding {} - Dataset {} - Ratio {}'.format(embedding_type, dataset, ratio))
            evaluate_result_splitchr(datapath, model_name, list_X_train, src_vocab_size, y_train, list_X_test, y_test, best_params, data_variants, device)
        elif model_name == 'Test1_CNN':
            print('Run: Test1_CNN | Split chromosome - Embedding {} - Dataset {} - Ratio {}'.format(embedding_type, dataset, ratio))
            evaluate_result_splitchr(datapath, model_name, list_X_train, src_vocab_size, y_train, list_X_test, y_test, best_params, data_variants, device)
        elif model_name == 'Transformer':
            print('Run: Test1_Transformer | Split chromosome - Embedding {} - Dataset {} - Ratio {}'.format(embedding_type, dataset, ratio))
            evaluate_result_splitchr(datapath, model_name, list_X_train, src_vocab_size, y_train, list_X_test, y_test, best_params, data_variants, device)
        elif model_name == 'Test2_Infinite_Transformer':
            print('Run: Test2_Infinite_Transformer | Split chromosome - Embedding {} - Dataset {} - Ratio {}'.format(embedding_type, dataset, ratio))
            evaluate_result_splitchr(datapath, model_name, list_X_train, src_vocab_size, y_train, list_X_test, y_test, best_params, data_variants, device)
        elif model_name == 'HyperMixer':
            print('Run: Test1_HyperMixer | Split chromosome - Embedding {} - Dataset {} - Ratio {}'.format(embedding_type, dataset, ratio))
            evaluate_result_splitchr(datapath, model_name, list_X_train, src_vocab_size, y_train, list_X_test, y_test, best_params, data_variants, device)

        

    elif model_type == 'Regular':
        print('Type: Regular')
        # prepare dataset
        X_train, y_train, X_test, y_test = load_split_regular_data(datapath, dataset, ratio)

        if embedding_type == 'kmer_nonoverlap':
            print('Embedding: {} | kmer={}'.format(embedding_type, kmer))
            X_kmer = seqs2kmer_nonoverlap(X_train, kmer=kmer)
            X_tokenizer = load_kmer_tokenizer(kmer=kmer)
            X_test_kmer = seqs2kmer_nonoverlap(X_test, kmer=kmer)
            x_maxlen = find_max_kmer_length(X_kmer, X_tokenizer) #401
            embedded_X_train = np.array(encode_sequences(X_kmer, X_tokenizer, max_length=x_maxlen))
            embedded_X_test = np.array(encode_sequences(X_test_kmer, X_tokenizer, max_length=x_maxlen))
            # src_vocab_size = X_chr1_tokenizer.get_vocab_size()
            src_vocab_size = len(X_tokenizer)

        elif embedding_type == 'kmer_overlap':
            print('Embedding: {} | kmer={}'.format(embedding_type, kmer))
            X_kmer = seqs2kmer_overlap(X_train, kmer=kmer)
            X_tokenizer = load_kmer_tokenizer(kmer=kmer)
            X_test_kmer = seqs2kmer_overlap(X_test, kmer=kmer)
            x_maxlen = find_max_kmer_length(X_kmer, X_tokenizer) #401
            embedded_X_train = np.array(encode_sequences(X_kmer, X_tokenizer, max_length=x_maxlen))
            embedded_X_test = np.array(encode_sequences(X_test_kmer, X_tokenizer, max_length=x_maxlen))
            # src_vocab_size = X_chr1_tokenizer.get_vocab_size()
            src_vocab_size = len(X_tokenizer)

        elif embedding_type == 'BPE':
            print('Embedding: {} | vocab_size={}'.format(embedding_type, vocab_size_bpe))
            # set vocab_size = 2048
            X_tokenizer = BPE_embed(X_train, train_ratio = ratio, chr_index=6, vocab_size=vocab_size_bpe)
            # Check max_len for each chrmosome in train dataset
            x_maxlen = choose_max_length(X_train, X_tokenizer)  # max_len_train = 2258 , test=2242 (BPE 2048)
                                                                # max_len = 2106 (BPE 4096)
            # Check max_len for each chrmosome in test dataset
            embedded_X_train = np.array(encode_BPE(X_train, X_tokenizer, x_maxlen)) # assign idices to each token[13, 29, 5, 52, 18, ...]
            embedded_X_test = np.array(encode_BPE(X_test, X_tokenizer, x_maxlen)) # assign idices to each token[13, 29, 5, 52, 18, ...]
            # seq_len = tensor_X.shape[1]
            src_vocab_size = X_tokenizer.get_vocab_size()

        if model_name == 'MLPMixer':
            print('Run: MLPMixer | all chromosome - Embedding {} - Dataset {} - Ratio {}'.format(embedding_type, dataset, ratio))
            evaluate_result_regular(datapath, model_name, embedded_X_train, src_vocab_size, y_train, embedded_X_test, y_test, best_params, data_variants, device)
        elif model_name == 'Test1_CNN':
            print('Run: Test1_CNN | all chromosome - Embedding {} - Dataset {} - Ratio {}'.format(embedding_type, dataset, ratio))
            evaluate_result_regular(datapath, model_name, embedded_X_train, src_vocab_size, y_train, embedded_X_test, y_test, best_params, data_variants, device)
        elif model_name == 'Test1_Transformer':
            print('Run: Test1_Transformer | all chromosome - Embedding {} - Dataset {} - Ratio {}'.format(embedding_type, dataset, ratio))
            evaluate_result_regular(datapath, model_name, embedded_X_train, src_vocab_size, y_train, embedded_X_test, y_test, best_params, data_variants, device)
        elif model_name == 'Test2_Infinite_Transformer':
            print('Run: Test2_Infinite_Transformer | all chromosome - Embedding {} - Dataset {} - Ratio {}'.format(embedding_type, dataset, ratio))
            evaluate_result_regular(datapath, model_name, embedded_X_train, src_vocab_size, y_train, embedded_X_test, y_test, best_params, data_variants, device)
        elif model_name == 'Test1_HyperMixer':
            print('Run: Test1_HyperMixer | all chromosome - Embedding {} - Dataset {} - Ratio {}'.format(embedding_type, dataset, ratio))
            evaluate_result_regular(datapath, model_name, embedded_X_train, src_vocab_size, y_train, embedded_X_test, y_test, best_params, data_variants, device)

    elif model_type == 'Augmentation_Chromosome':
        print('Type: Augmentation_Chromosome')