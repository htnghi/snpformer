import json 
import sklearn
import argparse
import numpy as np

from preprocess.snp_splitting import *
from preprocess.snp_tokenize import *
from preprocess.snp_embed import *

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold

from torch.optim import SGD, Adam
from torch.profiler import profile, record_function, ProfilerActivity
from fvcore.nn import FlopCountAnalysis, parameter_count_table
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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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
        max_seq_lens = best_params['max_seq_len']
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

    # training loop over epochs
    for epoch in range(num_epochs):
        if epoch < 1:
            with profile(activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True, with_flops=True) as prof:
                train_one_epoch(model, train_loader, loss_function, optimizer, device)
            print("Training Profile:")
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=5))
        else:
            train_one_epoch(model, train_loader, loss_function, optimizer, device)
    
    # predict result test
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True, with_flops=True) as prof:
        y_pred = predict(model, test_loader, device)
    
    # print profiled data
    print("Prediction Profile:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=5))

    # Calculating the number of parameters
    num_params = count_parameters(model)
    print(f"Number of parameters: {num_params}")
    
    # Calculating FLOPs using fvcore
    inputs = next(iter(train_loader))[0].to(device)
    flops = FlopCountAnalysis(model, inputs)
    print(f"FLOPs: {flops.total()}")

    # Printing detailed parameter count table
    print("Parameter Count Table:")
    print(parameter_count_table(model))

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