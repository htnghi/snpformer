import argparse
import numpy as np

from preprocess.snp_splitting import *
from preprocess.snp_tokenize import *
from preprocess.snp_embed import *

from model.transformer_allchr import *
from model.transformer_splitchr import *

from model.mlp_mixer_allchr import *
from model.mlp_mixer_splitchr import *

from model.hyper_mixer_allchr import *
from model.hyper_mixer_splitchr import *

from model.cnn_transformer_allchr import *
from model.cnn_transformer_splitchr import *

from model.transformer_infinite_allchr import * 
from model.transformer_infinite_splitchr import * 

from model.word2vec_transformer_splitchr import *
from model.word2vec_hyper_mixer_splitchr import *

if __name__ == '__main__':
    """
    Run the main.py file to start the program:
        + Process the input arguments
        + Read data
        + Preprocess data
        + Train models
        + Prediction
    """

    # ----------------------------------------------------
    # Process the arguments
    # ----------------------------------------------------
    parser = argparse.ArgumentParser()

    parser.add_argument("-ddi", "--data_dir", type=str,
                        default='/Users/nghihuynh/Documents/MscTUM_BioTech/thesis/transformer_based_snp',
                        help="Path to the data folder")
    
    parser.add_argument("-mod", "--model_type", type=str,
                        default='Chromosomesplit',
                        help="Type of model for training the phenotype prediction: Chromosomesplit, Regular, Augmentation_Chromosome")
    
    parser.add_argument("-llm", "--model_name", type=str,
                        default='HyperMixer',
                        help="Name of LL model for training the phenotype prediction: Transformer, MLPMixer, HyperMixer, CNN_Transformer, Test_CNN, Test1_Transformer, Test2_Infinite_Transformer")
    
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

    args = vars(parser.parse_args())

    # ----------------------------------------------------
    # Check available GPUs, if not, run on CPUs
    # ----------------------------------------------------
    dev = "cpu"
    if args["gpucuda"] >= 1 and torch.cuda.is_available(): 
        print("GPU CUDA available, using GPU for training the models.")
        dev = "cuda:" + str(args["gpucuda"]-1) # to get the idx of gpu device
    else:
        print("GPU CUDA not available, using CPU instead.")
    device = torch.device(dev)

    # ----------------------------------------------------
    # Parsing the input arguments
    # ----------------------------------------------------
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

    # print('-----------------------------------------------')
    # print('Input arguments: ')
    # print('   + data_dir: {}'.format(datapath))
    # print('   + model: {}'.format(model))
    # print('   + minmax_scale: {}'.format(minmax_scale))
    # print('   + standa_scale: {}'.format(standa_scale))
    # print('   + pca_fitting: {}'.format(pca_fitting))
    # print('   + dataset: pheno_{}'.format(dataset))
    # print('   + gpucuda: {}'.format(gpucuda))

    data_variants = [minmax_scale, standa_scale, pca_fitting, dataset]
    # print('   + data_variants: {}'.format(data_variants))
    # print('-----------------------------------------------\n')

    # ----------------------------------------------------
    # Read data and preprocess
    # ----------------------------------------------------

    # One_hot Encoding
    # read_data_pheno(datapath, 1)
    # read_data_realworld(datapath, 4)
    # filter_duplicate_snps(datapath, 1)
    # split_train_test_data(datapath, 1, ratio)
    # map_indices_easypheno(datapath, 4, ratio)
    # ----------------------------------------------------
    # Tune and evaluate the model performance
    # ----------------------------------------------------
    # set up parameters for tuning
    training_params_dict = {
        'num_trials': 60,
        'min_trials': 30,
        'percentile': 65,
        'optunaseed': 42,
        'num_epochs': 100,
        'early_stop': 20,
        'batch_size': 32
    }

    print('---------------------------------------------------------')
    # print('Tuning MLP with dataset pheno-{}, minmax={}, standard={}, pcafit={}'.format(dataset, minmax_scale, standa_scale, pca_fitting))
    print('---------------------------------------------------------\n')
    # len each chromosome_train: 2401, 1696, 2015, 1727, 2161
    
    
    if model_type == 'Chromosomesplit':
        print('Run Chromosomesplit model')
        
        if dataset == 4:
            X_chr1_train, X_chr2_train, X_chr3_train, X_chr4_train, X_chr5_train = split_into_chromosome_train_realworld(datapath, dataset, ratio)
            X_chr1_test, X_chr2_test, X_chr3_test, X_chr4_test, X_chr5_test = split_into_chromosome_test_realworld(datapath, dataset, ratio)
            y_train, y_test= load_split_y_data(datapath, dataset, ratio)
        else:
            X_chr1_train, X_chr2_train, X_chr3_train, X_chr4_train, X_chr5_train = split_into_chromosome_train(datapath, dataset, ratio)
            X_chr1_test, X_chr2_test, X_chr3_test, X_chr4_test, X_chr5_test = split_into_chromosome_test(datapath, dataset, ratio)
            y_train, y_test= load_split_y_data(datapath, dataset, ratio)

        if embedding_type == 'kmer_nonoverlap':
            print('Using {} with kmer ={}'.format(embedding_type, kmer))

            X_chr1_kmer = seqs2kmer_nonoverlap(X_chr1_train, kmer=kmer)
            X_chr2_kmer = seqs2kmer_nonoverlap(X_chr2_train, kmer=kmer)
            X_chr3_kmer = seqs2kmer_nonoverlap(X_chr3_train, kmer=kmer)
            X_chr4_kmer = seqs2kmer_nonoverlap(X_chr4_train, kmer=kmer)
            X_chr5_kmer = seqs2kmer_nonoverlap(X_chr5_train, kmer=kmer)

            # X_chr1_tokenizer = kmer_embed(X_chr1_kmer, kmer=6)
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

            list_X_train = [embedded_X_chr1, embedded_X_chr2, embedded_X_chr3, embedded_X_chr4, embedded_X_chr5]

            # embedded_X_test_chr1 = np.array(encode(X_test_chr1_kmer, X_chr1_tokenizer, 801)) # assign idices to each token[13, 29, 5, 52, 18, ...]
            # embedded_X_test_chr2 = np.array(encode(X_test_chr2_kmer, X_chr1_tokenizer, 566))
            # embedded_X_test_chr3 = np.array(encode(X_test_chr3_kmer, X_chr1_tokenizer, 672))
            # embedded_X_test_chr4 = np.array(encode(X_test_chr4_kmer, X_chr1_tokenizer, 576))
            # embedded_X_test_chr5 = np.array(encode(X_test_chr5_kmer, X_chr1_tokenizer, 721))

            embedded_X_test_chr1 = np.array(encode_sequences(X_test_chr1_kmer, X_chr1_tokenizer, max_length=x1_maxlen))
            embedded_X_test_chr2 = np.array(encode_sequences(X_test_chr2_kmer, X_chr1_tokenizer, max_length=x2_maxlen))
            embedded_X_test_chr3 = np.array(encode_sequences(X_test_chr3_kmer, X_chr1_tokenizer, max_length=x3_maxlen))
            embedded_X_test_chr4 = np.array(encode_sequences(X_test_chr4_kmer, X_chr1_tokenizer, max_length=x4_maxlen))
            embedded_X_test_chr5 = np.array(encode_sequences(X_test_chr5_kmer, X_chr1_tokenizer, max_length=x5_maxlen))


            list_X_test = [embedded_X_test_chr1, embedded_X_test_chr2, embedded_X_test_chr3, embedded_X_test_chr4, embedded_X_test_chr5]

            # src_vocab_size = X_chr1_tokenizer.get_vocab_size()
            src_vocab_size = len(X_chr1_tokenizer)

            if model_name == 'Transformer':
                print('Run Transformer model with Split chromosome - use {} embedding - Dataset {} - Ratio {}'.format(embedding_type, dataset, ratio))
                best_params = tuning_splitchr_transformer(datapath, list_X_train, src_vocab_size, y_train, data_variants, training_params_dict, device)
                evaluate_result_splitchr_transformer(datapath, list_X_train, src_vocab_size, y_train, list_X_test, y_test, best_params, data_variants, device)
            elif model_name == 'MLPMixer':
                print('Run MLPMixer model with Split chromosome - use {} embedding - Dataset {} - Ratio {}'.format(embedding_type, dataset, ratio))
                best_params = tuning_splitchr_mlpmixer(datapath, list_X_train, src_vocab_size, y_train, data_variants, training_params_dict, device)
                evaluate_result_splitchr_mlpmixer(datapath, list_X_train, src_vocab_size, y_train, list_X_test, y_test, best_params, data_variants, device)
            elif model_name == 'HyperMixer':
                print('Run HyperMixer model with Split chromosome - use {} embedding - Dataset {} - Ratio {}'.format(embedding_type, dataset, ratio))
                best_params = tuning_splitchr_hypermixer(datapath, list_X_train, src_vocab_size, y_train, data_variants, training_params_dict, device)
                evaluate_result_splitchr_hypermixer(datapath, list_X_train, src_vocab_size, y_train, list_X_test, y_test, best_params, data_variants, device)
            elif model_name == 'Test1_CNN':
                print('Run Test1_CNN model with Split chromosome - use {} embedding - Dataset {} - Ratio {}'.format(embedding_type, dataset, ratio))
                best_params = tuning_splitchr_cnntransformer_test1(datapath, list_X_train, src_vocab_size, y_train, data_variants, training_params_dict, device)
                evaluate_result_splitchr_cnntransformer_test1(datapath, list_X_train, src_vocab_size, y_train, list_X_test, y_test, best_params, data_variants, device)
            elif model_name == 'Test2_Infinite_Transformer':
                print('Run Test2_Infinite_Transformer model with Split chromosome - use {} embedding - Dataset {} - Ratio {}'.format(embedding_type, dataset, ratio))
                best_params = tuning_splitchr_transformer_test2_infinite(datapath, list_X_train, src_vocab_size, y_train, data_variants, training_params_dict, device)
                evaluate_result_splitchr_transformer_test2_infinite(datapath, list_X_train, src_vocab_size, y_train, list_X_test, y_test, best_params, data_variants, device)
        
        elif embedding_type == 'kmer_overlap':
            print('Using {} with kmer ={}'.format(embedding_type, kmer))

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


            list_X_test = [embedded_X_test_chr1, embedded_X_test_chr2, embedded_X_test_chr3, embedded_X_test_chr4, embedded_X_test_chr5]

            # src_vocab_size = X_chr1_tokenizer.get_vocab_size()
            src_vocab_size = len(X_chr1_tokenizer)

            if model_name == 'Transformer':
                print('Run Transformer model with Split chromosome - use {} embedding - Dataset {} - Ratio {}'.format(embedding_type, dataset, ratio))
                best_params = tuning_splitchr_transformer(datapath, list_X_train, src_vocab_size, y_train, data_variants, training_params_dict, device)
                evaluate_result_splitchr_transformer(datapath, list_X_train, src_vocab_size, y_train, list_X_test, y_test, best_params, data_variants, device)
            elif model_name == 'MLPMixer':
                print('Run MLPMixer model with Split chromosome - use {} embedding - Dataset {} - Ratio {}'.format(embedding_type, dataset, ratio))
                best_params = tuning_splitchr_mlpmixer(datapath, list_X_train, src_vocab_size, y_train, data_variants, training_params_dict, device)
                evaluate_result_splitchr_mlpmixer(datapath, list_X_train, src_vocab_size, y_train, list_X_test, y_test, best_params, data_variants, device)
            elif model_name == 'HyperMixer':
                print('Run HyperMixer model with Split chromosome - use {} embedding - Dataset {} - Ratio {}'.format(embedding_type, dataset, ratio))
                best_params = tuning_splitchr_hypermixer(datapath, list_X_train, src_vocab_size, y_train, data_variants, training_params_dict, device)
                evaluate_result_splitchr_hypermixer(datapath, list_X_train, src_vocab_size, y_train, list_X_test, y_test, best_params, data_variants, device)
            elif model_name == 'Test1_CNN':
                print('Run Test1_CNN model with Split chromosome - use {} embedding - Dataset {} - Ratio {}'.format(embedding_type, dataset, ratio))
                best_params = tuning_splitchr_cnntransformer_test1(datapath, list_X_train, src_vocab_size, y_train, data_variants, training_params_dict, device)
                evaluate_result_splitchr_cnntransformer_test1(datapath, list_X_train, src_vocab_size, y_train, list_X_test, y_test, best_params, data_variants, device)
            elif model_name == 'Test2_Infinite_Transformer':
                print('Run Test2_Infinite_Transformer model with Split chromosome - use {} embedding - Dataset {} - Ratio {}'.format(embedding_type, dataset, ratio))
                best_params = tuning_splitchr_transformer_test2_infinite(datapath, list_X_train, src_vocab_size, y_train, data_variants, training_params_dict, device)
                evaluate_result_splitchr_transformer_test2_infinite(datapath, list_X_train, src_vocab_size, y_train, list_X_test, y_test, best_params, data_variants, device)

        elif embedding_type == 'BPE':
            print('Using {} with vocab_size ={}'.format(embedding_type, vocab_size_bpe))

            X_chr1_tokenizer = BPE_embed(X_chr1_train, train_ratio = ratio, chr_index=1, vocab_size=vocab_size_bpe)
            X_chr2_tokenizer = BPE_embed(X_chr1_train, train_ratio = ratio, chr_index=2, vocab_size=vocab_size_bpe)
            X_chr3_tokenizer = BPE_embed(X_chr1_train, train_ratio = ratio, chr_index=3, vocab_size=vocab_size_bpe)
            X_chr4_tokenizer = BPE_embed(X_chr1_train, train_ratio = ratio, chr_index=4, vocab_size=vocab_size_bpe)
            X_chr5_tokenizer = BPE_embed(X_chr1_train, train_ratio = ratio, chr_index=5, vocab_size=vocab_size_bpe)

            # pad_token = torch.tensor([X_chr1_tokenizer.token_to_id("[PAD]")], dtype=torch.int64)

            # Check max_len for each chrmosome in train dataset
            x1_maxlen = choose_max_length(X_chr1_train, X_chr1_tokenizer)  # max_len = 460 
            x2_maxlen =choose_max_length(X_chr1_train, X_chr2_tokenizer)   # max_len = 650 
            x3_maxlen= choose_max_length(X_chr1_train, X_chr3_tokenizer)   # max_len = 635 
            x4_maxlen =choose_max_length(X_chr1_train, X_chr4_tokenizer)   # max_len = 643 
            x5_maxlen =choose_max_length(X_chr1_train, X_chr5_tokenizer)   # max_len = 621 

            # Check max_len for each chrmosome in test dataset
            # x1 = choose_max_length(X_chr1_test, X_chr1_tokenizer)  # max_len = 450 
            # x2 =choose_max_length(X_chr1_test, X_chr2_tokenizer)   # max_len = 642 
            # x3= choose_max_length(X_chr1_test, X_chr3_tokenizer)   # max_len = 631 
            # x4 =choose_max_length(X_chr1_test, X_chr4_tokenizer)   # max_len = 636 
            # x5 =choose_max_length(X_chr1_test, X_chr5_tokenizer)   # max_len = 616 


            embedded_X_chr1 = np.array(encode(X_chr1_train, X_chr1_tokenizer, x1_maxlen)) # assign idices to each token[13, 29, 5, 52, 18, ...]
            embedded_X_chr2 = np.array(encode(X_chr2_train, X_chr2_tokenizer, x2_maxlen))
            embedded_X_chr3 = np.array(encode(X_chr3_train, X_chr3_tokenizer, x3_maxlen))
            embedded_X_chr4 = np.array(encode(X_chr4_train, X_chr4_tokenizer, x4_maxlen))
            embedded_X_chr5 = np.array(encode(X_chr5_train, X_chr5_tokenizer, x5_maxlen))

            list_X_train = [embedded_X_chr1, embedded_X_chr2, embedded_X_chr3, embedded_X_chr4, embedded_X_chr5]

            embedded_X_test_chr1 = np.array(encode(X_chr1_test, X_chr1_tokenizer, x1_maxlen)) # assign idices to each token[13, 29, 5, 52, 18, ...]
            embedded_X_test_chr2 = np.array(encode(X_chr2_test, X_chr2_tokenizer, x2_maxlen))
            embedded_X_test_chr3 = np.array(encode(X_chr3_test, X_chr3_tokenizer, x3_maxlen))
            embedded_X_test_chr4 = np.array(encode(X_chr4_test, X_chr4_tokenizer, x4_maxlen))
            embedded_X_test_chr5 = np.array(encode(X_chr5_test, X_chr5_tokenizer, x5_maxlen))
        
            list_X_test = [embedded_X_test_chr1, embedded_X_test_chr2, embedded_X_test_chr3, embedded_X_test_chr4, embedded_X_test_chr5]

            # seq_len = tensor_X.shape[1]
            src_vocab_size = X_chr1_tokenizer.get_vocab_size()  

            if model_name == 'Transformer':
                print('Run Transformer model with Split chromosome - use {} embedding - Dataset {} - Ratio {}'.format(embedding_type, dataset, ratio))
                best_params = tuning_splitchr_transformer(datapath, list_X_train, src_vocab_size, y_train, data_variants, training_params_dict, device)
                evaluate_result_splitchr_transformer(datapath, list_X_train, src_vocab_size, y_train, list_X_test, y_test, best_params, data_variants, device)
            elif model_name == 'MLPMixer':
                print('Run MLPMixer model with Split chromosome - use {} embedding - Dataset {} - Ratio {}'.format(embedding_type, dataset, ratio))
                best_params = tuning_splitchr_mlpmixer(datapath, list_X_train, src_vocab_size, y_train, data_variants, training_params_dict, device)
                evaluate_result_splitchr_mlpmixer(datapath, list_X_train, src_vocab_size, y_train, list_X_test, y_test, best_params, data_variants, device)
            elif model_name == 'HyperMixer':
                print('Run HyperMixer model with Split chromosome - use {} embedding - Dataset {} - Ratio {}'.format(embedding_type, dataset, ratio))
                best_params = tuning_splitchr_hypermixer(datapath, list_X_train, src_vocab_size, y_train, data_variants, training_params_dict, device)
                evaluate_result_splitchr_hypermixer(datapath, list_X_train, src_vocab_size, y_train, list_X_test, y_test, best_params, data_variants, device)
            elif model_name == 'Test1_CNN':
                print('Run Test1_CNN model with Split chromosome - use {} embedding - Dataset {} - Ratio {}'.format(embedding_type, dataset, ratio))
                best_params = tuning_splitchr_cnntransformer_test1(datapath, list_X_train, src_vocab_size, y_train, data_variants, training_params_dict, device)
                evaluate_result_splitchr_cnntransformer_test1(datapath, list_X_train, src_vocab_size, y_train, list_X_test, y_test, best_params, data_variants, device)
            elif model_name == 'Test2_Infinite_Transformer':
                print('Run Test2_Infinite_Transformer model with Split chromosome - use {} embedding - Dataset {} - Ratio {}'.format(embedding_type, dataset, ratio))
                best_params = tuning_splitchr_transformer_test2_infinite(datapath, list_X_train, src_vocab_size, y_train, data_variants, training_params_dict, device)
                evaluate_result_splitchr_transformer_test2_infinite(datapath, list_X_train, src_vocab_size, y_train, list_X_test, y_test, best_params, data_variants, device)

        elif embedding_type == 'word2vec':
            print('Using {} '.format(embedding_type))

            # X_chr1_w2v = word2vec_embed(X_chr1_train)
            # print(X_chr1_w2v)
            # exit()

            X_chr1_kmer = np.array(seqs2kmer_nonoverlap_w2v(X_chr1_train, kmer=kmer))
            X_chr2_kmer = np.array(seqs2kmer_nonoverlap_w2v(X_chr2_train, kmer=kmer))
            X_chr3_kmer = np.array(seqs2kmer_nonoverlap_w2v(X_chr3_train, kmer=kmer))
            X_chr4_kmer = np.array(seqs2kmer_nonoverlap_w2v(X_chr4_train, kmer=kmer))
            X_chr5_kmer = np.array(seqs2kmer_nonoverlap_w2v(X_chr5_train, kmer=kmer))

            X_test_chr1_kmer = np.array(seqs2kmer_nonoverlap_w2v(X_chr1_test, kmer=kmer))
            X_test_chr2_kmer = np.array(seqs2kmer_nonoverlap_w2v(X_chr2_test, kmer=kmer))
            X_test_chr3_kmer = np.array(seqs2kmer_nonoverlap_w2v(X_chr3_test, kmer=kmer))
            X_test_chr4_kmer = np.array(seqs2kmer_nonoverlap_w2v(X_chr4_test, kmer=kmer))
            X_test_chr5_kmer = np.array(seqs2kmer_nonoverlap_w2v(X_chr5_test, kmer=kmer))


            list_X_train = [X_chr1_kmer, X_chr2_kmer, X_chr3_kmer, X_chr4_kmer, X_chr5_kmer]
            list_X_test = [X_test_chr1_kmer, X_test_chr2_kmer, X_test_chr3_kmer, X_test_chr4_kmer, X_test_chr5_kmer]

            if model_name == 'Transformer_word2vec':
                print('Run Transformer_Word2vec model with Split chromosome - use {} embedding - Dataset {} - Ratio {}'.format(embedding_type, dataset, ratio))
                best_params = tuning_splitchr_transformer_word2vec(datapath, list_X_train, y_train, data_variants, training_params_dict, device)
                evaluate_result_splitchr_transformer_word2vec(datapath, list_X_train, y_train, list_X_test, y_test, best_params, data_variants, device)
            elif model_name == 'HyperMixer':
                print('Run HyperMixer model with Split chromosome - use {} embedding - Dataset {} - Ratio {}'.format(embedding_type, dataset, ratio))
                best_params = tuning_splitchr_hypermixer_word2vec(datapath, list_X_train, y_train, data_variants, training_params_dict, device)
                evaluate_result_splitchr_hypermixer_word2vec(datapath, list_X_train, y_train, list_X_test, y_test, best_params, data_variants, device)
            

    elif model_type == 'Regular':
        print('Run Regular model')
        X_train, y_train, X_test, y_test = load_split_regular_data(datapath, dataset, ratio)

        if embedding_type == 'kmer_nonoverlap':
            print('Using {} with kmer ={}'.format(embedding_type, kmer))

            X_kmer = seqs2kmer_overlap(X_train, kmer=kmer)

            X_tokenizer = load_kmer_tokenizer(kmer=kmer)

            X_test_kmer = seqs2kmer_nonoverlap(X_test, kmer=kmer)

            x_maxlen = find_max_kmer_length(X_kmer, X_tokenizer) #401

            embedded_X_train = np.array(encode_sequences(X_kmer, X_tokenizer, max_length=x_maxlen))

            embedded_X_test = np.array(encode_sequences(X_test_kmer, X_tokenizer, max_length=x_maxlen))

            # src_vocab_size = X_chr1_tokenizer.get_vocab_size()
            src_vocab_size = len(X_tokenizer)

            if model_name == 'MLPMixer':
                print('Run MLPMixer model with all chromosome - use {} embedding - Dataset {} - Ratio {}'.format(embedding_type, dataset, ratio))
                best_params = tuning_regular_mlpmixer(datapath, embedded_X_train, src_vocab_size, y_train, data_variants, training_params_dict, device)
                evaluate_result_regular_mlpmixer(datapath, embedded_X_train, src_vocab_size, y_train, embedded_X_test, y_test, best_params, data_variants, device)
            elif model_name == 'Test1_CNN':
                print('Run Test1_CNN model with all chromosome - use {} embedding - Dataset {} - Ratio {}'.format(embedding_type, dataset, ratio))
                best_params = tuning_regular_cnntransformer_test1(datapath, embedded_X_train, src_vocab_size, y_train, data_variants, training_params_dict, device)
                evaluate_result_regular_cnntransformer_test1(datapath, embedded_X_train, src_vocab_size, y_train, embedded_X_test, y_test, best_params, data_variants, device)
            elif model_name == 'Test1_Transformer':
                print('Run Test1_Transformer model with all chromosome - use {} embedding - Dataset {} - Ratio {}'.format(embedding_type, dataset, ratio))
                best_params = tuning_regular_transformer_test1(datapath, embedded_X_train, src_vocab_size, y_train, data_variants, training_params_dict, device)
                evaluate_result_regular_transformer_test1(datapath, embedded_X_train, src_vocab_size, y_train, embedded_X_test, y_test, best_params, data_variants, device)
            elif model_name == 'Test2_Infinite_Transformer':
                print('Run Test2_Infinite_Transformer model with all chromosome - use {} embedding - Dataset {} - Ratio {}'.format(embedding_type, dataset, ratio))
                best_params = tuning_regular_transformer_test2_infinite(datapath, embedded_X_train, src_vocab_size, y_train, data_variants, training_params_dict, device)
                evaluate_result_regular_transformer_test2_infinite(datapath, embedded_X_train, src_vocab_size, y_train, embedded_X_test, y_test, best_params, data_variants, device)
            elif model_name == 'Test1_HyperMixer':
                print('Run Test1_HyperMixer model with all chromosome - use {} embedding - Dataset {} - Ratio {}'.format(embedding_type, dataset, ratio))
                best_params = tuning_regular_hypermixer_test1(datapath, embedded_X_train, src_vocab_size, y_train, data_variants, training_params_dict, device)
                evaluate_result_regular_hypermixer_test1(datapath, embedded_X_train, src_vocab_size, y_train, embedded_X_test, y_test, best_params, data_variants, device)

        elif embedding_type == 'kmer_overlap':
            print('Using {} with kmer ={}'.format(embedding_type, kmer))

            X_kmer = seqs2kmer_overlap(X_train, kmer=kmer)

            X_tokenizer = load_kmer_tokenizer(kmer=kmer)

            X_test_kmer = seqs2kmer_overlap(X_test, kmer=kmer)

            x_maxlen = find_max_kmer_length(X_kmer, X_tokenizer) #401

            embedded_X_train = np.array(encode_sequences(X_kmer, X_tokenizer, max_length=x_maxlen))

            embedded_X_test = np.array(encode_sequences(X_test_kmer, X_tokenizer, max_length=x_maxlen))

            # src_vocab_size = X_chr1_tokenizer.get_vocab_size()
            src_vocab_size = len(X_tokenizer)

            if model_name == 'MLPMixer':
                print('Run MLPMixer model with all chromosome - use {} embedding - Dataset {} - Ratio {}'.format(embedding_type, dataset, ratio))
                best_params = tuning_regular_mlpmixer(datapath, embedded_X_train, src_vocab_size, y_train, data_variants, training_params_dict, device)
                evaluate_result_regular_mlpmixer(datapath, embedded_X_train, src_vocab_size, y_train, embedded_X_test, y_test, best_params, data_variants, device)
            elif model_name == 'Test1_CNN':
                print('Run Test1_CNN model with all chromosome - use {} embedding - Dataset {} - Ratio {}'.format(embedding_type, dataset, ratio))
                best_params = tuning_regular_cnntransformer_test1(datapath, embedded_X_train, src_vocab_size, y_train, data_variants, training_params_dict, device)
                evaluate_result_regular_cnntransformer_test1(datapath, embedded_X_train, src_vocab_size, y_train, embedded_X_test, y_test, best_params, data_variants, device)
            elif model_name == 'Test1_Transformer':
                print('Run Test1_Transformer model with all chromosome - use {} embedding - Dataset {} - Ratio {}'.format(embedding_type, dataset, ratio))
                best_params = tuning_regular_transformer_test1(datapath, embedded_X_train, src_vocab_size, y_train, data_variants, training_params_dict, device)
                evaluate_result_regular_transformer_test1(datapath, embedded_X_train, src_vocab_size, y_train, embedded_X_test, y_test, best_params, data_variants, device)
            elif model_name == 'Test2_Infinite_Transformer':
                print('Run Test2_Infinite_Transformer model with all chromosome - use {} embedding - Dataset {} - Ratio {}'.format(embedding_type, dataset, ratio))
                best_params = tuning_regular_transformer_test2_infinite(datapath, embedded_X_train, src_vocab_size, y_train, data_variants, training_params_dict, device)
                evaluate_result_regular_transformer_test2_infinite(datapath, embedded_X_train, src_vocab_size, y_train, embedded_X_test, y_test, best_params, data_variants, device)
            elif model_name == 'Test1_HyperMixer':
                print('Run Test1_HyperMixer model with all chromosome - use {} embedding - Dataset {} - Ratio {}'.format(embedding_type, dataset, ratio))
                best_params = tuning_regular_hypermixer_test1(datapath, embedded_X_train, src_vocab_size, y_train, data_variants, training_params_dict, device)
                evaluate_result_regular_hypermixer_test1(datapath, embedded_X_train, src_vocab_size, y_train, embedded_X_test, y_test, best_params, data_variants, device)

        elif embedding_type == 'BPE':
            print('Using {} with vocab_size ={}'.format(embedding_type, vocab_size_bpe))

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
                print('Run MLPMixer model with all chromosome - use {} embedding - Dataset {} - Ratio {}'.format(embedding_type, dataset, ratio))
                best_params = tuning_regular_mlpmixer(datapath, embedded_X_train, src_vocab_size, y_train, data_variants, training_params_dict, device)
                evaluate_result_regular_mlpmixer(datapath, embedded_X_train, src_vocab_size, y_train, embedded_X_test, y_test, best_params, data_variants, device)
            elif model_name == 'Test1_CNN':
                print('Run Test1_CNN model with all chromosome - use {} embedding - Dataset {} - Ratio {}'.format(embedding_type, dataset, ratio))
                best_params = tuning_regular_cnntransformer_test1(datapath, embedded_X_train, src_vocab_size, y_train, data_variants, training_params_dict, device)
                evaluate_result_regular_cnntransformer_test1(datapath, embedded_X_train, src_vocab_size, y_train, embedded_X_test, y_test, best_params, data_variants, device)
            elif model_name == 'Test1_Transformer':
                print('Run Test1_Transformer model with all chromosome - use {} embedding - Dataset {} - Ratio {}'.format(embedding_type, dataset, ratio))
                best_params = tuning_regular_transformer_test1(datapath, embedded_X_train, src_vocab_size, y_train, data_variants, training_params_dict, device)
                evaluate_result_regular_transformer_test1(datapath, embedded_X_train, src_vocab_size, y_train, embedded_X_test, y_test, best_params, data_variants, device)
            elif model_name == 'Test2_Infinite_Transformer':
                print('Run Test2_Infinite_Transformer model with all chromosome - use {} embedding - Dataset {} - Ratio {}'.format(embedding_type, dataset, ratio))
                best_params = tuning_regular_transformer_test2_infinite(datapath, embedded_X_train, src_vocab_size, y_train, data_variants, training_params_dict, device)
                evaluate_result_regular_transformer_test2_infinite(datapath, embedded_X_train, src_vocab_size, y_train, embedded_X_test, y_test, best_params, data_variants, device)
            elif model_name == 'Test1_HyperMixer':
                print('Run Test1_HyperMixer model with all chromosome - use {} embedding - Dataset {} - Ratio {}'.format(embedding_type, dataset, ratio))
                best_params = tuning_regular_hypermixer_test1(datapath, embedded_X_train, src_vocab_size, y_train, data_variants, training_params_dict, device)
                evaluate_result_regular_hypermixer_test1(datapath, embedded_X_train, src_vocab_size, y_train, embedded_X_test, y_test, best_params, data_variants, device)


    elif model_type == 'Augmentation_Chromosome':
        set_seeds()
        X_train, y_train = load_train_augmented_data(datapath, dataset, ratio)
        X_test, y_test = load_test_augmented_data(datapath, dataset, ratio)


        if embedding_type == 'kmer_nonoverlap':

            X_kmer = seqs2kmer_nonoverlap(X_train, kmer=kmer)

            X_tokenizer = load_kmer_tokenizer(kmer=kmer)

            X_test_kmer = seqs2kmer_nonoverlap(X_test, kmer=kmer)

            x_maxlen = find_max_kmer_length(X_kmer, X_tokenizer)

            embedded_X_train = np.array(encode_sequences(X_kmer, X_tokenizer, x_maxlen))

            embedded_X_test = np.array(encode_sequences(X_test_kmer, X_tokenizer, x_maxlen))

            # src_vocab_size = X_chr1_tokenizer.get_vocab_size()
            src_vocab_size = len(X_tokenizer)

            if model_name == 'Transformer':
                print('Run Transformer model with all chromosome - use K-mer Non-overlap embedding')
                best_params = tuning_regular_transformer(datapath, embedded_X_train, src_vocab_size, y_train, data_variants, training_params_dict, device)
                evaluate_result_regular_transformer(datapath, embedded_X_train, src_vocab_size, y_train, embedded_X_test, y_test, best_params, data_variants, device)
            elif model_name == 'MLPMixer':
                print('Run MLPMixer model with all chromosome - use K-mer Non-overlap embedding')
                best_params = tuning_regular_mlpmixer(datapath, embedded_X_train, src_vocab_size, y_train, data_variants, training_params_dict, device)
                evaluate_result_regular_mlpmixer(datapath, embedded_X_train, src_vocab_size, y_train, embedded_X_test, y_test, best_params, data_variants, device)
            elif model_name == 'HyperMixer':
                print('Run HyperMixer model with all chromosome - use K-mer Non-overlap embedding')
                best_params = tuning_regular_hypermixer(datapath, embedded_X_train, src_vocab_size, y_train, data_variants, training_params_dict, device)
                evaluate_result_regular_hypermixer(datapath, embedded_X_train, src_vocab_size, y_train, embedded_X_test, y_test, best_params, data_variants, device)
            elif model_name == 'CNN_Transformer':
                print('Run CNN_Transformer model with all chromosome - use K-mer Non-overlap embedding')
                best_params = tuning_regular_convblocktransformer(datapath, embedded_X_train, src_vocab_size, y_train, data_variants, training_params_dict, device)
                evaluate_result_regular_convblocktransformer(datapath, embedded_X_train, src_vocab_size, y_train, embedded_X_test, y_test, best_params, data_variants, device)
            elif model_name == 'Test_CNN':
                print('Run Test_CNN model with all chromosome - use K-mer Non-overlap embedding')
                best_params = tuning_regular_cnntransformer(datapath, embedded_X_train, src_vocab_size, y_train, data_variants, training_params_dict, device)
                evaluate_result_regular_cnntransformer(datapath, embedded_X_train, src_vocab_size, y_train, embedded_X_test, y_test, best_params, data_variants, device)
            elif model_name == 'Test1_CNN':
                print('Run Test1_CNN model with all chromosome - use K-mer Non-overlap embedding')
                best_params = tuning_regular_cnntransformer_test1(datapath, embedded_X_train, src_vocab_size, y_train, data_variants, training_params_dict, device)
                evaluate_result_regular_cnntransformer_test1(datapath, embedded_X_train, src_vocab_size, y_train, embedded_X_test, y_test, best_params, data_variants, device)
            elif model_name == 'Test1_Transformer':
                print('Run Test1_Transformer model with all chromosome - use K-mer Non-overlap embedding')
                best_params = tuning_regular_transformer_test1(datapath, embedded_X_train, src_vocab_size, y_train, data_variants, training_params_dict, device)
                evaluate_result_regular_transformer_test1(datapath, embedded_X_train, src_vocab_size, y_train, embedded_X_test, y_test, best_params, data_variants, device)
            elif model_name == 'Test1_HyperMixer':
                print('Run Test1_HyperMixer model with all chromosome - use K-mer Non-overlap embedding')
                best_params = tuning_regular_hypermixer_test1(datapath, embedded_X_train, src_vocab_size, y_train, data_variants, training_params_dict, device)
                evaluate_result_regular_hypermixer_test1(datapath, embedded_X_train, src_vocab_size, y_train, embedded_X_test, y_test, best_params, data_variants, device)


        elif embedding_type == 'BPE':

            X_tokenizer = BPE_embed(X_train, train_ratio = ratio, chr_index=6, vocab_size=vocab_size_bpe)

            # Check max_len for each chrmosome in train dataset
            x_maxlen = choose_max_length(X_train, X_tokenizer)  # max_len_train = 2258 , test=2242


            embedded_X_train = np.array(encode_BPE(X_train, X_tokenizer, x_maxlen)) # assign idices to each token[13, 29, 5, 52, 18, ...]
            embedded_X_test = np.array(encode_BPE(X_test, X_tokenizer, x_maxlen)) # assign idices to each token[13, 29, 5, 52, 18, ...]

            # seq_len = tensor_X.shape[1]
            src_vocab_size = X_tokenizer.get_vocab_size()

            if model_name == 'Transformer':
                print('Run Transformer model with all chromosome - use BPE embedding')
                best_params = tuning_regular_transformer(datapath, embedded_X_train, src_vocab_size, y_train, data_variants, training_params_dict, device)
                evaluate_result_regular_transformer(datapath, embedded_X_train, src_vocab_size, y_train, embedded_X_test, y_test, best_params, data_variants, device)
            elif model_name == 'MLPMixer':
                print('Run MLPMixer model with all chromosome - use BPE embedding')
                best_params = tuning_regular_mlpmixer(datapath, embedded_X_train, src_vocab_size, y_train, data_variants, training_params_dict, device)
                evaluate_result_regular_mlpmixer(datapath, embedded_X_train, src_vocab_size, y_train, embedded_X_test, y_test, best_params, data_variants, device)
            elif model_name == 'HyperMixer':
                print('Run HyperMixer model with all chromosome - use BPE embedding')
                best_params = tuning_regular_hypermixer(datapath, embedded_X_train, src_vocab_size, y_train, data_variants, training_params_dict, device)
                evaluate_result_regular_hypermixer(datapath, embedded_X_train, src_vocab_size, y_train, embedded_X_test, y_test, best_params, data_variants, device)
            elif model_name == 'CNN_Transformer':
                print('Run CNN_Transformer model with all chromosome - use BPE embedding')
                best_params = tuning_regular_convblocktransformer(datapath, embedded_X_train, src_vocab_size, y_train, data_variants, training_params_dict, device)
                evaluate_result_regular_convblocktransformer(datapath, embedded_X_train, src_vocab_size, y_train, embedded_X_test, y_test, best_params, data_variants, device)
            elif model_name == 'Test_CNN':
                print('Run Test_CNN model with all chromosome - use K-mer Non-overlap embedding')
                best_params = tuning_regular_cnntransformer(datapath, embedded_X_train, src_vocab_size, y_train, data_variants, training_params_dict, device)
                evaluate_result_regular_cnntransformer(datapath, embedded_X_train, src_vocab_size, y_train, embedded_X_test, y_test, best_params, data_variants, device)
            elif model_name == 'Test1_CNN':
                print('Run Test1_CNN model with all chromosome - use K-mer Non-overlap embedding')
                best_params = tuning_regular_cnntransformer_test1(datapath, embedded_X_train, src_vocab_size, y_train, data_variants, training_params_dict, device)
                evaluate_result_regular_cnntransformer_test1(datapath, embedded_X_train, src_vocab_size, y_train, embedded_X_test, y_test, best_params, data_variants, device)
            elif model_name == 'Test1_Transformer':
                print('Run Test1_Transformer model with all chromosome - use K-mer Non-overlap embedding')
                best_params = tuning_regular_transformer_test1(datapath, embedded_X_train, src_vocab_size, y_train, data_variants, training_params_dict, device)
                evaluate_result_regular_transformer_test1(datapath, embedded_X_train, src_vocab_size, y_train, embedded_X_test, y_test, best_params, data_variants, device)
            elif model_name == 'Test2_Infinite_Transformer':
                print('Run Test2_Infinite_Transformer model with all chromosome - use K-mer Non-overlap embedding')
                best_params = tuning_regular_transformer_test2_infinite(datapath, embedded_X_train, src_vocab_size, y_train, data_variants, training_params_dict, device)
                evaluate_result_regular_transformer_test2_infinite(datapath, embedded_X_train, src_vocab_size, y_train, embedded_X_test, y_test, best_params, data_variants, device)
            elif model_name == 'Test1_HyperMixer':
                print('Run Test1_HyperMixer model with all chromosome - use K-mer Non-overlap embedding')
                best_params = tuning_regular_hypermixer_test1(datapath, embedded_X_train, src_vocab_size, y_train, data_variants, training_params_dict, device)
                evaluate_result_regular_hypermixer_test1(datapath, embedded_X_train, src_vocab_size, y_train, embedded_X_test, y_test, best_params, data_variants, device)

