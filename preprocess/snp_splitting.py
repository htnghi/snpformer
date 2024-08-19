import pandas as pd
import numpy as np

import torch
import random

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

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

# -------------------------------------------------------------
#  1. Read data + Split into train, test
# -------------------------------------------------------------
def read_data_pheno(datapath, type):
    """
    Read the raw data or impured data by pandas,
        + in: temporarily the data path is fixed
        + out: the returned format is pandas dataframe and write to .csv files
    """

    df_genotypes  = pd.read_csv(datapath + '/data/raw_geneotype_dataset.csv')
    df_phenotypes = pd.read_csv(datapath + '/data/phenotype_data.csv')

    # delete nan values from phenotye data
    df_pheno = df_phenotypes.dropna(subset=['sample_ids', 'pheno'+str(type)]) #only delte missing value in pheno1 column
    # print('Number of Corresponding phenotype values:\t%d' %df_pheno.shape[0])

    # select the samples id that we have in y_matrix.csv
    unique_ids_ymatrix = df_pheno['sample_ids'].unique()

    # filter only row with the sample ids in x_matrix.csv that fits with the ids in y_matrix.csv
    df_genotypes = df_genotypes[df_genotypes['sample_ids'].isin(unique_ids_ymatrix)]

    # get the list of common ids between two datasets
    common_sample_ids = df_genotypes['sample_ids'].unique()

    # fileter again the sample ids in y_matrix.csv
    df_pheno = df_pheno[df_pheno['sample_ids'].isin(common_sample_ids)]

    # then map the continuous_values of y to x
    phenotype_dict_arr = df_pheno.set_index('sample_ids').to_dict()['pheno'+str(type)]
    trans_phenotype_dict = {key: float(value) for key, value in phenotype_dict_arr.items()}
    df_genotypes['pheno'+str(type)] = df_genotypes['sample_ids'].map(trans_phenotype_dict) # add label column to genotypes data

    # create new X1, y1
    X = df_genotypes.iloc[:,1:df_genotypes.shape[1]-1]
    y = df_genotypes[['sample_ids', 'pheno'+str(type)]]

    # convert new dataset to csv
    X.to_csv(datapath + '/data/pheno' + str(type) + '/x_matrix.csv')
    y.to_csv(datapath + '/data/pheno' + str(type) + '/y_matrix.csv')
    # print('------------------------------------------------------------------\n')

def read_data_realworld(datapath, type):
    """
    Read the raw data or impured data by pandas,
        + in: temporarily the data path is fixed
        + out: the returned format is pandas dataframe and write to .csv files
    """

    df_genotypes  = pd.read_csv(datapath + '/data/raw_realworld_geneotype_dataset.csv')
    df_phenotypes = pd.read_csv(datapath + '/data/study_12_values.csv')

    # delete nan values from phenotye data
    df_pheno = df_phenotypes.dropna(subset=['sample_ids', 'FT10']) #only delte missing value in pheno1 column
    # print('Number of Corresponding phenotype values:\t%d' %df_pheno.shape[0])

    # select the samples id that we have in y_matrix.csv
    unique_ids_ymatrix = df_pheno['sample_ids'].unique()

    # filter only row with the sample ids in x_matrix.csv that fits with the ids in y_matrix.csv
    df_genotypes = df_genotypes[df_genotypes['sample_ids'].isin(unique_ids_ymatrix)]

    # get the list of common ids between two datasets
    common_sample_ids = df_genotypes['sample_ids'].unique()

    # fileter again the sample ids in y_matrix.csv
    df_pheno = df_pheno[df_pheno['sample_ids'].isin(common_sample_ids)]

    # then map the continuous_values of y to x
    phenotype_dict_arr = df_pheno.set_index('sample_ids').to_dict()['FT10']
    trans_phenotype_dict = {key: float(value) for key, value in phenotype_dict_arr.items()}
    df_genotypes['FT10'] = df_genotypes['sample_ids'].map(trans_phenotype_dict) # add label column to genotypes data

    # create new X1, y1
    X = df_genotypes.iloc[:,1:df_genotypes.shape[1]-1]
    y = df_genotypes[['sample_ids', 'FT10']]

    # convert new dataset to csv
    X.to_csv(datapath + '/data/pheno' + str(type) + '/x_matrix.csv')
    y.to_csv(datapath + '/data/pheno' + str(type) + '/y_matrix.csv')
    # print('------------------------------------------------------------------\n')

def filter_duplicate_snps(datapath, type):
        """
        Remove duplicate SNPs,
        i.e. SNPs that are completely the same for all samples and therefore do not add information.
        """
        print('Filter duplicate SNPs')
        X_full = pd.read_csv(datapath + '/data/pheno' + str(type) + '/x_matrix.csv')
        
        # Extract the sample_ids column
        sample_ids = X_full.iloc[:, 1]
        X_full_data = X_full.iloc[:, 2:]

        # Convert the DataFrame to a NumPy array of strings
        X_full_array = X_full_data.to_numpy(dtype=str)

        # Find unique columns and their indices
        uniques, index = np.unique(X_full_array, return_index=True, axis=1)

        # Identify similar columns and print detailed information
        original_indices = np.arange(X_full_data.shape[1])
        filtered_indices = np.setdiff1d(original_indices, np.sort(index))
        retained_indices = np.sort(index)
        
        similar_columns = {}
        
        for i in filtered_indices:
            similar_columns[X_full_data.columns[i]] = []
            for j in retained_indices:
                if np.array_equal(X_full_array[:, i], X_full_array[:, j]):
                    similar_columns[X_full_data.columns[i]].append(X_full_data.columns[j])
        
        print("Filtered columns and their similar retained columns:")
        for filtered_col, similar_cols in similar_columns.items():
            print(f"Filtered Column: {filtered_col}, Similar to Retained Columns: {similar_cols}")
        
        
        # Filter the DataFrame based on unique columns
        X_full_filtered = X_full_data.iloc[:, retained_indices]
        # X_full_filtered = X_full_data.iloc[:, np.sort(index)]
        # Add the sample_ids column back to the filtered DataFrame
        X_full_filtered.insert(0, 'sample_ids', sample_ids)
        print(X_full_filtered.shape)

        # convert new dataset to csv
        X_full_filtered.to_csv(datapath + '/data/pheno' + str(type) + '/x_matrix_filtered.csv')
        

def split_train_test_data(datapath, type, train_ratio):
    """
    Read the preprocessed data after matching input features and labels,
        + in: path to X and y
        + out: the X and y as type numpy array
    """
    
    X = pd.read_csv(datapath + '/data/pheno' + str(type) + '/x_matrix.csv')
    y = pd.read_csv(datapath + '/data/pheno' + str(type) + '/y_matrix.csv')

    X_nparray = X.iloc[:,2:]
    y_nparray = y.iloc[:,2]

    X_train, X_test, y_train, y_test = train_test_split(X_nparray, y_nparray, train_size=train_ratio, random_state=43, shuffle=True)
    
    label_train = str(int(train_ratio*10))
    label_test = str(int(10-train_ratio*10))

    X_train.to_csv(datapath + '/data/ratio_' + label_train + '_' + label_test + '/pheno' + str(type) + '/x_train.csv')
    y_train.to_csv(datapath + '/data/ratio_' + label_train + '_' + label_test + '/pheno' + str(type) + '/y_train.csv')
    X_test.to_csv(datapath + '/data/ratio_' + label_train + '_' + label_test + '/pheno' + str(type) + '/x_test.csv')
    y_test.to_csv(datapath + '/data/ratio_' + label_train + '_' + label_test + '/pheno' + str(type) + '/y_test.csv')


def map_indices_easypheno(datapath, type, train_ratio):

    label_train = str(int(train_ratio*10))
    label_test = str(int(10-train_ratio*10))
    # Load the CSV files
    x_full = pd.read_csv(datapath + '/data/pheno' + str(type) + '/x_matrix.csv', index_col=0)
    df_phenotype = pd.read_csv(datapath + '/data/pheno' + str(type) + '/y_matrix.csv', index_col=0)
    
    # Processing for Train set and test set
    train_indices = pd.read_csv(datapath + '/data/easypheno_indices/ratio_' + label_train + '_' + label_test + '/pheno' + str(type) + '/final_model_test_results_ratio' + label_train + label_test + '.csv')
    train_indices = train_indices.filter(['sample_ids_train'], axis=1).dropna().astype(int)

    test_indices = pd.read_csv(datapath + '/data/easypheno_indices/ratio_' + label_train + '_' + label_test + '/pheno' + str(type) + '/final_model_test_results_ratio' + label_train + label_test + '.csv')
    test_indices = test_indices.filter(['sample_ids_test'], axis=1).dropna().astype(int)

    # Merge the DataFrames based on the 'sample_ids' and 'sample_ids_train' columns
    merged_df1_df2_train = pd.merge(train_indices, x_full, left_on='sample_ids_train', right_on='sample_ids')
    # merged_df1_df2_train.drop(columns=['sample_ids'], axis=1, inplace=True)
    final_merged_df_train = pd.merge(merged_df1_df2_train, df_phenotype, left_on='sample_ids_train', right_on='sample_ids')
    # final_merged_df_train.drop(columns=['sample_ids'], axis=1, inplace=True)

    merged_df1_df2_test = pd.merge(test_indices, x_full, left_on='sample_ids_test', right_on='sample_ids')
    # merged_df1_df2_test.drop(columns=['sample_ids'], axis=1, inplace=True)
    final_merged_df_test = pd.merge(merged_df1_df2_test, df_phenotype, left_on='sample_ids_test', right_on='sample_ids')
    # final_merged_df_test.drop(columns=['sample_ids'], axis=1, inplace=True)

    # create new x_train, y_train
    x_train = final_merged_df_train.iloc[:,:final_merged_df_train.shape[1]-1]
    y_train = final_merged_df_train[['sample_ids_train', 'pheno'+str(type)]]
    x_test = final_merged_df_test.iloc[:,:final_merged_df_test.shape[1]-1]
    y_test = final_merged_df_test[['sample_ids_test', 'pheno'+str(type)]]

    # convert new dataset to csv
    x_train.to_csv(datapath + '/data/easypheno_indices/ratio_' + label_train + '_' + label_test + '/pheno' + str(type) + '/x_train.csv', index=False)
    y_train.to_csv(datapath + '/data/easypheno_indices/ratio_' + label_train + '_' + label_test + '/pheno' + str(type) + '/y_train.csv', index=False)
    x_test.to_csv(datapath + '/data/easypheno_indices/ratio_' + label_train + '_' + label_test + '/pheno' + str(type) + '/x_test.csv', index=False)
    y_test.to_csv(datapath + '/data/easypheno_indices/ratio_' + label_train + '_' + label_test + '/pheno' + str(type) + '/y_test.csv', index=False)

# -------------------------------------------------------------
#  2. Load data for Regular-based type model
# -------------------------------------------------------------
def load_split_regular_data(datapath, type, train_ratio):

    label_train = str(int(train_ratio*10))
    label_test = str(int(10-train_ratio*10))

    X_train = pd.read_csv(datapath + '/data/easypheno_indices/ratio_' + label_train + '_' + label_test + '/pheno' + str(type) + '/x_train.csv')
    y_train = pd.read_csv(datapath + '/data/easypheno_indices/ratio_' + label_train + '_' + label_test + '/pheno' + str(type) + '/y_train.csv')
    X_test = pd.read_csv(datapath + '/data/easypheno_indices/ratio_' + label_train + '_' + label_test + '/pheno' + str(type) + '/x_test.csv')
    y_test = pd.read_csv(datapath + '/data/easypheno_indices/ratio_' + label_train + '_' + label_test + '/pheno' + str(type) + '/y_test.csv')

    X_train_nparray, y_train_nparray = X_train.iloc[:,1:].to_numpy(), y_train.iloc[:,1].to_numpy()
    X_test_nparray, y_test_nparray = X_test.iloc[:,1:].to_numpy(), y_test.iloc[:,1].to_numpy()

    # Create a list of sequences 
    # from [['C' 'G' 'A' ... 'A' 'G' 'G'], [..], ..] --> ['CGA..AGG', '...',...]
    X_list_train = [''.join(seq) for seq in X_train_nparray]
    X_list_test = [''.join(seq) for seq in X_test_nparray]


    return X_list_train, y_train_nparray, X_list_test, y_test_nparray

# -------------------------------------------------------------
#  3. Load data for Chromosome-based type model
# -------------------------------------------------------------

def split_into_chromosome_train(datapath, type, train_ratio):

    label_train = str(int(train_ratio*10))
    label_test = str(int(10-train_ratio*10))

    X_train = pd.read_csv(datapath + '/data/easypheno_indices/ratio_' + label_train + '_' + label_test + '/pheno' + str(type) + '/x_train.csv')

    # Load data into a DataFrame object:
    df_train = pd.DataFrame(X_train)

    # Sorting the Column name. Format: chromosome_position
    # DataFrame.columns: return the column labels
    # for each column name: we'll provide a tuple (chromosome, position) to the sorting function. 
    # As tuples are sorted by comparing them field by field, this will effectively sort column name by chromosome, then by position.
    sorted_columns_train = sorted(df_train.columns[1:], key=lambda x: (int(x.split("_")[0]), int(x.split("_")[1])))

    # Sort the DataFrame based on the sorted columns
    sorted_df_train = df_train[["sample_ids_train"] + sorted_columns_train]

    # Filter groups of chromosomes
    chr1_columns = [column for column in sorted_df_train.columns if column.startswith('1_')]
    df_chr1 = sorted_df_train[['sample_ids_train'] + chr1_columns]

    chr2_columns = [column for column in sorted_df_train.columns if column.startswith('2_')]
    df_chr2 = sorted_df_train[['sample_ids_train'] + chr2_columns]

    chr3_columns = [column for column in sorted_df_train.columns if column.startswith('3_')]
    df_chr3 = sorted_df_train[['sample_ids_train'] + chr3_columns]

    chr4_columns = [column for column in sorted_df_train.columns if column.startswith('4_')]
    df_chr4 = sorted_df_train[['sample_ids_train'] + chr4_columns]

    chr5_columns = [column for column in sorted_df_train.columns if column.startswith('5_')]
    df_chr5 = sorted_df_train[['sample_ids_train'] + chr5_columns]

    # Convert into numpy array
    X_chr1, X_chr2, X_chr3, X_chr4, X_chr5 = df_chr1.iloc[:,1:].to_numpy(), df_chr2.iloc[:,1:].to_numpy(), df_chr3.iloc[:,1:].to_numpy(), df_chr4.iloc[:,1:].to_numpy(), df_chr5.iloc[:,1:].to_numpy()

    # Create a list of sequences 
    # from [['C' 'G' 'A' ... 'A' 'G' 'G'], [..], ..] --> ['CGA..AGG', '...',...]
    X_chr1_train = [''.join(seq) for seq in X_chr1]
    X_chr2_train = [''.join(seq) for seq in X_chr2]
    X_chr3_train = [''.join(seq) for seq in X_chr3]
    X_chr4_train = [''.join(seq) for seq in X_chr4]
    X_chr5_train = [''.join(seq) for seq in X_chr5]

    return X_chr1_train, X_chr2_train, X_chr3_train, X_chr4_train, X_chr5_train

def split_into_chromosome_test(datapath, type, train_ratio):

    label_train = str(int(train_ratio*10))
    label_test = str(int(10-train_ratio*10))

    X_test = pd.read_csv(datapath + '/data/easypheno_indices/ratio_' + label_train + '_' + label_test + '/pheno' + str(type) + '/x_test.csv')

    # Load data into a DataFrame object:
    df_train = pd.DataFrame(X_test)

    # Sorting the Column name. Format: chromosome_position
    # DataFrame.columns: return the column labels
    # for each column name: we'll provide a tuple (chromosome, position) to the sorting function. 
    # As tuples are sorted by comparing them field by field, this will effectively sort column name by chromosome, then by position.
    sorted_columns_train = sorted(df_train.columns[1:], key=lambda x: (int(x.split("_")[0]), int(x.split("_")[1])))

    # Sort the DataFrame based on the sorted columns
    sorted_df_train = df_train[["sample_ids_test"] + sorted_columns_train]

    # Filter groups of chromosomes
    chr1_columns = [column for column in sorted_df_train.columns if column.startswith('1_')]
    df_chr1 = sorted_df_train[['sample_ids_test'] + chr1_columns]

    chr2_columns = [column for column in sorted_df_train.columns if column.startswith('2_')]
    df_chr2 = sorted_df_train[['sample_ids_test'] + chr2_columns]

    chr3_columns = [column for column in sorted_df_train.columns if column.startswith('3_')]
    df_chr3 = sorted_df_train[['sample_ids_test'] + chr3_columns]

    chr4_columns = [column for column in sorted_df_train.columns if column.startswith('4_')]
    df_chr4 = sorted_df_train[['sample_ids_test'] + chr4_columns]

    chr5_columns = [column for column in sorted_df_train.columns if column.startswith('5_')]
    df_chr5 = sorted_df_train[['sample_ids_test'] + chr5_columns]

    # Convert into numpy array
    X_chr1, X_chr2, X_chr3, X_chr4, X_chr5 = df_chr1.iloc[:,1:].to_numpy(), df_chr2.iloc[:,1:].to_numpy(), df_chr3.iloc[:,1:].to_numpy(), df_chr4.iloc[:,1:].to_numpy(), df_chr5.iloc[:,1:].to_numpy()

    # Create a list of sequences 
    # from [['C' 'G' 'A' ... 'A' 'G' 'G'], [..], ..] --> ['CGA..AGG', '...',...]
    X_chr1_test = [''.join(seq) for seq in X_chr1]
    X_chr2_test = [''.join(seq) for seq in X_chr2]
    X_chr3_test = [''.join(seq) for seq in X_chr3]
    X_chr4_test = [''.join(seq) for seq in X_chr4]
    X_chr5_test = [''.join(seq) for seq in X_chr5]

    return X_chr1_test, X_chr2_test, X_chr3_test, X_chr4_test, X_chr5_test

def split_into_chromosome_train_realworld(datapath, type, train_ratio):

    label_train = str(int(train_ratio*10))
    label_test = str(int(10-train_ratio*10))

    X_train = pd.read_csv(datapath + '/data/easypheno_indices/ratio_' + label_train + '_' + label_test + '/pheno' + str(type) + '/x_train.csv')

    # Load data into a DataFrame object:
    df_train = pd.DataFrame(X_train)

    # Sorting the Column name. Format: chromosome_position
    # DataFrame.columns: return the column labels
    # for each column name: we'll provide a tuple (chromosome, position) to the sorting function. 
    # As tuples are sorted by comparing them field by field, this will effectively sort column name by chromosome, then by position.
    sorted_columns_train = sorted(df_train.columns[1:], key=lambda x: (int(x.split("_")[0]), int(x.split("_")[1])))

    # Sort the DataFrame based on the sorted columns
    sorted_df_train = df_train[["sample_ids_train"] + sorted_columns_train]

    # Filter groups of chromosomes
    chr1_columns = [column for column in sorted_df_train.columns if column.endswith('_1')]
    df_chr1 = sorted_df_train[['sample_ids_train'] + chr1_columns]

    chr2_columns = [column for column in sorted_df_train.columns if column.endswith('_2')]
    df_chr2 = sorted_df_train[['sample_ids_train'] + chr2_columns]

    chr3_columns = [column for column in sorted_df_train.columns if column.endswith('_3')]
    df_chr3 = sorted_df_train[['sample_ids_train'] + chr3_columns]

    chr4_columns = [column for column in sorted_df_train.columns if column.endswith('_4')]
    df_chr4 = sorted_df_train[['sample_ids_train'] + chr4_columns]

    chr5_columns = [column for column in sorted_df_train.columns if column.endswith('_5')]
    df_chr5 = sorted_df_train[['sample_ids_train'] + chr5_columns]

    # Convert into numpy array
    X_chr1, X_chr2, X_chr3, X_chr4, X_chr5 = df_chr1.iloc[:,1:].to_numpy(), df_chr2.iloc[:,1:].to_numpy(), df_chr3.iloc[:,1:].to_numpy(), df_chr4.iloc[:,1:].to_numpy(), df_chr5.iloc[:,1:].to_numpy()

    # Create a list of sequences 
    # from [['C' 'G' 'A' ... 'A' 'G' 'G'], [..], ..] --> ['CGA..AGG', '...',...]
    X_chr1_train = [''.join(seq) for seq in X_chr1]
    X_chr2_train = [''.join(seq) for seq in X_chr2]
    X_chr3_train = [''.join(seq) for seq in X_chr3]
    X_chr4_train = [''.join(seq) for seq in X_chr4]
    X_chr5_train = [''.join(seq) for seq in X_chr5]

    return X_chr1_train, X_chr2_train, X_chr3_train, X_chr4_train, X_chr5_train

def split_into_chromosome_test_realworld(datapath, type, train_ratio):

    label_train = str(int(train_ratio*10))
    label_test = str(int(10-train_ratio*10))

    X_test = pd.read_csv(datapath + '/data/easypheno_indices/ratio_' + label_train + '_' + label_test + '/pheno' + str(type) + '/x_test.csv')

    # Load data into a DataFrame object:
    df_train = pd.DataFrame(X_test)

    # Sorting the Column name. Format: chromosome_position
    # DataFrame.columns: return the column labels
    # for each column name: we'll provide a tuple (chromosome, position) to the sorting function. 
    # As tuples are sorted by comparing them field by field, this will effectively sort column name by chromosome, then by position.
    sorted_columns_train = sorted(df_train.columns[1:], key=lambda x: (int(x.split("_")[0]), int(x.split("_")[1])))

    # Sort the DataFrame based on the sorted columns
    sorted_df_train = df_train[["sample_ids_test"] + sorted_columns_train]

    # Filter groups of chromosomes
    chr1_columns = [column for column in sorted_df_train.columns if column.endswith('_1')]
    df_chr1 = sorted_df_train[['sample_ids_test'] + chr1_columns]

    chr2_columns = [column for column in sorted_df_train.columns if column.endswith('_2')]
    df_chr2 = sorted_df_train[['sample_ids_test'] + chr2_columns]

    chr3_columns = [column for column in sorted_df_train.columns if column.endswith('_3')]
    df_chr3 = sorted_df_train[['sample_ids_test'] + chr3_columns]

    chr4_columns = [column for column in sorted_df_train.columns if column.endswith('_4')]
    df_chr4 = sorted_df_train[['sample_ids_test'] + chr4_columns]

    chr5_columns = [column for column in sorted_df_train.columns if column.endswith('_5')]
    df_chr5 = sorted_df_train[['sample_ids_test'] + chr5_columns]

    # Convert into numpy array
    X_chr1, X_chr2, X_chr3, X_chr4, X_chr5 = df_chr1.iloc[:,1:].to_numpy(), df_chr2.iloc[:,1:].to_numpy(), df_chr3.iloc[:,1:].to_numpy(), df_chr4.iloc[:,1:].to_numpy(), df_chr5.iloc[:,1:].to_numpy()

    # Create a list of sequences 
    # from [['C' 'G' 'A' ... 'A' 'G' 'G'], [..], ..] --> ['CGA..AGG', '...',...]
    X_chr1_test = [''.join(seq) for seq in X_chr1]
    X_chr2_test = [''.join(seq) for seq in X_chr2]
    X_chr3_test = [''.join(seq) for seq in X_chr3]
    X_chr4_test = [''.join(seq) for seq in X_chr4]
    X_chr5_test = [''.join(seq) for seq in X_chr5]

    return X_chr1_test, X_chr2_test, X_chr3_test, X_chr4_test, X_chr5_test

def load_split_y_data(datapath, type, train_ratio):

    label_train = str(int(train_ratio*10))
    label_test = str(int(10-train_ratio*10))

    y_train = pd.read_csv(datapath + '/data/easypheno_indices/ratio_' + label_train + '_' + label_test + '/pheno' + str(type) + '/y_train.csv')
    y_test = pd.read_csv(datapath + '/data/easypheno_indices/ratio_' + label_train + '_' + label_test + '/pheno' + str(type) + '/y_test.csv')
    

    y_train_nparray = y_train.iloc[:,1].to_numpy()
    y_test_nparray = y_test.iloc[:,1].to_numpy()


    return y_train_nparray, y_test_nparray

# -------------------------------------------------------------
#  3. Load data for Augmentation_Chromosome type model
# -------------------------------------------------------------
def augment_data(X_chrs, y, augmentation_folds=2):
    augmented_X = []
    augmented_y = []
    unique_samples = set()
    
    # Determine the number of samples based on the length of the first chromosome
    num_samples = len(X_chrs[0])

    for fold in range(augmentation_folds):
        for i in range(num_samples):
            sample_chrs = [X_chrs[chr_idx][i] for chr_idx in range(len(X_chrs))]
            if fold > 0:  # Don't shuffle for the original fold (fold=0)
                random.shuffle(sample_chrs)          

            augmented_sample = ''.join(sample_chrs)
            if augmented_sample not in unique_samples:
                unique_samples.add(augmented_sample)
                augmented_X.append(augmented_sample)
                augmented_y.append(y[i])

    return augmented_X, np.asarray(augmented_y)

def load_train_augmented_data(datapath, type, train_ratio):

    label_train = str(int(train_ratio*10))
    label_test = str(int(10-train_ratio*10))

    X_train = pd.read_csv(datapath + '/data/easypheno_indices/ratio_' + label_train + '_' + label_test + '/pheno' + str(type) + '/x_train.csv')
    y_train = pd.read_csv(datapath + '/data/easypheno_indices/ratio_' + label_train + '_' + label_test + '/pheno' + str(type) + '/y_train.csv')

    # Load data into a DataFrame object:
    df_train = pd.DataFrame(X_train)

    # Sorting the Column name. Format: chromosome_position
    # DataFrame.columns: return the column labels
    # for each column name: we'll provide a tuple (chromosome, position) to the sorting function. 
    # As tuples are sorted by comparing them field by field, this will effectively sort column name by chromosome, then by position.
    sorted_columns_train = sorted(df_train.columns[1:], key=lambda x: (int(x.split("_")[0]), int(x.split("_")[1])))

    # Sort the DataFrame based on the sorted columns
    sorted_df_train = df_train[["sample_ids_train"] + sorted_columns_train]

    # Filter groups of chromosomes
    chr1_columns = [column for column in sorted_df_train.columns if column.startswith('1_')]
    df_chr1 = sorted_df_train[['sample_ids_train'] + chr1_columns]

    chr2_columns = [column for column in sorted_df_train.columns if column.startswith('2_')]
    df_chr2 = sorted_df_train[['sample_ids_train'] + chr2_columns]

    chr3_columns = [column for column in sorted_df_train.columns if column.startswith('3_')]
    df_chr3 = sorted_df_train[['sample_ids_train'] + chr3_columns]

    chr4_columns = [column for column in sorted_df_train.columns if column.startswith('4_')]
    df_chr4 = sorted_df_train[['sample_ids_train'] + chr4_columns]

    chr5_columns = [column for column in sorted_df_train.columns if column.startswith('5_')]
    df_chr5 = sorted_df_train[['sample_ids_train'] + chr5_columns]

    # Convert into numpy array
    X_chr1, X_chr2, X_chr3, X_chr4, X_chr5 = df_chr1.iloc[:,1:].to_numpy(), df_chr2.iloc[:,1:].to_numpy(), df_chr3.iloc[:,1:].to_numpy(), df_chr4.iloc[:,1:].to_numpy(), df_chr5.iloc[:,1:].to_numpy()

    # Create a list of sequences 
    # from [['C' 'G' 'A' ... 'A' 'G' 'G'], [..], ..] --> ['CGA..AGG', '...',...]
    X_chr1_train = [''.join(seq) for seq in X_chr1]
    X_chr2_train = [''.join(seq) for seq in X_chr2]
    X_chr3_train = [''.join(seq) for seq in X_chr3]
    X_chr4_train = [''.join(seq) for seq in X_chr4]
    X_chr5_train = [''.join(seq) for seq in X_chr5]

    y_train_nparray = y_train.iloc[:,1].to_numpy()

    num_augmentation_folds = 5  # Number of times to augment the data
    augmented_X_train, augmented_y_train = augment_data(
        [X_chr1_train, X_chr2_train, X_chr3_train, X_chr4_train, X_chr5_train],
        y_train_nparray,
        augmentation_folds=num_augmentation_folds
    )

    return augmented_X_train, augmented_y_train

def load_test_augmented_data(datapath, type, train_ratio):

    label_train = str(int(train_ratio*10))
    label_test = str(int(10-train_ratio*10))
    
    X_test = pd.read_csv(datapath + '/data/easypheno_indices/ratio_' + label_train + '_' + label_test + '/pheno' + str(type) + '/x_test.csv')
    y_test = pd.read_csv(datapath + '/data/easypheno_indices/ratio_' + label_train + '_' + label_test + '/pheno' + str(type) + '/y_test.csv')

    X_test_nparray, y_test_nparray = X_test.iloc[:,1:].to_numpy(), y_test.iloc[:,1].to_numpy()

    # Create a list of sequences 
    # from [['C' 'G' 'A' ... 'A' 'G' 'G'], [..], ..] --> ['CGA..AGG', '...',...]
    X_list_test = [''.join(seq) for seq in X_test_nparray]


    return X_list_test, y_test_nparray
