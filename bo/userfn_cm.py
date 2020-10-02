import matplotlib
matplotlib.use('Agg')

import os
import itertools
#import seaborn
import pandas as pd
import numpy as np
import sys
import sklearn
from scipy.spatial import distance
from scipy.sparse import csr_matrix
from scipy.sparse import issparse
import matplotlib.pyplot as plt
import math
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.metrics import mean_absolute_error
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from operator import itemgetter
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
from sklearn.metrics import make_scorer
import sklearn.metrics.pairwise
from sklearn import preprocessing
import json
import time
import pandas as pd
from sklearn.model_selection import ShuffleSplit
import ase.io
from dscribe.descriptors import CoulombMatrix
from dscribe.utils.stats import system_stats
import multiprocessing
from scipy.sparse import lil_matrix, save_npz
import ast
from functools import partial
from sklearn.utils import shuffle
from io import StringIO
import subprocess
from subprocess import *
import pickle
import pandas as pd
from scipy.sparse import hstack
from scipy.sparse import vstack
import tempfile

def f(x):
    
    filename = 'boss_outfile.txt'
    if os.path.exists(filename):
        append_write = 'a' # append if already exists
    else:
        append_write = 'w' # make a new file if not
    
    iteration_start=time.time()    
    
    
    ## KRR parameters
    alpha_exp = -x[0][0]
    gamma_exp = -x[0][1]

    alpha = 10**alpha_exp
    gamma = 10**gamma_exp
   

    # write variables to file
    f = open('variables.in', 'w')
    f.write(str(alpha))
    f.write("\n")
    f.write(str(gamma))
    f.close()


    time_cv_array = []
    MAE_list = []
    cv_time_list = []

    #### Load data

    data = pd.read_json("../data/data_train_1k.json")



    ###### extract xyz coordinates and HOMOs from dataframe
    homo_array = []
    out_mol = StringIO()

    for i, row in data.iterrows():
        homo = row[0][1]
        homo_array.append(homo)
        x = "".join(row.molecule)
        #print("x:", x)
        out_mol.write(x)

    homo = np.array(homo_array)
    homo = [float(x) for x in homo]
    #print(homo_train)
    ase_mol = list(ase.io.iread(out_mol, format="xyz"))

    ## Load statistics from the dataset
    stats = system_stats(ase_mol)
    atomic_numbers = stats["atomic_numbers"]
    max_atomic_number = stats["max_atomic_number"]
    min_atomic_number = stats["min_atomic_number"]
    min_distance = stats["min_distance"]

    cm_desc = CoulombMatrix(
        #n_atoms_max=max_atomic_number,
	n_atoms_max=29,
	permutation="sorted_l2",
	#sparse=True
    )


    
    ############# create CM for data ##############################################################################
    cm_start=time.time()
    cm = cm_desc.create(ase_mol)
    cm_end=time.time()
    cm_time = np.round(cm_end - cm_start, decimals=3)


    ################# split CM and homo array into 5 different parts    


    ### mbtr to csr
    #mbtr = mbtr_mol.tocsr()
    
    ## select 3 random rows of mbtr matrix
    #select_ind = np.array([0,2,4])
    #mbtr[select_ind, :]

    ## see contents: todense()
    
    ### define index
    index=np.arange(np.shape(cm)[0])
    ### shuffle index
    np.random.shuffle(index)
    ### return shuffled cm matrix
    shuffled_cm = cm[index, :]
    ### return shuffled homo array
    homo=np.array(homo)
    shuffled_homo = homo[index]
    # shuffled_homo.tolist()

    ### split data into 5 equal parts
    select_ind_1 = np.arange(0,200,1)
    cm_1 = shuffled_cm[select_ind_1, :]
    homo_1 = shuffled_homo[select_ind_1]

    select_ind_2 = np.arange(200,400,1)
    cm_2 = shuffled_cm[select_ind_2, :]
    homo_2 = shuffled_homo[select_ind_2]
    
    select_ind_3 = np.arange(400,600,1)
    cm_3 = shuffled_cm[select_ind_3, :]
    homo_3 = shuffled_homo[select_ind_3]

    select_ind_4 = np.arange(600, 800,1)
    cm_4 = shuffled_cm[select_ind_4, :]
    homo_4 = shuffled_homo[select_ind_4]

    select_ind_5 = np.arange(800, 1000,1)
    cm_5 = shuffled_cm[select_ind_5, :]
    homo_5 = shuffled_homo[select_ind_5]

    ##### arrange data into training and validation sets
    cm_train_1 = np.concatenate((cm_2, cm_3, cm_4, cm_5))
    print("cm_train_1:", cm_train_1)
    print("Length cm_train:", len(cm_train_1))
    print("Shape cm_train:", cm_train_1.shape)
    cm_val_1 = cm_1
    homo_train_1 = np.concatenate((homo_2, homo_3, homo_4, homo_5))
    homo_val_1 = homo_1

    cm_train_2 = np.concatenate((cm_3, cm_4, cm_5, cm_1))
    #print("Length cm_train:", cm_train_2.shape)
    cm_val_2 = cm_2
    #print("Length cm_val:", cm_val_2.shape)
    homo_train_2 = np.concatenate((homo_3, homo_4, homo_5, homo_1))
    #print("Length homo_train:", len(homo_train_2))
    homo_val_2 = homo_2
    #print("Length homo_val:", len(homo_val_2))

    cm_train_3 = np.concatenate((cm_4, cm_5, cm_1, cm_2))
    #print("Length cm_train:", cm_train_3.shape)
    cm_val_3 = cm_3
    #print("Length cm_val:", cm_val_3.shape)
    homo_train_3 = np.concatenate((homo_4, homo_5, homo_1, homo_2))
    homo_val_3 = homo_3
    
    cm_train_4 = np.concatenate((cm_5, cm_1, cm_2, cm_3))
    #print("Length cm_train:", cm_train_4.shape)
    cm_val_4 = cm_4
    #print("Length cm_val:", cm_val_4.shape)
    homo_train_4 = np.concatenate((homo_5, homo_1, homo_2, homo_3))
    homo_val_4 = homo_4

    cm_train_5 = np.concatenate((cm_1, cm_2, cm_3, cm_4))
    #print("Length cm_train:", cm_train_5.shape)
    cm_val_5 = cm_5
    #print("Length cm_val:", cm_val_5.shape)
    homo_train_5 = np.concatenate((homo_1, homo_2, homo_3, homo_4))
    homo_val_5 = homo_5

    cm_train = [cm_train_1, cm_train_2, cm_train_3, cm_train_4, cm_train_5]
    cm_val = [cm_val_1, cm_val_2, cm_val_3, cm_val_4, cm_val_5]
    homo_train = [homo_train_1, homo_train_2, homo_train_3, homo_train_4, homo_train_5]
    homo_val = [homo_val_1, homo_val_2, homo_val_3, homo_val_4, homo_val_5]
    


    ### KRR ###############
    for cm_train_i, homo_train_i, cm_val_i, homo_val_i in zip(cm_train, homo_train, cm_val, homo_val):
        cv_start = time.time()
	            
        model = KernelRidge(alpha=alpha, kernel='laplacian', gamma=gamma)

        model.fit(cm_train_i, homo_train_i)

        y_true = homo_val_i
        y_pred = model.predict(cm_val_i)

        MAE = mean_absolute_error(y_true, y_pred)
        cv_end = time.time()
        cv_time_list.append(np.round(cv_end - cv_start, decimals=3))
        MAE_list.append(MAE)
        print("MAE:", MAE)


    avg_MAE = np.mean(MAE_list)
    
    avg_cv_time = np.mean(cv_time_list)

    iteration_end=time.time()
    iteration_time = np.round(iteration_end - iteration_start, decimals=3)
    print("iteration time:", iteration_time)
    
    if os.path.isfile('results/df_results_cm.json'):
        df_results = pd.read_json('results/df_results_cm.json', orient='split')
        iteration = len(df_results) + 1
        print("iteration:", iteration)
        row = [iteration, avg_MAE, iteration_time, cm_time, avg_cv_time, alpha, gamma]
        df_results.loc[len(df_results)] = row
        df_results.to_json('results/df_results_cm.json', orient='split')
        print(df_results)
    else:
        df_results = pd.DataFrame([[1, avg_MAE, iteration_time, cm_time, avg_cv_time, alpha, gamma]], columns=['iteration', 'avg_MAE', 'iteration_time', 'cm_time', 'avg_cv_time','alpha', 'gamma'])
        df_results.to_json('results/df_results_cm.json', orient='split')

    
    return avg_MAE
