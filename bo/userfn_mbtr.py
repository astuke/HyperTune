import matplotlib
matplotlib.use('Agg')

import os
import itertools
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
from sklearn.model_selection import ShuffleSplit
import ase.io
from dscribe.descriptors import MBTR
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
    
    
    iteration_start=time.time()    
    
    
    ## KRR hyperparameters
    alpha_exp = -x[0][0]
    gamma_exp = -x[0][1]

    alpha = 10**alpha_exp
    gamma = 10**gamma_exp
   
    ### MBTR hyperparameters
    sigma2_exp = -x[0][2]
    sigma3_exp = -x[0][3]

    sigma1 = 0.2
    sigma2 = 10**sigma2_exp
    sigma3 = 10**sigma3_exp

    ### scaling for weighting function
    s2 = x[0][4]
    s3 = x[0][5]
    

    # write variables to file
    f = open('variables.in', 'w')
    f.write(str(alpha))
    f.write("\n")
    f.write(str(gamma))
    f.close()


    time_cv_array = []
    mbtr_start=time.time()

    #### Load training data

    data = pd.read_json("../data/data_train_1k.json")


    global create

    def create(i_samples):
        """This is the function that is called by each process but with different
        parts of the data.
        """
        n_i_samples = len(i_samples)
        i_res = lil_matrix((n_i_samples, n_features))
        for i, i_sample in enumerate(i_samples):
            feat = mbtr_desc.create(i_sample)
            i_res[i, :] = feat
            #print("{} %".format((i+1)/n_i_samples*100))
        return i_res

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

    ## define MBTR
    mbtr_desc = MBTR(
        species=atomic_numbers,
        k1={
            "geometry": {"function": "atomic_number"},
            "grid": {"min": min_atomic_number, "max": max_atomic_number, "n": 200, "sigma": sigma1},
        },
        k2={
            "geometry": {"function": "inverse_distance"},
            "grid": {"min": 0, "max": 1, "n": 200, "sigma": sigma2},
            "weighting": {"function": "exponential", "scale": s2, "cutoff": 1e-3},
        },
        k3={
            "geometry": {"function": "cosine"},
            "grid": {"min": -1, "max": 1, "n": 200, "sigma": sigma3},
            "weighting": {"function": "exponential", "scale": s3, "cutoff": 1e-3},
        },
        periodic=False,
        normalization="l2_each",
    )#.create(ase_train_cv)

    ############# create MBTR for data ##############################################################################

    # Split the data into roughly equivalent chunks for each process
    n_proc = 24  # How many processes are spawned
    k, m = divmod(len(ase_mol), n_proc)
    atoms_split = (ase_mol[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n_proc))
    n_features = int(mbtr_desc.get_number_of_features())

    # Initialize a pool of processes, and tell each process in the pool to
    # handle a different part of the data
    with multiprocessing.Pool(processes=n_proc) as pool:
        res = pool.map(create, atoms_split)  # pool.map keeps the order

    # Save results
    n_samples = len(ase_mol)
    mbtr_mol = lil_matrix((n_samples, n_features))

    i_sample = 0
    for i, i_res in enumerate(res):
        i_n_samples = i_res.shape[0]
        mbtr_mol[i_sample:i_sample+i_n_samples, :] = i_res
        i_sample += i_n_samples

    ################# split MBTR and homo array into 5 different parts    

    ### mbtr to csr
    mbtr = mbtr_mol.tocsr()
    
    ### define index
    index=np.arange(np.shape(mbtr)[0])
    ### shuffle index
    np.random.shuffle(index)
    ### return shuffled mbtr matrix
    shuffled_mbtr = mbtr[index, :]
    ### return shuffled homo array
    homo=np.array(homo)
    shuffled_homo = homo[index]

    ### split data into 5 equal parts
    select_ind_1 = np.arange(0,200,1)
    mbtr_1 = shuffled_mbtr[select_ind_1, :]
    homo_1 = shuffled_homo[select_ind_1]

    select_ind_2 = np.arange(200,400,1)
    mbtr_2 = shuffled_mbtr[select_ind_2, :]
    homo_2 = shuffled_homo[select_ind_2]
    
    select_ind_3 = np.arange(400,600,1)
    mbtr_3 = shuffled_mbtr[select_ind_3, :]
    homo_3 = shuffled_homo[select_ind_3]

    select_ind_4 = np.arange(600, 800,1)
    mbtr_4 = shuffled_mbtr[select_ind_4, :]
    homo_4 = shuffled_homo[select_ind_4]

    select_ind_5 = np.arange(800, 1000,1)
    mbtr_5 = shuffled_mbtr[select_ind_5, :]
    homo_5 = shuffled_homo[select_ind_5]
    

    ##### arrange data into training and validation sets
    mbtr_train_1 = vstack((mbtr_2, mbtr_3, mbtr_4, mbtr_5))
    mbtr_val_1 = mbtr_1
    homo_train_1 = np.concatenate((homo_2, homo_3, homo_4, homo_5), axis=0)
    homo_val_1 = homo_1

    mbtr_train_2 = vstack((mbtr_3, mbtr_4, mbtr_5, mbtr_1))
    mbtr_val_2 = mbtr_2
    homo_train_2 = np.concatenate((homo_3, homo_4, homo_5, homo_1), axis=0)
    homo_val_2 = homo_2

    mbtr_train_3 = vstack((mbtr_4, mbtr_5, mbtr_1, mbtr_2))
    mbtr_val_3 = mbtr_3
    homo_train_3 = np.concatenate((homo_4, homo_5, homo_1, homo_2), axis=0)
    homo_val_3 = homo_3
    
    mbtr_train_4 = vstack((mbtr_5, mbtr_1, mbtr_2, mbtr_3))
    mbtr_val_4 = mbtr_4
    homo_train_4 = np.concatenate((homo_5, homo_1, homo_2, homo_3), axis=0)
    homo_val_4 = homo_4

    mbtr_train_5 = vstack((mbtr_1, mbtr_2, mbtr_3, mbtr_4))
    mbtr_val_5 = mbtr_5
    homo_train_5 = np.concatenate((homo_1, homo_2, homo_3, homo_4), axis=0)
    homo_val_5 = homo_5
    
    print("Finished building MBTR")
    mbtr_end=time.time()
    mbtr_time = np.round(mbtr_end - mbtr_start, decimals=3)

    with open('mbtr_train_1.pkl', 'wb') as f1, open('mbtr_train_2.pkl', 'wb') as f2, open('mbtr_train_3.pkl', 'wb') as f3, open('mbtr_train_4.pkl', 'wb') as f4, open('mbtr_train_5.pkl', 'wb') as f5:
        pickle.dump(mbtr_train_1, f1)
        pickle.dump(mbtr_train_2, f2)
        pickle.dump(mbtr_train_3, f3)
        pickle.dump(mbtr_train_4, f4)
        pickle.dump(mbtr_train_5, f5)

    with open('mbtr_val_1.pkl', 'wb') as f1, open('mbtr_val_2.pkl', 'wb') as f2, open('mbtr_val_3.pkl', 'wb') as f3, open('mbtr_val_4.pkl', 'wb') as f4, open('mbtr_val_5.pkl', 'wb') as f5:
        pickle.dump(mbtr_val_1, f1)
        pickle.dump(mbtr_val_2, f2)
        pickle.dump(mbtr_val_3, f3)
        pickle.dump(mbtr_val_4, f4)
        pickle.dump(mbtr_val_5, f5)

    with open('homo_train_1.pkl', 'wb') as f1, open('homo_train_2.pkl', 'wb') as f2, open('homo_train_3.pkl', 'wb') as f3, open('homo_train_4.pkl', 'wb') as f4, open('homo_train_5.pkl', 'wb') as f5:
        pickle.dump(homo_train_1, f1)
        pickle.dump(homo_train_2, f2)
        pickle.dump(homo_train_3, f3)
        pickle.dump(homo_train_4, f4)
        pickle.dump(homo_train_5, f5)
    
    with open('homo_val_1.pkl', 'wb') as f1, open('homo_val_2.pkl', 'wb') as f2, open('homo_val_3.pkl', 'wb') as f3, open('homo_val_4.pkl', 'wb') as f4, open('homo_val_5.pkl', 'wb') as f5:
        pickle.dump(homo_val_1, f1)
        pickle.dump(homo_val_2, f2)
        pickle.dump(homo_val_3, f3)
        pickle.dump(homo_val_4, f4)
        pickle.dump(homo_val_5, f5)
     
    subprocess.call('submit_cv.sh')


    pp_start=time.time()
    
    file1 =  open('mae1.txt', 'r')
    mae1 = file1.read()
    output.write("mae1:"+ mae1+"\n")
    ftime1 =  open('cv_time_1.txt', 'r')
    time1 = ftime1.read()
    
    file2 =  open('mae2.txt', 'r')
    mae2 = file2.read()
    output.write("mae2:"+ mae2+"\n")
    ftime2 =  open('cv_time_2.txt', 'r')
    time2 = ftime2.read()
    
    file3 =  open('mae3.txt', 'r')
    mae3 = file3.read()
    output.write("mae3:"+ mae3+"\n")
    ftime3 =  open('cv_time_3.txt', 'r')
    time3 = ftime3.read()
    
    file4 =  open('mae4.txt', 'r')
    mae4 = file4.read()
    output.write("mae4:"+ mae4+"\n")
    ftime4 =  open('cv_time_4.txt', 'r')
    time4 = ftime4.read()
    
    file5 =  open('mae5.txt', 'r')
    mae5 = file5.read()
    output.write("mae5:"+ mae5+"\n")
    ftime5 =  open('cv_time_5.txt', 'r')
    time5 = ftime5.read()
    
    while not (mae1 and mae2 and mae3 and mae4 and mae5 and time1 and time2 and time3 and time4 and time5):
        if (mae1 and mae2 and mae3 and mae4 and mae5 and time1 and time2 and time3 and time4 and time5):                     
            break
        
        output.write("Waiting for all cv rounds to finish..."+"\n")
        time.sleep(5)
        
        file1 =  open('mae1.txt', 'r')
        mae1 = file1.read()
        output.write("mae1:"+ mae1+"\n")
        ftime1 =  open('cv_time_1.txt', 'r')
        time1 = ftime1.read()
        
        file2 =  open('mae2.txt', 'r')
        mae2 = file2.read()
        output.write("mae2:"+ mae2+"\n")
        ftime2 =  open('cv_time_2.txt', 'r')
        time2 = ftime2.read()
        
        file3 =  open('mae3.txt', 'r')
        mae3 = file3.read()
        output.write("mae3:"+ mae3+"\n")
        ftime3 =  open('cv_time_3.txt', 'r')
        time3 = ftime3.read()
        
        file4 =  open('mae4.txt', 'r')
        mae4 = file4.read()
        output.write("mae4:"+ mae4+"\n")
        ftime4 =  open('cv_time_4.txt', 'r')
        time4 = ftime4.read()
        
        file5 =  open('mae5.txt', 'r')
        mae5 = file5.read()
        output.write("mae5:"+ mae5+"\n")
        ftime5 =  open('cv_time_5.txt', 'r')
        time5 = ftime5.read()
    
    MAE_list = [mae1, mae2, mae3, mae4, mae5]
    output.write("All cv rounds finished."+"\n")
    output.write("MAEs of CV rounds:"+ str(MAE_list)+"\n")
    MAE_list = np.array(MAE_list).astype(np.float)
    avg_MAE = np.mean(MAE_list)
    output.write("Average MAE: %s eV" %avg_MAE+"\n")
    
    output.write("BREAKDOWN OF TIMINGS"+"\n")
    output.write("Time to load data and build MBTRs: %f s" %mbtr_time+"\n")
    
    cv_time_list = [time1, time2, time3, time4, time5]
    output.write("CV timings:"+ str(cv_time_list)+"\n")
    cv_time_list = np.array(cv_time_list).astype(np.float)
    avg_cv_time = np.mean(cv_time_list)
    output.write("Average time for CV loop: %f s" %avg_cv_time+"\n")
    
    pp_end=time.time()
    pp_time = np.round(pp_end - pp_start, decimals=3)
    output.write("Postprocessing time: %f s" %pp_time +"\n")

    iteration_end=time.time()
    iteration_time = np.round(iteration_end - iteration_start, decimals=3)
    output.write("Total iteration time: %f s" %iteration_time +"\n")
    output.close()
    
    if os.path.isfile('results/df_results_mbtr.json'):
        df_results = pd.read_json('results/df_results_mbtr.json', orient='split')
        iteration = len(df_results) + 1
        print("iteration:", iteration)
        row = [iteration, avg_MAE, iteration_time, mbtr_time, avg_cv_time, alpha, gamma, sigma2, sigma3]
        df_results.loc[len(df_results)] = row
        df_results.to_json('results/df_results_mbtr.json', orient='split')
    else:
        df_results = pd.DataFrame([[1, avg_MAE, iteration_time, mbtr_time, avg_cv_time, alpha, gamma, sigma2, sigma3]], columns=['iteration', 'avg_MAE', 'iteration_time', 'mbtr_time', 'avg_cv_time','alpha', 'gamma', 'sigma2', 'sigma3'])
        df_results.to_json('results/df_results_mbtr.json', orient='split')
    
    return avg_MAE
