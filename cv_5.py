import matplotlib
matplotlib.use('Agg')

import os
import itertools
#import seaborn
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
import multiprocessing
from scipy.sparse import lil_matrix, save_npz
import ast
from functools import partial
from sklearn.utils import shuffle
from io import StringIO
import pickle

start_time = time.time()
print("Starting cross-validation round 5")

with open("mbtr_train_5.pkl", "rb") as f1, open("mbtr_val_5.pkl", "rb") as f2, open("homo_train_5.pkl", "rb") as f3, open("homo_val_5.pkl", "rb") as f4:
    mbtr_train = pickle.load(f1)
    mbtr_val = pickle.load(f2)
    homo_train = pickle.load(f3)
    homo_val = pickle.load(f4)

infile = 'variables.in'
with open(infile) as fp:
    alpha = float(fp.readline())
    gamma = float(fp.readline())

outfile = open("mae5.txt", "w")

######################## KRR ####################################################################

model = KernelRidge(alpha=alpha, kernel='rbf', gamma=gamma)

model.fit(mbtr_train, homo_train)

y_true, y_pred = homo_val, model.predict(mbtr_val)

MAE = mean_absolute_error(y_true, y_pred)
#MAE_list.append(MAE)
print("MAE_5:", MAE)
outfile.write(str(MAE))
outfile.close()

end_time=time.time()
cv_time = np.round(end_time - start_time, decimals=3)
print("Time for cross-validation loop 5 (KRR):", cv_time)
timefile = open("cv_time_5.txt", "w")
timefile.write(str(cv_time))
