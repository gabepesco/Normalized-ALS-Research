import os
from tqdm import tqdm
import scipy.sparse as sp
import numpy as np
import functions
import pickle

def main():
    optimal_matrix = sp.load_npz('data/optimal_confidence_matrix.npz')
    csr = sp.csr_matrix(optimal_matrix)
    train, test, masked = functions.get_train_test_masked(csr)
    os.environ['MKL_NUM_THREADS'] = '1'
    model = functions.get_model(train, alpha=512, reg=.0625)
    pickle.dump(model, open("model.p", "wb"))

main()