import os
from tqdm import tqdm
import scipy.sparse as sp
import numpy as np
import functions
import pickle


def main():
    int_matrix = sp.load_npz('data/sorted_interaction_matrix.npz')
    sh, mb = functions.get_sh_mb(int_matrix)

    matrix = sp.load_npz('data/bm25_confidence_matrix.npz')
    train, test, masked = functions.get_train_test_masked(matrix)

    os.environ['MKL_NUM_THREADS'] = '1'
    model = functions.get_model(train, alpha=512, reg=.25)
    with open(r"data/bm25_model.pickle", "wb") as output_file:
        pickle.dump(model, output_file)

    # with open(r"data/bm25_model.pickle", "rb") as input_file:
    #     model = pickle.load(input_file)

    auc, sh_auc, mb_auc, lt_auc = functions.score_model(model, test, masked, sh, mb)
    with open(r"data/model_scores.pickle", "wb") as output_file:
        pickle.dump((auc, sh_auc, mb_auc, lt_auc), output_file)


main()
