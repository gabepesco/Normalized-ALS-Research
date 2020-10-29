import os
from tqdm import tqdm
import scipy.sparse as sp
import numpy as np
import functions
import pickle


def main():
    optimal_matrix = sp.load_npz('data/optimal_confidence_matrix.npz')
    sh, mb = functions.get_sh_mb(optimal_matrix)
    train, test, masked = functions.get_train_test_masked(optimal_matrix)

    os.environ['MKL_NUM_THREADS'] = '1'
    # model = functions.get_model(train, alpha=512, reg=.0625)
    # with open(r"data/model.pickle", "wb") as output_file:
    #     pickle.dump(model, output_file)

    with open(r"data/model.pickle", "rb") as input_file:
        model = pickle.load(input_file)

    sorted_recs = functions.get_recs(model, test)
    with open(r"data/sorted_recs.pickle", "wb") as output_file:
        pickle.dump(sorted_recs, output_file)

    auc, sh_auc, mb_auc, lt_auc = functions.get_scores(masked, sorted_recs, sh, mb)


main()
