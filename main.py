import os
import scipy.sparse as sp
import functions
import pickle
import csv

def main():
    int_matrix = sp.load_npz('data/pref_matrix.npz')
    sh, mb = functions.get_sh_mb(int_matrix)
    # total_interactions = int_matrix.sum()
    del int_matrix


    matrix_name = "pref"
    matrix_filename = f'data/{matrix_name}_matrix.npz'
    matrix = sp.load_npz(matrix_filename)
    train, test, masked = functions.get_train_test_masked(matrix)

    os.environ['MKL_NUM_THREADS'] = '1'

    for a in (128, 256, 512):
        alpha, reg, factors = a, .25, 128
        model_name = f'data/models/{matrix_name}_a{alpha}_r{reg}_f{factors}_model.pickle'

        try:
            with open(model_name, "rb") as input_file:
                model = pickle.load(input_file)

        except FileNotFoundError:
            model = functions.get_model(train, alpha=alpha, reg=reg, factors=factors)
            with open(model_name, 'wb') as output_file:
                pickle.dump(model, output_file)

        auc, sh_auc, mb_auc, lt_auc = functions.score_model(model, test, masked, sh, mb)
        results = [matrix_name, alpha, reg, factors, auc, sh_auc, mb_auc, lt_auc]
        with open('data/results.csv', 'a') as result_file:
            wr = csv.writer(result_file, dialect='excel')
            wr.writerow(results)


main()
