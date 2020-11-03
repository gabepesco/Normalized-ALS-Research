import os
import scipy.sparse as sp
import functions
import pickle
import csv


def main():
    # set these!
    matrix_name = "bm25_len_norm_conf"
    save_model = True

    matrix_filename = f'data/{matrix_name}_matrix.npz'
    matrix = sp.load_npz(matrix_filename)
    ratio = 65464776.0 / matrix.sum()  # total interactions

    alpha, reg, factors = round(3360 * ratio), 1.19, 128

    # set this to true if you want to save the model
    print(f'matrix: {matrix_name}, alpha: {alpha}, reg: {reg}, factors: {factors}')

    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['MKL_DEBUG_CPU_TYPE'] = '5'
    sh, mb = 2641, 22530

    train, test, masked = functions.get_train_test_masked(matrix)

    print(f'alpha: {alpha}, reg: {reg}, factors: {factors}')
    model_name = f'data/models/{matrix_name}_a{alpha}_r{reg}_f{factors}_model.pickle'

    try:
        with open(model_name, "rb") as input_file:
            model = pickle.load(input_file)

    except FileNotFoundError:
        model = functions.get_model(train, alpha=alpha, reg=reg, factors=factors)

        if save_model:
            try:
                with open(model_name, 'wb') as output_file:
                    pickle.dump(model, output_file)
            except FileNotFoundError:
                os.mkdir('data/models')
                with open(model_name, 'wb') as output_file:
                    pickle.dump(model, output_file)
    del train

    pgap, auc, sh_auc, mb_auc, lt_auc = functions.score_model(model, test, masked, sh, mb)
    results = [matrix_name, alpha, reg, factors, pgap, auc, sh_auc, mb_auc, lt_auc]
    with open('data/results.csv', 'a') as result_file:
        wr = csv.writer(result_file, dialect='excel')
        wr.writerow(results)


main()
