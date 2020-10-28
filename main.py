import parser
from tqdm import tqdm
import scipy.sparse as sp
import numpy as np

int_matrix, bm25_conf_matrix, len_norm_conf_matrix, optimal_conf_matrix = parser.get_matrices()

