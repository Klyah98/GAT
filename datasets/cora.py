import pickle
import torch
import scipy.sparse as sp
import numpy as np
from tqdm.notebook import tqdm

TRAIN_RANGE = (0, 140)
VAL_RANGE = (140, 140+500)
TEST_RANGE = (1708, 1708+1000)
NUM_INPUT_FEATURES = 1433
NUM_CLASSES = 7

def pickle_read(filepath):
    with open(filepath, 'rb') as f:
        file = pickle.load(f)
    return file


def normalize_features_sparse(node_features_sparse):
    """
    Normalize input feature vectors (becomes easier to learn)
    """
    node_features_sum = np.array(node_features_sparse.sum(-1))
    node_features_inv_sum = np.power(node_features_sum, -1).squeeze()
    node_features_inv_sum[np.isinf(node_features_inv_sum)] = 1.
    diagonal_inv_features_sum_matrix = sp.diags(node_features_inv_sum)
    return diagonal_inv_features_sum_matrix.dot(node_features_sparse)


def build_adjacency_matrix(adjacency_dict):
    num_nodes = len(adjacency_dict)
    adjacency_matrix = sp.csr_matrix(sp.diags([1]*num_nodes, shape=(num_nodes, num_nodes)))
    for node in tqdm(adjacency_dict.keys()):
        for neighbor in adjacency_dict[node]:
            adjacency_matrix[node, neighbor] = 1
    return adjacency_matrix


def load_cora(cora_path, device='cpu'):

    node_features = pickle_read(cora_path + 'node_features.csr')
    adjacency_dict = pickle_read(cora_path + 'adjacency_list.dict')
    node_labels = pickle_read(cora_path + 'node_labels.npy')
    
    adjacency_matrix = torch.tensor(build_adjacency_matrix(adjacency_dict).toarray(), dtype=torch.long, device=device)
    adjacency_matrix[adjacency_matrix == 0] = -100000
    node_labels = torch.tensor(node_labels, dtype=torch.long, device=device)
    node_features = torch.tensor(node_features.todense(), device=device)
    
    train_indices = torch.arange(TRAIN_RANGE[0], TRAIN_RANGE[1], device=device)
    val_indices = torch.arange(VAL_RANGE[0], VAL_RANGE[1], device=device)
    test_indices = torch.arange(TEST_RANGE[0], TEST_RANGE[1], device=device)
    return node_features, adjacency_matrix, node_labels, train_indices, val_indices, test_indices