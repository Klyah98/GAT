import torch
import pandas as pd
from tqdm.notebook import tqdm
import numpy as np


TRAIN_RANGE = (0, 800)
VAL_RANGE = (800, 2000)
TEST_RANGE = (2000, 3312)
NUM_INPUT_FEATURES = 3703
NUM_CLASSES = 6


def get_node_features(citeseer_path):    
    node_features = pd.DataFrame([], index=range(3312), columns=range(3705))
    with open(citeseer_path + 'citeseer.content', 'rb') as f:
        for i, line in tqdm(enumerate(f.readlines()), total=3312):
            node_features.loc[i, :] = (line.decode('utf-8').split('\t'))
        paper_names = node_features[0].values
        paper_labels = node_features[3704]
        unique_labels = {key: value for value, key in enumerate(paper_labels.unique())}
        paper_id_map = {key: value for value, key in enumerate(paper_names)}
        node_labels = [unique_labels[label] for label in paper_labels]
        node_features = node_features.drop(columns=[0, 3704]).astype(int).values
    return node_features, paper_id_map, node_labels


def get_adjacency_matrix(citeseer_path, paper_id_map):    
    adjacency_matrix = np.eye(3312)
    with open(citeseer_path + 'citeseer.cites', 'rb') as f:
        for line in f.readlines():
            citation = line.decode('utf-8').split()
            if citation[0] in paper_id_map.keys() and citation[1] in paper_id_map.keys():
                idx1, idx2 = paper_id_map[citation[0]], paper_id_map[citation[1]]
                adjacency_matrix[idx1, idx2] = 1
    return adjacency_matrix


def load_citeseer(citeseer_path, device='cpu'):
    node_features, paper_id_map, node_labels = get_node_features(citeseer_path)
    adjacency_matrix = get_adjacency_matrix(citeseer_path, paper_id_map)
    
    adjacency_matrix[adjacency_matrix == 0] = -100000
    adjacency_matrix = torch.tensor(adjacency_matrix, dtype=torch.long, device=device)
    node_labels = torch.tensor(node_labels, dtype=torch.long, device=device)
    node_features = torch.tensor(node_features, device=device).float()
    
    train_indices = torch.arange(TRAIN_RANGE[0], TRAIN_RANGE[1], device=device)
    val_indices = torch.arange(VAL_RANGE[0], VAL_RANGE[1], device=device)
    test_indices = torch.arange(TEST_RANGE[0], TEST_RANGE[1], device=device)
    return node_features, adjacency_matrix, node_labels, train_indices, val_indices, test_indices