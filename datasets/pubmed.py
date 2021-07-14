import pandas as pd
import pickle
import re
from tqdm.notebook import tqdm
import numpy as np
import torch

TRAIN_RANGE = (0, 7000)
VAL_RANGE = (7000, 14000)
TEST_RANGE = (14000, 19717)
NUM_INPUT_FEATURES = 500
NUM_CLASSES = 3

def get_feature_vectors(pubmed_path):
    with open(pubmed_path + 'Pubmed-Diabetes.NODE.paper.tab', 'rb') as f:
        for i, line in tqdm(enumerate(f.readlines())):
            if i == 0:
                continue
            elif i == 1:
                init_string = line.decode('utf-8')[15:]
                words = re.sub(r'\tnumeric:', '', init_string).split(':0.0')[:-1]
                dataset = pd.DataFrame(0, index=range(19717), columns=['paper_id', 'label'] + sorted(words))
                continue

            line_info = line.decode('utf-8').split('\t')[:-1]
            for j, elem in enumerate(line_info):
                if j == 0:
                    dataset.iloc[i-2, 0] = int(elem)
                elif j == 1:
                    dataset.iloc[i-2, 1] = int(elem.split('=')[1])
                else:
                    col_name = elem.split('=')[0]
                    col_value = float(elem.split('=')[1])
                    dataset.loc[i-2, col_name] = col_value

    paper_id_map = {key: value for value, key in enumerate(dataset['paper_id'].values)}
    labels = dataset['label'].values
    dataset = dataset.drop(columns=['paper_id', 'label']).values

    return dataset, labels, paper_id_map


def get_adjacency_matrix(pubmed_path, paper_id_map):    
    adjacency_matrix = np.zeros(shape=(19717, 19717)) - 100000

    with open(pubmed_path + 'Pubmed-Diabetes.DIRECTED.cites.tab', 'rb') as f:
        for line in f.readlines()[2:]:
            line_info = line.decode('utf-8').split('\t')
            paper_from = paper_id_map[int(line_info[1][6:])]
            paper_to = paper_id_map[int(line_info[3][6:])]
            adjacency_matrix[paper_from, paper_to] = 1
            
    return adjacency_matrix


def load_pubmed(pubmed_path, device='cpu'):
    node_features, node_labels, paper_id_map = get_feature_vectors(pubmed_path)
    adjacency_matrix = get_adjacency_matrix(pubmed_path, paper_id_map)
    
    
    adjacency_matrix = torch.tensor(adjacency_matrix, device=device).float()
    node_labels = torch.tensor(node_labels, dtype=torch.long, device=device)
    node_features = torch.tensor(node_features, device=device).float()
    
    train_indices = torch.arange(TRAIN_RANGE[0], TRAIN_RANGE[1], device=device)
    val_indices = torch.arange(VAL_RANGE[0], VAL_RANGE[1], device=device)
    test_indices = torch.arange(TEST_RANGE[0], TEST_RANGE[1], device=device)
    return node_features, adjacency_matrix, node_labels, train_indices, val_indices, test_indices