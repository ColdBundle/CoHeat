import os

import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset, DataLoader


def print_statistics(X, string):
    """
    Print the statistics of a dataset
    """
    print('-'*10 + string + '-'*10)
    print(f'Avg non-zeros in row:    {X.sum(1).mean(0).item():8.4f}')
    nonzero_row_indice, nonzero_col_indice = X.nonzero()
    unique_nonzero_row_indice = np.unique(nonzero_row_indice)
    unique_nonzero_col_indice = np.unique(nonzero_col_indice)
    print(f'Ratio of non-empty rows: {len(unique_nonzero_row_indice)/X.shape[0]:8.4f}')
    print(f'Ratio of non-empty cols: {len(unique_nonzero_col_indice)/X.shape[1]:8.4f}')
    print(f'Density of matrix:       {len(nonzero_row_indice)/(X.shape[0]*X.shape[1]):8.4f}')


class TrnDataset(Dataset):
    """
    Class of training dataset
    """
    def __init__(self, conf, u_b_pairs, u_b_graph, num_bundles, neg_sample=1):
        self.conf = conf
        self.u_b_pairs = u_b_pairs
        self.u_b_graph = u_b_graph
        self.num_bundles = num_bundles
        self.neg_sample = neg_sample

    def __getitem__(self, index):
        user_b, pos_bundle = self.u_b_pairs[index]
        all_bundles = [pos_bundle]

        while True:
            i = np.random.randint(self.num_bundles)
            if self.u_b_graph[user_b, i] == 0 and not i in all_bundles:                                                          
                all_bundles.append(i)                                                                                                   
                if len(all_bundles) == self.neg_sample+1:                                                                               
                    break                                                                                                               

        return torch.LongTensor([user_b]), torch.LongTensor(all_bundles)

    def __len__(self):
        return len(self.u_b_pairs)


class TestDataset(Dataset):
    """
    Class of test dataset
    """
    def __init__(self, u_b_pairs, u_b_graph, u_b_graph_train, num_users, num_bundles):
        self.u_b_pairs = u_b_pairs
        self.u_b_graph = u_b_graph
        self.train_mask_u_b = u_b_graph_train

        self.num_users = num_users
        self.num_bundles = num_bundles

        self.users = torch.arange(num_users, dtype=torch.long).unsqueeze(dim=1)
        self.bundles = torch.arange(num_bundles, dtype=torch.long)

    def __getitem__(self, index):
        u_b_grd = torch.from_numpy(self.u_b_graph[index].toarray()).squeeze()
        u_b_mask = torch.from_numpy(self.train_mask_u_b[index].toarray()).squeeze()

        return index, u_b_grd, u_b_mask

    def __len__(self):
        return self.u_b_graph.shape[0]


class Datasets():
    """
    Class of datasets
    """
    def __init__(self, conf):
        self.path = conf['data_path']
        self.name = conf['dataset']
        batch_size_train = conf['batch_size_train']
        batch_size_test = conf['batch_size_test']

        self.num_users, self.num_bundles, self.num_items = self.get_data_size()

        b_i_graph = self.get_bi()
        u_i_pairs, u_i_graph = self.get_ui()

        u_b_pairs_train, u_b_graph_train = self.get_ub("train")
        u_b_pairs_val, u_b_graph_val = self.get_ub("tune")
        u_b_pairs_test, u_b_graph_test = self.get_ub("test")

        self.bundle_train_data = TrnDataset(conf, u_b_pairs_train, u_b_graph_train, self.num_bundles)
        self.bundle_val_data = TestDataset(u_b_pairs_val, u_b_graph_val, u_b_graph_train, self.num_users, self.num_bundles)
        self.bundle_test_data = TestDataset(u_b_pairs_test, u_b_graph_test, u_b_graph_train, self.num_users, self.num_bundles)

        self.graphs = [u_b_graph_train, u_i_graph, b_i_graph]

        self.train_loader = DataLoader(self.bundle_train_data, batch_size=batch_size_train, shuffle=True, num_workers=10, drop_last=True)
        self.val_loader = DataLoader(self.bundle_val_data, batch_size=batch_size_test, shuffle=False, num_workers=20)
        self.test_loader = DataLoader(self.bundle_test_data, batch_size=batch_size_test, shuffle=False, num_workers=20)

        self.bundles_freq = np.asarray(u_b_graph_train.sum(0)).squeeze()

    def get_data_size(self):
        """
        Get the numbers of users, bundles, and items
        """
        name = self.name
        if "_" in name:
            name = name.split("_")[0]
        with open(os.path.join(self.path, self.name, f'{name}_data_size.txt'), 'r') as f:
            return [int(s) for s in f.readline().split('\t')][:3]

    def get_bi(self):
        """
        Get a bundle-item graph
        """
        with open(os.path.join(self.path, self.name, 'bundle_item.txt'), 'r') as f:
            b_i_pairs = list(map(lambda s: tuple(int(i) for i in s[:-1].split('\t')), f.readlines()))
        indices = np.array(b_i_pairs, dtype=np.int32)
        values = np.ones(len(b_i_pairs), dtype=np.float32)
        b_i_graph = sp.coo_matrix(
            (values, (indices[:, 0], indices[:, 1])), shape=(self.num_bundles, self.num_items)).tocsr()
        print_statistics(b_i_graph, 'B-I statistics')
        return b_i_graph

    def get_ui(self):
        """
        Get a user-item graph
        """
        with open(os.path.join(self.path, self.name, 'user_item.txt'), 'r') as f:
            u_i_pairs = list(map(lambda s: tuple(int(i) for i in s[:-1].split('\t')), f.readlines()))
        indices = np.array(u_i_pairs, dtype=np.int32)
        values = np.ones(len(u_i_pairs), dtype=np.float32)
        u_i_graph = sp.coo_matrix( 
            (values, (indices[:, 0], indices[:, 1])), shape=(self.num_users, self.num_items)).tocsr()
        print_statistics(u_i_graph, 'U-I statistics')
        return u_i_pairs, u_i_graph

    def get_ub(self, task):
        """
        Get a user-bundle graph
        """
        with open(os.path.join(self.path, self.name, f'user_bundle_{task}.txt'), 'r') as f:
            u_b_pairs = list(map(lambda s: tuple(int(i) for i in s[:-1].split('\t')), f.readlines()))
        indices = np.array(u_b_pairs, dtype=np.int32)
        values = np.ones(len(u_b_pairs), dtype=np.float32)
        u_b_graph = sp.coo_matrix(
            (values, (indices[:, 0], indices[:, 1])), shape=(self.num_users, self.num_bundles)).tocsr()
        print_statistics(u_b_graph, f'U-B statistics in {task}')
        return u_b_pairs, u_b_graph
