import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


def cal_bpr_loss(pred):
    """
    Compute BPR loss
    """
    negs = pred[:, 1].unsqueeze(1)
    pos = pred[:, 0].unsqueeze(1)
    loss = - torch.mean(torch.log(torch.sigmoid(pos - negs))) # [bs]
    return loss


def laplace_transform(graph):
    """
    Get a laplace transform of a graph
    """
    rowsum_sqrt = sp.diags(1/(np.sqrt(graph.sum(axis=1).A.ravel()) + 1e-8))
    colsum_sqrt = sp.diags(1/(np.sqrt(graph.sum(axis=0).A.ravel()) + 1e-8))
    graph = rowsum_sqrt @ graph @ colsum_sqrt
    return graph


def to_tensor(graph):
    """
    Convert to a sparse tensor
    """
    graph = graph.tocoo()
    values = graph.data
    indices = np.vstack((graph.row, graph.col))
    graph = torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(graph.shape))
    return graph


def np_edge_dropout(values, dropout_ratio):
    """
    Perform a edge dropout
    """
    mask = np.random.choice([0, 1], size=(len(values),), p=[dropout_ratio, 1-dropout_ratio])
    values = mask * values
    return values


class CoHeat(nn.Module):
    """
    Class of CoHeat model
    """
    def __init__(self, conf, raw_graph, bundles_freq):
        """
        Initialize the model
        """
        super().__init__()
        self.conf = conf
        self.device = self.conf["device"]
        self.embedding_size = conf["embedding_size"]
        self.embed_L2_norm = conf["lambda2"]
        self.num_users = conf["num_users"]
        self.num_bundles = conf["num_bundles"]
        self.num_items = conf["num_items"]

        self.init_emb()

        assert isinstance(raw_graph, list)
        self.ub_graph, self.ui_graph, self.bi_graph = raw_graph

        self.get_aff_graph_ori()
        self.get_hist_graph_ori()
        self.get_agg_graph_ori()

        self.get_aff_graph()
        self.get_hist_graph()
        self.get_agg_graph()

        self.num_layers = self.conf["num_layers"]

        self.bundles_freq = torch.FloatTensor(bundles_freq).to(self.device)

    def init_emb(self):
        """
        Initialize embeddings
        """
        self.users_feature = nn.Parameter(torch.FloatTensor(self.num_users, self.embedding_size))
        nn.init.xavier_normal_(self.users_feature)
        self.bundles_feature = nn.Parameter(torch.FloatTensor(self.num_bundles, self.embedding_size))
        nn.init.xavier_normal_(self.bundles_feature)
        self.items_feature = nn.Parameter(torch.FloatTensor(self.num_items, self.embedding_size))
        nn.init.xavier_normal_(self.items_feature)
        self.IL_layer = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
        nn.init.xavier_normal_(self.IL_layer.weight)
        self.BL_layer = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
        nn.init.xavier_normal_(self.BL_layer.weight)

    def get_aff_graph(self):
        """
        Get an affiliation-view graph
        """
        ui_graph = self.ui_graph
        device = self.device
        modification_ratio = self.conf["aff_ed_ratio"]

        item_level_graph = sp.bmat([[sp.csr_matrix((ui_graph.shape[0], ui_graph.shape[0])), ui_graph], [ui_graph.T, sp.csr_matrix((ui_graph.shape[1], ui_graph.shape[1]))]])
        if modification_ratio != 0:
            graph = item_level_graph.tocoo()
            values = np_edge_dropout(graph.data, modification_ratio)
            item_level_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()

        self.aff_view_graph = to_tensor(laplace_transform(item_level_graph)).to(device)

    def get_aff_graph_ori(self):
        """
        Get the original affiliation-view graph
        """
        ui_graph = self.ui_graph
        device = self.device
        item_level_graph = sp.bmat([[sp.csr_matrix((ui_graph.shape[0], ui_graph.shape[0])), ui_graph], [ui_graph.T, sp.csr_matrix((ui_graph.shape[1], ui_graph.shape[1]))]])
        self.aff_view_graph_ori = to_tensor(laplace_transform(item_level_graph)).to(device)

    def get_hist_graph(self):
        """
        Get a history-view graph
        """
        ub_graph = self.ub_graph
        device = self.device
        modification_ratio = self.conf["hist_ed_ratio"]

        bundle_level_graph = sp.bmat([[sp.csr_matrix((ub_graph.shape[0], ub_graph.shape[0])), ub_graph], [ub_graph.T, sp.csr_matrix((ub_graph.shape[1], ub_graph.shape[1]))]])

        if modification_ratio != 0:
            graph = bundle_level_graph.tocoo()
            values = np_edge_dropout(graph.data, modification_ratio)
            bundle_level_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()

        self.hist_view_graph = to_tensor(laplace_transform(bundle_level_graph)).to(device)

    def get_hist_graph_ori(self):
        """
        Get the original history-view graph
        """
        ub_graph = self.ub_graph
        device = self.device
        bundle_level_graph = sp.bmat([[sp.csr_matrix((ub_graph.shape[0], ub_graph.shape[0])), ub_graph], [ub_graph.T, sp.csr_matrix((ub_graph.shape[1], ub_graph.shape[1]))]])
        self.hist_view_graph_ori = to_tensor(laplace_transform(bundle_level_graph)).to(device)

    def get_agg_graph(self):
        """
        Get an aggregation graph
        """
        bi_graph = self.bi_graph
        device = self.device

        modification_ratio = self.conf["agg_ed_ratio"]
        graph = bi_graph.tocoo()
        values = np_edge_dropout(graph.data, modification_ratio)
        bi_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()

        bundle_size = bi_graph.sum(axis=1) + 1e-8
        bi_graph = sp.diags(1/bundle_size.A.ravel()) @ bi_graph
        self.bundle_agg_graph = to_tensor(bi_graph).to(device)

    def get_agg_graph_ori(self):
        """
        Get the origianl aggregation graph
        """
        bi_graph = self.bi_graph
        device = self.device

        bundle_size = bi_graph.sum(axis=1) + 1e-8
        bi_graph = sp.diags(1/bundle_size.A.ravel()) @ bi_graph
        self.bundle_agg_graph_ori = to_tensor(bi_graph).to(device)

    def one_propagate(self, graph, A_feature, B_feature):
        """
        One propagation
        """
        features = torch.cat((A_feature, B_feature), 0)
        all_features = [features]

        for i in range(self.num_layers):
            features = torch.spmm(graph, features)
            features = features / (i+2)
            all_features.append(F.normalize(features, p=2, dim=1))

        all_features = torch.stack(all_features, 1)
        all_features = torch.sum(all_features, dim=1).squeeze(1)

        A_feature, B_feature = torch.split(all_features, (A_feature.shape[0], B_feature.shape[0]), 0)

        return A_feature, B_feature

    def get_aff_bundle_rep(self, aff_items_feature, test):
        """
        Get affiliation-view bundle representations
        """
        if test:
            aff_bundles_feature = torch.matmul(self.bundle_agg_graph_ori, aff_items_feature)
        else:
            aff_bundles_feature = torch.matmul(self.bundle_agg_graph, aff_items_feature)

        return aff_bundles_feature

    def propagate(self, test=False):
        """
        Propagate the representations
        """
        # Affiliation-view
        if test:
            aff_users_feature, aff_items_feature = self.one_propagate(self.aff_view_graph_ori, self.users_feature, self.items_feature)
        else:
            aff_users_feature, aff_items_feature = self.one_propagate(self.aff_view_graph, self.users_feature, self.items_feature)

        aff_bundles_feature = self.get_aff_bundle_rep(aff_items_feature, test)

        # History-view
        if test:
            hist_users_feature, hist_bundles_feature = self.one_propagate(self.hist_view_graph_ori, self.users_feature, self.bundles_feature)
        else:
            hist_users_feature, hist_bundles_feature = self.one_propagate(self.hist_view_graph, self.users_feature, self.bundles_feature)

        users_feature = [aff_users_feature, hist_users_feature]
        bundles_feature = [aff_bundles_feature, hist_bundles_feature]

        return users_feature, bundles_feature

    def cal_a_loss(self, x, y):
        """
        Compute alignment loss
        """
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        return (x - y).norm(p=2, dim=1).pow(2).mean()

    def cal_u_loss(self, x):
        """
        Compute uniformity loss
        """
        x = F.normalize(x, dim=-1)
        return torch.pdist(x, p=2).pow(2).mul(-2).exp().mean().log()

    def cal_loss(self, users_feature, bundles_feature, bundles_gamma):
        """
        Compute BPR loss and contrastive loss
        """
        aff_users_feature, hist_users_feature = users_feature
        aff_bundles_feature, hist_bundles_feature = bundles_feature
        aff_bundles_feature_ = aff_bundles_feature * (1 - bundles_gamma.unsqueeze(2))
        hist_bundles_feature_ = hist_bundles_feature * bundles_gamma.unsqueeze(2)
        pred = torch.sum(aff_users_feature * aff_bundles_feature_, 2) + torch.sum(hist_users_feature * hist_bundles_feature_, 2)
        bpr_loss = cal_bpr_loss(pred)

        aff_bundles_feature = aff_bundles_feature[:, 0, :]
        hist_bundles_feature = hist_bundles_feature[:, 0, :]
        bundles_align = self.cal_a_loss(aff_bundles_feature, hist_bundles_feature)
        bundles_uniform = (self.cal_u_loss(aff_bundles_feature) + self.cal_u_loss(hist_bundles_feature)) / 2
        bundles_c_loss = bundles_align + bundles_uniform

        aff_users_feature = aff_users_feature[:, 0, :]
        hist_users_feature = hist_users_feature[:, 0, :]
        users_align = self.cal_a_loss(aff_users_feature, hist_users_feature)
        users_uniform = (self.cal_u_loss(aff_users_feature) + self.cal_u_loss(hist_users_feature)) / 2
        users_c_loss = users_align + users_uniform

        c_loss = (bundles_c_loss + users_c_loss) / 2

        return bpr_loss, c_loss

    def forward(self, batch, ED_drop=False, psi=1.):
        """
        Forward the model
        """
        if ED_drop:
            self.get_aff_graph()
            self.get_hist_graph()
            self.get_agg_graph()

        users, bundles = batch
        users_feature, bundles_feature = self.propagate()

        users_embedding = [i[users].expand(-1, bundles.shape[1], -1) for i in users_feature]
        bundles_embedding = [i[bundles] for i in bundles_feature]
        bundles_gamma = torch.tanh(self.bundles_freq / psi)
        bundles_gamma = bundles_gamma[bundles.flatten()].reshape(bundles.shape)

        bpr_loss, c_loss = self.cal_loss(users_embedding, bundles_embedding, bundles_gamma)

        return bpr_loss, c_loss

    def evaluate(self, propagate_result, users, psi=1.):
        """
        Evaluate the model
        """
        users_feature, bundles_feature = propagate_result
        aff_users_feature, hist_users_feature = [i[users] for i in users_feature]
        aff_bundles_feature, hist_bundles_feature = bundles_feature
        bundles_gamma = torch.tanh(self.bundles_freq / psi)
        aff_bundles_feature_ = aff_bundles_feature * (1 - bundles_gamma.unsqueeze(1))
        hist_bundles_feature_ = hist_bundles_feature * bundles_gamma.unsqueeze(1)
        scores = torch.mm(aff_users_feature, aff_bundles_feature_.t()) + torch.mm(hist_users_feature, hist_bundles_feature_.t())
        return scores
