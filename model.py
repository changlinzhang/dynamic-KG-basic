import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable

import numpy as np
import time
import random
from sklearn.metrics import f1_score
from collections import defaultdict

from encoders import Encoder
from aggregators import MeanAggregator

"""
Simple Unsupervised GraphSAGE model as well as examples running the model
on the ICEWS and GDELT datasets.
"""


class UnsupervisedGraphSage(nn.Module):

    def __init__(self, enc, config):  # not num_classes as parameter
        super(UnsupervisedGraphSage, self).__init__()
        self.enc = enc
        # self.xent = nn.CrossEntropyLoss()

        # self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim))
        init.xavier_uniform(self.weight)

        self.filter = config.filter
        self.optimizer = config.optimizer
        self.entity_total = config.entity_total
        self.relation_total = config.relation_total
        self.batch_size = config.batch_size
        self.neg_sample_weights = 1.0

        # config.learning_rate = args.learning_rate
        # config.hidden1 = args.hidden
        # config.dropout = args.dropout
        # config.entity_total, config.relation_total = get_total_number('./data/', 'stat_500.txt')
        # config.feature_size = args.feature_size
        # config.train_iters = args.train_iters

    def forward(self, nodes1, nodes2):
        embeds1 = self.enc(nodes1)
        embeds2 = self.enc(nodes2)
        return embeds1, embeds2

    def loss(self, nodes1, nodes2):
        embeds1, embeds2 = self.forward(nodes1, nodes2)
        return self._xent_loss(embeds1, embeds2, self.neg_outputs)

    def affinity(self, inputs1, inputs2):
        """ Affinity score between batch of inputs1 and inputs2.
        Args:
            inputs1: tensor of shape [batch_size x feature_size].
        """
        # shape: [batch_size, input_dim1]
        result = torch.sum(inputs1 * inputs2, axis=1)
        return result

    def neg_cost(self, inputs1, neg_samples, hard_neg_samples=None):
        """ For each input in batch, compute the sum of its affinity to negative samples.

        Returns:
            Tensor of shape [batch_size x num_neg_samples]. For each node, a list of affinities to
                negative samples is computed.
        """
        neg_aff = torch.matmul(inputs1, torch.t(neg_samples))
        return neg_aff

    def _xent_loss(self, inputs1, inputs2, neg_samples, hard_neg_samples=None):
        aff = self.affinity(inputs1, inputs2)
        neg_aff = self.neg_cost(inputs1, neg_samples, hard_neg_samples)

        loss_f = nn.CrossEntropyLoss()
        true_xent = loss_f(torch.ones_like(aff), aff)
        negative_xent = loss_f(torch.zeros_like(aff), neg_aff)
        loss = torch.sum(true_xent) + self.neg_sample_weights * torch.sum(negative_xent)

        return loss


def load_cora():
    num_nodes = 2708
    num_feats = 1433
    feat_data = np.zeros((num_nodes, num_feats))
    # labels = np.empty((num_nodes,1), dtype=np.int64)
    node_map = {}
    label_map = {}
    with open("cora/cora.content") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            feat_data[i,:] = map(float, info[1:-1])
            node_map[info[0]] = i
            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map)
            labels[i] = label_map[info[-1]]

    adj_lists = defaultdict(set)
    with open("cora/cora.cites") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
    return feat_data, labels, adj_lists


def run_cora():
    np.random.seed(1)
    random.seed(1)
    num_nodes = 2708
    feat_data, adj_lists = load_cora()
    features = nn.Embedding(2708, 1433)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
   # features.cuda()

    agg1 = MeanAggregator(features, cuda=True)
    enc1 = Encoder(features, 1433, 128, adj_lists, agg1, gcn=True, cuda=False)
    agg2 = MeanAggregator(lambda nodes : enc1(nodes).t(), cuda=False)
    enc2 = Encoder(lambda nodes : enc1(nodes).t(), enc1.embed_dim, 128, adj_lists, agg2,
            base_model=enc1, gcn=True, cuda=False)
    enc1.num_samples = 5
    enc2.num_samples = 5

    graphsage = UnsupervisedGraphSage(enc2)
#    graphsage.cuda()
    rand_indices = np.random.permutation(num_nodes)
    test = rand_indices[:1000]
    val = rand_indices[1000:1500]
    train = list(rand_indices[1500:])

    optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, graphsage.parameters()), lr=0.7)
    times = []
    for batch in range(100):
        batch_nodes = train[:256]
        random.shuffle(train)
        start_time = time.time()
        optimizer.zero_grad()
        loss = graphsage.loss(batch_nodes, 
                Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time-start_time)
        print(batch, loss.item())

    val_output = graphsage.forward(val) 
    print("Validation F1:", f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro"))
    print("Average batch time:", np.mean(times))


if __name__ == "__main__":
    run_cora()
