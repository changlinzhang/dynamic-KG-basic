import os
import math
import pickle
import numpy as np

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from LSTMLinear import LSTMModel

USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
	longTensor = torch.cuda.LongTensor
	floatTensor = torch.cuda.FloatTensor

else:
	longTensor = torch.LongTensor
	floatTensor = torch.FloatTensor


class TTransEModel(nn.Module):
	def __init__(self, config):
		super(TTransEModel, self).__init__()
		self.score = config.score
		self.learning_rate = config.learning_rate
		self.early_stopping_round = config.early_stopping_round
		self.L1_flag = config.L1_flag
		self.filter = config.filter
		self.embedding_size = config.embedding_size
		self.entity_total = config.entity_total
		self.relation_total = config.relation_total
		self.tem_total = config.tem_total
		self.batch_size = config.batch_size

		ent_weight = floatTensor(self.entity_total, self.embedding_size)
		rel_weight = floatTensor(self.relation_total, self.embedding_size)
		tem_weight = floatTensor(self.tem_total, self.embedding_size)
		# Use xavier initialization method to initialize embeddings of entities and relations
		nn.init.xavier_uniform(ent_weight)
		nn.init.xavier_uniform(rel_weight)
		nn.init.xavier_uniform(tem_weight)
		self.ent_embeddings = nn.Embedding(self.entity_total, self.embedding_size)
		self.rel_embeddings = nn.Embedding(self.relation_total, self.embedding_size)
		self.tem_embeddings = nn.Embedding(self.tem_total, self.embedding_size)
		self.ent_embeddings.weight = nn.Parameter(ent_weight)
		self.rel_embeddings.weight = nn.Parameter(rel_weight)
		self.tem_embeddings.weight = nn.Parameter(tem_weight)

		normalize_entity_emb = F.normalize(self.ent_embeddings.weight.data, p=2, dim=1)
		normalize_relation_emb = F.normalize(self.rel_embeddings.weight.data, p=2, dim=1)
		normalize_temporal_emb = F.normalize(self.tem_embeddings.weight.data, p=2, dim=1)
		self.ent_embeddings.weight.data = normalize_entity_emb
		self.rel_embeddings.weight.data = normalize_relation_emb
		self.tem_embeddings.weight.data = normalize_temporal_emb

	def forward(self, pos_h, pos_t, pos_r, pos_tem, neg_h, neg_t, neg_r, neg_tem):
		pos_h_e = self.ent_embeddings(pos_h)
		pos_t_e = self.ent_embeddings(pos_t)
		pos_r_e = self.rel_embeddings(pos_r)
		pos_tem_e = self.tem_embeddings(pos_tem)

		neg_h_e = self.ent_embeddings(neg_h)
		neg_t_e = self.ent_embeddings(neg_t)
		neg_r_e = self.rel_embeddings(neg_r)
		neg_tem_e = self.tem_embeddings(neg_tem)

		if self.L1_flag:
			pos = torch.sum(torch.abs(pos_h_e + pos_r_e + pos_tem_e - pos_t_e), 1)
			neg = torch.sum(torch.abs(neg_h_e + neg_r_e + neg_tem_e - neg_t_e), 1)
		else:
			pos = torch.sum((pos_h_e + pos_r_e + pos_tem_e - pos_t_e) ** 2, 1)
			neg = torch.sum((neg_h_e + neg_r_e + neg_tem_e - neg_t_e) ** 2, 1)
		return pos, neg


class TADistmultModel(nn.Module):
	def __init__(self, config):
		super(TADistmultModel, self).__init__()
		self.learning_rate = config.learning_rate
		self.early_stopping_round = config.early_stopping_round
		self.L1_flag = config.L1_flag
		self.filter = config.filter
		self.embedding_size = config.embedding_size
		self.entity_total = config.entity_total
		self.relation_total = config.relation_total
		self.tem_total = config.tem_total
		self.prefix_total = config.prefix_total
		self.batch_size = config.batch_size

		self.criterion = nn.Softplus()
		torch.nn.BCELoss()

		self.dropout = nn.Dropout(config.dropout)
		self.lstm = LSTMModel(self.embedding_size, n_layer=1)

		ent_weight = floatTensor(self.entity_total, self.embedding_size)
		rel_weight = floatTensor(self.relation_total, self.embedding_size)
		tem_weight = floatTensor(self.tem_total, self.embedding_size)
		prefix_weight = floatTensor(self.prefix_total, self.embedding_size)
		# Use xavier initialization method to initialize embeddings of entities and relations
		nn.init.xavier_uniform(ent_weight)
		nn.init.xavier_uniform(rel_weight)
		nn.init.xavier_uniform(tem_weight)
		nn.init.xavier_uniform(prefix_weight)
		self.ent_embeddings = nn.Embedding(self.entity_total, self.embedding_size)
		self.rel_embeddings = nn.Embedding(self.relation_total, self.embedding_size)
		self.tem_embeddings = nn.Embedding(self.tem_total, self.embedding_size)
		self.prefix_embeddings = nn.Embedding(self.prefix_total, self.embedding_size)
		self.ent_embeddings.weight = nn.Parameter(ent_weight)
		self.rel_embeddings.weight = nn.Parameter(rel_weight)
		self.tem_embeddings.weight = nn.Parameter(tem_weight)
		self.prefix_embeddings.weight = nn.Parameter(prefix_weight)

		normalize_entity_emb = F.normalize(self.ent_embeddings.weight.data, p=2, dim=1)
		normalize_relation_emb = F.normalize(self.rel_embeddings.weight.data, p=2, dim=1)
		normalize_temporal_emb = F.normalize(self.tem_embeddings.weight.data, p=2, dim=1)
		normalize_prefix_emb = F.normalize(self.prefix_embeddings.weight.data, p=2, dim=1)
		self.ent_embeddings.weight.data = normalize_entity_emb
		self.rel_embeddings.weight.data = normalize_relation_emb
		self.tem_embeddings.weight.data = normalize_temporal_emb
		self.prefix_embeddings.weight.data = normalize_prefix_emb

	def scoring(self, h, t, r):
		return torch.sum(h * t * r, 1, False)

	def forward(self, pos_h, pos_t, pos_r, pos_tem, neg_h, neg_t, neg_r, neg_tem):
		pos_h_e = self.ent_embeddings(pos_h)
		pos_t_e = self.ent_embeddings(pos_t)
		pos_rseq_e = self.get_rseq(pos_r, pos_tem)

		neg_h_e = self.ent_embeddings(neg_h)
		neg_t_e = self.ent_embeddings(neg_t)
		neg_rseq_e = self.get_rseq(neg_r, neg_tem)

		pos_h_e = self.dropout(pos_h_e)
		pos_t_e = self.dropout(pos_t_e)
		pos_rseq_e = self.dropout(pos_rseq_e)
		neg_h_e = self.dropout(neg_h_e)
		neg_t_e = self.dropout(neg_t_e)
		neg_rseq_e = self.dropout(neg_rseq_e)

		pos = self.scoring(pos_h_e, pos_t_e, pos_rseq_e)
		neg = self.scoring(neg_h_e, neg_t_e, neg_rseq_e)
		return pos, neg

	def get_rseq(self, pos_r, pos_tem):
		pos_r_e = self.rel_embeddings(pos_r)
		pos_r_e = pos_r_e.unsqueeze(0).transpose(0, 1)

		pos_prefix = pos_tem[:, :1]
		pos_tem = pos_tem[:, 1:].contiguous()

		pos_prefix_e = self.prefix_embeddings(pos_prefix)

		bs = pos_tem.shape[0]  # batch size
		tem_len = pos_tem.shape[1]
		pos_tem = pos_tem.view(bs * tem_len)
		token_e = self.tem_embeddings(pos_tem)
		token_e = token_e.view(bs, tem_len, self.embedding_size)
		# pos_seq_e = torch.cat((pos_r_e, token_e), 1)
		pos_seq_e = torch.cat((pos_r_e, pos_prefix_e), 1)
		pos_seq_e = torch.cat((pos_seq_e, token_e), 1)

		hidden_tem = self.lstm(pos_seq_e)
		hidden_tem = hidden_tem[0, :, :]
		pos_rseq_e = hidden_tem

		return pos_rseq_e


class DistmultModel(nn.Module):
	def __init__(self, config):
		super(DistmultModel, self).__init__()
		self.learning_rate = config.learning_rate
		self.early_stopping_round = config.early_stopping_round
		self.L1_flag = config.L1_flag
		self.filter = config.filter
		self.embedding_size = config.embedding_size
		self.entity_total = config.entity_total
		self.relation_total = config.relation_total
		self.batch_size = config.batch_size

		self.criterion = nn.Softplus()
		torch.nn.BCELoss()

		self.dropout = nn.Dropout(config.dropout)

		ent_weight = floatTensor(self.entity_total, self.embedding_size)
		rel_weight = floatTensor(self.relation_total, self.embedding_size)
		# Use xavier initialization method to initialize embeddings of entities and relations
		nn.init.xavier_uniform(ent_weight)
		nn.init.xavier_uniform(rel_weight)
		self.ent_embeddings = nn.Embedding(self.entity_total, self.embedding_size)
		self.rel_embeddings = nn.Embedding(self.relation_total, self.embedding_size)
		self.ent_embeddings.weight = nn.Parameter(ent_weight)
		self.rel_embeddings.weight = nn.Parameter(rel_weight)

		normalize_entity_emb = F.normalize(self.ent_embeddings.weight.data, p=2, dim=1)
		normalize_relation_emb = F.normalize(self.rel_embeddings.weight.data, p=2, dim=1)
		self.ent_embeddings.weight.data = normalize_entity_emb
		self.rel_embeddings.weight.data = normalize_relation_emb

	def scoring(self, h, t, r):
		return torch.sum(h * t * r, 1, False)

	def forward(self, pos_h, pos_t, pos_r, neg_h, neg_t, neg_r):
		pos_h_e = self.ent_embeddings(pos_h)
		pos_t_e = self.ent_embeddings(pos_t)
		pos_r_e = self.rel_embeddings(pos_r)

		neg_h_e = self.ent_embeddings(neg_h)
		neg_t_e = self.ent_embeddings(neg_t)
		neg_r_e = self.rel_embeddings(neg_r)

		pos_h_e = self.dropout(pos_h_e)
		pos_t_e = self.dropout(pos_t_e)
		pos_r_e = self.dropout(pos_r_e)
		neg_h_e = self.dropout(neg_h_e)
		neg_t_e = self.dropout(neg_t_e)
		neg_r_e = self.dropout(neg_r_e)

		pos = self.scoring(pos_h_e, pos_t_e, pos_r_e)
		neg = self.scoring(neg_h_e, neg_t_e, neg_r_e)
		return pos, neg


class TATransEModel(nn.Module):
	def __init__(self, config):
		super(TATransEModel, self).__init__()
		self.score = config.score
		self.learning_rate = config.learning_rate
		self.early_stopping_round = config.early_stopping_round
		self.L1_flag = config.L1_flag
		self.filter = config.filter
		self.embedding_size = config.embedding_size
		self.entity_total = config.entity_total
		self.relation_total = config.relation_total
		self.tem_total = config.tem_total
		self.prefix_total = config.prefix_total
		self.batch_size = config.batch_size

		self.dropout = nn.Dropout(config.dropout)
		self.lstm = LSTMModel(self.embedding_size, n_layer=1)

		ent_weight = floatTensor(self.entity_total, self.embedding_size)
		rel_weight = floatTensor(self.relation_total, self.embedding_size)
		tem_weight = floatTensor(self.tem_total, self.embedding_size)
		prefix_weight = floatTensor(self.prefix_total, self.embedding_size)
		# Use xavier initialization method to initialize embeddings of entities and relations
		nn.init.xavier_uniform(ent_weight)
		nn.init.xavier_uniform(rel_weight)
		nn.init.xavier_uniform(tem_weight)
		nn.init.xavier_uniform(prefix_weight)
		self.ent_embeddings = nn.Embedding(self.entity_total, self.embedding_size)
		self.rel_embeddings = nn.Embedding(self.relation_total, self.embedding_size)
		self.tem_embeddings = nn.Embedding(self.tem_total, self.embedding_size)
		self.prefix_embeddings = nn.Embedding(self.prefix_total, self.embedding_size)
		self.ent_embeddings.weight = nn.Parameter(ent_weight)
		self.rel_embeddings.weight = nn.Parameter(rel_weight)
		self.tem_embeddings.weight = nn.Parameter(tem_weight)
		self.prefix_embeddings.weight = nn.Parameter(prefix_weight)

		normalize_entity_emb = F.normalize(self.ent_embeddings.weight.data, p=2, dim=1)
		normalize_relation_emb = F.normalize(self.rel_embeddings.weight.data, p=2, dim=1)
		normalize_temporal_emb = F.normalize(self.tem_embeddings.weight.data, p=2, dim=1)
		normalize_prefix_emb = F.normalize(self.prefix_embeddings.weight.data, p=2, dim=1)
		self.ent_embeddings.weight.data = normalize_entity_emb
		self.rel_embeddings.weight.data = normalize_relation_emb
		self.tem_embeddings.weight.data = normalize_temporal_emb
		self.prefix_embeddings.weight.data = normalize_prefix_emb

	def forward(self, pos_h, pos_t, pos_r, pos_tem, neg_h, neg_t, neg_r, neg_tem):
		pos_h_e = self.ent_embeddings(pos_h)
		pos_t_e = self.ent_embeddings(pos_t)
		pos_rseq_e = self.get_rseq(pos_r, pos_tem)

		neg_h_e = self.ent_embeddings(neg_h)
		neg_t_e = self.ent_embeddings(neg_t)
		neg_rseq_e = self.get_rseq(neg_r, neg_tem)

		pos_h_e = self.dropout(pos_h_e)
		pos_t_e = self.dropout(pos_t_e)
		pos_rseq_e = self.dropout(pos_rseq_e)
		neg_h_e = self.dropout(neg_h_e)
		neg_t_e = self.dropout(neg_t_e)
		neg_rseq_e = self.dropout(neg_rseq_e)

		if self.L1_flag:
			pos = torch.sum(torch.abs(pos_h_e + pos_rseq_e - pos_t_e), 1)
			neg = torch.sum(torch.abs(neg_h_e + neg_rseq_e - neg_t_e), 1)
		# L2 distance
		else:
			pos = torch.sum((pos_h_e + pos_rseq_e - pos_t_e) ** 2, 1)
			neg = torch.sum((neg_h_e + neg_rseq_e - neg_t_e) ** 2, 1)
		return pos, neg

	def get_rseq(self, pos_r, pos_tem):
		pos_r_e = self.rel_embeddings(pos_r)
		pos_r_e = pos_r_e.unsqueeze(0).transpose(0, 1)

		pos_prefix = pos_tem[:, :1]
		pos_tem = pos_tem[:, 1:].contiguous()

		pos_prefix_e = self.prefix_embeddings(pos_prefix)

		bs = pos_tem.shape[0] # batch size
		tem_len = pos_tem.shape[1]
		pos_tem = pos_tem.view(bs * tem_len)
		token_e = self.tem_embeddings(pos_tem)
		token_e = token_e.view(bs, tem_len, self.embedding_size)
		# pos_seq_e = torch.cat((pos_r_e, token_e), 1)
		pos_seq_e = torch.cat((pos_r_e, pos_prefix_e), 1)
		pos_seq_e = torch.cat((pos_seq_e, token_e), 1)

		hidden_tem = self.lstm(pos_seq_e)
		hidden_tem = hidden_tem[0, :, :]
		pos_rseq_e = hidden_tem

		return pos_rseq_e


class TransEModel(nn.Module):
	def __init__(self, config):
		super(TransEModel, self).__init__()
		self.score = config.score
		self.learning_rate = config.learning_rate
		self.early_stopping_round = config.early_stopping_round
		self.L1_flag = config.L1_flag
		self.filter = config.filter
		self.embedding_size = config.embedding_size
		self.entity_total = config.entity_total
		self.relation_total = config.relation_total
		self.batch_size = config.batch_size

		self.dropout = nn.Dropout(config.dropout)

		ent_weight = floatTensor(self.entity_total, self.embedding_size)
		rel_weight = floatTensor(self.relation_total, self.embedding_size)
		# Use xavier initialization method to initialize embeddings of entities and relations
		nn.init.xavier_uniform(ent_weight)
		nn.init.xavier_uniform(rel_weight)
		self.ent_embeddings = nn.Embedding(self.entity_total, self.embedding_size)
		self.rel_embeddings = nn.Embedding(self.relation_total, self.embedding_size)
		self.ent_embeddings.weight = nn.Parameter(ent_weight)
		self.rel_embeddings.weight = nn.Parameter(rel_weight)

		normalize_entity_emb = F.normalize(self.ent_embeddings.weight.data, p=2, dim=1)
		normalize_relation_emb = F.normalize(self.rel_embeddings.weight.data, p=2, dim=1)
		self.ent_embeddings.weight.data = normalize_entity_emb
		self.rel_embeddings.weight.data = normalize_relation_emb

	def forward(self, pos_h, pos_t, pos_r, neg_h, neg_t, neg_r):
		pos_h_e = self.ent_embeddings(pos_h)
		pos_t_e = self.ent_embeddings(pos_t)
		pos_r_e = self.rel_embeddings(pos_r)

		neg_h_e = self.ent_embeddings(neg_h)
		neg_t_e = self.ent_embeddings(neg_t)
		neg_r_e = self.rel_embeddings(neg_r)

		pos_h_e = self.dropout(pos_h_e)
		pos_t_e = self.dropout(pos_t_e)
		pos_r_e = self.dropout(pos_r_e)
		neg_h_e = self.dropout(neg_h_e)
		neg_t_e = self.dropout(neg_t_e)
		neg_r_e = self.dropout(neg_r_e)

		if self.L1_flag:
			pos = torch.sum(torch.abs(pos_h_e + pos_r_e - pos_t_e), 1)
			neg = torch.sum(torch.abs(neg_h_e + neg_r_e - neg_t_e), 1)
		# L2 distance
		else:
			pos = torch.sum((pos_h_e + pos_r_e - pos_t_e) ** 2, 1)
			neg = torch.sum((neg_h_e + neg_r_e - neg_t_e) ** 2, 1)
		return pos, neg
