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
		self.tem_total = 32
		self.batch_size = config.batch_size

		self.criterion = nn.Softplus()
		torch.nn.BCELoss()

		self.dropout = nn.Dropout(config.dropout)
		self.lstm = LSTMModel(self.embedding_size, n_layer=1)

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

	def predict(self, e, r, tem):
		e_e = self.ent_embeddings(e)
		rseq_e = self.get_rseq(r, tem)
		pred = torch.mm(e_e * rseq_e, self.ent_embeddings.weight.transpose(1, 0))
		pred = F.sigmoid(pred)
		return pred

	def _calc(self, h, t, r):
		return - torch.sum(h * t * r, -1)

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

	def score(self, pos_h, pos_t, pos_r, pos_tem):
		pos_h_e = self.ent_embeddings(pos_h)
		pos_t_e = self.ent_embeddings(pos_t)
		pos_rseq_e = self.get_rseq(pos_r, pos_tem)
		pos = self.scoring(pos_h_e, pos_t_e, pos_rseq_e)
		return pos

	def loss(self, pos, neg):
		pos_y = torch.ones(pos.shape[0], 1)
		neg_y = -torch.ones(neg.shape[0], 1)
		score = torch.cat((pos, neg), 0)
		batch_y = torch.cat((pos_y, neg_y), 0)
		batch_y = autograd.Variable(batch_y.cuda())
		return torch.mean(self.criterion(score * batch_y))

	def get_rseq(self, pos_r, pos_tem):
		pos_r_e = self.rel_embeddings(pos_r)
		pos_r_e = pos_r_e.unsqueeze(0).transpose(0, 1)

		bs = pos_tem.shape[0] # batch size
		tem_len = pos_tem.shape[1]
		pos_tem = pos_tem.contiguous()
		pos_tem = pos_tem.view(bs * tem_len)
		token_e = self.tem_embeddings(pos_tem)
		token_e = token_e.view(bs, tem_len, self.embedding_size)
		pos_seq_e = torch.cat((pos_r_e, token_e), 1)
		# print(pos_seq_e.size())

		hidden_tem = self.lstm(pos_seq_e)
		hidden_tem = hidden_tem[0, :, :]
		pos_rseq_e = hidden_tem

		# print(pos_rseq_e)
		return pos_rseq_e


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
		self.tem_total = 32
		self.batch_size = config.batch_size

		self.dropout = nn.Dropout(config.dropout)
		self.lstm = LSTMModel(self.embedding_size, n_layer=1)

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

	def distmulti_predict(self, e, rseq_e):
		pred = torch.mm(e * rseq_e, self.ent_embeddings.weight.transpose(1, 0))
		pred = F.sigmoid(pred)
		return pred

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

		if self.score == 0:
			# TranE score
			# L1 distance
			if self.L1_flag:
				pos = torch.sum(torch.abs(pos_h_e + pos_rseq_e - pos_t_e), 1)
				neg = torch.sum(torch.abs(neg_h_e + neg_rseq_e - neg_t_e), 1)
			# L2 distance
			else:
				pos = torch.sum((pos_h_e + pos_rseq_e - pos_t_e) ** 2, 1)
				neg = torch.sum((neg_h_e + neg_rseq_e - neg_t_e) ** 2, 1)
		return pos, neg

	def unroll(self, data, unroll_len=4):
		result = None
		for i in range(len(data) - unroll_len):
			if i == 0:
				result = data[i: i+unroll_len].unsqueeze(0)
			else:
				result = torch.cat((result, data[i: i+unroll_len].unsqueeze(0)), 0)
		return result

	def get_rseq(self, pos_r, pos_tem, unroll_len=4):
		pos_r_e = self.rel_embeddings(pos_r)
		pos_r_e = pos_r_e.unsqueeze(0).transpose(0, 1)

		bs = pos_tem.shape[0] # batch size
		tem_len = pos_tem.shape[1]
		pos_tem = pos_tem.view(bs * tem_len)
		token_e = self.tem_embeddings(pos_tem)
		token_e = token_e.view(bs, tem_len, self.embedding_size)
		pos_seq_e = torch.cat((pos_r_e, token_e), 1)
		# print(pos_seq_e.size())

		hidden_tem = self.lstm(pos_seq_e)
		hidden_tem = hidden_tem[0, :, :]
		pos_rseq_e = hidden_tem

		# print(pos_rseq_e)
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
		self.tem_total = 32
		self.batch_size = config.batch_size

		self.dropout = nn.Dropout(config.dropout)
		self.lstm = LSTMModel(self.embedding_size, n_layer=1)

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

	def distmulti_predict(self, e, rseq_e):
		pred = torch.mm(e * rseq_e, self.ent_embeddings.weight.transpose(1, 0))
		pred = F.sigmoid(pred)
		return pred

	def forward(self, pos_h, pos_t, pos_r, pos_tem, neg_h, neg_t, neg_r, neg_tem):
		pos_h_e = self.ent_embeddings(pos_h)
		pos_t_e = self.ent_embeddings(pos_t)
		pos_rseq_e = self.rel_embeddings(pos_r)

		neg_h_e = self.ent_embeddings(neg_h)
		neg_t_e = self.ent_embeddings(neg_t)
		neg_rseq_e = self.rel_embeddings(neg_r)

		if self.score == 0:
			# TranE score
			# L1 distance
			if self.L1_flag:
				pos = torch.sum(torch.abs(pos_h_e + pos_rseq_e - pos_t_e), 1)
				neg = torch.sum(torch.abs(neg_h_e + neg_rseq_e - neg_t_e), 1)
			# L2 distance
			else:
				pos = torch.sum((pos_h_e + pos_rseq_e - pos_t_e) ** 2, 1)
				neg = torch.sum((neg_h_e + neg_rseq_e - neg_t_e) ** 2, 1)
		else:
			# DistMult score
			j = 0
			pos = None
			for h_e in pos_h_e:
				score = torch.squeeze(torch.mm((pos_h_e[j] * pos_t_e[j]).unsqueeze(0), pos_rseq_e[j].unsqueeze(0).transpose(1, 0)))
				if j == 0:
					pos = score.unsqueeze(0)
				else:
					pos = torch.cat((pos, score.unsqueeze(0)), 0)
				j += 1
			j = 0
			neg = None
			for h_e in neg_h_e:
				score_m = torch.squeeze(torch.mm((neg_h_e[j] * neg_t_e[j]).unsqueeze(0), neg_rseq_e[j].unsqueeze(0).transpose(1, 0)))
				score = score_m
				if j == 0:
					neg = score.unsqueeze(0)
				else:
					neg = torch.cat((neg, score.unsqueeze(0)), 0)
				j += 1
			# print(pos.size())
		return pos, neg
