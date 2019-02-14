import os
import math
import pickle
import numpy as np

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
	longTensor = torch.cuda.LongTensor
	floatTensor = torch.cuda.FloatTensor

else:
	longTensor = torch.LongTensor
	floatTensor = torch.FloatTensor


class LSTMModel(nn.Module):
	def __init__(self, in_dim, n_layer):
		super(LSTMModel, self).__init__()
		self.n_layer = n_layer
		self.hidden_dim = in_dim
		self.lstm = nn.LSTM(in_dim, self.hidden_dim, n_layer, batch_first=True)

	def forward(self, x):
		out, h = self.lstm(x)
		return h[0]


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

	def unroll(self, data, unroll_len=4):
		result = None
		for i in range(len(data) - unroll_len):
			if i == 0:
				result = data[i: i+unroll_len].unsqueeze(0)
			else:
				result = torch.cat((result, data[i: i+unroll_len].unsqueeze(0)), 0)
		return result

	def get_rseq(self, pos_r, pos_tem):
		pos_r_e = self.rel_embeddings(pos_r)

		pos_seq_e = None
		i = 0
		for tem in pos_tem:
			seq_e = pos_r_e[i].unsqueeze(0)
			for token in tem:
				token_e = self.tem_embeddings(token)
				seq_e = torch.cat((seq_e, token_e.unsqueeze(0)), 0)
			if i == 0:
				pos_seq_e = seq_e.unsqueeze(0)
			else:
				pos_seq_e = torch.cat((pos_seq_e, seq_e.unsqueeze(0)), 0)
			i += 1

		isFirst = True
		pos_rseq_e = None
		# add LSTM
		for seq_e in pos_seq_e:
			# unroll to get input for LSTM
			input_tem = self.unroll(seq_e)
			# input_tem = seq_e.unsqueeze(0) # unroll length = 1
			hidden_tem = self.lstm(input_tem)
			if isFirst:
				pos_rseq_e = hidden_tem[0,-1,:].unsqueeze(0)
				isFirst = False
			else:
				pos_rseq_e = torch.cat((pos_rseq_e, hidden_tem[0,-1,:].unsqueeze(0)), 0)
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
