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
		self.learning_rate = config.learning_rate
		self.early_stopping_round = config.early_stopping_round
		self.L1_flag = config.L1_flag
		self.filter = config.filter
		self.embedding_size = config.embedding_size
		self.entity_total = config.entity_total
		self.relation_total = config.relation_total
		self.tem_total = 32
		self.batch_size = config.batch_size

		self.lstm = LSTMModel(self.embedding_size, n_layer=1)

		ent_weight = floatTensor(self.entity_total, self.embedding_size)
		rel_weight = floatTensor(self.relation_total, self.embedding_size)
		tem_weight = floatTensor(self.tem_total, self.embedding_size)
		# Use xavier initialization method to initialize embeddings of entities and relations
		nn.init.xavier_uniform(ent_weight)
		nn.init.xavier_uniform(rel_weight)
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

	#TODO: should use tensor operations
	def unroll(self, data, unroll_len = 4):
		result = []
		for i in range(len(data) - unroll_len):
			result.append(data[i: i+unroll_len])
		return result

	def forward(self, pos_h, pos_t, pos_r, pos_tem, neg_h, neg_t, neg_r, neg_tem):
		pos_h_e = self.ent_embeddings(pos_h)
		pos_t_e = self.ent_embeddings(pos_t)
		pos_r_e = self.rel_embeddings(pos_r)
		print(np.shape(pos_r))
		print(np.shape(pos_tem))
		# print(np.shape(pos_r_e))
		isFirst = True
		pos_seq_e = None
		i = 0
		for tem in pos_tem:
			seq_e = pos_r_e[i].unsqueeze(0)
			for token in tem:
				token_e = self.tem_embeddings(token)
				seq_e = torch.cat((seq_e, token_e.unsqueeze(0)), 0)
				# print(token_e.unsqueeze(0).size())
			# print(np.shape(seq_e))
			if isFirst:
				pos_seq_e = seq_e.unsqueeze(0)
				isFirst = False
			else:
				pos_seq_e = torch.cat((pos_seq_e, seq_e.unsqueeze(0)), 0)
		# print(pos_seq_e.size())

		isFirst = True
		pos_rseq_e = None
		# add LSTM
		for seq_e in pos_seq_e:
			# unroll to get input for LSTM
			# input_tem = self.unroll(seq_e)
			input_tem = seq_e.unsqueeze(0) # unroll length = 1
			# print(input_tem.size())
			hidden_tem = self.lstm(input_tem)
			# print(hidden_tem.size())
			# print(hidden_tem[0,-1,:].size())
			if isFirst:
				pos_rseq_e = hidden_tem[0,-1,:].unsqueeze(0)
				isFirst = False
			else:
				pos_rseq_e = torch.cat((pos_rseq_e, hidden_tem[0,-1,:].unsqueeze(0)), 0)

		neg_h_e = self.ent_embeddings(neg_h)
		neg_t_e = self.ent_embeddings(neg_t)
		neg_r_e = self.rel_embeddings(neg_r)
		isFirst = True
		neg_seq_e = None
		i = 0
		for tem in neg_tem:
			seq_e = neg_r_e[i].unsqueeze(0)
			for token in tem:
				token_e = self.tem_embeddings(token)
				seq_e = torch.cat((seq_e, token_e.unsqueeze(0)), 0)
			if isFirst:
				neg_seq_e = seq_e.unsqueeze(0)
				isFirst = False
			else:
				neg_seq_e = torch.cat((neg_seq_e, seq_e.unsqueeze(0)), 0)

		isFirst = True
		neg_rseq_e = []
		for seq_e in neg_seq_e:
			input_tem = seq_e.unsqueeze(0)
			hidden_tem = self.lstm(input_tem)
			if isFirst:
				neg_rseq_e = hidden_tem[0,-1,:].unsqueeze(0)
				isFirst = False
			else:
				neg_rseq_e = torch.cat((neg_rseq_e, hidden_tem[0,-1,:].unsqueeze(0)), 0)

		# pos_tem_e = []
		# for tem in pos_tem:
		# 	tem_e = []
		# 	for token in tem:
		# 		token_e = self.tem_embeddings(token)
		# 		tem_e.append(token_e)
		# 	pos_tem_e.append(tem_e)
        #
		# pos_rseq_e = []
		# # add LSTM
		# pos_input_r = self.unroll(pos_r_e)
		# pos_hidden_r = self.lstm(pos_input_r)
		# pos_rseq_e.append(pos_hidden_r[0,-1,:])
		# for tem_e in pos_tem_e:
		# 	# unroll to get input for LSTM
		# 	input_tem = self.unroll(tem_e)
		# 	hidden_tem = self.lstm(input_tem)
		# 	print(np.shape(hidden_tem[0,-1:,:]))
		# 	pos_rseq_e.append(hidden_tem[0,-1:,:])
        #
		# neg_h_e = self.ent_embeddings(neg_h)
		# neg_t_e = self.ent_embeddings(neg_t)
		# neg_r_e = self.rel_embeddings(neg_r)
		# neg_tem_e = []
		# for tem in neg_tem:
		# 	tem_e = []
		# 	for token in tem:
		# 		token_e = self.tem_embeddings(token)
		# 		tem_e.append(token_e)
		# 	neg_tem_e.append(tem_e)
        #
		# neg_rseq_e = []
		# # add LSTM
		# neg_input_r = self.unroll(neg_r_e)
		# neg_hidden_r = self.lstm(neg_input_r)
		# neg_rseq_e.append(neg_hidden_r[0,-1,:])
		# for tem_e in pos_tem_e:
		# 	# unroll to get input for LSTM
		# 	input_tem = self.unroll(tem_e)
		# 	hidden_tem = self.lstm(input_tem)
		# 	print(np.shape(hidden_tem[0,-1,:]))
		# 	neg_rseq_e.append(hidden_tem[0,-1,:])

		print(pos_h_e.size())
		print(pos_rseq_e.size())
		print(pos_t_e.size())
		# L1 distance
		if self.L1_flag:
			pos = torch.sum(torch.abs(pos_h_e + pos_rseq_e - pos_t_e), 1)
			neg = torch.sum(torch.abs(neg_h_e + pos_rseq_e - neg_t_e), 1)
		# L2 distance
		else:
			pos = torch.sum((pos_h_e + pos_rseq_e - pos_t_e) ** 2, 1)
			neg = torch.sum((neg_h_e + pos_rseq_e - neg_t_e) ** 2, 1)
		return pos, neg
