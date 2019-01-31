#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-12-24 01:45:03
# @Author  : jimmy (jimmywangheng@qq.com)
# @Link    : http://sdcs.sysu.edu.cn
# @Version : $Id$

import os

import torch
torch.multiprocessing.set_start_method("spawn")
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import time
import datetime
import random

from utils import *
from data import *
from evaluation import *
import loss
import model

# from hyperboard import Agent

USE_CUDA = torch.cuda.is_available()

if USE_CUDA:
    longTensor = torch.cuda.LongTensor
    floatTensor = torch.cuda.FloatTensor

else:
    longTensor = torch.LongTensor
    floatTensor = torch.FloatTensor

"""
The meaning of parameters:
self.dataset: Which dataset is used to train the model? Such as 'FB15k', 'WN18', etc.
self.learning_rate: Initial learning rate (lr) of the model.
self.early_stopping_round: How many times will lr decrease? If set to 0, it remains constant.
self.L1_flag: If set to True, use L1 distance as dissimilarity; else, use L2.
self.embedding_size: The embedding size of entities and relations.
self.num_batches: How many batches to train in one epoch?
self.train_times: The maximum number of epochs for training.
self.margin: The margin set for MarginLoss.
self.filter: Whether to check a generated negative sample is false negative.
self.momentum: The momentum of the optimizer.
self.optimizer: Which optimizer to use? Such as SGD, Adam, etc.
self.loss_function: Which loss function to use? Typically, we use margin loss.
self.entity_total: The number of different entities.
self.relation_total: The number of different relations.
self.batch_size: How many instances is contained in one batch?
"""

class Config(object):
    def __init__(self):
        self.dataset = None
        self.learning_rate = 0.001
        self.early_stopping_round = 0
        self.L1_flag = True
        self.embedding_size = 100
        self.num_batches = 100
        self.train_times = 1000
        self.margin = 1.0
        self.filter = True
        self.momentum = 0.9
        self.optimizer = optim.Adam
        self.loss_function = loss.marginLoss
        self.entity_total = 0
        self.relation_total = 0
        self.batch_size = 0

if __name__ == "__main__":

    import argparse
    argparser = argparse.ArgumentParser()

    """
    The meaning of some parameters:
    seed: Fix the random seed. Except for 0, which means no setting of random seed.
    port: The port number used by hyperboard, 
    which is a demo showing training curves in real time.
    You can refer to https://github.com/WarBean/hyperboard to know more.
    num_processes: Number of processes used to evaluate the result.
    """

    argparser.add_argument('-d', '--dataset', type=str)
    argparser.add_argument('-l', '--learning_rate', type=float, default=0.001)
    argparser.add_argument('-es', '--early_stopping_round', type=int, default=0)
    argparser.add_argument('-L', '--L1_flag', type=int, default=1)
    argparser.add_argument('-em', '--embedding_size', type=int, default=100)
    argparser.add_argument('-nb', '--num_batches', type=int, default=100)
    argparser.add_argument('-n', '--train_times', type=int, default=1000)
    argparser.add_argument('-m', '--margin', type=float, default=1.0)
    argparser.add_argument('-f', '--filter', type=int, default=1)
    argparser.add_argument('-mo', '--momentum', type=float, default=0.9)
    argparser.add_argument('-s', '--seed', type=int, default=0)
    argparser.add_argument('-op', '--optimizer', type=int, default=1)
    argparser.add_argument('-lo', '--loss_type', type=int, default=0)
    argparser.add_argument('-p', '--port', type=int, default=5000)
    argparser.add_argument('-np', '--num_processes', type=int, default=4)

    args = argparser.parse_args()

    # Start the hyperboard agent
    # agent = Agent(username = 'vicky', password = 'vicky', address='127.0.0.1', port=args.port)

    if args.seed != 0:
        torch.manual_seed(args.seed)

    trainTotal, trainList, trainDict, trainTimes = load_quadruples('./data/icews14/', 'train2id.txt', 'train_tem.npy')
    validTotal, validList, validDict, validTimes = load_quadruples('./data/icews14/', 'valid2id.txt', 'valid_tem.npy')
    quadrupleTotal, quadrupleList, tripleDict, _ = load_quadruples('./data/icews14/', 'train2id.txt', 'train_tem.npy', 'valid2id.txt', 'valid_tem.npy', 'test2id.txt', 'test_tem.npy')
    config = Config()
    config.dataset = args.dataset
    config.learning_rate = args.learning_rate

    config.early_stopping_round = args.early_stopping_round

    if args.L1_flag == 1:
        config.L1_flag = True
    else:
        config.L1_flag = False

    config.embedding_size = args.embedding_size
    config.num_batches = args.num_batches
    config.train_times = args.train_times
    config.margin = args.margin

    if args.filter == 1:
        config.filter = True
    else:
        config.filter = False

    config.momentum = args.momentum

    if args.optimizer == 0:
        config.optimizer = optim.SGD
    elif args.optimizer == 1:
        config.optimizer = optim.Adam
    elif args.optimizer == 2:
        config.optimizer = optim.RMSprop

    if args.loss_type == 0:
        config.loss_function = loss.marginLoss

    config.entity_total, config.relation_total = get_total_number('./data/icews14/', 'stat.txt')
    config.time_total = 32
    config.batch_size = trainTotal // config.num_batches

    shareHyperparameters = {'dataset': args.dataset,
        'learning_rate': args.learning_rate,
        'early_stopping_round': args.early_stopping_round,
        'L1_flag': args.L1_flag,
        'embedding_size': args.embedding_size,
        'margin': args.margin,
        'filter': args.filter,
        'momentum': args.momentum,
        'seed': args.seed,
        'optimizer': args.optimizer,
        'loss_type': args.loss_type,
        }

    trainHyperparameters = shareHyperparameters.copy()
    trainHyperparameters.update({'type': 'train_loss'})

    validHyperparameters = shareHyperparameters.copy()
    validHyperparameters.update({'type': 'valid_loss'})

    hit10Hyperparameters = shareHyperparameters.copy()
    hit10Hyperparameters.update({'type': 'hit10'})

    meanrankHyperparameters = shareHyperparameters.copy()
    meanrankHyperparameters.update({'type': 'mean_rank'})

    loss_function = config.loss_function()

    # filename = '_'.join(
    #     #     ['l', str(args.learning_rate),
    #     #      'es', str(args.early_stopping_round),
    #     #      'L', str(args.L1_flag),
    #     #      'em', str(args.embedding_size),
    #     #      'nb', str(args.num_batches),
    #     #      'n', str(args.train_times),
    #     #      'm', str(args.margin),
    #     #      'f', str(args.filter),
    #     #      'mo', str(args.momentum),
    #     #      's', str(args.seed),
    #     #      'op', str(args.optimizer),
    #     #      'lo', str(args.loss_type),]) + '_TATransE.ckpt'
    filename = 'TATransE.ckpt'
    path_name = os.path.join('./model/', filename)
    if os.path.exists(path_name):
        model = torch.load(path_name)
    else:
        model = model.TATransEModel(config)

    if USE_CUDA:
        model.cuda()
        loss_function.cuda()

    optimizer = config.optimizer(model.parameters(), lr=config.learning_rate)
    margin = autograd.Variable(floatTensor([config.margin]))

    start_time = time.time()

    trainBatchList = getBatchList(trainList, config.num_batches)

    for epoch in range(config.train_times):
        total_loss = floatTensor([0.0])
        random.shuffle(trainBatchList)
        for batchList in trainBatchList:
            if config.filter == True:
                pos_h_batch, pos_t_batch, pos_r_batch, pos_time_batch, neg_h_batch, neg_t_batch, neg_r_batch, neg_time_batch = getBatch_filter_all(batchList,
                    config.entity_total, tripleDict)
            else:
                pos_h_batch, pos_t_batch, pos_r_batch, pos_time_batch, neg_h_batch, neg_t_batch, neg_r_batch, neg_time_batch = getBatch_raw_all(batchList,
                    config.entity_total)

            batch_entity_set = set(pos_h_batch + pos_t_batch + neg_h_batch + neg_t_batch)
            batch_relation_set = set(pos_r_batch + neg_r_batch)
            batch_entity_list = list(batch_entity_set)
            batch_relation_list = list(batch_relation_set)

            pos_h_batch = autograd.Variable(longTensor(pos_h_batch))
            pos_t_batch = autograd.Variable(longTensor(pos_t_batch))
            pos_r_batch = autograd.Variable(longTensor(pos_r_batch))
            pos_time_batch = autograd.Variable(longTensor(pos_time_batch))
            neg_h_batch = autograd.Variable(longTensor(neg_h_batch))
            neg_t_batch = autograd.Variable(longTensor(neg_t_batch))
            neg_r_batch = autograd.Variable(longTensor(neg_r_batch))
            neg_time_batch = autograd.Variable(longTensor(neg_time_batch))

            model.zero_grad()
            pos, neg = model(pos_h_batch, pos_t_batch, pos_r_batch, pos_time_batch, neg_h_batch, neg_t_batch, neg_r_batch, neg_time_batch)

            if args.loss_type == 0:
                losses = loss_function(pos, neg, margin)
            else:
                losses = loss_function(pos, neg)
            ent_embeddings = model.ent_embeddings(torch.cat([pos_h_batch, pos_t_batch, neg_h_batch, neg_t_batch]))
            rel_embeddings = model.rel_embeddings(torch.cat([pos_r_batch, neg_r_batch]))
            rseq_embeddings = model.get_rseq(torch.cat([pos_r_batch, neg_r_batch]), torch.cat([pos_time_batch, neg_time_batch]))
            losses = losses + loss.normLoss(ent_embeddings) + loss.normLoss(rel_embeddings) + loss.normLoss(rseq_embeddings)

            losses.backward()
            optimizer.step()
            total_loss += losses.data

        # agent.append(trainCurve, epoch, total_loss[0])

        if epoch % 10 == 0:
            now_time = time.time()
            print(now_time - start_time)
            print("Train total loss: %d %f" % (epoch, total_loss[0]))

        if epoch % 10 == 0:
            if config.filter == True:
                pos_h_batch, pos_t_batch, pos_r_batch, pos_time_batch, neg_h_batch, neg_t_batch, neg_r_batch, neg_time_batch = getBatch_filter_random(validList,
                    config.batch_size, config.entity_total, tripleDict)
            else:
                pos_h_batch, pos_t_batch, pos_r_batch, pos_time_batch, neg_h_batch, neg_t_batch, neg_r_batch, neg_time_batch = getBatch_raw_random(validList,
                    config.batch_size, config.entity_total)
            pos_h_batch = autograd.Variable(longTensor(pos_h_batch))
            pos_t_batch = autograd.Variable(longTensor(pos_t_batch))
            pos_r_batch = autograd.Variable(longTensor(pos_r_batch))
            pos_time_batch = autograd.Variable(longTensor(pos_time_batch))
            neg_h_batch = autograd.Variable(longTensor(neg_h_batch))
            neg_t_batch = autograd.Variable(longTensor(neg_t_batch))
            neg_r_batch = autograd.Variable(longTensor(neg_r_batch))
            neg_time_batch = autograd.Variable(longTensor(neg_time_batch))

            pos, neg = model(pos_h_batch, pos_t_batch, pos_r_batch, pos_time_batch, neg_h_batch, neg_t_batch, neg_r_batch, neg_time_batch)

            if args.loss_type == 0:
                losses = loss_function(pos, neg, margin)
            else:
                losses = loss_function(pos, neg)
            ent_embeddings = model.ent_embeddings(torch.cat([pos_h_batch, pos_t_batch, neg_h_batch, neg_t_batch]))
            rel_embeddings = model.rel_embeddings(torch.cat([pos_r_batch, neg_r_batch]))
            rseq_embeddings = model.get_rseq(torch.cat([pos_r_batch, neg_r_batch]), torch.cat([pos_time_batch, neg_time_batch]))
            losses = losses + loss.normLoss(ent_embeddings) + loss.normLoss(rel_embeddings) + loss.normLoss(rseq_embeddings)
            print("Valid batch loss: %d %f" % (epoch, losses.item()))
            # print("Valid batch loss: %d %f" % (epoch, losses.data[0]))
            # agent.append(validCurve, epoch, losses.data[0])

        if config.early_stopping_round > 0:
            if epoch == 0:
                ent_embeddings = model.ent_embeddings.weight.data.cpu().numpy()
                rel_embeddings = model.rel_embeddings.weight.data.cpu().numpy()
                L1_flag = model.L1_flag
                filter = model.filter
                hit10, best_meanrank = evaluation(validList, tripleDict, ent_embeddings, rel_embeddings,
                    L1_flag, filter, config.batch_size, num_processes=args.num_processes)
                # agent.append(hit10Curve, epoch, hit10)
                # agent.append(meanrankCurve, epoch, best_meanrank)
                # torch.save(model, os.path.join('./model/' + args.dataset, filename))
                torch.save(model, os.path.join('./model/', filename))
                best_epoch = 0
                meanrank_not_decrease_time = 0
                lr_decrease_time = 0
                #if USE_CUDA:
                    #model.cuda()

            # Evaluate on validation set for every 20 epochs
            elif epoch % 20 == 0:
                ent_embeddings = model.ent_embeddings.weight.data.cpu().numpy()
                rel_embeddings = model.rel_embeddings.weight.data.cpu().numpy()
                L1_flag = model.L1_flag
                filter = model.filter
                hit10, now_meanrank = evaluation(validList, tripleDict, ent_embeddings, rel_embeddings,
                    L1_flag, filter, config.batch_size, num_processes=args.num_processes)
                # agent.append(hit10Curve, epoch, hit10)
                # agent.append(meanrankCurve, epoch, now_meanrank)
                if now_meanrank < best_meanrank:
                    meanrank_not_decrease_time = 0
                    best_meanrank = now_meanrank
                    # torch.save(model, os.path.join('./model/' + args.dataset, filename))
                    torch.save(model, os.path.join('./model/', filename))
                else:
                    meanrank_not_decrease_time += 1
                    # If the result hasn't improved for consecutive 5 evaluations, decrease learning rate
                    if meanrank_not_decrease_time == 5:
                        lr_decrease_time += 1
                        if lr_decrease_time == config.early_stopping_round:
                            break
                        else:
                            optimizer.param_groups[0]['lr'] *= 0.5
                            meanrank_not_decrease_time = 0
                #if USE_CUDA:
                    #model.cuda()

        elif (epoch + 1) % 10 == 0 or epoch == 0:
            # torch.save(model, os.path.join('./model/' + args.dataset, filename))
            torch.save(model, os.path.join('./model/', filename))

    testTotal, testList, testDict, testTimes = load_quadruples('./data/icews14/', 'test2id.txt', 'test_tem.npy')

    ent_embeddings = model.ent_embeddings.weight.data.cpu().numpy()

    hit10Test, meanrankTest = evaluation(testList, tripleDict, model, ent_embeddings, head=0)

    writeList = [filename,
        'testSet', '%.6f' % hit10Test, '%.6f' % meanrankTest]

    # Write the result into file
    with open(os.path.join('./result/', 'icews2014.txt'), 'a') as fw:
        fw.write('\t'.join(writeList) + '\n')
