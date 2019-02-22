import os
import numpy as np
import time
import datetime
import random
import multiprocessing
import math
import torch
import torch.autograd as autograd
from torch.autograd import Variable

from sklearn.metrics.pairwise import pairwise_distances, cosine_similarity

from data import *

USE_CUDA = torch.cuda.is_available()

if USE_CUDA:
	longTensor = torch.cuda.LongTensor
	floatTensor = torch.cuda.FloatTensor

else:
	longTensor = torch.LongTensor
	floatTensor = torch.FloatTensor


def isHit10(triple, tree, cal_embedding, tripleDict, isTail):
    # If isTail == True, evaluate the prediction of tail entity
    if isTail == True:
        k = 0
        wrongCount = 0
        while wrongCount < 10:
            k += 15
            tail_dist, tail_ind = tree.query(cal_embedding, k=k)
            for elem in tail_ind[0][k - 15: k]:
                if triple.t == elem:
                    return True
                elif (triple.h, elem, triple.r) in tripleDict:
                    continue
                else:
                    wrongCount += 1
                    if wrongCount > 9:
                        return False
    # If isTail == False, evaluate the prediction of head entity
    else:
        k = 0
        wrongCount = 0
        while wrongCount < 10:
            k += 15
            head_dist, head_ind = tree.query(cal_embedding, k=k)
            for elem in head_ind[0][k - 15: k]:
                if triple.h == elem:
                    return True
                elif (elem, triple.t, triple.r) in tripleDict:
                    continue
                else:
                    wrongCount += 1
                    if wrongCount > 9:
                        return False

# Find the rank of ground truth tail in the distance array,
# If (head, num, rel) in tripleDict,
# skip without counting.
def argwhereTail(head, tail, rel, array, tripleDict):
    wrongAnswer = 0
    for num in array:
        if num == tail:
            return wrongAnswer
        elif (head, num, rel) in tripleDict:
            continue
        else:
            wrongAnswer += 1
    return wrongAnswer

# Find the rank of ground truth head in the distance array,
# If (head, num, rel) in tripleDict,
# skip without counting.
def argwhereHead(head, tail, rel, array, tripleDict):
    wrongAnswer = 0
    for num in array:
        if num == head:
            return wrongAnswer
        elif (num, tail, rel) in tripleDict:
            continue
        else:
            wrongAnswer += 1
    return wrongAnswer

def pairwise_L1_distances(A, B):
    dist = torch.sum(torch.abs(A.unsqueeze(1) - B.unsqueeze(0)), dim=2)
    return dist

def pairwise_L2_distances(A, B):
    AA = torch.sum(A ** 2, dim=1).unsqueeze(1)
    BB = torch.sum(B ** 2, dim=1).unsqueeze(0)
    dist = torch.mm(A, torch.transpose(B, 0, 1))
    dist *= -2
    dist += AA
    dist += BB
    return dist


def mrr_mr_hitk(scores, target, k=10):
    _, sorted_idx = torch.sort(scores)
    find_target = sorted_idx == target
    target_rank = torch.nonzero(find_target)[0, 0]
    return 1 / (target_rank+1), target_rank, int(target_rank < 1), int(target_rank < 3), int(target_rank < 10)


def evaluation_helper(testList, tripleDict, model, ent_embeddings, n_ent, L1_flag, filter, head=0):
    batch_s, batch_t, batch_r, batch_tem = getFourElements(testList)

    mrr_tot = 0
    mr_tot = 0
    hit1_tot = 0
    hit3_tot = 0
    hit10_tot = 0
    count = 0

    batch_size = len(batch_s)
    batch_r = Variable(longTensor(batch_r).cuda())
    batch_s = Variable(longTensor(batch_s).cuda())
    batch_t = Variable(longTensor(batch_t).cuda())
    batch_tem = Variable(longTensor(batch_tem).cuda())
    rel_var = Variable(batch_r.unsqueeze(1).expand(batch_size, n_ent).cuda())
    tem_var = Variable(batch_tem.unsqueeze(1).expand(batch_size, n_ent, batch_tem.size(-1)).cuda())
    src_var = Variable(batch_s.unsqueeze(1).expand(batch_size, n_ent).cuda())
    dst_var = Variable(batch_t.unsqueeze(1).expand(batch_size, n_ent).cuda())
    all_var = Variable(torch.arange(0, n_ent).unsqueeze(0).expand(batch_size, n_ent)
                       .type(torch.LongTensor).cuda(), volatile=True)
    batch_dst_scores = []
    batch_src_scores = []
    for i in range(batch_size):
        dst_scores = model.score(src_var[i], all_var[i], rel_var[i], tem_var[i]).data
        src_scores = model.score(all_var[i], dst_var[i], rel_var[i], tem_var[i]).data
        batch_dst_scores.append(dst_scores)
        batch_src_scores.append(src_scores)
    for s, r, t, dst_scores, src_scores in zip(batch_s, batch_r, batch_t, batch_dst_scores, batch_src_scores):
        # if filt:
        #     if tails[(s, r)]._nnz() > 1:
        #         tmp = dst_scores[t]
        #         dst_scores += tails[(s, r)].cuda() * 1e30
        #         dst_scores[t] = tmp
        #     if heads[(t, r)]._nnz() > 1:
        #         tmp = src_scores[s]
        #         src_scores += heads[(t, r)].cuda() * 1e30
        #         src_scores[s] = tmp
        mrr, mr, hit1, hit3, hit10 = mrr_mr_hitk(dst_scores, t)
        mrr_tot += mrr
        mr_tot += mr
        hit1_tot += hit1
        hit3_tot += hit3
        hit10_tot += hit10
        mrr, mr, hit1, hit3, hit10 = mrr_mr_hitk(src_scores, s)
        mrr_tot += mrr
        mr_tot += mr
        hit1_tot += hit1
        hit3_tot += hit3
        hit10_tot += hit10
        count += 2

    return hit1_tot, hit3_tot, hit10_tot, mr_tot, mrr_tot, count


def process_data(testList, tripleDict, model, ent_embeddings, n_ent, L1_flag, filter, L, head):
    hit1Count, hit3Count, hit10Count, totalRank, totalReRank, tripleCount = evaluation_helper(testList, tripleDict, model, ent_embeddings, n_ent, L1_flag, filter, head)

    L.append((hit1Count, hit3Count, hit10Count, totalRank, totalReRank, tripleCount))


# Use multiprocessing to speed up evaluation
def evaluation(testList, tripleDict, model, ent_embeddings, n_ent, L1_flag, filter, k=0, head=0):
    # embeddings are numpy like

    if k > len(testList):
        testList = random.choices(testList, k=k)
    elif k > 0:
        testList = random.sample(testList, k=k)

    L = []
    process_data(testList, tripleDict, model, ent_embeddings, n_ent, L1_flag, filter, L, head)

    resultList = list(L)

    # what is head?
    if head == 1 or head == 2:
        hit1 = sum([elem[0] for elem in resultList]) / len(testList)
        hit3 = sum([elem[1] for elem in resultList]) / len(testList)
        hit10 = sum([elem[2] for elem in resultList]) / len(testList)
        meanrank = sum([elem[3] for elem in resultList]) / len(testList)
        meanrerank = sum([elem[4] for elem in resultList]) / len(testList)
    else:
        hit1 = sum([elem[0] for elem in resultList]) / (2 * len(testList))
        hit3 = sum([elem[1] for elem in resultList]) / (2 * len(testList))
        hit10 = sum([elem[2] for elem in resultList]) / (2 * len(testList))
        meanrank = sum([elem[3] for elem in resultList]) / (2 * len(testList))
        meanrerank = sum([elem[4] for elem in resultList]) / (2 * len(testList))
        # hit1 = sum([elem[0] for elem in resultList])
        # hit3 = sum([elem[1] for elem in resultList])
        # hit10 = sum([elem[2] for elem in resultList])
        # meanrank = sum([elem[3] for elem in resultList])
        # meanrerank = sum([elem[4] for elem in resultList])

    print('Meanrank: %.6f' % meanrank)
    print('Meanrerank: %.6f' % meanrerank)
    print('Hit@1: %.6f' % hit1)
    print('Hit@3: %.6f' % hit3)
    print('Hit@10: %.6f' % hit10)

    return hit1, hit3, hit10, meanrank, meanrerank


def evaluation_batch(testList, tripleDict, model, ent_embeddings, n_ent, L1_flag, filter, k=0, head=0):
    # embeddings are numpy like

    if k > len(testList):
        testList = random.choices(testList, k=k)
    elif k > 0:
        testList = random.sample(testList, k=k)

    L = []
    process_data(testList, tripleDict, model, ent_embeddings, n_ent, L1_flag, filter, L, head)

    resultList = list(L)

    hit1 = sum([elem[0] for elem in resultList])
    hit3 = sum([elem[1] for elem in resultList])
    hit10 = sum([elem[2] for elem in resultList])
    meanrank = sum([elem[3] for elem in resultList])
    meanrerank = sum([elem[4] for elem in resultList])

    if head == 1 or head == 2:
        return hit1, hit3, hit10, meanrank, meanrerank, len(testList)
    else:
        return hit1, hit3, hit10, meanrank, meanrerank, 2 * len(testList)

