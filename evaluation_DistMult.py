import os
import numpy as np
import time
import datetime
import random
import multiprocessing
import math
import torch
import torch.autograd as autograd

from sklearn.metrics.pairwise import pairwise_distances, cosine_similarity, linear_kernel

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


def evaluation_helper(testList, tripleDict, model, ent_embeddings, rel_embeddings, L1_flag, filter, head=0):
    # embeddings are numpy likre
    headList, tailList, relList = getThreeElements(testList)
    h_e = ent_embeddings[headList]
    t_e = ent_embeddings[tailList]
    r_e = rel_embeddings[relList]

    c_t_e = h_e * r_e
    c_h_e = t_e * r_e
    dist = linear_kernel(c_t_e, ent_embeddings)

    rankArrayTail = np.argsort(-dist, axis=1)
    if filter == False:
        rankListTail = [int(np.argwhere(elem[1]==elem[0])) for elem in zip(tailList, rankArrayTail)]
    else:
        rankListTail = [argwhereTail(elem[0], elem[1], elem[2], elem[3], tripleDict)
                        for elem in zip(headList, tailList, relList, rankArrayTail)]

    dist = linear_kernel(c_h_e, ent_embeddings)

    rankArrayHead = np.argsort(-dist, axis=1)
    if filter == False:
        rankListHead = [int(np.argwhere(elem[1]==elem[0])) for elem in zip(headList, rankArrayHead)]
    else:
        rankListHead = [argwhereHead(elem[0], elem[1], elem[2], elem[3], tripleDict)
                        for elem in zip(headList, tailList, relList, rankArrayHead)]

    rankList = np.concatenate((rankListHead, rankListTail))
    return rankList


def process_data(testList, tripleDict, model, ent_embeddings, rel_embeddings, L1_flag, filter, L, head):
    rankList = evaluation_helper(testList, tripleDict, model, ent_embeddings, rel_embeddings, L1_flag, filter, head)

    L.append((rankList))


# Use multiprocessing to speed up evaluation
def evaluation(testList, tripleDict, model, ent_embeddings, rel_embeddings, L1_flag, filter, k=0, head=0):
    # embeddings are numpy like

    if k > len(testList):
        testList = random.choices(testList, k=k)
    elif k > 0:
        testList = random.sample(testList, k=k)

    L = []
    process_data(testList, tripleDict, model, ent_embeddings, rel_embeddings, L1_flag, filter, L, head)

    resultList = list(L)
    rankList = resultList

    return rankList


def evaluation_batch(testList, tripleDict, model, ent_embeddings, rel_embeddings, L1_flag, filter, k=0, head=0):
    # embeddings are numpy like

    if k > len(testList):
        testList = random.choices(testList, k=k)
    elif k > 0:
        testList = random.sample(testList, k=k)

    L = []
    process_data(testList, tripleDict, model, ent_embeddings, rel_embeddings, L1_flag, filter, L, head)

    resultList = list(L)
    rankList = resultList

    return rankList

