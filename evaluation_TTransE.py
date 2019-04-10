import os
import numpy as np
import time
import datetime
import random
import multiprocessing
import math
import torch
import torch.autograd as autograd

from sklearn.metrics.pairwise import pairwise_distances, cosine_similarity

from data import *

USE_CUDA = torch.cuda.is_available()

if USE_CUDA:
	longTensor = torch.cuda.LongTensor
	floatTensor = torch.cuda.FloatTensor

else:
	longTensor = torch.LongTensor
	floatTensor = torch.FloatTensor


# Find the rank of ground truth tail in the distance array,
# If (head, num, rel) in tripleDict,
# skip without counting.
def argwhereTail(head, tail, rel, time, array, quadrupleDict):
    wrongAnswer = 0
    for num in array:
        if num == tail:
            return wrongAnswer
        elif (head, num, rel, time) in quadrupleDict:
            continue
        else:
            wrongAnswer += 1
    return wrongAnswer

# Find the rank of ground truth head in the distance array,
# If (head, num, rel) in tripleDict,
# skip without counting.
def argwhereHead(head, tail, rel, time, array, quadrupleDict):
    wrongAnswer = 0
    for num in array:
        if num == head:
            return wrongAnswer
        elif (num, tail, rel, time) in quadrupleDict:
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


def evaluation_helper(testList, tripleDict, dict, model, ent_embeddings, rel_embeddings, tem_embeddings, L1_flag, filter, head=0):
    # embeddings are numpy likre
    headList, tailList, relList, timeList = getFourElements(testList)
    h_e = ent_embeddings[headList]
    t_e = ent_embeddings[tailList]
    r_e = rel_embeddings[relList]
    time_e = tem_embeddings[timeList]

    c_t_e = h_e + (r_e + time_e)
    c_h_e = t_e - (r_e + time_e)
    if L1_flag == True:
        dist = pairwise_distances(c_t_e, ent_embeddings, metric='manhattan')
    else:
        dist = pairwise_distances(c_t_e, ent_embeddings, metric='euclidean')

    rankArrayTail = np.argsort(dist, axis=1)
    if filter == False:
        rankListTail = [int(np.argwhere(elem[1]==elem[0])) for elem in zip(tailList, rankArrayTail)]
    else:
        rankListTail = [argwhereTail(elem[0], elem[1], elem[2], elem[3], elem[4], tripleDict)
                        for elem in zip(headList, tailList, relList, timeList, rankArrayTail)]

    if L1_flag == True:
        dist = pairwise_distances(c_h_e, ent_embeddings, metric='manhattan')
    else:
        dist = pairwise_distances(c_h_e, ent_embeddings, metric='euclidean')

    rankArrayHead = np.argsort(dist, axis=1)
    if filter == False:
        rankListHead = [int(np.argwhere(elem[1]==elem[0])) for elem in zip(headList, rankArrayHead)]
    else:
        rankListHead = [argwhereHead(elem[0], elem[1], elem[2], elem[3], elem[4], tripleDict)
                        for elem in zip(headList, tailList, relList, timeList, rankArrayHead)]

    # rankList = np.concatenate((rankListHead, rankListTail))
    return rankListHead, rankListTail


def process_data(testList, tripleDict, dict, model, ent_embeddings, rel_embeddings, tem_embeddings, L1_flag, filter, L, head):
    rankListHead, rankListTail = evaluation_helper(testList, tripleDict, dict, model, ent_embeddings, rel_embeddings, tem_embeddings, L1_flag, filter, head)

    L.append((rankListHead, rankListTail))


# Use multiprocessing to speed up evaluation
def evaluation(testList, tripleDict, dict, model, ent_embeddings, rel_embeddings, tem_embeddings, L1_flag, filter, k=0, head=0):
    # embeddings are numpy like

    if k > len(testList):
        testList = random.choices(testList, k=k)
    elif k > 0:
        testList = random.sample(testList, k=k)

    L = []
    process_data(testList, tripleDict, dict, model, ent_embeddings, rel_embeddings, tem_embeddings, L1_flag, filter, L, head)

    resultList = list(L)
    rankListHead = [elem[0] for elem in resultList]
    rankListTail = [elem[1] for elem in resultList]

    return rankListHead, rankListTail


def evaluation_batch(testList, tripleDict, dict, model, ent_embeddings, rel_embeddings, tem_embeddings, L1_flag, filter, k=0, head=0):
    # embeddings are numpy like

    if k > len(testList):
        testList = random.choices(testList, k=k)
    elif k > 0:
        testList = random.sample(testList, k=k)

    L = []
    process_data(testList, tripleDict, dict, model, ent_embeddings, rel_embeddings, tem_embeddings, L1_flag, filter, L, head)

    resultList = list(L)
    rankListHead = [elem[0] for elem in resultList]
    rankListTail = [elem[1] for elem in resultList]

    return rankListHead, rankListTail
