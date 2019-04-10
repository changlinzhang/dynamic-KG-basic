import os
import random
from copy import deepcopy
import numpy as np
import operator

from utils import Triple

# Change the head of a triple randomly,
# without checking whether it is a false negative sample.
def corrupt_head_raw(quadruple, entityTotal):
    newQuadruple = deepcopy(quadruple)
    oldHead = quadruple.s
    while True:
        newHead = random.randrange(entityTotal)
        if newHead != oldHead:
            break
    newQuadruple.s = newHead
    return newQuadruple

# Change the tail of a triple randomly,
# without checking whether it is a false negative sample.
def corrupt_tail_raw(quadruple, entityTotal):
    newQuadruple = deepcopy(quadruple)
    oldTail = newQuadruple.o
    while True:
        newTail = random.randrange(entityTotal)
        if newTail != oldTail:
            break
    newQuadruple.o = newTail
    return newQuadruple

# Change the head of a triple randomly,
# with checking whether it is a false negative sample.
# If it is, regenerate.
def corrupt_head_filter(quadruple, entityTotal, quadrupleDict):
    newQuadruple = deepcopy(quadruple)
    while True:
        newHead = random.randrange(entityTotal)
        if (newHead, newQuadruple.o, newQuadruple.r) not in quadrupleDict:
            break
    newQuadruple.s = newHead
    return newQuadruple

# Change the tail of a triple randomly,
# with checking whether it is a false negative sample.
# If it is, regenerate.
def corrupt_tail_filter(quadruple, entityTotal, quadrupleDict):
    newQuadruple = deepcopy(quadruple)
    while True:
        newTail = random.randrange(entityTotal)
        if (newQuadruple.s, newTail, newQuadruple.r) not in quadrupleDict:
            break
    newQuadruple.o = newTail
    return newQuadruple

# Split the tripleList into #num_batches batches
# def getBatchList(tripleList, num_batches):
#     batchSize = len(tripleList) // num_batches
#     batchList = [0] * num_batches
#     for i in range(num_batches - 1):
#         batchList[i] = tripleList[i * batchSize : (i + 1) * batchSize]
#     batchList[num_batches - 1] = tripleList[(num_batches - 1) * batchSize : ]
#     return batchList
def getBatchList(tripleList, batch_size):
    num_batches = len(tripleList) // batch_size + 1
    batchList = [0] * num_batches
    for i in range(num_batches - 1):
        batchList[i] = tripleList[i * batch_size : (i + 1) * batch_size]
    batchList[num_batches - 1] = tripleList[(num_batches - 1) * batch_size : ]
    return batchList

def getFourElements(quadrupleList):
    headList = [quadruple.s for quadruple in quadrupleList]
    tailList = [quadruple.o for quadruple in quadrupleList]
    relList = [quadruple.r for quadruple in quadrupleList]
    timeList = [quadruple.t for quadruple in quadrupleList]
    return headList, tailList, relList, timeList

def getFourElement(quadruple):
    return quadruple.s, quadruple.o, quadruple.r, quadruple.t

def getThreeElements(tripleList):
    headList = [triple.s for triple in tripleList]
    tailList = [triple.o for triple in tripleList]
    relList = [triple.r for triple in tripleList]
    return headList, tailList, relList

def getThreeElement(triple):
    return triple.s, triple.o, triple.r

# Use all the tripleList,
# and generate negative samples by corrupting head or tail with equal probabilities,
# without checking whether false negative samples exist.
def getBatch_raw_all(quadrupleList, entityTotal, mult_num = 1):
    newQuadrupleList = [corrupt_head_raw(quadruple, entityTotal) if random.random() < 0.5
        else corrupt_tail_raw(quadruple, entityTotal) for quadruple in quadrupleList]
    if mult_num > 1:
        for i in range(0, mult_num-1):
            newQuadrupleList2 = [corrupt_head_raw(quadruple, entityTotal) if random.random() < 0.5
                                else corrupt_tail_raw(quadruple, entityTotal) for quadruple in quadrupleList]
            newQuadrupleList.extend(newQuadrupleList2)
    ps, po, pr, pt = getFourElements(quadrupleList)
    ns, no, nr, nt = getFourElements(newQuadrupleList)
    return ps, po, pr, pt, ns, no, nr, nt

# Use all the tripleList,
# and generate negative samples by corrupting head or tail with equal probabilities,
# with checking whether false negative samples exist.
def getBatch_filter_all(quadrupleList, entityTotal, quadrupleDict, mult_num = 1):
    newQuadrupleList = [corrupt_head_filter(quadruple, entityTotal, quadrupleDict) if random.random() < 0.5
        else corrupt_tail_filter(quadruple, entityTotal, quadrupleDict) for quadruple in quadrupleList]
    if mult_num > 1:
        for i in range(0, mult_num-1):
            newQuadrupleList2 = [corrupt_head_filter(quadruple, entityTotal, quadrupleDict) if random.random() < 0.5
                            else corrupt_tail_filter(quadruple, entityTotal, quadrupleDict) for quadruple in
                            quadrupleList]
            newQuadrupleList.extend(newQuadrupleList2)
    ps, po, pr, pt = getFourElements(quadrupleList)
    ns, no, nr, nt = getFourElements(newQuadrupleList)
    return ps, po, pr, pt, ns, no, nr, nt

# Sample a batch of #batchSize triples from tripleList,
# and generate negative samples by corrupting head or tail with equal probabilities,
# without checking whether false negative samples exist.
def getBatch_raw_random(quadrupleList, batchSize, entityTotal):
    oldQuadrupleList = random.sample(quadrupleList, batchSize)
    newQuadrupleList = [corrupt_head_raw(quadruple, entityTotal) if random.random() < 0.5
        else corrupt_tail_raw(quadruple, entityTotal) for quadruple in oldQuadrupleList]
    ph, po, pr, pt = getFourElements(oldQuadrupleList)
    nh, no, nr, nt = getFourElements(newQuadrupleList)
    return ph, po, pr, pt, nh, no, nr, nt

# Sample a batch of #batchSize triples from tripleList,
# and generate negative samples by corrupting head or tail with equal probabilities,
# with checking whether false negative samples exist.
def getBatch_filter_random(quadrupleList, batchSize, entityTotal, quadrupleDict):
    oldQuadrupleList = random.sample(quadrupleList, batchSize)
    newQuadrupleList = [corrupt_head_filter(quadruple, entityTotal, quadrupleDict) if random.random() < 0.5
        else corrupt_tail_filter(quadruple, entityTotal, quadrupleDict) for quadruple in oldQuadrupleList]
    ph, po, pr, pt = getFourElements(oldQuadrupleList)
    nh, no, nr, nt = getFourElements(newQuadrupleList)
    return ph, po, pr, pt, nh, no, nr, nt


######################################
######################################
##############Triple##################
######################################
######################################

# Change the head of a triple randomly,
# without checking whether it is a false negative sample.
def corrupt_head_raw_tri(triple, entityTotal):
    newTriple = deepcopy(triple)
    oldHead = triple.s
    while True:
        newHead = random.randrange(entityTotal)
        if newHead != oldHead:
            break
    newTriple.s = newHead
    return newTriple

# Change the tail of a triple randomly,
# without checking whether it is a false negative sample.
def corrupt_tail_raw_tri(triple, entityTotal):
    newTriple = deepcopy(triple)
    oldTail = triple.o
    while True:
        newTail = random.randrange(entityTotal)
        if newTail != oldTail:
            break
    newTriple.o = newTail
    return newTriple

# Change the head of a triple randomly,
# with checking whether it is a false negative sample.
# If it is, regenerate.
def corrupt_head_filter_tri(triple, entityTotal, tripleDict):
    newTriple = deepcopy(triple)
    while True:
        newHead = random.randrange(entityTotal)
        if (newHead, newTriple.o, newTriple.r) not in tripleDict:
            break
    newTriple.s = newHead
    return newTriple

# Change the tail of a triple randomly,
# with checking whether it is a false negative sample.
# If it is, regenerate.
def corrupt_tail_filter_tri(triple, entityTotal, tripleDict):
    newTriple = deepcopy(triple)
    while True:
        newTail = random.randrange(entityTotal)
        if (newTriple.s, newTail, newTriple.r) not in tripleDict:
            break
    newTriple.o = newTail
    return newTriple


def getBatch_raw_random_tri(tripleList, batchSize, entityTotal):
    oldTripleList = random.sample(tripleList, batchSize)
    newTripleList = [corrupt_head_raw_tri(triple, entityTotal) if random.random() < 0.5
        else corrupt_tail_raw_tri(triple, entityTotal) for triple in oldTripleList]
    ph, pt ,pr = getThreeElements(oldTripleList)
    nh, nt, nr = getThreeElements(newTripleList)
    return ph, pt, pr, nh, nt, nr


# Use all the tripleList,
# and generate negative samples by corrupting head or tail with equal probabilities,
# without checking whether false negative samples exist.
def getBatch_raw_all_tri(tripleList, entityTotal):
    newTripleList = [corrupt_head_raw_tri(triple, entityTotal) if random.random() < 0.5
        else corrupt_tail_raw_tri(triple, entityTotal) for triple in tripleList]
    ph, pt ,pr = getThreeElements(tripleList)
    nh, nt, nr = getThreeElements(newTripleList)
    return ph, pt, pr, nh, nt, nr


# Sample a batch of #batchSize triples from tripleList,
# and generate negative samples by corrupting head or tail with equal probabilities,
# with checking whether false negative samples exist.
def getBatch_filter_random_tri(tripleList, batchSize, entityTotal, tripleDict):
    oldTripleList = random.sample(tripleList, batchSize)
    newTripleList = [corrupt_head_filter_tri(triple, entityTotal, tripleDict) if random.random() < 0.5
        else corrupt_tail_filter_tri(triple, entityTotal, tripleDict) for triple in oldTripleList]
    ph, pt ,pr = getThreeElements(oldTripleList)
    nh, nt, nr = getThreeElements(newTripleList)
    return ph, pt, pr, nh, nt, nr


# Use all the tripleList,
# and generate negative samples by corrupting head or tail with equal probabilities,
# with checking whether false negative samples exist.
def getBatch_filter_all_tri(tripleList, entityTotal, tripleDict):
    newTripleList = [corrupt_head_filter_tri(triple, entityTotal, tripleDict) if random.random() < 0.5
        else corrupt_tail_filter_tri(triple, entityTotal, tripleDict) for triple in tripleList]
    ph, pt ,pr = getThreeElements(tripleList)
    nh, nt, nr = getThreeElements(newTripleList)
    return ph, pt, pr, nh, nt, nr


def getTimestampBatchList(quadrupleList):
    batchList = []
    tmpList = []
    preTimestamp = []
    for i in range(len(quadrupleList)):
        if not operator.eq(quadrupleList[i].t, preTimestamp):
            if len(preTimestamp) != 0:
                batchList.append(deepcopy(tmpList))
            preTimestamp = quadrupleList[i].t
            tmpList = []
        tmpList.append(quadrupleList[i])
    batchList.append(deepcopy(tmpList))
    return batchList
