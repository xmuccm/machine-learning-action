# -*- coding: utf-8 -*-
# from numpy import *
import operator


def create_data_set():
    group = array([[[1.0, 1.1]], [1.0, 1.0], [0.0, 0.1], [0.0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(in_x, data_set, labels, k):
    data_set_size = data_set.shape[0]
    print(tile(in_x, (data_set_size, 1)))

    diffMat = tile(in_x, (data_set_size, 1)) - data_set
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

        # if classCount.has_key(voteIlabel):
        #    classCount[voteIlabel] += 1
        # else:
        #    classCount[voteIlabel] = 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]



if __name__ == "__main__":
    # data, lab = create_data_set()
    # classify0([0, 0.2], data, lab, 3)
    a = u"我打发生".encode("utf-8")
    print(type(a))
    print(a)

    a = u"我打发生"
    print(type(a))
    print(a)