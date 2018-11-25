# -*- coding: utf-8 -*-
from numpy import *
import operator
from os import listdir
import random


def create_data_set():
    group = array([[1.0, 1.1], [1.0, 1.0], [0.0, 0.1], [0.0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(in_x, data_set, labels, k):
    data_set_size = data_set.shape[0]
    diff_mat = tile(in_x, (data_set_size, 1)) - data_set
    distances = (diff_mat ** 2).sum(axis=1) ** 0.5
    sorted_dist_index = distances.argsort()
    class_count = {}
    for i in range(k):
        vote_ilabel = labels[sorted_dist_index[i]]
        class_count[vote_ilabel] = class_count.get(vote_ilabel, 0) + 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def auto_norm(data_set):
    min_val = data_set.min(0)
    max_val = data_set.max(0)
    ranges = max_val - min_val
    m = data_set.shape[0]
    norm_data_set = data_set - tile(min_val, (m, 1))
    norm_data_set = norm_data_set / tile(ranges, (m, 1))  # element wise divide
    return norm_data_set, ranges, min_val


def file2matrix(filename):
    fr = open(filename)
    number_of_lines = len(fr.readlines())  # get the number of lines in the file
    data_set = zeros((number_of_lines, 3))  # prepare matrix to return
    labs = []  # prepare labels return
    fr = open(filename)
    for index, line in enumerate(fr.readlines()):
        line = line.strip()
        list_from_line = line.split('\t')
        data_set[index, :] = list_from_line[0:3]
        labs.append(int(list_from_line[-1]))
    return data_set, labs


def dating_class_test():
    ho_ratio = 0.10  # hold out 10%
    dating_data_mat, dating_labels = file2matrix('EXTRAS/testSet.txt')  # load data setfrom file
    norm_mat, ranges, min_val = auto_norm(dating_data_mat)
    m = norm_mat.shape[0]
    num_test_vecs = int(m * ho_ratio)
    error_count = 0.0
    for i in range(num_test_vecs):
        classifier_result = classify0(norm_mat[i, :], norm_mat[num_test_vecs:m, :], dating_labels[num_test_vecs:m], 3)
        if classifier_result != dating_labels[i]:
            print("the classifier came back with: {}, the real answer is: {}".format(classifier_result, dating_labels[i]))
            error_count += 1.0
    print("the total error rate is: %f" % (error_count / float(num_test_vecs)))


def img2vector(filename):
    data_set = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        line_str = fr.readline()
        for j in range(32):
            data_set[0, 32 * i + j] = int(line_str[j])
    return data_set


def handwriting_class_test():
    hw_labels = []
    training_file_list = listdir('trainingDigits')           #load the training set
    m = len(training_file_list)
    training_mat = zeros((m,1024))
    for i in range(m):
        file_name_str = training_file_list[i]
        file_str = file_name_str.split('.')[0]     # take off .txt
        class_num_str = int(file_str.split('_')[0])
        hw_labels.append(class_num_str)
        training_mat[i, :] = img2vector('trainingDigits/%s' % file_name_str)

    test_file_list = listdir('testDigits')        #iterate through the test set
    error_count = 0.0
    m_test = len(test_file_list)
    for i in range(m_test):
        file_name_str = test_file_list[i]
        file_str = file_name_str.split('.')[0]     #take off .txt
        class_num_str = int(file_str.split('_')[0])
        vector_under_test = img2vector('testDigits/%s' % file_name_str)
        classifier_result = classify0(vector_under_test, training_mat, hw_labels, 3)
        if classifier_result != class_num_str:
            print("the classifier came back with: {}, the real answer is: {}".format(classifier_result, class_num_str))
            error_count += 1.0
    print("the total number of errors is: {}".format(error_count))
    print("the total error rate is: {}".format(error_count/float(m_test)))


if __name__ == "__main__":
    data, lab = create_data_set()
    print(classify0([0, 0.2], data, lab, 3))
    dating_class_test()
    handwriting_class_test()
