'''
Created on Oct 6, 2010

@author: Peter
'''
from numpy import *
import matplotlib
import matplotlib.pyplot as plt


def create_date_set(n):
    fw = open('testSet.txt', 'w')
    training_mat = zeros((n, 2))
    labels = zeros((n, 1))
    for i in range(n):
        r0, r1 = random.standard_normal(2)
        if random.uniform(0,1) <= 0.5:
            x0 = r0 + 9.0
            x1 = 1.0 * r1 + x0 - 9.0
            class_label = 0
        else:
            x0 = r0 + 2.0
            x1 = r1 + x0 - 2.0
            class_label = 1
        fw.write("%f\t%f\t%d\n" % (x0, x1, class_label))
        training_mat[i, :] = [x0, x1]
        labels[i] = class_label
    fw.close()
    return training_mat, labels


def draw_data(training_mat, labels):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    marker = [20 if i else 50 for i in labels]
    col = ['red' if i else 'blue' for i in labels]
    ax.scatter(training_mat[:, 0], training_mat[:, 1], s=marker, c=col)
    # plt.plot([0,1], label='going up')
    plt.show()


if __name__ == "__main__":
    training_mats, label = create_date_set(100)
    draw_data(training_mats, label)


