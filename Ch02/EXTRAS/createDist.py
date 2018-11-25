'''
Created on Oct 6, 2010

@author: Peter
'''
from numpy import *
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def create_data_set(n):
    # n = data_set.shape[0]   # number of points to create
    training_mat = zeros((n, 3))
    labels = zeros((n, 1))

    for i in range(n):
        [r0, r1] = random.standard_normal(2)    # 扰动
        my_class = random.uniform(0, 1)     # 决定类别
        if my_class <= 0.16:
            x1 = random.uniform(22000, 60000)
            x2 = 3 + 1.6 * r1
            lab = 1
        elif my_class <= 0.33:
            x1 = 6000 * r0 + 70000
            x2 = 10 + 3 * r1 + 2 * r0
            lab = 1
        elif my_class <= 0.66:
            x1 = 5000 * r0 + 10000
            x2 = 3 + 2.8 * r1
            lab = 2
        else:
            x1 = 10000*r0 + 35000
            x2 = 10 + 2.0*r1
            lab = 3
        x2 = 0 if x2 < 0 else x2
        x1 = 0 if x1 < 0 else x1
        training_mat[i, :] = [x1, x2, random.uniform(0.0, 1.7)]
        labels[i] = lab
    return training_mat, labels


def draw_data_set(data_set, labs):
    markers =[]
    colors =[]

    fw = open('testSet.txt','w')

    for (x1, x2, x3), lab in zip(data_set, labs):
        if lab == 1:
            markers.append(20)
            colors.append("red")
        elif lab == 2:
            markers.append(30)
            colors.append("green")
        elif lab == 3:
            markers.append(50)
            colors.append("blue")
        fw.write("{}\t{}\t{}\t{}\n" .format(x1, x2, x3, lab))
    fw.close()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(data_set[:, 0], data_set[:, 1], c=colors, s=markers)
    type1 = ax.scatter([-10], [-10], s=20, c='red')
    type2 = ax.scatter([-10], [-15], s=30, c='green')
    type3 = ax.scatter([-10], [-20], s=50, c='blue')
    ax.legend([type1, type2, type3], ["Class 1", "Class 2", "Class 3"], loc=4)
    #ax.axis([-5000,100000,-2,25])
    plt.xlabel('Frequent Flyier Miles Earned Per Year')
    plt.ylabel('Percentage of Body Covered By Tatoos')
    plt.show()


if __name__ == "__main__":
    data_set, labs = create_data_set(1000)
    draw_data_set(data_set, labs)
