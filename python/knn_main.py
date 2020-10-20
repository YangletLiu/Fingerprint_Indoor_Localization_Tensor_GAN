#coding=utf-8
import numpy as np
import scipy.io as scio
from sklearn import neighbors
import operator
import matplotlib
import matplotlib.pyplot as plt
import random
import math

#50cm * 50cm

def data_process():
    # Data PreProcess
    with open('Radio_map_3D.mat','rb') as fileholder:
        trainset = scio.loadmat(fileholder)
        data = np.array(trainset['Radio_map_3D'])

    print('Data read complete')
    return data

def test(data):  # 已知数据抽样添加噪声打乱顺序后作为未知测试数据

    labeled_data = [] # 加标签
    for index_i,i in enumerate(data,0):
        for index_j,j in enumerate(i,0):
            labeled_data.append(j)
    print('Data labeled finish')
    traintarget = range(62500)
    # 抽取部分样本添加噪声作为测试集
    testset = []
    testtarget = []
    for rand in range(500):
        index_i = random.randint(0,249)
        index_j = random.randint(0,249)
        testdata = []
        for item in data[index_i][index_j]:
            testdata.append(item + random.uniform(-10,10))
        testset.append(testdata)
        testtarget.append(index_i*25+index_j)
    print('Test set complete')
    n_neighbors = 1
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')
    clf.fit(labeled_data,traintarget)
    Z = clf.predict(testset)
    print (testtarget*10)
    print (Z)
    sum = 0
    for i in range(500):
        x1 = Z[i] % 250
        y1 = Z[i] / 250
        x2 = (testtarget[i]*10) % 250
        y2 = (testtarget[i]*10) / 250
        distance = math.sqrt(abs((x1-x2)**2 + (y1-y2)**2))
        print("distance: %f" % distance)
        sum += distance
    print('total result: ')
    print(sum / 500)

if __name__ == "__main__":
    data = data_process()
    test(data)

