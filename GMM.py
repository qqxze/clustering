import csv
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy import *


def loadDateSet(fileName):
    heade = ["x","y","label"]
    dateSet = pd.read_excel(fileName,header=None,names=heade)
    dateSet = dateSet.iloc[:,[0,1]]
    return dateSet
	
# 计算概率密度，
# 参数皆为array类型，过程中参数不变
def gaussian(x, mean, cov):
    dim = np.shape(cov)[0]  # 维度
    # 之所以加入单位矩阵是为了防止行列式为0的情况
    covdet = np.linalg.det(cov + np.eye(dim) * 0.01)  # 协方差矩阵的行列式
    covinv = np.linalg.inv(cov + np.eye(dim) * 0.01)  # 协方差矩阵的逆
    xdiff = x - mean
    # 概率密度
    prob = 1.0 / np.power(2 * np.pi, 1.0 * dim / 2) / np.sqrt(np.abs(covdet)) * np.exp(
        -1.0 / 2 * np.dot(np.dot(xdiff, covinv), xdiff))
    return prob


# 获取初始协方差矩阵
def getconvs(data, K):
    convs = [0] * K
    for i in range(K):
        # 初始的协方差矩阵源自于原始数据的协方差矩阵，且每个簇的初始协方差矩阵相同
        convs[i] = np.cov(data.T)
    return convs

def distEclud(vecA,vecB):
    return sqrt(sum(power(vecA - vecB, 2)))

#构建包含k个随机质心的集合
def randCent(dataSet,k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k, n)))
    for j in range(n):
        minJ = min(dataSet[:,j])
        rangeJ = float(max(dataSet[:,j]) - minJ)
        centroids[:,j] = minJ + rangeJ * random.rand(k,1)
    return  centroids

def kmeans(dateSet,k,distMeas = distEclud,createCent=randCent):
    dateSet = mat(dateSet)
    m = shape(dateSet)[0]
    clusterAssment = mat(zeros((m,2)))#簇分配结果矩阵 第一列记录簇索引值 第二列记录存储误差
    centroids = createCent(dateSet,k)
    clusterChanged = True #循环标志变量 如果为True 则继续迭代
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = inf;minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j,:],dateSet[i,:])#质心和 集合的 点的距离
                if distJI < minDist:
                    minDist = distJI;minIndex = j
            if clusterAssment[i,0] !=minIndex:
                clusterChanged = True
            clusterAssment[i,:] = minIndex,minIndex**2
        #print(centroids)
        for cent in range(k):
            ptsInClust = dateSet[nonzero(clusterAssment[:, 0].A == cent)[0]]
            centroids[cent,:] = mean(ptsInClust,axis=0)
    return np.array(centroids)
    #return centroids, clusterAssment

def visualresult(data, myCentroids, clustAssing, K):
    myCentroids = mat(myCentroids)
    data = mat(data)
    fig = plt.figure("gmm clustering")
    rect = [0.1, 0.1, 0.8, 0.8]
    scatterMarkers = ['s', 'o', '^']
    axprops = dict(xticks=[], yticks=[])
    ax0 = fig.add_axes(rect, label='ax0', **axprops)
    ax1 = fig.add_axes(rect, label='ax1', frameon=False)

    for i in range(K):
        ptsInCurrCluster = data[nonzero(clustAssing[:, 0].A == i)[0], :]  # 返回属于i标签的值
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:, 0].flatten().A[0], ptsInCurrCluster[:, 1].flatten().A[0], marker=markerStyle,
                    s=90)
    ax1.scatter(myCentroids[:, 0].flatten().A[0], myCentroids[:, 1].flatten().A[0], marker='+', s=300)
    plt.savefig("GMMout.png")
    plt.show()


def output(data, outputpath):
    with open(outputpath, "w", newline='') as f:
        headers = ["x","y","GMM_labels"]
        writer = csv.writer(f)
        writer.writerow(headers)
        for d in data:
            writer.writerow([d[0],d[1],d[2]+1])
    f.close()

def Gmm(data,K):
    N = np.shape(data)[0]  # 样本数目
    dim = np.shape(data)[1]  # 维度
    means = kmeans(data, K)
    print("km")
    print(means)
    convs = getconvs(data, K)
    pis = [1.0 / K] * K
    gammas = [np.zeros(K) for i in range(N)]  # *N 注意不能用 *N，否则N个array只指向一个地址
    loglikelyhood = 0
    oldloglikelyhood = 1
    while np.abs(loglikelyhood - oldloglikelyhood) > 0.0001:
        oldloglikelyhood = loglikelyhood
        # E_step
        for n in range(N):
            respons = [pis[k] * gaussian(data[n], means[k], convs[k]) for k in range(K)]
            sumrespons = np.sum(respons)
            for k in range(K):
                gammas[n][k] = respons[k] / sumrespons#gammas[n][k] 就可以理解为第k个高斯分布对观测值x_n发生的概率的贡献
        # M_step
        for k in range(K):
            nk = np.sum([gammas[n][k] for n in range(N)])#第k个单高斯分布对整个GMM的贡献
            means[k] = 1.0 / nk * np.sum([gammas[n][k] * data[n] for n in range(N)], axis=0)

            xdiffs = data - means[k]
            convs[k] = 1.0 / nk * np.sum([gammas[n][k] * xdiffs[n].reshape(dim, 1) * xdiffs[n] for n in range(N)], axis=0)
            pis[k] = 1.0 * nk / N #pik
        # 计算似然函数值
        loglikelyhood = np.sum(
            [np.log(np.sum([pis[k] * gaussian(data[n], means[k], convs[k]) for k in range(K)])) for n in range(N)])
    clusterAssment = mat(np.argmax(gammas, axis=1)).T
    datalabel = np.insert(data,2,values=np.array(clusterAssment)[:,0],axis=1)
    print("输出means")
    print(means)
    print("似然函数值")
    print(loglikelyhood)
    # print("第k个单高斯分布对整个GMM的贡献")
    # print(gammas)
    # print(len(gammas))
    # print("lables")
    # print(clusterAssment)
    # print('=='*10)
    return means,clusterAssment,datalabel

if __name__=='__main__':
    
    fileName = "chap_7_数据集.xlsx"
    outputpath = "GMMALL_label.csv"
    data = np.array(loadDateSet(fileName))
    m,v,d=Gmm(data,3)
    visualresult(data,m,v,3)
    output(d,outputpath)