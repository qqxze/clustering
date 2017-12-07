import csv
import os
import numpy as np
import pandas
from numpy import *
import pandas as pd

#导入数据
def loadDateSet(fileName):
    heade = ["x","y","label"]
    dateSet = pd.read_excel(fileName,header=None,names=heade)
    dateSet = dateSet.iloc[:,[0,1]]
    return dateSet

# 计算向量的欧氏距离
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

def KMeans(dateSet,k,distMeas = distEclud,createCent=randCent):
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
    return centroids,clusterAssment

def biKmeans(dataSet, k, distMeas=distEclud):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))
    centroid0 = mean(dataSet, axis=0).tolist()[0]#axis=0 按列
    centList =[centroid0] #create a list with one centroid
    for j in range(m):#calc initial Error
        clusterAssment[j,1] = distMeas(mat(centroid0), dataSet[j,:])**2
    while (len(centList) < k):
        lowestSSE = inf
        for i in range(len(centList)):
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A==i)[0],:]#get the data points currently in cluster i
            centroidMat, splitClustAss = KMeans(ptsInCurrCluster, 2, distMeas)
            sseSplit = sum(splitClustAss[:,1])#compare the SSE to the currrent minimum
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1])
            #print ("sseSplit, and notSplit: ",sseSplit,sseNotSplit)
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList) #change 1 to 3,4, or whatever
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
        #print ('the bestCentToSplit is: ',bestCentToSplit)
        #print ('the len of bestClustAss is: ', len(bestClustAss))
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]#replace a centroid with two best centroids
        centList.append(bestNewCents[1,:].tolist()[0])
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]= bestClustAss#reassign new clusters, and SSE
    datalabel = np.insert(np.array(dataMat),2,values=np.array(clusterAssment)[:,0],axis=1)
    return mat(centList), clusterAssment,datalabel


import matplotlib.pyplot as plt
def clusterClubs(datMat,myCentroids, clustAssing,numClust=3):
    
    fig = plt.figure("bikmeans clustering")
    rect=[0.1,0.1,0.8,0.8]
    scatterMarkers=['s', 'o', '^']
    axprops = dict(xticks=[], yticks=[])
    ax0=fig.add_axes(rect, label='ax0', **axprops)
    ax1=fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numClust):
        ptsInCurrCluster = datMat[nonzero(clustAssing[:,0].A==i)[0],:]#返回属于i标签的值
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:,0].flatten().A[0], ptsInCurrCluster[:,1].flatten().A[0], marker=markerStyle, s=90)
    ax1.scatter(myCentroids[:,0].flatten().A[0], myCentroids[:,1].flatten().A[0], marker='+', s=300)
    plt.savefig("Kmeansout.png")
    plt.show()




def output(data, outputpath):

    with open(outputpath, "w", newline='') as f:
        headers = ["x","y","KM_labels"]
        writer = csv.writer(f)
        writer.writerow(headers)
        for d in data:
            writer.writerow([d[0],d[1],d[2]+1])
    f.close()

if __name__=='__main__':
    fileName = "chap_7_数据集.xlsx"
    outputpath = "KMALL_label.csv"
    dataMat = mat(loadDateSet(fileName))
    centList,myNewAssing,d = biKmeans(dataMat,3)
    print("new质心集合")
    print(centList)
    output(d,outputpath)
    clusterClubs(dataMat,centList,myNewAssing)

