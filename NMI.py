# coding=utf-8
import numpy as np
import pandas as pd
import math
def NMI(A,B):
    # len(A) should be equal to len(B)
    total = len(A)
    A_ids = set(A)
    B_ids = set(B)
    #Mutual information
    MI = 0
    eps = 1.4e-45
    for idA in A_ids:
        for idB in B_ids:
            idAOccur = np.where(A==idA)
            idBOccur = np.where(B==idB)
            idABOccur = np.intersect1d(idAOccur,idBOccur)
            px = 1.0*len(idAOccur[0])/total
            py = 1.0*len(idBOccur[0])/total
            pxy = 1.0*len(idABOccur)/total
            MI = MI + pxy*math.log(pxy/(px*py)+eps,2)
    # Normalized Mutual information
    Hx = 0
    for idA in A_ids:
        idAOccurCount = 1.0*len(np.where(A==idA)[0])
        Hx = Hx - (idAOccurCount/total)*math.log(idAOccurCount/total+eps,2)
    Hy = 0
    for idB in B_ids:
        idBOccurCount = 1.0*len(np.where(B==idB)[0])
        Hy = Hy - (idBOccurCount/total)*math.log(idBOccurCount/total+eps,2)
    MIhat = 2.0*MI/(Hx+Hy)
    return MIhat

if __name__ == '__main__':

    A = pd.read_csv('GMM_label.csv')
    A = np.array(A["GMM_labels"])
    B = pd.read_csv('label.csv')
    B = np.array(B["labels"])
    C = pd.read_excel('chap_7_数据集.xlsx',header=None,names=['x','y','label'])
    C = np.array(C['label'])
    GMM_NMI = NMI(C,A)
    KMeans_NMI = NMI(C,B)
    print("GMM_NMI")
    print(GMM_NMI)
    print("KMeans_NMI")
    print(KMeans_NMI)
