import numpy as np
from time import time
Tstart_time = time()

def Searching_Top1(Ints_clustMem, Top1, idxCount, InsIdx2, FC, SC, fn2, fn3):
          
    recall = np.zeros((Top1.shape[0]), dtype=int)
    len_lcl = np.zeros((Top1.shape[0]), dtype=int)
    Idx = np.zeros((InsIdx2.shape[0], InsIdx2.shape[1]), dtype=int)
    Idx -= 1
       
    print("\nComputing Top 1 recall")
    for i in range(InsIdx2.shape[0]): 
        for j in range(InsIdx2.shape[1]) :
            Idx[i,j] = InsIdx2[i,j]
            if idxCount[InsIdx2[i,j]] > 0 and InsIdx2[i,j] != -1 :      
                ints_member = Ints_clustMem[InsIdx2[i,j]]
                len_lcl[i] += idxCount[InsIdx2[i,j]]
                found = len(np.intersect1d(Top1[i], ints_member))
                recall[i] += found 
                if len_lcl[i] > 100000: 
                    break
                
    TotalRecall =np.sum(recall) 
    AverageRecall = TotalRecall/Top1.shape[0]
    MeanCandidate = np.mean(len_lcl)
    print('================================')
    print('No. First Stage codebook : ', FC)
    print('No. Second Stage codebook : ', SC)
    print('Total Recall ', TotalRecall)
    print('Top1 Recall ', AverageRecall) 
    print('Mean Candidate ', int(MeanCandidate))
    print('================================')
    
    np.savetxt(fn2, Idx, delimiter=' ',fmt='%d')
    np.savetxt(fn3,len_lcl, delimiter=' ',fmt='%d')    
    return AverageRecall
