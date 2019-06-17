import numpy as np
from time import time
Tstart_time = time()

from lib import Searching_Methods
from lib import Evaluating

def Searching_by_Distance(MaxFC, NumSubCluster, Ints_clustMem, idxCount, Top1, FirstStage_Codebook, SecondStage_Codebook, START, STEP ):
    
    print('\n====> Distance-based ranking <====')
    
    query_sift = np.loadtxt('./input/Test_Data/Test.txt', dtype=float)    
    Method = 2
    testcases = []     
    CTop1_Recall  = np.zeros((MaxFC, Method), dtype=float)    

    cluster_idx = 0 
    
    for FC in range (START,MaxFC+1,STEP):
        
        Top1Recall  = np.zeros((MaxFC, Method), dtype=float)
                        
        r = 0
        for SC in range (NumSubCluster,NumSubCluster+1,1):
    
            print('\n--> Dist_QCD <---')
            usingProb = 0
            fn = './output_Dist/Candidate_Lists/QNN_M0_%d_%d.txt' %(FC,SC)
            fn2 = './output_candidate_lists_100k/M0_%d_%d.txt' %(FC,SC)
            fn3 = './output_candidate_count/M0_%d_%d.txt' %(FC,SC)

            InsIdx2 = Searching_Methods.createIndex_using_x_idx(query_sift, FirstStage_Codebook, SecondStage_Codebook, fn, usingProb, testcases, FC, SC)     
            Top1Recall[r,0] = Evaluating.Searching_Top1(Ints_clustMem, Top1, idxCount, InsIdx2, FC, SC, fn2, fn3)

            print('\n--> Dist_SLE <---')
            fn = './output_Dist/Candidate_Lists/QNN_M1_%d_%d.txt' %(FC,SC)
            fn2 = './output_candidate_lists_100k/M1_%d_%d.txt' %(FC,SC)
            fn3 = './output_candidate_count/M1_%d_%d.txt' %(FC,SC)

            InsIdx2 = Searching_Methods.createIndex_using_xplus_idx(query_sift, FirstStage_Codebook, SecondStage_Codebook, fn, testcases, FC, SC )             
            Top1Recall[r,1] = Evaluating.Searching_Top1(Ints_clustMem, Top1, idxCount, InsIdx2, FC, SC, fn2, fn3 )          
          
            r +=1
        
        fN_Top1 = './output_Dist/Recall/Recall_Top1_FC%d_x_idx.txt' %(FC)
        np.savetxt(fN_Top1, Top1Recall, delimiter=' ',fmt='%1.5f')
        CTop1_Recall[cluster_idx]  = Top1Recall[0]


        cluster_idx += 1
    
    fN_CTop1 = './output_Dist/Recall/CRecall_Top1_%d_Clusters_x_idx.txt' %(MaxFC)
    np.savetxt(fN_CTop1, CTop1_Recall, delimiter=' ',fmt='%1.5f')


#################
end_time = time()
time_taken = end_time - Tstart_time # time_taken is in seconds
hours, rest = divmod(time_taken,3600)
minutes, seconds = divmod(rest, 60)
print("This took %d hours %d minutes %d seconds" %(hours,minutes,seconds)) 
