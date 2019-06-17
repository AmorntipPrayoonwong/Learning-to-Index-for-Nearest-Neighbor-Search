import numpy as np
from keras.models import load_model

from lib import Searching_Methods
from lib import Evaluating    


def Searching(MaxFC, NumSubCluster, Zscore_var, Ints_clustMem, idxCount, Top1, FirstStage_Codebook, SecondStage_Codebook, START, STEP ):
    
    print('\n====> Probability-based ranking: Hierarchical Learning <====')
    
    model = load_model('My_Residual_Model.h5')
    
    query_sift = np.loadtxt('./input/Test_Data/Test.txt', dtype=float)
    npzfile = np.load('./output/(te)sort_bid_RawData_weightingProb.npz')
    
    Method = 3
    testcases = [] #[0,1,2,3]
    
    CTop1_Recall  = np.zeros((MaxFC, Method), dtype=float)
    
    cluster_idx = 0 
    
    for FC in range (START,MaxFC+1,STEP):
        
        Top1Recall  = np.zeros((MaxFC, Method), dtype=float)
        
        r = 0

        for SC in range (NumSubCluster,NumSubCluster+1,1):
            
            print('\n--> HProb_Raw+γstd <---')
            learning_idx = np.array(npzfile['p_idx'])
            p_prob = np.array(npzfile['p_prob'])
            fn = './output_TwoStage_Learning/Candidate_Lists/QNN_M6_%d_%d.txt' %(FC,SC)
            fn2 = './output_candidate_lists_100k/M6_%d_%d.txt' %(FC,SC)
            fn3 = './output_candidate_count/M6_%d_%d.txt' %(FC,SC)

            if FC == 1 :
                Z = FC * SC
            else:
                Z = Zscore_var            

            InsIdx2 = Searching_Methods.createIndex_using_TwoStageZscore_p_idx(query_sift, FirstStage_Codebook, SecondStage_Codebook, learning_idx, fn, p_prob, Z, testcases, FC, SC, model) 
    
            Top1Recall[r,0] = Evaluating.Searching_Top1(Ints_clustMem, Top1, idxCount, InsIdx2, FC, SC, fn2, fn3)
    
            print('\n--> HProb_Raw+γexp <---')
            learning_idx = np.array(npzfile['p_idx'])
            p_prob = np.array(npzfile['p_prob'])
            p_prob = 1 - p_prob 
    
            prob_SubC_Retrieve = np.exp(-p_prob/np.mean(p_prob))
            usingProb = 1
            fn = './output_TwoStage_Learning/Candidate_Lists/QNN_M7_%d_%d.txt' %(FC,SC)
            fn2 = './output_candidate_lists_100k/M7_%d_%d.txt' %(FC,SC)
            fn3 = './output_candidate_count/M7_%d_%d.txt' %(FC,SC)

            InsIdx2 = Searching_Methods.createIndex_using_TwoStageLearningProb(query_sift, FirstStage_Codebook, SecondStage_Codebook, learning_idx, fn, prob_SubC_Retrieve, usingProb, testcases, FC, SC, model) 
    
            Top1Recall[r,1] = Evaluating.Searching_Top1(Ints_clustMem, Top1, idxCount, InsIdx2, FC, SC, fn2, fn3)
    

            print('\n--> HProb_Raw+γSum <--- ==========')
            learning_idx = np.array(npzfile['p_idx'])
            prob_SubC_Retrieve = np.array(npzfile['p_prob'])
            usingProb = 2
            fn = './output_TwoStage_Learning/Candidate_Lists/QNN_M8_%d_%d.txt' %(FC,SC)
            fn2 = './output_candidate_lists_100k/M8_%d_%d.txt' %(FC,SC)
            fn3 = './output_candidate_count/M8_%d_%d.txt' %(FC,SC)

            InsIdx2 = Searching_Methods.createIndex_using_TwoStageLearningProb(query_sift, FirstStage_Codebook, SecondStage_Codebook, learning_idx, fn, prob_SubC_Retrieve, usingProb, testcases, FC, SC, model) 
    
            Top1Recall[r,2] = Evaluating.Searching_Top1(Ints_clustMem, Top1, idxCount, InsIdx2, FC, SC, fn2, fn3)
          
          
            r +=1
        fN_Top1 = './output_TwoStage_Learning/Recall/Recall_Top1_FC%d.txt' %(FC)
        np.savetxt(fN_Top1, Top1Recall, delimiter=' ',fmt='%1.4f')
        CTop1_Recall[cluster_idx]  = Top1Recall[0]

        cluster_idx += 1
    
    fN_CTop1 = './output_TwoStage_Learning/Recall/CRecall_Top1_%d_Clusters.txt' %(MaxFC)
    np.savetxt(fN_CTop1, CTop1_Recall, delimiter=' ',fmt='%1.4f')
    
