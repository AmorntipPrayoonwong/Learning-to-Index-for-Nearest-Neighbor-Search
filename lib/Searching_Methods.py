import numpy as np
from scipy.spatial import distance
from scipy import stats
from time import time
from copy import deepcopy

Tstart_time = time()

def cal_Time(start_time):
    end_time = time()
    time_taken = end_time - start_time 
    hours, rest = divmod(time_taken,3600)
    minutes, seconds = divmod(rest, 60)
    print("This took %d hours %d minutes %d seconds" %(hours,minutes,seconds)) 

def createIndex_using_x_idx(query_sift, FirstStage_Codebook, SecondStage_Codebook, fname,  usingProb, testcases, FC, SC):
    start_time = time()
    FcandidateSize = FC 
    ScandidateSize = SC  
    DScandidateSize = np.zeros((query_sift.shape[0], FC), dtype=int) 
    BucketSize = FcandidateSize *  ScandidateSize  
    
    # initial vars
    dist = np.zeros((query_sift.shape[0], FirstStage_Codebook.shape[0]), dtype=float)
    CNN = np.zeros((query_sift.shape[0], FcandidateSize), dtype=int)
    FCandidate = np.zeros((FcandidateSize, FirstStage_Codebook.shape[1]), dtype=float)

    Rdist = np.zeros((FcandidateSize, SecondStage_Codebook.shape[0]), dtype=float)
    Sorted_dist = np.zeros((query_sift.shape[0], FcandidateSize), dtype=float)
    InsIdx2 = np.zeros((query_sift.shape[0], FcandidateSize *  SC), dtype=int)
    InsIdx2 -= 1

    for i in range (query_sift.shape[0]):
        dist[i, :] = np.sqrt(np.sum(np.square(np.dot(np.ones((FirstStage_Codebook.shape[0], 1)), [query_sift[i, :]]) - FirstStage_Codebook), axis=-1))
        idx = np.argsort(dist[i, :]) 
        CNN[i,0:FcandidateSize] = idx[0:FcandidateSize]
        Sorted_dist[i,0:FcandidateSize] = dist[i,CNN[i,0:FcandidateSize]]
        DScandidateSize[i,0:FcandidateSize] = SC
                    
        DRCNN = np.zeros((FcandidateSize,FcandidateSize * ScandidateSize ), dtype=int) 
        DRCNN -= 1 
        DSorted_Rdist = np.zeros((FcandidateSize,FcandidateSize * ScandidateSize), dtype=float)
        DSorted_Rdist -=1
    
        for j in range(FcandidateSize):

           FCandidate[j] = query_sift[i] - FirstStage_Codebook[CNN[i,j]]
           Rdist[j, :] = np.sqrt(np.sum(np.square(np.dot(np.ones((SecondStage_Codebook.shape[0], 1)), [FCandidate[j, : ]]) - SecondStage_Codebook), axis=-1))
           idx2 = np.argsort(Rdist[j, :])
           if DScandidateSize[i,j] > len(idx2):
               DataSize = len(idx2)
           else:
               DataSize = DScandidateSize[i,j]
                         
           DRCNN[j,0:DataSize] = idx2[0:DataSize]
           DSorted_Rdist[j,0:DataSize] = Rdist[j,DRCNN[j,0:DataSize]]
           
        temp_candidate = []
        temp_candidateDist = []
        candidatelist = []
        candidatedist = []

        for j in range(FcandidateSize):
            for k in range(DScandidateSize[i,j]):
                temp_candidate = (CNN[i,j]*SecondStage_Codebook.shape[0])+(DRCNN[j,k]) 
                temp_candidateDist = Sorted_dist[i,j]+DSorted_Rdist[j,k]
                candidatelist.extend([temp_candidate])
                candidatedist.extend([temp_candidateDist])
                
        CandidateList = np.array(candidatelist)
        ReorderIdx = np.argsort(candidatedist)
        ReorderIdx2 = ReorderIdx[0:BucketSize]
        IdxSize = len(ReorderIdx2)
        InsIdx2[i,0:IdxSize] =  CandidateList[ReorderIdx2]

    cal_Time(start_time)

    print('\nEnd search then write output to file ')
    print(fname)
    np.savetxt(fname, InsIdx2, delimiter=' ',fmt='%d')

        
    return InsIdx2

def createIndex_using_xplus_idx(query_sift, FirstStage_Codebook, SecondStage_Codebook, fname, testcases, FC, SC):
    start_time = time()
    FcandidateSize = FC 
    ScandidateSize = SecondStage_Codebook.shape[0] 
    DScandidateSize = np.zeros((query_sift.shape[0], FC), dtype=int) 
    BucketSize = FcandidateSize *  SC  
   
    # initial vars
    dist = np.zeros((query_sift.shape[0], FirstStage_Codebook.shape[0]), dtype=float)
    CNN = np.zeros((query_sift.shape[0], FcandidateSize), dtype=int)
    FCandidate = np.zeros((FcandidateSize, FirstStage_Codebook.shape[1]), dtype=float)

    Rdist = np.zeros((FcandidateSize, SecondStage_Codebook.shape[0]), dtype=float)
    Sorted_dist = np.zeros((query_sift.shape[0], FcandidateSize), dtype=float)
    InsIdx2 = np.zeros((query_sift.shape[0], FcandidateSize *  SC), dtype=int)
    InsIdx2 -= 1

    for i in range (query_sift.shape[0]):
        dist[i, :] = np.sqrt(np.sum(np.square(np.dot(np.ones((FirstStage_Codebook.shape[0], 1)), [query_sift[i, :]]) - FirstStage_Codebook), axis=-1))
        idx = np.argsort(dist[i, :]) 
        CNN[i,0:FcandidateSize] = idx[0:FcandidateSize]
        Sorted_dist[i,0:FcandidateSize] = dist[i,CNN[i,0:FcandidateSize]]
        DScandidateSize[i,0:FcandidateSize] = ScandidateSize
                    
        DRCNN = np.zeros((FcandidateSize,FcandidateSize * ScandidateSize ), dtype=int) 
        DRCNN -= 1 
        DSorted_Rdist = np.zeros((FcandidateSize,FcandidateSize * ScandidateSize), dtype=float)
        DSorted_Rdist -=1
    
        for j in range(FcandidateSize):

           FCandidate[j] = query_sift[i] - FirstStage_Codebook[CNN[i,j]]
           Rdist[j, :] = np.sqrt(np.sum(np.square(np.dot(np.ones((SecondStage_Codebook.shape[0], 1)), [FCandidate[j, : ]]) - SecondStage_Codebook), axis=-1))
           idx2 = np.argsort(Rdist[j, :])
           DataSize = DScandidateSize[i,j]
           DRCNN[j,0:DataSize] = idx2[0:DataSize]
           DSorted_Rdist[j,0:DataSize] = Rdist[j,DRCNN[j,0:DataSize]]
           
        temp_candidate = []
        temp_candidateDist = []
        candidatelist = []
        candidatedist = []

        for j in range(FcandidateSize):
            for k in range(DScandidateSize[i,j]):
                temp_candidate = (CNN[i,j]*SecondStage_Codebook.shape[0])+(DRCNN[j,k]) 
                temp_candidateDist = Sorted_dist[i,j]+DSorted_Rdist[j,k]
                candidatelist.extend([temp_candidate])
                candidatedist.extend([temp_candidateDist])
                
        CandidateList = np.array(candidatelist)
        ReorderIdx = np.argsort(candidatedist)
        ReorderIdx2 = ReorderIdx[0:BucketSize]
        IdxSize = len(ReorderIdx2)
        InsIdx2[i,0:IdxSize] =  CandidateList[ReorderIdx2]

    cal_Time(start_time)

    np.savetxt(fname, InsIdx2, delimiter=' ',fmt='%d')

        
    return InsIdx2

def createIndex_using_LearningProb(query_sift, FirstStage_Codebook, SecondStage_Codebook, learning_idx, fname ,prob_SubC_Retrieve, usingProb, testcases, FC, SC):
    start_time = time()
    FcandidateSize = FC 
    ScandidateSize = SC 
    DScandidateSize = np.zeros((query_sift.shape[0], FC), dtype=int) 
    BucketSize = FcandidateSize *  ScandidateSize  
    
    CNN = np.zeros((query_sift.shape[0], FcandidateSize), dtype=int)
    FCandidate = np.zeros((FcandidateSize, FirstStage_Codebook.shape[1]), dtype=float)

    Rdist = np.zeros((FcandidateSize, SecondStage_Codebook.shape[0]), dtype=float)
    InsIdx2 = np.zeros((query_sift.shape[0], FcandidateSize *  SC), dtype=int)
    InsIdx2 -= 1

    for i in range (query_sift.shape[0]):
        idx = learning_idx[i]
        CNN[i,0:FcandidateSize] = idx[0:FcandidateSize]
        prob = prob_SubC_Retrieve[i, CNN[i]] 
        sp = sum(prob)
                       
        if usingProb > 0:
            if FC == 1:
                DScandidateSize[i,0:FcandidateSize] = SC
            else:                                                             
                BZ = BucketSize
                for j in range(FcandidateSize):
                    if prob_SubC_Retrieve[i,CNN[i,j]] > 0:
                        if usingProb == 1:
                            DScandidateSize[i,j] = (prob_SubC_Retrieve[i,CNN[i,j]] * BZ) 
                        if usingProb == 2:
                            DScandidateSize[i,j] = ((prob_SubC_Retrieve[i,CNN[i,j]] /sp) * BZ)  
                        
                    if BZ < 0:
                        BZ = 0               
                    BZ -= DScandidateSize[i,j]                                                                                   
        else:
            DScandidateSize[i,0:FcandidateSize] = SC
                    
        DRCNN = np.zeros((FcandidateSize,FcandidateSize * ScandidateSize ), dtype=int) 
        DRCNN -= 1 
        DSorted_Rdist = np.zeros((FcandidateSize,FcandidateSize * ScandidateSize), dtype=float)
        DSorted_Rdist -=1
    
        for j in range(FcandidateSize):

           FCandidate[j] = query_sift[i] - FirstStage_Codebook[CNN[i,j]]
           Rdist[j, :] = np.sqrt(np.sum(np.square(np.dot(np.ones((SecondStage_Codebook.shape[0], 1)), [FCandidate[j, : ]]) - SecondStage_Codebook), axis=-1))
           idx2 = np.argsort(Rdist[j, :])
           if DScandidateSize[i,j] > len(idx2):
               DataSize = len(idx2)
           else:
               DataSize = DScandidateSize[i,j]
           if DataSize > DRCNN.shape[1]:
               DataSize = DRCNN.shape[1]
                                     
           DRCNN[j,0:DataSize] = idx2[0:DataSize]
           DSorted_Rdist[j,0:DataSize] = Rdist[j,DRCNN[j,0:DataSize]]
           
        temp_candidate = []
        candidate = []
        Alternate = int(np.ceil(SC * 0.3))
        if FcandidateSize > 1:
            for r in range (Alternate):
                temp_candidate = (CNN[i,0]*SecondStage_Codebook.shape[0])+(DRCNN[0,r])
                candidate.extend([temp_candidate])
                temp_candidate = (CNN[i,1]*SecondStage_Codebook.shape[0])+(DRCNN[1,r])
                candidate.extend([temp_candidate])
            
            for j in range(FcandidateSize):
                if j < 2:
                    start = Alternate
                else:
                    start = 0
                    
                for k in range(start, DScandidateSize[i,j]):
                    temp_candidate = (CNN[i,j]*SecondStage_Codebook.shape[0])+(DRCNN[j,k]) 
                    candidate.extend([temp_candidate])
        else:
                for k in range(DScandidateSize[i,j]):
                    temp_candidate = (CNN[i,j]*SecondStage_Codebook.shape[0])+(DRCNN[j,k]) 
                    candidate.extend([temp_candidate])
            
            
#        for j in range(FcandidateSize):
#            for k in range(DScandidateSize[i,j]):
#                temp_candidate = (CNN[i,j]*SecondStage_Codebook.shape[0])+(DRCNN[j,k]) 
#                candidate.extend([temp_candidate])

        CandidateList = np.array(candidate)
        IdxSize = len(CandidateList)
        if IdxSize > BucketSize:
            IdxSize = BucketSize
        InsIdx2[i,0:IdxSize] =  CandidateList[0:IdxSize]
        
    cal_Time(start_time)
    np.savetxt(fname, InsIdx2, delimiter=' ',fmt='%d')        
    return InsIdx2

def createIndex_using_Zscore_p_idx(query_sift, FirstStage_Codebook, SecondStage_Codebook, learning_idx, fname, p_prob, Zscore_var, testcases, FC, SC):
    start_time = time()
    FcandidateSize = FC 
    ScandidateSize = SC 
    DScandidateSize = np.zeros((query_sift.shape[0], FC), dtype=int) 
    BucketSize = FcandidateSize *  ScandidateSize  
    
    CNN = np.zeros((query_sift.shape[0], FcandidateSize), dtype=int)
    FCandidate = np.zeros((FcandidateSize, FirstStage_Codebook.shape[1]), dtype=float)

    Rdist = np.zeros((FcandidateSize, SecondStage_Codebook.shape[0]), dtype=float)
    Sorted_dist = np.zeros((query_sift.shape[0], FcandidateSize), dtype=float)
    InsIdx2 = np.zeros((query_sift.shape[0], FcandidateSize *  SC), dtype=int)
    InsIdx2 -= 1

    for i in range (query_sift.shape[0]):
        idx = learning_idx[i]
        CNN[i,0:FcandidateSize] = idx[0:FcandidateSize]

        for candidate in range (FcandidateSize):    
            Sorted_dist[i, candidate] = distance.euclidean(query_sift[i],FirstStage_Codebook[CNN[i,candidate]]) 

#===========Test Outlier             
        SortedDist = p_prob[i,idx[0:Zscore_var]]
        ZScoreSortedList = stats.zscore(SortedDist)
        
        invZScore = ZScoreSortedList 

        sumZScore = sum(x for x in invZScore if x > 0) 
        probRL1 = invZScore[invZScore>0]
        probRL2 = probRL1/sumZScore  #Probability of how many data we want to retrieve from this cluster
        MaxProbRL2 = np.argmax(probRL2)
        
        len_probRL2 = len(probRL2)
        excess = 0
        for y in range(len_probRL2):
            if y < FC:
                DScandidateSize[i,y] = int(probRL2[y]*(FC*SC)) + excess 
                if DScandidateSize[i,y] > SecondStage_Codebook.shape[0]:
                    excess = DScandidateSize[i,y] -  (SecondStage_Codebook.shape[0]*(1-(y/10)))
                    DScandidateSize[i,y] = SecondStage_Codebook.shape[0] * (1-(y/10))
                else:
                    excess = 0
                    
            else :
                if MaxProbRL2 < FcandidateSize:
                    DScandidateSize[i,MaxProbRL2] += int(probRL2[y]*(FC*SC))
                else:
                    DScandidateSize[i,0] += int(probRL2[y]*(FC*SC)) 
                                                   
        if i in testcases:
            print("++++++++++++++++++++++++++++++++++++++++")
            print(i)
            print("First Stage cluster------------")
            print(CNN[i])
            print(DScandidateSize[i])
            
        DRCNN = np.zeros((FcandidateSize,FcandidateSize * ScandidateSize ), dtype=int) 
        DRCNN -= 1 
        DSorted_Rdist = np.zeros((FcandidateSize,FcandidateSize * ScandidateSize), dtype=float)
        DSorted_Rdist -=1
        if len_probRL2 < FcandidateSize:
            RANGE = len_probRL2
            
        else:
            RANGE = FcandidateSize
            
           
            
        for j in range(RANGE):

           FCandidate[j] = query_sift[i] - FirstStage_Codebook[CNN[i,j]]
           Rdist[j, :] = np.sqrt(np.sum(np.square(np.dot(np.ones((SecondStage_Codebook.shape[0], 1)), [FCandidate[j, : ]]) - SecondStage_Codebook), axis=-1))
           idx2 = np.argsort(Rdist[j, :])
           if DScandidateSize[i,j] > len(idx2):
               DataSize = len(idx2)
           else:
               DataSize = DScandidateSize[i,j]
           DRCNN[j,0:DataSize] = idx2[0:DataSize]
           DSorted_Rdist[j,0:DataSize] = Rdist[j,DRCNN[j,0:DataSize]]



        temp_candidate = []   
        candidate = [] 
        
        Alternate = int(np.ceil(SC * 0.3))
        if FcandidateSize > 1:
            for r in range (Alternate):
                temp_candidate = (CNN[i,0]*SecondStage_Codebook.shape[0])+(DRCNN[0,r])
                candidate.extend([temp_candidate])
                temp_candidate = (CNN[i,1]*SecondStage_Codebook.shape[0])+(DRCNN[1,r])
                candidate.extend([temp_candidate])
            
            for j in range(RANGE):
                if j < 2:
                    start = Alternate
                else:
                    start = 0
                    
                for k in range(start, DScandidateSize[i,j]):
                    temp_candidate = (CNN[i,j]*SecondStage_Codebook.shape[0])+(DRCNN[j,k]) 
                    candidate.extend([temp_candidate])
        else:
                for k in range(DScandidateSize[i,j]):
                    temp_candidate = (CNN[i,j]*SecondStage_Codebook.shape[0])+(DRCNN[j,k]) 
                    candidate.extend([temp_candidate])
        
        
#        
#        for j in range(RANGE):
#            for k in range(DScandidateSize[i,j]):
#                temp_candidate = (CNN[i,j]*SecondStage_Codebook.shape[0])+(DRCNN[j,k])                 
#                candidate.extend([temp_candidate])

        CandidateList = np.array(candidate)
        IdxSize = len(CandidateList)
        if IdxSize > BucketSize:
            IdxSize = BucketSize
        InsIdx2[i,0:IdxSize] =  CandidateList[0:IdxSize]



    cal_Time(start_time)

    np.savetxt(fname, InsIdx2, delimiter=' ',fmt='%d')

    
    return InsIdx2


########### =============================== Two Stage Learning ===================================================#############################################

def createIndex_using_TwoStageLearningProb(query_sift, FirstStage_Codebook, SecondStage_Codebook, learning_idx, fname ,prob_SubC_Retrieve, usingProb, testcases, FC, SC, model):
    FNormalize = np.loadtxt('./input/Train_Data/Traininng_Set/Train_Residual_CR.txt', dtype=float)
    NNormalize = np.max(FNormalize)        
    
    start_time = time()
    FcandidateSize = FC 
    ScandidateSize = SC 
    DScandidateSize = np.zeros((query_sift.shape[0], FC), dtype=int) 
    BucketSize = FcandidateSize *  ScandidateSize  
    
    CNN = np.zeros((query_sift.shape[0], FcandidateSize), dtype=int)
    FCandidate = np.zeros((FcandidateSize, FirstStage_Codebook.shape[1]), dtype=float)

    InsIdx2 = np.zeros((query_sift.shape[0], FcandidateSize *  ScandidateSize), dtype=int)
    InsIdx2 -= 1

    for i in range (query_sift.shape[0]):
        idx = learning_idx[i]
        CNN[i,0:FcandidateSize] = idx[0:FcandidateSize]
        prob = prob_SubC_Retrieve[i, CNN[i]]        
        sp = sum(prob)
        
                
        if usingProb > 0: 
            if FC == 1:
                DScandidateSize[i,0:FcandidateSize] = SC
            else:                                                             
                BZ = BucketSize
                for j in range(FcandidateSize):
                    if prob_SubC_Retrieve[i,CNN[i,j]] > 0:
                        if usingProb == 1:
                            DScandidateSize[i,j] = (prob_SubC_Retrieve[i,CNN[i,j]] * BZ) 
                        if usingProb == 2:
                            DScandidateSize[i,j] = ((prob_SubC_Retrieve[i,CNN[i,j]] /sp) * BZ) 

                        if BZ < 0:
                            BZ = 0               
                    BZ -= DScandidateSize[i,j]                                                                                   
        else:
            DScandidateSize[i,0:FcandidateSize] = SC
                    
        DRCNN = np.zeros((FcandidateSize,FcandidateSize * ScandidateSize ), dtype=int) 
        DRCNN -= 1 
        DSorted_Rdist = np.zeros((FcandidateSize,FcandidateSize * ScandidateSize), dtype=float)
        DSorted_Rdist -=1
        
        x_test = np.zeros((FcandidateSize,SecondStage_Codebook.shape[1]*2), dtype=float)
        
        for j in range(FcandidateSize):

           FCandidate[j] = query_sift[i] - FirstStage_Codebook[CNN[i,j]]

           x_test[j,0:SecondStage_Codebook.shape[1]] = FirstStage_Codebook[CNN[i,j]]
           x_test[j,SecondStage_Codebook.shape[1]:SecondStage_Codebook.shape[1]*2] = FCandidate[j]
           DataSize = DScandidateSize[i,j]

        x_test /= NNormalize

        p_test = model.predict(x_test)   
        
        pTest = deepcopy(p_test)        
        p_idx = np.zeros((p_test.shape[0], BucketSize), dtype=int) 
        p_idx -=1
        for q in range (FC): 
            c = 0
            for t in range (DScandidateSize[i,q]):
                p_idx[q,c] = np.argmax(pTest[q])
                pTest[q,p_idx[q,c]] = -1        
                c += 1
                
        for j in range(FcandidateSize):
            DataSize = DScandidateSize[i,j]
            DRCNN[j, 0:DataSize] = p_idx[j,0:DataSize]

        temp_candidate = []
        candidate = []
        
        Alternate = int(np.ceil(SC * 0.3)) 
        if FcandidateSize > 1:
            for r in range (Alternate):
                temp_candidate = (CNN[i,0]*SecondStage_Codebook.shape[0])+(DRCNN[0,r])
                candidate.extend([temp_candidate])
                temp_candidate = (CNN[i,1]*SecondStage_Codebook.shape[0])+(DRCNN[1,r])
                candidate.extend([temp_candidate])
            
            for j in range(FcandidateSize):
                if j < 2:
                    start = Alternate
                else:
                    start = 0
                    
                for k in range(start, DScandidateSize[i,j]):
                    temp_candidate = (CNN[i,j]*SecondStage_Codebook.shape[0])+(DRCNN[j,k]) 
                    candidate.extend([temp_candidate])
        else:
                for k in range(DScandidateSize[i,j]):
                    temp_candidate = (CNN[i,j]*SecondStage_Codebook.shape[0])+(DRCNN[j,k]) 
                    candidate.extend([temp_candidate])

        
#        for j in range(FcandidateSize):
#            for k in range(DScandidateSize[i,j]):
#                temp_candidate = (CNN[i,j]*SecondStage_Codebook.shape[0])+(DRCNN[j,k]) 
#                candidate.extend([temp_candidate])

        CandidateList = np.array(candidate)
        IdxSize = len(CandidateList)
        if IdxSize > BucketSize:
            IdxSize = BucketSize
        InsIdx2[i,0:IdxSize] =  CandidateList[0:IdxSize]

    cal_Time(start_time)

    np.savetxt(fname, InsIdx2, delimiter=' ',fmt='%d')
        
    return InsIdx2

def createIndex_using_TwoStageZscore_p_idx(query_sift, FirstStage_Codebook, SecondStage_Codebook, learning_idx, fname, p_prob, Zscore_var, testcases, FC, SC, model):
    FNormalize = np.loadtxt('./input/Train_Data/Traininng_Set/Train_Residual_CR.txt', dtype=float)
    NNormalize = np.max(FNormalize)        

    start_time = time()
    FcandidateSize = FC 
    ScandidateSize = SC 
    DScandidateSize = np.zeros((query_sift.shape[0], FC), dtype=int) 
    BucketSize = FcandidateSize *  ScandidateSize  
    
    CNN = np.zeros((query_sift.shape[0], FcandidateSize), dtype=int)
    FCandidate = np.zeros((FcandidateSize, FirstStage_Codebook.shape[1]), dtype=float)

    InsIdx2 = np.zeros((query_sift.shape[0], FcandidateSize *  ScandidateSize), dtype=int)
    InsIdx2 -= 1

    for i in range (query_sift.shape[0]):
        idx = learning_idx[i]
        CNN[i,0:FcandidateSize] = idx[0:FcandidateSize]
        
#===========Test Outlier             
        SortedDist = p_prob[i,idx[0:Zscore_var]]
        ZScoreSortedList = stats.zscore(SortedDist)
        
        invZScore = ZScoreSortedList 

        sumZScore = sum(x for x in invZScore if x > 0) 
        probRL1 = invZScore[invZScore>0]
        probRL2 = probRL1/sumZScore  #Probability of how many data we want to retrieve from this cluster
        MaxProbRL2 = np.argmax(probRL2)
        
        len_probRL2 = len(probRL2)
        excess = 0
        for y in range(len_probRL2):
            if y < FC:
                DScandidateSize[i,y] = int(probRL2[y]*(FC*SC)) + excess 
                if DScandidateSize[i,y] > SecondStage_Codebook.shape[0]:
                    excess = DScandidateSize[i,y] -  (SecondStage_Codebook.shape[0]*(1-(y/10)))
                    DScandidateSize[i,y] = SecondStage_Codebook.shape[0] * (1-(y/10))
                else:
                    excess = 0
                    
            else :
                if MaxProbRL2 < FcandidateSize:
                    DScandidateSize[i,MaxProbRL2] += int(probRL2[y]*(FC*SC))
                else:
                    DScandidateSize[i,0] += int(probRL2[y]*(FC*SC)) 
                                                   
        if i in testcases:
            print("++++++++++++++++++++++++++++++++++++++++")
            print(i)
            print("First Stage cluster------------")
            print(CNN[i])
            print(DScandidateSize[i])
            
        DRCNN = np.zeros((FcandidateSize,FcandidateSize * ScandidateSize ), dtype=int) 
        DRCNN -= 1 
        DSorted_Rdist = np.zeros((FcandidateSize,FcandidateSize * ScandidateSize), dtype=float)
        DSorted_Rdist -=1
        if len_probRL2 < FcandidateSize:
            RANGE = len_probRL2
            
        else:
            RANGE = FcandidateSize
            
        x_test = np.zeros((FcandidateSize,SecondStage_Codebook.shape[1]*2), dtype=float)
            
        for j in range(RANGE):

           FCandidate[j] = query_sift[i] - FirstStage_Codebook[CNN[i,j]]
           x_test[j,0:SecondStage_Codebook.shape[1]] = FirstStage_Codebook[CNN[i,j]]
           x_test[j,SecondStage_Codebook.shape[1]:SecondStage_Codebook.shape[1]*2] = FCandidate[j]
           DataSize = DScandidateSize[i,j]

        x_test /= NNormalize

        p_test = model.predict(x_test)   
        
        pTest = deepcopy(p_test)
        p_idx = np.zeros((p_test.shape[0], BucketSize), dtype=int) 
        p_idx -=1
        for q in range (RANGE): 
            c = 0
            for t in range (DScandidateSize[i,q]):
                p_idx[q,c] = np.argmax(pTest[q])
                pTest[q,p_idx[q,c]] = -1        
                c += 1
                                
        for j in range(FcandidateSize):
            DataSize = DScandidateSize[i,j]
            DRCNN[j, 0:DataSize] = p_idx[j,0:DataSize]
            
            
        temp_candidate = []   
        candidate = [] 
        
        Alternate = int(np.ceil(SC * 0.3)) 
        if FcandidateSize > 1:
            for r in range (Alternate):
                temp_candidate = (CNN[i,0]*SecondStage_Codebook.shape[0])+(DRCNN[0,r])
                candidate.extend([temp_candidate])
                temp_candidate = (CNN[i,1]*SecondStage_Codebook.shape[0])+(DRCNN[1,r])
                candidate.extend([temp_candidate])
            
            for j in range(RANGE):
                if j < 2:
                    start = Alternate
                else:
                    start = 0
                    
                for k in range(start, DScandidateSize[i,j]):
                    temp_candidate = (CNN[i,j]*SecondStage_Codebook.shape[0])+(DRCNN[j,k]) 
                    candidate.extend([temp_candidate])
        else:
                for k in range(DScandidateSize[i,j]):
                    temp_candidate = (CNN[i,j]*SecondStage_Codebook.shape[0])+(DRCNN[j,k]) 
                    candidate.extend([temp_candidate])        
        
#        for j in range(RANGE):
#            for k in range(DScandidateSize[i,j]):
#                temp_candidate = (CNN[i,j]*SecondStage_Codebook.shape[0])+(DRCNN[j,k])                 
#                candidate.extend([temp_candidate])

        CandidateList = np.array(candidate)
        IdxSize = len(CandidateList)
        if IdxSize > BucketSize:
            IdxSize = BucketSize
        InsIdx2[i,0:IdxSize] =  CandidateList[0:IdxSize]
            
    cal_Time(start_time)


    np.savetxt(fname, InsIdx2, delimiter=' ',fmt='%d')
    
    return InsIdx2

 