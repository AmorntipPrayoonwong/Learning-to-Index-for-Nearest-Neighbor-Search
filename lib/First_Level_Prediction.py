import numpy as np
from keras.models import load_model
from copy import deepcopy

def First_Model_Prediction(Zscore_var):
    BucketSize = Zscore_var
    First_Level_Model = load_model('My_Model_RawData_WeightingProb.h5')
    query = np.loadtxt('./input/Test_Data/Test.txt', dtype=float)
    query /=  np.max(query)    
    x_test = query
    p_test = First_Level_Model.predict(x_test)
    #P_IDX = np.argsort(-p_test)
    
    pTest = deepcopy(p_test)
    P_IDX = np.zeros((query.shape[0], BucketSize), dtype=int) 
    for q in range (query.shape[0]): 
        for t in range (BucketSize):
            P_IDX[q,t] = np.argmax(pTest[q])
            pTest[q,P_IDX[q,t]] = -1
    
    np.savez_compressed('./output/(te)sort_bid_RawData_weightingProb', p_idx=P_IDX, p_prob = p_test)

