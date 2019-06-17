from __future__ import absolute_import
from __future__ import print_function

import numpy as np
from time import time
start_time = time()

from lib import Searching_Dist_QCD_Dist_SLE
from lib import Searching_OneStage_Learning_RawData
from lib import Searching_TwoStage_Learning_RawData
from lib import First_Level_Prediction


MaxFirstStageCluster = 15
NumSubCluster = 10 
Zscore_var =  100  
START = 15
STEP = 1
 
############# Load Data #############################

print('\nLoading input files')
npzfile = np.load('./output/Idx2Codebooks.npz')
Ints_clustMem = np.array(npzfile['idx'])
npzfile = np.load('./output/Idx2CountMember.npz')
idxCount = np.array(npzfile['idxCount'])
Top1 = np.loadtxt('./input/Test_Data/Top1_Test.txt', dtype=int)
FirstStage_Codebook = np.loadtxt('./input/FirstStageCodebook.txt', dtype=float)
SecondStage_Codebook = np.loadtxt('./input/SecondStageCodebook.txt', dtype=float)

############## The Coarse Search ##############################################
print ("\nThe coarse search results.")
print ("\nRetrieve the candidate second-level clusters for SIFT1B RVQ before performing the ADC.")
print ("\nCalculate the top-1 recall along with the candidate second-level clusters")

############## Searching by using Learning Method #############################
First_Level_Prediction.First_Model_Prediction(Zscore_var)
Searching_TwoStage_Learning_RawData.Searching(MaxFirstStageCluster, NumSubCluster, Zscore_var, Ints_clustMem, idxCount, Top1, FirstStage_Codebook, SecondStage_Codebook, START, STEP)
Searching_OneStage_Learning_RawData.Searching(MaxFirstStageCluster, NumSubCluster, Zscore_var, Ints_clustMem, idxCount, Top1, FirstStage_Codebook, SecondStage_Codebook, START, STEP)

################# Searching by using Distance #############################	
Searching_Dist_QCD_Dist_SLE.Searching_by_Distance(MaxFirstStageCluster, NumSubCluster, Ints_clustMem, idxCount, Top1, FirstStage_Codebook, SecondStage_Codebook, START, STEP)

#################
end_time = time()
time_taken = end_time - start_time # time_taken is in seconds
hours, rest = divmod(time_taken,3600)
minutes, seconds = divmod(rest, 60)
print("This took %d hours %d minutes %d seconds" %(hours,minutes,seconds)) 




