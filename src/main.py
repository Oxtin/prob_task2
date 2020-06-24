import pandas as pd
import numpy as np
import random
import sys
import time
from sklearn.metrics import ndcg_score
from evaluation_t2 import t2_evaluation

if len(sys.argv) > 1:
	dataSet = int(sys.argv[1])
else:
	exit(1)

tStart = time.time()
sample_size = 2000
total_size = 0
if dataSet == 0: # training data
	total_size = 2468285
elif dataSet == 1: # testing data
	total_size = 200000
else:
	exit(2)

# Reservoir sampling
index_list = np.array([i for i in range(sample_size)])
for i in range(sample_size, total_size):
	r = random.randint(0, i)
	if r < sample_size:
		index_list[r] = i
print("index-list:", index_list)

# =========== EVALUTATION ================
print("Init the evaluation...")
# init the evaluation
if dataSet == 0:
	t2 = t2_evaluation("../public/raw_csv_data/t2-data.csv")
else:
	t2 = t2_evaluation("../public/raw_csv_data/t2-test-data.csv")

'''
# directly compute the rank of the gt_rule
raw_gt_rank = t2.get_gt_rule_final()
gt_rank = raw_gt_rank[:,1,:].flatten()
gt_rank[gt_rank==-1] = 0
f = open("output.txt", "w")
for e in gt_rank:
	f.write(str(int(e)))
	f.write('\n')
f.close()
'''

# compute the sample rank
sample_rank = t2.sample_evaluation(index_list)

# read the pre-calculated gt_rank array
gt_rank = []
if dataSet == 0:
	f = open("../public/pre_calculated/t2-data-gt-rank.txt")
else:
	f = open("../public/pre_calculated/t2-test-data-gt-rank.txt")
lines = f.readlines()
for line in lines:
	gt_rank.append(int(line.split('\n')[0]))
f.close()

# compute the ndcg score
sample_rank = [sample_rank]
gt_rank = [gt_rank]
score = t2.evaluation(sample_rank, gt_rank)
print("======= RESULT =======")
print("nDCG score:", score)
tEnd = time.time()
print("The whole process costs %f sec" % (tEnd - tStart))
