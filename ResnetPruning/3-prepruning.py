import numpy as np
import index_sum
# layer1_1
layer1_1_1similar = np.load("./cal_similar/layer1_1_1similar.npy",allow_pickle=True)
layer1_1_1similar = layer1_1_1similar.tolist()
layer1_1_1index = np.load("./cal_similar/layer1_1_1index.npy",allow_pickle=True)
layer1_1_1index = layer1_1_1index.tolist()
layer1_1_1norm = np.load("./cal_similar/layer1_1_1norm.npy",allow_pickle=True)
layer1_1_1norm = layer1_1_1norm.tolist()
# layer1_2
layer1_1_2similar = np.load("./cal_similar/layer1_1_2similar.npy",allow_pickle=True)
layer1_1_2similar = layer1_1_2similar.tolist()
# print('similar1_1',similar1_1)
layer1_1_2index = np.load("./cal_similar/layer1_1_2index.npy",allow_pickle=True)
layer1_1_2index = layer1_1_2index.tolist()
layer1_1_2norm = np.load("./cal_similar/layer1_1_2norm.npy",allow_pickle=True)
layer1_1_2norm = layer1_1_2norm.tolist()

layer1_2_1similar = np.load("./cal_similar/layer1_2_1similar.npy",allow_pickle=True)
layer1_2_1similar = layer1_2_1similar.tolist()
layer1_2_1index = np.load("./cal_similar/layer1_2_1index.npy",allow_pickle=True)
layer1_2_1index = layer1_2_1index.tolist()
layer1_2_1norm = np.load("./cal_similar/layer1_2_1norm.npy",allow_pickle=True)
layer1_2_1norm = layer1_2_1norm.tolist()

layer1_2_2similar = np.load("./cal_similar/layer1_2_2similar.npy",allow_pickle=True)
layer1_2_2similar = layer1_2_2similar.tolist()
layer1_2_2index = np.load("./cal_similar/layer1_2_2index.npy",allow_pickle=True)
layer1_2_2index = layer1_2_2index.tolist()
layer1_2_2norm = np.load("./cal_similar/layer1_2_2norm.npy",allow_pickle=True)
layer1_2_2norm = layer1_2_2norm.tolist()

layer1_3_1similar = np.load("./cal_similar/layer1_3_1similar.npy",allow_pickle=True)
layer1_3_1similar = layer1_3_1similar.tolist()
layer1_3_1index = np.load("./cal_similar/layer1_3_1index.npy",allow_pickle=True)
layer1_3_1index = layer1_3_1index.tolist()
layer1_3_1norm = np.load("./cal_similar/layer1_3_1norm.npy",allow_pickle=True)
layer1_3_1norm = layer1_3_1norm.tolist()

layer1_3_2similar = np.load("./cal_similar/layer1_3_2similar.npy",allow_pickle=True)
layer1_3_2similar = layer1_3_2similar.tolist()
layer1_3_2index = np.load("./cal_similar/layer1_3_2index.npy",allow_pickle=True)
layer1_3_2index = layer1_3_2index.tolist()
layer1_3_2norm = np.load("./cal_similar/layer1_3_2norm.npy",allow_pickle=True)
layer1_3_2norm = layer1_3_2norm.tolist()

layer1_3_3similar = np.load("./cal_similar/layer1_3_3similar.npy",allow_pickle=True)
layer1_3_3similar = layer1_3_3similar.tolist()
layer1_3_3index = np.load("./cal_similar/layer1_3_3index.npy",allow_pickle=True)
layer1_3_3index = layer1_3_3index.tolist()
layer1_3_3norm = np.load("./cal_similar/layer1_3_3norm.npy",allow_pickle=True)
layer1_3_3norm = layer1_3_3norm.tolist()

layer2_1_1similar = np.load("./cal_similar/layer2_1_1similar.npy",allow_pickle=True)
layer2_1_1similar = layer2_1_1similar.tolist()
# print('similar1_1',similar1_1)
layer2_1_1index = np.load("./cal_similar/layer2_1_1index.npy",allow_pickle=True)
layer2_1_1index = layer2_1_1index.tolist()
layer2_1_1norm = np.load("./cal_similar/layer2_1_1norm.npy",allow_pickle=True)
layer2_1_1norm = layer2_1_1norm.tolist()
# layer1_2
layer2_1_2similar = np.load("./cal_similar/layer2_1_2similar.npy",allow_pickle=True)
layer2_1_2similar = layer2_1_2similar.tolist()
# print('similar1_1',similar1_1)
layer2_1_2index = np.load("./cal_similar/layer2_1_2index.npy",allow_pickle=True)
layer2_1_2index = layer2_1_2index.tolist()
layer2_1_2norm = np.load("./cal_similar/layer2_1_2norm.npy",allow_pickle=True)
layer2_1_2norm = layer2_1_2norm.tolist()

layer2_2_1similar = np.load("./cal_similar/layer2_2_1similar.npy",allow_pickle=True)
layer2_2_1similar = layer2_2_1similar.tolist()
layer2_2_1index = np.load("./cal_similar/layer2_2_1index.npy",allow_pickle=True)
layer2_2_1index = layer2_2_1index.tolist()
layer2_2_1norm = np.load("./cal_similar/layer2_2_1norm.npy",allow_pickle=True)
layer2_2_1norm = layer2_2_1norm.tolist()

layer2_2_2similar = np.load("./cal_similar/layer2_2_2similar.npy",allow_pickle=True)
layer2_2_2similar = layer2_2_2similar.tolist()
layer2_2_2index = np.load("./cal_similar/layer2_2_2index.npy",allow_pickle=True)
layer2_2_2index = layer2_2_2index.tolist()
layer2_2_2norm = np.load("./cal_similar/layer2_2_2norm.npy",allow_pickle=True)
layer2_2_2norm = layer2_2_2norm.tolist()

layer2_3_1similar = np.load("./cal_similar/layer2_3_1similar.npy",allow_pickle=True)
layer2_3_1similar = layer2_3_1similar.tolist()
layer2_3_1index = np.load("./cal_similar/layer2_3_1index.npy",allow_pickle=True)
layer2_3_1index = layer2_3_1index.tolist()
layer2_3_1norm = np.load("./cal_similar/layer2_3_1norm.npy",allow_pickle=True)
layer2_3_1norm = layer2_3_1norm.tolist()

layer2_3_2similar = np.load("./cal_similar/layer2_3_2similar.npy",allow_pickle=True)
layer2_3_2similar = layer2_3_2similar.tolist()
layer2_3_2index = np.load("./cal_similar/layer2_3_2index.npy",allow_pickle=True)
layer2_3_2index = layer2_3_2index.tolist()
layer2_3_2norm = np.load("./cal_similar/layer2_3_2norm.npy",allow_pickle=True)
layer2_3_2norm = layer2_3_2norm.tolist()

layer2_4_1similar = np.load("./cal_similar/layer2_4_1similar.npy",allow_pickle=True)
layer2_4_1similar = layer2_4_1similar.tolist()
layer2_4_1index = np.load("./cal_similar/layer2_4_1index.npy",allow_pickle=True)
layer2_4_1index = layer2_4_1index.tolist()
layer2_4_1norm = np.load("./cal_similar/layer2_4_1norm.npy",allow_pickle=True)
layer2_4_1norm = layer2_4_1norm.tolist()

layer2_4_2similar = np.load("./cal_similar/layer2_4_2similar.npy",allow_pickle=True)
layer2_4_2similar = layer2_4_2similar.tolist()
layer2_4_2index = np.load("./cal_similar/layer2_4_2index.npy",allow_pickle=True)
layer2_4_2index = layer2_4_2index.tolist()
layer2_4_2norm = np.load("./cal_similar/layer2_4_2norm.npy",allow_pickle=True)
layer2_4_2norm = layer2_4_2norm.tolist()

layer2_4_3similar = np.load("./cal_similar/layer2_4_3similar.npy",allow_pickle=True)
layer2_4_3similar = layer2_4_3similar.tolist()
layer2_4_3index = np.load("./cal_similar/layer2_4_3index.npy",allow_pickle=True)
layer2_4_3index = layer2_4_3index.tolist()
layer2_4_3norm = np.load("./cal_similar/layer2_4_3norm.npy",allow_pickle=True)
layer2_4_3norm = layer2_4_3norm.tolist()

layer3_1_1similar = np.load("./cal_similar/layer3_1_1similar.npy",allow_pickle=True)
layer3_1_1similar = layer3_1_1similar.tolist()
layer3_1_1index = np.load("./cal_similar/layer3_1_1index.npy",allow_pickle=True)
layer3_1_1index = layer3_1_1index.tolist()
layer3_1_1norm = np.load("./cal_similar/layer3_1_1norm.npy",allow_pickle=True)
layer3_1_1norm = layer3_1_1norm.tolist()
# layer1_2
layer3_1_2similar = np.load("./cal_similar/layer3_1_2similar.npy",allow_pickle=True)
layer3_1_2similar = layer3_1_2similar.tolist()
layer3_1_2index = np.load("./cal_similar/layer3_1_2index.npy",allow_pickle=True)
layer3_1_2index = layer3_1_2index.tolist()
layer3_1_2norm = np.load("./cal_similar/layer3_1_2norm.npy",allow_pickle=True)
layer3_1_2norm = layer3_1_2norm.tolist()

layer3_2_1similar = np.load("./cal_similar/layer3_2_1similar.npy",allow_pickle=True)
layer3_2_1similar = layer3_2_1similar.tolist()
layer3_2_1index = np.load("./cal_similar/layer3_2_1index.npy",allow_pickle=True)
layer3_2_1index = layer3_2_1index.tolist()
layer3_2_1norm = np.load("./cal_similar/layer3_2_1norm.npy",allow_pickle=True)
layer3_2_1norm = layer3_2_1norm.tolist()

layer3_2_2similar = np.load("./cal_similar/layer3_2_2similar.npy",allow_pickle=True)
layer3_2_2similar = layer3_2_2similar.tolist()
layer3_2_2index = np.load("./cal_similar/layer3_2_2index.npy",allow_pickle=True)
layer3_2_2index = layer3_2_2index.tolist()
layer3_2_2norm = np.load("./cal_similar/layer3_2_2norm.npy",allow_pickle=True)
layer3_2_2norm = layer3_2_2norm.tolist()

layer3_3_1similar = np.load("./cal_similar/layer3_3_1similar.npy",allow_pickle=True)
layer3_3_1similar = layer3_3_1similar.tolist()
layer3_3_1index = np.load("./cal_similar/layer3_3_1index.npy",allow_pickle=True)
layer3_3_1index = layer3_3_1index.tolist()
layer3_3_1norm = np.load("./cal_similar/layer3_3_1norm.npy",allow_pickle=True)
layer3_3_1norm = layer3_3_1norm.tolist()

layer3_3_2similar = np.load("./cal_similar/layer3_3_2similar.npy",allow_pickle=True)
layer3_3_2similar = layer3_3_2similar.tolist()
layer3_3_2index = np.load("./cal_similar/layer3_3_2index.npy",allow_pickle=True)
layer3_3_2index = layer3_3_2index.tolist()
layer3_3_2norm = np.load("./cal_similar/layer3_3_2norm.npy",allow_pickle=True)
layer3_3_2norm = layer3_3_2norm.tolist()

layer3_4_1similar = np.load("./cal_similar/layer3_4_1similar.npy",allow_pickle=True)
layer3_4_1similar = layer3_4_1similar.tolist()
layer3_4_1index = np.load("./cal_similar/layer3_4_1index.npy",allow_pickle=True)
layer3_4_1index = layer3_4_1index.tolist()
layer3_4_1norm = np.load("./cal_similar/layer3_4_1norm.npy",allow_pickle=True)
layer3_4_1norm = layer3_4_1norm.tolist()

layer3_4_2similar = np.load("./cal_similar/layer3_4_2similar.npy",allow_pickle=True)
layer3_4_2similar = layer3_4_2similar.tolist()
layer3_4_2index = np.load("./cal_similar/layer3_4_2index.npy",allow_pickle=True)
layer3_4_2index = layer3_4_2index.tolist()
layer3_4_2norm = np.load("./cal_similar/layer3_4_2norm.npy",allow_pickle=True)
layer3_4_2norm = layer3_4_2norm.tolist()

layer3_5_1similar = np.load("./cal_similar/layer3_5_1similar.npy",allow_pickle=True)
layer3_5_1similar = layer3_5_1similar.tolist()
layer3_5_1index = np.load("./cal_similar/layer3_5_1index.npy",allow_pickle=True)
layer3_5_1index = layer3_5_1index.tolist()
layer3_5_1norm = np.load("./cal_similar/layer3_5_1norm.npy",allow_pickle=True)
layer3_5_1norm = layer3_5_1norm.tolist()

layer3_5_2similar = np.load("./cal_similar/layer3_5_2similar.npy",allow_pickle=True)
layer3_5_2similar = layer3_5_2similar.tolist()
layer3_5_2index = np.load("./cal_similar/layer3_5_2index.npy",allow_pickle=True)
layer3_5_2index = layer3_5_2index.tolist()
layer3_5_2norm = np.load("./cal_similar/layer3_5_2norm.npy",allow_pickle=True)
layer3_5_2norm = layer3_5_2norm.tolist()

layer3_6_1similar = np.load("./cal_similar/layer3_6_1similar.npy",allow_pickle=True)
layer3_6_1similar = layer3_6_1similar.tolist()
layer3_6_1index = np.load("./cal_similar/layer3_6_1index.npy",allow_pickle=True)
layer3_6_1index = layer3_6_1index.tolist()
layer3_6_1norm = np.load("./cal_similar/layer3_6_1norm.npy",allow_pickle=True)
layer3_6_1norm = layer3_6_1norm.tolist()

layer3_6_2similar = np.load("./cal_similar/layer3_6_2similar.npy",allow_pickle=True)
layer3_6_2similar = layer3_6_2similar.tolist()
layer3_6_2index = np.load("./cal_similar/layer3_6_2index.npy",allow_pickle=True)
layer3_6_2index = layer3_6_2index.tolist()
layer3_6_2norm = np.load("./cal_similar/layer3_6_2norm.npy",allow_pickle=True)
layer3_6_2norm = layer3_6_2norm.tolist()

layer3_6_3similar = np.load("./cal_similar/layer3_6_3similar.npy",allow_pickle=True)
layer3_6_3similar = layer3_6_3similar.tolist()
layer3_6_3index = np.load("./cal_similar/layer3_6_3index.npy",allow_pickle=True)
layer3_6_3index = layer3_6_3index.tolist()
layer3_6_3norm = np.load("./cal_similar/layer3_6_3norm.npy",allow_pickle=True)
layer3_6_3norm = layer3_6_3norm.tolist()

layer4_1_1similar = np.load("./cal_similar/layer4_1_1similar.npy",allow_pickle=True)
layer4_1_1similar = layer4_1_1similar.tolist()
layer4_1_1index = np.load("./cal_similar/layer4_1_1index.npy",allow_pickle=True)
layer4_1_1index = layer4_1_1index.tolist()
layer4_1_1norm = np.load("./cal_similar/layer4_1_1norm.npy",allow_pickle=True)
layer4_1_1norm = layer4_1_1norm.tolist()
# layer1_2
layer4_1_2similar = np.load("./cal_similar/layer4_1_2similar.npy",allow_pickle=True)
layer4_1_2similar = layer4_1_2similar.tolist()
# print('similar1_1',similar1_1)
layer4_1_2index = np.load("./cal_similar/layer4_1_2index.npy",allow_pickle=True)
layer4_1_2index = layer4_1_2index.tolist()
layer4_1_2norm = np.load("./cal_similar/layer4_1_2norm.npy",allow_pickle=True)
layer4_1_2norm = layer4_1_2norm.tolist()

layer4_2_1similar = np.load("./cal_similar/layer4_2_1similar.npy",allow_pickle=True)
layer4_2_1similar = layer4_2_1similar.tolist()
layer4_2_1index = np.load("./cal_similar/layer4_2_1index.npy",allow_pickle=True)
layer4_2_1index = layer4_2_1index.tolist()
layer4_2_1norm = np.load("./cal_similar/layer4_2_1norm.npy",allow_pickle=True)
layer4_2_1norm = layer4_2_1norm.tolist()

layer4_2_2similar = np.load("./cal_similar/layer4_2_2similar.npy",allow_pickle=True)
layer4_2_2similar = layer4_2_2similar.tolist()
layer4_2_2index = np.load("./cal_similar/layer4_2_2index.npy",allow_pickle=True)
layer4_2_2index = layer4_2_2index.tolist()
layer4_2_2norm = np.load("./cal_similar/layer4_2_2norm.npy",allow_pickle=True)
layer4_2_2norm = layer4_2_2norm.tolist()

layer4_3_1similar = np.load("./cal_similar/layer4_3_1similar.npy",allow_pickle=True)
layer4_3_1similar = layer4_3_1similar.tolist()
layer4_3_1index = np.load("./cal_similar/layer4_3_1index.npy",allow_pickle=True)
layer4_3_1index = layer4_3_1index.tolist()
layer4_3_1norm = np.load("./cal_similar/layer4_3_1norm.npy",allow_pickle=True)
layer4_3_1norm = layer4_3_1norm.tolist()

layer4_3_2similar = np.load("./cal_similar/layer4_3_2similar.npy",allow_pickle=True)
layer4_3_2similar = layer4_3_2similar.tolist()
layer4_3_2index = np.load("./cal_similar/layer4_3_2index.npy",allow_pickle=True)
layer4_3_2index = layer4_3_2index.tolist()
layer4_3_2norm = np.load("./cal_similar/layer4_3_2norm.npy",allow_pickle=True)
layer4_3_2norm = layer4_3_2norm.tolist()

layer4_3_3similar = np.load("./cal_similar/layer4_3_3similar.npy",allow_pickle=True)
layer4_3_3similar = layer4_3_3similar.tolist()
layer4_3_3index = np.load("./cal_similar/layer4_3_3index.npy",allow_pickle=True)
layer4_3_3index = layer4_3_3index.tolist()
layer4_3_3norm = np.load("./cal_similar/layer4_3_3norm.npy",allow_pickle=True)
layer4_3_3norm = layer4_3_3norm.tolist()

# layer1_1_1 (64 -> 56)
remain =[]
remove_total = []
cfg_before = []
cfg_before.append(64)
remain.append(64)
remove_total.append([])
layer1_1_1norm = np.sum(layer1_1_1norm, axis=(0,)) / 40
sum1, sum2, sum3 = index_sum.index_sum(layer1_1_1similar)
index1, index2, index3 = index_sum.index_cut(layer1_1_1index,sum1,sum2,sum3)
layer1_1_1remove = index_sum.remove_index1_1_1(sum1,index1,sum2,index2,sum3,index3,layer1_1_1norm)
# layer1_1_1remove =[]
print('layer1_1_1remove',layer1_1_1remove)
layer1_1_1remain = len(layer1_1_1norm) - len(layer1_1_1remove)
print('len(layer1_1_1remove)',len(layer1_1_1remove))
cfg_before.append(len(layer1_1_1norm))
remain.append(layer1_1_1remain)
remove_total.append(layer1_1_1remove)

# layer1_1_2 (64 -> 56)
layer1_1_2norm = np.sum(layer1_1_2norm, axis=(0,))
sum1, sum2, sum3 = index_sum.index_sum(layer1_1_2similar)
index1, index2, index3 = index_sum.index_cut(layer1_1_2index,sum1,sum2,sum3)
layer1_1_2remove = index_sum.remove_index1_1_1(sum1,index1,sum2,index2,sum3,index3,layer1_1_2norm)
# layer1_1_2remove =[]
layer1_1_2remain = len(layer1_1_2norm) - len(layer1_1_2remove)
cfg_before.append(len(layer1_1_2norm))
remain.append(layer1_1_2remain)
remove_total.append(layer1_1_2remove)

# layer1_1_3 = layer1_3_3
#   (256 -> 248)
layer1_3_3norm = np.sum(layer1_3_3norm,axis=(0,))
sum1, sum2, sum3 = index_sum.index_sum(layer1_3_3similar)
index1,index2,index3 = index_sum.index_cut(layer1_3_3index,sum1,sum2,sum3)
layer1_3_3remove = index_sum.remove_index1_1_1(sum1,index1,sum2,index2,sum3,index3,layer1_3_3norm)
# layer1_3_3remove =[]
print('layer1_3_3remove',layer1_3_3remove)
layer1_3_3remain = len(layer1_3_3norm) - len(layer1_3_3remove)
cfg_before.append(len(layer1_3_3norm))
layer1_1_3remove=layer1_3_3remove
layer1_1_3remain=layer1_3_3remain
print('layer1_3_3remain',layer1_3_3remain)
remain.append(layer1_1_3remain)
remove_total.append(layer1_1_3remove)
cfg_before.append(len(layer1_3_3norm))
layer1_1_3remove=layer1_3_3remove
layer1_1_3remain=layer1_3_3remain
print('layer1_3_3remain',layer1_3_3remain)
remain.append(layer1_1_3remain)
remove_total.append(layer1_1_3remove)

# layer1_2_1 (64 -> 56)
layer1_2_1norm = np.sum(layer1_2_1norm, axis=(0,))
sum1, sum2, sum3 = index_sum.index_sum(layer1_2_1similar)
index1, index2, index3 = index_sum.index_cut(layer1_2_1index,sum1,sum2,sum3)
layer1_2_1remove = index_sum.remove_index1_1_1(sum1,index1,sum2,index2,sum3,index3,layer1_2_1norm)
# layer1_2_1remove =[]
layer1_2_1remain = len(layer1_2_1norm) - len(layer1_2_1remove)

cfg_before.append(len(layer1_2_1norm))
remain.append(layer1_2_1remain)
remove_total.append(layer1_2_1remove)

# layer1_2_2 (64 -> 56)
layer1_2_2norm = np.sum(layer1_2_2norm, axis=(0,))
sum1, sum2, sum3 = index_sum.index_sum(layer1_2_2similar)
index1, index2, index3 = index_sum.index_cut(layer1_2_2index,sum1,sum2,sum3)
layer1_2_2remove = index_sum.remove_index1_1_1(sum1,index1,sum2,index2,sum3,index3,layer1_2_2norm)
# layer1_2_2remove =[]
layer1_2_2remain = len(layer1_2_2norm) - len(layer1_2_2remove)

cfg_before.append(len(layer1_2_2norm))
remain.append(layer1_2_2remain)
remove_total.append(layer1_2_2remove)

cfg_before.append(len(layer1_3_3norm))
layer1_2_3remove=layer1_3_3remove
layer1_2_3remain=layer1_3_3remain
# print('layer1_3_3remain',layer1_3_3remain)
remain.append(layer1_2_3remain)
remove_total.append(layer1_2_3remove)

# layer1_3_1 (64 -> 56)
layer1_3_1norm = np.sum(layer1_3_1norm, axis=(0,))
sum1, sum2, sum3 = index_sum.index_sum(layer1_3_1similar)
index1, index2, index3 = index_sum.index_cut(layer1_3_1index,sum1,sum2,sum3)
layer1_3_1remove = index_sum.remove_index1_1_1(sum1,index1,sum2,index2,sum3,index3,layer1_3_1norm)
# layer1_3_1remove =[]
layer1_3_1remain = len(layer1_3_1norm) - len(layer1_3_1remove)
cfg_before.append(len(layer1_3_1norm))
remain.append(layer1_3_1remain)
remove_total.append(layer1_3_1remove)

# layer1_3_2 (64 -> 56)
layer1_3_2norm = np.sum(layer1_3_2norm, axis=(0,))
sum1, sum2, sum3 = index_sum.index_sum(layer1_3_2similar)
index1, index2, index3 = index_sum.index_cut(layer1_3_2index,sum1,sum2,sum3)
layer1_3_2remove = index_sum.remove_index1_1_1(sum1,index1,sum2,index2,sum3,index3,layer1_3_2norm)
# layer1_3_2remove =[]
layer1_3_2remain = len(layer1_3_2norm) - len(layer1_3_2remove)
cfg_before.append(len(layer1_3_2norm))
remain.append(layer1_3_2remain)
remove_total.append(layer1_3_2remove)
# layer1_3_3  (256 -> 242)
cfg_before.append(len(layer1_3_3norm))
layer1_3_3remain = len(layer1_3_3norm) - len(layer1_3_3remove)
print('layer1_3_3remain',layer1_3_3remain)
remain.append(layer1_3_3remain)
remove_total.append(layer1_3_3remove)

# layer2_1_1  (128 -> 242)
layer2_1_1norm = np.sum(layer2_1_1norm, axis=(0,))
sum1, sum2, sum3 = index_sum.index_sum(layer2_1_1similar)
index1, index2, index3 = index_sum.index_cut(layer2_1_1index,sum1,sum2,sum3)
layer2_1_1remove = index_sum.remove_index2_1_1(sum1,index1,sum2,index2,sum3,index3,layer2_1_1norm)
# layer2_1_1remove =[]

layer2_1_1remain = len(layer2_1_1norm) - len(layer2_1_1remove)

'''11'''
cfg_before.append(len(layer2_1_1norm))
remain.append(layer2_1_1remain)
remove_total.append(layer2_1_1remove)

# layer2_1_2  (128 -> 248)
layer2_1_2norm = np.sum(layer2_1_2norm, axis=(0,))
sum1, sum2, sum3 = index_sum.index_sum(layer2_1_2similar)
index1, index2, index3 = index_sum.index_cut(layer2_1_2index,sum1,sum2,sum3)
layer2_1_2remove = index_sum.remove_index2_1_1(sum1,index1,sum2,index2,sum3,index3,layer2_1_2norm)
# layer2_1_2remove =[]
layer2_1_2remain = len(layer2_1_2norm) - len(layer2_1_2remove)
'''12'''
cfg_before.append(len(layer2_1_2norm))
remain.append(layer2_1_2remain)
remove_total.append(layer2_1_2remove)

# layer2_1_3
layer2_4_3norm = np.sum(layer2_4_3norm, axis=(0,))
sum1, sum2, sum3 = index_sum.index_sum(layer2_4_3similar)
index1, index2, index3 = index_sum.index_cut(layer2_4_3index,sum1,sum2,sum3)
layer2_4_3remove = index_sum.remove_index2_1_1(sum1,index1,sum2,index2,sum3,index3,layer2_4_3norm)
# layer2_4_3remove =[]
layer2_4_3remain = len(layer2_4_3norm) - len(layer2_4_3remove)
'''13'''
cfg_before.append(len(layer2_4_3norm))
layer2_1_3remove=layer2_4_3remove
layer2_1_3remain=layer2_4_3remain
# print('layer2_1_3remain',layer2_1_3remain)
# input()
remain.append(layer2_1_3remain)
remove_total.append(layer2_1_3remove)
'''14'''
cfg_before.append(len(layer2_4_3norm))
layer2_1_3remove=layer2_4_3remove
layer2_1_3remain=layer2_4_3remain
print('layer2_1_3remain',layer2_1_3remain)
remain.append(layer2_1_3remain)
remove_total.append(layer2_1_3remove)

# layer2_2_1  (128 -> 248)
layer2_2_1norm = np.sum(layer2_2_1norm, axis=(0,))
sum1, sum2, sum3 = index_sum.index_sum(layer2_2_1similar)
index1, index2, index3 = index_sum.index_cut(layer2_2_1index,sum1,sum2,sum3)
layer2_2_1remove = index_sum.remove_index2_1_1(sum1,index1,sum2,index2,sum3,index3,layer2_2_1norm)
# layer2_2_1remove =[]
layer2_2_1remain = len(layer2_2_1norm) - len(layer2_2_1remove)
'''15'''
cfg_before.append(len(layer2_2_1norm))
remain.append(layer2_2_1remain)
remove_total.append(layer2_2_1remove)

# layer2_2_2  (128 -> 248)
layer2_2_2norm = np.sum(layer2_2_2norm, axis=(0,))
sum1, sum2, sum3 = index_sum.index_sum(layer2_2_2similar)
index1, index2, index3 = index_sum.index_cut(layer2_2_2index,sum1,sum2,sum3)
layer2_2_2remove = index_sum.remove_index2_1_1(sum1,index1,sum2,index2,sum3,index3,layer2_2_2norm)
# layer2_2_2remove =[]
layer2_2_2remain = len(layer2_2_2norm) - len(layer2_2_2remove)
cfg_before.append(len(layer2_2_2norm))
remain.append(layer2_2_2remain)
remove_total.append(layer2_2_2remove)

# layer2_2_3
cfg_before.append(len(layer2_4_3norm))
layer2_2_3remove=layer2_4_3remove
layer2_2_3remain=layer2_4_3remain
print('layer2_2_3remain',layer2_2_3remain)
remain.append(layer2_2_3remain)
remove_total.append(layer2_2_3remove)

# layer2_3_1  (128 -> 248)
layer2_3_1norm = np.sum(layer2_3_1norm, axis=(0,))
sum1, sum2, sum3 = index_sum.index_sum(layer2_3_1similar)
index1, index2, index3 = index_sum.index_cut(layer2_3_1index,sum1,sum2,sum3)
layer2_3_1remove = index_sum.remove_index2_1_1(sum1,index1,sum2,index2,sum3,index3,layer2_3_1norm)
# layer2_3_1remove =[]
layer2_3_1remain = len(layer2_3_1norm) - len(layer2_3_1remove)
cfg_before.append(len(layer2_3_1norm))
remain.append(layer2_3_1remain)
remove_total.append(layer2_3_1remove)

# layer2_3_2  (128 -> 248)
layer2_3_2norm = np.sum(layer2_3_2norm, axis=(0,))
sum1, sum2, sum3 = index_sum.index_sum(layer2_3_2similar)
index1, index2, index3 = index_sum.index_cut(layer2_3_2index,sum1,sum2,sum3)
layer2_3_2remove = index_sum.remove_index2_1_1(sum1,index1,sum2,index2,sum3,index3,layer2_3_2norm)
# layer2_3_2remove =[]
layer2_3_2remain = len(layer2_3_2norm) - len(layer2_3_2remove)
cfg_before.append(len(layer2_3_2norm))
remain.append(layer2_3_2remain)
remove_total.append(layer2_3_2remove)

#layer2_3_3
cfg_before.append(len(layer2_4_3norm))
layer2_3_3remove=layer2_4_3remove
layer2_3_3remain=layer2_4_3remain
print('layer2_3_3remain',layer2_3_3remain)
remain.append(layer2_3_3remain)
remove_total.append(layer2_3_3remove)

# layer2_4_1  (128 -> 248)
layer2_4_1norm = np.sum(layer2_4_1norm, axis=(0,))
sum1, sum2, sum3 = index_sum.index_sum(layer2_4_1similar)
index1, index2, index3 = index_sum.index_cut(layer2_4_1index,sum1,sum2,sum3)
layer2_4_1remove = index_sum.remove_index2_1_1(sum1,index1,sum2,index2,sum3,index3,layer2_4_1norm)
# layer2_4_1remove =[]
layer2_4_1remain = len(layer2_4_1norm) - len(layer2_4_1remove)
cfg_before.append(len(layer2_4_1norm))
remain.append(layer2_4_1remain)
remove_total.append(layer2_4_1remove)

# layer2_4_2  (128 -> 248)
layer2_4_2norm = np.sum(layer2_4_2norm, axis=(0,))
sum1, sum2, sum3 = index_sum.index_sum(layer2_4_2similar)
index1, index2, index3 = index_sum.index_cut(layer2_4_2index,sum1,sum2,sum3)
layer2_4_2remove = index_sum.remove_index2_1_1(sum1,index1,sum2,index2,sum3,index3,layer2_4_2norm)
# layer2_4_2remove =[]
layer2_4_2remain = len(layer2_4_2norm) - len(layer2_4_2remove)
cfg_before.append(len(layer2_4_2norm))
remain.append(layer2_4_2remain)
remove_total.append(layer2_4_2remove)

# layer2_4_3  (128 -> 248)
cfg_before.append(len(layer2_4_3norm))
remain.append(layer2_4_3remain)
remove_total.append(layer2_4_3remove)

# layer3_1_1  (128 -> 248)
layer3_1_1norm = np.sum(layer3_1_1norm, axis=(0,))
sum1, sum2, sum3 = index_sum.index_sum(layer3_1_1similar)
index1, index2, index3 = index_sum.index_cut(layer3_1_1index,sum1,sum2,sum3)
layer3_1_1remove = index_sum.remove_index3_1_1(sum1,index1,sum2,index2,sum3,index3,layer3_1_1norm)
# layer3_1_1remove =[]
layer3_1_1remain = len(layer3_1_1norm) - len(layer3_1_1remove)
cfg_before.append(len(layer3_1_1norm))
remain.append(layer3_1_1remain)
remove_total.append(layer3_1_1remove)

# layer3_1_2  (128 -> 248)
layer3_1_2norm = np.sum(layer3_1_2norm, axis=(0,))
sum1, sum2, sum3 = index_sum.index_sum(layer3_1_2similar)
index1, index2, index3 = index_sum.index_cut(layer3_1_2index,sum1,sum2,sum3)
layer3_1_2remove = index_sum.remove_index3_1_1(sum1,index1,sum2,index2,sum3,index3,layer3_1_2norm)
# layer3_1_2remove =[]
layer3_1_2remain = len(layer3_1_2norm) - len(layer3_1_2remove)
cfg_before.append(len(layer3_1_2norm))
remain.append(layer3_1_2remain)
remove_total.append(layer3_1_2remove)

# layer3_1_3
layer3_6_3norm = np.sum(layer3_6_3norm, axis=(0,))
sum1, sum2, sum3 = index_sum.index_sum(layer3_6_3similar)
index1, index2, index3 = index_sum.index_cut(layer3_6_3index,sum1,sum2,sum3)
layer3_6_3remove = index_sum.remove_index3_1_1(sum1,index1,sum2,index2,sum3,index3,layer3_6_3norm)
# layer3_6_3remove =[]
layer3_6_3remain = len(layer3_6_3norm) - len(layer3_6_3remove)
cfg_before.append(len(layer3_6_3norm))
layer3_1_3remove = layer3_6_3remove
layer3_1_3remain = layer3_6_3remain
remain.append(layer3_1_3remain)
remove_total.append(layer3_1_3remove)

cfg_before.append(len(layer3_6_3norm))
layer3_1_3remove = layer3_6_3remove
layer3_1_3remain = layer3_6_3remain
remain.append(layer3_1_3remain)
remove_total.append(layer3_1_3remove)

# layer3_2_1  (128 -> 248)
layer3_2_1norm = np.sum(layer3_2_1norm, axis=(0,))
sum1, sum2, sum3 = index_sum.index_sum(layer3_2_1similar)
index1, index2, index3 = index_sum.index_cut(layer3_2_1index,sum1,sum2,sum3)
layer3_2_1remove = index_sum.remove_index3_1_1(sum1,index1,sum2,index2,sum3,index3,layer3_2_1norm)
# layer3_2_1remove =[]
layer3_2_1remain = len(layer3_2_1norm) - len(layer3_2_1remove)
cfg_before.append(len(layer3_2_1norm))
remain.append(layer3_2_1remain)
remove_total.append(layer3_2_1remove)

# layer3_2_2  (128 -> 248)
layer3_2_2norm = np.sum(layer3_2_2norm, axis=(0,))
sum1, sum2, sum3 = index_sum.index_sum(layer3_2_2similar)
index1, index2, index3 = index_sum.index_cut(layer3_2_2index,sum1,sum2,sum3)
layer3_2_2remove = index_sum.remove_index3_1_1(sum1,index1,sum2,index2,sum3,index3,layer3_2_2norm)
# layer3_2_2remove =[]
layer3_2_2remain = len(layer3_2_2norm) - len(layer3_2_2remove)
cfg_before.append(len(layer3_2_2norm))
remain.append(layer3_2_2remain)
remove_total.append(layer3_2_2remove)

# layer3_2_3
cfg_before.append(len(layer3_6_3norm))
layer3_2_3remove = layer3_6_3remove
layer3_2_3remain = layer3_6_3remain
remain.append(layer3_2_3remain)
remove_total.append(layer3_2_3remove)

# layer3_3_1  (128 -> 248)
layer3_3_1norm = np.sum(layer3_3_1norm, axis=(0,))
sum1, sum2, sum3 = index_sum.index_sum(layer3_3_1similar)
index1, index2, index3 = index_sum.index_cut(layer3_3_1index,sum1,sum2,sum3)
layer3_3_1remove = index_sum.remove_index3_1_1(sum1,index1,sum2,index2,sum3,index3,layer3_3_1norm)
# layer3_3_1remove =[]
layer3_3_1remain = len(layer3_3_1norm) - len(layer3_3_1remove)
cfg_before.append(len(layer3_3_1norm))
remain.append(layer3_3_1remain)
remove_total.append(layer3_3_1remove)

# layer3_3_2  (128 -> 248)
layer3_3_2norm = np.sum(layer3_3_2norm, axis=(0,))
sum1, sum2, sum3 = index_sum.index_sum(layer3_3_2similar)
index1, index2, index3 = index_sum.index_cut(layer3_3_2index,sum1,sum2,sum3)
layer3_3_2remove = index_sum.remove_index3_1_1(sum1,index1,sum2,index2,sum3,index3,layer3_3_2norm)
# layer3_3_2remove =[]
layer3_3_2remain = len(layer3_3_2norm) - len(layer3_3_2remove)
cfg_before.append(len(layer3_3_2norm))
remain.append(layer3_3_2remain)
remove_total.append(layer3_3_2remove)

# layer3_3_3
cfg_before.append(len(layer3_6_3norm))
layer3_3_3remove = layer3_6_3remove
layer3_3_3remain = layer3_6_3remain
remain.append(layer3_3_3remain)
remove_total.append(layer3_3_3remove)

# layer3_4_1  (128 -> 248)
layer3_4_1norm = np.sum(layer3_4_1norm, axis=(0,))
sum1, sum2, sum3 = index_sum.index_sum(layer3_4_1similar)
index1, index2, index3 = index_sum.index_cut(layer3_4_1index,sum1,sum2,sum3)
layer3_4_1remove = index_sum.remove_index3_1_1(sum1,index1,sum2,index2,sum3,index3,layer3_4_1norm)
# layer3_4_1remove =[]
layer3_4_1remain = len(layer3_4_1norm) - len(layer3_4_1remove)
cfg_before.append(len(layer3_4_1norm))
remain.append(layer3_4_1remain)
remove_total.append(layer3_4_1remove)

# layer3_4_2  (128 -> 248)
layer3_4_2norm = np.sum(layer3_4_2norm, axis=(0,))
sum1, sum2, sum3 = index_sum.index_sum(layer3_4_2similar)
index1, index2, index3 = index_sum.index_cut(layer3_4_2index,sum1,sum2,sum3)
layer3_4_2remove = index_sum.remove_index3_1_1(sum1,index1,sum2,index2,sum3,index3,layer3_4_2norm)
# layer3_4_2remove =[]
layer3_4_2remain = len(layer3_4_2norm) - len(layer3_4_2remove)
cfg_before.append(len(layer3_4_2norm))
remain.append(layer3_4_2remain)
remove_total.append(layer3_4_2remove)

# layer3_4_3
cfg_before.append(len(layer3_6_3norm))
layer3_4_3remove = layer3_6_3remove
layer3_4_3remain = layer3_6_3remain
remain.append(layer3_4_3remain)
remove_total.append(layer3_4_3remove)

# layer3_5_1  (128 -> 248)
layer3_5_1norm = np.sum(layer3_5_1norm, axis=(0,))
sum1, sum2, sum3 = index_sum.index_sum(layer3_5_1similar)
index1, index2, index3 = index_sum.index_cut(layer3_5_1index,sum1,sum2,sum3)
layer3_5_1remove = index_sum.remove_index3_1_1(sum1,index1,sum2,index2,sum3,index3,layer3_5_1norm)
# layer3_5_1remove =[]
layer3_5_1remain = len(layer3_5_1norm) - len(layer3_5_1remove)
cfg_before.append(len(layer3_5_1norm))
remain.append(layer3_5_1remain)
remove_total.append(layer3_5_1remove)

# layer3_5_2  (128 -> 248)
layer3_5_2norm = np.sum(layer3_5_2norm, axis=(0,))
sum1, sum2, sum3 = index_sum.index_sum(layer3_5_2similar)
index1, index2, index3 = index_sum.index_cut(layer3_5_2index,sum1,sum2,sum3)
layer3_5_2remove = index_sum.remove_index3_1_1(sum1,index1,sum2,index2,sum3,index3,layer3_5_2norm)
# layer3_5_2remove =[]
layer3_5_2remain = len(layer3_5_2norm) - len(layer3_5_2remove)
cfg_before.append(len(layer3_5_2norm))
remain.append(layer3_5_2remain)
remove_total.append(layer3_5_2remove)

# layer3_5_3
cfg_before.append(len(layer3_6_3norm))
layer3_5_3remove = layer3_6_3remove
layer3_5_3remain = layer3_6_3remain
remain.append(layer3_5_3remain)
remove_total.append(layer3_5_3remove)

# layer3_6_1  (128 -> 248)
layer3_6_1norm = np.sum(layer3_6_1norm, axis=(0,))
sum1, sum2, sum3 = index_sum.index_sum(layer3_6_1similar)
index1, index2, index3 = index_sum.index_cut(layer3_6_1index,sum1,sum2,sum3)
layer3_6_1remove = index_sum.remove_index3_1_1(sum1,index1,sum2,index2,sum3,index3,layer3_6_1norm)
# layer3_6_1remove =[]
layer3_6_1remain = len(layer3_6_1norm) - len(layer3_6_1remove)
cfg_before.append(len(layer3_6_1norm))
remain.append(layer3_6_1remain)
remove_total.append(layer3_6_1remove)

# layer3_6_2  (128 -> 248)
layer3_6_2norm = np.sum(layer3_6_2norm, axis=(0,))
sum1, sum2, sum3 = index_sum.index_sum(layer3_6_2similar)
index1, index2, index3 = index_sum.index_cut(layer3_6_2index,sum1,sum2,sum3)
layer3_6_2remove = index_sum.remove_index3_1_1(sum1,index1,sum2,index2,sum3,index3,layer3_6_2norm)
# layer3_6_2remove =[]
layer3_6_2remain = len(layer3_6_2norm) - len(layer3_6_2remove)
cfg_before.append(len(layer3_6_2norm))
remain.append(layer3_6_2remain)
remove_total.append(layer3_6_2remove)

# layer3_6_3  (128 -> 248)
cfg_before.append(len(layer3_6_3norm))
remain.append(layer3_6_3remain)
remove_total.append(layer3_6_3remove)

# layer4_1_1  (128 -> 248)
layer4_1_1norm = np.sum(layer4_1_1norm, axis=(0,))
sum1, sum2, sum3 = index_sum.index_sum(layer4_1_1similar)
index1, index2, index3 = index_sum.index_cut(layer4_1_1index,sum1,sum2,sum3)
layer4_1_1remove = index_sum.remove_index4_1_1(sum1,index1,sum2,index2,sum3,index3,layer4_1_1norm)
# layer4_1_1remove =[]
layer4_1_1remain = len(layer4_1_1norm) - len(layer4_1_1remove)
cfg_before.append(len(layer4_1_1norm))
remain.append(layer4_1_1remain)
remove_total.append(layer4_1_1remove)

# layer4_1_2  (128 -> 248)
layer4_1_2norm = np.sum(layer4_1_2norm, axis=(0,))
sum1, sum2, sum3 = index_sum.index_sum(layer4_1_2similar)
index1, index2, index3 = index_sum.index_cut(layer4_1_2index,sum1,sum2,sum3)
layer4_1_2remove = index_sum.remove_index4_1_1(sum1,index1,sum2,index2,sum3,index3,layer4_1_2norm)
# layer4_1_2remove =[]
layer4_1_2remain = len(layer4_1_2norm) - len(layer4_1_2remove)
cfg_before.append(len(layer4_1_2norm))
remain.append(layer4_1_2remain)
remove_total.append(layer4_1_2remove)

#layer4_1_3
layer4_3_3norm = np.sum(layer4_3_3norm, axis=(0,))
sum1, sum2, sum3 = index_sum.index_sum(layer4_3_3similar)
index1, index2, index3 = index_sum.index_cut(layer4_3_3index,sum1,sum2,sum3)
layer4_3_3remove = index_sum.remove_index4_1_1(sum1,index1,sum2,index2,sum3,index3,layer4_3_3norm)
# layer4_3_3remove =[]
layer4_3_3remain = len(layer4_3_3norm) - len(layer4_3_3remove)
cfg_before.append(len(layer4_3_3norm))
layer4_1_3remove = layer4_3_3remove
layer4_1_3remain = layer4_3_3remain
remain.append(layer4_1_3remain)
remove_total.append(layer4_1_3remove)

cfg_before.append(len(layer4_3_3norm))
layer4_1_3remove = layer4_3_3remove
layer4_1_3remain = layer4_3_3remain
remain.append(layer4_1_3remain)
remove_total.append(layer4_1_3remove)

# layer4_2_1  (128 -> 248)
layer4_2_1norm = np.sum(layer4_2_1norm, axis=(0,))
sum1, sum2, sum3 = index_sum.index_sum(layer4_2_1similar)
index1, index2, index3 = index_sum.index_cut(layer4_2_1index,sum1,sum2,sum3)
layer4_2_1remove = index_sum.remove_index4_1_1(sum1,index1,sum2,index2,sum3,index3,layer4_2_1norm)
# layer4_2_1remove =[]
layer4_2_1remain = len(layer4_2_1norm) - len(layer4_2_1remove)
cfg_before.append(len(layer4_2_1norm))
remain.append(layer4_2_1remain)
remove_total.append(layer4_2_1remove)

# layer4_2_2  (128 -> 248)
layer4_2_2norm = np.sum(layer4_2_2norm, axis=(0,))
sum1, sum2, sum3 = index_sum.index_sum(layer4_2_2similar)
index1, index2, index3 = index_sum.index_cut(layer4_2_2index,sum1,sum2,sum3)
layer4_2_2remove = index_sum.remove_index4_1_1(sum1,index1,sum2,index2,sum3,index3,layer4_2_2norm)
# layer4_2_2remove =[]
layer4_2_2remain = len(layer4_2_2norm) - len(layer4_2_2remove)
cfg_before.append(len(layer4_2_2norm))
remain.append(layer4_2_2remain)
remove_total.append(layer4_2_2remove)

# layer4_2_3
cfg_before.append(len(layer4_3_3norm))
layer4_2_3remove = layer4_3_3remove
layer4_2_3remain = layer4_3_3remain
remain.append(layer4_2_3remain)
remove_total.append(layer4_2_3remove)

# layer4_3_1  (128 -> 248)
layer4_3_1norm = np.sum(layer4_3_1norm, axis=(0,))
sum1, sum2, sum3 = index_sum.index_sum(layer4_3_1similar)
index1, index2, index3 = index_sum.index_cut(layer4_3_1index,sum1,sum2,sum3)
layer4_3_1remove = index_sum.remove_index4_1_1(sum1,index1,sum2,index2,sum3,index3,layer4_3_1norm)
# layer4_3_1remove =[]
layer4_3_1remain = len(layer4_3_1norm) - len(layer4_3_1remove)
cfg_before.append(len(layer4_3_1norm))
remain.append(layer4_3_1remain)
remove_total.append(layer4_3_1remove)

# layer4_3_2  (128 -> 248)
layer4_3_2norm = np.sum(layer4_3_2norm, axis=(0,))
sum1, sum2, sum3 = index_sum.index_sum(layer4_3_2similar)
index1, index2, index3 = index_sum.index_cut(layer4_3_2index,sum1,sum2,sum3)
layer4_3_2remove = index_sum.remove_index4_1_1(sum1,index1,sum2,index2,sum3,index3,layer4_3_2norm)
# layer4_3_2remove =[]
layer4_3_2remain = len(layer4_3_2norm) - len(layer4_3_2remove)
cfg_before.append(len(layer4_3_2norm))
remain.append(layer4_3_2remain)
remove_total.append(layer4_3_2remove)

# layer4_3_3  (128 -> 248)
cfg_before.append(len(layer4_3_3norm))
remain.append(layer4_3_3remain)
remove_total.append(layer4_3_3remove)


# cfg_before.append(len(layer4_3_3norm))
# remain.append(layer4_3_3remain)
# remove_total.append(layer4_3_3remove)

print('cfg_before',cfg_before)
print('remain',remain)
# print(len(remain))
# #
remove_total = np.array(remove_total)
np.save("./layer_remove_index/remove_total.npy", remove_total)
remain = np.array(remain)
np.save("./layer_remove_index/remain.npy", remain)
cfg_before = np.array(cfg_before)
np.save("./layer_remove_index/cfg_before.npy", cfg_before)