import numpy as np
import index_sum
# conv1_1
similar_1 = np.load("./cal_similar/similar_1.npy",allow_pickle=True)
similar_1 = similar_1.tolist()
# print('similar1_1',similar1_1)
index_1 = np.load("./cal_similar/index_1.npy",allow_pickle=True)
index_1 = index_1.tolist()
L1_norm_1 = np.load("./cal_similar/L1_norm_1.npy",allow_pickle=True)
L1_norm_1 = L1_norm_1.tolist()

# conv2
similar_2 = np.load("./cal_similar/similar_2.npy",allow_pickle=True)
similar_2 = similar_2.tolist()
index_2 = np.load("./cal_similar/index_2.npy",allow_pickle=True)
index_2 = index_2.tolist()
L1_norm_2 = np.load("./cal_similar/L1_norm_2.npy",allow_pickle=True)
L1_norm_2 = L1_norm_2.tolist()

# conv3
similar_3 = np.load("./cal_similar/similar_3.npy",allow_pickle=True)
similar_3 = similar_3.tolist()
index_3 = np.load("./cal_similar/index_3.npy",allow_pickle=True)
index_3 = index_3.tolist()
L1_norm_3 = np.load("./cal_similar/L1_norm_3.npy",allow_pickle=True)
L1_norm_3 = L1_norm_3.tolist()

# conv4
similar_4 = np.load("./cal_similar/similar_4.npy",allow_pickle=True)
similar_4 = similar_4.tolist()
index_4 = np.load("./cal_similar/index_4.npy",allow_pickle=True)
index_4 = index_4.tolist()
L1_norm_4 = np.load("./cal_similar/L1_norm_4.npy",allow_pickle=True)
L1_norm_4 = L1_norm_4.tolist()

# conv5
similar_5 = np.load("./cal_similar1/similar_5.npy",allow_pickle=True)
similar_5 = similar_5.tolist()
index_5 = np.load("./cal_similar1/index_5.npy",allow_pickle=True)
index_5 = index_5.tolist()
L1_norm_5 = np.load("./cal_similar1/L1_norm_5.npy",allow_pickle=True)
L1_norm_5 = L1_norm_5.tolist()

# conv6
similar_6 = np.load("./cal_similar/similar_6.npy",allow_pickle=True)
similar_6 = similar_6.tolist()
index_6 = np.load("./cal_similar/index_6.npy",allow_pickle=True)
index_6 = index_6.tolist()
L1_norm_6 = np.load("./cal_similar/L1_norm_6.npy",allow_pickle=True)
L1_norm_6 = L1_norm_6.tolist()
# print(L1_norm_6)
# input()
# conv7
similar_7 = np.load("./cal_similar/similar_7.npy",allow_pickle=True)
similar_7 = similar_7.tolist()
index_7 = np.load("./cal_similar/index_7.npy",allow_pickle=True)
index_7 = index_7.tolist()
L1_norm_7 = np.load("./cal_similar/L1_norm_7.npy",allow_pickle=True)
L1_norm_7 = L1_norm_7.tolist()

# conv8
similar_8 = np.load("./cal_similar/similar_8.npy",allow_pickle=True)
similar_8 = similar_8.tolist()
index_8 = np.load("./cal_similar/index_8.npy",allow_pickle=True)
index_8 = index_8.tolist()
L1_norm_8 = np.load("./cal_similar/L1_norm_8.npy",allow_pickle=True)
L1_norm_8 = L1_norm_8.tolist()

# conv1_1bottleneck
L1_norm_1 = np.sum(L1_norm_1,axis=(0,))/40
remain =[]
remove_total = []
cfg = []
cfg.append(32)
cfg.append(32)
cfg.append(len(L1_norm_1))
remove_total.append([])
remove_total.append([])
sum1, sum2, sum3 = index_sum.index_sum(similar_1)
index1,index2,index3 = index_sum.index_cut(index_1,sum1,sum2,sum3)
remove_1 = index_sum.remove_index1(sum1,index1,sum2,index2,sum3,index3,L1_norm_1)
remove_total.append(remove_1)
remain_1 = len(L1_norm_1)-len(remove_1)
remain.append(32)
remain.append(32)
remain.append(remain_1)

# conv2
L1_norm_2 = np.sum(L1_norm_2,axis=(0,))/40
sum1_2_1, sum1_2_2, sum1_2_3 = index_sum.index_sum(similar_2)
index1_2_1,index1_2_2,index1_2_3 = index_sum.index_cut(index_2, sum1_2_1, sum1_2_2, sum1_2_3)
remove_2 = index_sum.remove_index2(sum1_2_1, index1_2_1, sum1_2_2, index1_2_2, sum1_2_3, index1_2_3, L1_norm_2)
# conv2_2bottleneck
cfg.append(6*len(L1_norm_1))
cfg.append(6*len(L1_norm_1))
cfg.append(len(L1_norm_2))
remove_2_1bottle = remove_1
remove_2_2bottle = remove_1
remove_total.append(remove_2_1bottle)
remove_total.append(remove_2_2bottle)
remove_total.append(remove_2)
remain.append(6*len(L1_norm_1)-len(remove_2_1bottle))
remain.append(6*len(L1_norm_1)-len(remove_2_2bottle))
remain.append(len(L1_norm_2)-len(remove_2))
# conv2_3bottleneck
cfg.append(6*len(L1_norm_2))
cfg.append(6*len(L1_norm_2))
cfg.append(len(L1_norm_2))
remove_3_1bottle = remove_2
remove_3_2bottle = remove_2
remove_total.append(remove_3_1bottle)
remove_total.append(remove_3_2bottle)
remove_total.append(remove_2)
remain.append(6*len(L1_norm_2)-len(remove_3_1bottle))
remain.append(6*len(L1_norm_2)-len(remove_3_2bottle))
remain.append(len(L1_norm_2)-len(remove_2))

# conv3
L1_norm_3 = np.sum(L1_norm_3,axis=(0,))/40
sum1_2_1, sum1_2_2, sum1_2_3 = index_sum.index_sum(similar_3)
index1_2_1,index1_2_2,index1_2_3 = index_sum.index_cut(index_3,sum1_2_1,sum1_2_2,sum1_2_3)
remove_3 = index_sum.remove_index3(sum1_2_1,index1_2_1,sum1_2_2,index1_2_2,sum1_2_3,index1_2_3,L1_norm_3)
#conv3_4bottleneck
cfg.append(6*len(L1_norm_2))
cfg.append(6*len(L1_norm_2))
cfg.append(len(L1_norm_3))
remove_4_1bottle = remove_2
remove_4_2bottle = remove_2
remove_total.append(remove_4_1bottle)
remove_total.append(remove_4_2bottle)
remove_total.append(remove_3)
remain.append(6*len(L1_norm_2)-len(remove_4_1bottle))
remain.append(6*len(L1_norm_2)-len(remove_4_2bottle))
remain.append(len(L1_norm_3)-len(remove_3))
# 3_5bottlenechneck
cfg.append(6*len(L1_norm_3))
cfg.append(6*len(L1_norm_3))
cfg.append(len(L1_norm_3))
remove_5_1bottle = remove_3
remove_5_2bottle = remove_3
remove_total.append(remove_5_1bottle)
remove_total.append(remove_5_2bottle)
remove_total.append(remove_3)
remain.append(6*len(L1_norm_3)-len(remove_5_1bottle))
remain.append(6*len(L1_norm_3)-len(remove_5_2bottle))
remain.append(len(L1_norm_3)-len(remove_3))

# 3_6bottlenechneck
cfg.append(6*len(L1_norm_3))
cfg.append(6*len(L1_norm_3))
cfg.append(len(L1_norm_3))
remove_6_1bottle = remove_3
remove_6_2bottle = remove_3
remove_total.append(remove_6_1bottle)
remove_total.append(remove_6_2bottle)
remove_total.append(remove_3)
remain.append(6*len(L1_norm_3)-len(remove_6_1bottle))
remain.append(6*len(L1_norm_3)-len(remove_6_2bottle))
remain.append(len(L1_norm_3)-len(remove_3))

# conv4
L1_norm_4 = np.sum(L1_norm_4,axis=(0,))/40
sum1_2_1, sum1_2_2, sum1_2_3 = index_sum.index_sum(similar_4)
index1_2_1,index1_2_2,index1_2_3 = index_sum.index_cut(index_4,sum1_2_1,sum1_2_2,sum1_2_3)
remove_4 = index_sum.remove_index4(sum1_2_1,index1_2_1,sum1_2_2,index1_2_2,sum1_2_3,index1_2_3,L1_norm_4)
# conv4_7bottleneck
cfg.append(6*len(L1_norm_3))
cfg.append(6*len(L1_norm_3))
cfg.append(len(L1_norm_4))
remove_7_1bottle = remove_3
print(len(remove_7_1bottle))
remove_7_2bottle = remove_3
remove_total.append(remove_7_1bottle)
remove_total.append(remove_7_2bottle)
remove_total.append(remove_4)
remain.append(6*len(L1_norm_3)-len(remove_7_1bottle))
remain.append(6*len(L1_norm_3)-len(remove_7_2bottle))
remain.append(len(L1_norm_4)-len(remove_4))
# conv4_8bottleneck
cfg.append(6*len(L1_norm_4))
cfg.append(6*len(L1_norm_4))
cfg.append(len(L1_norm_4))
remove_8_1bottle = remove_4
print(len(remove_8_1bottle))
remove_8_2bottle = remove_4
remove_total.append(remove_8_1bottle)
remove_total.append(remove_8_2bottle)
remove_total.append(remove_4)
remain.append(6*len(L1_norm_4)-len(remove_8_1bottle))
remain.append(6*len(L1_norm_4)-len(remove_8_2bottle))
remain.append(len(L1_norm_4)-len(remove_4))
# conv4_9bottleneck
cfg.append(6*len(L1_norm_4))
cfg.append(6*len(L1_norm_4))
cfg.append(len(L1_norm_4))
remove_9_1bottle = remove_4
remove_9_2bottle = remove_4
remove_total.append(remove_9_1bottle)
remove_total.append(remove_9_2bottle)
remove_total.append(remove_4)
remain.append(6*len(L1_norm_4)-len(remove_9_1bottle))
remain.append(6*len(L1_norm_4)-len(remove_9_2bottle))
remain.append(len(L1_norm_4)-len(remove_4))
# conv4_10bottleneck
cfg.append(6*len(L1_norm_4))
cfg.append(6*len(L1_norm_4))
cfg.append(len(L1_norm_4))
remove_10_1bottle = remove_4
remove_10_2bottle = remove_4
remove_total.append(remove_10_1bottle)
remove_total.append(remove_10_2bottle)
remove_total.append(remove_4)
remain.append(6*len(L1_norm_4)-len(remove_10_1bottle))
remain.append(6*len(L1_norm_4)-len(remove_10_2bottle))
remain.append(len(L1_norm_4)-len(remove_4))

# conv5
L1_norm_5 = np.sum(L1_norm_5,axis=(0,))/40
sum1_2_1, sum1_2_2, sum1_2_3 = index_sum.index_sum(similar_5)
index1_2_1,index1_2_2,index1_2_3 = index_sum.index_cut(index_5,sum1_2_1,sum1_2_2,sum1_2_3)
remove_5 = index_sum.remove_index5(sum1_2_1,index1_2_1,sum1_2_2,index1_2_2,sum1_2_3,index1_2_3,L1_norm_5)
# conv5_11bottleneck
cfg.append(6*len(L1_norm_4))
cfg.append(6*len(L1_norm_4))
cfg.append(len(L1_norm_5))
remove_11_1bottle = remove_4
remove_11_2bottle = remove_4
remove_total.append(remove_11_1bottle)
remove_total.append(remove_11_2bottle)
remove_total.append(remove_5)
remain.append(6*len(L1_norm_4)-len(remove_11_1bottle))
remain.append(6*len(L1_norm_4)-len(remove_11_2bottle))
remain.append(len(L1_norm_5)-len(remove_5))
# conv5_12bottleneck
cfg.append(6*len(L1_norm_5))
cfg.append(6*len(L1_norm_5))
cfg.append(len(L1_norm_5))
remove_12_1bottle = remove_5
remove_12_2bottle = remove_5
remove_total.append(remove_12_1bottle)
remove_total.append(remove_12_2bottle)
remove_total.append(remove_5)
remain.append(6*len(L1_norm_5)-len(remove_12_1bottle))
remain.append(6*len(L1_norm_5)-len(remove_12_2bottle))
remain.append(len(L1_norm_5)-len(remove_5))
# conv5_13bottleneck
cfg.append(6*len(L1_norm_5))
cfg.append(6*len(L1_norm_5))
cfg.append(len(L1_norm_5))
remove_13_1bottle = remove_5
remove_13_2bottle = remove_5
remove_total.append(remove_13_1bottle)
remove_total.append(remove_13_2bottle)
remove_total.append(remove_5)
remain.append(6*len(L1_norm_5)-len(remove_13_1bottle))
remain.append(6*len(L1_norm_5)-len(remove_13_2bottle))
remain.append(len(L1_norm_5)-len(remove_5))
# conv6
L1_norm_6 = np.sum(L1_norm_6,axis=(0,))/40
# print('L1_norm_6',L1_norm_6)
# input()
sum1_2_1, sum1_2_2, sum1_2_3 = index_sum.index_sum(similar_6)
index1_2_1,index1_2_2,index1_2_3 = index_sum.index_cut(index_6,sum1_2_1,sum1_2_2,sum1_2_3)
remove_6 = index_sum.remove_index6(sum1_2_1,index1_2_1,sum1_2_2,index1_2_2,sum1_2_3,index1_2_3,L1_norm_6)
# conv6_14bottleneck
cfg.append(6*len(L1_norm_5))
cfg.append(6*len(L1_norm_5))
cfg.append(len(L1_norm_6))
remove_14_1bottle = remove_5
remove_14_2bottle = remove_5
remove_total.append(remove_14_1bottle)
remove_total.append(remove_14_2bottle)
remove_total.append(remove_6)
remain.append(6*len(L1_norm_5)-len(remove_14_1bottle))
remain.append(6*len(L1_norm_5)-len(remove_14_2bottle))
remain.append(len(L1_norm_6)-len(remove_6))
# conv6_15bottleneck
cfg.append(6*len(L1_norm_6))
cfg.append(6*len(L1_norm_6))
cfg.append(len(L1_norm_6))
remove_15_1bottle = remove_6
remove_15_2bottle = remove_6
remove_total.append(remove_15_1bottle)
remove_total.append(remove_15_2bottle)
remove_total.append(remove_6)
remain.append(6*len(L1_norm_6)-len(remove_15_1bottle))
remain.append(6*len(L1_norm_6)-len(remove_15_2bottle))
remain.append(len(L1_norm_6)-len(remove_6))
# conv6_16bottleneck
cfg.append(6*len(L1_norm_6))
cfg.append(6*len(L1_norm_6))
cfg.append(len(L1_norm_6))
remove_16_1bottle = remove_6
remove_16_2bottle = remove_6
remove_total.append(remove_16_1bottle)
remove_total.append(remove_16_2bottle)
remove_total.append(remove_6)
remain.append(6*len(L1_norm_6)-len(remove_16_1bottle))
remain.append(6*len(L1_norm_6)-len(remove_16_2bottle))
remain.append(len(L1_norm_6)-len(remove_6))
#
# conv7
L1_norm_7 = np.sum(L1_norm_7,axis=(0,))/40
sum1_2_1, sum1_2_2, sum1_2_3 = index_sum.index_sum(similar_7)
index1_2_1,index1_2_2,index1_2_3 = index_sum.index_cut(index_7,sum1_2_1,sum1_2_2,sum1_2_3)
remove_7 = index_sum.remove_index7(sum1_2_1,index1_2_1,sum1_2_2,index1_2_2,sum1_2_3,index1_2_3,L1_norm_7)
# conv7_17bottleneck
cfg.append(6*len(L1_norm_6))
cfg.append(6*len(L1_norm_6))
cfg.append(len(L1_norm_7))
remove_17_1bottle = remove_6
remove_17_2bottle = remove_6
remove_total.append(remove_17_1bottle)
remove_total.append(remove_17_2bottle)
remove_total.append(remove_7)
remain.append(6*len(L1_norm_6)-len(remove_17_1bottle))
remain.append(6*len(L1_norm_6)-len(remove_17_2bottle))
remain.append(len(L1_norm_7)-len(remove_7))
# conv8
L1_norm_8 = np.sum(L1_norm_8,axis=(0,))/40
sum1_2_1, sum1_2_2, sum1_2_3 = index_sum.index_sum(similar_8)
index1_2_1,index1_2_2,index1_2_3 = index_sum.index_cut(index_8,sum1_2_1,sum1_2_2,sum1_2_3)
remove_8 = index_sum.remove_index8(sum1_2_1,index1_2_1,sum1_2_2,index1_2_2,sum1_2_3,index1_2_3,L1_norm_8)
cfg.append(len(L1_norm_8))
remove_total.append(remove_8)
remain.append(len(L1_norm_8)-len(remove_8))

print('cfg',cfg)
# print('remove_total',remove_total)
print('remain',remain)
cfg_before = np.array(cfg)
np.save("./bottle_remove/cfg_before.npy", cfg_before)
remove_total = np.array(remove_total)
np.save("./bottle_remove/remove_total.npy", remove_total)
remain = np.array(remain)
np.save("./bottle_remove/remain.npy", remain)
# print('remain',remain)