import numpy as np
import index_sum
# conv1_1
similar1_1 = np.load("./cal_similar/similar1_1.npy",allow_pickle=True)
similar1_1 = similar1_1.tolist()
# print('similar1_1',similar1_1)
index1_1 = np.load("./cal_similar/index1_1.npy",allow_pickle=True)
index1_1 = index1_1.tolist()
L1_norm1_1 = np.load("./cal_similar/L1_norm1_1.npy",allow_pickle=True)
L1_norm1_1 = L1_norm1_1.tolist()

# conv1_2
similar1_2 = np.load("./cal_similar/similar1_2.npy",allow_pickle=True)
similar1_2 = similar1_2.tolist()
index1_2 = np.load("./cal_similar/index1_2.npy",allow_pickle=True)
index1_2 = index1_2.tolist()
L1_norm1_2 = np.load("./cal_similar/L1_norm1_2.npy",allow_pickle=True)
L1_norm1_2 = L1_norm1_2.tolist()
# print('index1_2',index1_2)

# conv2_1
similar2_1 = np.load("./cal_similar/similar2_1.npy",allow_pickle=True)
similar2_1 = similar2_1.tolist()
index2_1 = np.load("./cal_similar/index2_1.npy",allow_pickle=True)
index2_1 = index2_1.tolist()
L1_norm2_1 = np.load("./cal_similar/L1_norm2_1.npy", allow_pickle=True)
L1_norm2_1 = L1_norm2_1.tolist()
# print('index2_1',index2_1)

# conv2_2
similar2_2 = np.load("./cal_similar/similar2_2.npy",allow_pickle=True)
similar2_2 = similar2_2.tolist()
index2_2 = np.load("./cal_similar/index2_2.npy",allow_pickle=True)
index2_2 = index2_2.tolist()
L1_norm2_2 = np.load("./cal_similar/L1_norm2_2.npy", allow_pickle=True)
L1_norm2_2 = L1_norm2_2.tolist()


# conv3_1
similar3_1 = np.load("./cal_similar/similar3_1.npy",allow_pickle=True)
similar3_1 = similar3_1.tolist()
index3_1 = np.load("./cal_similar/index3_1.npy",allow_pickle=True)
index3_1 = index3_1.tolist()
L1_norm3_1 = np.load("./cal_similar/L1_norm3_1.npy", allow_pickle=True)
L1_norm3_1 = L1_norm3_1.tolist()

# conv3_2
similar3_2 = np.load("./cal_similar/similar3_2.npy",allow_pickle=True)
similar3_2 = similar3_2.tolist()
index3_2 = np.load("./cal_similar/index3_2.npy",allow_pickle=True)
index3_2 = index3_2.tolist()
L1_norm3_2 = np.load("./cal_similar/L1_norm3_2.npy", allow_pickle=True)
L1_norm3_2 = L1_norm3_2.tolist()

# conv3_3
similar3_3 = np.load("./cal_similar/similar3_3.npy",allow_pickle=True)
similar3_3 = similar3_3.tolist()
index3_3 = np.load("./cal_similar/index3_3.npy",allow_pickle=True)
index3_3 = index3_3.tolist()
L1_norm3_3 = np.load("./cal_similar/L1_norm3_3.npy", allow_pickle=True)
L1_norm3_3 = L1_norm3_3.tolist()

# conv4_1
similar4_1 = np.load("./cal_similar/similar4_1.npy",allow_pickle=True)
similar4_1 = similar4_1.tolist()
index4_1 = np.load("./cal_similar/index4_1.npy",allow_pickle=True)
index4_1 = index4_1.tolist()
L1_norm4_1 = np.load("./cal_similar/L1_norm4_1.npy", allow_pickle=True)
L1_norm4_1 = L1_norm4_1.tolist()

# conv4_2
similar4_2 = np.load("./cal_similar/similar4_2.npy",allow_pickle=True)
similar4_2 = similar4_2.tolist()
index4_2 = np.load("./cal_similar/index4_2.npy",allow_pickle=True)
index4_2 = index4_2.tolist()
L1_norm4_2 = np.load("./cal_similar/L1_norm4_2.npy", allow_pickle=True)
L1_norm4_2 = L1_norm4_2.tolist()

# conv4_3
similar4_3 = np.load("./cal_similar/similar4_3.npy",allow_pickle=True)
similar4_3 = similar4_3.tolist()
index4_3 = np.load("./cal_similar/index4_3.npy",allow_pickle=True)
index4_3 = index4_3.tolist()
L1_norm4_3 = np.load("./cal_similar/L1_norm4_3.npy", allow_pickle=True)
L1_norm4_3 = L1_norm4_3.tolist()

# conv5_1
similar5_1 = np.load("./cal_similar/similar5_1.npy",allow_pickle=True)
similar5_1 = similar5_1.tolist()
index5_1 = np.load("./cal_similar/index5_1.npy",allow_pickle=True)
index5_1 = index5_1.tolist()
L1_norm5_1 = np.load("./cal_similar/L1_norm5_1.npy", allow_pickle=True)
L1_norm5_1 = L1_norm5_1.tolist()

# conv5_2
similar5_2 = np.load("./cal_similar/similar5_2.npy",allow_pickle=True)
similar5_2 = similar5_2.tolist()
index5_2 = np.load("./cal_similar/index5_2.npy",allow_pickle=True)
index5_2 = index5_2.tolist()
L1_norm5_2 = np.load("./cal_similar/L1_norm5_2.npy", allow_pickle=True)
L1_norm5_2 = L1_norm5_2.tolist()

# conv5_3
similar5_3 = np.load("./cal_similar/similar5_3.npy",allow_pickle=True)
similar5_3 = similar5_3.tolist()
index5_3 = np.load("./cal_similar/index5_3.npy",allow_pickle=True)
index5_3 = index5_3.tolist()
L1_norm5_3 = np.load("./cal_similar/L1_norm5_3.npy", allow_pickle=True)
L1_norm5_3 = L1_norm5_3.tolist()


# linear1
linear_norm1 = np.load("./cal_similar/linear_norm_1.npy", allow_pickle=True)
linear_norm1 = linear_norm1.tolist()

# linear2
linear_norm2 = np.load("./cal_similar/linear_norm_2.npy", allow_pickle=True)
linear_norm2 = linear_norm2.tolist()

# conv1_1
L1_norm1_1 = np.sum(L1_norm1_1,axis=(0,))/40
remain =[]
remove_total = []
sum1, sum2, sum3 = index_sum.index_sum(similar1_1)
index1,index2,index3 = index_sum.index_cut(index1_1,sum1,sum2,sum3)
remove1_1 = index_sum.remove_index(sum1,index1,sum2,index2,sum3,index3,L1_norm1_1)
# remove1_1 = []
remove_total.append(remove1_1)
# print('remove1_1',remove1_1)
# print('len(remove1_1)',len(remove1_1))
remain1_1 = 64-len(remove1_1)
print('remain1_1',remain1_1)
remain.append(remain1_1)
# input()
# conv1_2
L1_norm1_2 = np.sum(L1_norm1_2,axis=(0,))/40
# print('L1_norm1_2',L1_norm1_2)
# input()
sum1_2_1, sum1_2_2, sum1_2_3 = index_sum.index_sum(similar1_2)
index1_2_1,index1_2_2,index1_2_3 = index_sum.index_cut(index1_2,sum1_2_1,sum1_2_2,sum1_2_3)
remove1_2 = index_sum.remove_index(sum1_2_1,index1_2_1,sum1_2_2,index1_2_2,sum1_2_3,index1_2_3,L1_norm1_2)
# remove1_2=[]
remove_total.append(remove1_2)
print('remove1_2',remove1_2)
remain1_2 = 64-len(remove1_2)
print('remain1_2',remain1_2)
remain.append(remain1_2)
print('remain',remain)

# conv2_1
L1_norm2_1 = np.sum(L1_norm2_1,axis=(0,))/40
# print('L1_norm2_1',L1_norm2_1)
sum1_2_1, sum1_2_2, sum1_2_3 = index_sum.index_sum(similar2_1)
index1_2_1,index1_2_2,index1_2_3 = index_sum.index_cut(index2_1,sum1_2_1,sum1_2_2,sum1_2_3)
remove2_1 = index_sum.remove_index(sum1_2_1,index1_2_1,sum1_2_2,index1_2_2,sum1_2_3,index1_2_3,L1_norm2_1)
# remove2_1=[]
remove_total.append(remove2_1)
print('remove2_1',remove2_1)
remain2_1 = 128-len(remove2_1)
print('remain2_1',remain2_1)
remain.append(remain2_1)
# input()
# conv2_2
L1_norm2_2 = np.sum(L1_norm2_2,axis=(0,))/40
sum2_2_1, sum2_2_2, sum2_2_3 = index_sum.index_sum(similar2_2)
index2_2_1,index2_2_2,index2_2_3 = index_sum.index_cut(index2_2,sum2_2_1,sum2_2_2,sum2_2_3)
remove2_2 = index_sum.remove_index(sum2_2_1,index2_2_1,sum2_2_2,index2_2_2,sum2_2_3,index2_2_3,L1_norm2_2)
# remove2_2=[]
remove_total.append(remove2_2)
print('remove2_2',remove2_2)
remain2_2 = 128-len(remove2_2)
print('remain2_2',remain2_2)
remain.append(remain2_2)
# #
# conv3_1
L1_norm3_1 = np.sum(L1_norm3_1,axis=(0,))/40
sum2_2_1, sum2_2_2, sum2_2_3 = index_sum.index_sum(similar3_1)
index2_2_1,index2_2_2,index2_2_3 = index_sum.index_cut(index3_1,sum2_2_1,sum2_2_2,sum2_2_3)
remove3_1 = index_sum.remove_index(sum2_2_1,index2_2_1,sum2_2_2,index2_2_2,sum2_2_3,index2_2_3,L1_norm3_1)
# remove3_1 =[]
remove_total.append(remove3_1)
print('remove3_1',remove3_1)
# print('len(remove3_1)',len(remove3_1))
remain3_1 = 256-len(remove3_1)
print('remain3_1',remain3_1)
remain.append(remain3_1)
#
# conv3_2
L1_norm3_2 = np.sum(L1_norm3_2,axis=(0,))/40
sum2_2_1, sum2_2_2, sum2_2_3 = index_sum.index_sum(similar3_2)
index2_2_1,index2_2_2,index2_2_3 = index_sum.index_cut(index3_2,sum2_2_1,sum2_2_2,sum2_2_3)
remove3_2 = index_sum.remove_index(sum2_2_1,index2_2_1,sum2_2_2,index2_2_2,sum2_2_3,index2_2_3,L1_norm3_2)
# remove3_2 =[]
remove_total.append(remove3_2)
# print('len(remove3_2)',len(remove3_2))
remain3_2 = 256-len(remove3_2)
print('remain3_2',remain3_2)
remain.append(remain3_2)

# conv3_3
L1_norm3_3 = np.sum(L1_norm3_3,axis=(0,))/40
sum2_2_1, sum2_2_2, sum2_2_3 = index_sum.index_sum(similar3_3)
index2_2_1,index2_2_2,index2_2_3 = index_sum.index_cut(index3_3,sum2_2_1,sum2_2_2,sum2_2_3)
remove3_3 = index_sum.remove_index(sum2_2_1,index2_2_1,sum2_2_2,index2_2_2,sum2_2_3,index2_2_3,L1_norm3_3)
# remove3_3 =[]
remove_total.append(remove3_3)
print('len(remove3_3)',len(remove3_3))
remain3_3 = 256-len(remove3_3)
print('remain3_3',remain3_3)
remain.append(remain3_3)

# conv4_1
L1_norm4_1 = np.sum(L1_norm4_1,axis=(0,))/40
sum2_2_1, sum2_2_2, sum2_2_3 = index_sum.index_sum(similar4_1)
index2_2_1,index2_2_2,index2_2_3 = index_sum.index_cut(index4_1,sum2_2_1,sum2_2_2,sum2_2_3)
remove4_1 = index_sum.remove_index(sum2_2_1,index2_2_1,sum2_2_2,index2_2_2,sum2_2_3,index2_2_3,L1_norm4_1)
# remove4_1 =[]
remove_total.append(remove4_1)
# print('len(remove4_1)',len(remove4_1))
remain4_1 = 512-len(remove4_1)
print('remain4_1',remain4_1)
remain.append(remain4_1)

# conv4_2
L1_norm4_2 = np.sum(L1_norm4_2,axis=(0,))/40
# print('L1_norm4_2',L1_norm4_2)
# input()
sum2_2_1, sum2_2_2, sum2_2_3 = index_sum.index_sum(similar4_2)
index2_2_1,index2_2_2,index2_2_3 = index_sum.index_cut(index4_2,sum2_2_1,sum2_2_2,sum2_2_3)
remove4_2 = index_sum.remove_index(sum2_2_1,index2_2_1,sum2_2_2,index2_2_2,sum2_2_3,index2_2_3,L1_norm4_2)
# remove4_2 =[]
remove_total.append(remove4_2)
# print('len(remove4_2)',len(remove4_2))
remain4_2 = 512-len(remove4_2)
print('remain4_2',remain4_2)
remain.append(remain4_2)

# conv4_3
L1_norm4_3 = np.sum(L1_norm4_3,axis=(0,))/40
sum2_2_1, sum2_2_2, sum2_2_3 = index_sum.index_sum(similar4_3)
index2_2_1,index2_2_2,index2_2_3 = index_sum.index_cut(index4_3,sum2_2_1,sum2_2_2,sum2_2_3)
remove4_3 = index_sum.remove_index(sum2_2_1,index2_2_1,sum2_2_2,index2_2_2,sum2_2_3,index2_2_3,L1_norm4_3)
# remove4_3 =[]
remove_total.append(remove4_3)
# print('len(remove4_3)',len(remove4_3))
remain4_3 = 512-len(remove4_3)
print('remain4_3',remain4_3)
remain.append(remain4_3)

# conv5_1
L1_norm5_1 = np.sum(L1_norm5_1,axis=(0,))/40
sum2_2_1, sum2_2_2, sum2_2_3 = index_sum.index_sum(similar5_1)
index2_2_1,index2_2_2,index2_2_3 = index_sum.index_cut(index5_1,sum2_2_1,sum2_2_2,sum2_2_3)
remove5_1 = index_sum.remove_index(sum2_2_1,index2_2_1,sum2_2_2,index2_2_2,sum2_2_3,index2_2_3,L1_norm5_1)
# remove5_1 =[]
# arg_5_1 = np.argsort(L1_norm5_1)
# new_l5_1 = sorted(L1_norm5_1)
# print('arg_5_1',arg_5_1)
# print('new_l5_1',new_l5_1)
# remove5_1 = index_sum.cal_5_1(arg_5_1,new_l5_1)
remove_total.append(remove5_1)
print('len(remove5_1)',len(remove5_1))
remain5_1 = 512-len(remove5_1)
print('remain5_1',remain5_1)
remain.append(remain5_1)

# conv5_2
L1_norm5_2 = np.sum(L1_norm5_2,axis=(0,))/40
sum2_2_1, sum2_2_2, sum2_2_3 = index_sum.index_sum(similar5_2)
index2_2_1,index2_2_2,index2_2_3 = index_sum.index_cut(index5_2,sum2_2_1,sum2_2_2,sum2_2_3)
remove5_2 = index_sum.remove_index(sum2_2_1,index2_2_1,sum2_2_2,index2_2_2,sum2_2_3,index2_2_3,L1_norm5_2)
# remove5_2=[]
# arg_5_2 = np.argsort(L1_norm5_2)
# new_l5_2 = sorted(L1_norm5_2)
# remove5_2 = index_sum.cal_5_2(arg_5_2,new_l5_2)
remove_total.append(remove5_2)
remain5_2 = 512-len(remove5_2)
print('remain5_2',remain5_2)
remain.append(remain5_2)

# conv5_3
L1_norm5_3 = np.sum(L1_norm5_3,axis=(0,))/40
print('L1_norm5_3',L1_norm5_3)
sum2_2_1, sum2_2_2, sum2_2_3 = index_sum.index_sum(similar5_3)
index2_2_1,index2_2_2,index2_2_3 = index_sum.index_cut(index5_3,sum2_2_1,sum2_2_2,sum2_2_3)
remove5_3 = index_sum.remove_index(sum2_2_1,index2_2_1,sum2_2_2,index2_2_2,sum2_2_3,index2_2_3,L1_norm5_3)
# remove5_3 =[]
# arg_5_3 = np.argsort(L1_norm5_3)
# new_l5_3 = sorted(L1_norm5_3)
# remove5_3 = index_sum.cal_5_3(arg_5_3,new_l5_3)
remove_total.append(remove5_3)
# print('len(remove5_3)',len(remove5_3))
remain5_3 = 512-len(remove5_3)
print('remain5_3',remain5_3)
remain.append(remain5_3)

#linear1
linear_norm1_sum = np.sum(linear_norm1,axis=(0,))/40
arg_linear1 = np.argsort(linear_norm1_sum) # 返回升序的索引值
remove_linear1 = arg_linear1[0:2048]
# remove_linear1=[]
remove_total.append(remove_linear1)
remain_linear1 = 4096-len(remove_linear1)
print('remain_linear1',remain_linear1)
remain.append(remain_linear1)

#linear2
linear_norm2_sum = np.sum(linear_norm2,axis=(0,))/40
arg_linear2 = np.argsort(linear_norm2_sum)
remove_linear2 = arg_linear1[0:2048]
# remove_linear2=[]
remove_total.append(remove_linear2)
remain_linear2 = 4096-len(remove_linear2)
remain.append(remain_linear2)
# print('remain_linear2',remain_linear2)
print('remain',remain)
#
print('remove_total',remove_total)
remove_total = np.array(remove_total)
np.save("./remove_index/remove_total.npy", remove_total)
remain = np.array(remain)
np.save("./remove_index/remain.npy", remain)
print('remain',remain)