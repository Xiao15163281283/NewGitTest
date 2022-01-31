import torch
import mobilenet_v2_ts
from torchvision import transforms
import load
from torch.autograd import Variable
from thop import profile
import conv_remove
import cal
import fenlei
import index_sum
import torch.nn as nn
from operator import itemgetter
import numpy as np

model = mobilenet_v2_ts.MobileNetV2()
checkpoint = torch.load('./0.7031model/model_0.7031.pth.tar')
model.load_state_dict(checkpoint['state_dict'])
best_prec1 = checkpoint['best_prec1']
samplesDataset = load.samplesDataset('./data/train_samples',
                                     transform=transforms.Compose([
                                         transforms.RandomCrop(32, padding=4),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                     ]))
data_loader = torch.utils.data.DataLoader(samplesDataset, batch_size=128, shuffle=False)

# print(len(samplesDataset))
init_seed = 1
torch.cuda.manual_seed(init_seed)

model.eval()
model = model.cuda()

batch_size = 128

conv_1_norm=[]
conv_2_norm=[]
conv_3_norm=[]
conv_4_norm=[]
conv_5_norm=[]
conv_6_norm=[]
conv_7_norm=[]
conv_8_norm=[]
j=0
for data, target in data_loader:

    print('j',j)
    data, target = data.cuda(), target.cuda()
    with torch.no_grad():
        data, target = Variable(data), Variable(target)
        output, conv_1, conv_2, conv_3, conv_4, conv_5, conv_6, conv_7, conv_8 = model(
            data)

        L1_norm1_1 = fenlei.L1(conv_1, batch_size)
        conv_1_norm.append(L1_norm1_1)

        L1_norm1_2 = fenlei.L1(conv_2, batch_size)
        conv_2_norm.append(L1_norm1_2)

        L1_norm2_1 = fenlei.L1(conv_3, batch_size)
        conv_3_norm.append(L1_norm2_1)

        L1_norm2_2 = fenlei.L1(conv_4, batch_size)
        conv_4_norm.append(L1_norm2_2)

        L1_norm3_1 = fenlei.L1(conv_5, batch_size)
        conv_5_norm.append(L1_norm3_1)

        L1_norm3_2 = fenlei.L1(conv_6, batch_size)
        conv_6_norm.append(L1_norm3_2)

        L1_norm3_3 = fenlei.L1(conv_7, batch_size)
        conv_7_norm.append(L1_norm3_3)

        L1_norm4_1 = fenlei.L1(conv_8, batch_size)
        conv_8_norm.append(L1_norm4_1)

        j += 1

index_1 = fenlei.cal_index1_1(conv_1_norm)
index_2 = fenlei.cal_index1_2(conv_2_norm)
index_3 = fenlei.cal_index2_1(conv_3_norm)
index_4 = fenlei.cal_index2_2(conv_4_norm)
index_5 = fenlei.cal_index3_1(conv_5_norm)
index_6 = fenlei.cal_index3_2(conv_6_norm)
index_7 = fenlei.cal_index3_3(conv_7_norm)
index_8 = fenlei.cal_index4_1(conv_8_norm)

similar_1 =[]
similar_2 =[]
similar_3 =[]
similar_4 =[]
similar_5 =[]
similar_6 =[]
similar_7 =[]
similar_8 =[]
j=0
for data, target in data_loader:

    print('j',j)
    data, target = data.cuda(), target.cuda()
    with torch.no_grad():
        data, target = Variable(data), Variable(target)
        output, conv_1, conv_2, conv_3, conv_4, conv_5, conv_6, conv_7, conv_8 = model(
            data)

        conv_1 = conv_1.permute(1, 0, 2, 3)
        new_conv_1 = conv_1.reshape(conv_1.shape[0], -1)
        # print(new_conv1_1.shape)
        conv_2 = conv_2.permute(1, 0, 2, 3)
        new_conv_2 = conv_2.reshape(conv_2.shape[0], -1)
        #
        conv_3 = conv_3.permute(1, 0, 2, 3)
        new_conv_3 = conv_3.reshape(conv_3.shape[0], -1)

        conv_4 = conv_4.permute(1, 0, 2, 3)
        new_conv_4 = conv_4.reshape(conv_4.shape[0], -1)

        conv_5 = conv_5.permute(1, 0, 2, 3)
        new_conv_5 = conv_5.reshape(conv_5.shape[0], -1)
#
        conv_6 = conv_6.permute(1, 0, 2, 3)
        new_conv_6 = conv_6.reshape(conv_6.shape[0], -1)
#
        conv_7 = conv_7.permute(1, 0, 2, 3)
        new_conv_7 = conv_7.reshape(conv_7.shape[0], -1)
        #
        conv_8 = conv_8.permute(1, 0, 2, 3)
        new_conv_8 = conv_8.reshape(conv_8.shape[0], -1)


        new_s1_1, index1_1_1 = cal.Similarity(index_1,new_conv_1)
        new_s1_2, index1_1_2 = cal.Similarity(index_2, new_conv_2)
        # print('index2_1',index2_1)
        new_s2_1, index1_2_1 = cal.Similarity(index_3, new_conv_3)
        new_s2_2, index1_2_2 = cal.Similarity(index_4, new_conv_4)
        new_s3_1, index1_3_1 = cal.Similarity(index_5, new_conv_5)
        new_s3_2, index1_3_2 = cal.Similarity(index_6, new_conv_6)
        new_s3_3, index1_3_3 = cal.Similarity(index_7, new_conv_7)
        new_s4_1, index1_4_1 = cal.Similarity(index_8, new_conv_8)

        similar_1.extend(new_s1_1)
        similar_2.extend(new_s1_2)
        similar_3.extend(new_s2_1)
        similar_4.extend(new_s2_2)
        similar_5.extend(new_s3_1)
        similar_6.extend(new_s3_2)
        similar_7.extend(new_s3_3)
        similar_8.extend(new_s4_1)
#
#         # break
        j += 1


similar_1 = np.array(similar_1)
index_1 = np.array(index1_1_1)
# print('new_index1_1',index1_1)
L1_norm_1 = np.array(conv_1_norm)
np.save("./cal_similar/similar_1.npy", similar_1)
np.save("./cal_similar/index_1.npy", index_1)
np.save("./cal_similar/L1_norm_1.npy", L1_norm_1)
similar_2 = np.array(similar_2)
index_2 = np.array(index1_1_2)

L1_norm_2 = np.array(conv_2_norm)
np.save("./cal_similar/similar_2.npy", similar_2)
np.save("./cal_similar/index_2.npy", index_2)
np.save("./cal_similar/L1_norm_2.npy", L1_norm_2)

similar_3 = np.array(similar_3)
index_3 = np.array(index1_2_1)
L1_norm_3 = np.array(conv_3_norm)
np.save("./cal_similar/similar_3.npy", similar_3)
np.save("./cal_similar/index_3.npy", index_3)
np.save("./cal_similar/L1_norm_3.npy", L1_norm_3)


similar_4 = np.array(similar_4)
index_4 = np.array(index1_2_2)
L1_norm_4 = np.array(conv_4_norm)
np.save("./cal_similar/similar_4.npy", similar_4)
np.save("./cal_similar/index_4.npy", index_4)
np.save("./cal_similar/L1_norm_4.npy", L1_norm_4)

similar_5 = np.array(similar_5)
index_5 = np.array(index1_3_1)
L1_norm_5 = np.array(conv_5_norm)
np.save("./cal_similar/similar_5.npy", similar_5)
np.save("./cal_similar/index_5.npy", index_5)
np.save("./cal_similar/L1_norm_5.npy", L1_norm_5)

similar_6 = np.array(similar_6)
index_6 = np.array(index1_3_2)
L1_norm_6 = np.array(conv_6_norm)
np.save("./cal_similar/similar_6.npy", similar_6)
np.save("./cal_similar/index_6.npy", index_6)
np.save("./cal_similar/L1_norm_6.npy", L1_norm_6)
similar_7 = np.array(similar_7)
index_7 = np.array(index1_3_3)
L1_norm_7 = np.array(conv_7_norm)
np.save("./cal_similar/similar_7.npy", similar_7)
np.save("./cal_similar/index_7.npy", index_7)
np.save("./cal_similar/L1_norm_7.npy", L1_norm_7)
similar_8= np.array(similar_8)
index_8 = np.array(index1_4_1)
L1_norm_8 = np.array(conv_8_norm)
np.save("./cal_similar/similar_8.npy", similar_8)
np.save("./cal_similar/index_8.npy", index_8)
np.save("./cal_similar/L1_norm_8.npy", L1_norm_8)