import torch
import resnet_mod
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

model = resnet_mod.Resnet50()
checkpoint = torch.load('./0.7786model/model_0.7786.pth.tar')
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

print(len(samplesDataset))
init_seed = 1
torch.cuda.manual_seed(init_seed)

model.eval()
model.cuda()

batch_size = 128
#layer1(7个)
layer1_1_1norm=[]
layer1_1_2norm=[]
layer1_2_1norm=[]
layer1_2_2norm=[]
layer1_3_1norm=[]
layer1_3_2norm=[]
layer1_3_3norm=[]
#layer2_1(9个)
layer2_1_1norm=[]
layer2_1_2norm=[]
layer2_2_1norm=[]
layer2_2_2norm=[]
layer2_3_1norm=[]
layer2_3_2norm=[]
layer2_4_1norm=[]
layer2_4_2norm=[]
layer2_4_3norm=[]
#layer3_1(13个)
layer3_1_1norm=[]
layer3_1_2norm=[]
layer3_2_1norm=[]
layer3_2_2norm=[]
layer3_3_1norm=[]
layer3_3_2norm=[]
layer3_4_1norm=[]
layer3_4_2norm=[]
layer3_5_1norm=[]
layer3_5_2norm=[]
layer3_6_1norm=[]
layer3_6_2norm=[]
layer3_6_3norm=[]
#layer4_1(7个)
layer4_1_1norm=[]
layer4_1_2norm=[]
layer4_2_1norm=[]
layer4_2_2norm=[]
layer4_3_1norm=[]
layer4_3_2norm=[]
layer4_3_3norm=[]
j=0
for data, target in data_loader:

    print('j',j)
    data, target = data.cuda(), target.cuda()
    with torch.no_grad():
        data, target = Variable(data), Variable(target)
        x,layer1_1_1,layer1_1_2,layer1_2_1,layer1_2_2,layer1_3_1,layer1_3_2,layer1_3_3,layer2_1_1,layer2_1_2,layer2_2_1,layer2_2_2,layer2_3_1,layer2_3_2,layer2_4_1,layer2_4_2,layer2_4_3,layer3_1_1,layer3_1_2,layer3_2_1,layer3_2_2,layer3_3_1,layer3_3_2,layer3_4_1,layer3_4_2,layer3_5_1,layer3_5_2,layer3_6_1,layer3_6_2,layer3_6_3,layer4_1_1,layer4_1_2,layer4_2_1,layer4_2_2,layer4_3_1,layer4_3_2,layer4_3_3, = model(data)
        # print(layer1_1_1.shape)
        L1_layer1_1_1 = fenlei.L1(layer1_1_1, batch_size)
        layer1_1_1norm.append(L1_layer1_1_1)
        L1_layer1_1_2 = fenlei.L1(layer1_1_2, batch_size)
        layer1_1_2norm.append(L1_layer1_1_2)

        L1_layer1_2_1 = fenlei.L1(layer1_2_1, batch_size)
        layer1_2_1norm.append(L1_layer1_2_1)

        L1_layer1_2_2 = fenlei.L1(layer1_2_2, batch_size)
        layer1_2_2norm.append(L1_layer1_2_2)

        L1_layer1_3_1 = fenlei.L1(layer1_3_1, batch_size)
        layer1_3_1norm.append(L1_layer1_3_1)
        L1_layer1_3_2 = fenlei.L1(layer1_3_2, batch_size)
        layer1_3_2norm.append(L1_layer1_3_2)
        L1_layer1_3_3 = fenlei.L1(layer1_3_3, batch_size)
        layer1_3_3norm.append(L1_layer1_3_3)

        L1_layer2_1_1 = fenlei.L1(layer2_1_1, batch_size)
        layer2_1_1norm.append(L1_layer2_1_1)
        L1_layer2_1_2 = fenlei.L1(layer2_1_2, batch_size)
        layer2_1_2norm.append(L1_layer2_1_2)

        L1_layer2_2_1 = fenlei.L1(layer2_2_1, batch_size)
        layer2_2_1norm.append(L1_layer2_2_1)
        L1_layer2_2_2 = fenlei.L1(layer2_2_2, batch_size)
        layer2_2_2norm.append(L1_layer2_2_2)

        L1_layer2_3_1 = fenlei.L1(layer2_3_1, batch_size)
        layer2_3_1norm.append(L1_layer2_3_1)
        L1_layer2_3_2 = fenlei.L1(layer2_3_2, batch_size)
        layer2_3_2norm.append(L1_layer2_3_2)

        L1_layer2_4_1 = fenlei.L1(layer2_4_1, batch_size)
        layer2_4_1norm.append(L1_layer2_4_1)
        L1_layer2_4_2 = fenlei.L1(layer2_4_2, batch_size)
        layer2_4_2norm.append(L1_layer2_4_2)
        L1_layer2_4_3 = fenlei.L1(layer2_4_3, batch_size)
        layer2_4_3norm.append(L1_layer2_4_3)

        L1_layer3_1_1 = fenlei.L1(layer3_1_1, batch_size)
        layer3_1_1norm.append(L1_layer3_1_1)
        L1_layer3_1_2 = fenlei.L1(layer3_1_2, batch_size)
        layer3_1_2norm.append(L1_layer3_1_2)

        L1_layer3_2_1 = fenlei.L1(layer3_2_1, batch_size)
        layer3_2_1norm.append(L1_layer3_2_1)
        L1_layer3_2_2 = fenlei.L1(layer3_2_2, batch_size)
        layer3_2_2norm.append(L1_layer3_2_2)

        L1_layer3_3_1 = fenlei.L1(layer3_3_1, batch_size)
        layer3_3_1norm.append(L1_layer3_3_1)
        L1_layer3_3_2 = fenlei.L1(layer3_3_2, batch_size)
        layer3_3_2norm.append(L1_layer3_3_2)

        L1_layer3_4_1 = fenlei.L1(layer3_4_1, batch_size)
        layer3_4_1norm.append(L1_layer3_4_1)
        L1_layer3_4_2 = fenlei.L1(layer3_4_2, batch_size)
        layer3_4_2norm.append(L1_layer3_4_2)

        L1_layer3_5_1 = fenlei.L1(layer3_5_1, batch_size)
        layer3_5_1norm.append(L1_layer3_5_1)
        L1_layer3_5_2 = fenlei.L1(layer3_5_2, batch_size)
        layer3_5_2norm.append(L1_layer3_5_2)

        L1_layer3_6_1 = fenlei.L1(layer3_6_1, batch_size)
        layer3_6_1norm.append(L1_layer3_6_1)
        L1_layer3_6_2 = fenlei.L1(layer3_6_2, batch_size)
        layer3_6_2norm.append(L1_layer3_6_2)
        L1_layer3_6_3 = fenlei.L1(layer3_6_3, batch_size)
        layer3_6_3norm.append(L1_layer3_6_3)

        L1_layer4_1_1 = fenlei.L1(layer4_1_1, batch_size)
        layer4_1_1norm.append(L1_layer4_1_1)
        L1_layer4_1_2 = fenlei.L1(layer4_1_2, batch_size)
        layer4_1_2norm.append(L1_layer4_1_2)

        L1_layer4_2_1 = fenlei.L1(layer4_2_1, batch_size)
        layer4_2_1norm.append(L1_layer4_2_1)
        L1_layer4_2_2 = fenlei.L1(layer4_2_2, batch_size)
        layer4_2_2norm.append(L1_layer4_2_2)

        L1_layer4_3_1 = fenlei.L1(layer4_3_1, batch_size)
        layer4_3_1norm.append(L1_layer4_3_1)
        L1_layer4_3_2 = fenlei.L1(layer4_3_2, batch_size)
        layer4_3_2norm.append(L1_layer4_3_2)
        L1_layer4_3_3 = fenlei.L1(layer4_3_3, batch_size)
        layer4_3_3norm.append(L1_layer4_3_3)

        j += 1

index1_1_1 = fenlei.cal_index1_1(layer1_1_1norm)
index1_1_2 = fenlei.cal_index1_1(layer1_1_2norm)

index1_2_1 = fenlei.cal_index1_1(layer1_2_1norm)
index1_2_2 = fenlei.cal_index1_1(layer1_2_2norm)

index1_3_1 = fenlei.cal_index1_1(layer1_3_1norm)
index1_3_2 = fenlei.cal_index1_1(layer1_3_2norm)
index1_3_3 = fenlei.cal_index2_1(layer1_3_3norm)

index2_1_1 = fenlei.cal_index2_1(layer2_1_1norm)
index2_1_2 = fenlei.cal_index2_1(layer2_1_2norm)

index2_2_1 = fenlei.cal_index2_1(layer2_2_1norm)
index2_2_2 = fenlei.cal_index2_1(layer2_2_2norm)

index2_3_1 = fenlei.cal_index2_1(layer2_3_1norm)
index2_3_2 = fenlei.cal_index2_1(layer2_3_2norm)

index2_4_1 = fenlei.cal_index2_1(layer2_4_1norm)
index2_4_2 = fenlei.cal_index2_1(layer2_4_2norm)
index2_4_3 = fenlei.cal_index3_1(layer2_4_3norm)

index3_1_1 = fenlei.cal_index2_1(layer3_1_1norm)
index3_1_2 = fenlei.cal_index2_1(layer3_1_2norm)
index3_2_1 = fenlei.cal_index2_1(layer3_2_1norm)
index3_2_2 = fenlei.cal_index2_1(layer3_2_2norm)

index3_3_1 = fenlei.cal_index2_1(layer3_3_1norm)
index3_3_2 = fenlei.cal_index2_1(layer3_3_2norm)

index3_4_1 = fenlei.cal_index2_1(layer3_4_1norm)
index3_4_2 = fenlei.cal_index2_1(layer3_4_2norm)

index3_5_1 = fenlei.cal_index2_1(layer3_5_1norm)
index3_5_2 = fenlei.cal_index2_1(layer3_5_2norm)

index3_6_1 = fenlei.cal_index2_1(layer3_6_1norm)
index3_6_2 = fenlei.cal_index2_1(layer3_6_2norm)
index3_6_3 = fenlei.cal_index4_1(layer3_6_3norm)

index4_1_1 = fenlei.cal_index3_1(layer4_1_1norm)
index4_1_2 = fenlei.cal_index3_1(layer4_1_2norm)
index4_2_1 = fenlei.cal_index3_1(layer4_2_1norm)
index4_2_2 = fenlei.cal_index3_1(layer4_2_2norm)

index4_3_1 = fenlei.cal_index3_1(layer4_3_1norm)
index4_3_2 = fenlei.cal_index3_1(layer4_3_2norm)

index4_3_3 = fenlei.cal_index5_1(layer4_3_3norm)

layer1_1_1similar=[]
layer1_1_2similar=[]
layer1_2_1similar=[]
layer1_2_2similar=[]
layer1_3_1similar=[]
layer1_3_2similar=[]
layer1_3_3similar=[]
#layer2_1(9个)
layer2_1_1similar=[]
layer2_1_2similar=[]
layer2_2_1similar=[]
layer2_2_2similar=[]
layer2_3_1similar=[]
layer2_3_2similar=[]
layer2_4_1similar=[]
layer2_4_2similar=[]
layer2_4_3similar=[]
#layer3_1(13个)
layer3_1_1similar=[]
layer3_1_2similar=[]
layer3_2_1similar=[]
layer3_2_2similar=[]
layer3_3_1similar=[]
layer3_3_2similar=[]
layer3_4_1similar=[]
layer3_4_2similar=[]
layer3_5_1similar=[]
layer3_5_2similar=[]
layer3_6_1similar=[]
layer3_6_2similar=[]
layer3_6_3similar=[]
#layer4_1(7个)
layer4_1_1similar=[]
layer4_1_2similar=[]
layer4_2_1similar=[]
layer4_2_2similar=[]
layer4_3_1similar=[]
layer4_3_2similar=[]
layer4_3_3similar=[]
j=0
for data, target in data_loader:

    print('j',j)
    data, target = data.cuda(), target.cuda()
    with torch.no_grad():
        data, target = Variable(data), Variable(target)
        x, layer1_1_1, layer1_1_2, layer1_2_1, layer1_2_2, layer1_3_1, layer1_3_2, layer1_3_3, layer2_1_1, layer2_1_2, layer2_2_1, layer2_2_2, layer2_3_1, layer2_3_2, layer2_4_1, layer2_4_2, layer2_4_3, layer3_1_1, layer3_1_2, layer3_2_1, layer3_2_2, layer3_3_1, layer3_3_2, layer3_4_1, layer3_4_2, layer3_5_1, layer3_5_2, layer3_6_1, layer3_6_2, layer3_6_3, layer4_1_1, layer4_1_2, layer4_2_1, layer4_2_2, layer4_3_1, layer4_3_2, layer4_3_3 = model(
            data)

        layer1_1_1 = layer1_1_1.permute(1, 0, 2, 3)
        new_layer1_1_1 = layer1_1_1.reshape(layer1_1_1.shape[0], -1)
        # print(new_conv1_1.shape)
        layer1_1_2 = layer1_1_2.permute(1, 0, 2, 3)
        new_layer1_1_2 = layer1_1_2.reshape(layer1_1_2.shape[0], -1)

        layer1_2_1 = layer1_2_1.permute(1, 0, 2, 3)
        new_layer1_2_1 = layer1_2_1.reshape(layer1_2_1.shape[0], -1)
        layer1_2_2 = layer1_2_2.permute(1, 0, 2, 3)
        new_layer1_2_2 = layer1_2_2.reshape(layer1_2_2.shape[0], -1)

        layer1_3_1 = layer1_3_1.permute(1, 0, 2, 3)
        new_layer1_3_1 = layer1_3_1.reshape(layer1_3_1.shape[0], -1)
        layer1_3_2 = layer1_3_2.permute(1, 0, 2, 3)
        new_layer1_3_2 = layer1_3_2.reshape(layer1_3_2.shape[0], -1)
        layer1_3_3 = layer1_3_3.permute(1, 0, 2, 3)
        new_layer1_3_3 = layer1_3_3.reshape(layer1_3_3.shape[0], -1)

        layer2_1_1 = layer2_1_1.permute(1, 0, 2, 3)
        new_layer2_1_1 = layer2_1_1.reshape(layer2_1_1.shape[0], -1)
        # print(new_conv1_1.shape)
        layer2_1_2 = layer2_1_2.permute(1, 0, 2, 3)
        new_layer2_1_2 = layer2_1_2.reshape(layer2_1_2.shape[0], -1)

        layer2_2_1 = layer2_2_1.permute(1, 0, 2, 3)
        new_layer2_2_1 = layer2_2_1.reshape(layer2_2_1.shape[0], -1)
        layer2_2_2 = layer2_2_2.permute(1, 0, 2, 3)
        new_layer2_2_2 = layer2_2_2.reshape(layer2_2_2.shape[0], -1)

        layer2_3_1 = layer2_3_1.permute(1, 0, 2, 3)
        new_layer2_3_1 = layer2_3_1.reshape(layer2_3_1.shape[0], -1)
        layer2_3_2 = layer2_3_2.permute(1, 0, 2, 3)
        new_layer2_3_2 = layer2_3_2.reshape(layer2_3_2.shape[0], -1)

        layer2_4_1 = layer2_4_1.permute(1, 0, 2, 3)
        new_layer2_4_1 = layer2_4_1.reshape(layer2_4_1.shape[0], -1)
        layer2_4_2 = layer2_4_2.permute(1, 0, 2, 3)
        new_layer2_4_2 = layer2_4_2.reshape(layer2_4_2.shape[0], -1)
        layer2_4_3 = layer2_4_3.permute(1, 0, 2, 3)
        new_layer2_4_3 = layer2_4_3.reshape(layer2_4_3.shape[0], -1)

        layer3_1_1 = layer3_1_1.permute(1, 0, 2, 3)
        new_layer3_1_1 = layer3_1_1.reshape(layer3_1_1.shape[0], -1)
        # print(new_conv1_1.shape)
        layer3_1_2 = layer3_1_2.permute(1, 0, 2, 3)
        new_layer3_1_2 = layer3_1_2.reshape(layer3_1_2.shape[0], -1)

        layer3_2_1 = layer3_2_1.permute(1, 0, 2, 3)
        new_layer3_2_1 = layer3_2_1.reshape(layer3_2_1.shape[0], -1)
        layer3_2_2 = layer3_2_2.permute(1, 0, 2, 3)
        new_layer3_2_2 = layer3_2_2.reshape(layer3_2_2.shape[0], -1)

        layer3_3_1 = layer3_3_1.permute(1, 0, 2, 3)
        new_layer3_3_1 = layer3_3_1.reshape(layer3_3_1.shape[0], -1)
        layer3_3_2 = layer3_3_2.permute(1, 0, 2, 3)
        new_layer3_3_2 = layer3_3_2.reshape(layer3_3_2.shape[0], -1)

        layer3_4_1 = layer3_4_1.permute(1, 0, 2, 3)
        new_layer3_4_1 = layer3_4_1.reshape(layer3_4_1.shape[0], -1)
        layer3_4_2 = layer3_4_2.permute(1, 0, 2, 3)
        new_layer3_4_2 = layer3_4_2.reshape(layer3_4_2.shape[0], -1)

        layer3_5_1 = layer3_5_1.permute(1, 0, 2, 3)
        new_layer3_5_1 = layer3_5_1.reshape(layer3_5_1.shape[0], -1)
        layer3_5_2 = layer3_5_2.permute(1, 0, 2, 3)
        new_layer3_5_2 = layer3_5_2.reshape(layer3_5_2.shape[0], -1)

        layer3_6_1 = layer3_6_1.permute(1, 0, 2, 3)
        new_layer3_6_1 = layer3_6_1.reshape(layer3_6_1.shape[0], -1)
        layer3_6_2 = layer3_6_2.permute(1, 0, 2, 3)
        new_layer3_6_2 = layer3_6_2.reshape(layer3_6_2.shape[0], -1)
        layer3_6_3 = layer3_6_3.permute(1, 0, 2, 3)
        new_layer3_6_3 = layer3_6_3.reshape(layer3_6_3.shape[0], -1)

        layer4_1_1 = layer4_1_1.permute(1, 0, 2, 3)
        new_layer4_1_1 = layer4_1_1.reshape(layer4_1_1.shape[0], -1)
        # print(new_conv1_1.shape)
        layer4_1_2 = layer4_1_2.permute(1, 0, 2, 3)
        new_layer4_1_2 = layer4_1_2.reshape(layer4_1_2.shape[0], -1)

        layer4_2_1 = layer4_2_1.permute(1, 0, 2, 3)
        new_layer4_2_1 = layer4_2_1.reshape(layer4_2_1.shape[0], -1)
        layer4_2_2 = layer4_2_2.permute(1, 0, 2, 3)
        new_layer4_2_2 = layer4_2_2.reshape(layer4_2_2.shape[0], -1)

        layer4_3_1 = layer4_3_1.permute(1, 0, 2, 3)
        new_layer4_3_1 = layer4_3_1.reshape(layer4_3_1.shape[0], -1)
        layer4_3_2 = layer4_3_2.permute(1, 0, 2, 3)
        new_layer4_3_2 = layer4_3_2.reshape(layer4_3_2.shape[0], -1)
        layer4_3_3 = layer4_3_3.permute(1, 0, 2, 3)
        new_layer4_3_3 = layer4_3_3.reshape(layer4_3_3.shape[0], -1)

        new_s1_1_1, layer1_1_1index = cal.Similarity(index1_1_1,new_layer1_1_1)
        new_s1_1_2, layer1_1_2index = cal.Similarity(index1_1_2, new_layer1_1_2)
        new_s1_2_1, layer1_2_1index = cal.Similarity(index1_2_1, new_layer1_2_1)
        new_s1_2_2, layer1_2_2index = cal.Similarity(index1_2_2, new_layer1_2_2)

        new_s1_3_1, layer1_3_1index = cal.Similarity(index1_3_1, new_layer1_3_1)
        new_s1_3_2, layer1_3_2index = cal.Similarity(index1_3_2, new_layer1_3_2)
        new_s1_3_3, layer1_3_3index = cal.Similarity(index1_3_3, new_layer1_3_3)

        new_s2_1_1, layer2_1_1index = cal.Similarity(index2_1_1, new_layer2_1_1)
        new_s2_1_2, layer2_1_2index = cal.Similarity(index2_1_2, new_layer2_1_2)

        new_s2_2_1, layer2_2_1index = cal.Similarity(index2_2_1, new_layer2_2_1)
        new_s2_2_2, layer2_2_2index = cal.Similarity(index2_2_2, new_layer2_2_2)

        new_s2_3_1, layer2_3_1index = cal.Similarity(index2_3_1, new_layer2_3_1)
        new_s2_3_2, layer2_3_2index = cal.Similarity(index2_3_2, new_layer2_3_2)

        new_s2_4_1, layer2_4_1index = cal.Similarity(index2_4_1, new_layer2_4_1)
        new_s2_4_2, layer2_4_2index = cal.Similarity(index2_4_2, new_layer2_4_2)
        new_s2_4_3, layer2_4_3index = cal.Similarity(index2_4_3, new_layer2_4_3)

        new_s3_1_1, layer3_1_1index = cal.Similarity(index3_1_1, new_layer3_1_1)
        new_s3_1_2, layer3_1_2index = cal.Similarity(index3_1_2, new_layer3_1_2)

        new_s3_2_1, layer3_2_1index = cal.Similarity(index3_2_1, new_layer3_2_1)
        new_s3_2_2, layer3_2_2index = cal.Similarity(index3_2_2, new_layer3_2_2)

        new_s3_3_1, layer3_3_1index = cal.Similarity(index3_3_1, new_layer3_3_1)
        new_s3_3_2, layer3_3_2index = cal.Similarity(index3_3_2, new_layer3_3_2)

        new_s3_4_1, layer3_4_1index = cal.Similarity(index3_4_1, new_layer3_4_1)
        new_s3_4_2, layer3_4_2index = cal.Similarity(index3_4_2, new_layer3_4_2)

        new_s3_5_1, layer3_5_1index = cal.Similarity(index3_5_1, new_layer3_5_1)
        new_s3_5_2, layer3_5_2index = cal.Similarity(index3_5_2, new_layer3_5_2)

        new_s3_6_1, layer3_6_1index = cal.Similarity(index3_6_1, new_layer3_6_1)
        new_s3_6_2, layer3_6_2index = cal.Similarity(index3_6_2, new_layer3_6_2)
        new_s3_6_3, layer3_6_3index = cal.Similarity(index3_6_3, new_layer3_6_3)

        new_s4_1_1, layer4_1_1index = cal.Similarity(index4_1_1, new_layer4_1_1)
        new_s4_1_2, layer4_1_2index = cal.Similarity(index4_1_2, new_layer4_1_2)
        # print('index2_1',index2_1)
        new_s4_2_1, layer4_2_1index = cal.Similarity(index4_2_1, new_layer4_2_1)
        new_s4_2_2, layer4_2_2index = cal.Similarity(index4_2_2, new_layer4_2_2)

        new_s4_3_1, layer4_3_1index = cal.Similarity(index4_3_1, new_layer4_3_1)
        new_s4_3_2, layer4_3_2index = cal.Similarity(index4_3_2, new_layer4_3_2)
        new_s4_3_3, layer4_3_3index = cal.Similarity(index4_3_3, new_layer4_3_3)

        layer1_1_1similar.extend(new_s1_1_1)
        layer1_1_2similar.extend(new_s1_1_2)

        layer1_2_1similar.extend(new_s1_2_1)
        layer1_2_2similar.extend(new_s1_2_2)

        layer1_3_1similar.extend(new_s1_3_1)
        layer1_3_2similar.extend(new_s1_3_2)
        layer1_3_3similar.extend(new_s1_3_3)

        layer2_1_1similar.extend(new_s2_1_1)
        layer2_1_2similar.extend(new_s2_1_2)

        layer2_2_1similar.extend(new_s2_2_1)
        layer2_2_2similar.extend(new_s2_2_2)

        layer2_3_1similar.extend(new_s2_3_1)
        layer2_3_2similar.extend(new_s2_3_2)

        layer2_4_1similar.extend(new_s2_4_1)
        layer2_4_2similar.extend(new_s2_4_2)
        layer2_4_3similar.extend(new_s2_4_3)

        layer3_1_1similar.extend(new_s3_1_1)
        layer3_1_2similar.extend(new_s3_1_2)

        layer3_2_1similar.extend(new_s3_2_1)
        layer3_2_2similar.extend(new_s3_2_2)

        layer3_3_1similar.extend(new_s3_3_1)
        layer3_3_2similar.extend(new_s3_3_2)

        layer3_4_1similar.extend(new_s3_4_1)
        layer3_4_2similar.extend(new_s3_4_2)

        layer3_5_1similar.extend(new_s3_5_1)
        layer3_5_2similar.extend(new_s3_5_2)

        layer3_6_1similar.extend(new_s3_6_1)
        layer3_6_2similar.extend(new_s3_6_2)
        layer3_6_3similar.extend(new_s3_6_3)

        layer4_1_1similar.extend(new_s4_1_1)
        layer4_1_2similar.extend(new_s4_1_2)

        layer4_2_1similar.extend(new_s4_2_1)
        layer4_2_2similar.extend(new_s4_2_2)

        layer4_3_1similar.extend(new_s4_3_1)
        layer4_3_2similar.extend(new_s4_3_2)
        layer4_3_3similar.extend(new_s4_3_3)

        j += 1


# print('index1_2',index1_2)
layer1_1_1similar = np.array(layer1_1_1similar)
layer1_1_1index = np.array(layer1_1_1index)
layer1_1_1norm = np.array(layer1_1_1norm)
np.save("./cal_similar/layer1_1_1similar.npy", layer1_1_1similar)
np.save("./cal_similar/layer1_1_1index.npy", layer1_1_1index)
np.save("./cal_similar/layer1_1_1norm.npy", layer1_1_1norm)

layer1_1_2similar = np.array(layer1_1_2similar)
layer1_1_2index = np.array(layer1_1_2index)
layer1_1_2norm = np.array(layer1_1_2norm)
np.save("./cal_similar/layer1_1_2similar.npy", layer1_1_2similar)
np.save("./cal_similar/layer1_1_2index.npy", layer1_1_2index)
np.save("./cal_similar/layer1_1_2norm.npy", layer1_1_2norm)

layer1_2_1similar = np.array(layer1_2_1similar)
layer1_2_1index = np.array(layer1_2_1index)
layer1_2_1norm = np.array(layer1_2_1norm)
np.save("./cal_similar/layer1_2_1similar.npy", layer1_2_1similar)
np.save("./cal_similar/layer1_2_1index.npy", layer1_2_1index)
np.save("./cal_similar/layer1_2_1norm.npy", layer1_2_1norm)

layer1_2_2similar = np.array(layer1_2_2similar)
layer1_2_2index = np.array(layer1_2_2index)
layer1_2_2norm = np.array(layer1_2_2norm)
np.save("./cal_similar/layer1_2_2similar.npy", layer1_2_2similar)
np.save("./cal_similar/layer1_2_2index.npy", layer1_2_2index)
np.save("./cal_similar/layer1_2_2norm.npy", layer1_2_2norm)

layer1_3_1similar = np.array(layer1_3_1similar)
layer1_3_1index = np.array(layer1_3_1index)
layer1_3_1norm = np.array(layer1_3_1norm)
np.save("./cal_similar/layer1_3_1similar.npy", layer1_3_1similar)
np.save("./cal_similar/layer1_3_1index.npy", layer1_3_1index)
np.save("./cal_similar/layer1_3_1norm.npy", layer1_3_1norm)

layer1_3_2similar = np.array(layer1_3_2similar)
layer1_3_2index = np.array(layer1_3_2index)
layer1_3_2norm = np.array(layer1_3_2norm)
np.save("./cal_similar/layer1_3_2similar.npy", layer1_3_2similar)
np.save("./cal_similar/layer1_3_2index.npy", layer1_3_2index)
np.save("./cal_similar/layer1_3_2norm.npy", layer1_3_2norm)

layer1_3_3similar = np.array(layer1_3_3similar)
layer1_3_3index = np.array(layer1_3_3index)
layer1_3_3norm = np.array(layer1_3_3norm)
np.save("./cal_similar/layer1_3_3similar.npy", layer1_3_3similar)
np.save("./cal_similar/layer1_3_3index.npy", layer1_3_3index)
np.save("./cal_similar/layer1_3_3norm.npy", layer1_3_3norm)


layer2_1_1similar = np.array(layer2_1_1similar)
layer2_1_1index = np.array(layer2_1_1index)
layer2_1_1norm = np.array(layer2_1_1norm)
np.save("./cal_similar/layer2_1_1similar.npy", layer2_1_1similar)
np.save("./cal_similar/layer2_1_1index.npy", layer2_1_1index)
np.save("./cal_similar/layer2_1_1norm.npy", layer2_1_1norm)

layer2_1_2similar = np.array(layer2_1_2similar)
layer2_1_2index = np.array(layer2_1_2index)
layer2_1_2norm = np.array(layer2_1_2norm)
np.save("./cal_similar/layer2_1_2similar.npy", layer2_1_2similar)
np.save("./cal_similar/layer2_1_2index.npy", layer2_1_2index)
np.save("./cal_similar/layer2_1_2norm.npy", layer2_1_2norm)

layer2_2_1similar = np.array(layer2_2_1similar)
layer2_2_1index = np.array(layer2_2_1index)
layer2_2_1norm = np.array(layer2_2_1norm)
np.save("./cal_similar/layer2_2_1similar.npy", layer2_2_1similar)
np.save("./cal_similar/layer2_2_1index.npy", layer2_2_1index)
np.save("./cal_similar/layer2_2_1norm.npy", layer2_2_1norm)

layer2_2_2similar = np.array(layer2_2_2similar)
layer2_2_2index = np.array(layer2_2_2index)
layer2_2_2norm = np.array(layer2_2_2norm)
np.save("./cal_similar/layer2_2_2similar.npy", layer2_2_2similar)
np.save("./cal_similar/layer2_2_2index.npy", layer2_2_2index)
np.save("./cal_similar/layer2_2_2norm.npy", layer2_2_2norm)

layer2_3_1similar = np.array(layer2_3_1similar)
layer2_3_1index = np.array(layer2_3_1index)
layer2_3_1norm = np.array(layer2_3_1norm)
np.save("./cal_similar/layer2_3_1similar.npy", layer2_3_1similar)
np.save("./cal_similar/layer2_3_1index.npy", layer2_3_1index)
np.save("./cal_similar/layer2_3_1norm.npy", layer2_3_1norm)

layer2_3_2similar = np.array(layer2_3_2similar)
layer2_3_2index = np.array(layer2_3_2index)
layer2_3_2norm = np.array(layer2_3_2norm)
np.save("./cal_similar/layer2_3_2similar.npy", layer2_3_2similar)
np.save("./cal_similar/layer2_3_2index.npy", layer2_3_2index)
np.save("./cal_similar/layer2_3_2norm.npy", layer2_3_2norm)

layer2_4_1similar = np.array(layer2_4_1similar)
layer2_4_1index = np.array(layer2_4_1index)
layer2_4_1norm = np.array(layer2_4_1norm)
np.save("./cal_similar/layer2_4_1similar.npy", layer2_4_1similar)
np.save("./cal_similar/layer2_4_1index.npy", layer2_4_1index)
np.save("./cal_similar/layer2_4_1norm.npy", layer2_4_1norm)

layer2_4_2similar = np.array(layer2_4_2similar)
layer2_4_2index = np.array(layer2_4_2index)
layer2_4_2norm = np.array(layer2_4_2norm)
np.save("./cal_similar/layer2_4_2similar.npy", layer2_4_2similar)
np.save("./cal_similar/layer2_4_2index.npy", layer2_4_2index)
np.save("./cal_similar/layer2_4_2norm.npy", layer2_4_2norm)

layer2_4_3similar = np.array(layer2_4_3similar)
layer2_4_3index = np.array(layer2_4_3index)
layer2_4_3norm = np.array(layer2_4_3norm)
np.save("./cal_similar/layer2_4_3similar.npy", layer2_4_3similar)
np.save("./cal_similar/layer2_4_3index.npy", layer2_4_3index)
np.save("./cal_similar/layer2_4_3norm.npy", layer2_4_3norm)


layer3_1_1similar = np.array(layer3_1_1similar)
layer3_1_1index = np.array(layer3_1_1index)
layer3_1_1norm = np.array(layer3_1_1norm)
np.save("./cal_similar/layer3_1_1similar.npy", layer3_1_1similar)
np.save("./cal_similar/layer3_1_1index.npy", layer3_1_1index)
np.save("./cal_similar/layer3_1_1norm.npy", layer3_1_1norm)

layer3_1_2similar = np.array(layer3_1_2similar)
layer3_1_2index = np.array(layer3_1_2index)
layer3_1_2norm = np.array(layer3_1_2norm)
np.save("./cal_similar/layer3_1_2similar.npy", layer3_1_2similar)
np.save("./cal_similar/layer3_1_2index.npy", layer3_1_2index)
np.save("./cal_similar/layer3_1_2norm.npy", layer3_1_2norm)

layer3_2_1similar = np.array(layer3_2_1similar)
layer3_2_1index = np.array(layer3_2_1index)
layer3_2_1norm = np.array(layer3_2_1norm)
np.save("./cal_similar/layer3_2_1similar.npy", layer3_2_1similar)
np.save("./cal_similar/layer3_2_1index.npy", layer3_2_1index)
np.save("./cal_similar/layer3_2_1norm.npy", layer3_2_1norm)

layer3_2_2similar = np.array(layer3_2_2similar)
layer3_2_2index = np.array(layer3_2_2index)
layer3_2_2norm = np.array(layer3_2_2norm)
np.save("./cal_similar/layer3_2_2similar.npy", layer3_2_2similar)
np.save("./cal_similar/layer3_2_2index.npy", layer3_2_2index)
np.save("./cal_similar/layer3_2_2norm.npy", layer3_2_2norm)

layer3_3_1similar = np.array(layer3_3_1similar)
layer3_3_1index = np.array(layer3_3_1index)
layer3_3_1norm = np.array(layer3_3_1norm)
np.save("./cal_similar/layer3_3_1similar.npy", layer3_3_1similar)
np.save("./cal_similar/layer3_3_1index.npy", layer3_3_1index)
np.save("./cal_similar/layer3_3_1norm.npy", layer3_3_1norm)

layer3_3_2similar = np.array(layer3_3_2similar)
layer3_3_2index = np.array(layer3_3_2index)
layer3_3_2norm = np.array(layer3_3_2norm)
np.save("./cal_similar/layer3_3_2similar.npy", layer3_3_2similar)
np.save("./cal_similar/layer3_3_2index.npy", layer3_3_2index)
np.save("./cal_similar/layer3_3_2norm.npy", layer3_3_2norm)

layer3_4_1similar = np.array(layer3_4_1similar)
layer3_4_1index = np.array(layer3_4_1index)
layer3_4_1norm = np.array(layer3_4_1norm)
np.save("./cal_similar/layer3_4_1similar.npy", layer3_4_1similar)
np.save("./cal_similar/layer3_4_1index.npy", layer3_4_1index)
np.save("./cal_similar/layer3_4_1norm.npy", layer3_4_1norm)

layer3_4_2similar = np.array(layer3_4_2similar)
layer3_4_2index = np.array(layer3_4_2index)
layer3_4_2norm = np.array(layer3_4_2norm)
np.save("./cal_similar/layer3_4_2similar.npy", layer3_4_2similar)
np.save("./cal_similar/layer3_4_2index.npy", layer3_4_2index)
np.save("./cal_similar/layer3_4_2norm.npy", layer3_4_2norm)

layer3_5_1similar = np.array(layer3_5_1similar)
layer3_5_1index = np.array(layer3_5_1index)
layer3_5_1norm = np.array(layer3_5_1norm)
np.save("./cal_similar/layer3_5_1similar.npy", layer3_5_1similar)
np.save("./cal_similar/layer3_5_1index.npy", layer3_5_1index)
np.save("./cal_similar/layer3_5_1norm.npy", layer3_5_1norm)

layer3_5_2similar = np.array(layer3_5_2similar)
layer3_5_2index = np.array(layer3_5_2index)
layer3_4_2norm = np.array(layer3_5_2norm)
np.save("./cal_similar/layer3_5_2similar.npy", layer3_5_2similar)
np.save("./cal_similar/layer3_5_2index.npy", layer3_5_2index)
np.save("./cal_similar/layer3_5_2norm.npy", layer3_5_2norm)

layer3_6_1similar = np.array(layer3_6_1similar)
layer3_6_1index = np.array(layer3_6_1index)
layer3_6_1norm = np.array(layer3_6_1norm)
np.save("./cal_similar/layer3_6_1similar.npy", layer3_6_1similar)
np.save("./cal_similar/layer3_6_1index.npy", layer3_6_1index)
np.save("./cal_similar/layer3_6_1norm.npy", layer3_6_1norm)

layer3_6_2similar = np.array(layer3_6_2similar)
layer3_6_2index = np.array(layer3_6_2index)
layer3_6_2norm = np.array(layer3_6_2norm)
np.save("./cal_similar/layer3_6_2similar.npy", layer3_6_2similar)
np.save("./cal_similar/layer3_6_2index.npy", layer3_6_2index)
np.save("./cal_similar/layer3_6_2norm.npy", layer3_6_2norm)

layer3_6_3similar = np.array(layer3_6_3similar)
layer3_6_3index = np.array(layer3_6_3index)
layer3_6_3norm = np.array(layer3_6_3norm)
np.save("./cal_similar/layer3_6_3similar.npy", layer3_6_3similar)
np.save("./cal_similar/layer3_6_3index.npy", layer3_6_3index)
np.save("./cal_similar/layer3_6_3norm.npy", layer3_6_3norm)

layer4_1_1similar = np.array(layer4_1_1similar)
layer4_1_1index = np.array(layer4_1_1index)
layer4_1_1norm = np.array(layer4_1_1norm)
np.save("./cal_similar/layer4_1_1similar.npy", layer4_1_1similar)
np.save("./cal_similar/layer4_1_1index.npy", layer4_1_1index)
np.save("./cal_similar/layer4_1_1norm.npy", layer4_1_1norm)

layer4_1_2similar = np.array(layer4_1_2similar)
layer4_1_2index = np.array(layer4_1_2index)
layer4_1_2norm = np.array(layer4_1_2norm)
np.save("./cal_similar/layer4_1_2similar.npy", layer4_1_2similar)
np.save("./cal_similar/layer4_1_2index.npy", layer4_1_2index)
np.save("./cal_similar/layer4_1_2norm.npy", layer4_1_2norm)

layer4_2_1similar = np.array(layer4_2_1similar)
layer4_2_1index = np.array(layer4_2_1index)
layer4_2_1norm = np.array(layer4_2_1norm)
np.save("./cal_similar/layer4_2_1similar.npy", layer4_2_1similar)
np.save("./cal_similar/layer4_2_1index.npy", layer4_2_1index)
np.save("./cal_similar/layer4_2_1norm.npy", layer4_2_1norm)

layer4_2_2similar = np.array(layer4_2_2similar)
layer4_2_2index = np.array(layer4_2_2index)
layer4_2_2norm = np.array(layer4_2_2norm)
np.save("./cal_similar/layer4_2_2similar.npy", layer4_2_2similar)
np.save("./cal_similar/layer4_2_2index.npy", layer4_2_2index)
np.save("./cal_similar/layer4_2_2norm.npy", layer4_2_2norm)

layer4_3_1similar = np.array(layer4_3_1similar)
layer4_3_1index = np.array(layer4_3_1index)
layer4_3_1norm = np.array(layer4_3_1norm)
np.save("./cal_similar/layer4_3_1similar.npy", layer4_3_1similar)
np.save("./cal_similar/layer4_3_1index.npy", layer4_3_1index)
np.save("./cal_similar/layer4_3_1norm.npy", layer4_3_1norm)

layer4_3_2similar = np.array(layer4_3_2similar)
layer4_3_2index = np.array(layer4_3_2index)
layer4_3_2norm = np.array(layer4_3_2norm)
np.save("./cal_similar/layer4_3_2similar.npy", layer4_3_2similar)
np.save("./cal_similar/layer4_3_2index.npy", layer4_3_2index)
np.save("./cal_similar/layer4_3_2norm.npy", layer4_3_2norm)

layer4_3_3similar = np.array(layer4_3_3similar)
layer4_3_3index = np.array(layer4_3_3index)
layer4_3_3norm = np.array(layer4_3_3norm)
np.save("./cal_similar/layer4_3_3similar.npy", layer4_3_3similar)
np.save("./cal_similar/layer4_3_3index.npy", layer4_3_3index)
np.save("./cal_similar/layer4_3_3norm.npy", layer4_3_3norm)
