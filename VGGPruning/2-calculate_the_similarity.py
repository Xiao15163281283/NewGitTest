import torch
import vgg_ts
from torchvision import transforms
import load
from torch.autograd import Variable
import cal
import fenlei
import torch.nn as nn
import numpy as np

model = vgg_ts.vgg16_bn()
checkpoint = torch.load('./model_0.7372/model_best.pth.tar')
model.load_state_dict(checkpoint)
samplesDataset = load.samplesDataset('./data/train_samples',
                                     transform=transforms.Compose([
                                         transforms.RandomCrop(32, padding=4),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                     ]))
data_loader = torch.utils.data.DataLoader(samplesDataset, batch_size=128, shuffle=False)


class PearsonCorrelation(nn.Module):
    def forward(self,tensor_1,tensor_2):
        x = tensor_1
        y = tensor_2
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)
        vx += 0.01
        vy += 0.01
        cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        return cost


def Linear(conv1_1,batch_size):
    new_conv1_1 = conv1_1
    weight_copy1_1 = new_conv1_1.abs().clone()
    weight_copy1_1 = weight_copy1_1.cpu().numpy()
    L1_norm1_1 = np.sum(weight_copy1_1,axis=(0,))/batch_size
    return L1_norm1_1


def L1(conv1_1,batch_size):
    conv1_1 = conv1_1.permute(1, 0, 2, 3)  # (batch,dim,h,w)--->(dim,batch,h,w)
    new_conv1_1 = conv1_1.reshape(conv1_1.shape[0], -1)  # (dim,batch,h,w)--->(dim,batch*h*w)
    weight_copy1_1 = new_conv1_1.abs().clone()
    weight_copy1_1 = weight_copy1_1.cpu().numpy()
    L1_norm1_1 = np.sum(weight_copy1_1, axis=(1,))/batch_size
    return L1_norm1_1

def index(x,y,list1):
    arg_max_index1 = x[y.index(list1[0]):y.index(list1[len(list1)-1])]
    arg_max_index2 = x[y.index(list1[len(list1)-1])]
    arg_max1_1_1 = np.concatenate((arg_max_index1, [arg_max_index2]))
    index = arg_max1_1_1.tolist()
    return index


def Similarity(index1_1,new_conv1_1):
    new_conv1_1 = new_conv1_1
    new_s1_1_1 = []
    index1_1_1 = []
    for l in range(3):
        tmp = []
        for ii in range(len(index1_1[l])):
            index0 = index1_1[l][ii]
            for j in range(ii + 1, len(index1_1[l])):
                index1 = index1_1[l][j]
                s = PearsonCorrelation()
                cost = s(new_conv1_1[index0],new_conv1_1[index1])
                cost = cost.item()
                cost = round(cost, 3)
                tmp.append(cost)
                index1_1_1.append((index0, index1))
                j += 1
            ii += 1
        new_s1_1_1.append(tmp)
        l += 1
    return new_s1_1_1,index1_1_1

init_seed = 1
torch.cuda.manual_seed(init_seed)
model.eval()
model = model.cuda()
batch_size = 128

conv1_1_norm=[]
conv1_2_norm=[]
conv2_1_norm=[]
conv2_2_norm=[]
conv3_1_norm=[]
conv3_2_norm=[]
conv3_3_norm=[]
conv4_1_norm=[]
conv4_2_norm=[]
conv4_3_norm=[]
conv5_1_norm=[]
conv5_2_norm=[]
conv5_3_norm=[]
linear_norm_1 = []
linear_norm_2 = []
j=0
for data, target in data_loader:
    print('j',j)
    data, target = data.cuda(), target.cuda()
    with torch.no_grad():
        data, target = Variable(data), Variable(target)
        output, conv1_1, conv1_2, conv2_1, conv2_2, conv3_1, conv3_2, conv3_3, conv4_1, conv4_2, conv4_3, conv5_1, conv5_2, conv5_3, linear1, linear2 = model(data)

        L1_norm1_1 = L1(conv1_1, batch_size)
        conv1_1_norm.append(L1_norm1_1)
        L1_norm1_2 = L1(conv1_2, batch_size)
        conv1_2_norm.append(L1_norm1_2)

        L1_norm2_1 = L1(conv2_1, batch_size)
        conv2_1_norm.append(L1_norm2_1)
        L1_norm2_2 = L1(conv2_2, batch_size)
        conv2_2_norm.append(L1_norm2_2)

        L1_norm3_1 = L1(conv3_1, batch_size)
        conv3_1_norm.append(L1_norm3_1)
        L1_norm3_2 = L1(conv3_2, batch_size)
        conv3_2_norm.append(L1_norm3_2)
        L1_norm3_3 = L1(conv3_3, batch_size)
        conv3_3_norm.append(L1_norm3_3)

        L1_norm4_1 = L1(conv4_1, batch_size)
        conv4_1_norm.append(L1_norm4_1)
        L1_norm4_2 = L1(conv4_2, batch_size)
        conv4_2_norm.append(L1_norm4_2)
        L1_norm4_3 = L1(conv4_3, batch_size)
        conv4_3_norm.append(L1_norm4_3)

        L1_norm5_1 = L1(conv5_1, batch_size)
        conv5_1_norm.append(L1_norm5_1)
        L1_norm5_2 = L1(conv5_2, batch_size)
        conv5_2_norm.append(L1_norm5_2)
        L1_norm5_3 = L1(conv5_3, batch_size)
        conv5_3_norm.append(L1_norm5_3)

        linear_norm1 = Linear(linear1, batch_size)
        linear_norm_1.append(linear_norm1)
        linear_norm2 = Linear(linear2, batch_size)
        linear_norm_2.append(linear_norm2)
        j += 1

index1_1 = fenlei.cal_index1_1(conv1_1_norm)
index1_2 = fenlei.cal_index1_2(conv1_2_norm)

index2_1 = fenlei.cal_index2_1(conv2_1_norm)
index2_2 = fenlei.cal_index2_2(conv2_2_norm)

index3_1 = fenlei.cal_index3_1(conv3_1_norm)
index3_2 = fenlei.cal_index3_2(conv3_2_norm)
index3_3 = fenlei.cal_index3_3(conv3_3_norm)

index4_1 = fenlei.cal_index4_1(conv4_1_norm)
index4_2 = fenlei.cal_index4_2(conv4_2_norm)
index4_3 = fenlei.cal_index4_3(conv4_3_norm)

index5_1 = fenlei.cal_index5_1(conv5_1_norm)
index5_2 = fenlei.cal_index5_2(conv5_2_norm)
index5_3 = fenlei.cal_index5_3(conv5_3_norm)

similar1_1 =[]
similar1_2 =[]
similar2_1 =[]
similar2_2 =[]
similar3_1 =[]
similar3_2 =[]
similar3_3 =[]
similar4_1 =[]
similar4_2 =[]
similar4_3 =[]
similar5_1 =[]
similar5_2 =[]
similar5_3 =[]
j=0
for data, target in data_loader:
    print('j',j)
    data, target = data.cuda(), target.cuda()
    with torch.no_grad():
        data, target = Variable(data), Variable(target)
        output, conv1_1, conv1_2, conv2_1, conv2_2, conv3_1, conv3_2, conv3_3, conv4_1, conv4_2, conv4_3, conv5_1, conv5_2, conv5_3, linear1, linear2 = model(data)

        conv1_1 = conv1_1.permute(1, 0, 2, 3)
        new_conv1_1 = conv1_1.reshape(conv1_1.shape[0], -1)
        conv1_2 = conv1_2.permute(1, 0, 2, 3)
        new_conv1_2 = conv1_2.reshape(conv1_2.shape[0], -1)

        conv2_1 = conv2_1.permute(1, 0, 2, 3)
        new_conv2_1 = conv2_1.reshape(conv2_1.shape[0], -1)
        conv2_2 = conv2_2.permute(1, 0, 2, 3)
        new_conv2_2 = conv2_2.reshape(conv2_2.shape[0], -1)

        conv3_1 = conv3_1.permute(1, 0, 2, 3)
        new_conv3_1 = conv3_1.reshape(conv3_1.shape[0], -1)
        conv3_2 = conv3_2.permute(1, 0, 2, 3)
        new_conv3_2 = conv3_2.reshape(conv3_2.shape[0], -1)
        conv3_3 = conv3_3.permute(1, 0, 2, 3)
        new_conv3_3 = conv3_3.reshape(conv3_3.shape[0], -1)

        conv4_1 = conv4_1.permute(1, 0, 2, 3)
        new_conv4_1 = conv4_1.reshape(conv4_1.shape[0], -1)
        conv4_2 = conv4_2.permute(1, 0, 2, 3)
        new_conv4_2 = conv4_2.reshape(conv4_2.shape[0], -1)
        conv4_3 = conv4_3.permute(1, 0, 2, 3)
        new_conv4_3 = conv4_3.reshape(conv4_3.shape[0], -1)

        conv5_1 = conv5_1.permute(1, 0, 2, 3)
        new_conv5_1 = conv5_1.reshape(conv5_1.shape[0], -1)
        conv5_2 = conv5_2.permute(1, 0, 2, 3)
        new_conv5_2 = conv5_2.reshape(conv5_2.shape[0], -1)
        conv5_3 = conv5_3.permute(1, 0, 2, 3)
        new_conv5_3 = conv5_3.reshape(conv5_3.shape[0], -1)

        new_s1_1, index1_1_1 = cal.Similarity(index1_1,new_conv1_1)
        new_s1_2, index1_1_2 = cal.Similarity(index1_2, new_conv1_2)
        # print('index2_1',index2_1)
        new_s2_1, index1_2_1 = cal.Similarity(index2_1, new_conv2_1)
        new_s2_2, index1_2_2 = cal.Similarity(index2_2, new_conv2_2)

        new_s3_1, index1_3_1 = cal.Similarity(index3_1, new_conv3_1)
        new_s3_2, index1_3_2 = cal.Similarity(index3_2, new_conv3_2)
        new_s3_3, index1_3_3 = cal.Similarity(index3_3, new_conv3_3)

        new_s4_1, index1_4_1 = cal.Similarity(index4_1, new_conv4_1)
        new_s4_2, index1_4_2 = cal.Similarity(index4_2, new_conv4_2)
        new_s4_3, index1_4_3 = cal.Similarity(index4_3, new_conv4_3)

        new_s5_1, index1_5_1 = cal.Similarity(index5_1, new_conv5_1)
        new_s5_2, index1_5_2 = cal.Similarity(index5_2, new_conv5_2)
        new_s5_3, index1_5_3 = cal.Similarity(index5_3, new_conv5_3)

        similar1_1.extend(new_s1_1)
        similar1_2.extend(new_s1_2)
        similar2_1.extend(new_s2_1)
        similar2_2.extend(new_s2_2)

        similar3_1.extend(new_s3_1)
        similar3_2.extend(new_s3_2)
        similar3_3.extend(new_s3_3)

        similar4_1.extend(new_s4_1)
        similar4_2.extend(new_s4_2)
        similar4_3.extend(new_s4_3)

        similar5_1.extend(new_s5_1)
        similar5_2.extend(new_s5_2)
        similar5_3.extend(new_s5_3)
        # break
        j += 1

similar1_1 = np.array(similar1_1)
index1_1 = np.array(index1_1_1)
L1_norm1_1 = np.array(conv1_1_norm)
np.save("./cal_similar/similar1_1.npy", similar1_1)
np.save("./cal_similar/index1_1.npy", index1_1)
np.save("./cal_similar/L1_norm1_1.npy", L1_norm1_1)

similar1_2 = np.array(similar1_2)
index1_2 = np.array(index1_1_2)
L1_norm1_2 = np.array(conv1_2_norm)
np.save("./cal_similar/similar1_2.npy", similar1_2)
np.save("./cal_similar/index1_2.npy", index1_2)
np.save("./cal_similar/L1_norm1_2.npy", L1_norm1_2)

similar2_1 = np.array(similar2_1)
index2_1 = np.array(index1_2_1)
L1_norm2_1 = np.array(conv2_1_norm)
np.save("./cal_similar/similar2_1.npy", similar2_1)
np.save("./cal_similar/index2_1.npy", index2_1)
np.save("./cal_similar/L1_norm2_1.npy", L1_norm2_1)

similar2_2 = np.array(similar2_2)
index2_2 = np.array(index1_2_2)
L1_norm2_2 = np.array(conv2_2_norm)
np.save("./cal_similar/similar2_2.npy", similar2_2)
np.save("./cal_similar/index2_2.npy", index2_2)
np.save("./cal_similar/L1_norm2_2.npy", L1_norm2_2)

similar3_1 = np.array(similar3_1)
index3_1 = np.array(index1_3_1)
L1_norm3_1 = np.array(conv3_1_norm)
np.save("./cal_similar/similar3_1.npy", similar3_1)
np.save("./cal_similar/index3_1.npy", index3_1)
np.save("./cal_similar/L1_norm3_1.npy", L1_norm3_1)

similar3_2 = np.array(similar3_2)
index3_2 = np.array(index1_3_2)
L1_norm3_2 = np.array(conv3_2_norm)
np.save("./cal_similar/similar3_2.npy", similar3_2)
np.save("./cal_similar/index3_2.npy", index3_2)
np.save("./cal_similar/L1_norm3_2.npy", L1_norm3_2)

similar3_3 = np.array(similar3_3)
index3_3 = np.array(index1_3_3)
L1_norm3_3 = np.array(conv3_3_norm)
np.save("./cal_similar/similar3_3.npy", similar3_3)
np.save("./cal_similar/index3_3.npy", index3_3)
np.save("./cal_similar/L1_norm3_3.npy", L1_norm3_3)

similar4_1 = np.array(similar4_1)
index4_1 = np.array(index1_4_1)
L1_norm4_1 = np.array(conv4_1_norm)
np.save("./cal_similar/similar4_1.npy", similar4_1)
np.save("./cal_similar/index4_1.npy", index4_1)
np.save("./cal_similar/L1_norm4_1.npy", L1_norm4_1)

similar4_2 = np.array(similar4_2)
index4_2 = np.array(index1_4_2)
L1_norm4_2 = np.array(conv4_2_norm)
np.save("./cal_similar/similar4_2.npy", similar4_2)
np.save("./cal_similar/index4_2.npy", index4_2)
np.save("./cal_similar/L1_norm4_2.npy", L1_norm4_2)

similar4_3 = np.array(similar4_3)
index4_3 = np.array(index1_4_3)
L1_norm4_3 = np.array(conv4_3_norm)
np.save("./cal_similar/similar4_3.npy", similar4_3)
np.save("./cal_similar/index4_3.npy", index4_3)
np.save("./cal_similar/L1_norm4_3.npy", L1_norm4_3)

similar5_1 = np.array(similar5_1)
index5_1 = np.array(index1_5_1)
L1_norm5_1 = np.array(conv5_1_norm)
np.save("./cal_similar/similar5_1.npy", similar5_1)
np.save("./cal_similar/index5_1.npy", index5_1)
np.save("./cal_similar/L1_norm5_1.npy", L1_norm5_1)

similar5_2 = np.array(similar5_2)
index5_2 = np.array(index1_5_2)
L1_norm5_2 = np.array(conv5_2_norm)
np.save("./cal_similar/similar5_2.npy", similar5_2)
np.save("./cal_similar/index5_2.npy", index5_2)
np.save("./cal_similar/L1_norm5_2.npy", L1_norm5_2)
#
similar5_3 = np.array(similar5_3)
index5_3 = np.array(index1_5_3)
L1_norm5_3 = np.array(conv5_3_norm)
np.save("./cal_similar/similar5_3.npy", similar5_3)
np.save("./cal_similar/index5_3.npy", index5_3)
np.save("./cal_similar/L1_norm5_3.npy", L1_norm5_3)

linear_norm_1 = np.array(linear_norm_1)
np.save("./cal_similar/linear_norm_1.npy", linear_norm_1)
linear_norm_2 = np.array(linear_norm_2)
np.save("./cal_similar/linear_norm_2.npy", linear_norm_2)




