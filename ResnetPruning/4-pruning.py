import torch
import numpy as np
import resnet_new
import resnet_prune1_1
import argparse
import os
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
# from feature_pruning_index import cfg_before,remove
parser = argparse.ArgumentParser(description='resnet cifar100 prune')
parser.add_argument('--dataset', type=str, default='cifar100',
                    help='training dataset (default: cifar100)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--model', default='./0.7786model/model_0.7786.pth.tar', type=str, metavar='PATH',
                    help='path to the model (default: none)')
parser.add_argument('--save', default='./layer_prune', type=str, metavar='PATH',
                    help='path to save pruned model (default: none)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
if not os.path.exists(args.save):
    os.makedirs(args.save)


def test(model):
    kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])),
        batch_size=args.test_batch_size, shuffle=False, **kwargs)

    model.eval()
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            data, target = Variable(data), Variable(target)
            output = model(data)
            # #print('output', output)
            # #print('===========================')
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    print('\nTest set: Accuracy: {}/{} ({:.1f}%)\n'.format(
        correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    return correct / float(len(test_loader.dataset))


cfg_before = list(np.load('./layer_remove_index/cfg_before.npy',allow_pickle=True))
#print('cfg_before',cfg_before)

remove = list(np.load('./layer_remove_index/remove_total.npy', allow_pickle=True))
# print('remove',remove)
remain = list(np.load('./layer_remove_index/remain.npy', allow_pickle=True))
#print(remain)
# Pre-pruning
cfg_mask = []
layer_id = 0
for i, j in zip(cfg_before, remove):
    out_channels = i
    # #print('out_channels', out_channels)
    # #print('j', j)
    mask = torch.ones(out_channels)
    # #print('mask', mask.shape)
    mask[j] = 0
    # #print('mask', mask.shape)
    cfg_mask.append(mask)
#print('cfg_mask',len(cfg_mask))

# pruning
before_model = resnet_new.Resnet50()


"""参数未复制"""
if args.cuda:
    before_model.cuda()
if args.model:
    if os.path.isfile(args.model):
        #print("=> loading checkpoint '{}'".format(args.model))
        checkpoint = torch.load(args.model)
        # args.start_epoch = checkpoint['epoch']
        # best_prec1 = checkpoint['best_prec1']
        before_model.load_state_dict(checkpoint['state_dict'])


acc1 = test(before_model)
# for parameters in before_model.parameters():
#     print(parameters)
# input()
# print('acc1',acc1)

newmodel = resnet_prune1_1.Resnet50()

if args.cuda:
    newmodel.cuda()

start_mask = torch.ones(3)
layer_id_in_cfg = 0
end_mask = cfg_mask[layer_id_in_cfg]

for [m0, m1] in zip(before_model.modules(), newmodel.modules()):
    # print('m0',m0)
    # print('m1',m1)
    if isinstance(m0, nn.BatchNorm2d):
        if layer_id_in_cfg == 3:
            idx1 = np.squeeze(np.argwhere(np.asarray(cfg_mask[layer_id_in_cfg].cpu().numpy())))
            if idx1.size == 1:
                idx1 = np.resize(idx1,(1,))
            m1.weight.data = m0.weight.data[idx1.tolist()].clone()
            m1.bias.data = m0.bias.data[idx1.tolist()].clone()
            m1.running_mean = m0.running_mean[idx1.tolist()].clone()
            m1.running_var = m0.running_var[idx1.tolist()].clone()
            start_mask = cfg_mask[0]
            layer_id_in_cfg += 1
            if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
                #print('layer_id_in_cfg',layer_id_in_cfg)
                end_mask = cfg_mask[layer_id_in_cfg]

        elif layer_id_in_cfg == 13:
            #print(True)
            idx1 = np.squeeze(np.argwhere(np.asarray(cfg_mask[layer_id_in_cfg].cpu().numpy())))
            #print('idx1',idx1)
            if idx1.size == 1:
                idx1 = np.resize(idx1,(1,))
            m1.weight.data = m0.weight.data[idx1.tolist()].clone()
            m1.bias.data = m0.bias.data[idx1.tolist()].clone()
            m1.running_mean = m0.running_mean[idx1.tolist()].clone()
            m1.running_var = m0.running_var[idx1.tolist()].clone()
            start_mask = cfg_mask[10]
            layer_id_in_cfg += 1
            if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
                end_mask = cfg_mask[layer_id_in_cfg]


        elif layer_id_in_cfg == 26:
            #print(True)
            idx1 = np.squeeze(np.argwhere(np.asarray(cfg_mask[layer_id_in_cfg].cpu().numpy())))
            #print('idx1',idx1)
            if idx1.size == 1:
                idx1 = np.resize(idx1,(1,))
            m1.weight.data = m0.weight.data[idx1.tolist()].clone()
            m1.bias.data = m0.bias.data[idx1.tolist()].clone()
            m1.running_mean = m0.running_mean[idx1.tolist()].clone()
            m1.running_var = m0.running_var[idx1.tolist()].clone()
            start_mask = cfg_mask[23]
            layer_id_in_cfg += 1
            #print('layer_id_in_cfg',layer_id_in_cfg)
            #print('start_mask',start_mask)
            if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
                #print('layer_id_in_cfg',layer_id_in_cfg)
                end_mask = cfg_mask[layer_id_in_cfg]
            #print('end_mask', end_mask)

        elif layer_id_in_cfg == 45:
            #print(True)
            idx1 = np.squeeze(np.argwhere(np.asarray(cfg_mask[layer_id_in_cfg].cpu().numpy())))
            #print('idx1',idx1)
            if idx1.size == 1:
                idx1 = np.resize(idx1,(1,))
            m1.weight.data = m0.weight.data[idx1.tolist()].clone()
            m1.bias.data = m0.bias.data[idx1.tolist()].clone()
            m1.running_mean = m0.running_mean[idx1.tolist()].clone()
            m1.running_var = m0.running_var[idx1.tolist()].clone()
            start_mask = cfg_mask[42]
            layer_id_in_cfg += 1
            #print('layer_id_in_cfg',layer_id_in_cfg)
            #print('start_mask',start_mask)
            if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
                #print('layer_id_in_cfg',layer_id_in_cfg)
                end_mask = cfg_mask[layer_id_in_cfg]
            #print('end_mask', end_mask)
        else:
            #print(True)
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            #print('idx1', idx1)
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            m1.weight.data = m0.weight.data[idx1.tolist()].clone()
            m1.bias.data = m0.bias.data[idx1.tolist()].clone()
            m1.running_mean = m0.running_mean[idx1.tolist()].clone()
            m1.running_var = m0.running_var[idx1.tolist()].clone()
            layer_id_in_cfg += 1
            start_mask = end_mask
            #print('layer_id_in_cfg', layer_id_in_cfg)
            #print('start_mask', start_mask)
            if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
                #print('layer_id_in_cfg', layer_id_in_cfg)
                end_mask = cfg_mask[layer_id_in_cfg]
            #print('end_mask', end_mask)

    elif isinstance(m0, nn.Conv2d):
        #print(True,True)
        #print('layer_id_in_cfg',layer_id_in_cfg)
        if layer_id_in_cfg == 4:
            #print(True,True,True)
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            # #print('idx0', idx0)
            # #print('idx1', idx1)
            # input()
            #print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
            w1 = w1[idx1.tolist(), :, :, :].clone()
            m1.weight.data = w1.clone()

        if layer_id_in_cfg == 14:
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            # #print('idx0', idx0)
            # #print('idx1', idx1)
            # input()
            #print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
            w1 = w1[idx1.tolist(), :, :, :].clone()
            m1.weight.data = w1.clone()

        if layer_id_in_cfg == 27:
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            # #print('idx0', idx0)
            # #print('idx1', idx1)
            # input()
            #print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
            w1 = w1[idx1.tolist(), :, :, :].clone()
            m1.weight.data = w1.clone()

        if layer_id_in_cfg == 46:
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            # #print('idx0', idx0)
            # #print('idx1', idx1)
            # input()
            #print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
            w1 = w1[idx1.tolist(), :, :, :].clone()
            m1.weight.data = w1.clone()
        else:
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            #print('idx0', idx0)
            #print('idx1',idx1)
            #print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
            w1 = w1[idx1.tolist(), :, :, :].clone()
            m1.weight.data = w1.clone()
    elif isinstance(m0, nn.Linear):
        # print(True,True)
        # print('layer_id_in_cfg',layer_id_in_cfg)
        # print(len(cfg_mask))
        if layer_id_in_cfg == len(cfg_mask):
            # print(True)
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            m1.weight.data = m0.weight.data[:, idx1].clone()
            m1.bias.data = m0.bias.data.clone()
            # print('m1.weight.shape', m1.weight.shape)
            # print('m1.bias.shape', m1.bias.shape)
        else:
            end_mask = cfg_mask[layer_id_in_cfg]
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            # print('idx0', len(idx0))
            # print('idx1',len(idx1))
            # print(idx0.size)
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            # print(m0.weight.data.shape[1])
            # print('idx0', len(idx0))
            # print(len(idx1))
            w1 = m0.weight.data[:, idx0].clone()
            w1 = w1[idx1.tolist(), :].clone()
            m1.weight.data = w1.clone()
            # print('w1.shape',m1.weight.shape)
            # print(m0.bias.shape)
            # w2 = m0.bias.data[:, idx0.tolist()].clone()
            w2 = m0.bias.data[idx1.tolist()].clone()
            m1.bias.data = w2.clone()
            layer_id_in_cfg += 1
            start_mask = end_mask
    # input()


torch.save({'state_dict': newmodel.state_dict()}, os.path.join(args.save, 'pruned.pth.tar'))

# input()
model = newmodel
model.state_dict()
# for parameters in model.parameters():
#     print(parameters)
# input()
acc = test(newmodel)
print('acc',acc)

num_parameters = sum([param.nelement() for param in newmodel.parameters()])
with open(os.path.join(args.save, "prune.txt"), "w") as fp:
    fp.write("Number of parameters: \n"+str(num_parameters)+"\n")
    fp.write("Test accuracy: \n"+str(acc)+"\n")