import torch
import numpy as np
import vgg_A_conv1_linear1_1
import vgg_A
import argparse
import os
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
parser = argparse.ArgumentParser(description='vgg cifar100 prune')
parser.add_argument('--dataset', type=str, default='cifar100',
                    help='training dataset (default: cifar100)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--depth', type=int, default=16,help='depth of the vgg')
parser.add_argument('--model', default='./model_0.7372/model_best.pth.tar', type=str, metavar='PATH',
                    help='path to the model (default: none)')
parser.add_argument('--save', default='./remove_prune', type=str, metavar='PATH',
                    help='path to save pruned model (default: none)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
if not os.path.exists(args.save):
    os.makedirs(args.save)


def test(model):
    kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}
    if args.dataset == 'cifar10':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(r'./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
            batch_size=args.test_batch_size, shuffle=False, **kwargs)
    elif args.dataset == 'cifar100':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])),
            batch_size=args.test_batch_size, shuffle=False, **kwargs)
    else:
        raise ValueError("No valid dataset is given.")
    model.eval()
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            data, target = Variable(data), Variable(target)
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    print('\nTest set: Accuracy: {}/{} ({:.1f}%)\n'.format(
        correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    return correct / float(len(test_loader.dataset))


cfg_before = list(np.load('./remove_index/cfg_before.npy',allow_pickle=True))
print('cfg_before',cfg_before)

remove = list(np.load('./remove_index/remove_total.npy', allow_pickle=True))
print('remove',len(remove))

# Pre-pruning
cfg_mask = []
layer_id = 0
for i, j in zip(cfg_before, remove):
    out_channels = i
    mask = torch.ones(out_channels)
    mask[j] = 0
    cfg_mask.append(mask)
print('cfg_mask',len(cfg_mask))
model = vgg_A.vgg16_bn()
if args.cuda:
    model.cuda()
if args.model:
    if os.path.isfile(args.model):
        print("=> loading checkpoint '{}'".format(args.model))
        checkpoint = torch.load(args.model)
        model.load_state_dict(checkpoint)
        print("=> loaded checkpoint '{}' ".format(args.model, ))

acc1 = test(model)
print('acc1',acc1)
newmodel = vgg_A_conv1_linear1_1.vgg16_bn()

if args.cuda:
    newmodel.cuda()

start_mask = torch.ones(3)
layer_id_in_cfg = 0
end_mask = cfg_mask[layer_id_in_cfg]
for [m0, m1] in zip(model.modules(), newmodel.modules()):
    if isinstance(m0, nn.BatchNorm2d):
        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
        if idx1.size == 1:
            idx1 = np.resize(idx1,(1,))
        m1.weight.data = m0.weight.data[idx1.tolist()].clone()
        m1.bias.data = m0.bias.data[idx1.tolist()].clone()
        m1.running_mean = m0.running_mean[idx1.tolist()].clone()
        m1.running_var = m0.running_var[idx1.tolist()].clone()
        layer_id_in_cfg += 1
        start_mask = end_mask
        if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
            end_mask = cfg_mask[layer_id_in_cfg]

    elif isinstance(m0, nn.Conv2d):
        idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
        if idx0.size == 1:
            idx0 = np.resize(idx0, (1,))
        if idx1.size == 1:
            idx1 = np.resize(idx1, (1,))
        w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
        w1 = w1[idx1.tolist(), :, :, :].clone()
        m1.weight.data = w1.clone()

    elif isinstance(m0, nn.Linear):
        if layer_id_in_cfg == len(cfg_mask):
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            m1.weight.data = m0.weight.data[:, idx1].clone()
            m1.bias.data = m0.bias.data.clone()
        else:
            end_mask = cfg_mask[layer_id_in_cfg]
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            w1 = m0.weight.data[:, idx0].clone()
            w1 = w1[idx1.tolist(), :].clone()
            m1.weight.data = w1.clone()
            w2 = m0.bias.data[idx1.tolist()].clone()
            m1.bias.data = w2.clone()
            layer_id_in_cfg += 1
            start_mask = end_mask

torch.save({'state_dict': newmodel.state_dict()}, os.path.join(args.save, 'pruned.pth.tar'))
model = newmodel
params = model.state_dict()
acc = test(model)
print('acc',acc)
num_parameters = sum([param.nelement() for param in newmodel.parameters()])
with open(os.path.join(args.save, "prune.txt"), "w") as fp:
    fp.write("Number of parameters: \n"+str(num_parameters)+"\n")
    fp.write("Test accuracy: \n"+str(acc)+"\n")