import numpy as np


def index(x,y,list1):
    arg_max_index1 = x[y.index(list1[0]):y.index(list1[len(list1)-1])]
    arg_max_index2 = x[y.index(list1[len(list1)-1])]
    arg_max1_1_1 = np.concatenate((arg_max_index1, [arg_max_index2]))
    index = arg_max1_1_1.tolist()
    return index


def remove_in(remove_index, remove_pre):
    l = []
    for i in range(0, len(remove_pre), 2):
        tmp1 = remove_pre[i]
        tmp2 = remove_pre[i + 1]
        if tmp1 < tmp2:
            l.append(i)
        else:
            l.append(i + 1)
    new = []
    for i in l:
        new.append(remove_index[i])
    list2 = []
    for s in new:
        if s not in list2:
            list2.append(s)
    return list2


def circle(df, cent1, cent2, cent3, list1, list2, list3, result):
    for i in range(len(df)):
        a = np.abs(df[i] - cent1)
        b = np.abs(df[i] - cent2)
        c = np.abs(df[i] - cent3)
        if a < b and a < c:
            list1.append(df[i])
        if b < a and b < c:
            list2.append(df[i])
        if c < a and c < b:
            list3.append(df[i])
        if a == b and a == c:
            list1.append(df[i])
    nd1 = np.array(list1)
    nd2 = np.array(list2)
    nd3 = np.array(list3)
    cent_n1 = round(np.sum(nd1)/len(nd1),3)
    cent_n2 = round(np.sum(nd2)/len(nd2),3)
    cent_n3 = round(np.sum(nd3)/len(nd3),3)
    if cent_n1 != cent1 or cent_n2 != cent2 or cent_n3 != cent3:
        cent1 = cent_n1
        cent2 = cent_n2
        cent3 = cent_n3
        result = 1
    else:
        result = 0

    return result, cent1, cent2, cent3


def means1(x):
    cent1 = x[5]
    cent2 = x[20]
    cent3 = x[40]
    list1 = []
    list2 = []
    list3 = []
    result = 1
    while result:
        if cent1 == 0 or cent2 == 0 or cent3 == 0:
            break
        list1.clear()
        list2.clear()
        list3.clear()
        result, cent1, cent2, cent3 = circle(x, cent1, cent2, cent3, list1, list2, list3, result)
    return list1, list2, list3


def means2(x):
    cent1 = x[20]
    cent2 = x[60]
    cent3 = x[90]
    list1 = []
    list2 = []
    list3 = []
    result = 1
    while result:
        if cent1 == 0 or cent2 == 0 or cent3 == 0:
            break
        list1.clear()
        list2.clear()
        list3.clear()
        result, cent1, cent2, cent3 = circle(x, cent1, cent2, cent3, list1, list2, list3, result)
    return list1, list2, list3


def means3(x):
    cent1 = x[60]
    cent2 = x[120]
    cent3 = x[200]
    list1 = []
    list2 = []
    list3 = []
    result = 1
    while result:
        if cent1 == 0 or cent2 == 0 or cent3 == 0:
            break
        list1.clear()
        list2.clear()
        list3.clear()
        result, cent1, cent2, cent3 = circle(x, cent1, cent2, cent3, list1, list2, list3, result)
    return list1, list2, list3


def means4(x):
    cent1 = x[10]
    cent2 = x[150]
    cent3 = x[300]
    list1 = []
    list2 = []
    list3 = []
    result = 1
    while result:
        if cent1 == 0 or cent2 == 0 or cent3 == 0:
            break
        list1.clear()
        list2.clear()
        list3.clear()
        result, cent1, cent2, cent3 = circle(x, cent1, cent2, cent3, list1, list2, list3, result)
    return list1, list2, list3


def means5(x):
    cent1 = x[10]
    cent2 = x[100]
    cent3 = x[450]
    list1 = []
    list2 = []
    list3 = []
    result = 1
    while result:
        if cent1 == 0 or cent2 == 0 or cent3 == 0:
            break
        list1.clear()
        list2.clear()
        list3.clear()
        result, cent1, cent2, cent3 = circle(x, cent1, cent2, cent3, list1, list2, list3, result)
    return list1, list2, list3


def cal_index1_1(conv1_1_norm):
    L1_norm1_1 = np.sum(conv1_1_norm, axis=(0,)) / 40
    arg_max1_1 = np.argsort(L1_norm1_1)
    new_L1_norm1_1 = sorted(L1_norm1_1)
    list = []
    list1, list2, list3 = means1(new_L1_norm1_1)
    list.append(list1)
    list.append(list2)
    list.append(list3)
    index1_1 = []
    for i in range(3):
        index_1 = index(arg_max1_1, new_L1_norm1_1, list[i])
        index1_1.append(index_1)
    return index1_1


def cal_index1_2(conv1_2_norm):
    L1_norm1_1 = np.sum(conv1_2_norm, axis=(0,)) / 40
    arg_max1_1 = np.argsort(L1_norm1_1)
    new_L1_norm1_1 = sorted(L1_norm1_1)
    list = []
    list1, list2, list3 = means1(new_L1_norm1_1)
    list.append(list1)
    list.append(list2)
    list.append(list3)
    index1_1 = []
    for i in range(3):
        index_1 = index(arg_max1_1, new_L1_norm1_1, list[i])
        index1_1.append(index_1)
    return index1_1


def cal_index2_1(conv2_1_norm):
    L1_norm1_1 = np.sum(conv2_1_norm, axis=(0,)) / 40
    arg_max1_1 = np.argsort(L1_norm1_1)
    new_L1_norm1_1 = sorted(L1_norm1_1)
    list = []
    list1, list2, list3 = means2(new_L1_norm1_1)
    list.append(list1)
    list.append(list2)
    list.append(list3)
    index1_1 = []
    for i in range(3):
        index_1 = index(arg_max1_1, new_L1_norm1_1, list[i])
        index1_1.append(index_1)
    return index1_1


def cal_index2_2(conv2_2_norm):
    L1_norm1_1 = np.sum(conv2_2_norm, axis=(0,)) / 40
    arg_max1_1 = np.argsort(L1_norm1_1)
    new_L1_norm1_1 = sorted(L1_norm1_1)
    list = []
    list1, list2, list3 = means2(new_L1_norm1_1)
    list.append(list1)
    list.append(list2)
    list.append(list3)
    index1_1 = []
    for i in range(3):
        index_1 = index(arg_max1_1, new_L1_norm1_1, list[i])
        index1_1.append(index_1)
    return index1_1


def cal_index3_1(conv3_1_norm):
    L1_norm1_1 = np.sum(conv3_1_norm, axis=(0,)) / 40
    arg_max1_1 = np.argsort(L1_norm1_1)
    new_L1_norm1_1 = sorted(L1_norm1_1)
    list = []
    list1, list2, list3 = means3(new_L1_norm1_1)
    list.append(list1)
    list.append(list2)
    list.append(list3)
    index1_1 = []
    for i in range(3):
        index_1 = index(arg_max1_1, new_L1_norm1_1, list[i])
        index1_1.append(index_1)
    return index1_1


def cal_index3_2(conv3_2_norm):
    L1_norm1_1 = np.sum(conv3_2_norm, axis=(0,)) / 40
    arg_max1_1 = np.argsort(L1_norm1_1)
    new_L1_norm1_1 = sorted(L1_norm1_1)
    list = []
    list1, list2, list3 = means3(new_L1_norm1_1)
    list.append(list1)
    list.append(list2)
    list.append(list3)
    index1_1 = []
    for i in range(3):
        index_1 = index(arg_max1_1, new_L1_norm1_1, list[i])
        index1_1.append(index_1)
    return index1_1


def cal_index3_3(conv3_3_norm):
    L1_norm1_1 = np.sum(conv3_3_norm, axis=(0,)) / 40
    arg_max1_1 = np.argsort(L1_norm1_1)
    new_L1_norm1_1 = sorted(L1_norm1_1)
    list = []
    list1, list2, list3 = means3(new_L1_norm1_1)
    list.append(list1)
    list.append(list2)
    list.append(list3)
    index1_1 = []
    for i in range(3):
        index_1 = index(arg_max1_1, new_L1_norm1_1, list[i])
        index1_1.append(index_1)
    return index1_1


def cal_index4_1(conv4_1_norm):
    L1_norm1_1 = np.sum(conv4_1_norm, axis=(0,)) / 40
    arg_max1_1 = np.argsort(L1_norm1_1)
    new_L1_norm1_1 = sorted(L1_norm1_1)
    list = []
    list1, list2, list3 = means4(new_L1_norm1_1)
    list.append(list1)
    list.append(list2)
    list.append(list3)
    index1_1 = []
    for i in range(3):
        index_1 = index(arg_max1_1, new_L1_norm1_1, list[i])
        index1_1.append(index_1)
    return index1_1


def cal_index4_2(conv4_2_norm):
    L1_norm1_1 = np.sum(conv4_2_norm, axis=(0,)) / 40
    arg_max1_1 = np.argsort(L1_norm1_1)
    new_L1_norm1_1 = sorted(L1_norm1_1)
    list = []
    list1, list2, list3 = means4(new_L1_norm1_1)
    list.append(list1)
    list.append(list2)
    list.append(list3)
    index1_1 = []
    for i in range(3):
        index_1 = index(arg_max1_1, new_L1_norm1_1, list[i])
        index1_1.append(index_1)
    return index1_1


def cal_index4_3(conv4_3_norm):
    L1_norm1_1 = np.sum(conv4_3_norm, axis=(0,)) / 40
    arg_max1_1 = np.argsort(L1_norm1_1)
    new_L1_norm1_1 = sorted(L1_norm1_1)
    list = []
    list1, list2, list3 = means4(new_L1_norm1_1)
    list.append(list1)
    list.append(list2)
    list.append(list3)
    index1_1 = []
    for i in range(3):
        index_1 = index(arg_max1_1, new_L1_norm1_1, list[i])
        index1_1.append(index_1)
    return index1_1


def cal_index5_1(conv5_1_norm):
    L1_norm1_1 = np.sum(conv5_1_norm, axis=(0,)) / 40
    arg_max1_1 = np.argsort(L1_norm1_1)
    new_L1_norm1_1 = sorted(L1_norm1_1)
    list = []
    list1, list2, list3 = means5(new_L1_norm1_1)
    list.append(list1)
    list.append(list2)
    list.append(list3)
    index1_1 = []
    for i in range(3):
        index_1 = index(arg_max1_1, new_L1_norm1_1, list[i])
        index1_1.append(index_1)
    return index1_1


def cal_index5_2(conv5_2_norm):
    L1_norm1_1 = np.sum(conv5_2_norm, axis=(0,)) / 40
    arg_max1_1 = np.argsort(L1_norm1_1)
    new_L1_norm1_1 = sorted(L1_norm1_1)
    list = []
    list1, list2, list3 = means5(new_L1_norm1_1)
    list.append(list1)
    list.append(list2)
    list.append(list3)
    index1_1 = []
    for i in range(3):
        index_1 = index(arg_max1_1, new_L1_norm1_1, list[i])
        index1_1.append(index_1)
    return index1_1


def cal_index5_3(conv5_3_norm):
    L1_norm1_1 = np.sum(conv5_3_norm, axis=(0,)) / 40
    arg_max1_1 = np.argsort(L1_norm1_1)
    new_L1_norm1_1 = sorted(L1_norm1_1)
    list = []
    list1, list2, list3 = means5(new_L1_norm1_1)
    list.append(list1)
    list.append(list2)
    list.append(list3)
    index1_1 = []
    for i in range(3):
        index_1 = index(arg_max1_1, new_L1_norm1_1, list[i])
        index1_1.append(index_1)
    return index1_1