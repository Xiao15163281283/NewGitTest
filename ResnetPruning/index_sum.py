import numpy as np
from operator import itemgetter
import fenlei


def index_sum(similar1_1):
    # print(len(similar1_1))
    list1 = []
    list2 = []
    list3 = []
    # print('similar1_1',similar1_1)
    for i in range(0, len(similar1_1), 3):
        list1.append(similar1_1[i])
    # print('list1',list1)
    sum1 = np.sum(list1, axis=0) / 40
    # print('sum1',sum1)

    for i in range(1, len(similar1_1), 3):
        list2.append(similar1_1[i])
    sum2 = np.sum(list2, axis=0) / 40

    for i in range(2, len(similar1_1), 3):
        list3.append(similar1_1[i])
    sum3 = np.sum(list3, axis=0) / 40

    return sum1, sum2, sum3


def index_cut(index1_1_1, sum1, sum2, sum3):
    index1 = []
    for i in range(0, len(sum1)):
        index1.append(index1_1_1[i])
    # print('len(sum2)',len(sum2))
    index2 = []
    for i in range(len(sum1), len(sum1) + len(sum2)):
        index2.append(index1_1_1[i])
    # print('len(sum3)', len(sum3))
    index3 = []
    for i in range(len(sum1) + len(sum2), len(sum1) + len(sum2) + len(sum3)):
        index3.append(index1_1_1[i])

    return index1, index2, index3


def remove_index1_1_1(sum1, index1, sum2, index2, sum3, index3, new_L1_norm1_1):
    print('layer1_1_remove==========================================')
    index = []
    val = []
    for index_, value in sorted(enumerate(sum1), key=itemgetter(1)):
        index.append(index_)
        val.append(value)
    # print('val', val)
    # print('index',index)
    l2 = []
    for l1, every in enumerate(val):
        if every > 0.9:
            l2.append(index[l1])
    pre_remove = []
    #print('index1', index1)
    for i in l2:
        tmp = index1[i]
        pre_remove.append(tmp)
    # #print('pre_remove',pre_remove)
    remove_pre = []
    indexx = []
    remove = []
    # 找到对应的L1范数
    for inde in pre_remove:
        for ii in inde:
            indexx.append(ii)
            remove_pre.append((new_L1_norm1_1[ii]))
    # #print('remove_pre',remove_pre)
    # print('indexx',indexx)
    remove1_index = fenlei.remove_in(indexx, remove_pre)
    # # print('remove1_index',remove1_index)
    # #print('len(remove1_index)',len(remove1_index))
    remove.extend(remove1_index)
    index = []
    val = []
    for index_, value in sorted(enumerate(sum2), key=itemgetter(1)):
        index.append(index_)
        val.append(value)
    # print('val', val)
    # print('index', index)
    l2 = []
    for l1, every in enumerate(val):
        if every > 0.9:
            l2.append(index[l1])
    pre_remove = []
    #print('index2', index2)
    for i in l2:
        tmp = index2[i]
        pre_remove.append(tmp)
    # #print('pre_remove', pre_remove)
    remove_pre = []
    indexx = []
    # 找到对应的L1范数
    for inde in pre_remove:
        for ii in inde:
            indexx.append(ii)
            remove_pre.append((new_L1_norm1_1[ii]))
    # #print('remove_pre', remove_pre)
    # print('indexx', indexx)
    remove2_index = fenlei.remove_in(indexx, remove_pre)
    # # print('remove2_index',remove2_index)
    # # print('len(remove2_index)', len(remove2_index))
    remove.extend(remove2_index)
    index = []
    val = []
    for index_, value in sorted(enumerate(sum3), key=itemgetter(1)):
        index.append(index_)
        val.append(value)
    # print('val',val)
    # print('index',index)
    l2 = []
    for l1, every in enumerate(val):
        if every > 0.9:
            l2.append(index[l1])
    # print('l2', l2)
    #print('index3', index3)
    pre_remove = []
    for i in l2:
        tmp = index3[i]
        pre_remove.append(tmp)
    # #print('pre_remove', pre_remove)
    remove_pre = []
    indexx = []
    # 找到对应的L1范数
    for inde in pre_remove:
        for ii in inde:
            indexx.append(ii)
            remove_pre.append((new_L1_norm1_1[ii]))
    # #print('remove_pre', remove_pre)
    # print('indexx', indexx)
    remove3_index = fenlei.remove_in(indexx, remove_pre)
    # #print('remove_index3', remove3_index)
    # #print('len(remove3_index)', len(remove3_index))
    remove.extend(remove3_index)

    return remove

def remove_index2_1_1(sum1, index1, sum2, index2, sum3, index3, new_L1_norm1_1):
    print('layer1_1_remove==========================================')
    index = []
    val = []
    for index_, value in sorted(enumerate(sum1), key=itemgetter(1)):
        index.append(index_)
        val.append(value)
    # print('val', val)
    # print('index',index)
    l2 = []
    for l1, every in enumerate(val):
        if every > 0.8:
            l2.append(index[l1])
    pre_remove = []
    #print('index1', index1)
    for i in l2:
        tmp = index1[i]
        pre_remove.append(tmp)
    #print('pre_remove',pre_remove)
    remove_pre = []
    indexx = []
    remove = []
    # 找到对应的L1范数
    for inde in pre_remove:
        for ii in inde:
            indexx.append(ii)
            remove_pre.append((new_L1_norm1_1[ii]))
    #print('remove_pre',remove_pre)
    # print('indexx',indexx)
    remove1_index = fenlei.remove_in(indexx, remove_pre)
    # print('remove1_index',remove1_index)
    #print('len(remove1_index)',len(remove1_index))
    remove.extend(remove1_index)
    index = []
    val = []
    for index_, value in sorted(enumerate(sum2), key=itemgetter(1)):
        index.append(index_)
        val.append(value)
    # print('val', val)
    # print('index', index)
    l2 = []
    for l1, every in enumerate(val):
        if every > 0.8:
            l2.append(index[l1])
    pre_remove = []
    #print('index2', index2)
    for i in l2:
        tmp = index2[i]
        pre_remove.append(tmp)
    #print('pre_remove', pre_remove)
    remove_pre = []
    indexx = []
    # 找到对应的L1范数
    for inde in pre_remove:
        for ii in inde:
            indexx.append(ii)
            remove_pre.append((new_L1_norm1_1[ii]))
    #print('remove_pre', remove_pre)
    # print('indexx', indexx)
    remove2_index = fenlei.remove_in(indexx, remove_pre)
    # print('remove2_index',remove2_index)
    # print('len(remove2_index)', len(remove2_index))
    remove.extend(remove2_index)
    index = []
    val = []
    for index_, value in sorted(enumerate(sum3), key=itemgetter(1)):
        index.append(index_)
        val.append(value)
    # print('val',val)
    # print('index',index)
    l2 = []
    for l1, every in enumerate(val):
        if every > 0.8:
            l2.append(index[l1])
    # print('l2', l2)
    #print('index3', index3)
    pre_remove = []
    for i in l2:
        tmp = index3[i]
        pre_remove.append(tmp)
    #print('pre_remove', pre_remove)
    remove_pre = []
    indexx = []
    # 找到对应的L1范数
    for inde in pre_remove:
        for ii in inde:
            indexx.append(ii)
            remove_pre.append((new_L1_norm1_1[ii]))
    #print('remove_pre', remove_pre)
    # print('indexx', indexx)
    remove3_index = fenlei.remove_in(indexx, remove_pre)
    #print('remove_index3', remove3_index)
    #print('len(remove3_index)', len(remove3_index))
    remove.extend(remove3_index)

    return remove

def remove_index3_1_1(sum1, index1, sum2, index2, sum3, index3, new_L1_norm1_1):
    print('layer1_1_remove==========================================')
    index = []
    val = []
    for index_, value in sorted(enumerate(sum1), key=itemgetter(1)):
        index.append(index_)
        val.append(value)
    # print('val', val)
    # print('index',index)
    l2 = []
    for l1, every in enumerate(val):
        if every > 0.9:
            l2.append(index[l1])
    pre_remove = []
    #print('index1', index1)
    for i in l2:
        tmp = index1[i]
        pre_remove.append(tmp)
    #print('pre_remove',pre_remove)
    remove_pre = []
    indexx = []
    remove = []
    # 找到对应的L1范数
    for inde in pre_remove:
        for ii in inde:
            indexx.append(ii)
            remove_pre.append((new_L1_norm1_1[ii]))
    #print('remove_pre',remove_pre)
    # print('indexx',indexx)
    remove1_index = fenlei.remove_in(indexx, remove_pre)
    # print('remove1_index',remove1_index)
    #print('len(remove1_index)',len(remove1_index))
    remove.extend(remove1_index)
    index = []
    val = []
    for index_, value in sorted(enumerate(sum2), key=itemgetter(1)):
        index.append(index_)
        val.append(value)
    # print('val', val)
    # print('index', index)
    l2 = []
    for l1, every in enumerate(val):
        if every > 0.9:
            l2.append(index[l1])
    pre_remove = []
    #print('index2', index2)
    for i in l2:
        tmp = index2[i]
        pre_remove.append(tmp)
    #print('pre_remove', pre_remove)
    remove_pre = []
    indexx = []
    # 找到对应的L1范数
    for inde in pre_remove:
        for ii in inde:
            indexx.append(ii)
            remove_pre.append((new_L1_norm1_1[ii]))
    #print('remove_pre', remove_pre)
    # print('indexx', indexx)
    remove2_index = fenlei.remove_in(indexx, remove_pre)
    # print('remove2_index',remove2_index)
    # print('len(remove2_index)', len(remove2_index))
    remove.extend(remove2_index)
    index = []
    val = []
    for index_, value in sorted(enumerate(sum3), key=itemgetter(1)):
        index.append(index_)
        val.append(value)
    # print('val',val)
    # print('index',index)
    l2 = []
    for l1, every in enumerate(val):
        if every > 0.9:
            l2.append(index[l1])
    # print('l2', l2)
    #print('index3', index3)
    pre_remove = []
    for i in l2:
        tmp = index3[i]
        pre_remove.append(tmp)
    #print('pre_remove', pre_remove)
    remove_pre = []
    indexx = []
    # 找到对应的L1范数
    for inde in pre_remove:
        for ii in inde:
            indexx.append(ii)
            remove_pre.append((new_L1_norm1_1[ii]))
    #print('remove_pre', remove_pre)
    # print('indexx', indexx)
    remove3_index = fenlei.remove_in(indexx, remove_pre)
    #print('remove_index3', remove3_index)
    #print('len(remove3_index)', len(remove3_index))
    remove.extend(remove3_index)

    return remove

def remove_index4_1_1(sum1, index1, sum2, index2, sum3, index3, new_L1_norm1_1):
    print('layer1_1_remove==========================================')
    index = []
    val = []
    for index_, value in sorted(enumerate(sum1), key=itemgetter(1)):
        index.append(index_)
        val.append(value)
    # print('val', val)
    # print('index',index)
    l2 = []
    for l1, every in enumerate(val):
        if every > 0.99:
            l2.append(index[l1])
    pre_remove = []
    #print('index1', index1)
    for i in l2:
        tmp = index1[i]
        pre_remove.append(tmp)
    #print('pre_remove',pre_remove)
    remove_pre = []
    indexx = []
    remove = []
    # 找到对应的L1范数
    for inde in pre_remove:
        for ii in inde:
            indexx.append(ii)
            remove_pre.append((new_L1_norm1_1[ii]))
    #print('remove_pre',remove_pre)
    # print('indexx',indexx)
    remove1_index = fenlei.remove_in(indexx, remove_pre)
    # print('remove1_index',remove1_index)
    #print('len(remove1_index)',len(remove1_index))
    remove.extend(remove1_index)
    index = []
    val = []
    for index_, value in sorted(enumerate(sum2), key=itemgetter(1)):
        index.append(index_)
        val.append(value)
    # print('val', val)
    # print('index', index)
    l2 = []
    for l1, every in enumerate(val):
        if every > 0.99:
            l2.append(index[l1])
    pre_remove = []
    #print('index2', index2)
    for i in l2:
        tmp = index2[i]
        pre_remove.append(tmp)
    #print('pre_remove', pre_remove)
    remove_pre = []
    indexx = []
    # 找到对应的L1范数
    for inde in pre_remove:
        for ii in inde:
            indexx.append(ii)
            remove_pre.append((new_L1_norm1_1[ii]))
    #print('remove_pre', remove_pre)
    # print('indexx', indexx)
    remove2_index = fenlei.remove_in(indexx, remove_pre)
    # print('remove2_index',remove2_index)
    # print('len(remove2_index)', len(remove2_index))
    remove.extend(remove2_index)
    index = []
    val = []
    for index_, value in sorted(enumerate(sum3), key=itemgetter(1)):
        index.append(index_)
        val.append(value)
    # print('val',val)
    # print('index',index)
    l2 = []
    for l1, every in enumerate(val):
        if every > 0.99:
            l2.append(index[l1])
    # print('l2', l2)
    #print('index3', index3)
    pre_remove = []
    for i in l2:
        tmp = index3[i]
        pre_remove.append(tmp)
    #print('pre_remove', pre_remove)
    remove_pre = []
    indexx = []
    # 找到对应的L1范数
    for inde in pre_remove:
        for ii in inde:
            indexx.append(ii)
            remove_pre.append((new_L1_norm1_1[ii]))
    #print('remove_pre', remove_pre)
    # print('indexx', indexx)
    remove3_index = fenlei.remove_in(indexx, remove_pre)
    #print('remove_index3', remove3_index)
    #print('len(remove3_index)', len(remove3_index))
    remove.extend(remove3_index)

    return remove




















