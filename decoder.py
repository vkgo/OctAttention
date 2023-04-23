'''
Author: fuchy@stu.pku.edu.cn
Date: 2021-09-17 23:30:48
LastEditTime: 2021-12-02 22:18:56
LastEditors: FCY
Description: decoder
FilePath: /compression/decoder.py
All rights reserved.
'''
#%%
import  numpy as np
import torch
from tqdm import tqdm
from Octree import DeOctree, dec2bin
import pt 
from dataset import default_loader as matloader
from collections import deque
import os 
import time
from networkTool import *
from encoderTool import bpttRepeatTime,generate_square_subsequent_mask
from encoder import model,list_orifile
import numpyAc
batch_size = 1 

#%%
'''
description: decode bin file to occupancy code
param {str;input bin file name} binfile
param {N*1 array; occupancy code, only used for check} oct_data_seq
param {model} model
param {int; Context window length} bptt
return {N*1,float}occupancy code,time
'''
def decodeOct(binfile,oct_data_seq,model,bptt):
    model.eval()
    with torch.no_grad():
        elapsed = time.time()

        KfatherNode = [[255,0,0]]*levelNumK
        nodeQ = deque()
        oct_seq = []
        src_mask = generate_square_subsequent_mask(bptt).to(device)

        input = torch.zeros((bptt,batch_size,levelNumK,3)).long().to(device)
        padinginbptt = torch.zeros((bptt,batch_size,levelNumK,3)).long().to(device)
        bpttMovSize = bptt//bpttRepeatTime
        # input torch.Size([256, 32, 4, 3]) bptt,batch_sz,kparent,[oct,level,octant]
        # all of [oct,level,octant] default is zero

        output = model(input,src_mask,[])

        freqsinit = torch.softmax(output[-1],1).squeeze().cpu().detach().numpy()  # 将模型输出转换为概率分布
        
        oct_len = len(oct_data_seq)

        dec = numpyAc.arithmeticDeCoding(None,oct_len,255,binfile) # 初始化算术解码器

        root =  decodeNode(freqsinit,dec) # 解码根节点
        nodeId = 0
        
        KfatherNode = KfatherNode[3:]+[[root,1,1]] + [[root,1,1]] # for padding for first row # ( the parent of root node is root itself)
        
        nodeQ.append(KfatherNode) 
        oct_seq.append(root) #decode the root  
        
        with tqdm(total=  oct_len+10) as pbar:
            while True:
                father = nodeQ.popleft()
                childOcu = dec2bin(father[-1][0])
                childOcu.reverse()
                faterLevel = father[-1][1] 
                for i in range(8):
                    if(childOcu[i]):
                        faterFeat = [[father+[[root,faterLevel+1,i+1]]]] # Fill in the information of the node currently decoded [xi-1, xi level, xi octant]
                        faterFeatTensor = torch.Tensor(faterFeat).long().to(device)
                        faterFeatTensor[:,:,:,0] -= 1

                        # shift bptt window
                        offsetInbpttt = (nodeId)%(bpttMovSize) # the offset of current node in the bppt window
                        if offsetInbpttt==0: # a new bptt window
                            input = torch.vstack((input[bpttMovSize:],faterFeatTensor,padinginbptt[0:bpttMovSize-1]))
                        else:
                            input[bptt-bpttMovSize+offsetInbpttt] = faterFeatTensor

                        output = model(input,src_mask,[])
                        
                        Pro = torch.softmax(output[offsetInbpttt+bptt-bpttMovSize],1).squeeze().cpu().detach().numpy()

                        root =  decodeNode(Pro,dec)
                        nodeId += 1
                        pbar.update(1)
                        KfatherNode = father[1:]+[[root,faterLevel+1,i+1]]
                        nodeQ.append(KfatherNode)
                        if(root==256 or nodeId==oct_len):
                            assert len(oct_data_seq) == nodeId # for check oct num
                            Code = oct_seq
                            return Code,time.time() - elapsed
                        oct_seq.append(root)
                    assert oct_data_seq[nodeId] == root # for check

def decodeNode(pro,dec):
    root = dec.decode(np.expand_dims(pro,0))
    return root+1


if __name__=="__main__":
    list_orifile = ['testplyfiles/boxer_viewdep_vox9.ply']
    for oriFile in list_orifile:  # 遍历encoder.py中的原始文件列表
        ptName = os.path.basename(oriFile)[:-4]  # 提取原始文件的基本名称（不带扩展名）
        matName = 'Data/testPly/' + ptName + '.mat'  # 构造对应的.mat文件名
        binfile = expName + '/data/' + ptName + '.bin'  # 构造对应的.bin文件名
        cell, mat = matloader(matName)  # 加载.mat文件，获取其单元格数据和矩阵数据

        # 读取Sideinfo
        oct_data_seq = np.transpose(mat[cell[0, 0]]).astype(int)[:, -1:, 0]  # 提取八叉树节点占用信息作为检查数据

        p = np.transpose(mat[cell[1, 0]]['Location'])  # 提取原始点云数据
        offset = np.transpose(mat[cell[2, 0]]['offset'])  # 提取解码时需要的偏移量
        qs = mat[cell[2, 0]]['qs'][0]  # 提取解码时需要的缩放因子

        Code, elapsed = decodeOct(binfile, oct_data_seq, model, bptt)  # 解码二进制文件以获取八叉树占用信息，同时计算解码耗时
        print('decode succee, time:', elapsed)  # 打印解码成功及耗时信息
        print('oct len:', len(Code))  # 打印八叉树占用信息长度

        # DeOctree
        ptrec = DeOctree(Code)  # 从八叉树占用信息重构点云
        # Dequantization
        DQpt = (ptrec * qs + offset)  # 对重构点云进行逆量化操作
        pt.write_ply_data(expName + "/temp/test/rec.ply", DQpt)  # 将重构点云写入.ply文件
        pt.pcerror(p, DQpt, None, '-r 1', None).wait()  # 计算原始点云和重构点云之间的误差
