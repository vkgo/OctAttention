'''
Author: fuchy@stu.pku.edu.cn
Date: 2021-09-17 23:30:48
LastEditTime: 2021-12-02 22:18:56
LastEditors: FCY
Description: dataPrepare helper
FilePath: /compression/Preparedata/data.py
All rights reserved.
'''
from Octree import GenOctree,GenKparentSeq
import pt as pointCloud
import numpy as np
import os
import hdf5storage

def dataPrepare(fileName,saveMatDir='Data',qs=1,ptNamePrefix='',offset='min',qlevel=None,rotation=False,normalize=False):
    '''
    这个函数dataPrepare的主要目的是准备并保存点云数据到一个.mat文件。
    它接受一系列参数，包括文件名、保存路径、量化步长、点云名称前缀、偏移量、量化等级、是否旋转和是否标准化等。
    流程：
    检查保存路径是否存在，如果不存在，则创建相应的目录。
    从文件中读取点云数据。
    如果需要标准化，将点云数据标准化到[-1,1]^3范围内。
    如果需要旋转，对点云数据进行旋转。
    根据参数offset，对点云数据进行平移。
    如果提供了量化等级，根据量化等级计算量化步长。
    对点云数据进行量化，去除重复的点，并生成八叉树（Octree）。
    生成四叉树（K-parent）序列。
    保存点云数据、相关信息和四叉树序列到一个.mat文件。
    返回生成的.mat文件路径、量化后的点云数据和原始点云数据。
    '''
    if not os.path.exists(saveMatDir):
        os.makedirs(saveMatDir)
    ptName = ptNamePrefix+os.path.splitext(os.path.basename(fileName))[0]
    p = pointCloud.ptread(fileName) # 原始点云

    refPt = p
    if normalize is True: # normalize pc to [-1,1]^3
        p = p - np.mean(p,axis=0)
        p = p/abs(p).max()
        refPt = p

    if rotation:
        refPt = refPt[:,[0,2,1]]
        refPt[:,2] = - refPt[:,2]

    if offset is 'min':
        offset = np.min(refPt,0)

    points = refPt - offset

    if qlevel is not None:
        qs = (points.max() - points.min())/(2**qlevel-1)
    #  将点云坐标进行四舍五入，并去除重复的点
    pt = np.round(points/qs)
    pt,idx = np.unique(pt,axis=0,return_index=True)
    pt = pt.astype(int)
    # pointCloud.write_ply_data('pori.ply',np.hstack((pt,c)),attributeName=['reflectance'],attriType=['uint16'])
    code,Octree,QLevel = GenOctree(pt)
    DataSturct = GenKparentSeq(Octree,4)

    ptcloud = {'Location':refPt}
    Info = {'qs':qs,'offset':offset,'Lmax':QLevel,'name':ptName,'levelSID':np.array([Octreelevel.node[-1].nodeid for Octreelevel in Octree])}
    patchFile = {'patchFile':(np.concatenate((np.expand_dims(DataSturct['Seq'],2),DataSturct['Level'],DataSturct['Pos']),2), ptcloud, Info)}
    hdf5storage.savemat(os.path.join(saveMatDir,ptName+'.mat'), patchFile, format='7.3', oned_as='row', store_python_metadata=True)
    DQpt = (pt*qs+offset)
    return os.path.join(saveMatDir,ptName+'.mat'),DQpt,refPt #  .mat 文件的路径、量化后的点云数据（在此不变化，但排序过）、坐标对齐零点的点云数据。