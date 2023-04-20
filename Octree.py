'''
Author: fuchy@stu.pku.edu.cn
Description: Octree 
FilePath: /compression/Octree.py
All rights reserved.
'''
import numpy as np
from OctreeCPP.Octreewarpper import GenOctree
class CNode():
    def __init__(self,nodeid=0,childPoint=[[]]*8,parent=0,oct=0,pos = np.array([0,0,0]),octant = 0) -> None:
        self.nodeid = nodeid
        self.childPoint=childPoint.copy()
        self.parent = parent
        self.oct = oct # occupancyCode 1~255
        self.pos = pos
        self.octant = octant # 1~8

class COctree():
    def __init__(self,node=[],level=0) -> None:
        self.node = node.copy()
        self.level=level

def dec2bin(n, count=8): 
    """returns the binary of integer n, using count number of digits""" 
    return [int((n >> y) & 1) for y in range(count-1, -1, -1)]

def dec2binAry(x, bits):
    mask = np.expand_dims(2**np.arange(bits-1,-1,-1),1).T              
    return (np.bitwise_and(np.expand_dims(x,1), mask)!=0).astype(int) 

def bin2decAry(x):
    if(x.ndim==1):
        x = np.expand_dims(x,0)
    bits = x.shape[1]
    mask = np.expand_dims(2**np.arange(bits-1,-1,-1),1)
    return x.dot(mask).astype(int)

def Morton(A):
    A =  A.astype(int)
    n = np.ceil(np.log2(np.max(A)+1)).astype(int)   
    x = dec2binAry(A[:,0],n)                         
    y = dec2binAry(A[:,1],n)
    z = dec2binAry(A[:,2],n)
    m = np.stack((x,y,z),2)                           
    m = np.transpose(m,(0, 2, 1))                     
    mcode = np.reshape(m,(A.shape[0],3*n),order='F')  
    return mcode
 
def DeOctree(Codes):
    Codes = np.squeeze(Codes)
    occupancyCode = np.flip(dec2binAry(Codes,8),axis=1)  
    codeL = occupancyCode.shape[0]                        
    N = np.ones((30),int) 
    codcal = 0
    L = 0
    while codcal+N[L]<=codeL:
        L +=1
        try:
            N[L+1] = np.sum(occupancyCode[codcal:codcal+N[L],:])
        except:
            assert 0
        codcal = codcal+N[L]
    Lmax = L
    Octree = [COctree() for _ in range(Lmax+1)]
    proot = [np.array([0,0,0])]
    Octree[0].node = proot
    codei = 0
    for L in range(1,Lmax+1):
        childNode = []  # the node of next level
        for currentNode in Octree[L-1].node: # bbox of currentNode
            code = occupancyCode[codei,:]
            for bit in np.where(code==1)[0].tolist():
                newnode =currentNode+(np.array(dec2bin(bit, count=3))<<(Lmax-L))# bbox of childnode
                childNode.append(newnode)
            codei+=1
        Octree[L].node = childNode.copy()
    points = np.array(Octree[Lmax].node)
    return points

def GenKparentSeq(Octree,K):
    # 获取八叉树的层数
    LevelNum = len(Octree)
    # 获取最后一个节点的ID（即节点总数）
    nodeNum = Octree[-1].node[-1].nodeid
    # 初始化Seq数组，用于存储每个节点的占用编码序列
    Seq = np.ones((nodeNum, K), 'int') * 255
    # 初始化LevelOctant数组，用于存储每个节点的层级和八叉区信息
    LevelOctant = np.zeros((nodeNum, K, 2), 'int')  # Level and Octant
    # 初始化Pos数组，用于存储每个节点的位置信息
    Pos = np.zeros((nodeNum, K, 3), 'int')  # padding 0
    # 初始化ChildID列表，用于存储每个节点的子节点ID（未填充）
    ChildID = [[] for _ in range(nodeNum)]
    # 初始化根节点的占用编码、层级和位置信息
    Seq[0, K-1] = Octree[0].node[0].oct
    LevelOctant[0, K-1, 0] = 1
    LevelOctant[0, K-1, 1] = 1
    Pos[0, K-1, :] = Octree[0].node[0].pos
    # 将根节点的父节点ID设置为1
    Octree[0].node[0].parent = 1
    # 初始化节点计数器
    n = 0
    # 遍历八叉树的每一层
    for L in range(0, LevelNum):
        # 遍历当前层的每个节点
        for node in Octree[L].node:
            # 更新当前节点在Seq、LevelOctant和Pos数组中的信息
            Seq[n, K-1] = node.oct
            Seq[n, 0:K-1] = Seq[node.parent-1, 1:K]
            LevelOctant[n, K-1, :] = [L+1, node.octant]
            LevelOctant[n, 0:K-1] = LevelOctant[node.parent-1, 1:K, :]
            Pos[n, K-1] = node.pos
            Pos[n, 0:K-1, :] = Pos[node.parent-1, 1:K, :]
            # 如果当前层为最大层级，不做额外操作
            if (L == LevelNum-1):
                pass
            # 更新节点计数器
            n += 1
    # 确保节点计数器与节点总数相等
    assert n == nodeNum
    # 将四个数组组合成一个字典并返回
    DataStruct = {'Seq': Seq, 'Level': LevelOctant, 'ChildID': ChildID, 'Pos': Pos}
    return DataStruct
    # Seq是一个二维数组，用于存储八叉树中节点的占用编码序列。在八叉树结构中，
    # 每个节点可以有0到8个子节点，占用编码用一个整数（0-255）表示节点拥有哪些子节点。
    # 例如，占用编码为9的节点表示其具有第一和第四个子节点。
    # Seq的每一行对应一个节点，每一列对应一个八叉树层级。Seq中的值表示节点在该层级的占用编码。
    # 每个节点的序列从其父节点序列继承，然后在最后一列添加自己的占用编码。
    # 通过Seq数组，我们可以在每个层级跟踪节点的子节点信息，以了解八叉树结构的层次关系和节点分布。

    # LevelOctant有两个元素：
    # 第一个元素表示节点所在的层级。这是一个整数，表示节点在八叉树结构中的深度。根节点的层级为1，随着向下遍历树结构，层级逐渐增加。
    # 第二个元素表示节点所在的八叉区。这是一个整数（1-8），表示节点在其父节点的哪个子区域中。每个父节点有8个子区域，分别用1到8的整数表示。