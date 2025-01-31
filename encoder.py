'''
Author: fuchy@stu.pku.edu.cn
Description: this file encodes point cloud
FilePath: /compression/encoder.py
All rights reserved.
'''
from numpy import mod
from Preparedata.data import dataPrepare
from encoderTool import main
from networkTool import reload,CPrintl,expName,device
from octAttention import model
import glob,datetime,os
import pt as pointCloud
############## warning ###############
## decoder.py relys on this model here
## do not move this lines to somewhere else
model = model.to(device)
saveDic = reload(None,'./Exp_2/total_feature2/checkpoint/encoder_epoch_01200970.pth')
# saveDic = reload(None,'./Exp/Obj/checkpoint/encoder_epoch_008000110.pth')
model.load_state_dict(saveDic['encoder'])

###########Objct##############
list_orifile = ['file/Ply/2851.ply']
# list_orifile = ['testplyfiles/boxer_viewdep_vox9.ply']
if __name__=="__main__":
    printl = CPrintl(expName+'/encoderPLY.txt')
    printl('_'*50,'OctAttention V0.4','_'*50)
    printl(datetime.datetime.now().strftime('%Y-%m-%d:%H:%M:%S'))
    printl('load checkpoint', saveDic['path'])
    for oriFile in list_orifile:
        printl(oriFile)
        if (os.path.getsize(oriFile)>300*(1024**2)):#300M
            printl('too large!')
            continue
        ptName = os.path.splitext(os.path.basename(oriFile))[0] 
        for qs in [1]:
            ptNamePrefix = ptName
            matFile,DQpt,refPt = dataPrepare(oriFile,saveMatDir='./Data/testPly',qs=qs,ptNamePrefix='',rotation=False) #  .mat 文件的路径、量化后的点云数据和处理后的点云数据。
            # please set `rotation=True` in the `dataPrepare` function when processing MVUB data
            result = main(matFile,model,actualcode=True,printl =printl) # actualcode=False: bin file will not be generated
            # result is a dict. {'binsize':binsz, 'ptnum':ptNum, 'octlen':oct_len}
            print('_'*50,'pc_error','_'*50)
            pointCloud.pcerror(refPt,DQpt,None,'-r 1023',None).wait()