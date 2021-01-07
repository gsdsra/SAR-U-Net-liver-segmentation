import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


path = 'E:\pycharmprojects\own_liver_segmentation\csv\Finall_gai.csv'
data = pd.read_csv(path)

#---------------箱形图---------------------
dice_unet,dice_resunet,dice_se_resunet,dice_ours= data['dice_unet'],data['dice_resunet'],data['dice_se_resunet'],data['dice_ours']
# dice = list(dice)

figure,axes=plt.subplots() #得到画板、轴

axes.boxplot([dice_unet,dice_resunet,dice_se_resunet,dice_ours],patch_artist=False,labels=['U-Net','ResUnet','SE-ResUnet','Ours'],showmeans=True,boxprops = {'color':'orangered'}) #描点上色

plt.ylabel("Dice",fontdict={'fontsize':12, 'color':'k'})

plt.title("Performance of Segmentation",fontdict={'fontsize':'18', 'color':'k'})

axes.xaxis.grid(True)
axes.yaxis.grid(True)
axes.yaxis.grid(True)
axes.yaxis.grid(True)

plt.savefig('png/dice.png')
plt.show() #展示\


#-------------------散点图-----------------------------
# fig, ax = plt.subplots()
# x = data['epoch']
# msd_unet = data['msd_unet']
# msd_resunet = data['msd_resunet']
# msd_se_resunet = data['msd_se_resunet']
# msd_ours = data['msd_ours']
# # y_2 = data['msd']
# ax.scatter(x,msd_unet,s=80,c='purple',marker='^',label='U-Net')
# ax.scatter(x,msd_resunet,s=80,c='red',marker='+',label='ResUnet')
# ax.scatter(x,msd_se_resunet,s=80,c='blue',marker='p',label='SE-ResUnet')
# ax.scatter(x,msd_ours,s=80,c='black',marker='8',label='Ours')
#
# plt.ylabel("MSD(mm)",fontdict={'fontsize':12, 'color':'k'})
# plt.title("Performance of Segmentation",fontdict={'fontsize':'18', 'color':'k'})
#
# ax.legend()                                         #显示图例
# plt.savefig('png/msd.png')
# plt.show()
