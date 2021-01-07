#----------------------------路径相关参数---------------------------

raw_dataset_path = './data'  #没做预处理的输入数据路径

raw_ct_path = '/home/haishan/Data/dataPeiQing/PeiQing/liver_segmentation_gai/data/raw' #原始ct路径路径

raw_seg_path = '/home/haishan/Data/dataPeiQing/PeiQing/liver_segmentation_gai/data/label' #金标准的数据路径

fixed_dataset_path = './fixed' #预处理后的数据集根路径

fixed_ct_path = '/home/haishan/Data/dataPeiQing/PeiQing/liver_segmentation_gai/fixed/raw' #预处理后的原始ct路径

fixed_seg_path = '/home/haishan/Data/dataPeiQing/PeiQing/liver_segmentation_gai/fixed/label' #预处理后的金标准数据路径






#----------------------路径相关参数----------------------------------------

#------------------------------训练数据获取相关参数---------------------------

upper , lower = 200 ,-200  #CT数据灰度阶段窗口

down_scale = 0.5 #横断面降采样因子

size = 48 #使用48张连续切片作为网络的输入

slice_thickness = 1  # 将所有数据在z轴的spacing归一化到1mm

expand_slice = 20  # 仅使用包含肝脏以及肝脏上下20张切片作为训练样本

#------------------------------训练数据获取相关参数---------------------------

