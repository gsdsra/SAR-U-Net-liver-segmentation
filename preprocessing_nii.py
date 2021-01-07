import numpy as np
import os
import SimpleITK as sitk
import random
from scipy import ndimage
import parameter as para
from tqdm import tqdm

class LITS_fix:
    def __init__(self, raw_dataset_path, fixed_dataset_path):
        self.raw_root_path = raw_dataset_path
        self.fixed_path = fixed_dataset_path

        if not os.path.exists(self.fixed_path):  # 创建保存目录
            os.makedirs(self.fixed_path + 'data')
            os.makedirs(self.fixed_path + 'label')

        self.fix_data()  # 对原始图像进行修剪并保存

    def fix_data(self):
        # upper = 200
        # lower = -200
        # expand_slice = 20  # 轴向外侧扩张的slice数量
        # size = 48  # 取样的slice数量

        print('the raw dataset total numbers of samples is :', len(os.listdir(para.raw_ct_path)))
        for ct_file in tqdm(os.listdir(para.raw_ct_path )):
            print(ct_file)

            # 将CT和金标准入读内存
            ct = sitk.ReadImage(os.path.join(para.raw_ct_path , ct_file), sitk.sitkInt16)
            ct_array = sitk.GetArrayFromImage(ct)

            seg = sitk.ReadImage(os.path.join(para.raw_seg_path , ct_file.replace('volume', 'segmentation')),
                                 sitk.sitkInt8)
            seg_array = sitk.GetArrayFromImage(seg)

            print(ct_array.shape, seg_array.shape)

            # 将灰度值在阈值之外的截断掉
            # ct_array[ct_array > para.upper] = para.upper
            # ct_array[ct_array < para.lower] = para.lower

            # 对CT数据在横断面上进行降采样(下采样),并进行重采样,将所有数据的z轴的spacing调整到1mm
            ct_array = ndimage.zoom(ct_array,(ct.GetSpacing()[-1] / para.slice_thickness, para.down_scale, para.down_scale),order=3)
            seg_array = ndimage.zoom(seg_array, (ct.GetSpacing()[-1] / para.slice_thickness, para.down_scale, para.down_scale), order=0)
            print(ct_array.shape, seg_array.shape)

            # 找到肝脏区域开始和结束的slice，并各向外扩张
            z = np.any(seg_array, axis=(1, 2))
            start_slice, end_slice = np.where(z)[0][[0, -1]]

            #俩个方向上个扩张个slice
            start_slice = max(0, start_slice - para.expand_slice)
            end_slice = min(seg_array.shape[0], end_slice + para.expand_slice)

            # 如果这时候剩下的slice数量不足size，直接放弃，这样的数据很少
            if end_slice - start_slice < para.size - 1:  # 过滤掉不足以生成一个切片块的原始样本
                continue

            print(str(start_slice) + '--' + str(end_slice))

            ct_array = ct_array[start_slice:end_slice + 1, :, :]  # 截取原始CT影像中包含肝脏区间及拓张的所有切片
            seg_array = seg_array[start_slice:end_slice + 1, :, :]

            #最终将数据保存为nii文件
            new_ct = sitk.GetImageFromArray(ct_array)

            new_ct.SetDirection(ct.GetDirection())
            new_ct.SetOrigin(ct.GetOrigin())
            new_ct.SetSpacing((ct.GetSpacing()[0] * int(1 / para.down_scale),
                               ct.GetSpacing()[1] * int(1 / para.down_scale), para.slice_thickness))

            new_seg = sitk.GetImageFromArray(seg_array)

            new_seg.SetDirection(ct.GetDirection())
            new_seg.SetOrigin(ct.GetOrigin())
            new_seg.SetSpacing((ct.GetSpacing()[0], ct.GetSpacing()[1], para.slice_thickness))

            sitk.WriteImage(new_ct, os.path.join(para.fixed_ct_path, ct_file))
            sitk.WriteImage(new_seg, os.path.join(para.fixed_seg_path,
                                                  ct_file.replace('volume', 'segmentation')))

def main():
    raw_dataset_path = '/home/haishan/Data/dataPeiQing/PeiQing/liver_segmentation_gai/data/'
    fixed_dataset_path = '/home/haishan/Data/dataPeiQing/PeiQing/liver_segmentation_gai/fixed/'

    LITS_fix(raw_dataset_path, fixed_dataset_path)


if __name__ == '__main__':
    main()