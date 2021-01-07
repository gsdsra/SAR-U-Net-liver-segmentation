import os
import shutil
import nibabel as nib
import glob
import imageio
import numpy as np
import multiprocessing
from utils.preprocessing_utils import sitk2slices, sitk2labels
import SimpleITK as sitk


if __name__ == '__main__':
    fixed_raw_path = 'fixed/raw/'
    fixed_label_path = 'fixed/label/'
    tr_path = 'fixed/save_first_rot_equal/tr/'
    ts_path = 'fixed/save_first_rot_equal/ts/'
    raw_path = 'raw/'
    label_path = 'label/'

    for i in range(31):
        print(i)
        ct = sitk.ReadImage(fixed_raw_path +'volume-'+str(i)+'.nii',sitk.sitkInt16)
        ct_array = np.rot90(sitk.GetArrayFromImage(ct))
        ct_array = np.rot90(ct_array)

        seg = sitk.ReadImage(fixed_label_path+'segmentation-'+str(i)+'.nii',sitk.sitkInt16)
        seg_array = np.rot90(sitk.GetArrayFromImage(seg))
        seg_array = np.rot90(seg_array)

        slices_in_order = sitk2slices(ct_array,0,400)
        labels_in_order = sitk2labels(seg_array)
        for n in range(len(slices_in_order)):
            imageio.imwrite(ts_path+raw_path+str(i)+'_'+str(n).zfill(4)+'.png', slices_in_order[n].astype(np.uint8))
            imageio.imwrite(ts_path+label_path+str(i)+'_'+str(n).zfill(4)+'.png', labels_in_order[n].astype(np.uint8))

    for i in range(31, 131):
        print(i)
        ct = sitk.ReadImage(fixed_raw_path+ 'volume-' + str(i) + '.nii', sitk.sitkInt16)
        ct_array = np.rot90(sitk.GetArrayFromImage(ct))
        ct_array = np.rot90(ct_array)

        seg = sitk.ReadImage(fixed_label_path+ 'segmentation-' + str(i) + '.nii', sitk.sitkInt16)
        seg_array = np.rot90(sitk.GetArrayFromImage(seg))
        seg_array = np.rot90(seg_array)

        slices_in_order = sitk2slices(ct_array,0,400)
        labels_in_order = sitk2labels(seg_array)
        for n in range(len(slices_in_order)):
            imageio.imwrite(tr_path+raw_path+str(i)+'_'+str(n).zfill(4)+'.png', slices_in_order[n].astype(np.uint8))
            imageio.imwrite(tr_path+label_path+str(i)+'_'+str(n).zfill(4)+'.png', labels_in_order[n].astype(np.uint8))