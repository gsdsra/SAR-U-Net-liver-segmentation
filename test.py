import imageio
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import SimpleITK as sitk

import utils.metrics as m
# from model.unet.unet_model import UNet
# # from unet import UNet
# from model.FCN.FCN import *
# from model.attention_unet.attention_unet import AttU_Net
# from model.resunet.resunet import DeepResUNet
# from model.se_resunet_plus.se_resunet_plus import SeResUNet
# from model.se_p_attresunet.se_p_attresunet import SE_P_AttU_Net
from models.se_p_resunet.se_p_resunet import Se_PPP_ResUNet

from utils.preprocessing_utils import sitk2slices, sitk2labels
from utils.surface import Surface
# from utils.test_utils import (draw_contours, draw_many_slices, imwrite,
#                               remove_fragment)


if __name__ == '__main__':
    LITS_fixed_data_path = '/home/haishan/Data/dataPeiQing/PeiQing/liver_segmentation_gai/fixed/LITS_FIXED/'
    model_path = '/home/haishan/Data/dataPeiQing/PeiQing/liver_segmentation_gai/final checkpoints/ResUnet_ce/liver_segmentation_Resunet_ce_on_LITS_dataset_iter_292000.pth'
    prediction_path = '/home/haishan/Data/dataPeiQing/PeiQing/liver_segmentation_gai/green_purple_results_thickness2/results_Resunet_ce_NoS/'

    device = torch.device('cuda:0')
    # model =  UNet(1, 2).to(device)
    # model = Se_PPP_ResUNet(1,2,deep_supervision=False).to(device)
    # vgg_model = VGGNet()
    # model = FCNs(pretrained_net=vgg_model, n_class=2).to(device)
    # model = AttU_Net(1,2).to(device)
    model = Se_PPP_ResUNet(1,2).to(device)
    # model = SeResUNet(1, 2, deep_supervision=False).to(device)
    # model = SE_P_AttU_Net(1,2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    sm = nn.Softmax(dim=1)

    idx_list = []
    dice_list = []
    iou_list = []
    voe_list = []
    rvd_list = []
    assd_list = []
    msd_list = []

    for i in range(31):
        print(i)
        idx_list.append(i)
        # volume = nib.load(LITS_data_path + 'volume-' + str(i) + '.nii').get_fdata()
        # vxlspacing = nib.load(LITS_data_path + 'volume-' + str(i) + '.nii').header.get_zooms()[:3]
        # segmentation = nib.load(LITS_data_path + 'segmentation-' + str(i) + '.nii').get_fdata()

        ct = sitk.ReadImage(LITS_fixed_data_path + 'volume-' + str(i) + '.nii',sitk.sitkInt16)
        vxlspacing = ct.GetSpacing()
        ct_array = np.rot90(sitk.GetArrayFromImage(ct))
        ct_array = np.rot90(ct_array)

        seg = sitk.ReadImage(LITS_fixed_data_path + 'segmentation-' + str(i) + '.nii',sitk.sitkInt16)
        seg_array = np.rot90(sitk.GetArrayFromImage(seg))
        seg_array = np.rot90(seg_array)

        slices_in_order = sitk2slices(ct_array,0,400)
        labels_in_order = sitk2labels(seg_array)

        predictions_in_order = []

        for slice in slices_in_order:
            slice = torch.from_numpy(slice).float() / 255.
            # print(slice.shape)   #torch.Size([256, 256])
            output = model(slice.unsqueeze(0).unsqueeze(0).to(device))
            # print(output.shape) torch.Size([1, 2, 256, 256])
            prediction = sm(output)
            # print(prediction.shape)    #torch.Size([1, 2, 256, 256])
            _, prediction = torch.max(prediction, dim=1) #返回每一行中最大值的那个元素，且返回其索引（返回最大元素在这一行的列索引）
            # print(prediction.shape)     #torch.Size([1, 256, 256])
            # print(prediction)
            prediction = prediction.squeeze(0).cpu().detach().numpy().astype(np.uint8)
            # print(prediction.shape) (256, 256)
            predictions_in_order.append(prediction)

        # overlay_in_orders = draw_many_slices(slices_in_order, predictions_in_order, labels_in_order)
        # imwrite(predictions_in_order, overlay_in_orders, prediction_path+str(i) + '/')

        # metrics
        v_prediction = np.stack(predictions_in_order).astype(np.uint)
        v_label = np.stack(labels_in_order).astype(np.uint)

        dice = m.dc(v_prediction, v_label)
        dice_list.append(dice)
        print('DICE:', dice)

        iou = m.jc(v_prediction, v_label)
        iou_list.append(iou)
        print('IOU:', iou)

        voe = 1 - iou
        voe_list.append(voe)
        print('VOE:', voe)

        rvd = m.ravd(v_prediction, v_label)
        rvd_list.append(rvd)
        print('RVD:', rvd)

        if np.count_nonzero(v_prediction) == 0 or np.count_nonzero(v_label) == 0:
            assd = 0
            msd = 0
        else:
            evalsurf = Surface(v_prediction, v_label, physical_voxel_spacing=vxlspacing, mask_offset=[0., 0., 0.],
                               reference_offset=[0., 0., 0.])
            assd = evalsurf.get_average_symmetric_surface_distance()
            msd = m.hd(v_label, v_prediction, voxelspacing=vxlspacing)

        assd_list.append(assd)
        print('ASSD:', assd)

        msd_list.append(msd)
        print('MSD:', msd)

    metric_data = {'dice': dice_list, 'iou': iou_list, 'voe': voe_list, 'rvd': rvd_list, 'assd': assd_list,
                   'msd': msd_list}
    csv_data = pd.DataFrame(metric_data, idx_list)
    csv_data.to_csv('metrics_thickness2/metrics_Resunet_ce_NoS.csv')