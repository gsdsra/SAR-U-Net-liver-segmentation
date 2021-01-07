import argparse
from utils.trainer import trainer
from utils.dataset import LITS_dataset, make_dataloaders
import torch
import torch.nn as nn
# from model.unet import UNet
# from model.attention_unet.attention_unet import AttU_Net
# from model.resunet.resunet import DeepResUNet
# from model.se_resunet_plus.se_resunet_plus import SeResUNet
# from model.resunet.res_net import Res_UNet
# from model.se_p_attresunet.se_p_attresunet import SE_P_AttU_Net
# from model.unet.unet_model import UNet
# from model.resunet_plusplus.resnet_plusplus import ResUnetPlusPlus
import numpy as np
import torch.nn as nn
from models.se_p_resunet.se_p_resunet import Se_PPP_ResUNet
# from model.FCN.FCN import *


if __name__ == '__main__':
    device = torch.device('cuda:0')

    LEARNING_RATE = 1e-3
    LR_DECAY_STEP = 2
    LR_DECAY_FACTOR = 0.5
    WEIGHT_DECAY = 5e-4
    BATCH_SIZE = 4
    MAX_EPOCHS = 60
    # MODEL = DeepResUNet(2).to(device)
    # MODEL = AttU_Net(1,2).to(device)
    # MODEL = SeResUNet(1, 2, deep_supervision=False).to(device)
    # MODEL = SE_P_AttU_Net(1,2).to(device)
    # MODEL = UNet(1, 2).to(device)
    # vgg_model = VGGNet()
    # MODEL = FCNs(pretrained_net=vgg_model, n_class=2).to(device)
    # MODEL = ResUnetPlusPlus(1,filters=[32, 64, 128, 256, 512]).to(device)
    # MODEL = Res_UNet(1,2).to(device)
    MODEL = Se_PPP_ResUNet(1,2,deep_supervision=False).to(device)
    OPTIMIZER = torch.optim.Adam(MODEL.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    LR_SCHEDULER = torch.optim.lr_scheduler.StepLR(OPTIMIZER, step_size=LR_DECAY_STEP, gamma=LR_DECAY_FACTOR)
    # CRITERION = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([0.75,1])).float(),size_average=True).to(device)
    CRITERION = nn.CrossEntropyLoss().to(device)
    # CRITERION = DiceLoss().to(device)
    # CRITERION = TverskyLoss().to(device)
    # CRITERION = WCELoss().to(device)

    tr_path_raw = 'fixed/save_first_rot/tr/raw'
    tr_path_label = 'fixed/save_first_rot/tr/label'
    ts_path_raw = 'fixed/save_first_rot/ts/raw'
    ts_path_label = 'fixed/save_first_rot/ts/label'

    # checkpoints_dir = 'checkpoints'
    checkpoints_dir = 'final checkpoints/Se_PPP_ResUNet_ce_NoS'
    checkpoint_frequency = 1000
    dataloaders = make_dataloaders(tr_path_raw, tr_path_label, ts_path_raw, ts_path_label, BATCH_SIZE, n_workers=4)
    comment = 'liver_segmentation_Se_PPP_ResUNet_ce_NoS_on_LITS_dataset_'
    verbose_train = 1
    verbose_val = 500

    trainer = trainer(MODEL, OPTIMIZER, LR_SCHEDULER, CRITERION, dataloaders, comment, verbose_train, verbose_val, checkpoint_frequency, MAX_EPOCHS, checkpoint_dir=checkpoints_dir, device=device)
    trainer.train()



