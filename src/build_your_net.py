from thop import profile
import torch

from .network_factory.resnet_feature import BackBone_ResNet
from .network_factory.mobilenet_v2_feature import  BackBone_MobileNet
from .network_factory.part_group_network import Part_Group_Network
from .network_factory.pose_hrnet import BackBone_HRNet

from myutils import  get_model_summary

import logging
logger = logging.getLogger(__name__)

def bulid_up_network(config,criterion):

    # if config.model.use_backbone:
    #     logger.info("backbone of architecture is {}".format(config.model.backbone_net_name))

    if config.model.backbone_net_name=="resnet":
        backbone = BackBone_ResNet(config,is_train=True)

    if config.model.backbone_net_name=="mobilenet_v2":
        backbone = BackBone_MobileNet(config,is_train=True)

    if config.model.backbone_net_name=="hrnet":
        backbone = BackBone_HRNet(config,is_train=True)

    Arch = Part_Group_Network(config.model.keypoints_num, criterion, backbone, **config.model.part_group_config)

    if config.model.use_pretrained:
        Arch.load_pretrained(config.model.pretrained)

    
        
    logger.info("\n\nbackbone: params and flops")
    logger.info(get_model_summary(backbone,torch.randn(1, 3, config.model.input_size.h,config.model.input_size.w)))

    logger.info("\n\nwhole architecture: params and flops")
    logger.info(get_model_summary(Arch,torch.randn(1, 3, config.model.input_size.h,config.model.input_size.w)))

    logger.info("=========== thop statistics ==========")
    dump = torch.randn(1, 3, config.model.input_size.h,config.model.input_size.w)
    flops, params = profile( backbone, inputs=(dump,),  )
    logger.info(">>> total params of BackBone: {:.2f}M\n>>> total FLOPS of Backbone: {:.3f} G\n".format(
                    (params / 1000000.0),(flops / 1000000000.0)))
    flops, params = profile(Arch, inputs=(dump,),  )
    logger.info(">>> total params of Whole Model: {:.2f}M\n>>> total FLOPS of Model: {:.3f} G\n".format(
                        (params / 1000000.0),(flops / 1000000000.0)))

    return Arch
