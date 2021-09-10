# -*- coding: UTF-8 -*-
#!/usr/bin/python
# Written by Sen Yang (yangsenius@seu.edu.cn)

import torch
import numpy as np 
import logging
from src.network_factory.body_parts import parts_mode

logger = logging.getLogger(__name__)
# data control-flow mode
#@torch.jit.script
class MSELoss(torch.nn.Module):

    def __init__(self,):
       super(MSELoss,self).__init__()
        
    def forward(self, preds, heatmap_gt, weight):
        # logger.info("(preds - heatmap_gt)**2 is {}".format(((preds - heatmap_gt)**2).mean(dim=3).mean(dim=2)))
        # logger.info("loss is {}".format(weight * ((preds-heatmap_gt)**2).mean(dim=3).mean(dim=2)))
        losses = 0.5 * weight * ((preds-heatmap_gt)**2).mean(dim=3).mean(dim=2)
        back_loss = losses.mean(dim=1).mean(dim=0)
        return back_loss


class Weight_MSELoss(torch.nn.Module):

    def __init__(self, ):
        super(Weight_MSELoss, self).__init__()

    def forward(self, preds, heatmap_gt, weight, joints_used):
        back_loss = 0
        for i, joints in enumerate(joints_used):
            joints_weight = torch.zeros((heatmap_gt.size(0), heatmap_gt.size(1))).to(heatmap_gt.device)
            for j in joints:
                joints_weight[:,j] = 1
            loss = 0.5 * weight * joints_weight * ((preds[:,i] - heatmap_gt) ** 2).mean(dim=3).mean(dim=2)
            losses = loss.mean(dim=1).mean(dim=0)
            back_loss += losses
        return back_loss

# class Weight_MSELoss(torch.nn.Module):
#
#     def __init__(self, ):
#         super(Weight_MSELoss, self).__init__()
#
#     def forward(self, preds, heatmap_gt, weight, joints_used):
#         back_loss = 0
#         for i, p in enumerate(preds):
#             predictions_new = torch.zeros(
#                 size=heatmap_gt.size()
#             ).to(p.device)
#             joints_weight = torch.zeros((heatmap_gt.size(0), heatmap_gt.size(1))).to(heatmap_gt.device)
#             joints_weight[:,joints_used[i]] = 1
#             predictions_new[:, joints_used[i], :, :] = p
#             loss = 0.5 * weight * joints_weight * ((predictions_new - heatmap_gt) ** 2).mean(dim=3).mean(dim=2)
#             losses = loss.mean(dim=1).mean(dim=0)
#             back_loss += losses
#         return back_loss

def test():

    # preds = torch.randn(24,16,64,64)
    # heatmap_gt = torch.randn(24,16,64,64)
    # weight = torch.ones((24,16))
    # weight[1][2] = 0
    # joints_weight = torch.ones((24,16))
    # joints_weight[1][1] = 0
    # MSELoss = Weight_MSELoss()
    # #traced_loss,losses = torch.jit.trace(MSELoss, (preds , heatmap_gt, weight ))
    # loss = MSELoss(preds , heatmap_gt, weight, joints_weight)
    # print(loss)
    # origin_p = torch.zeros(
    #     size=(8, 12, 16, 64, 64)
    # )
    # a = torch.ones(12, 2, 64, 64)
    # a[:, 1, :, :] = 2
    # id = [11, 14]
    # origin_p[1, :, id, :, :] = a
    # print(origin_p[1][1][14])
    parts = parts_mode("mpii", 8)
    origin_parts = []
    for part_name, nums in parts.items():
        origin_parts.append(nums)

if __name__ == '__main__':
    test()
    