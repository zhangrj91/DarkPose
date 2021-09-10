import torch
import torch.nn as nn
import os
from src.architecture.part_group_arch import *
import logging
from .blocks import *

logger = logging.getLogger(__name__)
BN_MOMENTUM = 0.1

class Merge(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(Merge, self).__init__()
        self.conv = Conv(x_dim, y_dim, 1, relu=False, bn=False)

    def forward(self, x):
        return self.conv(x)

class Part_Group_Level(nn.Module):

    def __init__(self, block, origin_parts = None, **group_level_config):
        super(Part_Group_Level, self).__init__()

        self.in_channel = group_level_config['channel']
        self.channel = group_level_config['channel']
        self.out_channel = group_level_config['out_channel']
        self.num_branches = group_level_config['parts_num']
        self.num_blocks = group_level_config['block_num']

        self.origin_parts = origin_parts

        self.branches, self.preds, self.merge_features, self.merge_preds = self._make_branches(
            self.num_branches, block, self.num_blocks, self.in_channel, self.channel, self.out_channel)

        self.relu = nn.ReLU(True)

        self.group_final_pred = Conv(self.out_channel, self.out_channel, 1, 1, relu=False)

    def _make_one_branch(self, branch_index, block, num_blocks, in_channel, channel,
                         stride=1):
        # downsample = None
        # layers = []
        #
        # #1*1 conv change channel
        # if in_channel != out_channel:
        #     layers.append(nn.Conv2d(
        #             self.in_channel, out_channel,
        #             kernel_size=1, stride=stride, bias=False
        #         ))
        #     layers.append(nn.BatchNorm2d(out_channel, momentum=BN_MOMENTUM))
        #
        # #BottleNeck
        # if stride != 1 or self.in_channel != out_channel * block.expansion:
        #     downsample = nn.Sequential(
        #         nn.Conv2d(
        #             self.in_channel, out_channel * block.expansion,
        #             kernel_size=1, stride=stride, bias=False
        #         ),
        #         nn.BatchNorm2d(out_channel * block.expansion, momentum=BN_MOMENTUM),
        #     )
        #
        # layers.append(block(self.in_channel, out_channel, stride, downsample))
        # in_channel = out_channel * block.expansion
        # for i in range(num_blocks - 1):
        #     layers.append(block(in_channel, out_channel))
        #
        # return nn.Sequential(*layers)

        layers = []

        #1*1 conv change channel
        if in_channel != channel:
            layers.append(nn.Conv2d(
                    self.in_channel, channel,
                    kernel_size=1, stride=stride, bias=False
                ))
            layers.append(nn.BatchNorm2d(channel, momentum=BN_MOMENTUM))

        for i in range(num_blocks):
            layers.append(block(in_channel, channel))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, in_channel, channel, out_channel):
        branches = []
        merge_features = []
        merg_preds = []
        preds = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, in_channel, channel)
            )
            if self.origin_parts is None:
                preds.append(Conv(channel, out_channel, 1, 1, relu=False, bn=True))
                merg_preds.append(Merge(out_channel, channel))
            else:
                preds.append(Conv(channel, len(self.origin_parts[i]), 1, 1, relu=False, bn=True))
                merg_preds.append(Merge(len(self.origin_parts[i]), channel))

            merge_features.append(Merge(channel, channel))

        return nn.ModuleList(branches), nn.ModuleList(preds), nn.ModuleList(merge_features), nn.ModuleList(merg_preds)

    def get_num_inchannels(self):
        return self.channel

    def forward(self, x, joints_used = None):
        assert self.origin_parts is not None or joints_used is not None
        merged_features = []
        if self.origin_parts is not None:
            predictions = torch.zeros(
                size=(self.num_branches, x.size(0), self.out_channel, x.size(2), x.size(3))
            ).to(x.device)
            for i in range(self.num_branches):
                predictions_new = torch.zeros(
                    size=(self.num_branches, x.size(0), self.out_channel, x.size(2), x.size(3))
                ).to(x.device)
                feature = self.branches[i](x)
                prediction = self.preds[i](feature)
                predictions_new[i,:,self.origin_parts[i],:,:] = prediction
                predictions += predictions_new
                merged_features.append(x + self.merge_features[i](feature) + self.merge_preds[i](prediction))
        else:
            predictions = torch.zeros(
                size=(self.num_branches, x[0].size(0), self.out_channel, x[0].size(2), x[0].size(3))
            ).to(x[0].device)
            for i in range(self.num_branches):
                predictions_new = torch.zeros(
                    size=(self.num_branches, x[0].size(0), self.out_channel, x[0].size(2), x[0].size(3))
                ).to(x[0].device)
                feature = self.branches[i](x[i])
                prediction = self.preds[i](feature)
                predictions_new[i, :, joints_used[i], :, :] = prediction[:, joints_used[i]]
                predictions += predictions_new
                merged_features.append(x[i] + self.merge_features[i](feature) + self.merge_preds[i](prediction))

        return self.group_final_pred(torch.sum(predictions, dim=0)), merged_features
