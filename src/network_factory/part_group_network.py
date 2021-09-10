import torch
import torch.nn as nn
import os
from src.architecture.part_group_arch import *
from .part_group_level import *
from .body_parts import parts_mode
from .blocks import *
from src.network_factory.pose_hrnet import BasicBlock
from src.architecture.grouping_cell import EmptyOp

import logging

logger = logging.getLogger(__name__)


class Part_Group_Network(nn.Module):

    def __init__(self, keypoints_num, criterion, backbone, **part_group_config):

        super(Part_Group_Network, self).__init__()

        self.part_group_config = part_group_config
        self.level_1_config = part_group_config['level_1_config']
        self.grouping_1_2_config = part_group_config['grouping_1_2_config']
        self.level_2_config = part_group_config['level_2_config']
        self.grouping_2_3_config = part_group_config['grouping_2_3_config']
        self.level_3_config = part_group_config['level_3_config']
        self.grouping_3_4_config = part_group_config['grouping_3_4_config']
        self.level_4_config = part_group_config['level_4_config']
        self.grouping_4_5_config = part_group_config['grouping_4_5_config']
        self.level_5_config = part_group_config['level_5_config']
        self.backbone = backbone
        self.criterion = criterion
        self.keypoints_num = keypoints_num
        self.use_main_parts = True

        backbone_feature_num = self.backbone.feature_num

        self.level_num = part_group_config['level_num']

        self.level_1_parts_num = self.level_1_config['parts_num']

        self.block = Residual  # Residual Block

        # origin parts grouped by huaman
        self.parts = parts_mode(part_group_config['dataset_name'], self.level_1_parts_num)

        logger.info("\nbody parts is {}".format(self.parts))

        self.origin_predict_layer = Conv(self.level_1_config['channel'], self.keypoints_num, 1, 1, relu=False)

        self.origin_parts = []
        for part_name, nums in self.parts.items():
            self.origin_parts.append(nums)

        self.all_parts = []
        self.all_parts.append([i for i in range(self.keypoints_num)])

        self.group_levels = nn.ModuleList()
        self.group_levels.append(Part_Group_Level(self.block, self.origin_parts, **self.level_1_config))
        for i in range(1, self.level_num):
            self.group_levels.append(Part_Group_Level(self.block, **eval('self.level_{}_config'.format(i + 1))))

        self.groupings = nn.ModuleList()
        for i in range(self.level_num - 1):
            self.groupings.append(
                Part_Grouping_Arch(self.criterion, **eval('self.grouping_{}_{}_config'.format(i + 1, i + 2))))

        self.all_parts.append(self.origin_parts)
        previous_part_used = self.origin_parts

        for i in range(1, self.level_num):
            part_groups = self.groupings[i - 1].part_groups_()
            level_parts_used = []
            for part_cell in part_groups:
                cell_parts_used = []
                for j, part_op in enumerate(part_cell._ops):
                    if not isinstance(part_op, EmptyOp):
                        for joint in previous_part_used[j]:
                            if joint not in cell_parts_used:
                                cell_parts_used.append(joint)
                level_parts_used.append(cell_parts_used)

            previous_part_used = level_parts_used
            self.all_parts.append(level_parts_used)

        self.prune_masks = []
        for i in range(self.level_num-1):
            self.prune_masks.append(torch.ones([eval("self.level_{}_config".format(i+2))['parts_num'], eval("self.level_{}_config".format(i+1))['parts_num']]))

        self.main_parts = []
        self.main_part_features = []

        self.each_group_parts = []
        self.each_group_features = []

    def forward(self, x):

        shared_feature = self.backbone(x)

        all_level_predictions = []

        origin_predictions = self.origin_predict_layer(shared_feature)

        all_level_predictions.append(origin_predictions)

        predictions, features = self.group_levels[0](shared_feature)

        all_level_predictions.append(predictions)

        self.main_part_features = []
        for i in range(1, self.level_num):
            main_part_feature = []
            if self.use_main_parts and self.main_parts:
                for j in range(len(self.main_parts[i-1])):
                    main_part_feature.append(features[self.main_parts[i-1][j]])
                self.main_part_features.append(main_part_feature)
                # groups = []
                # for p in self.prune_masks[i-1]:
                #     each_group = []
                #     for k in range(len(p)):
                #         if p[k] == 1:
                #             each_group.append(features[k])
                #     groups.append(each_group)
                # self.each_group_parts.append(groups)
            features = self.groupings[i - 1](features, self.prune_masks[i-1])
            predictions, features = self.group_levels[i](features, self.all_parts[i])
            all_level_predictions.append(predictions)

        # return predictions
        return torch.stack(all_level_predictions, 1)

    def new(self):
        """
        create a new model and initialize it with current arch parameters.
        However, its weights are left untouched.
        :return:
        """
        part_group_config = self.part_group_config
        model_new = Part_Group_Network(self.keypoints_num, self.criterion, self.backbone, **part_group_config)

        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def arch_parameters(self):

        self.all_group_arch_parameters = []

        for grouping in self.groupings:

            if grouping.search_alpha:
                self.all_group_arch_parameters.append(grouping.alphas)
            if grouping.search_beta:
                self.all_group_arch_parameters.append(grouping.betas)

        return self.all_group_arch_parameters

    def arch_parameters_random_search(self):

        for grouping in self.groupings:

            # beta control the fabrics outside the cell

            grouping._arch_parameters = []
            if grouping.search_alpha:
                grouping.alphas = nn.Parameter(torch.randn(grouping.parts_num, grouping.num_ops))
                grouping._arch_parameters.extend(grouping.alphas)

    def loss(self, x, target, target_weight, info=None):
        kpts = self(x)
        loss = self.criterion(kpts, target, target_weight, self.all_parts)
        if self.use_main_parts and self.main_part_features:
            n = 0
            l = 0
            for mfs in self.main_part_features:
                for i in range(len(mfs)):
                    for j in range(i+1, len(mfs)):
                        losses = 0.5 * ((mfs[i] - mfs[j]) ** 2).mean(dim=3).mean(dim=2)
                        back_loss = losses.mean(dim=1).mean(dim=0)
                        l = l + back_loss
                        n = n + 1
                        # l += torch.mul(fs[i], fs[j])
                        # n += 1
            loss = loss - l/n

            # l_e = 0
            # n_e = 0
            # for efs_level in self.each_group_parts:
            #     for efs in efs_level:
            #         for i in range(len(efs)):
            #             for j in range(i+1, len(efs)):
            #                 losses = 0.5 * ((efs[i] - efs[j]) ** 2).mean(dim=3).mean(dim=2)
            #                 back_loss = losses.mean(dim=1).mean(dim=0)
            #                 l_e = l_e + back_loss
            #                 n_e = n_e + 1
            # loss = loss + l_e/n_e

        return loss

    def load_pretrained(self, pretrained=''):

        if os.path.exists(pretrained):
            pretrained_state_dict = torch.load(pretrained)
            model_dict = {}
            state_dict = self.state_dict()
            for k, v in pretrained_state_dict.items():
                if k in state_dict:
                    if 'final_layer' in k:  # final_layer is excluded
                        continue
                    model_dict[k] = v
            state_dict.update(model_dict)
            self.load_state_dict(state_dict)

            logger.info('=> loading pretrained model in {}'.format(pretrained))

        else:
            logger.info('=> no pretrained found in {}!'.format(pretrained))

    def _print_info(self):
        if hasattr(self.backbone, "_print_info"):
            self.backbone._print_info()
        for g in self.groupings:
            g._print_info()
        logger.info("---------------------------------------")
        for i, parts in enumerate(self.all_parts):
            logger.info('group parts no.{} is {}'.format(i, parts))
