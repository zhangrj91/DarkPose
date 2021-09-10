import copy

import torch
import torch.nn as nn
import os
from src.architecture.part_group_arch import *
from .part_group_level import *
from .body_parts import parts_mode
from .blocks import *
from src.network_factory.pose_hrnet import BasicBlock
from src.architecture.grouping_cell import EmptyOp
from src.network_factory.derived_part_group_level import Derived_Part_Group_Level
from src.architecture.derived_part_group_arch import Derived_Part_Grouping_Arch

import logging

logger = logging.getLogger(__name__)


class Stacked_Group_Network(nn.Module):

    def __init__(self, pruned_arch):

        super(Stacked_Group_Network, self).__init__()

        self.part_group_config = pruned_arch.part_group_config
        self.level_1_config = pruned_arch.part_group_config['level_1_config']
        self.grouping_1_2_config = pruned_arch.part_group_config['grouping_1_2_config']
        self.level_2_config = pruned_arch.part_group_config['level_2_config']
        self.grouping_2_3_config = pruned_arch.part_group_config['grouping_2_3_config']
        self.level_3_config = pruned_arch.part_group_config['level_3_config']
        self.grouping_3_4_config = pruned_arch.part_group_config['grouping_3_4_config']
        self.level_4_config = pruned_arch.part_group_config['level_4_config']
        self.backbone = pruned_arch.backbone
        self.criterion = pruned_arch.criterion
        self.keypoints_num = pruned_arch.keypoints_num

        backbone_feature_num = self.backbone.feature_num

        self.level_num = pruned_arch.part_group_config['level_num']

        self.level_1_parts_num = self.level_1_config['parts_num']

        self.block = Residual  # Residual Block

        # origin parts grouped by huaman
        self.parts = parts_mode(pruned_arch.part_group_config['dataset_name'], self.level_1_parts_num)

        logger.info("\nbody parts is {}".format(self.parts))

        self.origin_predict_layer = Conv(self.level_1_config['channel'], self.keypoints_num, 1, 1, relu=False)

        self.origin_parts = pruned_arch.origin_parts
        self.all_parts = pruned_arch.all_parts

        self.group_levels = pruned_arch.group_levels

        self.groupings = pruned_arch.groupings
        logger.info(self.groupings)

        self.group_levels_1 = copy.deepcopy(pruned_arch.group_levels)
        self.groupings_1 = copy.deepcopy(pruned_arch.groupings)

        self.group_levels_2 = copy.deepcopy(pruned_arch.group_levels)
        self.groupings_2 = copy.deepcopy(pruned_arch.groupings)

        self.prune_masks = pruned_arch.prune_masks

    def forward(self, x):

        shared_feature = self.backbone(x)

        all_level_predictions = []

        origin_predictions = self.origin_predict_layer(shared_feature)

        all_level_predictions.append(origin_predictions)

        predictions, features = self.group_levels[0](shared_feature)

        all_level_predictions.append(predictions)

        for i in range(1, self.level_num):
            features = self.groupings[i - 1](features)

            predictions, features = self.group_levels[i](features)

            all_level_predictions.append(predictions)

        predictions, features = self.group_levels_1[0](features[0])

        all_level_predictions.append(predictions)

        for i in range(1, self.level_num):
            features = self.groupings_1[i - 1](features)

            predictions, features = self.group_levels_1[i](features)

            all_level_predictions.append(predictions)

        predictions, features = self.group_levels_1[0](features[0])

        all_level_predictions.append(predictions)

        for i in range(1, self.level_num):
            features = self.groupings_2[i - 1](features)

            predictions, features = self.group_levels_2[i](features)

            all_level_predictions.append(predictions)

        # return predictions
        return torch.stack(all_level_predictions, 1)

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
