import torch
import torch.nn as nn
import torch.nn.functional as F

from .operators import OPS, Connections
from .grouping_cell import GCell

import logging

logger = logging.getLogger(__name__)


class Part_Grouping_Arch(nn.Module):

    def __init__(self, criterion, name="part_group", **Group_Config):
        super(Part_Grouping_Arch, self).__init__()
        self.arch_name = name
        self.criterion = criterion
        self.group_config = Group_Config

        self.grouping_stage_num = Group_Config['grouping_stage_num']
        self.next_parts_num = Group_Config['next_parts_num']
        self.pre_parts_num = Group_Config['pre_parts_num']
        self.operators = Group_Config['operators']
        self.channel = Group_Config['channel']
        self.search_alpha = Group_Config['search_alpha']
        self.search_beta = Group_Config['search_beta']

        self.parts_group = nn.ModuleList()
        self.alphas = []
        self.betas = []
        self.num_ops = len(self.operators)  # 2
        for i in range(self.next_parts_num):
            self.parts_group.append(GCell(self.grouping_stage_num, i, self.pre_parts_num,
                                          self.channel, self.channel, self.operators))
            self.alphas.append(nn.Parameter(1e-3 * torch.randn(self.pre_parts_num, self.num_ops)))
            self.betas.append(nn.Parameter(1e-3 * torch.randn(self.pre_parts_num)))

        self._arch_parameters = []
        if self.search_alpha:
            self._arch_parameters.append(self.alphas)
        if self.search_beta:
            self._arch_parameters.append(self.betas)

    def forward(self, input, prune_mask):
        input_num = len(input)
        assert input_num == self.pre_parts_num

        output = []

        for i in range(self.next_parts_num):
            output.append(self.parts_group[i](input, self.alphas[i], self.betas[i]*prune_mask[i]))

        return output

    def new(self):
        """
        create a new model and initialize it with current arch parameters.
        However, its weights are left untouched.
        :return:
        """
        group_config = self.group_config
        model_new = Part_Grouping_Arch(self.criterion, **group_config)

        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def arch_parameters(self):
        return self._arch_parameters

    def part_groups_(self):
        return self.parts_group

    def arch_parameters_random_search(self):

        self._arch_parameters = []

        if self.search_alpha:
            for i in range(self.next_parts_num):
                self.alphas.append(nn.Parameter(1e-3 * torch.randn(self.pre_parts_num, self.num_ops)))
        if self.search_beta:
            for i in range(self.next_parts_num):
                self.betas.append(nn.Parameter(1e-3 * torch.randn(self.pre_parts_num)))

        self._arch_parameters.append(self.alphas)
        self._arch_parameters.append(self.betas)

    # def loss(self, x, target, target_weight):
    #
    #     kpts = self(x)
    #     loss = self.criterion(kpts, target, target_weight)
    #
    #     return loss

    #################  show information function #######################################
    def _print_info(self):
        logger.info(
            "\n========================== {} Architecture Configuration ======================".format(self.arch_name))
        logger.info(
            "grouping_stage_number is {} ".format(self.grouping_stage_num))
        logger.info(
            "previous level parts group has {} parts, and the next parts group has {} parts".format(self.pre_parts_num, self.next_parts_num)
        )
        logger.info(
            "Channel for this grouping arch is {}".format(self.channel)
        )
        logger.info("Search Space of ALPHA is {}, optimization is {}".format(self.alphas[0].shape,
                                                                                               self.search_alpha))
        logger.info("operators used  in each  node are {}".format(self.operators))
        logger.info(">>> total params of Model: {:.2f}M".format(sum(p.numel() for p in self.parameters()) / 1000000.0))
        self._show_weight(original_value=True)
        logger.info("=========================================================================++++++++++++")

    def _show_weight(self, original_value=False):

        logger.info("alpha value is {}".format(self.alphas))
        logger.info("beta value is {}".format(self.betas))

        # if self.alphas.size(0) == 1:
        #     operators = [(x, round(y, 3)) for (x, y) in zip(self.operators_used, value[0])]  # squeeze
        #     logger.info("=>the single edge is mixed in:{}".format(operators))
        # if self.alphas.size(0) > 1:
        #
        #     for id, alpha in enumerate(value):
        #         operators = [(x, round(y, 3)) for (x, y) in zip(self.operators_used, alpha)]
        #         logger.info("=>the {} edge is mixed in:{}".format(id + 1, operators))