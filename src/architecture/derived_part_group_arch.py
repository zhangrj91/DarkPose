import torch
import torch.nn as nn
import torch.nn.functional as F

from .operators import OPS, Connections
from .grouping_cell import GCell
from .grouping_cell import MixedOp, Connection_Combination

import logging

logger = logging.getLogger(__name__)


class Derived_Part_Grouping_Arch(nn.Module):

    def __init__(self, part_grouping_arch):
        super(Derived_Part_Grouping_Arch, self).__init__()
        self.arch_name = part_grouping_arch.arch_name
        self.criterion = part_grouping_arch.criterion
        self.group_config = part_grouping_arch.group_config

        self.grouping_stage_num = part_grouping_arch.grouping_stage_num
        self.next_parts_num = part_grouping_arch.next_parts_num
        self.pre_parts_num = part_grouping_arch.pre_parts_num
        self.operators = part_grouping_arch.operators
        self.channel = part_grouping_arch.channel
        self.search_alpha = part_grouping_arch.search_alpha
        self.search_beta = part_grouping_arch.search_beta

        self.alphas = part_grouping_arch.alphas
        self.betas = part_grouping_arch.betas
        self.num_ops = len(self.operators)  # 3

        self.part_groups = nn.ModuleList()
        for next_num in range(len(self.alphas)):
            cells = nn.ModuleList()
            for pre_num in range(len(self.alphas[0])):
                if isinstance(part_grouping_arch.parts_group[next_num]._ops[pre_num], MixedOp):
                    cells.append(part_grouping_arch.parts_group[next_num]._ops[pre_num].ops[self.alphas[next_num][pre_num].argmax().data])
                else:
                    cells.append(OPS["Zero"](self.channel))
            self.part_groups.append(cells)

        self.cell_connections = Connection_Combination()

    def forward(self, input):
        input_num = len(input)
        assert input_num == self.pre_parts_num

        output = []

        input = list(input)
        for i, cell in enumerate(self.part_groups):
            mixed_op = []
            for j, h in enumerate(input):
                mixed_op.append(cell[j](h))
            output.append(self.cell_connections(mixed_op, self.betas[i]))

        return output

    def new(self):
        """
        create a new model and initialize it with current arch parameters.
        However, its weights are left untouched.
        :return:
        """
        group_config = self.group_config
        model_new = Derived_Part_Grouping_Arch(self.criterion, **group_config)

        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def arch_parameters(self):
        return self._arch_parameters

    def part_groups_(self):
        return self.part_groups


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

        # logger.info("alpha value is {}".format(self.alphas))
        logger.info("beta value is {}".format(self.betas))

        # if self.alphas.size(0) == 1:
        #     operators = [(x, round(y, 3)) for (x, y) in zip(self.operators_used, value[0])]  # squeeze
        #     logger.info("=>the single edge is mixed in:{}".format(operators))
        # if self.alphas.size(0) > 1:
        #
        #     for id, alpha in enumerate(value):
        #         operators = [(x, round(y, 3)) for (x, y) in zip(self.operators_used, alpha)]
        #         logger.info("=>the {} edge is mixed in:{}".format(id + 1, operators))