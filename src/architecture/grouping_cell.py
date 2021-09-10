import torch.nn.functional as F
import torch.nn as nn
from .operators import *


class MixedOp(nn.Module):
    def __init__(self, channel, ops_used):
        super(MixedOp, self).__init__()
        self.ops = nn.ModuleList()
        self.ops_used = ops_used
        self.channel = channel
        for name in ops_used:
            op = OPS[name](channel)  # only consideration the input channel in a cell
            if 'pool' in name:
                op = nn.Sequential(op, nn.BatchNorm2d(channel, affine=False))

            self.ops.append(op)

    def forward(self, input, alphas):
        # output = sum(all of alpha *operator(x))

        # normalize
        alphas = F.softmax(alphas, dim=-1)

        output = len(self.ops_used) * sum([alpha * op.to(input.device)(input) for alpha, op in zip(alphas, self.ops)])  # len_use
        return output

class EmptyOp(nn.Module):
    def __init__(self, channel, ops_used):
        super(EmptyOp, self).__init__()
        self.ops = nn.ModuleList()
        self.ops_used = ops_used
        self.channel = channel
        for name in ops_used:
            op = OPS["Zero"](channel)  # only consideration the input channel in a cell
            if 'pool' in name:
                op = nn.Sequential(op, nn.BatchNorm2d(channel, affine=False))

            self.ops.append(op)

    def forward(self, input, alphas):
        # output = sum(all of alpha *operator(x))

        # normalize
        alphas = F.softmax(alphas, dim=-1)

        output = len(self.ops_used) * sum([alpha * op.to(input.device)(input) for alpha, op in zip(alphas, self.ops)])  # len_use
        return output

# Connection_Combination is outside the cell

class Connection_Combination(nn.Module):
    "combine 8 nodes of part by 'beta' weights to become an input node "

    def __init__(self, ):
        super(Connection_Combination, self).__init__()

    def forward(self, prevs, betas):
        betas = F.softmax(betas, dim=-1)
        mix = 0
        for i in range(len(prevs)):
            mix += betas[i] * prevs[i]

        mix = F.relu(mix)

        return mix

class GCell(nn.Module):

    def __init__(self, grouping_stage_num, part_number, prev_part_num, c_prev, channel,
                 operators_used = ["zero"]):

        super(GCell, self).__init__()

        #level number this cell belong to
        self.grouping_stage_num = grouping_stage_num
        #which part this cell represent
        self.part_num = part_number

        # output channel
        self.channel = channel

        # previous part level part amount
        self._prev_fmultipliers = c_prev

        # self.block_multiplier = block_multiplier
        self._ops = nn.ModuleList()

        for i in range(prev_part_num):
            op = MixedOp(channel, operators_used)#op是构建两个节点之间的混合
            self._ops.append(op)#所有边的混合操作添加到ops

        self.cell_connections = Connection_Combination()

        # self.ReLUConvBN = ReLUConvBN(self.C_in, self.C_out, 1, 1, 0)

        # self._initialize_weights()

    def forward(self, input, alpha_weights, beta_weights):
        input = list(input)
        mixed_op = []
        for j, h in enumerate(input):
            mixed_op.append(self._ops[j](h, alpha_weights[j]))
        output = self.cell_connections(mixed_op, beta_weights)

        return output
