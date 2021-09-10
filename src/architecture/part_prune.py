import torch
import torch.nn as nn
import torch.nn.functional as F
from .part_group_arch import Part_Grouping_Arch
from .grouping_cell import EmptyOp
from src.network_factory.derived_part_group_network import Derived_Part_Group_Network

import logging
logger = logging.getLogger(__name__)

def Choose_Main_Part(Arch, prune_level):
    with torch.no_grad():
        all_parts = []
        all_parts.append(Arch.all_parts[0])
        for n, m in Arch.named_modules():
            # find all the 'part_group_Arch'
            if isinstance(m, Part_Grouping_Arch) and str(prune_level) in n:
                prune_mask = torch.ones([len(m.alphas), len(m.parts_group[0]._ops)])
                logger.info("n in Arch.named_modules is {}".format(n))
                # one-shot-search
                logger.info("n in Arch.named_modules is {}".format(n))
                # one-shot-search
                scores = torch.ones([len(m.alphas), len(m.alphas[0])])
                for i, (alpha, beta) in enumerate(zip(m.alphas, m.betas)):
                    alpha = F.softmax(alpha, dim=-1)
                    beta = F.softmax(beta, dim=-1)
                    for j, (a, b) in enumerate(zip(alpha, beta)):
                        score = abs((1 - a[0]) * b)
                        scores[i][j] = score
                ranking = torch.sort(scores, descending=True)[1]
                chosen_main_part = []
                chosen_main_part.append(ranking[0][0])
                for i in range(1, len(ranking)):
                    for r in ranking[i]:
                        if r not in chosen_main_part:
                            chosen_main_part.append(r)
                            break

    return chosen_main_part

def Prune_new(Arch, prune_level, alpha_prune = False, chosen_main_part = None):
    with torch.no_grad():
        all_parts = []
        all_parts.append(Arch.all_parts[0])
        for n, m in Arch.named_modules():
            # find all the 'part_group_Arch'
            if isinstance(m, Part_Grouping_Arch) and str(prune_level) in n:
                prune_mask = torch.ones([len(m.alphas), len(m.parts_group[0]._ops)])
                logger.info("n in Arch.named_modules is {}".format(n))
                # one-shot-search
                logger.info("n in Arch.named_modules is {}".format(n))

                for i, (alpha, beta) in enumerate(zip(m.alphas, m.betas)):
                    alpha = F.softmax(alpha, dim=-1)
                    # lowest = 100
                    # lowest_location = 0
                    # if beta_prune == True:
                    #     for j, b in enumerate(beta):
                    #         if prune_mask[i][j] != 0:
                    #             if lowest > abs(b):
                    #                 lowest_location = j
                    #                 lowest = abs(b)
                    #     prune_mask[i][lowest_location] = 0
                    #     m.parts_group[i]._ops[beta.argmin().data] = EmptyOp(m.parts_group[i]._ops[beta.argmin().data].channel,
                    #                                        m.parts_group[i]._ops[beta.argmin().data].ops_used)
                    if alpha_prune == True:
                        for j, aa in enumerate(alpha):
                            if aa.argmax().data == 0:
                                if j != chosen_main_part[i]:
                                    prune_mask[i][j] = 0

                all_parts.append(Arch.all_parts[1])
                previous_part_used = all_parts[1]
                Arch.prune_masks[prune_level] = prune_mask

                for i in range(Arch.level_num - 1):
                    left_parts = [p for p in all_parts[0]]
                    for p_m in Arch.prune_masks[i]:
                        for j in range(len(p_m)):
                            if p_m[j] == 1:
                                for joint in previous_part_used[j]:
                                    if joint in left_parts:
                                        left_parts.remove(joint)

                    best_groupings = {str(lp): -100.0 for lp in left_parts}
                    best_groupings_location = {str(lp):torch.zeros([2]) for lp in left_parts}
                    if left_parts:
                        for p in left_parts:
                            for j in range(len(previous_part_used)):
                                if p in previous_part_used[j]:
                                    for k, (alpha, beta) in enumerate(zip(m.alphas, m.betas)):
                                        alpha = F.softmax(alpha, dim=-1)
                                        beta = F.softmax(beta, dim=-1)
                                        score = abs((1 - alpha[j][0]) * (beta[j]))
                                        if score > best_groupings[str(p)]:
                                            best_groupings[str(p)] = score
                                            best_groupings_location[str(p)] = torch.tensor([k, j])

                    for l in best_groupings_location.values():
                        prune_mask[l[0]][l[1]] = 1

                    if i == prune_level:
                        Arch.prune_masks[i] = prune_mask
                    level_parts_used = []
                    for p_m in Arch.prune_masks[i]:
                        cell_parts_used = []
                        for j in range(len(p_m)):
                            if p_m[j] == 1:
                                for joint in previous_part_used[j]:
                                    if joint not in cell_parts_used:
                                        cell_parts_used.append(joint)
                        level_parts_used.append(cell_parts_used)

                    previous_part_used = level_parts_used
                    all_parts.append(level_parts_used)

                logger.info("left_parts are {}".format(left_parts))

        logger.info("new all_parts is {}".format(all_parts))
        Arch.all_parts = all_parts
        logger.info(Arch.prune_masks)

    return Arch

def Prune(Arch, prune_level, alpha_prune = False, beta_prune= False, beta_indice = 0):
    with torch.no_grad():
        all_parts = []
        all_parts.append(Arch.all_parts[0])
        for n, m in Arch.named_modules():
            # find all the 'part_group_Arch'
            if isinstance(m, Part_Grouping_Arch) and str(prune_level) in n:
                prune_mask = torch.ones([len(m.alphas), len(m.parts_group[0]._ops)])
                logger.info("n in Arch.named_modules is {}".format(n))
                # one-shot-search
                logger.info("n in Arch.named_modules is {}".format(n))
                # one-shot-search
                scores = torch.ones([len(m.alphas), len(m.alphas[0])])
                for i, (alpha, beta) in enumerate(zip(m.alphas, m.betas)):
                    alpha = F.softmax(alpha, dim=-1)
                    beta = F.softmax(beta, dim=-1)
                    for j, (a, b) in enumerate(zip(alpha, beta)):
                        score = abs((1 - a[0]) * b)
                        scores[i][j] = score
                ranking = torch.sort(scores, descending=True)[1]
                chosen_main_part = []
                chosen_main_part.append(ranking[0][0])
                for i in range(1, len(ranking)):
                    for r in ranking[i]:
                        if r not in chosen_main_part:
                            chosen_main_part.append(r)
                            break

                for i, (alpha, beta) in enumerate(zip(m.alphas, m.betas)):
                    alpha = F.softmax(alpha, dim=-1)
                    # lowest = 100
                    # lowest_location = 0
                    # if beta_prune == True:
                    #     for j, b in enumerate(beta):
                    #         if prune_mask[i][j] != 0:
                    #             if lowest > abs(b):
                    #                 lowest_location = j
                    #                 lowest = abs(b)
                    #     prune_mask[i][lowest_location] = 0
                    #     m.parts_group[i]._ops[beta.argmin().data] = EmptyOp(m.parts_group[i]._ops[beta.argmin().data].channel,
                    #                                        m.parts_group[i]._ops[beta.argmin().data].ops_used)
                    if alpha_prune == True:
                        for j, aa in enumerate(alpha):
                            if aa.argmax().data == 0:
                                if j != chosen_main_part[i]:
                                    prune_mask[i][j] = 0

                all_parts.append(Arch.all_parts[1])
                previous_part_used = all_parts[1]
                Arch.prune_masks[prune_level] = prune_mask

                for i in range(Arch.level_num - 1):
                    left_parts = [p for p in all_parts[0]]
                    for p_m in Arch.prune_masks[i]:
                        for j in range(len(p_m)):
                            if p_m[j] == 1:
                                for joint in previous_part_used[j]:
                                    if joint in left_parts:
                                        left_parts.remove(joint)

                    best_groupings = {str(lp): -100.0 for lp in left_parts}
                    best_groupings_location = {str(lp):torch.zeros([2]) for lp in left_parts}
                    if left_parts:
                        for p in left_parts:
                            for j in range(len(previous_part_used)):
                                if p in previous_part_used[j]:
                                    for k, (alpha, beta) in enumerate(zip(m.alphas, m.betas)):
                                        alpha = F.softmax(alpha, dim=-1)
                                        beta = F.softmax(beta, dim=-1)
                                        score = abs((1 - alpha[j][0]) * (beta[j]))
                                        if score > best_groupings[str(p)]:
                                            best_groupings[str(p)] = score
                                            best_groupings_location[str(p)] = torch.tensor([k, j])

                    for l in best_groupings_location.values():
                        prune_mask[l[0]][l[1]] = 1

                    if i == prune_level:
                        Arch.prune_masks[i] = prune_mask
                    level_parts_used = []
                    for p_m in Arch.prune_masks[i]:
                        cell_parts_used = []
                        for j in range(len(p_m)):
                            if p_m[j] == 1:
                                for joint in previous_part_used[j]:
                                    if joint not in cell_parts_used:
                                        cell_parts_used.append(joint)
                        level_parts_used.append(cell_parts_used)

                    previous_part_used = level_parts_used
                    all_parts.append(level_parts_used)

                logger.info("left_parts are {}".format(left_parts))

        logger.info("new all_parts is {}".format(all_parts))
        Arch.all_parts = all_parts
        logger.info(Arch.prune_masks)

    return Arch

def build_new_arch(Arch, prune_level):
    for n, m in Arch.named_modules():
        # find all the 'part_group_Arch'
        if isinstance(m, Part_Grouping_Arch) and str(prune_level) in n:
            prune_mask = Arch.prune_masks[prune_level]
            for i in range(len(prune_mask)):
                for j in range(len(prune_mask[0])):
                    if prune_mask[i][j] == 0:
                        m.parts_group[i]._ops[j] = EmptyOp(m.parts_group[i]._ops[j].channel, m.parts_group[i]._ops[j].ops_used)

    new_Arch = Derived_Part_Group_Network(Arch, prune_level)
    return new_Arch