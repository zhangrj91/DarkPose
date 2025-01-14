# # ------------------------------------------------------------------------------
# # Copyright (c) Microsoft
# # Licensed under the MIT License.
# # The code is based on HigherHRNet-Human-Pose-Estimation.
# # (https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation)
# # Modified by Zigang Geng (aa397601@mail.ustc.edu.cn).
# # ------------------------------------------------------------------------------
#
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
#
#
# import torch
#
#
# def get_one_stage_outputs(outputs):
#     if len(outputs) == 1:
#         return outputs[0]
#     else:
#         temp = outputs[0]
#         for j in range(1, len(outputs)):
#             temp = torch.max(temp, outputs[j])
#         return temp
#
#
# def get_locations(output_h, output_w, device):
#     shifts_x = torch.arange(
#         0, output_w, step=1,
#         dtype=torch.float32, device=device
#     )
#     shifts_y = torch.arange(
#         0, output_h, step=1,
#         dtype=torch.float32, device=device
#     )
#     shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
#     shift_x = shift_x.reshape(-1)
#     shift_y = shift_y.reshape(-1)
#     locations = torch.stack((shift_x, shift_y), dim=1)
#
#     return locations
#
#
# def get_reg_kpts(offset, num_joints):
#     _, h, w = offset.shape
#     offset = offset.permute(1, 2, 0).reshape(h*w, num_joints, 2)
#     locations = get_locations(h, w, offset.device)
#     locations = locations[:, None, :].expand(-1, num_joints, -1)
#     kpts = locations - offset
#
#     return kpts
#
#
# def get_multi_stage_outputs(
#         cfg, model, image, with_flip=False
# ):
#     num_joints = cfg.DATASET.NUM_JOINTS-1
#     dataset = cfg.DATASET.DATASET
#     heatmaps_avg = 0
#     num_heatmaps = 0
#     heatmaps = []
#     reg_kpts_list = []
#
#     # forward
#     all_outputs, all_offsets = model(image)
#     outputs = [get_one_stage_outputs(out)
#                for out in all_outputs]
#     offset = all_offsets[0][-1]
#     h, w = offset.shape[2:]
#     reg_kpts = get_reg_kpts(offset[0], num_joints)
#     reg_kpts = reg_kpts.contiguous().view(h*w, 2*num_joints).permute(1,
#                                                                      0).contiguous().view(1, -1, h, w)
#     reg_kpts_list.append(reg_kpts)
#
#     if with_flip:
#         if 'coco' in dataset:
#             flip_index_heat = FLIP_CONFIG['COCO_WITH_CENTER'] \
#                 if cfg.DATASET.WITH_CENTER else FLIP_CONFIG['COCO']
#             flip_index_offset = FLIP_CONFIG['COCO']
#         elif 'mpii' in dataset:
#             flip_index_heat = FLIP_CONFIG['CROWDPOSE_WITH_CENTER'] \
#                 if cfg.DATASET.WITH_CENTER else FLIP_CONFIG['CROWDPOSE']
#             flip_index_offset = FLIP_CONFIG['CROWDPOSE']
#         else:
#             raise ValueError(
#                 'Please implement flip_index for new dataset: %s.' % dataset)
#
#         new_image = torch.zeros_like(image)
#         new_image_2x = torch.zeros_like(image)
#
#         image = torch.flip(image, [3])
#         new_image[:, :, :, :-3] = image[:, :, :, 3:]
#         new_image_2x[:, :, :, :-1] = image[:, :, :, 1:]
#
#         all_outputs_flip, all_offsets_flip = model(new_image)
#         outputs_flip = [get_one_stage_outputs(all_outputs_flip[0])]
#         if len(cfg.DATASET.OUTPUT_SIZE) > 1:
#             all_outputs_flip, _ = model(new_image_2x)
#             outputs_flip.append(get_one_stage_outputs(all_outputs_flip[1]))
#
#         offset_flip = all_offsets_flip[0][-1]
#         reg_kpts_flip = get_reg_kpts(offset_flip[0], num_joints)
#         reg_kpts_flip = reg_kpts_flip[:, flip_index_offset, :]
#         reg_kpts_flip[:, :, 0] = w - reg_kpts_flip[:, :, 0] - 1
#         reg_kpts_flip = reg_kpts_flip.contiguous().view(
#             h*w, 2*num_joints).permute(1, 0).contiguous().view(1, -1, h, w)
#         reg_kpts_list.append(torch.flip(reg_kpts_flip, [3]))
#     else:
#         outputs_flip = None
#
#     for i, output in enumerate(outputs):
#         c = output.shape[1]
#         if cfg.LOSS.WITH_HEATMAPS_LOSS[i] and cfg.TEST.WITH_HEATMAPS[i]:
#             num_heatmaps += 1
#             if num_heatmaps > 1:
#                 heatmaps_avg[:, :c] += output
#             else:
#                 heatmaps_avg += output
#
#     if num_heatmaps > 0:
#         heatmaps_avg[:, :c] /= num_heatmaps
#         heatmaps.append(heatmaps_avg)
#
#     if with_flip:
#         heatmaps_avg = 0
#         num_heatmaps = 0
#         for i in range(len(outputs_flip)):
#             output = outputs_flip[i]
#
#             output = torch.flip(output, [3])
#             outputs.append(output)
#             c = output.shape[1]
#
#             if cfg.LOSS.WITH_HEATMAPS_LOSS[i] and cfg.TEST.WITH_HEATMAPS[i]:
#                 num_heatmaps += 1
#                 if 'coco' in dataset:
#                     flip_index_heat = FLIP_CONFIG['COCO_WITH_CENTER'] \
#                         if c == num_joints+1 else FLIP_CONFIG['COCO']
#                 elif 'crowd_pose' in dataset:
#                     flip_index_heat = FLIP_CONFIG['CROWDPOSE_WITH_CENTER'] \
#                         if c == num_joints+1 else FLIP_CONFIG['CROWDPOSE']
#                 else:
#                     raise ValueError(
#                         'Please implement flip_index for new dataset: %s.' % dataset)
#
#                 if num_heatmaps > 1:
#                     heatmaps_avg[:, :c] += output[:, flip_index_heat, :, :]
#                 else:
#                     heatmaps_avg += \
#                         output[:, flip_index_heat, :, :]
#         if num_heatmaps > 0:
#             heatmaps_avg[:, :c] /= num_heatmaps
#             heatmaps.append(heatmaps_avg)
#
#     return outputs, heatmaps, reg_kpts_list
#
#
# def aggregate_results(
#         cfg, final_heatmaps, final_kpts, heatmaps, kpts
# ):
#
#     heatmaps_avg = (heatmaps[0] + heatmaps[1])/2.0 if cfg.TEST.FLIP_TEST \
#         else heatmaps[0]
#
#     if final_heatmaps is None:
#         final_heatmaps = [heatmaps_avg]
#     else:
#         final_heatmaps.append(heatmaps_avg)
#
#     kpts_avg = (kpts[0] + kpts[1])/2.0 if cfg.TEST.FLIP_TEST \
#         else kpts[0]
#
#     kpts_avg = torch.nn.functional.interpolate(
#         kpts_avg*(heatmaps_avg.shape[2]/kpts_avg.shape[2]),
#         size=(heatmaps_avg.shape[2], heatmaps_avg.shape[3]),
#         mode='nearest'
#     )
#
#     if final_kpts is None:
#         final_kpts = [kpts_avg]
#     else:
#         final_kpts.append(kpts_avg)
#
#     return final_heatmaps, final_kpts


# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np

from src.hrutils.transforms import transform_preds


def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


def get_final_preds(config, batch_heatmaps, center, scale):
    coords, maxvals = get_max_preds(batch_heatmaps)

    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    # post-processing
    if config.test.POST_PROCESS:
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                hm = batch_heatmaps[n][p]
                px = int(math.floor(coords[n][p][0] + 0.5))
                py = int(math.floor(coords[n][p][1] + 0.5))
                if 1 < px < heatmap_width-1 and 1 < py < heatmap_height-1:
                    diff = np.array(
                        [
                            hm[py][px+1] - hm[py][px-1],
                            hm[py+1][px]-hm[py-1][px]
                        ]
                    )
                    coords[n][p] += np.sign(diff) * .25

    preds = coords.copy()

    # Transform back
    for i in range(coords.shape[0]):
        preds[i] = transform_preds(
            coords[i], center[i], scale[i], [heatmap_width, heatmap_height]
        )

    return preds, maxvals
