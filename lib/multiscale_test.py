import torch
import cv2
from utils.transforms import get_affine_transform,flip_back
from utils.vis import save_debug_images
from core.function import AverageMeter, _print_name_value
from core.evaluate import accuracy
from core.inference import get_final_preds
import numpy as np
import multiprocessing
import time
import logging
import os

logger = logging.getLogger(__name__)

def read_scaled_image(image_file, s, center, scale, image_size, COLOR_RGB, DATA_FORMAT, image_transform):
    if DATA_FORMAT == 'zip':
        from utils import zipreader
        data_numpy = zipreader.imread(image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    else:
        data_numpy = cv2.imread(image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if COLOR_RGB:
        data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)
    trans = get_affine_transform(center, s * scale, 0, image_size)
    images_warp = cv2.warpAffine(data_numpy, trans, tuple(image_size), flags=cv2.INTER_LINEAR)
    return image_transform(images_warp)


def scale_back_output(output_hm, s, output_size):
    hm_size = [output_hm.size(3), output_hm.size(2)]
    # original_max_val1, _ = torch.max(output_hm, dim=2, keepdim=True)
    # original_max_val2, _ = torch.max(original_max_val1, dim=3, keepdim=True)
    if s != 1.0:
        hm_w_margin = int(abs(1.0 - s) * hm_size[0] / 2.0)
        hm_h_margin = int(abs(1.0 - s) * hm_size[1] / 2.0)
        if s < 1.0:
            hm_padding = torch.nn.ZeroPad2d((hm_w_margin, hm_w_margin, hm_h_margin, hm_h_margin))
            resized_hm = hm_padding(output_hm)
        else:
            resized_hm = output_hm[:, :, hm_h_margin:hm_size[1] - hm_h_margin, hm_w_margin:hm_size[0] - hm_w_margin]
        resized_hm = torch.nn.functional.interpolate(
            resized_hm,
            size=(output_size[0], output_size[1]),
            mode='bilinear',  # bilinear bicubic area
            align_corners=False
        )
    else:
        resized_hm = output_hm
        if hm_size[0] != output_size[0] or hm_size[1] != output_size[1]:
            resized_hm = torch.nn.functional.interpolate(
                resized_hm,
                size=(output_size[0], output_size[1]),
                mode='bilinear',  # bilinear bicubic area
                align_corners=False
            )

    # max_val1, _ = torch.max(resized_hm, dim=2, keepdim=True)
    # max_val2, _ = torch.max(max_val1, dim=3, keepdim=True)
    # resized_hm = resized_hm/max_val2*original_max_val2

    # resized_hm = resized_hm / torch.amax(resized_hm, dim=[2, 3], keepdim=True)
    # resized_hm = torch.nn.functional.normalize(resized_hm, dim=[2, 3], p=1)
    # resized_hm = resized_hm/(torch.sum(resized_hm, dim=[2, 3], keepdim=True) + 1e-9)
    return resized_hm


def validate(config, val_loader, val_dataset, model, criterion, output_dir, tb_log_dir, writer_dict=None, test_scale=None, image_transform=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros((num_samples, config.MODEL.NUM_JOINTS, 3), dtype=np.float32)
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0

    # PRINT_FREQ = min(config.PRINT_FREQ//10, 5)
    PRINT_FREQ = config.PRINT_FREQ
    thread_pool = multiprocessing.Pool(multiprocessing.cpu_count())

    image_size = np.array([config.MODEL.IMAGE_SIZE[1], config.MODEL.IMAGE_SIZE[0]])
    final_test_scale = test_scale if test_scale is not None else config.TEST.SCALE_FACTOR
    with torch.no_grad():
        end = time.time()

        start_time = time.time()
        for i, (input, target, target_weight, meta) in enumerate(val_loader):
            # compute output
            # print("Batch", i, "Batch Size", input.size(0))

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            outputs = []
            hm_size = None
            for sidx, s in enumerate(sorted(final_test_scale, reverse=True)):
                print("Test Scale", s)
                if s != 1.0:
                    image_files = meta["image"]
                    centers = meta["center"].numpy()
                    scales = meta["scale"].numpy()

                    images_resized = thread_pool.starmap(read_scaled_image, [(image_file,
                                                                              s,
                                                                              center,
                                                                              scale,
                                                                              image_size,
                                                                              config.DATASET.COLOR_RGB,
                                                                              config.DATASET.DATA_FORMAT,
                                                                              image_transform) for (image_file, center, scale) in zip(image_files, centers, scales)])
                    images_resized = torch.stack(images_resized, dim=0)
                else:
                    images_resized = input

                model_outputs = model(images_resized)
                if isinstance(model_outputs, list):
                    model_outputs = model_outputs[-1]

                if config.TEST.FLIP_TEST:
                    print("Test Flip")
                    input_flipped = images_resized.flip(3)
                    output_flipped = model(input_flipped)
                    if isinstance(output_flipped, list):
                        output_flipped = output_flipped[-1]

                    output_flipped = flip_back(output_flipped.cpu().numpy(), val_dataset.flip_pairs)
                    output_flipped = torch.from_numpy(output_flipped.copy()).cuda()

                    # feature is not aligned, shift flipped heatmap for higher accuracy
                    output_flipped[:, :, :, 1:] = output_flipped.clone()[:, :, :, 0:-1]

                    model_outputs = 0.5 * (model_outputs + output_flipped)

                hm_size = [model_outputs.size(3), model_outputs.size(2)]
                # hm_size = image_size
                # hm_size = [128, 128]
                output_flipped_resized = scale_back_output(model_outputs, s, hm_size)
                outputs.append(output_flipped_resized)

            for indv_output in outputs:
                _, avg_acc, _, _ = accuracy(indv_output.cpu().numpy(), target.cpu().numpy())
                print("Indv Accuracy", avg_acc)

            output = torch.stack(outputs, dim=0).mean(dim=0)

            target = scale_back_output(target, 1.0, hm_size)
            loss = criterion(output, target, target_weight)

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(), target.cpu().numpy())
            print("Avg Accuracy", avg_acc)
            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            preds, maxvals = get_final_preds(config, output.clone().cpu().numpy(), c, s)

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])

            idx += num_images

            if i % PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(i, len(val_loader), batch_time=batch_time, loss=losses, acc=acc)
                logger.info(msg)

                prefix = '{}_{}'.format(os.path.join(output_dir, 'val'), i)
                save_debug_images(config, input, meta, target, pred*4, output, prefix)

        total_duration = time.time() - start_time
        logger.info("Total test time: {:.1f}".format(total_duration))
        name_values, perf_indicator = val_dataset.evaluate(config, all_preds, output_dir, all_boxes, image_path, filenames, imgnums)

        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar('valid_loss', losses.avg, global_steps)
            writer.add_scalar('valid_acc', acc.avg, global_steps)
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars('valid', dict(name_value), global_steps)
            else:
                writer.add_scalars('valid', dict(name_values), global_steps)
            writer_dict['valid_global_steps'] = global_steps + 1

    thread_pool.close()
    thread_pool.join()
    return perf_indicator