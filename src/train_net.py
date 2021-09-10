import torch
from myutils import AverageMeter,save_batch_image_with_joints
from timeit import default_timer as timer
import logging
import os
import torch.nn as nn
from torch.nn import init
from .architecture.part_prune import Prune
    # define the initial function to init the layer's parameters for the network

logger = logging.getLogger(__name__)


def weigth_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight.data)
        init.constant_(m.bias.data, 0.1)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()

def train(epoch, train_queue, arch_queue ,model,search_arch,criterion,optimizer,lr,search_strategy,
            output_dir, logger, config, args):
    # when the search_strategy is `None` or `sync `, the arch_queue is None

    loss = AverageMeter()
    weigth_init(model)
    model.train()
    if arch_queue is not None:
        valid_iter = iter(arch_queue)
    
    # only update W for several epoches before update the alpha
    NAS_begin = config.train.arch_search_epoch

    current_search_strategy = search_strategy if epoch >= NAS_begin else 'None'

    logger.info("Current Architecture Search strategy is {} . *{} Search begin in epoch: {}"
                    .format(current_search_strategy,   config.train.arch_search_strategy,NAS_begin))

    for iters, (x, train_heatmap_gt, train_kpt_visible,  train_info) in enumerate(train_queue):

        start = timer()

        x = x.cuda(non_blocking=True)
        train_heatmap_gt = train_heatmap_gt.cuda(non_blocking=True)
        train_kpt_visible = train_kpt_visible.float().cuda(non_blocking=True)

        if  search_strategy=='first_order_gradient' or search_strategy=='second_order_gradient':

            x_valid , valid_heatmap_gt, valid_kpt_visible, valid_info = next(valid_iter)
            x_valid = x_valid.cuda(non_blocking=True)
            valid_heatmap_gt = valid_heatmap_gt.cuda(non_blocking=True)
            valid_kpt_visible = valid_kpt_visible.cuda(non_blocking=True)

            search_arch.step(x, train_heatmap_gt, train_kpt_visible, x_valid, valid_heatmap_gt, valid_kpt_visible, lr, optimizer,
                             search_strategy= current_search_strategy,
                             # Note : NOT `search_strategy`, because `current_search_strategy` can be none in early epoches
                             weight_optimization_flag = config.train.arch_search_weight_optimization_flag)
        
        kpts = model(x)
        # for name, param in model.named_parameters():
        #     print("--------------")
        #     if "branches" in name and "group" in name:
        #         print(name)
        #         print(param.grad)
        all_parts = model.module.all_parts if isinstance(model, torch.nn.DataParallel) else model.all_parts
        backward_loss = criterion(kpts,train_heatmap_gt.to(kpts.device), train_kpt_visible.to(kpts.device), all_parts)
        # backward_loss = criterion(kpts, train_heatmap_gt.to(kpts[0].device), train_kpt_visible.to(kpts[0].device),
        #                           model.origin_parts)

        #backward_loss = model.loss(x, train_heatmap_gt, train_kpt_visible,info=train_info)
        optimizer.zero_grad()
        backward_loss.backward()
        loss.update(backward_loss.item(), x.size(0))

        optimizer.step()
        time = timer() - start

        if iters % 100 == 0:
            # torch.cuda.empty_cache()
            if args.debug:

                save_batch_image_with_joints(   x,
                                                train_info['keypoints'],
                                                train_kpt_visible.unsqueeze(-1).cpu(),
                                                os.path.join(output_dir,'debug_image_'+str(iters)))
                
            logger.info('epoch: {}   \titers:[{}|{}]   \tloss:{:.6f}({:.5f})  \tfeed-speed:{:.2f} samples/s' #  \tembedloss:{:.8f}'
                        .format(epoch,iters,len(train_queue),loss.val, loss.avg ,len(x)/time))#,embedding_loss))

        # break

    if isinstance(model, torch.nn.DataParallel):
        model.module._print_info()
    else:
        model._print_info()


    # if args.show_arch_value or epoch % 10==0: # alpha and beta will be constant when nas is `none`
    #     logger.info("=========>current architecture's values before evaluate")
    #     if hasattr(model,"backbone"):
    #         if hasattr(model.backbone,"alphas"):
    #             model.backbone._show_alpha(original_value=True)
    #         for g in model.groupings:
    #             g._show_alpha(original_value=False)
        
