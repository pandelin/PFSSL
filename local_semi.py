import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
import copy
from easydict import EasyDict
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Normalize, ToPILImage
from data.fssl_liver_dataset import LiverDataset
from models import deeplabv3
from utils.loss_functions import DSCLoss, DiceLoss_Model
from utils.logger import logger as logging
from utils.utils import *
from utils.mask_generator import BoxMaskGenerator, AddMaskParamsToBatch, SegCollate
from utils.ramps import sigmoid_rampup
from utils.torch_utils import seed_torch
from utils.model_init import init_weight
from utils.utils import write_csv, log_metrics
from fedgraph.util import weight_flatten_all_params, weight_flatten_all
import torchmetrics
from tqdm import tqdm
import torchvision.transforms.functional as TF
import numpy as np
from models.unet2d import UNet2D
from torch.nn.functional import one_hot
from utils.metirc import MultDice
import time
import hashlib
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

all_best_test_iou = [0.] * 6
all_best_test_dice = [0.] * 6

def create_model(ema=False):
    # Network definition
    # model = UNet2D(3, 2).to(device)
    model = deeplabv3.__dict__["UNet"](in_channels=3, out_channels=2).to(device)
    if ema:
        for param in model.parameters():
            param.detach_()
    return model

def model_ema(model):
    for param in model.parameters():
        param.detach_()
    return model

def local_train(args, round, nets_this_round, local_train_models, cluster_models, local_dls):
    best_test_dice_list, best_test_iou_list = [], []
    for net_id, net in nets_this_round.items():  #
        label_local_dl, un_label_local_dl, test_local_dl = local_dls[net_id]

        cluster_model = None
        if round > 0 and cluster_models is not None:
            cluster_model = cluster_models[net_id].to(device)

        local_train_model = local_train_models[net_id]

        # start training
        best_test_iou, best_test_dice, teacher = local_train_net(args, round, net, local_train_model, cluster_model,
                                                                 net_id, label_local_dl, un_label_local_dl, test_local_dl)
        best_test_dice_list.append(best_test_dice)
        best_test_iou_list.append(best_test_iou)
        net.to('cpu')
        # nets_this_round[net_id] = teacher

    return best_test_dice_list, best_test_iou_list

def local_train_net(args, round, net, local_train_model, cluster_model, net_id, label_local_dl, un_label_local_dl, test_local_dl):
    # args = get_args()
    seed_torch(args.seed)
    # Project Saving Path
    project_path = os.path.join(args.project, str(net_id))
    ensure_dir(project_path)
    save_path = os.path.join(project_path, 'weights')
    os.makedirs(save_path, exist_ok=True)
    train_csv_path = os.path.join(project_path, 'train.csv')
    test_csv_path = os.path.join(project_path, 'test.csv')

    # Tensorboard & Statistics Results & Logger
    tb_dir = project_path + '/tensorboard{}'.format(time.strftime("%b%d_%d-%H-%M", time.localtime()))
    writer = SummaryWriter(tb_dir)
    metrics = EasyDict()
    metrics.train_loss = []
    metrics.train_loss2 = []
    metrics.val_loss = []
    logger = logging(os.path.join(project_path, 'train_val.log'))
    # logger.info('PyTorch Version {}\n Experiment{}'.format(torch.__version__, project_path))
    # init log
    runs_log_name = os.path.join(project_path, 'log')
    my_writer = SummaryWriter(os.path.join(runs_log_name, f'res_log'))

    # Load Data
    train_labeled_dataloader = label_local_dl
    train_unlabeled_dataloader = un_label_local_dl
    iters = len(label_local_dl)
    test_dataloader, length = test_local_dl, len(test_local_dl)
    test_iters = len(test_dataloader)

    # Load Model & EMA
    student1 = local_train_model.to(device)
    # student1 = copy.deepcopy(net).to(device)
    student2 = net.to(device)
    teacher = model_ema(copy.deepcopy(net)).to(device)
    # teacher.detach_model()
    # best_model_wts = copy.deepcopy(teacher.state_dict())
    best_epoch = 0
    best_loss = 100
    alpha = 0.1
    global all_best_test_iou
    global all_best_test_dice

    # Criterion & Optimizer & LR Schedule
    # optimizer = optim.AdamW(student.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))
    # criterion = DSCLoss(num_classes=args.num_classes, intra_weights=args.intra_weights, inter_weight=args.inter_weight,
    #                     device=device)
    dice_loss = DiceLoss_Model()
    criterion = DSCLoss(num_classes=args.num_classes, intra_weights=args.intra_weights, inter_weight=args.inter_weight,
                        device=device)
    criterion_u = DSCLoss(num_classes=args.num_classes, intra_weights=args.intra_weights,
                          inter_weight=args.inter_weight, device=device)
    criterion_c = DSCLoss(num_classes=args.num_classes, intra_weights=args.intra_weights,
                          inter_weight=args.inter_weight, device=device)
    optimizer1 = optim.AdamW(student1.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))
    optimizer2 = optim.AdamW(student2.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))

    # metric
    # train_metric_iou = torchmetrics.JaccardIndex(task="multiclass", num_classes=2).to(device)
    # test_metric_iou = torchmetrics.JaccardIndex(task="multiclass", num_classes=2).to(device)
    train_metric_iou = torchmetrics.JaccardIndex(task="multilabel", num_labels=2).to(device)
    train_metric_dice = MultDice().to(device)
    test_metric_iou = torchmetrics.JaccardIndex(task="multilabel", num_labels=2).to(device)
    test_metric_dice = MultDice().to(device)
    best_test_iou = 0.
    best_test_dice = 0.

    # Train
    since = time.time()
    logger.info('start training')
    for epoch in range(1, args.num_epochs + 1):
        epoch_metrics = EasyDict()
        epoch_metrics.train_loss = []
        epoch_metrics.train_loss2 = []
        epoch_metrics.test_loss = []
        epoch_metrics.un_sup_loss = []
        epoch_metrics.sup_loss = []
        pbar = range(iters)
        iter_train_labeled_dataloader = iter(train_labeled_dataloader)
        iter_train_unlabeled_dataloader = iter(train_unlabeled_dataloader)

        ############################
        # Train
        ############################
        train_dice_results = [[], []]
        student1.train()
        student2.train()
        teacher.train()
        train_bar = tqdm(train_labeled_dataloader)
        for idx, _ in enumerate(train_bar):
            image, label, imageA1, imageA2, imgname = next(iter_train_labeled_dataloader)
            image, label = image.to(device), label.to(device)
            imageA1, imageA2 = imageA1.to(device), imageA2.to(device)
            uimage, _, uimageA1, uimageA2, _ = next(iter_train_unlabeled_dataloader)
            uimage = uimage.to(device)
            uimageA1, uimageA2 = uimageA1.to(device), uimageA2.to(device)
            '''
            Step 1
            '''
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            ###########################
                # supervised path #
            ###########################
            pred1_data = student1(image)  # [1,3,128,128]
            pred2_data = student2(imageA1)
            pred_data = teacher(image)
            # pred_feature = torch.softmax(preds, dim=1) # [1,2,128,128]
            # loss_sup = criterion(pred, label.squeeze(1).long())

            loss1_sup = criterion(pred1_data, label.squeeze(1).long())
            loss2_sup = criterion(pred2_data, label.squeeze(1).long())
            loss_sup = 0.5 * (loss1_sup + loss2_sup)

            label = label.squeeze(dim=1).long()
            label_hot = one_hot(label, num_classes=2).permute(0, 3, 1, 2).contiguous()
            pred_feature = torch.softmax(pred_data, dim=1)
            pred_max = torch.argmax(pred_feature, dim=1)
            pred_train = one_hot(pred_max, num_classes=2).permute(0, 3, 1, 2).contiguous()

            # cal metric
            train_miou = train_metric_iou(pred_train, label_hot)
            train_dice = train_metric_dice(pred_train, label_hot)
            dices = dice_score_batch(prediction=pred_train, target=label_hot)
            for b in range(len(dices)):
                for i in range(2):
                    train_dice_results[i].append(dices[b][i].item())

            ###########################
                # unsupervised path #
            ###########################
            preds_u = teacher(uimage)
            pred_u_feature = torch.softmax(preds_u, dim=1)
            pseudo = torch.argmax(pred_u_feature, dim=1)

            # cal relationship
            label_ratio = cal_mask_ratio(label)
            pseudo_ratio = cal_mask_ratio(pseudo)
            label_patch_num = calculate_patch_num(label_ratio)
            pseudo_patch_num = calculate_patch_num(pseudo_ratio)
            topk = min(label_patch_num, pseudo_patch_num)
            topk = args.fusion_ratio * topk
            patch_num = 16  # batch_size=8
            h, w = args.image_size // patch_num, args.image_size // patch_num
            s = h
            unfolds = torch.nn.Unfold(kernel_size=(h, w), stride=s).to(device)
            folds = torch.nn.Fold(output_size=(args.image_size, args.image_size), kernel_size=(h, w), stride=s).to(
                device)
            # boundary detection
            edge_label = get_edge_by_sobel(label, 5)
            edge_pseudo = get_edge_by_sobel(pseudo, 5)

            # Estimate the uncertainty map
            with torch.no_grad():
                uncertainty_map1 = pred_u_feature
                uncertainty_map11 = -1.0 * torch.sum(uncertainty_map1 * torch.log(uncertainty_map1 + 1e-6), dim=1,
                                                    keepdim=True)
                uncertainty_map11 = uncertainty_map11 * edge_label.unsqueeze(1)
                uncertainty_map3 = pred_feature
                uncertainty_map33 = -1.0 * torch.sum(uncertainty_map3 * torch.log(uncertainty_map3 + 1e-6), dim=1,
                                                     keepdim=True)
                uncertainty_map33 = uncertainty_map33 * edge_pseudo.unsqueeze(1)
                B, C = image.shape[0], image.shape[1]
                # for student 1
                x1 = unfolds(uncertainty_map11)  # B x C*kernel_size[0]*kernel_size[1] x L
                x1 = x1.view(B, 1, h, w, -1)  # B x C x h x w x L
                x1_mean = torch.mean(x1, dim=(1, 2, 3))  # B x L
                _, x1_max_index = torch.sort(x1_mean, dim=1, descending=True)  # B x L B x L
                x1_min_index = sort_mean(x1_mean)
                # for label
                x3 = unfolds(uncertainty_map33)  # B x C*kernel_size[0]*kernel_size[1] x L
                x3 = x3.view(B, 1, h, w, -1)  # B x C x h x w x L
                x3_mean = torch.mean(x3, dim=(1, 2, 3))  # B x L
                _, x3_max_index = torch.sort(x3_mean, dim=1, descending=True)  # B x L B x L

                # replace
                img_unfold = unfolds(image).view(B, C, h, w, -1)  # B x C x h x w x L
                imgu_unfold = unfolds(uimage).view(B, C, h, w, -1)  # B x C x h x w x L
                img_pseudo_unfold = unfolds(label.unsqueeze(1).float()).view(B, 1, h, w, -1)  # B x C x h x w x
                imgu_pseudo_unfold = unfolds(pseudo.unsqueeze(1).float()).view(B, 1, h, w, -1)  # B x C x h x w x
                for i in range(B):
                    imgu_unfold[i, :, :, :, x1_min_index[i, :topk]] = img_unfold[i, :, :, :, x3_max_index[i, :topk]]
                    imgu_pseudo_unfold[i, :, :, :, x1_min_index[i, :topk]] = img_pseudo_unfold[i, :, :, :,
                                                                             x3_max_index[i, :topk]]

            image2 = folds(imgu_unfold.view(B, C * h * w, -1))
            label2 = folds(imgu_pseudo_unfold.view(B, 1 * h * w, -1)).squeeze(1).long()

            pred1_u = student1(image2)
            pred2_u = student2(image2)

            pseudo1 = torch.softmax(pred1_u, dim=1)
            pseudo1 = torch.argmax(pseudo1, dim=1)
            pseudo2 = torch.softmax(pred2_u, dim=1)
            pseudo2 = torch.argmax(pseudo2, dim=1)

            # MPS loss
            loss1_mps = criterion_c(pred1_u, label2.detach())
            loss2_mps = criterion_c(pred2_u, label2.detach())
            loss_mps = (loss1_mps + loss2_mps) * 0.5
            # CPS loss
            loss1_u = criterion_u(pred1_u, pseudo2.detach())
            loss2_u = criterion_u(pred2_u, pseudo1.detach())
            loss_cps = (loss1_u + loss2_u) * 0.5

            loss_u = (loss_cps + loss_mps) * alpha
            lambda_ = sigmoid_rampup(current=idx + len(pbar) * (epoch - 1), rampup_length=len(pbar) * 5)
            loss = loss_sup + lambda_ * loss_u

            # cal cos similarity
            if round > 0 and cluster_model is not None:
                flatten_model = weight_flatten_all(student2.state_dict())
                loss2 = args.lam * torch.dot(cluster_model, flatten_model) / torch.linalg.norm(flatten_model)
                loss2.requires_grad_(True)
                loss2.backward(retain_graph=True)

            loss.requires_grad_(True)
            loss.backward()
            optimizer1.step()
            optimizer2.step()
            teacher.weighted_update(student1, student2, ema_decay=0.99, cur_step=idx + len(pbar) * (epoch - 1))

            writer.add_scalar('train_sup_loss', loss_sup.item(), idx + len(pbar) * (epoch-1))
            # writer.add_scalar('train_cps_loss', loss_cps.item(), idx + len(pbar) * (epoch-1))
            writer.add_scalar('train_loss', loss.item(), idx + len(pbar) * (epoch-1))
            epoch_metrics.train_loss.append(loss.item())
            epoch_metrics.un_sup_loss.append(loss_u.item())
            epoch_metrics.sup_loss.append(loss_sup.item())
            if round > 0 and cluster_model is not None:
                epoch_metrics.train_loss2.append(loss2.item())
            train_bar.set_description(f'loss_sup:{loss.item()}')

        metrics.train_loss.append(np.mean(epoch_metrics.train_loss))
        logger.info(f"Average: Epoch/Epoches {epoch}/{args.num_epochs} train_loss {np.mean(epoch_metrics.train_loss)}, "+
                    f"un_sup_loss {np.mean(epoch_metrics.un_sup_loss)}, sup_loss {np.mean(epoch_metrics.sup_loss)}, "+
                    f"train_loss2 {np.mean(epoch_metrics.train_loss2):.5f},")
        if np.mean(epoch_metrics.train_loss) <= best_loss:
            best_loss = np.mean(epoch_metrics.train_loss)
            torch.save(teacher.state_dict(), os.path.join(save_path, 'best.pth'))
        torch.save(teacher.state_dict(), os.path.join(save_path, 'last.pth'))

        # save result
        train_miou_avg = train_metric_iou.compute()
        train_dice_avg = train_metric_dice.compute()
        # get avg Dice
        mean_dice_class_1 = np.mean(train_dice_results[0])
        mean_dice_class_2 = np.mean(train_dice_results[1])
        train_metrics_list = [train_dice_avg.item(), train_miou_avg.item(), np.mean(epoch_metrics.train_loss),
                              np.mean(epoch_metrics.un_sup_loss), np.mean(epoch_metrics.sup_loss),
                              np.mean(epoch_metrics.train_loss2), mean_dice_class_1, mean_dice_class_2]
        write_csv(train_csv_path, epoch, train_metrics_list)
        log_metrics(my_writer, train_metrics_list, epoch, "train")
        print(f"round{round} | client{net_id} | epoch-{epoch}, train_metric:{train_metrics_list}")
        train_metric_iou.reset()
        train_metric_dice.reset()

        # test stage
        teacher.eval()
        test_dice_results =  [[], []]
        test_bar = tqdm(test_dataloader)
        with torch.no_grad():
            for idx, (image, label, _) in enumerate(test_bar):
                image, label = image.to(device), label.to(device)
                pred_test_data = teacher(image)  # [1,2,128,128]

                label = label.squeeze(dim=1).long()
                label_hot = one_hot(label, num_classes=2).permute(0, 3, 1, 2).contiguous()
                pred_feature = torch.softmax(pred_test_data, dim=1)
                pred_max = torch.argmax(pred_feature, dim=1)
                pred_test = one_hot(pred_max, num_classes=2).permute(0, 3, 1, 2).contiguous()

                loss_test = criterion(pred_test_data, label.squeeze(1).long())  #
                epoch_metrics.test_loss.append(loss_test.item())

                # cal metric
                test_miou = test_metric_iou(pred_test, label_hot)
                test_dice = test_metric_dice(pred_test, label_hot)
                test_dices = dice_score_batch(prediction=pred_test, target=label_hot)
                for b in range(len(test_dices)):
                    for i in range(2):
                        test_dice_results[i].append(test_dices[b][i].item())

            # save result
            test_miou_avg = test_metric_iou.compute()
            test_dice_avg = test_metric_dice.compute()
            # get avg Dice
            mean_dice_class_1 = np.mean(test_dice_results[0])
            mean_dice_class_2 = np.mean(test_dice_results[1])
            test_metrics_list = [test_dice_avg.item(), test_miou_avg.item(), np.mean(epoch_metrics.test_loss),
                                 mean_dice_class_1, mean_dice_class_2]
            write_csv(test_csv_path, epoch, test_metrics_list)
            log_metrics(my_writer, test_metrics_list, epoch, "test")
            print(f"round{round} | client{net_id} | epoch-{epoch}, test_metric:{test_metrics_list}")
            if test_miou_avg.item() > best_test_iou:
                best_test_iou = test_miou_avg.item()
            if test_dice_avg.item() > best_test_dice:
                best_test_dice = test_dice_avg.item()
            if test_miou_avg.item() > all_best_test_iou[net_id]:
                all_best_test_iou[net_id] = test_miou_avg.item()
                torch.save(teacher.state_dict(), os.path.join(save_path, 'best_test_iou.pth'))
            if test_dice_avg.item() > all_best_test_dice[net_id]:
                all_best_test_dice[net_id] = test_dice_avg.item()
                torch.save(teacher.state_dict(), os.path.join(save_path, 'best_test_dice.pth'))

            test_metric_iou.reset()
            test_metric_dice.reset()

    # Train finish

    time_elapsed = time.time() - since
    logger.info('Training completed in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    logger.info(project_path)
    logger.info('TRAINING FINISHED!')
    return best_test_iou, best_test_dice, teacher


if __name__ == '__main__':
    local_train_net()
