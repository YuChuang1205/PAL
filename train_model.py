#!/usr/bin/python3
# coding = gbk
"""
@Author : yuchuang
@Time :
@desc:
"""
from tqdm import tqdm
import torch.optim as optim
from loss.Edge_loss import edgeSCE_loss
# from model.MSDA.MSDA_no_sigmoid import MSDANet_No_Sigmoid
from components.metric_new_crop import *
from datetime import datetime
import torch
import os
import sys
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from components.utils_all_edge_copy_paste_final_2_img_path import get_loaders, make_dir
from components.cal_mean_std import Calculate_mean_std
import cv2
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
from utilts import *
from torch.autograd import Variable
import math


##############################################
choose_model = 'MSDA'   ##choose model in [ACM, ALC, MLCL, ALCL, DNA, GGL, UIU, MSDA]
model_func = access_model(choose_model)
choose_dataset = 'SIRST3'  ## choose dataset in [SIRST3, IRSTD_1K_point, NUDT_SIRST_1_1_point, SIRST_1_1_point_new]
choose_dataset_type = 'masks_coarse' ## choose dataset_type in [mask, masks_coarse, masks_centroid]   "mask": "Full supervision"; "masks_centroid": "Centroid point supervision"; "mask_coarse": "Coarse point supervision"
################################################

# Hyperparameters etc.
LEARNING_RATE = 1e-3  #default: 1e-3;    When the choose_model = 'GGL' or 'ALCL', "LEARNING_RATE = 5e-4" can be considered to obtain relatively more stable training.  
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_BATCH_SIZE = 16   #default: 16
TEST_BATCH_SIZE = 1
TEST_PATCH_BATCH_SIZE = 16
NUM_EPOCHS = 400     #default: 400
num_start_test_epochs = 100   #default: 100
NUM_WORKERS = 4
# IMAGE_SIZE = 256
PATCH_SIZE = 256
PATCH_SIZE_test = 1024
CP_PROBABILITY = 0.5
# lose_point_ratio = 0.2
lose_point_ratio_init = 0.2
alarm_point_ration = 5
clear_epoch_gap = 5
clear_inital_ratio = 0.2   #  default:0.2.   [0, clear_inital_ratio]: model pre-start phase
final_epoch_ratio = 0.8    # default:0.8.     [clear_inital_ratio, final_epoch_ratio] : model enhancement phase; [final_epoch_ratio,1]: model refinement phase
PIN_MEMORY = True
LOAD_MODEL = False
fixed_length = 120
thresh_Tb = 0.5
thresh_k = 0.5
degen = 0.97


## 方法1：直接使用当前文件名作为名字生成对应的文件夹作为工作空间
# # ----------------获取当前运行文件的文件名------------------#
# # 获取当前正在运行的文件的绝对路径
# file_path = os.path.abspath(sys.argv[0])
# # 获取文件名
# file_name = os.path.basename(file_path)
# # 去掉扩展名
# file_name_without_ext = os.path.splitext(file_name)[0]
# # # ------------------------------------------------------#
# MODEL_NAME = file_name_without_ext


# ## 方法2：根据给定的choose_model, choose_dataset, choose_dataset_type生成对应文件夹作为工作空间
# MODEL_NAME = choose_model + '__' + choose_dataset + '__' +choose_dataset_type


## 方法3：根据给定的choose_model, choose_dataset, choose_dataset_type生成对应文件夹作为工作空间,且加一个时间码确保文件夹的独一无二性,不存在覆盖问题 (推荐！！！)
MODEL_NAME = choose_model + '__' + choose_dataset + '__' + choose_dataset_type + '__' + str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))



root_path = os.path.abspath('.')
input_path = os.path.join(root_path,'dataset',choose_dataset)
demo_train_input_path = os.path.join(input_path,MODEL_NAME)

TRAIN_IMG_DIR = demo_train_input_path + "/train/choose/img"
TRAIN_MASK_DIR = demo_train_input_path + "/train/choose/mask"
train_points_dir = demo_train_input_path + "/train/choose/points"
VAL_IMG_DIR = input_path + "/val/img"
VAL_MASK_DIR = input_path + "/val/mask"

nc_img_dir =  demo_train_input_path + "/train/no_choose/img"
nc_mask_dir =  demo_train_input_path + "/train/no_choose/mask_pred"
nc_points_dir = demo_train_input_path + "/train/no_choose/points"

num_images = len(os.listdir(VAL_MASK_DIR))

origin_dir = input_path + "/origin"
origin_img_dir = input_path + "/origin/img"
origin_points_dir = os.path.join(input_path,'origin',choose_dataset_type)




def PadImg(image, times=32):
    h, w, c = image.shape
    pad_height = math.ceil(h / times) * times - h
    pad_width = math.ceil(w / times) * times - w
    image = np.pad(image, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant')
    return image

def test_pred(img, net, batch_size = TEST_PATCH_BATCH_SIZE, choose_model = choose_model):
    b, c, h, w = img.shape
    #print(img.shape)
    patch_size = PATCH_SIZE_test
    stride = PATCH_SIZE_test

    if h > patch_size and w > patch_size:
        # Unfold the image into patches
        img_unfold = F.unfold(img, kernel_size=patch_size, stride=stride)
        img_unfold = img_unfold.reshape(b, c, patch_size, patch_size, -1).permute(0, 4, 1, 2, 3)
        # print(img_unfold.shape)
        patch_num = img_unfold.size(1)

        preds_list = []
        for i in range(0, patch_num, batch_size):
            end = min(i + batch_size, patch_num)
            batch_patches = img_unfold[:, i:end, :, :, :].reshape(-1, c, patch_size, patch_size)
            batch_patches = Variable(batch_patches.float())
            batch_preds = net.forward(batch_patches)
            if choose_model == 'DNA':
                preds_list.append(batch_preds[-1])
            elif choose_model == 'UIU':
                preds_list.append(batch_preds[0])
            else:
                preds_list.append(batch_preds)

        # Concatenate all the patch predictions
        preds_unfold = torch.cat(preds_list, dim=0).permute(1, 2, 3, 0)
        preds_unfold = preds_unfold.reshape(b, -1, patch_num)
        preds = F.fold(preds_unfold, kernel_size=patch_size, stride=stride, output_size=(h, w))
    else:
        preds = net.forward(img)
        if choose_model == 'DNA':
            preds = preds[-1]
        elif choose_model == 'UIU':
            preds = preds[0]
        else:
            pass
    return preds


class SirstDataset_test(Dataset):
    def __init__(self, image_dir, img_name_list, transform=None, mode='None'):
        self.image_dir = image_dir
        self.transform = transform
        self.image_list = img_name_list
        self.mode = mode

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.image_list[index])
        image = np.array(Image.open(img_path).convert("RGB"))
        h, w, _ = image.shape
        # image = cv2.resize(image, (self.image_height, self.image_width))

        if (self.mode == 'test'):
            times = 32
            h, w, c = image.shape
            pad_height = math.ceil(h / times) * times - h
            pad_width = math.ceil(w / times) * times - w
            image = np.pad(image, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant')
            if self.transform is not None:
                augmentations = self.transform(image=image)
                image = augmentations["image"]
            return image, self.image_list[index], h, w

        else:
            print("mode输入的格式不对")
            sys.exit(0)



def main():
    cal_mean, cal_std = Calculate_mean_std(origin_img_dir)
    #print(cal_mean, cal_std)

    def train_fn(loader, model, optimizer, loss_fn, scaler, epoch, choose_model=choose_model):
        model.train()
        loop = tqdm(loader, ncols=fixed_length)
        iou_metric.reset()
        nIoU_metric.reset()
        train_losses = []
        for batch_idx, (data, targets, edge) in enumerate(loop):
            data = data.to(device=DEVICE).clone().detach()
            targets = targets.unsqueeze(1).to(device=DEVICE).clone().detach()
            edge = edge.to(device=DEVICE).clone().detach()

            if choose_model == 'DNA':
                with torch.cuda.amp.autocast():
                    predictions_no_sigmoid = model(data)
                    loss_0 = loss_fn(predictions_no_sigmoid[0], targets, edge)
                    loss_1 = loss_fn(predictions_no_sigmoid[1], targets, edge)
                    loss_2 = loss_fn(predictions_no_sigmoid[2], targets, edge)
                    loss_3 = loss_fn(predictions_no_sigmoid[3], targets, edge)
                    loss = torch.mean(loss_0 + loss_1 + loss_2 + loss_3)
                predictions = torch.sigmoid(predictions_no_sigmoid[-1])

            elif choose_model == 'UIU':
                with torch.cuda.amp.autocast():
                    predictions_no_sigmoid = model(data)
                    loss_0 = loss_fn(predictions_no_sigmoid[0], targets, edge)
                    loss_1 = loss_fn(predictions_no_sigmoid[1], targets, edge)
                    loss_2 = loss_fn(predictions_no_sigmoid[2], targets, edge)
                    loss_3 = loss_fn(predictions_no_sigmoid[3], targets, edge)
                    loss_4 = loss_fn(predictions_no_sigmoid[4], targets, edge)
                    loss_5 = loss_fn(predictions_no_sigmoid[5], targets, edge)
                    loss_6 = loss_fn(predictions_no_sigmoid[6], targets, edge)
                    loss = torch.mean(loss_0 + loss_1 + loss_2 + loss_3 + loss_4 + loss_5 + loss_6)
                predictions = torch.sigmoid(predictions_no_sigmoid[0])

            else:
                with torch.cuda.amp.autocast():
                    predictions_no_sigmoid = model(data)
                    loss = loss_fn(predictions_no_sigmoid, targets, edge)
                predictions = torch.sigmoid(predictions_no_sigmoid)

            iou_metric.update(predictions, targets)
            nIoU_metric.update(predictions, targets)
            _, IoU = iou_metric.get()
            _, nIoU = nIoU_metric.get()

            # backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            train_losses.append(loss.item())
            scaler.step(optimizer)
            scaler.update()

            # update tqdm loop
            loop.set_description(f"train epoch is：{epoch + 1} ")
            loop.set_postfix(loss=loss.item(), IoU=IoU.item(), nIoU=nIoU.item())

        return IoU, nIoU, np.mean(train_losses)

    def val_fn(loader, model, loss_fn, epoch):
        model.eval()
        loop = tqdm(loader, ncols=fixed_length)
        iou_metric.reset()
        nIoU_metric.reset()
        FA_PD_metric.reset()
        eval_losses = []
        with torch.no_grad():
            for batch_idx, (x, y, h, w) in enumerate(loop):
                x = x.to(device=DEVICE).clone().detach()
                y = y.unsqueeze(1).to(device=DEVICE).clone().detach()
                # print(y.shape)
                preds_no_sigmoid = test_pred(x, model)
                # preds_no_sigmoid = preds_no_sigmoid.cpu().data.numpy()
                preds_no_sigmoid = preds_no_sigmoid[0, :, :h, :w]
                y = y[0, :, :h, :w]
                eval_losses = 0
                preds = torch.sigmoid(preds_no_sigmoid)
                iou_metric.update(preds, y)
                nIoU_metric.update(preds, y)
                FA_PD_metric.update(preds, y)
                _, IoU = iou_metric.get()
                _, nIoU = nIoU_metric.get()
                FA, PD = FA_PD_metric.get(num_images)
                loop.set_description(f"test epoch is：{epoch + 1} ")
                loop.set_postfix(IoU=IoU.item(), nIoU=nIoU.item())
        return IoU, nIoU, np.mean(eval_losses), FA, PD

    train_transform = A.Compose(
        [
            # A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.SomeOf([
                A.VerticalFlip(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.Transpose(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightness(limit=0.3, p=0.2),
                A.RandomContrast(limit=0.3, p=0.2),
                A.Rotate(limit=45, p=0.3),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0, rotate_limit=0, p=0.5),
                A.ShiftScaleRotate(shift_limit=0, scale_limit=0.2, rotate_limit=0, p=0.5),
                A.GaussNoise(var_limit=(10.0, 50.0), mean=0, always_apply=False, p=0.2),
                A.NoOp(),
                A.NoOp(),
            ], 3, p=0.5),
            A.Normalize(
                mean=cal_mean,
                std=cal_std,
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    val_transforms = A.Compose(
        [
            # A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=cal_mean,
                std=cal_std,
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    test_transforms = A.Compose(
        [
            #A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
            A.Normalize(
                mean=cal_mean,
                std=cal_std,
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    if choose_model == 'DNA' or choose_model == 'UIU':
        model = model_func(mode='train').to(DEVICE)
    else:
        model = model_func().to(DEVICE)

    loss_fn = edgeSCE_loss
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    iou_metric = SigmoidMetric()
    nIoU_metric = SamplewiseSigmoidMetric(1, score_thresh=0.5)
    FA_PD_metric = PD_FA_2(1)


    scaler = torch.cuda.amp.GradScaler()

    best_mIoU = 0
    best_nIoU = 0
    best_PD = 0
    bestmIoUandPD = 0
    bestmIoUandPD_mIoU = 0
    bestmIoUandPD_nIoU = 0
    bestmIoUandPD_FA = 0
    bestmIoUandPD_PD = 0
    bestmIoUandPD_epoch = 0
    best_mIoU_nIoU = 0
    best_mIoU_FA = 0
    best_mIoU_PD = 0
    best_mIoU_epoch = 0
    best_nIoU_mIoU = 0
    best_nIoU_FA = 0
    best_nIoU_PD = 0
    best_nIoU_epoch = 0
    bestPD_mIou = 0
    bestPD_nIou = 0
    bestPD_FA = 0
    bestPD_epoch = 0
    num_epoch = []
    num_train_loss = []
    num_test_loss = []
    num_mioU = []
    num_nioU = []


    save_model_file_path = os.path.join(root_path, 'work_dirs', MODEL_NAME)
    make_dir(save_model_file_path)
    save_file_name = os.path.join(save_model_file_path, MODEL_NAME + '.txt')
    save_bestmiouandPD_file_name = os.path.join(save_model_file_path,
                                                'bestmIoUandPD_checkpoint_' + MODEL_NAME + ".pth.tar")
    save_best_miou_file_name = os.path.join(save_model_file_path,
                                            'best_mIoU_checkpoint_' + MODEL_NAME + ".pth.tar")
    save_best_PD_file_name = os.path.join(save_model_file_path,
                                          'best_PD_checkpoint_' + MODEL_NAME + ".pth.tar")

    save_file = open(save_file_name, 'a')
    start_epoch = 0
    RESUME = False
    if RESUME:
        # torch.cuda.empty_cache()
        path_checkpoint = "./work_dirs/"
        checkpoint = torch.load(path_checkpoint)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_mIoU = checkpoint['best_mIoU']
        best_nIoU = checkpoint['best_nIoU']
        best_PD = checkpoint['best_PD']
        bestmIoUandPD = checkpoint['bestmIoUandPD']
    # save_file = open('./work_dirs/result_' + MODEL_NAME + '.txt', 'a')
    save_file.write(
        '\n---------------------------------------start--------------------------------------------------\n')
    save_file.write(datetime.now().strftime("%Y-%m-%d, %H:%M:%S\n"))


##################################################################################
    ######生成数据
    if choose_dataset_type == 'masks_coarse' or choose_dataset_type == 'masks_centroid':
        data_inital_make_add_points(origin_img_dir, origin_points_dir, TRAIN_IMG_DIR, TRAIN_MASK_DIR, train_points_dir,
                                    nc_img_dir,
                                    nc_mask_dir, nc_points_dir, crop_size=10)
    else:
        pass
####################################################################################

    for epoch in range(start_epoch, NUM_EPOCHS):
        print(epoch)
        if choose_dataset_type == 'mask':
            train_loader, val_loader = get_loaders(
                origin_img_dir,
                origin_points_dir,
                VAL_IMG_DIR,
                VAL_MASK_DIR,
                PATCH_SIZE,
                TRAIN_BATCH_SIZE,
                TEST_BATCH_SIZE,
                train_transform,
                val_transforms,
                NUM_WORKERS,
                PIN_MEMORY,
            )

        elif choose_dataset_type == 'masks_coarse' or choose_dataset_type == 'masks_centroid':
            train_ds = SirstDataset_test(
                image_dir=TRAIN_IMG_DIR,
                img_name_list=os.listdir(TRAIN_IMG_DIR),
                transform=test_transforms,
                mode='test'
            )

            train_loader_label_update = DataLoader(
                train_ds,
                batch_size=TEST_BATCH_SIZE,
                num_workers=NUM_WORKERS,
                pin_memory=PIN_MEMORY,
                shuffle=False,
            )

            test_ds = SirstDataset_test(
                image_dir=nc_img_dir,
                img_name_list=os.listdir(nc_img_dir),
                transform=test_transforms,
                mode='test'
            )

            test_loader = DataLoader(
                test_ds,
                batch_size=TEST_BATCH_SIZE,
                num_workers=NUM_WORKERS,
                pin_memory=PIN_MEMORY,
                shuffle=False,
            )

            if (epoch > int(NUM_EPOCHS * clear_inital_ratio) and epoch <= int(NUM_EPOCHS * final_epoch_ratio)):
                model.eval()
                if epoch % clear_epoch_gap != 0:
                    pass
                else:
                    lose_point_ratio = lose_point_ratio_init + (epoch - (NUM_EPOCHS * clear_inital_ratio) + 1) / (NUM_EPOCHS * (final_epoch_ratio - clear_inital_ratio)) * (1 - lose_point_ratio_init)
                    print("开始标签的自更新--------")
                    for idx, (img, name, h, w) in tqdm(enumerate(train_loader_label_update)):
                        # print(idx)
                        img = img.to(device=DEVICE)
                        with torch.no_grad():
                            output = test_pred(img, model)
                            output = output[:, :, :h, :w]

                            output = torch.sigmoid(output)
                            output = output.cpu().data.numpy()

                        for i in range(output.shape[0]):
                            pred = output[i]
                            pred = pred[0]
                            pred = np.array(pred, dtype='float32')

                            mask_label_in_path = os.path.join(TRAIN_MASK_DIR, name[i])
                            prev_label = cv2.imread(mask_label_in_path, cv2.IMREAD_GRAYSCALE) / 255
                            current_binary_pred = update_gt_update_degen_corr(pred, prev_label, thresh_Tb, thresh_k, [h, w], degen=degen)
                            current_binary_pred = current_binary_pred * 255
                            cv2.imwrite(os.path.join(TRAIN_MASK_DIR, name[i]), current_binary_pred)

                    print("开始认识并学习困难样本-----------")
                    for idx, (img, name, h, w) in tqdm(enumerate(test_loader)):
                        # print(idx)
                        img = img.to(device=DEVICE)
                        with torch.no_grad():
                            output = test_pred(img, model)
                            output = output[:, :, :h, :w]

                            output = torch.sigmoid(output)
                            output = output.cpu().data.numpy()

                        for i in range(output.shape[0]):
                            pred = output[i]
                            pred = pred[0]
                            pred = np.array(pred, dtype='float32')
                            pred = cv2.resize(pred, (int(w[i]), int(h[i])))
                            pred = np.where(pred > 0.5, 255, 0)
                            # print(os.path.join(nc_mask_dir, name[i]))
                            cv2.imwrite(os.path.join(nc_mask_dir, name[i]), pred)


                    new_choose_list = deal_pred_mask_and_true_point_in(nc_img_dir, nc_mask_dir,
                                                                       nc_points_dir, TRAIN_IMG_DIR,
                                                                       TRAIN_MASK_DIR, train_points_dir,
                                                                       lose_point_ratio=lose_point_ratio,
                                                                       alarm_point_ration=alarm_point_ration)
                    print(new_choose_list)
                    print(len(new_choose_list))
                    save_file.write(f"当前epoch:{epoch + 1}  进入张数:{len(new_choose_list)} \n")
                    deal_gen_mask_error_aera(nc_mask_dir, nc_points_dir, new_choose_list)
                    if epoch > int(NUM_EPOCHS * clear_inital_ratio) and epoch <= int(NUM_EPOCHS * final_epoch_ratio):
                        hard_sample_in(nc_img_dir, nc_mask_dir, nc_points_dir, TRAIN_IMG_DIR, TRAIN_MASK_DIR,
                                       train_points_dir, new_choose_list)
                        print("此轮数据转移完成！！！！！")
                    else:
                        print("epochs进入的通道有误，请检查！！！！！")
                        sys.exit(0)



            elif epoch > int(NUM_EPOCHS * final_epoch_ratio):
                model.eval()
                if epoch % clear_epoch_gap != 0:
                    pass
                else:
                    print("开始标签的自更新--------")
                    for idx, (img, name, h, w) in tqdm(enumerate(train_loader_label_update)):
                        # print(idx)
                        img = img.to(device=DEVICE)
                        with torch.no_grad():
                            output = test_pred(img, model)
                            output = output[:, :, :h, :w]

                            output = torch.sigmoid(output)
                            output = output.cpu().data.numpy()

                        for i in range(output.shape[0]):
                            pred = output[i]
                            pred = pred[0]
                            pred = np.array(pred, dtype='float32')

                            mask_label_in_path = os.path.join(TRAIN_MASK_DIR, name[i])
                            prev_label = cv2.imread(mask_label_in_path, cv2.IMREAD_GRAYSCALE) / 255
                            current_binary_pred = update_gt_update_degen_corr(pred, prev_label, thresh_Tb, thresh_k, [h, w], degen=degen)
                            current_binary_pred = current_binary_pred * 255
                            cv2.imwrite(os.path.join(TRAIN_MASK_DIR, name[i]), current_binary_pred)
            else:
                pass

            train_loader, val_loader = get_loaders(
                TRAIN_IMG_DIR,
                TRAIN_MASK_DIR,
                VAL_IMG_DIR,
                VAL_MASK_DIR,
                PATCH_SIZE,
                TRAIN_BATCH_SIZE,
                TEST_BATCH_SIZE,
                train_transform,
                val_transforms,
                NUM_WORKERS,
                PIN_MEMORY,
            )

        else:
            print("choose_dataset_type赋予的值不对，应该为[mask, masks_coarse, masks_centroid]中的一个")
            sys.exit(0)

        train_mioU, train_nioU, train_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler, epoch)

        if epoch + 1 > num_start_test_epochs:
            mioU, nioU, test_loss, fa, pd = val_fn(val_loader, model, loss_fn, epoch)
            # mIoUandPD = 0.5 * mioU + 0.5 * pd

            # save model
            checkpoint = {
                'epoch': epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                'best_mIoU': best_mIoU,
                'best_nIoU': best_nIoU,
                'best_PD': best_PD,
                'bestmIoUandPD': bestmIoUandPD,

            }

            # if bestmIoUandPD < mIoUandPD:
            #     bestmIoUandPD = mIoUandPD
            #     bestmIoUandPD_mIoU = mioU
            #     bestmIoUandPD_nIoU = nioU
            #     bestmIoUandPD_FA = fa
            #     bestmIoUandPD_PD = pd
            #     bestmIoUandPD_epoch = epoch
            #     torch.save(checkpoint, save_bestmiouandPD_file_name)
            # else:
            #     bestmIoUandPD = bestmIoUandPD

            if best_mIoU < mioU:
                best_mIoU = mioU
                best_mIoU_nIoU = nioU
                best_mIoU_FA = fa
                best_mIoU_PD = pd
                best_mIoU_epoch = epoch
                torch.save(checkpoint, save_best_miou_file_name)
            else:
                best_mIoU = best_mIoU

            # if best_PD < pd:
            #     best_PD = pd
            #     bestPD_mIou = mioU
            #     bestPD_nIou = nioU
            #     bestPD_FA = fa
            #     bestPD_epoch = epoch
            #     torch.save(checkpoint, save_best_PD_file_name)
            # else:
            #     best_PD = best_PD

            print(f"当前epoch:{epoch + 1}  train_mioU:{round(train_mioU, 4)}  train_nioU:{round(train_nioU, 4)} \n"
                  f"当前epoch:{epoch + 1}  mioU:{round(mioU, 4)}  nioU:{round(nioU, 4)}  FA:{round(fa * 1000000, 3)}  PD:{round(pd, 4)} \n"
                  f"best_epoch:{best_mIoU_epoch + 1}  best_miou:{round(best_mIoU, 4)}  b_niou:{round(best_mIoU_nIoU, 4)}  FA:{round(best_mIoU_FA * 1000000, 3)}  PD:{round(best_mIoU_PD, 4)}\n"
                  # f"best_epoch:{bestPD_epoch + 1}  b_miou:{round(bestPD_mIou, 4)}  best_niou:{round(bestPD_nIou, 4)}  FA:{round(bestPD_FA * 1000000, 3)}  PD:{round(best_PD, 4)}\n"
                  # f"best_epoch:{bestmIoUandPD_epoch + 1}  bestmIoUandPD:{round(bestmIoUandPD, 4)}  bestmIoUandPD_miou:{round(bestmIoUandPD_mIoU, 4)}  bestmIoUandPD_nioU:{round(bestmIoUandPD_nIoU, 4)}  FA:{round(bestmIoUandPD_FA * 1000000, 3)}  PD:{round(bestmIoUandPD_PD, 4)}"
                  )

            save_file.write(
                f"当前epoch:{epoch + 1}  train_mioU:{round(train_mioU, 4)}  train_nioU:{round(train_nioU, 4)} \n")
            save_file.write(
                f"epoch is:{epoch + 1}  mioU:{round(mioU, 4)}  nioU:{round(nioU, 4)}  FA:{round(fa, 3)}  PD:{round(pd, 4)}\n")
            save_file.write(
                f"best_epoch:{best_mIoU_epoch + 1}  best_miou:{round(best_mIoU, 4)}  b_niou:{round(best_mIoU_nIoU, 4)}  FA:{round(best_mIoU_FA * 1000000, 3)}  PD:{round(best_mIoU_PD, 4)}\n")
            # save_file.write(
            #     f"best_epoch:{bestPD_epoch + 1}  b_miou:{round(bestPD_mIou, 4)}  best_niou:{round(bestPD_nIou, 4)}  FA:{round(bestPD_FA * 1000000, 3)}  PD:{round(best_PD, 4)}\n")
            # save_file.write(
            #     f"best_epoch:{bestmIoUandPD_epoch + 1}  bestmIoUandPD:{round(bestmIoUandPD, 4)}  bestmIoUandPD_miou:{round(bestmIoUandPD_mIoU, 4)}  bestmIoUandPD_nioU:{round(bestmIoUandPD_nIoU, 4)}   FA:{round(bestmIoUandPD_FA * 1000000, 3)}  PD:{round(bestmIoUandPD_PD, 4)}\n")

        else:
            mioU = 0
            nioU = 0
            test_loss = 100000
            fa = 100000
            pd = 0
            mIoUandPD = 0.5 * mioU + 0.5 * pd
            # save model
            checkpoint = {
                'epoch': epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                'best_mIoU': best_mIoU,
                'best_nIoU': best_nIoU,
                'best_PD': best_PD,
                'bestmIoUandPD': bestmIoUandPD,

            }

            print(f"当前epoch:{epoch + 1}  train_mioU:{round(train_mioU, 4)}  train_nioU:{round(train_nioU, 4)}")
            save_file.write(
                f"当前epoch:{epoch + 1}  train_mioU:{round(train_mioU, 4)}  train_nioU:{round(train_nioU, 4)} \n")

        if (epoch + 1) % 20 == 0:
            save_model_file_name = os.path.join(save_model_file_path,
                                                'model_' + MODEL_NAME + '_epoch_' + str(epoch + 1) + '.pth.tar')
            # torch.save(checkpoint, save_model_file_name)
        else:
            pass

        num_epoch.append(epoch + 1)
        num_train_loss.append(train_loss)
        num_test_loss.append(test_loss)
        num_mioU.append(mioU)
        num_nioU.append(nioU)


    save_file.write(datetime.now().strftime("%Y-%m-%d, %H:%M:%S\n"))
    save_file.write('\n---------------------------------------end--------------------------------------------------\n')



if __name__ == "__main__":
    check_path(demo_train_input_path)
    if choose_dataset_type == 'mask':
        pass
    elif choose_dataset_type == 'masks_coarse' or choose_dataset_type == 'masks_centroid':
        make_dir(TRAIN_IMG_DIR)
        make_dir(TRAIN_MASK_DIR)
        make_dir(train_points_dir)

        make_dir(nc_img_dir)
        make_dir(nc_mask_dir)
        make_dir(nc_points_dir)
    else:
        print("choose_dataset_type赋予的值不对，应该为[mask, masks_coarse, masks_centroid]中的一个")
        sys.exit(0)
    main()
