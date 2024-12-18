
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage import measure

TEST_BATCH_SIZE = 1
class SigmoidMetric():
    def __init__(self, score_thresh=0.5):
        self.score_thresh = score_thresh
        self.reset()

    def update(self, pred, labels):
        correct, labeled = self.batch_pix_accuracy(pred, labels)
        inter, union = self.batch_intersection_union(pred, labels)

        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union

    def get(self):
        """Gets the current evaluation result."""
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return pixAcc, mIoU

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0

    def batch_pix_accuracy(self, output, target):
        assert output.shape == target.shape
        output = output.cpu().detach().numpy()
        target = target.cpu().detach().numpy()

        predict = (output > self.score_thresh).astype('int64')  # P
        target = (target > self.score_thresh).astype('int64')
        # -----------------------------#在这个之前必须变为0、1
        pixel_labeled = np.sum(target > 0)  # T
        pixel_correct = np.sum((predict == target) * (target > 0))  # TP
        assert pixel_correct <= pixel_labeled
        return pixel_correct, pixel_labeled

    def batch_intersection_union(self, output, target):
        mini = 1
        maxi = 1  # nclass
        nbins = 1  # nclass
        predict = (output.cpu().detach().numpy() > self.score_thresh).astype('int64')  # P
        target = (target.cpu().detach().numpy() > self.score_thresh).astype('int64')  # T
        # target = target.cpu().numpy().astype('int64')  # T
        intersection = predict * (predict == target)  # TP


        # areas of intersection and union
        area_inter, _ = np.histogram(intersection, bins=nbins, range=(mini, maxi)) # 统计二值化图像中像素值为1的像素数量
        area_pred, _ = np.histogram(predict, bins=nbins, range=(mini, maxi))
        area_lab, _ = np.histogram(target, bins=nbins, range=(mini, maxi))
        area_union = area_pred + area_lab - area_inter
        assert (area_inter <= area_union).all()
        return area_inter, area_union


class SamplewiseSigmoidMetric():
    def __init__(self, nclass, score_thresh=0.5):
        self.nclass = nclass
        self.score_thresh = score_thresh
        self.reset()

    def update(self, preds, labels):
        """Updates the internal evaluation result."""
        inter_arr, union_arr = self.batch_intersection_union(preds, labels)
        self.total_inter = np.append(self.total_inter, inter_arr)
        self.total_union = np.append(self.total_union, union_arr)

    def get(self):
        """Gets the current evaluation result."""
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return IoU, mIoU

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.total_inter = np.array([])
        self.total_union = np.array([])
        self.total_correct = np.array([])
        self.total_label = np.array([])

    def batch_intersection_union(self, output, target):
        """nIoU"""
        # inputs are tensor
        # the category 0 is ignored class, typically for background / boundary
        mini = 1
        maxi = 1  # nclass
        nbins = 1  # nclass

        predict = (output.cpu().detach().numpy() > self.score_thresh).astype('int64')  # P
        target = (target.cpu().detach().numpy() > self.score_thresh).astype('int64')  # T
        # target = target.cpu().detach().numpy().astype('int64')  # T
        intersection = predict * (predict == target)  # TP

        num_sample = intersection.shape[0]
        area_inter_arr = np.zeros(num_sample)
        area_pred_arr = np.zeros(num_sample)
        area_lab_arr = np.zeros(num_sample)
        area_union_arr = np.zeros(num_sample)

        for b in range(num_sample):
            # areas of intersection and union
            area_inter, _ = np.histogram(intersection[b], bins=nbins, range=(mini, maxi))
            area_inter_arr[b] = area_inter

            area_pred, _ = np.histogram(predict[b], bins=nbins, range=(mini, maxi))
            area_pred_arr[b] = area_pred

            area_lab, _ = np.histogram(target[b], bins=nbins, range=(mini, maxi))
            area_lab_arr[b] = area_lab

            area_union = area_pred + area_lab - area_inter
            area_union_arr[b] = area_union

            assert (area_inter <= area_union).all()

        return area_inter_arr, area_union_arr


class PD_FA_2():
    def __init__(self, nclass):
        super(PD_FA_2, self).__init__()
        self.nclass = nclass
        self.image_area_total = []
        self.image_area_match = []
        self.FA = 0
        self.PD = 0
        self.target = 0
        self.all_pixel = 0
    def update(self, preds, labels):


        predits  = np.array((preds > 0.5).cpu()).astype('int64')

        for i in range(predits.shape[0]):
            self.image_h = predits.shape[-2]
            self.image_w = predits.shape[-1]
            self.all_pixel += self.image_h * self.image_w


        labelss = np.array((labels > 0.5).cpu()).astype('int64') # P

        image = measure.label(predits, connectivity=2)# 寻找最大连通域，二维图像当connectivity=2时代表8连通.
        # print('image.size', image.shape)
        #print(image)
        coord_image = measure.regionprops(image)# 返回所有连通区块的属性列表# 属性列表中包含了每个连通区块的一些统计信息，比如面积、中心坐标等
        #print(coord_image)
        label = measure.label(labelss, connectivity=2)
        coord_label = measure.regionprops(label)

        self.target    += len(coord_label) # 标签总小目标数len(coord_label)
        self.image_area_total = []
        self.image_area_match = []
        self.distance_match   = []
        self.dismatch         = []

        for K in range(len(coord_image)):
            area_image = np.array(coord_image[K].area) # coord_image[K].area——第K个连通区域(目标)中，区域内像素点总数
            self.image_area_total.append(area_image) #预测图像中各连通区域的面积列表
        # 比较标签图像中的每个连通域的质心与预测图像中的连通域的质心之间的距离，如果距离小于3，则将预测图像连通域的面积和距离添加到相应的列表中，同时删除已匹配的预测图像连通域。
        for i in range(len(coord_label)):
            centroid_label = np.array(list(coord_label[i].centroid)) # coord_label[i].centroid标签连通区域i的质心坐标，centroid_label标签中目标的坐标集
            for m in range(len(coord_image)):
                centroid_image = np.array(list(coord_image[m].centroid)) # coord_image[m].centroid预测连通区域m的质心坐标，centroid_image预测图像中目标的坐标集
                distance = np.linalg.norm(centroid_image - centroid_label) #计算当前标签图像连通域 i 的质心与预测图像连通域 m 的质心之间的欧氏距离。
                area_image = np.array(coord_image[m].area) # 获取当前预测图像连通域 m 的面积。
                if distance < 3: # 如果质心距离小于3（这里的3是一个阈值，可以根据实际情况调整）。
                    self.distance_match.append(distance) # 将匹配的质心距离添加到 self.distance_match 列表中
                    self.image_area_match.append(area_image) # 将匹配的预测图像连通域面积添加到 self.image_area_match 列表中。

                    del coord_image[m] # 从 coord_image 列表中删除连通域 m，因为它已经匹配到了。
                    break

        self.dismatch = np.sum(self.image_area_total)-np.sum(self.image_area_match)
        self.FA += self.dismatch
        self.PD+=len(self.distance_match) # 预测到的小目标数

    def get(self,img_num):
        # print("imgae_w:", self.image_w)
        # print("imgae_h:", self.image_h)
        Final_FA = self.FA / self.all_pixel
        Final_PD = self.PD / self.target
        #print("预测的目标点PD",self.PD)
        #print("预测的目标点target",self.target)

        return Final_FA,Final_PD


    def reset(self):
        self.FA  = 0
        self.PD  = 0
        self.target = 0
        self.all_pixel = 0

if __name__ == '__main__':
    pred = torch.rand(8, 1, 512, 512)
    target = torch.rand(8, 1, 512, 512)
    m1 = SigmoidMetric()
    m2 = SamplewiseSigmoidMetric(nclass=1, score_thresh=0.5)
    m1.update(pred, target)
    m2.update(pred, target)
    pixAcc, mIoU = m1.get()
    _, nIoU = m2.get()
