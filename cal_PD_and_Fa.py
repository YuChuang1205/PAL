#!/usr/bin/python3
# coding = gbk
"""
@Author : yuchuang,zhaojinmiao
@Time :
@desc:
"""
import os
import cv2
import numpy as np
import sys
from skimage import measure
import os
from os.path import join, isfile


def cal_Pd_Fa(input_pred, input_true):
    image = measure.label(input_pred, connectivity=2)  
    coord_image = measure.regionprops(image)
    label = measure.label(input_true, connectivity=2)
    coord_label = measure.regionprops(label)
    target = len(coord_label)
    image_area_total = []
    image_area_match = []
    distance_match = []
    dismatch = []
    for K in range(len(coord_image)):
        area_image = np.array(coord_image[K].area)  
        #print(area_image)
        image_area_total.append(area_image)  
        
    for i in range(len(coord_label)):
        centroid_label = np.array(list(coord_label[i].centroid)) 
        for m in range(len(coord_image)):
            centroid_image = np.array(list(coord_image[m].centroid))  
            distance = np.linalg.norm(centroid_image - centroid_label)  
            area_image = np.array(coord_image[m].area)  
            if distance < 3:  
                distance_match.append(distance)  
                image_area_match.append(area_image)  
                del coord_image[m]  
                break
   
    FA = np.sum(image_area_total) - np.sum(image_area_match)
    PD = len(distance_match) 
    return target, FA, PD

# IMAGE_SIZE = 512
root_path = os.path.abspath('.')
# input_path = os.path.join(root_path, 'input')
dataset_path = os.path.join(root_path,'dataset','SIRST3')  
test_dataset_path = os.path.join(dataset_path,'val')

input_pred_path = os.path.join(test_dataset_path, 'pre_results')
input_true_path = os.path.join(test_dataset_path, 'mask')

# img_num = count_images_in_folder(input_pred_path)
# print(img_num)
input_pred_list = os.listdir(input_true_path)
img_num = len(input_pred_list)
print(img_num)
input_pred_list.sort()
target_all = 0
FA_all = 0
PD_all = 0
all_pixel = 0
for i in range(len(input_pred_list)):
    print("正在处理：", input_pred_list[i])
    img_name = input_pred_list[i]
    input_pred_img_path = os.path.join(input_pred_path, img_name)
    input_true_img_path = os.path.join(input_true_path, img_name)
    input_pred = cv2.imread(input_pred_img_path, cv2.IMREAD_GRAYSCALE)
    print(input_pred.shape)
    all_pixel += input_pred.shape[0] * input_pred.shape[1]
    input_pred = np.where(input_pred>127,255,0)
    input_true = cv2.imread(input_true_img_path, cv2.IMREAD_GRAYSCALE)
    target, FA, PD = cal_Pd_Fa(input_pred, input_true)
    #print(FA)
    target_all = target_all + target
    FA_all = FA_all + FA
    PD_all = PD_all + PD
Pd = PD_all / target_all
Fa = FA_all / all_pixel

print("Pd为：", Pd)
print("Fa为：", Fa*1000000)
print("Done!!!!!!!!!!")





