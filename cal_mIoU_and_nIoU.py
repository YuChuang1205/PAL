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


def cal_iou(input_pred, input_true):

    input_pred = input_pred/255
    input_true = input_true/255

    inter_count = np.sum(input_pred*input_true)
    outer_count = np.sum(input_pred) + np.sum(input_true) - inter_count


    return inter_count, outer_count

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

root_path = os.path.abspath('.')
# input_path = os.path.join(root_path, 'input')

dataset_path = os.path.join(root_path,'dataset','SIRST3')
test_dataset_path = os.path.join(dataset_path,'val')

input_pred_path = os.path.join(test_dataset_path, 'pre_results')
input_true_path = os.path.join(test_dataset_path, 'mask')

input_pred_list = os.listdir(input_true_path)

input_pred_list.sort()

inter_count_all = 0
outer_count_all = 0
niou_list = []
for i in range(len(input_pred_list)):
    print("正在处理：", input_pred_list[i])
    img_name = input_pred_list[i]
    input_pred_img_path = os.path.join(input_pred_path, img_name)
    input_true_img_path = os.path.join(input_true_path, img_name)
    input_pred = cv2.imread(input_pred_img_path, cv2.IMREAD_GRAYSCALE)
    #input_pred = sigmoid(input_pred)
    input_pred = np.where(input_pred>127,255,0)
    #print(input_pred)
    input_true = cv2.imread(input_true_img_path, cv2.IMREAD_GRAYSCALE)
    inter_count, outer_count = cal_iou(input_pred, input_true)
    inter_count_all = inter_count_all + inter_count
    outer_count_all = outer_count_all + outer_count
    niou_list.append(inter_count/(outer_count+1e-6))

niou = np.mean(niou_list)
miou = inter_count_all/outer_count_all

print("moiu为：", miou)
print("niou为：", niou)
print("Done!!!!!!!!!!")





