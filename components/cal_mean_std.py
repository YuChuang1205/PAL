import numpy as np
from PIL import Image
import os
#

def Calculate_mean_std(img_dir):
    img_list = os.listdir(img_dir)

    mean_list = []
    std_list = []

    for i in range(len(img_list)):
        #print(i)
        img_path = os.path.join(img_dir,img_list[i])
        img = np.array(Image.open(img_path).convert("L"))
        mean_list.append(img.mean())
        std_list.append(img.std())
    mean_out = np.mean(mean_list)/255
    std_out = np.mean(std_list)/255
    print("路径为：", img_dir)
    print("数据集均值为：", mean_out)
    print("数据集方差为：", std_out)

    return mean_out, std_out

