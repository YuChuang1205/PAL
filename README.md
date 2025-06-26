## The official complete code for paper "From Easy to Hard: Progressive Active Learning Framework for Infrared Small Target Detection with Single Point Supervision" [[Paper/arXiv](https://arxiv.org/abs/2412.11154)] [Accepted by ICCV2025]  

<!--
<div align="center">
In this project demo, we have integrated multiple SIRST detection networks (ACM, ALC, MLCL-Net, ALCL-Net, DNANet, GGL-Net, UIUNet, MSDA-Net), label forms (Full supervision, Coarse single-point supervision, Centroid single-point supervision), and datasets (SIRST, NUDT, IRSTD-1K and SIRST3). At the same time, more networks and functions can be integrated into the project later. We hope we can contribute to the development of this field.
</div><br>
-->

In this project demo, we have integrated multiple SIRST detection networks ([**ACM**](https://arxiv.org/abs/2009.14530), [**ALC**](https://arxiv.org/abs/2012.08573), [**MLCL-Net**](https://doi.org/10.1016/j.infrared.2022.104107), [**ALCL-Net**](https://ieeexplore.ieee.org/document/9785618), [**DNANet**](https://arxiv.org/abs/2106.00487), [**GGL-Net**](https://ieeexplore.ieee.org/abstract/document/10230271), [**UIUNet**](https://arxiv.org/abs/2212.00968), [**MSDA-Net**](https://arxiv.org/abs/2406.02037)), label forms (Full supervision, Coarse single-point supervision, Centroid single-point supervision), and datasets ([**SIRST**](https://ieeexplore.ieee.org/document/9423171), [**NUDT-SIRST**](https://ieeexplore.ieee.org/document/9864119), [**IRSTD-1k**](https://ieeexplore.ieee.org/document/9880295) and [**SIRST3**](https://arxiv.org/pdf/2304.01484)). At the same time, more networks and functions can be integrated into the project later. We hope we can contribute to the development of this field.

<p align="center">
  <img src="imgs/Main results.png" alt="Main results" width="600"/></br>
</p>  

<div align="center">
Comparison of different methods on the SIRST3 dataset. <i>CNN Full</i>, <i>CNN Coarse</i>, and <i>CNN Centroid</i> denote CNN-based methods under full supervision, coarse and centroid point supervision.
</div><br>


<!--
<div align="center">
  Comparison of different methods on the SIRST3 dataset. CNN Full, CNN Coarse, and CNN Centroid denote CNN-based methods under full supervision, coarse and centroid point supervision. The curve trend of CNNs equipped with the PAL framework is basically consistent with that of full supervision, whereas CNNs with the LESPS framework is not. In addition, compared with the LESPS, using our PAL framework can improve by 8.53%-29.1% and 1.07%-12.03% in the IoU and Pd.
</div><br>
-->

## Overview

We consider that an excellent learning process should be from easy to hard and take into account the learning ability of the current learner (model) rather than directly treating all tasks (samples) equally. Inspired by organisms gradually adapting to the environment and continuously accumulating knowledge, we first propose an innovative progressive active learning idea, which emphasizes that the network progressively and actively recognizes and learns more hard samples to achieve continuous performance enhancement. For details, please see [[Paper/arXiv](https://arxiv.org/abs/2412.11154)].
   
<p align="center">
  <img src="imgs/PAL framework.png" alt="PAL framework" width="800"/></br>
</p>


## Datasets
1. Original datasets
* **NUDT-SIRST** [[Original dataset](https://pan.baidu.com/s/1WdA_yOHDnIiyj4C9SbW_Kg?pwd=nudt)] [[paper](https://ieeexplore.ieee.org/document/9864119)]
* **SIRST** [[Original dataset](https://github.com/YimianDai/sirst)] [[paper](https://ieeexplore.ieee.org/document/9423171)]
* **IRSTD-1k** [[Original dataset](https://drive.google.com/file/d/1JoGDGF96v4CncKZprDnoIor0k1opaLZa/view)] [[paper](https://ieeexplore.ieee.org/document/9880295)]
* **SIRST3** [[Original dataset](https://github.com/XinyiYing/LESPS)] [[paper](https://arxiv.org/pdf/2304.01484)]

2. The labels are processed according to the "coarse_anno.m" and "centroid_anno.m" files in the "tools" folder to produce coarse point labels and centroid point labels. (**You can also skip this step and use the complete dataset in step 3 directly.**)

3. The datasets we created from original datasets (**can be used directly in our demo**)
     
* [💎 Download the dataset required by our code!!!](https://pan.baidu.com/s/1_QIs9zUM_7MqJgwzO2aC0Q?pwd=1234)  

  
## How to use our code
1. Download the dataset
   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Click [download datasets](https://pan.baidu.com/s/1_QIs9zUM_7MqJgwzO2aC0Q?pwd=1234) 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Unzip the downloaded compressed package to the root directory of the project.

2. Creat a Anaconda Virtual Environment

    ```
    conda create -n PAL python=3.8 
    conda activate PAL 
    ```
3. Configure the running environment
   
   ```
    pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
    pip install segmentation_models_pytorch -i https://pypi.tuna.tsinghua.edu.cn/simple
    pip install PyWavelets -i https://pypi.tuna.tsinghua.edu.cn/simple
    pip install scikit-image -i https://pypi.tuna.tsinghua.edu.cn/simple
    pip install albumentations==1.3.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
    pip install scikit-learn matplotlib thop h5py SimpleITK scikit-image medpy yacs torchinfo
    ```
4. Training the model  
   
    The default model, dataset and label forms are MSDA-Net, SIRST3, and coarse point labels. If you need to train others, please modify the corresponding setting in "train_model.py". Just change the settings to your choice. It's very simple. For details, please see the beginning of the code of "train_model.py".<br>
    ```
    python train_model.py
    ```
5. Testing the Model  
     
    The default model, dataset and label forms are MSDA-Net, SIRST3, and coarse point labels. If you need to test others, please modify the corresponding setting in "test_model.py". Notably, in the "test_model.py" file, you also need to assign the name of the folder where the weight file is located to the "test_dir_name" variable so that the program can find the corresponding model weights. For details, please see the beginning of the code of "test_model.py".
    ```
    python test_model.py
    ```
6. Performance Evaluation
    Use "cal_mIoU_and_nIoU.py" and "cal_PD_and_Fa.py" for performance evaluation. Notably, the corresponding folder path should be replaced. default：SIRST3.
    ```
    python cal_mIoU_and_nIoU.py
    python cal_PD_and_Fa.py
    ```
    
## Results  

* **Quantative Results on the SIRST3 dataset with Coarse point labels:**
<p align="center">
  <img src="imgs/Results on the SIRST3 with coarse point label.png" alt="Results on the SIRST3 with coarse point label" width="800"/>
</p>  

* **Quantative Results on the three individual datasets with Coarse point labels:**
<p align="center">
  <img src="imgs/Results on the three separate dataset with coarse point label.png" alt="Results on the three separate dataset with coarse point label" width="800"/>
</p>  
  
* **Quantative Results on the SIRST3 dataset with Centroid point labels:**
<p align="center">
  <img src="imgs/Results on the SIRST3 with centroid point label.png" alt="Results on the SIRST3 with centroid point label" width="800"/>
</p>   

* **Quantative Results on the three individual datasets with Centroid point labels:**
<p align="center">
  <img src="imgs/Results on the three separate dataset with centroid point label.png" alt="Results on the three separate dataset with centroid point label" width="800"/>
</p>  

* **Qualitative results on the SIRST3 dataset with Coarse point labels:** (Red denotes the correct detections, blue denotes the false detections, and yellow denotes the missed detections.)
<p align="center">
  <img src="imgs/Visualization on the SIRST3 with coarse point label.png" alt="Visualization on the SIRST3 with coarse point label" width="800"/>
</p>  

* **Qualitative results on the SIRST3 dataset with Centroid point labels:** (Red denotes the correct detections, blue denotes the false detections, and yellow denotes the missed detections.)
<p align="center">
  <img src="imgs/Visualization on the SIRST3 with centroid point label.png" alt="Visualization on the SIRST3 with centroid point label" width="800"/>
</p>  



## Citation

If you find this repo helpful, please give us a 🤩**star**🤩. Please consider citing the **PAL** if it benefits your project. <br>  

BibTeX reference is as follows.
```
@misc{yu2024easyhardprogressiveactive,
      title={From Easy to Hard: Progressive Active Learning Framework for Infrared Small Target Detection with Single Point Supervision}, 
      author={Chuang Yu and Jinmiao Zhao and Yunpeng Liu and Sicheng Zhao and Yimian Dai and Xiangyu Yue},
      year={2024},
      eprint={2412.11154},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.11154}, 
}
```

word reference is as follows.
```
Chuang Yu, Jinmiao Zhao, Yunpeng Liu, Sicheng Zhao, Yimian Dai, Xiangyu Yue. From Easy to Hard: Progressive Active Learning Framework for Infrared Small Target Detection with Single Point Supervision. arXiv preprint arXiv:2412.11154, 2024.
```
 


## Other link

1. My homepage: [[YuChuang](https://github.com/YuChuang1205)]
2. "MSDA-Net" demo: [[Link](https://github.com/YuChuang1205/MSDA-Net)]  
