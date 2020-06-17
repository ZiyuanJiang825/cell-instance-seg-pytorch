import torch
import numpy as np
import cv2
from sklearn.cluster import MeanShift
import imageio
import os.path as osp
import matplotlib.pyplot as plt

def meanshift(model, result_path, dataloader, S): #S is the shape of picture
    for iter, (x, _ )in enumerate(dataloader):
        y_exp = []
        x = x.float()
        y = model(x)
        y = torch.squeeze(y).detach().numpy()
        y = y.astype(np.float32)
        y = cv2.resize(y, (S, S), interpolation=cv2.INTER_NEAREST)
        for i in range(y.shape[0]):
            for j in range(y.shape[0]):
                if y[i, j] >= 0.5:
                    y_exp.append([i, j])
        y_exp = np.array(y_exp)
        print(iter, y_exp.shape[0]) #第几张图片以及图片大小
        y = y.astype(np.uint8)
        R = 22.0 #带宽
        #R = 52.0 #dataset2使用
        cluster_model = MeanShift(bandwidth=R, bin_seeding=True)
        cluster_model.fit(y_exp)
        labels = cluster_model.labels_
        labels = [la + 1 for la in labels]  # 每个标签都加1，因为label从0开始算
        for i in range(len(y_exp)):
            xi = y_exp[i, 0]
            yi = y_exp[i, 1]
            y[xi, yi] = labels[i]
        imageio.imwrite(osp.join(result_path, 'mask{:0>3d}.tif'.format(iter)), y.astype(np.uint16))
        # 以下可视化聚类结果
        '''
        height, width = y.shape[:2]
        label = np.unique(y)
        visual_img = np.zeros((height, width, 3))
        for lab in label:
            if lab == 0:
                continue
            color = np.random.randint(low=0, high=255, size=3)
            visual_img[y == lab, :] = color
        visual_img = visual_img.astype(np.uint8)
        plt.imshow(visual_img)
        plt.pause(0.01)
        plt.show()
        '''