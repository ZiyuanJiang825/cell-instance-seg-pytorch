import os
import cv2
import torch.utils.data as data

class CellDataset1(data.Dataset): #对应数据集1
    #创造数据集
    def __init__(self, train_path, train_gt_path, transform = None, target_transform = None):
        n = len(os.listdir(train_path))
        imgs = []
        for i in range(n):
            img = os.path.join(train_path, "t%03d.tif" % i)
            mask = os.path.join(train_gt_path, "man_seg%03d.tif" % i)
            imgs.append([img,mask])
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x_path, y_path = self.imgs[index]
        img_x = cv2.imread(x_path, -1)
        img_y = cv2.imread(y_path, -1)
        img_x = cv2.resize(img_x, (640, 640), interpolation=cv2.INTER_NEAREST)
        img_y = cv2.resize(img_y, (640, 640), interpolation=cv2.INTER_NEAREST)
        img_y[img_y != 0] = 1
        img_x = img_x / 1.0
        img_y = img_y / 1.0
        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            img_y = self.target_transform(img_y)
        return img_x, img_y

    def __len__(self):
        return len(self.imgs)

class CellDataset2(data.Dataset): #对应数据集2
    #创造数据集
    def __init__(self, train_path, train_gt_path, transform = None, target_transform = None):
        n = len(os.listdir(train_path))
        if n<10:
          n=6
        else:
          n=167
        imgs = []
        for i in range(n):
            img = os.path.join(train_path, "t%03d.tif" % i)
            mask = os.path.join(train_gt_path, "man_seg%03d.tif" % i)
            imgs.append([img,mask])
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x_path, y_path = self.imgs[index]
        img_x = cv2.imread(x_path, -1)
        img_y = cv2.imread(y_path, -1)
        img_x = cv2.resize(img_x,(512,512),interpolation=cv2.INTER_NEAREST) #已经是512*512了
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
        img_y = cv2.erode(img_y,  kernel, iterations=2)
        img_y[img_y != 0] = 1
        img_x = img_x / 1.0
        img_y = img_y / 1.0
        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            img_y = self.target_transform(img_y)
        return img_x, img_y

    def __len__(self):
        return len(self.imgs)