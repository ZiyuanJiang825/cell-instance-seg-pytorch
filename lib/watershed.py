import cv2
import torch
import numpy as np
import imageio
import matplotlib.pyplot as plt

def watershed(model, result_path, dataloader, S): #S is the shape of picture
    for iter, (x, _) in enumerate(dataloader):
        x = x.float()
        img = model(x)
        img = torch.squeeze(img).detach().numpy()
        img[img >= 0.5] = 255
        img[img < 0.5] = 0
        img = img.astype(np.uint8)
        gray = img
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        thresh[thresh == 0] = 10
        thresh[thresh == 255] = 0
        thresh[thresh == 10] = 255
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        sure_bg = cv2.dilate(opening, kernel, iterations=2)
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)  # DIST_L1 DIST_C只能 对应掩膜为3    DIST_L2 可以为3或者5
        ret, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        ret, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        markers = cv2.watershed(img, markers)
        for i in range(np.max(markers) + 1):
            color = np.random.randint(low=0, high=255, size=3)
            img[markers == i] = color
        markers = markers - 1
        markers[markers == -2] = 0
        y = cv2.resize(markers, (S, S), interpolation=cv2.INTER_NEAREST)
        y = y.astype(np.uint16)
        '''
        plt.imshow(y)
        plt.show()
        '''
        imageio.imwrite(osp.join(result_path, 'mask{:0>3d}.tif'.format(iter)), y.astype(np.uint16))