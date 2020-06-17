import torch
from model.unet import UNet
from model.unet_plus import NestedUNet
from lib.meanshift import meanshift
from lib.watershed import watershed
from torchvision.transforms import transforms as T
from torch.utils.data import DataLoader
from dataset.celldataset import CellDataset1, CellDataset2

if __name__ == "__main__":
    #如果要对dataset2进行预测，则相应路径均需改变。
    model = UNet(1, 1)
    #model.load_state_dict(torch.load("pretrain/weights_80.pth", map_location='cpu'))
    x_transform = T.Compose([
      T.ToTensor(),
      T.Normalize([0.5], [0.5])])
    y_transform = T.ToTensor()
    model.eval()
    result_path = 'dataset/dataset1/test_RES'
    cell_dataset = CellDataset1('dataset/dataset1/test/', 'dataset/dataset1/train_GT/SEG/', transform=x_transform,
                             target_transform=y_transform)
    dataloader = DataLoader(cell_dataset)
    #选择任一方法进行后处理
    meanshift(model, result_path, dataloader, 628) #如果是dataset2，则将最后一个数改为500
    #watershed(model, result_path, dataloader, 628)
