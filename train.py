import torch
from torchvision.transforms import transforms as T
from torch import optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import numpy as np
import time
from torch.utils.data import random_split
import random
from model.unet import UNet
from model.unet_plus import NestedUNet
from dataset.celldataset import CellDataset1, CellDataset2
from lib.iou import iou


def train_model(model, criterion, optimizer, batch_size, train_dataload, test_dataload, scheduler, num_epochs, last_epoch):
    train_size = len(train_dataload.dataset) #数据集的大小
    test_size = len(test_dataload.dataset)
    #train_acc_ls = np.load('train_acc.npy') #如果需要中途重新开始训练，则取消这段注释
    #test_acc_ls = np.load('test_acc.npy')
    train_acc_ls = []
    test_acc_ls = []
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+last_epoch, num_epochs - 1))
        print('-' * 10)
        model.train() #训练模式
        epoch_loss = 0 #每个epoch的loss
        train_acc = 0 #训练集的准确率
        test_acc = 0 #测试集的准确率
        time_start=time.time()
        for step, (x, y) in enumerate(train_dataload):
            optimizer.zero_grad()
            x = x.type(torch.FloatTensor)
            y = y.type(torch.FloatTensor)
            inputs = x.to(device)
            labels = y.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            train_acc += iou(outputs.cpu(), y.cpu())
        #接下来进行测试集的操作
        model.eval()
        for step,(x, y) in enumerate(test_dataload):
          x = x.type(torch.FloatTensor)
          y = y.type(torch.FloatTensor)
          x = x.to(device)
          y_pred = model(x)
          test_acc += iou(y_pred.cpu(), y.cpu())
        time_end=time.time()
        print("epoch %d loss:%0.3f" % (epoch+last_epoch, epoch_loss))
        print("Train Accuracy:%0.3f" % (train_acc / train_size * batch_size))
        print("Test Accuracy:%0.3f" % (test_acc / test_size * batch_size))
        print("Training Time:%0.3f" % (time_end - time_start))
        #train_acc_ls = np.append(train_acc_ls, train_acc / train_size * batch_size)
        #test_acc_ls = np.append(test_acc_ls, test_acc / test_size * batch_size)
        train_acc_ls.append(train_acc / train_size * batch_size)
        test_acc_ls.append(test_acc / test_size * batch_size)
        scheduler.step() #更新学习率
        if (epoch+last_epoch) % 10 == 0:
          torch.save(model.state_dict(),'weights_%d.pth' % (epoch+last_epoch)) #模型保存
          np.save('train_acc.npy', np.array(train_acc_ls))
          np.save('test_acc.npy', np.array(test_acc_ls))
          print ("Model saved successfully!")

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device) #GPU能否使用
    model = UNet(1, 1).to(device)
    #model = NestedUNet(1, 1).to(device)
    # model.load_state_dict(torch.load("pretrain/weights_80.pth",map_location='cpu'))
    # 设置随机数种子，保证复现能力
    torch.backends.cudnn.deterministic = True
    random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    np.random.seed(1)
    batch_size = 2
    learning_rate = 0.001
    criterion = torch.nn.BCELoss()
    optimizer = optim.Adam([{'params': model.parameters(), 'initial_lr': learning_rate}], lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8, last_epoch=0)  # 每10个epoch衰减0.8,注意last_epoch的设置！！
    x_transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.5], [0.5])])
    y_transform = T.ToTensor()
    cell_dataset = CellDataset1('dataset/dataset1/train/', 'dataset/dataset1/train_GT/SEG/', transform=x_transform,
                                target_transform=y_transform)
    #cell_dataset = CellDataset2('dataset/dataset2/train/', 'dataset/dataset2/train_GT/SEG/', transform=x_transform,
    #                           target_transform=y_transform) #对应数据集2
    train_size = int(0.8 * len(cell_dataset)) #划分训练集和验证集,8:2
    test_size = len(cell_dataset) - train_size
    train_dataset, test_dataset = random_split(cell_dataset, [train_size, test_size])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    train_model(model, criterion, optimizer, batch_size, train_dataloader, test_dataloader, scheduler, 300, 0)