import torch
from torch import nn
from denoisingUnet import denoising_unet, device, batch_size
from torch.utils.data import DataLoader
from palsarLoader import PalsarLoader
from sentinelLoader import SentinelLoader
from torchvision import transforms

epochs = 50

def binary_cross_entropy_loss(pred, label):
    criterion_ce = nn.BCELoss()
    ce_loss = criterion_ce(pred, label)
    return ce_loss

def main():
    traindataset = PalsarLoader("../../data/train", train=True)
    # traindataset = SentinelLoader("../data/train", train=True)
    traindataloader = DataLoader(traindataset, batch_size=batch_size, shuffle=True, drop_last=True)

    unetmodel = denoising_unet()
    unetmodel = unetmodel.to(device)

    optimizer = torch.optim.Adam(unetmodel.parameters(), lr=1e-4)

    # 添加学习率调度器
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    for epoch in range(epochs):
        print('epoch={}'.format(epoch))
        unetmodel.train()
        for idx, (x, label) in enumerate(traindataloader):
            x, label = x.to(device), label.to(device)
            x.requires_grad_(True)

            a, b, c, a_up, b_up, c_up, d, pred = unetmodel(x)

            loss1 = binary_cross_entropy_loss(a_up, label)
            loss2 = binary_cross_entropy_loss(b_up, label)
            loss3 = binary_cross_entropy_loss(c_up, label)
            loss4 = binary_cross_entropy_loss(d, label)
            loss5 = binary_cross_entropy_loss(pred, label)
            loss = loss1 + loss2 + loss3 + loss4 + loss5

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('------epoch={}, idx={}, Binary CE Loss={}'.format(epoch, idx, loss.item()))

        # 学习率调度器步进
        scheduler.step()

        # 保存模型参数
        # if((epoch+1)%10 == 0):
        #     torch.save(unetmodel.state_dict(), "./msd_p/denoising_unet_p_stop_epoch{}.pth".format(epoch+1))
        torch.save(unetmodel.state_dict(), "./msd/p_sc.pth")

    print("Training completed, with a total of 60000 iterations!")

if __name__ == '__main__':
    main()
