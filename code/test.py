import math
import torch
from denoisingUnet import denoising_unet, device, batch_size
from torch.utils.data import DataLoader
from palsarLoader import PalsarLoader
from sentinelLoader import SentinelLoader
from evaluation import *
from torchvision import transforms

unloader = transforms.ToPILImage()

def main():
    # testdataset = PalsarLoader("../../data/test", train=False)
    testdataset = SentinelLoader("../../data/test", train=False)
    testdataloader = DataLoader(testdataset, batch_size=batch_size, shuffle=False, drop_last=True)

    unetmodel = denoising_unet() #
    unetmodel = unetmodel.to(device)
    unetmodel.load_state_dict(torch.load('./msd/denoising_unet_s_epoch30.pth'))  # 加载训练好的模型

    DiceScore = []
    HD95 = []
    Com = []
    ACC = []

    Cor = []
    IoU = []


    with torch.no_grad():
        unetmodel.eval()
        # file = open("./score/du_p_only_C.txt", "a") #
        for idx, (x, label) in enumerate(testdataloader):
            x, label = x.to(device), label.to(device)  #conv(x),label [b,1,256,256]

            a, b, c, A,B,C,D, pred = unetmodel(x)

            pred = torch.where(pred > 0.5, 1., 0.)

            # predimg = unloader(a[0])
            # predimg.save("./img_p0_new/{}_aa_1.jpg".format(idx))
            #
            # predimg = unloader(b[0])
            # predimg.save("./img_p0_new/{}_bb_1.jpg".format(idx))
            #
            # predimg = unloader(c[0])
            # predimg.save("./img_p0_new/{}_cc_1.jpg".format(idx))
            #
            # predimg = unloader(A[0])
            # predimg.save("./img_p0_new/{}_A_1.jpg".format(idx))
            #
            # predimg = unloader(B[0])
            # predimg.save("./img_p0_new/{}_B_1.jpg".format(idx))
            #
            # predimg = unloader(C[0])
            # predimg.save("./img_p0_new/{}_C_1.jpg".format(idx))
            #
            # predimg = unloader(D[0])
            # predimg.save("./img_p0_new/{}_D_1.jpg".format(idx))
            #
            # predimg = unloader(pred[0])
            # predimg.save("./img_p0_new/{}_pred_1.jpg".format(idx))

            A = torch.where(A > 0.5, 1., 0.)
            B = torch.where(B > 0.5, 1., 0.)
            C = torch.where(C > 0.5, 1., 0.)
            D = torch.where(D > 0.5, 1., 0.)
            img = (A + B + C + D + pred)/5.0
            img = torch.where(img > 0.5, 1., 0.)

            # predimg = unloader(img[0])
            # predimg.save("./img_s_final/{}_img_1.jpg".format(idx))
            #
            # predimg = unloader(img[1])
            # predimg.save("./img_s_final/{}_img_2.jpg".format(idx))
            #
            # predimg = unloader(img[2])
            # predimg.save("./img_s_final/{}_img_3.jpg".format(idx))
            #
            # predimg = unloader(img[3])
            # predimg.save("./img_s_final/{}_img_4.jpg".format(idx))
            #
            # predimg = unloader(img[4])
            # predimg.save("./img_s_final/{}_img_5.jpg".format(idx))
            #
            # predimg = unloader(img[5])
            # predimg.save("./img_s_final/{}_img_6.jpg".format(idx))
            #
            # predimg = unloader(img[6])
            # predimg.save("./img_s_final/{}_img_7.jpg".format(idx))
            #
            # predimg = unloader(img[7])
            # predimg.save("./img_s_final/{}_img_8.jpg".format(idx))

            label = torch.where(label>0.5, 1., 0.)

            diceScore, hd95, com, cor, acc, iou = calculate_metric_percase(img, label)

            if not math.isnan(diceScore):
                DiceScore.append(diceScore)

            if not math.isnan(com):
                Com.append(com)

            if not math.isnan(acc):
                ACC.append(acc)

            if not math.isnan(hd95):
                HD95.append(hd95)

            if not math.isnan(cor):
                Cor.append(cor)

            if not math.isnan(iou):
                IoU.append(iou)

            print("idx={},diceScore={},hd95={},com={},acc={},cor={},iou={} \n".format(idx, diceScore, hd95, com, acc, cor, iou))



            # file.write("{},{},{},{},{} \n".format(idx, diceScore, hd95, com, acc))

    print("total mean bce_diceloss = {}".format(sum(DiceScore) / len(DiceScore)))
    print("total mean hd95 = {}".format(sum(HD95) / len(HD95)))
    print("total mean com = {}".format(sum(Com) / len(Com)))

    print("total mean acc = {}".format(sum(ACC) / len(ACC)))

    print("total mean cor = {}".format(sum(Cor) / len(Cor)))
    print("total mean iou = {}".format(sum(IoU) / len(IoU)))

    # file.write("total mean bce_diceloss = {}\n".format(sum(DiceScore) / len(DiceScore)))
    # file.write("total mean hd95 = {}\n".format(sum(HD95) / len(HD95)))
    # file.write("total mean com = {}\n".format(sum(Com) / len(Com)))
    # file.write("total mean acc = {}\n".format(sum(ACC) / len(ACC)))
    #
    # file.close()



if __name__ == '__main__':
    main()
