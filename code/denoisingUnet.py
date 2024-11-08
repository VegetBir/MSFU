import torch
from torch import nn
from torch.nn import functional as F

batch_size = 8
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class conv_block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(conv_block, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size = 3, padding = 1, bias = False),
            nn.BatchNorm2d(out_channel),
            nn.SiLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
        )
        self.mid_conv = nn.Conv2d(in_channel, out_channel, kernel_size = 3, padding = 1)
        self.final_conv = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(out_channel * 2, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.SiLU(),
        )
    def forward(self, x):
        x1 = self.layers(x)
        x2 = self.mid_conv(x)
        x3 = torch.cat([x1, x2], dim = 1)
        x4 = self.final_conv(x3)
        return x4

class channel_attention(nn.Module):
    def __init__(self, size):
        super(channel_attention, self).__init__()
        self.mp = nn.AdaptiveMaxPool2d(1)
        self.w = nn.Sequential(
            nn.Linear(size,size//2,bias=False),
            nn.Sigmoid(),
            nn.Linear(size//2,size,bias=False),
            nn.Sigmoid(),
        )
    def forward(self, x):
        b, c, h, w = x.shape
        mp = self.mp(x).reshape(b, c)
        w = self.w(mp)
        w = w.reshape(b, c, 1, 1)
        out = w * x
        return out

class feature_extraction_net(nn.Module):
    def __init__(self, size, in_channel, out_channel = 2):
        super(feature_extraction_net, self).__init__()
        self.attention = nn.Sequential(
            channel_attention(size),
        )
        self.layers1 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size = 3, padding = 1, bias = False),
            nn.BatchNorm2d(in_channel),
        )
        self.layers2 = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding = 1),
        )
    def forward(self, x):
        x1 = self.attention(x)
        x2 = self.layers1(x1)
        x3 = x2 + x1
        x4 = self.layers2(x3)
        x5 = F.softmax(x4, dim=1)
        pred = x5[:,0:1,:,:]
        return pred

# [b,c,h,w] -> [b,c*2,h/2,w/2]
class down_sample(nn.Module):
    def __init__(self,channel):
        super(down_sample,self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channel,channel*2,kernel_size=3,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(channel*2),
            nn.SiLU()
        )
    def forward(self,x):
        return self.layers(x)

# [b,c,h,w] -> [b,c/2,h*2,w*2]
class up_sample(nn.Module):
    def __init__(self,channel):
        super(up_sample,self).__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(channel,channel//2,kernel_size=4,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(channel//2),
            nn.SiLU(),
        )
    def forward(self,x):
        up = self.layers(x)
        return up

class denoising_unet(nn.Module):
    def __init__(self):
        super(denoising_unet, self).__init__()
        self.init_conv = nn.Conv2d(1, 64, kernel_size = 3, padding = 1)
        self.ec1 = conv_block(64, 64)
        self.ed1 = down_sample(64)
        self.ec2 = conv_block(128, 128)
        self.ed2 = down_sample(128)
        self.ec3 = conv_block(256, 256)
        self.ed3 = down_sample(256)

        self.mid = conv_block(512, 512)

        self.du1 = up_sample(512)
        self.dc1 = conv_block(512, 256)
        self.du2 = up_sample(256)
        self.dc2 = conv_block(256, 128)
        self.du3 = up_sample(128)
        self.dc3 = conv_block(128, 64)

        self.fen1 = feature_extraction_net(512, 512)
        self.fen2 = feature_extraction_net(256, 256)
        self.fen3 = feature_extraction_net(128, 128)
        self.fen4 = feature_extraction_net(64, 64)
        self.fen5 = feature_extraction_net(4, 4)

    def forward(self, x):
        i1 = self.init_conv(x)
        x1 = self.ec1(i1)
        d1 = self.ed1(x1)
        x2 = self.ec2(d1)
        d2 = self.ed2(x2)
        x3 = self.ec3(d2)
        d3 = self.ed3(x3)

        x4 = self.mid(d3)

        u1 = self.du1(x4)
        u1 = torch.cat([x3, u1], dim = 1)
        x5 = self.dc1(u1)
        u2 = self.du2(x5)
        u2 = torch.cat([x2, u2], dim = 1)
        x6 = self.dc2(u2)
        u3 = self.du3(x6)
        u3 = torch.cat([x1, u3], dim = 1)
        x7 = self.dc3(u3)

        a = self.fen1(x4)
        b = self.fen2(x5)
        c = self.fen3(x6)
        d = self.fen4(x7)

        a_up = F.interpolate(a, size=(256, 256), mode='bilinear',align_corners=False)
        b_up = F.interpolate(b, size=(256, 256), mode='bilinear',align_corners=False)
        c_up = F.interpolate(c, size=(256, 256), mode='bilinear',align_corners=False)

        fm = torch.cat([a_up, b_up, c_up, d], dim = 1)
        pred = self.fen5(fm)

        return a, b, c, a_up, b_up, c_up, d, pred


if __name__ == '__main__':
    x = torch.rand([8,1,256,256], device = 'cuda:0')
    m = denoising_unet().to('cuda:0')
    a, b, c, a_up, b_up, c_up, d, pred = m(x)
    print(a_up.shape)
    print(a_up[0][0][0])

    net = denoising_unet()
    print(sum(p.numel() for p in net.parameters()))

