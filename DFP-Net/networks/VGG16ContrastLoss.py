import torch.nn as nn
import torch
from torchvision import models

class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()

        vgg_pretrained_features = models.vgg16(pretrained=True).features

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()

        for x in range(10):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(10, 17):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(17, 31):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        f_maxpool1 = self.slice1(X)
        f_maxpool2 = self.slice2(f_maxpool1)
        f_maxpool3 = self.slice3(f_maxpool2)

        return [f_maxpool1, f_maxpool2, f_maxpool3]

class ContrastLoss(nn.Module):#ablation 消融
    def __init__(self, ablation=False):

        super(ContrastLoss, self).__init__()
        self.vgg = Vgg16().cuda()
        self.mse = nn.MSELoss()
        self.weights = [0.4, 0.6, 1.0]
        self.ab = ablation

    def forward(self, a, p, n):
        a_vgg, p_vgg, n_vgg = self.vgg(a), self.vgg(p), self.vgg(n)
        loss = 0

        d_ap, d_an = 0, 0
        for i in range(len(a_vgg)):
            d_ap = self.mse(a_vgg[i], p_vgg[i].detach())
            if not self.ab:
                d_an = self.mse(a_vgg[i], n_vgg[i].detach())
                contrastive = d_ap / (d_an + 1e-7)
            else:
                contrastive = d_ap

            loss += self.weights[i] * contrastive

        return loss

if __name__=="__main__":
    x=torch.ones(1,3,256,256).cuda()
    y=torch.zeros(1,3,256,256).cuda()
    z=torch.zeros(1,3,256,256).cuda()
    l=ContrastLoss().cuda()
    loss=l(x,y,z).cuda()
    print(Vgg16())
    print(loss)