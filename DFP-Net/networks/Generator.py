import torch
import torch.nn as nn
from .PreGenerator import Preprocess
from .up_net import UNet
from .down_net import DehazeNet as down_net
from .final_net import DehazeNet as final_net


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.Preprocess=Preprocess()
        self.up_net=UNet()
        self.down_net=down_net()
        self.final_net=final_net()

    def forward(self, x):

        x_global, x_local, res_dcp_global, res_dcp_local = self.Preprocess(x)

        res_local = self.up_net(x_local,res_dcp_local)

        res_global = self.down_net(x_global,res_dcp_global)

        out = self.final_net(res_global,res_local)

        return out


if __name__=="__main__":
    x=torch.ones(1,3,256,256).cuda()
    net=Generator().cuda()
    a=net(x)
    print("a的shape",a.shape)


