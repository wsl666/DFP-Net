import torch
import torch.nn as nn
import functools
import torch.nn.functional as F
import numpy as np


class Discriminator_global(nn.Module):
    """Defines a global discriminator """

    def __init__(self, input_nc=3):
        super(Discriminator_global, self).__init__()

        # A bunch of convolutions one after another
        model = [nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                 nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(64, 128, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(128),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(128, 256, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(256),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(256, 512, 4, padding=1),
                  nn.InstanceNorm2d(512),
                  nn.LeakyReLU(0.2, inplace=True)]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

        self.fc = nn.Linear(900, 1)

    def forward(self, x):
        x = self.model(x)
        # flatten
        x=x.view(x.size()[0], -1)

        x=self.fc(x)

        return x



class Discriminator_local(nn.Module):
    """Defines a local discriminator ,we  use  patchgan as local discriminator"""

    def __init__(self, input_nc=3, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """

        super(Discriminator_local, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, padding=1, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class Discriminator_pixel(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc=3, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(Discriminator_pixel, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)


# Defines the Multiscale-PatchGAN discriminator with the specified arguments.
class MultiDiscriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=64, n_layers=5, norm_layer=nn.InstanceNorm2d, use_sigmoid=False):
        super(MultiDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # cannot deal with use_sigmoid=True case at thie moment
        assert (use_sigmoid == False)

        kw = 4
        padw = int(np.ceil((kw - 1) / 2))
        scale1 = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, 3):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            scale1 += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        self.scale1 = nn.Sequential(*scale1)
        scale1_output = []
        scale1_output += [
            nn.Conv2d(ndf * nf_mult, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        scale1_output += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # compress to 1 channel
        self.scale1_output = nn.Sequential(*scale1_output)

        scale2 = []
        nf_mult = nf_mult
        for n in range(3, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            scale2 += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        scale2 += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        scale2 += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            scale2 += [nn.Sigmoid()]

        self.scale2 = nn.Sequential(*scale2)

    def forward(self, input):

        scale1 = self.scale1(input)
        output1 = self.scale1_output(scale1)
        output2 = self.scale2(scale1)

        return output1, output2



# Defines the Multiscale-PatchGAN discriminator with the specified arguments.带谱归一化的patchgan
class MultiDiscriminator_SP(nn.Module):
    def __init__(self, input_nc=3, ndf=64, n_layers=5, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(MultiDiscriminator_SP, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # cannot deal with use_sigmoid=True case at thie moment
        assert (use_sigmoid == False)

        kw = 4
        padw = int(np.ceil((kw - 1) / 2))
        scale1 = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, 3):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            scale1 += [
                nn.utils.spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias)),
                nn.LeakyReLU(0.2, True)
            ]

        self.scale1 = nn.Sequential(*scale1)
        scale1_output = []
        scale1_output += [
            nn.utils.spectral_norm(nn.Conv2d(ndf * nf_mult, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias)),
            nn.LeakyReLU(0.2, True)
        ]
        scale1_output += [nn.utils.spectral_norm(nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw))]  # compress to 1 channel
        self.scale1_output = nn.Sequential(*scale1_output)

        scale2 = []
        nf_mult = nf_mult
        for n in range(3, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            scale2 += [
                nn.utils.spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias)),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        scale2 += [
            nn.utils.spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias)),
            nn.LeakyReLU(0.2, True)
        ]

        scale2 += [nn.utils.spectral_norm(nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw))]

        if use_sigmoid:
            scale2 += [nn.Sigmoid()]

        self.scale2 = nn.Sequential(*scale2)

    def forward(self, input):

        scale1 = self.scale1(input)
        output1 = self.scale1_output(scale1)
        output2 = self.scale2(scale1)

        return output1, output2

if __name__ =="__main__":
    x=torch.ones(1,3,576,576)
    D=MultiDiscriminator_SP()
    print(D)
    a,b=D(x)
    print(a.shape)
    print(b.shape)
