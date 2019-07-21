import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class WGANDiscriminator(nn.Module):
    def __init__(self, size=224):
        super(WGANDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.InstanceNorm2d(64),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.LeakyReLU(inplace=True),
            nn.InstanceNorm2d(128),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.InstanceNorm2d(128),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.LeakyReLU(inplace=True),
            nn.InstanceNorm2d(256),
            nn.Conv2d(256, 512, 3, 2, 1),
            nn.LeakyReLU(inplace=True))
        self.out_dim = size//16 * size//16 * 512
        self.linear = nn.Linear(self.out_dim, 1)

    def forward(self, input):
        output = self.model(input)
        output = output.view(-1, self.out_dim)
        return self.linear(output)


def calc_gradient_penalty(netD, real, fake, LAMBDA=10):
    batch_size, ch, h, w = real.size()
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, real.nelement()//batch_size).contiguous().view(batch_size, ch, h, w)
    alpha = alpha.cuda()

    interpolates = alpha * real.detach() + ((1 - alpha) * fake.detach())
    interpolates = interpolates.cuda()
    interpolates.requires_grad_(True)
    
    disc_interpolates = netD(interpolates)
    gradients = torch.autograd.grad(outputs=disc_interpolates,
                                    inputs=interpolates,
                                    grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                                    create_graph=True,
                                    retain_graph=True,
                                    only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) -1) ** 2).mean() * LAMBDA
    return gradient_penalty
    
