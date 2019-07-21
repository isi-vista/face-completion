import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm

def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, mean=1, std=0.02)
            nn.init.constant_(m.bias.data, 0)
            
    return init_fun

class PatchDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # receptive fild size of ~50pixels
        self.model = nn.Sequential(nn.Conv2d(3, 64, 5, 2, 0),
                                   nn.LeakyReLU(inplace=True),
                                   nn.Conv2d(64, 128, 5, 2, 0),
                                   nn.LeakyReLU(inplace=True),
                                   nn.InstanceNorm2d(128),
                                   # start ---dilation----
                                   nn.ReflectionPad2d(4),
                                   nn.Conv2d(128, 128, 5, 1, 0, dilation=4),
                                   nn.LeakyReLU(inplace=True),
                                   nn.InstanceNorm2d(128),
                                   nn.Conv2d(128, 128, 3, 1, 0, dilation=4),
                                   nn.LeakyReLU(inplace=True),
                                   nn.InstanceNorm2d(128),
                                   nn.Conv2d(128, 128, 3, 1, 0, dilation=2),
                                   nn.LeakyReLU(inplace=True),
                                   nn.InstanceNorm2d(128),
                                   # end ---dilation----
                                   nn.Conv2d(128, 256, 3, 2, 0),
                                   nn.LeakyReLU(inplace=True),
                                   nn.Conv2d(256, 256, 3, 1, 0),
                                   nn.Sigmoid())
        self.model.apply(weights_init('xavier'))

    @staticmethod
    def get_patches(batch_size, img_size):
        if img_size == 128:
            valid = torch.ones([batch_size, 256, 2, 2], requires_grad=False)
            fake = torch.zeros([batch_size, 256, 2, 2], requires_grad=False)
        elif img_size == 224:
            valid = torch.ones([batch_size, 256, 14, 14], requires_grad=False)
            fake = torch.zeros([batch_size, 256, 14, 14], requires_grad=False)
        else:
            raise NotImplementedError()
        return valid, fake

    def forward(self, img):
        return self.model(img)


if __name__ == "__main__":
    img1 = torch.randn(1, 3, 224, 224)
    img2 = torch.randn(1, 3, 128, 128)
    print("PatchDiscriminator")
    model = PatchDiscriminator()
    with torch.no_grad():
        output1 = model(img1)
        output2 = model(img2)
    print('output-224', output1.size())
    print('output-128', output2.size())
        

            
            
            
