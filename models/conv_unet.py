import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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

class Conv(nn.Module):
    """ Soft gating for the input """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, normalize=True):
        super().__init__()
        self.normalize = normalize
        self.relu = nn.ELU(inplace=True)
        # feature branch
        self.feature_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                      stride, padding, dilation, groups, bias)
        self.feature_conv.apply(weights_init('xavier'))
        if self.normalize:
            self.feature_norm = nn.InstanceNorm2d(out_channels)

    def forward(self, inputfeat):
        feats = self.feature_conv(inputfeat)
        out = self.relu(feats)
        # apply normalization after non-linearity
        if self.normalize:
            out = self.feature_norm(out)
        return out
    
class ConvUnet(nn.Module):
    def __init__(self):
        super().__init__()
        # encoder blocks 
        self.enc_0 = nn.Sequential(nn.ReflectionPad2d(1),
                                   Conv(4, 64, 3, 1, 0, normalize=False))
        self.enc_1 = nn.Sequential(nn.ReflectionPad2d(1),
                                   Conv(64, 128, 3, 2, 0))
        self.enc_2 = nn.Sequential(nn.ReflectionPad2d(1),
                                   Conv(128, 128, 3, 1, 0))
        self.enc_3 = nn.Sequential(nn.ReflectionPad2d(1),
                                   Conv(128, 128, 3, 1, 0))
        self.enc_4 = nn.Sequential(nn.ReflectionPad2d(1),
                                   Conv(128, 256, 3, 2, 0))
        # dilation blocks 
        self.dil_0 = nn.Sequential(nn.ReflectionPad2d(2),
                                   Conv(256, 256, 3, 1, 0, dilation=2))
        self.dil_1 = nn.Sequential(nn.ReflectionPad2d(2),
                                   Conv(256, 256, 3, 1, 0, dilation=2))
        self.dil_2 = nn.Sequential(nn.ReflectionPad2d(2),
                                   Conv(256, 256, 3, 1, 0, dilation=2))
        self.dil_3 = nn.Sequential(nn.ReflectionPad2d(2),
                                   Conv(256, 256, 3, 1, 0, dilation=2))
        # decoder blocks
        self.dec_5 = nn.Sequential(nn.ReflectionPad2d(1),
                                   Conv(256, 256, 3, 1, 0))
        self.dec_4 = nn.Sequential(nn.ReflectionPad2d(1),
                                   Conv(256, 256, 3, 1, 0))
        self.dec_3 = nn.Sequential(nn.ReflectionPad2d(1),
                                   Conv(256, 128, 3, 1, 0))
        self.dec_2 = nn.Sequential(nn.ReflectionPad2d(1),
                                   Conv(128, 128, 3, 1, 0))
        self.dec_1 = nn.Sequential(nn.ReflectionPad2d(1),
                                   Conv(128, 64, 3, 1, 0))
        self.dec_0 = nn.Sequential(nn.ReflectionPad2d(1),
                                   Conv(64, 32, 3, 1, 0),
                                   nn.ReflectionPad2d(1),
                                   Conv(32, 32, 3, 1, 0, normalize=False))
        self.post_dec = nn.Sequential(nn.Conv2d(32, 3, 1, 1, 0),
                                      nn.Tanh())

    def forward(self, occl_img, mask):
        # remove pixel information in the masked area.
        feat_cat = torch.cat([occl_img * mask, mask], 1)

        # encoder block
        feat_cat = self.enc_0(feat_cat)
        feat_cat = self.enc_1(feat_cat)
        feat_cat = self.enc_2(feat_cat)
        feat_cat = self.enc_3(feat_cat)
        feat_cat = self.enc_4(feat_cat)
        
        # dilation block
        feat_cat = self.dil_0(feat_cat)
        feat_cat = self.dil_1(feat_cat)
        feat_cat = self.dil_2(feat_cat)
        feat_cat = self.dil_3(feat_cat)
        
        feat_cat = self.dec_5(feat_cat)
        feat_cat = self.dec_4(feat_cat)
        feat_cat = F.upsample(feat_cat, scale_factor=2)
        feat_cat = self.dec_3(feat_cat)
        feat_cat = self.dec_2(feat_cat)
        feat_cat = F.upsample(feat_cat, scale_factor=2)
        feat_cat = self.dec_1(feat_cat)
        feat_cat = self.dec_0(feat_cat)
        feat_cat = self.post_dec(feat_cat)
        return feat_cat
    

if __name__ == '__main__':
    print('testing gated conv...')
    model = Conv(3, 64, 3, 1, 1)
    img = torch.randn(1, 3, 224, 224)
    mask = torch.randn(1, 1, 224, 224)
    with torch.no_grad():
        output = model(img)
    print(output.size())
    print('testing unet arch...')
    model = ConvUnet()
    with torch.no_grad():
        output = model(img, mask)
    print(output.size())
