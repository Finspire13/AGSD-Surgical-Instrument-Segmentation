import torch
import torch.nn as nn
import torchvision.models as models


def get_seg_block(scale_factor, in_channels, out_channels, transposed_conv, align_corners):
    
    module_list = []
    
    if scale_factor is not None:
        
        if align_corners:
            module_list.append(
                nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
            )
        else:
            module_list.append(
                nn.Upsample(scale_factor=scale_factor)
            )
    
    if transposed_conv:
        module_list.append(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, padding=1)
        )
        module_list.append(
            nn.ReLU(inplace=True)
        )
        module_list.append(
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, padding=1)
        )
        module_list.append(
            nn.ReLU(inplace=True)
        )
    else:
        module_list.append(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )
        module_list.append(
            nn.ReLU(inplace=True)
        )
        module_list.append(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )
        module_list.append(
            nn.ReLU(inplace=True)
        )        
    
    return nn.Sequential(*module_list)
        

class MyUNet(nn.Module):

    def __init__(self, transposed_conv=True, align_corners=False):
        super(MyUNet, self).__init__()
        
        self.backbone = list(models.resnet18(pretrained=True).children())
        self.backbone_bottom = nn.Sequential(*self.backbone[:4])
        self.backbone_layer1 = self.backbone[4]
        self.backbone_layer2 = self.backbone[5]
        self.backbone_layer3 = self.backbone[6]
        self.backbone_layer4 = self.backbone[7]

        self.scale_factor = 32  # 4x1x2x2x2

        self.seg_block1 = get_seg_block(2, 512, 64, transposed_conv, align_corners)
        self.seg_block2 = get_seg_block(2, 320, 32, transposed_conv, align_corners)
        self.seg_block3 = get_seg_block(2, 160, 16, transposed_conv, align_corners)
        self.seg_block4 = get_seg_block(None, 80, 16, transposed_conv, align_corners)
        self.seg_block5 = get_seg_block(4, 80, 8, transposed_conv, align_corners)

        self.seg_head = nn.Linear(11, 1)

    def forward(self, x_0):

        assert(x_0.shape[2] % self.scale_factor == 0)
        assert(x_0.shape[3] % self.scale_factor == 0)

        x_1 = self.backbone_bottom(x_0)
        x_2 = self.backbone_layer1(x_1)
        x_3 = self.backbone_layer2(x_2)
        x_4 = self.backbone_layer3(x_3)
        x_5 = self.backbone_layer4(x_4)

        x_4_u = self.seg_block1(x_5) 
        x_4_c = torch.cat((x_4_u, x_4), 1) 

        x_3_u = self.seg_block2(x_4_c)  
        x_3_c = torch.cat((x_3_u, x_3), 1)

        x_2_u = self.seg_block3(x_3_c)  
        x_2_c = torch.cat((x_2_u, x_2), 1) 

        x_1_u = self.seg_block4(x_2_c) 
        x_1_c = torch.cat((x_1_u, x_1), 1)

        x_0_u = self.seg_block5(x_1_c)  
        x_0_c = torch.cat((x_0_u, x_0), 1)

        x_seg = x_0_c.permute(0, 2, 3, 1)
        x_seg = self.seg_head(x_seg)  

        return x_seg
