import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

# from model.resnet import ResNet18
from model.resnet import ResNet18_all as ResNet18
from utils import hadamard_product, feature_crop
'''
Implement with 3D Hourglass Network
'''

def weight_init_xavier_uniform(submodule):
    if isinstance(submodule, torch.nn.Conv3d):
        torch.nn.init.xavier_normal_(submodule.weight.data)
    elif isinstance(submodule, torch.nn.BatchNorm3d):
        submodule.weight.data.fill_(1.0)
        submodule.bias.data.zero_()

class Residual3D(nn.Module):
    def __init__(self, use_bn, input_channels, out_channels, mid_channels):
        super(Residual3D, self).__init__()
        self.use_bn = use_bn
        self.out_channels   = out_channels
        self.input_channels = input_channels
        self.mid_channels   = mid_channels

        self.down_channel = nn.Conv3d(input_channels, self.mid_channels, kernel_size = 1)
        self.AcFunc       = nn.ReLU()
        if use_bn:
            self.bn_0 = nn.BatchNorm3d(num_features = self.mid_channels)
            self.bn_1 = nn.BatchNorm3d(num_features = self.mid_channels)
            self.bn_2 = nn.BatchNorm3d(num_features = self.out_channels)

        self.conv = nn.Conv3d(self.mid_channels, self.mid_channels, kernel_size = 3, padding = 1)

        self.up_channel = nn.Conv3d(self.mid_channels, out_channels, kernel_size= 1)

        if input_channels != out_channels:
            self.trans = nn.Conv3d(input_channels, out_channels, kernel_size = 1)
    
        for m in self.modules():
            weight_init_xavier_uniform(m)


    def forward(self, inputs):
        x = self.down_channel(inputs)
        if self.use_bn:
            x = self.bn_0(x)
        x = self.AcFunc(x)

        x = self.conv(x)
        if self.use_bn:
            x = self.bn_1(x)
        x = self.AcFunc(x)

        x = self.up_channel(x)

        if self.input_channels != self.out_channels:
            x += self.trans(inputs)
        else:
            x += inputs

        if self.use_bn:
            x = self.bn_2(x)
        
        return self.AcFunc(x)

class HourGlassBlock3D(nn.Module):
    def __init__(self, block_count, residual_each_block, input_channels, mid_channels, use_bn, stack_index):
        super(HourGlassBlock3D, self).__init__()

        self.block_count         = block_count
        self.residual_each_block = residual_each_block
        self.use_bn              = use_bn
        self.stack_index         = stack_index
        self.input_channels      = input_channels
        self.mid_channels        = mid_channels

        if self.block_count == 0: #inner block
            self.process = nn.Sequential()
            for _ in range(residual_each_block * 3):
                self.process.add_module(
                    name = 'inner_{}_{}'.format(self.stack_index, _),
                    module = Residual3D(input_channels = input_channels, out_channels = input_channels, mid_channels = mid_channels, use_bn = use_bn)
                )
        else:
            #down sampling
            self.down_sampling = nn.Sequential()
            self.down_sampling.add_module(
                name = 'down_sample_{}_{}'.format(self.stack_index, self.block_count), 
                module = nn.MaxPool3d(kernel_size = 2, stride = 2)
            )
            for _ in range(residual_each_block):
                self.down_sampling.add_module(
                    name = 'residual_{}_{}_{}'.format(self.stack_index, self.block_count, _),
                    module = Residual3D(input_channels = input_channels, out_channels = input_channels, mid_channels = mid_channels, use_bn = use_bn)
                )
            
            #up sampling
            self.up_sampling = nn.Sequential()
            self.up_sampling.add_module(
                name = 'up_sample_{}_{}'.format(self.stack_index, self.block_count),
                # module = nn.Upsample(scale_factor=2, mode='bilinear')
                module = nn.Upsample(scale_factor=2, mode='nearest')
                # module = nn.Upsample(scale_factor=2, mode='trilinear')
            )
            for _ in range(residual_each_block):
                self.up_sampling.add_module(
                    name   = 'residual_{}_{}_{}'.format(self.stack_index, self.block_count, _),
                    module = Residual3D(input_channels = input_channels, out_channels = input_channels, mid_channels = mid_channels, use_bn = use_bn)
                )

            #sub hour glass
            self.sub_hg = HourGlassBlock3D(
                block_count         = self.block_count - 1, 
                residual_each_block = self.residual_each_block, 
                input_channels      = self.input_channels,
                mid_channels        = self.mid_channels,
                use_bn              = self.use_bn,
                stack_index         = self.stack_index
            )
            
            # trans
            self.trans = nn.Sequential()
            for _ in range(residual_each_block):
                self.trans.add_module(
                    name = 'trans_{}_{}_{}'.format(self.stack_index, self.block_count, _),
                    module = Residual3D(input_channels = input_channels, out_channels = input_channels, mid_channels = mid_channels, use_bn = use_bn)
            )

        for m in self.modules():
            weight_init_xavier_uniform(m)


    def forward(self, inputs):
        if self.block_count == 0:
            return self.process(inputs)
        else:
            down_sampled        = self.down_sampling(inputs)
            transed             = self.trans(down_sampled)
            sub_net_output      = self.sub_hg(down_sampled)
            res                 = self.up_sampling(transed + sub_net_output)

            return res

class ContBatchNorm3d(nn.modules.batchnorm._BatchNorm):
    def forward(self, input):
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

class EDUpTransition(nn.Module):
    def __init__(self, inChans, num_classes=16, last_layer=True):
        super(EDUpTransition, self).__init__()

        self.upsample = nn.Upsample(scale_factor=2)

        self.conv = nn.Conv3d(inChans, inChans, kernel_size=3, padding=1)
        self.bn = ContBatchNorm3d(inChans)
        self.relu = nn.ReLU()
        self.last_conv = nn.Conv3d(in_channels=inChans, out_channels=num_classes, kernel_size=1)

        for m in self.modules():
            weight_init_xavier_uniform(m)

    def forward(self, x):
        # print('decoder input : ', x.shape)
        out = self.upsample(x)
        out = self.conv(out)
        out = self.bn(out)
        out = self.relu(out)
        out64 = out

        out = self.last_conv(out)
        # out = F.sigmoid(out)
        # out = torch.softmax(out, dim=1)

        return out64, out
    
class NewUpTransition(nn.Module):
    def __init__(self, inChans):
        super(NewUpTransition, self).__init__()

        self.upsample = nn.ConvTranspose3d(inChans, inChans, 4, 2, 1)
        # self.upsample = nn.ConvTranspose3d(inChans, inChans, 3, 2, 1)
        # self.upsample = nn.ConvTranspose3d(inChans, inChans, 2, 2, 0)
        self.bn = ContBatchNorm3d(inChans)
        # self.bn = nn.BatchNorm3d(inChans)
        self.relu = nn.ReLU()

        for m in self.modules():
            weight_init_xavier_uniform(m)

    def forward(self, x):
        # print('decoder input : ', x.shape)
        out = self.upsample(x)
        out = self.bn(out)
        out = self.relu(out)

        return out

class HourGlass3D(nn.Module):
    def __init__(self, nStack, nBlockCount, nResidualEachBlock, nMidChannels, nChannels, nJointCount, bUseBn):
        super(HourGlass3D, self).__init__()

        self.nStack             = nStack
        self.nBlockCount        = nBlockCount
        self.nResidualEachBlock = nResidualEachBlock
        self.nChannels          = nChannels
        self.nMidChannels       = nMidChannels
        self.nJointCount        = nJointCount
        self.bUseBn             = bUseBn

        self.pre_process = nn.Sequential(
            nn.Conv3d(1, nChannels, kernel_size = 3, padding = 1),
            Residual3D(use_bn = bUseBn, input_channels = nChannels, out_channels = nChannels, mid_channels = nMidChannels),
            nn.MaxPool3d(kernel_size = 2, stride = 2),
        )

        self.hg = nn.ModuleList()
        for _ in range(nStack):
            self.hg.append(
                HourGlassBlock3D(
                    block_count = nBlockCount, 
                    residual_each_block = nResidualEachBlock,
                    input_channels = nChannels, 
                    mid_channels = nMidChannels, 
                    use_bn = bUseBn,
                    stack_index = _
                )
            )

        self.blocks = nn.ModuleList()
        for _ in range(nStack - 1):
            self.blocks.append(
                nn.Sequential(
                    Residual3D(
                        use_bn = bUseBn, input_channels = nChannels, out_channels = nChannels, mid_channels = nMidChannels 
                    ),
                    Residual3D(
                        use_bn = bUseBn, input_channels = nChannels, out_channels = nChannels, mid_channels = nMidChannels
                    )
                )
            )
        
        self.intermediate_supervision = nn.ModuleList()
        for _ in range(nStack): # to 64 x 64 x joint_count
            self.intermediate_supervision.append(
                 nn.Sequential(
                    nn.Conv3d(nChannels, nJointCount, kernel_size = 1, stride = 1),
                    # nn.ReLU()
                 )
            )

        self.normal_feature_channel = nn.ModuleList()
        for _ in range(nStack - 1):
            self.normal_feature_channel.append(
                Residual3D(
                    use_bn = bUseBn, input_channels = nJointCount, out_channels = nChannels, mid_channels = nMidChannels
                )
            )
        self.upsample = NewUpTransition(128)
        self.inter_hm_head = HeatmapHead()
        self.hm_head = HeatmapHead()
        self.box_head = OffsetHead()

        self.linear_upsample = nn.Upsample(scale_factor=2)

        self.score_head = ResNet18(inchannels=129, num_classes=3)

        for m in self.modules():
            weight_init_xavier_uniform(m)

    def forward(self, inputs):
        o = [] #outputs include intermediate supervision result
        x = self.pre_process(inputs)

        # print('pre_process input : ', inputs.shape, ' -> output : ', x.shape)
        for _ in range(self.nStack):
            '''
            hg output :  torch.Size([1, 128, 64, 64, 64])
            intermediate_supervision output :  torch.Size([1, 16, 64, 64, 64])
            '''
            o1 = self.hg[_](x)
            o2 = self.intermediate_supervision[_](o1)

            if _ == self.nStack - 2:
                out = self.inter_hm_head(o2)
                out = out.sigmoid_()
                o.append(out)


            if _ == self.nStack - 1:
                # out64 = self.upsample(o1)
                out64 = self.upsample(o2)
                out = self.hm_head(out64)
                out = out.sigmoid_()
                o.append(out)

                offset = self.box_head(out64)
                o.append(offset)

                pred_hadamard_center = hadamard_product(out.squeeze(0))
                o.append(pred_hadamard_center)

                linear_upsample_feat = self.linear_upsample(o2)
                concat_feature = torch.cat([inputs, linear_upsample_feat], dim=1)
                crop_features = feature_crop(concat_feature, pred_hadamard_center)
                score_output = self.score_head(crop_features)
                o.append(score_output)

                break

            o2 = self.normal_feature_channel[_](o2)
            o1 = self.blocks[_](o1)
            x = o1 + o2 + x
            # x = o1 + x

        return o

class HeatmapHead(nn.Module):
    def __init__(self):
        super(HeatmapHead, self).__init__()
        
        cnv_dim = 128
        out_dim = 16

        self.hm_module = nn.Sequential(
            nn.Conv3d(cnv_dim, cnv_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(cnv_dim, out_dim, (1, 1, 1))
        )

        for m in self.modules():
            weight_init_xavier_uniform(m)

    def forward(self, feat):
        hm = self.hm_module(feat)
        return hm

class OffsetHead(nn.Module):
    def __init__(self):
        super(OffsetHead, self).__init__()
        
        cnv_dim = 128

        self.offset_module = nn.Sequential(
            nn.Conv3d(cnv_dim, cnv_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(cnv_dim, 3, (1, 1, 1))
        )
        
        k = 8
        kernel = (k, k ,k)
        s = 8
        p = 0
        stride = (s ,s, s)
        self.avgpool = nn.AvgPool3d(kernel_size=kernel, stride=stride, padding=p)

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 16)
        self.relu = nn.ReLU()
        
        for m in self.modules():
            weight_init_xavier_uniform(m)

    def forward(self, feat):
        x = self.offset_module(feat)

        x = self.avgpool(x)

        x = rearrange(x, 'b c d h w -> b c (d h w)')

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)

        x = rearrange(x, 'b d c -> b c d')
        
        return x
    

# if __name__ == '__main__':
#     net = HourGlass3D(
#         nStack = 2,
#         nBlockCount = 4,
#         nResidualEachBlock = 1,
#         nMidChannels = 128,
#         nChannels = 128,
#         nJointCount = 128,
#         bUseBn = True,
#     ).cuda()

#     y = net(torch.randn(1, 1, 64, 128, 128).cuda())
    
#     for yy in y:
#         print('yy : ', yy.shape)