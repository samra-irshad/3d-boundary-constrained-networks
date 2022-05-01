import torch.nn as nn
from functools import partial
import torch.nn.functional as F
import torch

class upsample_3d(nn.Module):
    def __init__(self):
        super(upsample_3d, self).__init__()
        self.upsample = partial(self._interpolate, mode='nearest')

    def forward(self, encoder_features, x):
        output_size = encoder_features.size()[2:]
        return self.upsample(x, output_size)

    @staticmethod
    def _interpolate(x, size, mode):
        return F.interpolate(x, size=size, mode=mode)


class DoubleConv_3d(nn.Module):
    """(convolution => [BN] => LReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class unet3d(nn.Module):
    def __init__(self, in_ch=1,out_ch=9):
        super(unet3d, self).__init__()

        filters = [16, 32, 64, 128,256]
        
        self.pool = nn.MaxPool3d(kernel_size=2)
        self.Up=upsample_3d()

        self.conv0_0 = DoubleConv_3d(in_ch, filters[0])
        self.conv1_0 = DoubleConv_3d(filters[0], filters[1])
        self.conv2_0 = DoubleConv_3d(filters[1], filters[2])
        self.conv3_0 = DoubleConv_3d(filters[2], filters[3])  
        self.conv4_0 = DoubleConv_3d(filters[3], filters[4])     
           
        self.conv0_1 = DoubleConv_3d(filters[0] + filters[1], filters[0])
        self.conv1_1 = DoubleConv_3d(filters[1] + filters[2], filters[1])
        self.conv2_1 = DoubleConv_3d(filters[2] + filters[3], filters[2])
        self.conv3_1 = DoubleConv_3d(filters[3] + filters[4], filters[3])
                   
        self.final = nn.Conv3d(filters[0], out_ch, kernel_size=1)
      
    def forward(self, x):
        
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.Up(x3_0,x4_0)], 1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.Up(x2_0,x3_1)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.Up(x1_0,x2_1)], 1))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.Up(x0_0,x1_1)], 1))
        output0_4 = self.final(x0_1)
        return output0_4


class unetplus3d(nn.Module):
    def __init__(self, in_ch=1,out_ch=9):
        super(unetplus3d, self).__init__()

        filters = [16, 32, 64,128,256]

        self.pool = nn.MaxPool3d(kernel_size=2)
        self.Up=upsample_3d()

        self.conv0_0 = DoubleConv_3d(in_ch, filters[0])
        self.conv1_0 = DoubleConv_3d(filters[0], filters[1])
        self.conv2_0 = DoubleConv_3d(filters[1], filters[2])
        self.conv3_0 = DoubleConv_3d(filters[2], filters[3])   
        self.conv4_0 = DoubleConv_3d(filters[3], filters[4])    
        
        self.conv0_1 = DoubleConv_3d(filters[0] + filters[1], filters[0])
        self.conv1_1 = DoubleConv_3d(filters[1] + filters[2], filters[1])
        self.conv2_1 = DoubleConv_3d(filters[2] + filters[3], filters[2])
        self.conv3_1 = DoubleConv_3d(filters[3] + filters[4], filters[3])

        self.conv0_2 = DoubleConv_3d(filters[0]*2 + filters[1], filters[0])
        self.conv1_2 = DoubleConv_3d(filters[1]*2 + filters[2], filters[1])
        self.conv2_2 = DoubleConv_3d(filters[2]*2 + filters[3], filters[2])
               
        self.conv0_3 = DoubleConv_3d(filters[0]*3 + filters[1], filters[0])
        self.conv1_3 = DoubleConv_3d(filters[1]*3 + filters[2], filters[1])
        self.conv0_4 = DoubleConv_3d(filters[0]*4 + filters[1], filters[0])

        self.final = nn.Conv3d(filters[0], out_ch, kernel_size=1)      
        
    def forward(self, x):
        
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.Up(x0_0,x1_0)], 1))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.Up(x1_0,x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.Up(x0_1,x1_1)], 1))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.Up(x2_0,x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.Up(x1_0,x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.Up(x0_0,x1_2)], 1))
        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.Up(x3_0,x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1,self.Up(x2_1,x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1,x1_2,self.Up(x1_2,x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1,x0_2,x0_3,self.Up(x0_3,x1_3)], 1))            
        output0_4 = self.final(x0_4)
        return output0_4

class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(conv_block, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, inputs):
        outputs = self.double_conv(inputs)
        return outputs

class gating_block(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size):
        super(gating_block, self).__init__()

        self.conv = nn.Sequential(nn.Conv3d(in_ch, out_ch, kernel_size, padding=0),
                                       nn.BatchNorm3d(out_ch),
                                       nn.ReLU(inplace=True),
                                       )
        
    def forward(self, inputs):
        outputs = self.conv(inputs)
        return outputs
class io_concatenation(nn.Module):
    def __init__(self):
        super(io_concatenation, self).__init__()
    def forward(self, x,g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        # theta => (b, c, t, h, w) -> (b, i_c, t, h, w) -> (b, i_c, thw)
        # phi   => (b, g_d) -> (b, i_c)
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        # g (b, c, t', h', w') -> phi_g (b, i_c, t', h', w')
        #  Relu(theta_x + phi_g + bias) -> f = (b, i_c, thw) -> (b, i_c, t/s1, h/s2, w/s3)
        phi_g = F.upsample(self.phi(g), size=theta_x_size[2:], mode=self.upsample_mode)
        f = F.relu(theta_x + phi_g, inplace=True)

        #  psi^T * f -> (b, psi_i_c, t/s1, h/s2, w/s3)
        sigm_psi_f = F.sigmoid(self.psi(f))

        # upsample the attentions and multiply
        sigm_psi_f = F.upsample(sigm_psi_f, size=input_size[2:], mode=self.upsample_mode)
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)

        return W_y, sigm_psi_f

class _GridAttentionBlockND(nn.Module):
    def __init__(self, in_channels, gating_channels, inter_channels=None, dimension=3, mode='concatenation',
                 sub_sample_factor=(2,2,2)):
        super(_GridAttentionBlockND, self).__init__()

        assert dimension in [2, 3]
        assert mode in ['concatenation', 'concatenation_debug', 'concatenation_residual']

        # Downsampling rate for the input featuremap
        if isinstance(sub_sample_factor, tuple): self.sub_sample_factor = sub_sample_factor
        elif isinstance(sub_sample_factor, list): self.sub_sample_factor = tuple(sub_sample_factor)
        else: self.sub_sample_factor = tuple([sub_sample_factor]) * dimension

        # Default parameter set
        self.mode = mode
        self.dimension = dimension
        
        self.sub_sample_kernel_size = self.sub_sample_factor

        # Number of channels (pixel dimensions)
        self.in_channels = in_channels
        self.gating_channels = gating_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            bn = nn.BatchNorm3d
            self.upsample_mode = 'trilinear'
        elif dimension == 2:
            conv_nd = nn.Conv2d
            bn = nn.BatchNorm2d
            self.upsample_mode = 'bilinear'
        else:
            raise NotImplemented

        # Output transform
        self.W = nn.Sequential(
            conv_nd(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0),
            bn(self.in_channels),
        )

        # Theta^T * x_ij + Phi^T * gating_signal + bias
        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=self.sub_sample_kernel_size, stride=self.sub_sample_factor, padding=0, bias=False)
        self.phi = conv_nd(in_channels=self.gating_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0, bias=True)
        self.psi = conv_nd(in_channels=self.inter_channels, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)
  
        # Define the operation
        if mode == 'concatenation':
            self.operation_function = io_concatenation()
        elif mode == 'concatenation_debug':
            self.operation_function = self._concatenation_debug
        elif mode == 'concatenation_residual':
            self.operation_function = self._concatenation_residual
        else:
            raise NotImplementedError('Unknown operation function.')


    def forward(self, x, g):
        '''
        :param x: (b, c, t, h, w)
        :param g: (b, g_d)
        :return:
        '''
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)
        #print('x',x.shape)
        #print('g',g.shape)


        # theta => (b, c, t, h, w) -> (b, i_c, t, h, w) -> (b, i_c, thw)
        # phi   => (b, g_d) -> (b, i_c)
        theta_x = self.theta(x)
       
        theta_x_size = theta_x.size()

        # g (b, c, t', h', w') -> phi_g (b, i_c, t', h', w')
        #  Relu(theta_x + phi_g + bias) -> f = (b, i_c, thw) -> (b, i_c, t/s1, h/s2, w/s3)
        phi_g = F.upsample(self.phi(g), size=theta_x_size[2:], mode=self.upsample_mode)
        
        f = F.relu(theta_x + phi_g, inplace=True)
        #print('phi_g', phi_g.shape)
        #print('f', f.shape)

        #  psi^T * f -> (b, psi_i_c, t/s1, h/s2, w/s3)
        sigm_psi_f = F.sigmoid(self.psi(f))
        #print('sigm_psi_f', sigm_psi_f.shape)

        # upsample the attentions and multiply
        sigm_psi_f = F.upsample(sigm_psi_f, size=input_size[2:], mode=self.upsample_mode)
        #print('sigm_psi_f', sigm_psi_f.shape)

        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)
        #print('W_y', W_y.shape)
        return W_y, sigm_psi_f

    def _concatenation(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)
        

        # theta => (b, c, t, h, w) -> (b, i_c, t, h, w) -> (b, i_c, thw)
        # phi   => (b, g_d) -> (b, i_c)
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        # g (b, c, t', h', w') -> phi_g (b, i_c, t', h', w')
        #  Relu(theta_x + phi_g + bias) -> f = (b, i_c, thw) -> (b, i_c, t/s1, h/s2, w/s3)
        phi_g = F.upsample(self.phi(g), size=theta_x_size[2:], mode=self.upsample_mode)
        f = F.relu(theta_x + phi_g, inplace=True)

        #  psi^T * f -> (b, psi_i_c, t/s1, h/s2, w/s3)
        sigm_psi_f = F.sigmoid(self.psi(f))

        # upsample the attentions and multiply
        sigm_psi_f = F.upsample(sigm_psi_f, size=input_size[2:], mode=self.upsample_mode)
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)

        return W_y, sigm_psi_f

    def _concatenation_debug(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        # theta => (b, c, t, h, w) -> (b, i_c, t, h, w) -> (b, i_c, thw)
        # phi   => (b, g_d) -> (b, i_c)
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        # g (b, c, t', h', w') -> phi_g (b, i_c, t', h', w')
        #  Relu(theta_x + phi_g + bias) -> f = (b, i_c, thw) -> (b, i_c, t/s1, h/s2, w/s3)
        phi_g = F.upsample(self.phi(g), size=theta_x_size[2:], mode=self.upsample_mode)
        f = F.softplus(theta_x + phi_g)

        #  psi^T * f -> (b, psi_i_c, t/s1, h/s2, w/s3)
        sigm_psi_f = F.sigmoid(self.psi(f))

        # upsample the attentions and multiply
        sigm_psi_f = F.upsample(sigm_psi_f, size=input_size[2:], mode=self.upsample_mode)
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)

        return W_y, sigm_psi_f


    def _concatenation_residual(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        # theta => (b, c, t, h, w) -> (b, i_c, t, h, w) -> (b, i_c, thw)
        # phi   => (b, g_d) -> (b, i_c)
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        # g (b, c, t', h', w') -> phi_g (b, i_c, t', h', w')
        #  Relu(theta_x + phi_g + bias) -> f = (b, i_c, thw) -> (b, i_c, t/s1, h/s2, w/s3)
        phi_g = F.upsample(self.phi(g), size=theta_x_size[2:], mode=self.upsample_mode)
        f = F.relu(theta_x + phi_g, inplace=True)

        #  psi^T * f -> (b, psi_i_c, t/s1, h/s2, w/s3)
        f = self.psi(f).view(batch_size, 1, -1)
        sigm_psi_f = F.softmax(f, dim=2).view(batch_size, 1, *theta_x.size()[2:])

        # upsample the attentions and multiply
        sigm_psi_f = F.upsample(sigm_psi_f, size=input_size[2:], mode=self.upsample_mode)
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)

        return W_y, sigm_psi_f


class GridAttentionBlock3D(_GridAttentionBlockND):
    def __init__(self, in_channels, gating_channels, inter_channels=None, mode='concatenation',
                 sub_sample_factor=(2,2,2)):
        super(GridAttentionBlock3D, self).__init__(in_channels,
                                                   inter_channels=inter_channels,
                                                   gating_channels=gating_channels,
                                                   dimension=3, mode=mode,
                                                   sub_sample_factor=sub_sample_factor,
                                                   )

class atten_block(nn.Module):
    def __init__(self, in_size, gate_size, inter_size, nonlocal_mode, sub_sample_factor):
        super(atten_block, self).__init__()
        self.gate_block_1 = GridAttentionBlock3D(in_channels=in_size, gating_channels=gate_size,
                                                 inter_channels=inter_size, mode=nonlocal_mode,
                                                 sub_sample_factor= sub_sample_factor)
        self.combine_gates = nn.Sequential(nn.Conv3d(in_size*2, in_size, kernel_size=1, stride=1, padding=0),
                                           nn.BatchNorm3d(in_size),
                                           nn.ReLU(inplace=True)
                                           )

    def forward(self, input, gating_signal):
        gate_1, attention_1 = self.gate_block_1(input, gating_signal)        
       
        return gate_1, attention_1


class attenunet_up(nn.Module):
    def __init__(self, in_size, out_size, pad):
        super(attenunet_up, self).__init__()
        self.conv = conv_block(in_size + out_size, out_size)
        self.up = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear')
        self.pad=pad

       
    def forward(self, inputs1, inputs2):
        
        outputs2 = self.up(inputs2)
        outputs2 = F.pad(outputs2, self.pad)
        return self.conv(torch.cat([inputs1, outputs2], 1))


class attenunet3d(nn.Module):

    def __init__(self, out_ch=9, in_ch=1,
                 nonlocal_mode='concatenation', attention_dsample=(2,2,2)):
        super(attenunet3d, self).__init__()
        filters = [16,32,64,128,256]
        self.pool = nn.MaxPool3d(kernel_size=2)
        
        self.conv1 = conv_block(in_ch, filters[0])
        self.conv2 = conv_block(filters[0], filters[1])
        self.conv3 = conv_block(filters[1], filters[2])
        self.conv4 = conv_block(filters[2], filters[3])
        self.conv5 = conv_block(filters[3], filters[4])
        self.gating = gating_block(filters[4], filters[4], 1)

       
        self.attentionblock2 = atten_block(in_size=filters[1], gate_size=filters[2], inter_size=filters[1],
                                                   nonlocal_mode=nonlocal_mode, sub_sample_factor= attention_dsample)
        self.attentionblock3 = atten_block(in_size=filters[2], gate_size=filters[3], inter_size=filters[2],
                                                   nonlocal_mode=nonlocal_mode, sub_sample_factor= attention_dsample)
        self.attentionblock4 = atten_block(in_size=filters[3], gate_size=filters[4], inter_size=filters[3],
                                                   nonlocal_mode=nonlocal_mode, sub_sample_factor= attention_dsample)

        # upsampling
        self.up_concat4 = attenunet_up(filters[4], filters[3], [0,0,0,0,0,0])
        self.up_concat3 = attenunet_up(filters[3], filters[2], [0,0,0,0,0,0])
        self.up_concat2 = attenunet_up(filters[2], filters[1], [0,0,0,0,0,0])
        self.up_concat1 = attenunet_up(filters[1], filters[0], [0,0,0,0,0,0])

        self.final = nn.Conv3d(filters[0], out_ch, 1)
        
       
    def forward(self, inputs):
        # Feature Extraction
        conv1 = self.conv1(inputs)
        conv1_p = self.pool(conv1)
        conv2 = self.conv2(conv1_p)
        conv2_p = self.pool(conv2)

        conv3 = self.conv3(conv2_p)
        conv3_p = self.pool(conv3)

        conv4 = self.conv4(conv3_p)
        conv4_p = self.pool(conv4)

        # Gating Signal Generation
        center = self.conv5(conv4_p)
        gating = self.gating(center)

        # Attention Mechanism
        # Upscaling Part (Decoder)
        g_conv4, att4 = self.attentionblock4(conv4, gating)
        up4 = self.up_concat4(g_conv4, center)
        g_conv3, att3 = self.attentionblock3(conv3, up4)
        up3 = self.up_concat3(g_conv3, up4)
        g_conv2, att2 = self.attentionblock2(conv2, up3)
        up2 = self.up_concat2(g_conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        final=self.final(up1)
        return final
