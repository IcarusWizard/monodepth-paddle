import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DispHead(nn.Layer):
    def __init__(self, feature_dim):
        super().__init__()
        self.conv = nn.Conv2D(feature_dim, 2, 3, 1, padding=1, weight_attr=nn.initializer.XavierUniform())

    def forward(self, x):
        return 0.3 * F.sigmoid(self.conv(x))

class ResConv(nn.Layer):
    def __init__(self, input_dim, output_dim, stride):
        super().__init__()
        self.do_proj = input_dim != output_dim or stride == 2

        self.res = nn.Sequential(
            nn.Conv2D(input_dim, output_dim, 1, 1, weight_attr=nn.initializer.XavierUniform()),
            nn.ELU(),
            nn.Conv2D(output_dim, output_dim, 3, stride, padding=1, weight_attr=nn.initializer.XavierUniform()),
            nn.ELU(),
            nn.Conv2D(output_dim, 4 * output_dim, 1, 1, weight_attr=nn.initializer.XavierUniform())
        )

        self.shortcut = nn.Conv2D(input_dim, 4 * output_dim, 1, stride, weight_attr=nn.initializer.XavierUniform()) or nn.Identity()

    def forward(self, x):
        return F.elu(self.res(x) + self.shortcut(x))

class ResBlock(nn.Layer):
    def __init__(self, input_dim, output_dim, blocks):
        super().__init__()
        assert blocks > 1
        self.blocks = nn.Sequential(*[ResConv(input_dim if i == 0 else output_dim * 4, output_dim, 2 if i == blocks - 1 else 1) for i in range(blocks)])
        
    def forward(self, x):
        return self.blocks(x)

class MonodepthModel(nn.Layer):

    def __init__(self, encoder_type='resnet', do_stereo=False, use_deconv=False):
        super().__init__()
        self.encoder_type = encoder_type
        self.do_stereo = do_stereo
        self.use_deconv = use_deconv

        self.input_dim = 6 if self.do_stereo else 3

        if self.encoder_type == 'vgg':
            self.build_vgg()
        elif self.encoder_type == 'resnet':
            self.build_resnet50()
    
    def forward(self, left, right=None):
        if self.do_stereo: assert right is not None
        x = paddle.concat([left, right], axis=1) if self.do_stereo else left

        if self.encoder_type == 'vgg':
            conv1 = self.conv1(x) # H/2
            conv2 = self.conv2(conv1) # H/4
            conv3 = self.conv3(conv2) # H/8
            conv4 = self.conv4(conv3) # H/16
            conv5 = self.conv5(conv4) # H/32
            conv6 = self.conv6(conv5) # H/64
            conv7 = self.conv7(conv6) # H/128

            skip1 = conv1
            skip2 = conv2
            skip3 = conv3
            skip4 = conv4
            skip5 = conv5
            skip6 = conv6
        
            upconv7 = self.upconv7(conv7) #H/64
            concat7 = paddle.concat([upconv7, skip6], 1)
            iconv7  = self.iconv7(concat7)

            upconv6 = self.upconv6(iconv7) #H/32
            concat6 = paddle.concat([upconv6, skip5], 1)
            iconv6  = self.iconv6(concat6)

            upconv5 = self.upconv5(iconv6) #H/16
            concat5 = paddle.concat([upconv5, skip4], 1)
            iconv5  = self.iconv5(concat5)

            upconv4 = self.upconv4(iconv5) #H/8
            concat4 = paddle.concat([upconv4, skip3], 1)
            iconv4  = self.iconv4(concat4)
            disp4 = self.disp4(iconv4)
            udisp4  = self.udisp4(disp4)

            upconv3 = self.upconv3(iconv4) #H/4
            concat3 = paddle.concat([upconv3, skip2, udisp4], 1)
            iconv3  = self.iconv3(concat3)
            disp3 = self.disp3(iconv3)
            udisp3  = self.udisp3(disp3)

            upconv2 = self.upconv2(iconv3) #H/2
            concat2 = paddle.concat([upconv2, skip1, udisp3], 1)
            iconv2  = self.iconv2(concat2)
            disp2 = self.disp2(iconv2)
            udisp2  = self.udisp2(disp2)

            upconv1 = self.upconv1(iconv2) #H
            concat1 = paddle.concat([upconv1, udisp2], 1)
            iconv1  = self.iconv1(concat1)
            disp1 = self.disp1(iconv1)

        elif self.encoder_type == 'resnet':
            conv1 = self.conv1(x) # H/2  -   64D
            pool1 = self.pool1(conv1) # H/4  -   64D
            conv2 = self.conv2(pool1) # H/8  -  256D
            conv3 = self.conv3(conv2) # H/16 -  512D
            conv4 = self.conv4(conv3) # H/32 - 1024D
            conv5 = self.conv5(conv4) # H/64 - 2048D

            skip1 = conv1
            skip2 = pool1
            skip3 = conv2
            skip4 = conv3
            skip5 = conv4

            upconv6 = self.upconv6(conv5) #H/32
            concat6 = paddle.concat([upconv6, skip5], 1)
            iconv6  = self.iconv6(concat6)

            upconv5 = self.upconv5(iconv6) #H/16
            concat5 = paddle.concat([upconv5, skip4], 1)
            iconv5  = self.iconv5(concat5)

            upconv4 = self.upconv4(iconv5) #H/8
            concat4 = paddle.concat([upconv4, skip3], 1)
            iconv4  = self.iconv4(concat4)
            disp4 = self.disp4(iconv4)
            udisp4  = self.udisp4(disp4)

            upconv3 = self.upconv3(iconv4) #H/4
            concat3 = paddle.concat([upconv3, skip2, udisp4], 1)
            iconv3  = self.iconv3(concat3)
            disp3 = self.disp3(iconv3)
            udisp3  = self.udisp3(disp3)

            upconv2 = self.upconv2(iconv3) #H/2
            concat2 = paddle.concat([upconv2, skip1, udisp3], 1)
            iconv2  = self.iconv2(concat2)
            disp2 = self.disp2(iconv2)
            udisp2  = self.udisp2(disp2)

            upconv1 = self.upconv1(iconv2) #H
            concat1 = paddle.concat([upconv1, udisp2], 1)
            iconv1  = self.iconv1(concat1)
            disp1 = self.disp1(iconv1)

        disp_est  = [disp1, disp2, disp3, disp4]
        disp_left_est  = [paddle.unsqueeze(d[:,0,:,:], 1) for d in disp_est]
        disp_right_est = [paddle.unsqueeze(d[:,1,:,:], 1) for d in disp_est]

        return disp_left_est, disp_right_est 

    def conv(self, input_dim, output_dim, kernel_size, stride, activation=nn.ELU):
        return nn.Sequential(
            nn.Conv2D(input_dim, output_dim, kernel_size, stride, padding=kernel_size//2, weight_attr=nn.initializer.XavierUniform()),
            activation()
        )

    def conv_block(self, input_dim, output_dim, kernel_size):
        return nn.Sequential(
            self.conv(input_dim, output_dim, kernel_size, 1),
            self.conv(output_dim, output_dim, kernel_size, 2)
        )

    def upconv(self, input_dim, output_dim, kernel_size, scale):
        return nn.Sequential(
            nn.UpsamplingNearest2D(scale_factor=scale),
            self.conv(input_dim, output_dim, kernel_size, 1)
        )

    def deconv(self, input_dim, output_dim, kernel_size, scale):
        return nn.Conv2DTranspose(input_dim, output_dim, kernel_size, scale, padding=kernel_size//2, weight_attr=nn.initializer.XavierUniform())

    def build_vgg(self):
        #set convenience functions
        conv = self.conv
        if self.use_deconv:
            upconv = self.deconv
        else:
            upconv = self.upconv

        self.conv1 = self.conv_block(self.input_dim, 32, 7) # H/2
        self.conv2 = self.conv_block(32, 64, 5) # H/4
        self.conv3 = self.conv_block(64, 128, 3) # H/8
        self.conv4 = self.conv_block(128, 256, 3) # H/16
        self.conv5 = self.conv_block(256, 512, 3) # H/32
        self.conv6 = self.conv_block(512, 512, 3) # H/64
        self.conv7 = self.conv_block(512, 512, 3) # H/128
        
        self.upconv7 = upconv(512, 512, 3, 2) #H/64
        self.iconv7  = conv(1024, 512, 3, 1)

        self.upconv6 = upconv(512, 512, 3, 2) #H/32
        self.iconv6  = conv(1024, 512, 3, 1)

        self.upconv5 = upconv(512, 256, 3, 2) #H/16
        self.iconv5  = conv(512, 256, 3, 1)

        self.upconv4 = upconv(256, 128, 3, 2) #H/8
        self.iconv4  = conv(256, 128, 3, 1)
        self.disp4 = DispHead(128)
        self.udisp4  = nn.UpsamplingNearest2D(scale_factor=2)

        self.upconv3 = upconv(128, 64, 3, 2) #H/4
        self.iconv3  = conv(128 + 2, 64, 3, 1)
        self.disp3 = DispHead(64)
        self.udisp3  = nn.UpsamplingNearest2D(scale_factor=2)

        self.upconv2 = upconv(64, 32, 3, 2) #H/2
        self.iconv2  = conv(64 + 2, 32, 3, 1)
        self.disp2 = DispHead(32)
        self.udisp2  = nn.UpsamplingNearest2D(scale_factor=2)

        self.upconv1 = upconv(32, 16, 3, 2) #H
        self.iconv1  = conv(16 + 2, 16, 3, 1)
        self.disp1 = DispHead(16)

    def build_resnet50(self):
        #set convenience functions
        conv = self.conv
        if self.use_deconv:
            upconv = self.deconv
        else:
            upconv = self.upconv

        self.conv1 = self.conv(self.input_dim, 64, 7, 2) # H/2  -   64D
        self.pool1 = nn.MaxPool2D(3, stride=2, padding=1) # H/4  -   64D
        self.conv2 = ResBlock(64, 64, 3) # H/8  -  256D
        self.conv3 = ResBlock(256, 128, 4) # H/16 -  512D
        self.conv4 = ResBlock(512, 256, 6) # H/32 - 1024D
        self.conv5 = ResBlock(1024, 512, 3) # H/64 - 2048D

        self.upconv6 = upconv(2048, 512, 3, 2) #H/32
        self.iconv6  = conv(1536, 512, 3, 1)

        self.upconv5 = upconv(512, 256, 3, 2) #H/16
        self.iconv5  = conv(768, 256, 3, 1)

        self.upconv4 = upconv(256, 128, 3, 2) #H/8
        self.iconv4  = conv(384, 128, 3, 1)
        self.disp4 = DispHead(128)
        self.udisp4 = nn.UpsamplingNearest2D(scale_factor=2)

        self.upconv3 = upconv(128, 64, 3, 2) #H/4
        self.iconv3  = conv(128 + 2, 64, 3, 1)
        self.disp3 = DispHead(64)
        self.udisp3  = nn.UpsamplingNearest2D(scale_factor=2)

        self.upconv2 = upconv(64, 32, 3, 2) #H/2
        self.iconv2  = conv(96 + 2, 32, 3, 1)
        self.disp2 = DispHead(32)
        self.udisp2  = nn.UpsamplingNearest2D(scale_factor=2)

        self.upconv1 = upconv(32, 16, 3, 2) #H
        self.iconv1  = conv(16 + 2, 16, 3, 1)
        self.disp1 = DispHead(16)

if __name__ == '__main__':
    model = MonodepthModel('resnet')
    left = paddle.randn((2, 3, 256, 512), dtype=paddle.float32)
    right = paddle.randn((2, 3, 256, 512), dtype=paddle.float32)
    disp_left_est, disp_right_est = model(left, right)
    
    from loss import get_monodepth_loss
    loss = get_monodepth_loss(disp_left_est, disp_right_est, left, right)
    print(loss)